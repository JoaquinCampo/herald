"""Predictor export parity tests.

Two parity bars:
1. ExportedPredictor.score(X) matches sklearn predict_proba[:, 1] on
   the SAME standardized inputs (loader correctness).
2. End-to-end export round-trip: fit -> save -> load -> score equals
   in-memory sklearn predict_proba on the same fit, on a fixed batch.
"""

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from herald.phase4_online_features import CHEAP_ALL_FEATURE_ORDER
from herald.phase4_predictor import (
    ARTIFACT_VERSION,
    DEFAULT_HORIZON,
    DEFAULT_QUANTILE,
    ExportedPredictor,
    _matrix_from_df,
    fit_lr_all_cheap,
    load_predictor_state,
    save_predictor,
    sigmoid_scalar,
)

FIXTURE = Path("tests/fixtures/phase4_online_parity_run.parquet")
PHASE2 = Path("results/phase2/dataset/phase2_tokens.parquet")


# --- sigmoid + matrix shape ----------------------------------------


def test_sigmoid_scalar_matches_definition() -> None:
    for z in (-5.0, -1.0, 0.0, 1.0, 5.0):
        assert sigmoid_scalar(z) == pytest.approx(1.0 / (1.0 + math.exp(-z)))


def test_matrix_from_df_handles_nulls_and_nans() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, None, 3.0],
            "b": [None, float("nan"), 2.0],
        }
    )
    X = _matrix_from_df(df, ("a", "b"))
    assert X.shape == (3, 2)
    # nulls and NaN both -> 0
    assert X[1, 0] == 0.0
    assert X[1, 1] == 0.0
    assert X[0, 1] == 0.0


def test_matrix_from_df_rejects_missing_columns() -> None:
    df = pl.DataFrame({"a": [1.0]})
    with pytest.raises(ValueError, match="missing feature columns"):
        _matrix_from_df(df, ("a", "b"))


# --- synthetic end-to-end parity -----------------------------------


def _synthetic_dataset(n: int = 400, seed: int = 7) -> pl.DataFrame:
    """Build a small dataframe with the 24 lr_all_cheap columns +
    the future_sum_js_25 label, for hermetic parity testing.

    The label is correlated with `entropy` so the resulting LR is
    non-degenerate and thresholding at p90 produces a real binary
    split with both classes populated.
    """
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {}
    # entropy seeds the label
    entropy = rng.gamma(2.0, 0.5, size=n)
    for c in CHEAP_ALL_FEATURE_ORDER:
        if c == "entropy":
            cols[c] = entropy
        elif c == "token_pos":
            cols[c] = np.arange(n, dtype=float)
        elif c == "relative_progress":
            cols[c] = np.linspace(0.0, 1.0, n)
        elif c == "compression_ratio":
            cols[c] = np.full(n, 0.875)
        else:
            cols[c] = rng.normal(size=n)
    label = entropy * 0.5 + rng.normal(0, 0.1, size=n)
    cols["future_sum_js_25"] = label
    return pl.DataFrame(cols)


def test_fit_lr_all_cheap_returns_complete_state() -> None:
    df = _synthetic_dataset()
    state = fit_lr_all_cheap(df)
    assert state["artifact_version"] == ARTIFACT_VERSION
    assert state["model"] == "lr_all_cheap"
    assert state["label"]["horizon"] == DEFAULT_HORIZON
    assert state["label"]["column"] == "future_sum_js_25"
    assert state["label"]["quantile"] == DEFAULT_QUANTILE
    assert len(state["feature_order"]) == len(CHEAP_ALL_FEATURE_ORDER)
    assert state["feature_order"] == list(CHEAP_ALL_FEATURE_ORDER)
    assert len(state["scaler"]["mean"]) == len(CHEAP_ALL_FEATURE_ORDER)
    assert len(state["scaler"]["scale"]) == len(CHEAP_ALL_FEATURE_ORDER)
    assert len(state["logreg"]["coef"]) == len(CHEAP_ALL_FEATURE_ORDER)
    assert state["logreg"]["classes"] == [0, 1]
    assert state["calibration"] is None
    assert state["training"]["n_pos"] > 0
    assert state["training"]["n_neg"] > 0


def test_export_matches_sklearn_predict_proba(tmp_path: Path) -> None:
    """The headline parity bar: round-trip fit/save/load gives the
    same probabilities as the in-memory sklearn pipeline.
    """
    df = _synthetic_dataset()
    state = fit_lr_all_cheap(df)
    out = save_predictor(state, tmp_path / "phase4_lr.json")
    assert out.exists()
    loaded = ExportedPredictor.from_path(out)

    # Re-fit the same pipeline in memory exactly the way the freeze
    # path did, and compare predict_proba on the SAME rows.
    X = _matrix_from_df(df, CHEAP_ALL_FEATURE_ORDER)
    threshold = float(
        df["future_sum_js_25"].quantile(
            DEFAULT_QUANTILE, interpolation="linear"
        )
    )
    y = (df["future_sum_js_25"].to_numpy() >= threshold).astype(int)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(Xs, y)
    sklearn_probs = clf.predict_proba(Xs)[:, 1]

    exported_probs = loaded.score(X)
    assert exported_probs.shape == sklearn_probs.shape
    max_abs = float(np.max(np.abs(sklearn_probs - exported_probs)))
    # Bit-equivalent up to fp64 sigmoid roundoff.
    assert max_abs < 1e-12, f"sklearn vs exported diverge: {max_abs}"


def test_export_score_matches_for_fixed_batch(tmp_path: Path) -> None:
    """Independently of refit, a saved -> loaded artifact produces
    deterministic outputs for any fixed input batch.
    """
    df = _synthetic_dataset()
    state = fit_lr_all_cheap(df)
    path = save_predictor(state, tmp_path / "phase4_lr.json")

    a = ExportedPredictor.from_path(path)
    b = ExportedPredictor.from_path(path)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, len(CHEAP_ALL_FEATURE_ORDER))).astype(np.float64)
    pa = a.score(X)
    pb = b.score(X)
    assert np.array_equal(pa, pb)
    assert pa.shape == (50,)
    assert (pa >= 0).all() and (pa <= 1).all()


def test_score_one_matches_score_batch() -> None:
    df = _synthetic_dataset()
    state = fit_lr_all_cheap(df)
    pred = ExportedPredictor.from_state(state)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, len(CHEAP_ALL_FEATURE_ORDER))).astype(np.float64)
    batch = pred.score(X)
    for i in range(X.shape[0]):
        row = dict(zip(CHEAP_ALL_FEATURE_ORDER, X[i].tolist()))
        assert pred.score_one(row) == pytest.approx(batch[i])


def test_score_rejects_wrong_shape() -> None:
    df = _synthetic_dataset()
    state = fit_lr_all_cheap(df)
    pred = ExportedPredictor.from_state(state)
    with pytest.raises(ValueError, match="incompatible"):
        pred.score(np.zeros((3, 5)))
    with pytest.raises(ValueError, match="incompatible"):
        pred.score(np.zeros(24))


def test_load_predictor_state_round_trip(tmp_path: Path) -> None:
    state = {
        "artifact_version": 1,
        "model": "lr_all_cheap",
        "feature_order": list(CHEAP_ALL_FEATURE_ORDER),
        "scaler": {
            "mean": [0.0] * 24,
            "scale": [1.0] * 24,
            "with_mean": True,
            "with_std": True,
        },
        "logreg": {
            "coef": [0.0] * 24,
            "intercept": 0.0,
            "classes": [0, 1],
        },
        "label": {
            "base": "future_sum_js",
            "horizon": 25,
            "column": "future_sum_js_25",
            "binarization": "ge_threshold",
            "threshold": 1.5,
            "quantile": 0.9,
        },
        "training": {"n_rows": 100, "n_pos": 10, "n_neg": 90, "seed": 42},
        "calibration": None,
    }
    p = save_predictor(state, tmp_path / "x.json")
    reload = load_predictor_state(p)
    assert reload == state


# --- parity vs real Phase 2 dataset (skipped if absent) ------------


@pytest.mark.skipif(
    not PHASE2.exists(), reason="real Phase 2 dataset not present"
)
def test_real_phase2_export_matches_sklearn_on_fixed_batch() -> None:
    """End-to-end on the real dataset: fit -> save -> load -> score
    matches in-memory sklearn predict_proba on the same rows.
    """
    df = pl.read_parquet(PHASE2)
    state = fit_lr_all_cheap(df)
    pred = ExportedPredictor.from_state(state)

    df_nn = df.filter(pl.col("future_sum_js_25").is_not_null())
    threshold = float(
        df_nn["future_sum_js_25"].quantile(0.9, interpolation="linear")
    )
    y = (df_nn["future_sum_js_25"].to_numpy() >= threshold).astype(int)
    X = _matrix_from_df(df_nn, CHEAP_ALL_FEATURE_ORDER)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(Xs, y)
    sklearn_probs = clf.predict_proba(Xs[:5000])[:, 1]
    exported_probs = pred.score(X[:5000])

    max_abs = float(np.max(np.abs(sklearn_probs - exported_probs)))
    assert max_abs < 1e-10, f"diverge on real dataset: {max_abs}"
