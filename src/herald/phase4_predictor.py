"""Phase 4 controller predictor: freeze + load lr_all_cheap.

Exports the fitted Phase 2 v1 predictor (`lr_all_cheap` on
`future_sum_js_25` thresholded at training-fold p90) as a portable
JSON artifact that the runtime controller can score against without
sklearn at inference time.

The exported artifact pins the feature order, the StandardScaler
mean/scale, and the LogisticRegression coefficients/intercept. A
small loader implements
    z = ((X - mean) / scale) @ coef + intercept
    p = 1 / (1 + exp(-z))
which is bit-equivalent to sklearn's
`LogisticRegression.predict_proba(...)[:, 1]` for binary
classification on the same scaled inputs.

Calibration: the artifact has a `calibration` slot reserved for an
isotonic mapping (fit on prompt-disjoint calibration prompts in a
separate stage). Until that's wired, the slot is None and the
caller must treat raw scores as ranks, not probabilities.

CPU-only — no torch import.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from herald.phase4_online_features import CHEAP_ALL_FEATURE_ORDER

ARTIFACT_VERSION = 1
DEFAULT_HORIZON = 25
DEFAULT_LABEL_BASE = "future_sum_js"
DEFAULT_QUANTILE = 0.90
DEFAULT_MAX_ITER = 200
DEFAULT_SEED = 42


# --- training -------------------------------------------------------


def _label_column(horizon: int, base: str = DEFAULT_LABEL_BASE) -> str:
    return f"{base}_{horizon}"


def _matrix_from_df(df: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Select feature columns, fill nulls with 0, NaN-clean.

    Mirrors the offline `_clean_X` + `fill_null(0.0)` path used by
    `predictor_baselines._logreg_score`. Bit-exact match is the
    parity bar.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns in dataframe: {missing}")
    arr = (
        df.select(list(columns)).fill_null(0.0).to_numpy().astype(np.float64)
    )
    cleaned: np.ndarray = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return cleaned


def fit_lr_all_cheap(
    df: pl.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    label_base: str = DEFAULT_LABEL_BASE,
    quantile: float = DEFAULT_QUANTILE,
    feature_order: tuple[str, ...] = CHEAP_ALL_FEATURE_ORDER,
    max_iter: int = DEFAULT_MAX_ITER,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Fit lr_all_cheap on `df` and return the export-ready state dict.

    Mirrors `predictor_baselines._logreg_score` exactly: StandardScaler
    with_mean+with_std, LogisticRegression(max_iter, class_weight=
    "balanced", solver="lbfgs"), predict_proba[:, 1] is the head.

    Drops rows where the continuous label is null. Threshold is the
    `quantile`-th quantile of the non-null label values in `df`
    (training-fold p90 by default).
    """
    label_col = _label_column(horizon, label_base)
    if label_col not in df.columns:
        raise ValueError(f"label column missing: {label_col}")

    df = df.filter(pl.col(label_col).is_not_null())
    if df.height < 4:
        raise ValueError(
            f"need >=4 training rows after null-drop; got {df.height}"
        )

    q_val = df[label_col].quantile(quantile, interpolation="linear")
    if q_val is None:
        raise ValueError(
            f"quantile({quantile}) of {label_col} is null; "
            "training fold has no usable labels"
        )
    threshold = float(q_val)
    y = (df[label_col].to_numpy() >= threshold).astype(np.int64)
    if y.sum() == 0 or y.sum() == y.shape[0]:
        raise ValueError(
            "label degenerated to single class after thresholding"
        )

    X = _matrix_from_df(df, feature_order)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=seed,
    )
    clf.fit(Xs, y)

    # sklearn LR for binary stores classes_ = [0, 1]; coef_ has
    # shape (1, n_features) and corresponds to class 1. predict_proba
    # column 1 is the positive-class prob.
    classes = clf.classes_.tolist()
    if classes != [0, 1]:
        raise RuntimeError(f"unexpected sklearn class order: {classes}")

    return {
        "artifact_version": ARTIFACT_VERSION,
        "model": "lr_all_cheap",
        "label": {
            "base": label_base,
            "horizon": horizon,
            "column": label_col,
            "binarization": "ge_threshold",
            "threshold": threshold,
            "quantile": quantile,
        },
        "feature_order": list(feature_order),
        "scaler": {
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "with_mean": True,
            "with_std": True,
        },
        "logreg": {
            "coef": clf.coef_[0].astype(float).tolist(),
            "intercept": float(clf.intercept_[0]),
            "classes": classes,
            "max_iter": max_iter,
            "solver": "lbfgs",
            "class_weight": "balanced",
        },
        "training": {
            "n_rows": int(df.height),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "seed": seed,
        },
        "calibration": None,
    }


# --- IO --------------------------------------------------------------


def save_predictor(state: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return path


def load_predictor_state(path: Path) -> dict[str, Any]:
    state: dict[str, Any] = json.loads(path.read_text())
    return state


# --- inference -------------------------------------------------------


@dataclass(slots=True)
class ExportedPredictor:
    """Loader-side wrapper around the JSON artifact.

    `score` returns positive-class probability per row; bit-equivalent
    to sklearn's `predict_proba(X)[:, 1]` for the same standardized
    LogisticRegression.
    """

    feature_order: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    horizon: int
    label_column: str
    threshold: float

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "ExportedPredictor":
        return cls(
            feature_order=tuple(state["feature_order"]),
            mean=np.asarray(state["scaler"]["mean"], dtype=np.float64),
            scale=np.asarray(state["scaler"]["scale"], dtype=np.float64),
            coef=np.asarray(state["logreg"]["coef"], dtype=np.float64),
            intercept=float(state["logreg"]["intercept"]),
            horizon=int(state["label"]["horizon"]),
            label_column=str(state["label"]["column"]),
            threshold=float(state["label"]["threshold"]),
        )

    @classmethod
    def from_path(cls, path: Path) -> "ExportedPredictor":
        return cls.from_state(load_predictor_state(path))

    def score(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probability per row, shape (n,)."""
        if X.ndim != 2 or X.shape[1] != self.coef.shape[0]:
            raise ValueError(
                f"X shape {X.shape} incompatible with feature_order "
                f"({self.coef.shape[0]} columns)"
            )
        Xc = np.nan_to_num(
            X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
        )
        z = (Xc - self.mean) / self.scale @ self.coef + self.intercept
        # numerically stable sigmoid
        return _sigmoid(z)

    def score_one(self, feature_dict: dict[str, float]) -> float:
        """Score a single online feature row by name lookup."""
        x = np.asarray(
            [feature_dict.get(c, 0.0) for c in self.feature_order],
            dtype=np.float64,
        ).reshape(1, -1)
        return float(self.score(x)[0])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Stable sigmoid; matches scipy.special.expit."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def sigmoid_scalar(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)
