"""Tests for the grace-window alarm feature assembly and bundle.

The live controller must feed the frozen alarm exactly the features it
was trained on. These tests pin the two risky constructions against
independent reimplementations of the training pipeline:

- `hybrid_block_summary` vs the vectorized `hyb_summaries` used by the
  AIMD feasibility scripts (copied verbatim below).
- `preswitch_features` vs `derive_features` on the full reference
  matrix (and, when local sweep artifacts exist, vs the actual stored
  parquet `feat__` values).
"""

from pathlib import Path

import numpy as np
import pytest

from herald.features import FEATURE_NAMES, IncrementalDerived, derive_features
from herald.grace_window import (
    AlarmBundle,
    assemble_alarm_row,
    assemble_alarm_row_from_state,
    hyb_feature_names,
    hybrid_block_summary,
    preswitch_features,
)
from herald.switch_risk import featurize

RNG = np.random.default_rng(7)
N_RAW = len(FEATURE_NAMES)


def vectorized_hyb_summaries(
    blocks: np.ndarray,
    lengths: np.ndarray,
    trailing: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """Verbatim copy of the training-side construction (scratchpad
    aimd_feasibility.py / xgb_scores_robustness.py hyb_summaries)."""
    n = blocks.shape[0]
    b = blocks[:, :k].astype(np.float32)
    m = np.minimum(lengths, k)
    step = np.arange(k)[None, :, None]
    b = np.where(step < m[:, None, None], b, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(b, axis=1)
        mn = np.nanmin(b, axis=1)
        mx = np.nanmax(b, axis=1)
    step0 = b[:, 0]
    last = b[np.arange(n), np.maximum(m - 1, 0)]
    slope = (last - step0) / np.maximum(m - 1, 1).astype(np.float32)[:, None]
    dtrail = mean - trailing
    parts = [step0, mean, mn, mx, slope, dtrail]
    for p in parts:
        p[m == 0] = np.nan
    names = [
        f"hyb__{stat}_{fname}_k{k}"
        for stat in ("step0", "mean", "min", "max", "slope", "dtrail")
        for fname in FEATURE_NAMES
    ]
    return np.concatenate(parts, axis=1), names


def random_block(n_steps: int) -> np.ndarray:
    return RNG.normal(size=(n_steps, N_RAW)).astype(np.float32)


def summary_as_array(summary: dict[str, float | None], k: int) -> np.ndarray:
    names = hyb_feature_names(k)
    return np.asarray(
        [np.nan if summary[n] is None else summary[n] for n in names],
        dtype=np.float32,
    )


class TestHybridBlockSummary:
    def test_names_match_training_order(self) -> None:
        _, expected = vectorized_hyb_summaries(
            np.zeros((1, 2, N_RAW)), np.array([2]), np.zeros((1, N_RAW)), 2
        )
        assert hyb_feature_names(2) == expected

    @pytest.mark.parametrize("n_steps", [0, 1, 2, 5])
    @pytest.mark.parametrize("k", [2, 4])
    def test_parity_with_vectorized(self, n_steps: int, k: int) -> None:
        block = random_block(n_steps)
        trailing = RNG.normal(size=N_RAW).astype(np.float32)
        padded = np.full((1, max(k, n_steps, 1), N_RAW), np.nan)
        if n_steps:
            padded[0, :n_steps] = block
        expected, _ = vectorized_hyb_summaries(
            padded,
            np.array([n_steps]),
            trailing[None, :].astype(np.float32),
            k,
        )
        got = summary_as_array(hybrid_block_summary(block, trailing, k=k), k)
        np.testing.assert_allclose(
            got, expected[0], rtol=1e-5, equal_nan=True
        )

    def test_nan_trailing_gives_nan_dtrail(self) -> None:
        block = random_block(2)
        trailing = np.full(N_RAW, np.nan, dtype=np.float32)
        summary = hybrid_block_summary(block, trailing, k=2)
        assert summary["hyb__dtrail_entropy_k2"] is None
        assert summary["hyb__mean_entropy_k2"] is not None


class TestPreswitchFeatures:
    def test_matches_full_matrix_row(self) -> None:
        raw = random_block(60).astype(np.float32)
        raw[0, FEATURE_NAMES.index("kl_prev")] = np.nan
        derived, names = derive_features(raw)
        s = 41
        got = preswitch_features(raw, s)
        assert set(got) == {f"feat__{n}" for n in names}
        for j, name in enumerate(names):
            want = float(derived[s, j])
            have = got[f"feat__{name}"]
            if np.isnan(want):
                assert have is None
            else:
                assert have == pytest.approx(want, rel=1e-6)

    def test_truncation_invariance(self) -> None:
        raw = random_block(80).astype(np.float32)
        s = 33
        full = preswitch_features(raw, s)
        truncated = preswitch_features(raw[: s + 1], s)
        assert full == truncated

    def test_s_out_of_range_raises(self) -> None:
        raw = random_block(10)
        with pytest.raises(ValueError):
            preswitch_features(raw, 10)


class TestAssembleAlarmRow:
    def test_row_feeds_featurize(self) -> None:
        raw = random_block(40)
        block = random_block(2)
        row = assemble_alarm_row(
            ref_raw=raw, s=16, block=block, ratio=0.5, k=2
        )
        assert row["task"] == "ifeval"
        assert row["ratio"] == 0.5
        cols = sorted(c for c in row if c.startswith("feat__")) + list(
            hyb_feature_names(2)
        )
        mat = featurize([row], cols)
        assert mat.shape == (1, 3 + 1 + len(cols))
        assert mat[0, 2] == 1.0  # ifeval one-hot

    def test_trailing_window_matches_extraction(self) -> None:
        """dtrail must use the mean of raw rows s-8..s-1, and the block
        must round-trip through float16, per extract_hybrid_streams.py
        (the training blocks were stored as float16)."""
        raw = random_block(40)
        block = random_block(2)
        s = 20
        row = assemble_alarm_row(
            ref_raw=raw, s=s, block=block, ratio=0.25, k=2
        )
        trailing = raw[s - 8 : s].mean(axis=0)
        quantized = block.astype(np.float16).astype(np.float32)
        expected = hybrid_block_summary(quantized, trailing, k=2)
        for name in hyb_feature_names(2):
            assert row[name] == expected[name]

    def test_s_zero_has_nan_trailing(self) -> None:
        raw = random_block(1)
        block = random_block(2)
        row = assemble_alarm_row(
            ref_raw=raw, s=0, block=block, ratio=0.25, k=2
        )
        assert row["hyb__dtrail_entropy_k2"] is None


class TestAlarmBundle:
    def _tiny_bundle(self, tmp_path: Path) -> AlarmBundle:
        import xgboost as xgb

        # featurize layout: 3 task one-hots + ratio + 2 feature cols.
        x = RNG.normal(size=(200, 6)).astype(np.float32)
        y = (x[:, 4] > 0).astype(np.float32)
        boosters = []
        for seed in (0, 1, 2):
            boosters.append(
                xgb.train(
                    {
                        "objective": "binary:logistic",
                        "seed": seed,
                        "max_depth": 2,
                    },
                    xgb.DMatrix(x, label=y),
                    num_boost_round=5,
                )
            )
        return AlarmBundle(
            compressor="expected_attention",
            k=2,
            epsilon=0.03,
            theta=0.4,
            feature_cols=["feat__entropy", "feat__max_prob"],
            boosters=boosters,
            meta={"split_seed": 0},
        )

    def test_roundtrip(self, tmp_path: Path) -> None:
        bundle = self._tiny_bundle(tmp_path)
        out = tmp_path / "bundle"
        bundle.save(out)
        loaded = AlarmBundle.load(out)
        assert loaded.compressor == bundle.compressor
        assert loaded.theta == bundle.theta
        assert loaded.k == bundle.k
        assert loaded.epsilon == bundle.epsilon
        assert loaded.feature_cols == bundle.feature_cols
        assert loaded.meta["split_seed"] == 0
        assert len(loaded.boosters) == 3

    def test_score_is_seed_mean_and_stable(self, tmp_path: Path) -> None:
        import xgboost as xgb

        bundle = self._tiny_bundle(tmp_path)
        row = {
            "task": "ifeval",
            "ratio": 0.5,
            "feat__entropy": 1.2,
            "feat__max_prob": 0.9,
        }
        mat = featurize([row], bundle.feature_cols)
        expected = float(
            np.mean([b.predict(xgb.DMatrix(mat))[0] for b in bundle.boosters])
        )
        assert bundle.score(row) == pytest.approx(expected, rel=1e-6)
        out = tmp_path / "bundle"
        bundle.save(out)
        loaded = AlarmBundle.load(out)
        assert loaded.score(row) == pytest.approx(expected, rel=1e-6)

    def test_commit_rule(self, tmp_path: Path) -> None:
        bundle = self._tiny_bundle(tmp_path)
        assert bundle.commits(bundle.theta) is True
        assert bundle.commits(bundle.theta + 1e-6) is False


SWEEP_DIR = Path("results/sweep/llama/ifeval")
PARQUET = Path("results/predictor/switch_dataset_attn.parquet")


@pytest.mark.skipif(
    not (SWEEP_DIR.exists() and PARQUET.exists()),
    reason="local sweep artifacts not present",
)
class TestArtifactParity:
    """End-to-end parity against the real recorded dataset."""

    def test_preswitch_matches_parquet_feat_columns(self) -> None:
        import pandas as pd

        from herald.storage import safe_id

        df = pd.read_parquet(PARQUET)
        df = df[
            (df["task"] == "ifeval")
            & (df["compressor"] == "expected_attention")
            & (df["s"] > 0)
        ]
        row = df.iloc[137]
        ref_raw = np.load(
            SWEEP_DIR / "references" / f"{safe_id(row['prompt_id'])}.npy"
        ).astype(np.float32)
        got = preswitch_features(ref_raw, int(row["s"]))
        engine = IncrementalDerived()
        for raw_row in ref_raw[: int(row["s"]) + 1]:
            engine.update(raw_row)
        incremental = engine.preswitch_features()
        assert incremental == got
        checked = 0
        for name, value in got.items():
            if name not in df.columns or name.startswith("feat__attn_"):
                continue
            want = row[name]
            if value is None:
                assert np.isnan(want)
            else:
                assert value == pytest.approx(float(want), rel=1e-4), name
            checked += 1
        assert checked > 100

    def test_block_summary_matches_recorded_stream(self) -> None:
        import pandas as pd

        from herald.storage import safe_id

        df = pd.read_parquet(
            PARQUET, columns=["task", "compressor", "ratio", "prompt_id", "s"]
        )
        df = df[
            (df["task"] == "ifeval") & (df["compressor"] == "knorm")
        ].reset_index(drop=True)
        row = df.iloc[731]
        hyb_path = (
            SWEEP_DIR
            / "hybrid_features"
            / f"{row['compressor']}__{row['ratio']:.4f}"
            / f"{safe_id(row['prompt_id'])}__s{int(row['s'])}.npy"
        )
        if not hyb_path.exists():
            pytest.skip("hybrid feature file missing")
        hyb = np.load(hyb_path).astype(np.float32)
        ref_raw = np.load(
            SWEEP_DIR / "references" / f"{safe_id(row['prompt_id'])}.npy"
        ).astype(np.float32)
        live_row = assemble_alarm_row(
            ref_raw=ref_raw,
            s=int(row["s"]),
            block=hyb[:2],
            ratio=float(row["ratio"]),
            k=2,
        )
        engine = IncrementalDerived()
        for raw_row in ref_raw[: int(row["s"]) + 1]:
            engine.update(raw_row)
        incremental_row = assemble_alarm_row_from_state(
            state=engine,
            block=hyb[:2],
            ratio=float(row["ratio"]),
            k=2,
        )
        assert incremental_row == live_row
        s = int(row["s"])
        trailing = (
            ref_raw[max(0, s - 8) : s].mean(axis=0)
            if s > 0
            else np.full(N_RAW, np.nan, dtype=np.float32)
        )
        padded = np.full((1, 16, N_RAW), np.nan, dtype=np.float16)
        m = min(2, hyb.shape[0])
        padded[0, :m] = hyb[:m].astype(np.float16)
        expected, names = vectorized_hyb_summaries(
            padded,
            np.array([m]),
            trailing[None, :].astype(np.float32),
            2,
        )
        for j, name in enumerate(names):
            want = expected[0, j]
            have = live_row[name]
            if np.isnan(want):
                assert have is None, name
            else:
                assert have == pytest.approx(float(want), rel=1e-5), name
