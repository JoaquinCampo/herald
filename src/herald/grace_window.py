"""Grace-window alarm: feature assembly and the frozen alarm bundle.

The live controller attempts compression at a grid point s, observes
the first k post-switch tokens, and asks a frozen alarm model whether
to commit or roll back. This module builds the alarm's input features
with exact parity to the training pipeline:

- pre-switch features: `derive_features` on the reference logit stream,
  row s (the same construction as the switch dataset's `feat__`
  columns);
- post-switch block summary: step0/mean/min/max/slope/dtrail of the
  first k raw feature rows, with dtrail against the trailing-8 mean of
  the reference stream (the same construction as the recorded hybrid
  stream blocks; blocks round-trip through float16 because the
  training artifact stored them as float16);
- the matrix layout comes from `herald.switch_risk.featurize`, the
  function the alarm was trained through.

`AlarmBundle` is the frozen deployable: the 3-seed XGBoost ensemble,
the calibrated threshold theta, and the exact feature column order.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from herald.features import FEATURE_NAMES, IncrementalDerived, derive_features
from herald.switch_risk import featurize

TRAIL_WINDOW = 8
HYB_STATS: tuple[str, ...] = (
    "step0",
    "mean",
    "min",
    "max",
    "slope",
    "dtrail",
)


def hyb_feature_names(k: int) -> list[str]:
    """Post-switch summary column names, in training order."""
    return [
        f"hyb__{stat}_{fname}_k{k}"
        for stat in HYB_STATS
        for fname in FEATURE_NAMES
    ]


def hybrid_block_summary(
    block: np.ndarray, trailing: np.ndarray, *, k: int
) -> dict[str, float | None]:
    """Summarize the first k post-switch feature rows of one run.

    `block` is `(m, n_raw)` raw per-token features (m may be < k when
    the run ended early; m == 0 yields all-None). `trailing` is the
    `(n_raw,)` trailing reference mean (NaN when s == 0). NaN-aware
    aggregation matches the vectorized training construction: raw NaNs
    (kl_prev at step 0) are skipped by mean/min/max.
    """
    n_raw = len(FEATURE_NAMES)
    names = hyb_feature_names(k)
    b = np.asarray(block, dtype=np.float32)[:k]
    m = b.shape[0]
    if m == 0:
        return {name: None for name in names}
    if b.shape[1] != n_raw:
        raise ValueError(f"block has {b.shape[1]} columns, expected {n_raw}")
    with np.errstate(all="ignore"):
        mean = np.nanmean(b, axis=0)
        mn = np.nanmin(b, axis=0)
        mx = np.nanmax(b, axis=0)
    step0 = b[0]
    last = b[m - 1]
    slope = (last - step0) / np.float32(max(m - 1, 1))
    dtrail = mean - np.asarray(trailing, dtype=np.float32)
    values = np.concatenate([step0, mean, mn, mx, slope, dtrail])
    return {
        name: None if np.isnan(v) else float(v)
        for name, v in zip(names, values, strict=True)
    }


def preswitch_features(
    ref_raw: np.ndarray, s: int
) -> dict[str, float | None]:
    """Derived reference features at the switch position s.

    `ref_raw` is the raw `(steps, n_raw)` reference stream; row s must
    exist (the distribution at the switch point). Every derived column
    is causal, so only rows <= s are used.
    """
    if not 0 <= s < ref_raw.shape[0]:
        raise ValueError(
            f"switch position {s} outside reference stream of "
            f"{ref_raw.shape[0]} steps"
        )
    derived, names = derive_features(
        np.asarray(ref_raw[: s + 1], dtype=np.float32)
    )
    return {
        f"feat__{name}": None if np.isnan(v) else float(v)
        for name, v in zip(names, derived[s], strict=True)
    }


def assemble_alarm_row(
    *,
    ref_raw: np.ndarray,
    s: int,
    block: np.ndarray,
    ratio: float,
    k: int,
) -> dict[str, Any]:
    """Build one alarm input row from live streams.

    The block round-trips through float16 to match the stored training
    artifact exactly; the trailing mean is the float32 reference rows
    s-8..s-1 (NaN at s == 0), per the extraction script.
    """
    n_raw = len(FEATURE_NAMES)
    if s > 0:
        trailing = np.asarray(ref_raw, dtype=np.float32)[
            max(0, s - TRAIL_WINDOW) : s
        ].mean(axis=0)
    else:
        trailing = np.full(n_raw, np.nan, dtype=np.float32)
    quantized = (
        np.asarray(block, dtype=np.float32)
        .astype(np.float16)
        .astype(np.float32)
    )
    row: dict[str, Any] = {"task": "ifeval", "ratio": float(ratio)}
    row.update(preswitch_features(ref_raw, s))
    row.update(hybrid_block_summary(quantized, trailing, k=k))
    return row


def assemble_alarm_row_from_state(
    *,
    state: IncrementalDerived,
    block: np.ndarray,
    ratio: float,
    k: int,
) -> dict[str, Any]:
    """Build one alarm input row from incremental reference state."""
    trailing = state.trailing_raw_mean(TRAIL_WINDOW)
    quantized = (
        np.asarray(block, dtype=np.float32)
        .astype(np.float16)
        .astype(np.float32)
    )
    row: dict[str, Any] = {"task": "ifeval", "ratio": float(ratio)}
    row.update(state.preswitch_features())
    row.update(hybrid_block_summary(quantized, trailing, k=k))
    return row


@dataclass
class AlarmBundle:
    """Frozen grace-window alarm: ensemble, threshold, feature order."""

    compressor: str
    k: int
    epsilon: float
    theta: float
    feature_cols: list[str]
    boosters: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def score(self, row: dict[str, Any]) -> float:
        """Ensemble-mean alarm score for one assembled row."""
        import xgboost as xgb

        mat = featurize([row], self.feature_cols)
        dmat = xgb.DMatrix(mat)
        preds = [float(b.predict(dmat)[0]) for b in self.boosters]
        return float(np.mean(preds))

    def commits(self, score: float) -> bool:
        """Replay commit rule: commit at the first score <= theta."""
        return score <= self.theta

    def save(self, bundle_dir: Path) -> None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "compressor": self.compressor,
            "k": self.k,
            "epsilon": self.epsilon,
            "theta": self.theta,
            "feature_cols": self.feature_cols,
            "n_boosters": len(self.boosters),
            "meta": self.meta,
        }
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        for i, booster in enumerate(self.boosters):
            booster.save_model(str(bundle_dir / f"booster_{i}.json"))

    @classmethod
    def load(cls, bundle_dir: Path) -> "AlarmBundle":
        import xgboost as xgb

        manifest = json.loads((bundle_dir / "manifest.json").read_text())
        boosters = []
        for i in range(int(manifest["n_boosters"])):
            booster = xgb.Booster()
            booster.load_model(str(bundle_dir / f"booster_{i}.json"))
            boosters.append(booster)
        return cls(
            compressor=str(manifest["compressor"]),
            k=int(manifest["k"]),
            epsilon=float(manifest["epsilon"]),
            theta=float(manifest["theta"]),
            feature_cols=list(manifest["feature_cols"]),
            boosters=boosters,
            meta=dict(manifest["meta"]),
        )
