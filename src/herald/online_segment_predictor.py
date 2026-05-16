"""Streaming online HERALD score from per-token signals.

Wraps the Phase 2 v2 segment-level XGBoost predictor for online use:
buffer per-token cheap features + token ids, fire a prediction at every
K-token segment boundary. The score is the model's calibrated probability
that the first catastrophic onset (looping or non-termination) lands in
the next K tokens.

Parity with offline pipeline:
- Per-segment aggregates use the same polars expressions as
  `scripts/build_phase2_v2_segments.py::aggregate_tokens_to_segments`.
- Surface repetition features call the same
  `surface_features_per_segment` helper.
- Cross-segment dynamics (cum_mean, roll8_std, prev/delta, roll4 mean/max,
  hand-crafted) match `scripts/extend_phase2_v2_segments.py`.

Cost: re-runs polars over a small per-segment dataframe (<= 32 rows).
Each prediction is single-digit milliseconds on CPU; the dominant cost
in HERALD is extract_signals on the GPU side, not this.

Caller pattern:

    pred = StreamingHeraldPredictor.from_paths(
        model_path=Path("results/.../model_fold0.json"),
        feature_cols=summary["feature_cols"],
        press="streaming_llm",
        compression_ratio=0.875,
    )
    for sig, tok in token_stream:
        out = pred.update(sig, tok)
        if out is not None:
            print(out["seg_idx"], out["score"])
"""

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from herald.config import TokenSignals

CHEAP_FEATURES: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
)


def _segment_aggregate_polars(
    cheap_buffer: dict[str, list[float]],
    token_pos_buffer: list[int],
    seg_idx: int,
    k: int,
) -> dict[str, float]:
    """Per-segment aggregates for the latest K tokens.

    Mirrors aggregate_tokens_to_segments in
    scripts/build_phase2_v2_segments.py: emits {f}_mean, {f}_last,
    {f}_max, {f}_min, {f}_std, {f}_slope for each cheap feature, plus
    seg_end_tok and seg_n_tok.
    """
    start = seg_idx * k
    end = (seg_idx + 1) * k
    pos_slice = token_pos_buffer[start:end]
    n = len(pos_slice)
    if n == 0:
        return {}

    cols = {
        "token_pos": pos_slice,
        **{f: cheap_buffer[f][start:end] for f in CHEAP_FEATURES},
    }
    df = pl.DataFrame(cols)

    agg: dict[str, float] = {}
    for f in CHEAP_FEATURES:
        agg[f"{f}_mean"] = df[f].mean()
        agg[f"{f}_last"] = df[f].last()
        agg[f"{f}_max"] = df[f].max()
        agg[f"{f}_min"] = df[f].min()
        agg[f"{f}_std"] = df[f].std()
    agg["seg_end_tok"] = df["token_pos"].max()
    agg["seg_n_tok"] = df["token_pos"].count()

    # Slope: Cov(f, t) / Var(t) (least-squares).
    t = df["token_pos"].to_numpy().astype(np.float64)
    t_mean = t.mean()
    t_dev = t - t_mean
    t_var = (t_dev**2).sum()
    for f in CHEAP_FEATURES:
        x = df[f].to_numpy().astype(np.float64)
        x_mean = x.mean()
        if math.isnan(x_mean) or t_var <= 0:
            agg[f"{f}_slope"] = math.nan
            continue
        cov = ((x - x_mean) * t_dev).sum()
        agg[f"{f}_slope"] = float(cov / (t_var + 1e-9))

    # Cast to plain Python floats (polars returns numpy scalars sometimes)
    return {
        k_: (float(v) if v is not None else math.nan) for k_, v in agg.items()
    }


def _surface_features(
    token_ids: list[int], k: int, seg_end: int
) -> dict[str, float]:
    """Causal surface repetition features at seg_end.

    Mirrors surface_features_per_segment from
    scripts/build_phase2_v2_segments.py for a single seg_end.
    """
    arr = np.asarray(token_ids, dtype=np.int64)
    t = seg_end
    if t < 0 or t >= len(arr):
        return {
            "top1_streak": 0,
            "n_unique_64": 0,
            "last_k_repeat_count": 0,
            "bigram_dup_rate_64": 0.0,
            "max_window20_repeats": 0,
        }

    streak = 1
    for i in range(t - 1, -1, -1):
        if arr[i] == arr[t]:
            streak += 1
        else:
            break

    win64 = arr[max(0, t - 63) : t + 1]
    n_unique_64 = int(np.unique(win64).size)

    if len(win64) >= 2:
        recent_bg = list(zip(win64[:-1], win64[1:], strict=True))
        history = arr[: max(0, t - 63)]
        if len(history) >= 2:
            hist_bg = set(zip(history[:-1], history[1:], strict=True))
        else:
            hist_bg = set()
        bigram_dup = sum(1 for bg in recent_bg if bg in hist_bg) / len(
            recent_bg
        )
    else:
        bigram_dup = 0.0

    if t + 1 >= k:
        target = arr[t + 1 - k : t + 1]
        target_bytes = target.tobytes()
        history = arr[: t + 1 - k]
        count = 0
        if len(history) >= k:
            for i in range(len(history) - k + 1):
                if history[i : i + k].tobytes() == target_bytes:
                    count += 1
        last_k_repeat_count = count
    else:
        last_k_repeat_count = 0

    window_size = 20
    if t + 1 >= window_size:
        seen: dict[bytes, int] = {}
        max_rep = 0
        for i in range(t + 1 - window_size + 1):
            key = arr[i : i + window_size].tobytes()
            seen[key] = seen.get(key, 0) + 1
            max_rep = max(max_rep, seen[key])
        max_window20_repeats = max_rep
    else:
        max_window20_repeats = 0

    return {
        "top1_streak": int(streak),
        "n_unique_64": int(n_unique_64),
        "last_k_repeat_count": int(last_k_repeat_count),
        "bigram_dup_rate_64": float(bigram_dup),
        "max_window20_repeats": int(max_window20_repeats),
    }


_DYNAMICS_META = {
    "run_id",
    "prompt_id",
    "press",
    "compression_ratio",
    "seg_idx",
    "seg_end_tok",
    "looping_onset",
    "non_termination_onset",
    "first_onset",
    "num_tokens_generated",
    "relative_progress",
    "y",
}


def _apply_cross_segment_dynamics(seg_df: pl.DataFrame) -> pl.DataFrame:
    """Mirror scripts/extend_phase2_v2_segments.py for the streaming case.

    Operates on a single run's seg_history (no `over("run_id")` needed
    because there is only one run); uses the same expression set so the
    last row matches the offline parquet's last row for the same run.
    """
    df = seg_df.sort("seg_idx")
    feat_cols = [
        c
        for c in df.columns
        if c not in _DYNAMICS_META
        and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    df = df.with_columns(
        [
            (pl.col(c).cum_sum() / (pl.col(c).cum_count() + 1e-9)).alias(
                f"{c}_cum_mean"
            )
            for c in feat_cols
        ]
    )
    df = df.with_columns(
        [
            pl.col(c)
            .rolling_std(window_size=8, min_samples=2)
            .alias(f"{c}_roll8_std")
            for c in feat_cols
        ]
    )
    df = df.with_columns(
        [pl.col(c).shift(1).alias(f"{c}_prev") for c in feat_cols]
    )
    df = df.with_columns(
        [
            (pl.col(c) - pl.col(f"{c}_prev")).alias(f"{c}_delta")
            for c in feat_cols
        ]
    )
    df = df.with_columns(
        [
            pl.col(c)
            .rolling_mean(window_size=4, min_samples=1)
            .alias(f"{c}_roll4_mean")
            for c in feat_cols
        ]
    )
    df = df.with_columns(
        [
            pl.col(c)
            .rolling_max(window_size=4, min_samples=1)
            .alias(f"{c}_roll4_max")
            for c in feat_cols
        ]
    )
    df = df.with_columns(
        [
            pl.col("max_window20_repeats")
            .cum_max()
            .alias("cum_max_window20_repeats"),
            pl.col("top1_streak").cum_max().alias("cum_max_top1_streak"),
            (pl.col("n_unique_64") - pl.col("n_unique_64").cum_max()).alias(
                "n_unique_64_below_peak"
            ),
        ]
    )
    return df


@dataclass(slots=True)
class StreamingHeraldPredictor:
    """Online segment-level HERALD score.

    Construct via `from_paths` for the typical case. `update(sig, tok)`
    is called once per generated token; it returns None except at the
    K-token boundary where it returns the prediction dict.
    """

    model: xgb.XGBClassifier
    feature_cols: list[str]
    press: str
    compression_ratio: float
    k: int = 16
    max_budget: int = 512
    isotonic: object | None = None
    _token_pos: int = -1
    _cheap: dict[str, list[float]] = field(default_factory=dict)
    _token_ids: list[int] = field(default_factory=list)
    _token_pos_buffer: list[int] = field(default_factory=list)
    _seg_history: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def from_paths(
        cls,
        model_path: Path,
        feature_cols: list[str],
        press: str,
        compression_ratio: float,
        k: int = 16,
        max_budget: int = 512,
        isotonic_path: Path | None = None,
    ) -> "StreamingHeraldPredictor":
        clf = xgb.XGBClassifier()
        clf.load_model(str(model_path))
        iso = None
        if isotonic_path is not None and isotonic_path.exists():
            iso = json.loads(isotonic_path.read_text())
        return cls(
            model=clf,
            feature_cols=list(feature_cols),
            press=press,
            compression_ratio=float(compression_ratio),
            k=k,
            max_budget=max_budget,
            isotonic=iso,
            _cheap={f: [] for f in CHEAP_FEATURES},
        )

    def reset(self) -> None:
        self._token_pos = -1
        self._cheap = {f: [] for f in CHEAP_FEATURES}
        self._token_ids = []
        self._token_pos_buffer = []
        self._seg_history = []

    def update(
        self, sig: TokenSignals, token_id: int
    ) -> dict[str, float] | None:
        """Push token; return prediction at K-token boundary, else None."""
        self._token_pos += 1
        self._token_ids.append(int(token_id))
        self._token_pos_buffer.append(self._token_pos)
        for f in CHEAP_FEATURES:
            v = getattr(sig, f, math.nan)
            if v is None:
                v = math.nan
            self._cheap[f].append(float(v))
        if (self._token_pos + 1) % self.k == 0:
            return self._predict()
        return None

    def _predict(self) -> dict[str, float]:
        seg_idx = self._token_pos // self.k
        agg = _segment_aggregate_polars(
            self._cheap, self._token_pos_buffer, seg_idx, self.k
        )
        surface = _surface_features(self._token_ids, self.k, self._token_pos)
        seg_row: dict[str, float] = {**agg, **surface, "seg_idx": seg_idx}
        self._seg_history.append(seg_row)

        seg_df = pl.from_pandas(pd.DataFrame(self._seg_history))
        ext = _apply_cross_segment_dynamics(seg_df)
        latest = ext.row(-1, named=True)

        feature_dict: dict[str, float] = {}
        for col in self.feature_cols:
            if col == "position_in_budget":
                v = float(self._token_pos + 1) / float(self.max_budget)
                feature_dict[col] = min(1.0, max(0.0, v))
            elif col == "compression_ratio":
                feature_dict[col] = float(self.compression_ratio)
            elif col.startswith("press_"):
                want = col.removeprefix("press_")
                feature_dict[col] = 1.0 if want == self.press else 0.0
            else:
                v = latest.get(col, math.nan)
                if v is None:
                    v = math.nan
                feature_dict[col] = float(v)

        vec = np.asarray(
            [feature_dict[c] for c in self.feature_cols],
            dtype=np.float32,
        ).reshape(1, -1)
        score = float(self.model.predict_proba(vec)[0, 1])

        calibrated = score
        if self.isotonic is not None:
            calibrated = _apply_isotonic(score, self.isotonic)

        return {
            "seg_idx": seg_idx,
            "seg_end_tok": self._token_pos,
            "score": score,
            "calibrated": calibrated,
        }


def _apply_isotonic(score: float, model: object) -> float:
    """Apply a serialized isotonic mapping (sorted xs+ys lookup).

    Expected format: {"x": [...], "y": [...]} both sorted ascending,
    matching `IsotonicRegression.X_thresholds_`/`y_thresholds_`. Linear
    interpolation between knots; clamped at the endpoints.
    """
    if not isinstance(model, dict):
        return score
    xs = np.asarray(model.get("x", []), dtype=np.float64)
    ys = np.asarray(model.get("y", []), dtype=np.float64)
    if xs.size == 0 or ys.size != xs.size:
        return score
    if score <= xs[0]:
        return float(ys[0])
    if score >= xs[-1]:
        return float(ys[-1])
    return float(np.interp(score, xs, ys))


def predict_run(
    pred: StreamingHeraldPredictor,
    stream: Iterable[tuple[TokenSignals, int]],
) -> list[dict[str, float]]:
    """Drain a token stream, return per-segment predictions for one run."""
    pred.reset()
    out: list[dict[str, float]] = []
    for sig, tok in stream:
        r = pred.update(sig, tok)
        if r is not None:
            out.append(r)
    return out
