"""Online (streaming) feature state for the Phase 4 controller.

Reproduces the Phase 2 cheap-feature schema token by token, so the
runtime predictor sees the same feature vector that
`lr_all_cheap` was trained on (24 columns).

Key invariant: the produced feature vector is bit-equivalent (within
the parity thresholds documented in the implementation plan) to the
offline `phase2_tokens.parquet` row at the same `token_pos`, on
non-null inputs.

Implementation choice: incremental scalar/ring-buffer state, *not*
re-running the polars pipeline per step. This keeps update() O(1)
amortized in feature count and W=32, so per-token use stays under
the millisecond budget called for in the ultimate-goal doc.

NaN-handling tracks the offline polars pipeline exactly. The Phase 2
parquet stores NaN (not null) for missing values like `delta_h`,
`kl_div`, `top10_jaccard` at t=0. Polars treats NaN as a numeric
value, so NaN propagates through rolling-window sums and the EWMA
recursion: any window touching a NaN input yields NaN, and once a
NaN enters EWMA state every subsequent output is NaN.

This module mirrors that: deque entries are floats (NaN allowed),
`_rolling_mean` and `_rolling_std` return NaN when any window
value is NaN, EWMA state stays NaN once seeded with NaN. The
parity test against `phase2_tokens.parquet` therefore agrees on
both NaN and non-NaN cells.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from herald.config import TokenSignals

# Mirrors src/herald/predictor_dataset.py — keep in sync.
ROLLING_TARGETS: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
)
ROLLING_WINDOWS: tuple[int, ...] = (8, 32)
EWMA_HALF_LIVES: tuple[int, ...] = (8, 32)

# Mirrors src/herald/predictor_baselines.py CHEAP_ALL_FEATURES.
# Order is load-bearing: it determines the column order of the
# numpy feature vector consumed by the exported lr_all_cheap model.
CHEAP_ALL_FEATURE_ORDER: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "kl_div",
    "top10_jaccard",
    "tail_mass",
    "logit_range",
    "eff_vocab_size",
    "entropy_mean_8",
    "entropy_std_8",
    "entropy_mean_32",
    "entropy_ewma_hl8",
    "entropy_ewma_hl32",
    "top1_prob_mean_8",
    "top1_prob_mean_32",
    "top1_prob_ewma_hl8",
    "kl_div_mean_8",
    "h_alts_mean_8",
    "top10_jaccard_mean_8",
    "top10_jaccard_mean_32",
    "token_pos",
    "relative_progress",
    "compression_ratio",
)


def _ewma_alpha(half_life: int) -> float:
    return float(1.0 - 0.5 ** (1.0 / half_life))


@dataclass(slots=True)
class _RollingState:
    """Per-target ring buffer + scalar EWMA state."""

    values: deque[float] = field(
        default_factory=lambda: deque(maxlen=max(ROLLING_WINDOWS))
    )
    ewma: dict[int, float | None] = field(
        default_factory=lambda: {hl: None for hl in EWMA_HALF_LIVES}
    )


def _rolling_mean(buf: deque[float], window: int) -> float:
    """Mean of last `window` values; NaN if any value is NaN.

    Mirrors polars `rolling_mean(window_size=window, min_samples=1)`
    on a NaN-containing column: NaN propagates through the sum so
    any window touching a NaN yields NaN. Empty buffer also -> NaN.
    """
    if window <= 0 or not buf:
        return math.nan
    tail = list(buf)[-window:]
    n = len(tail)
    s = 0.0
    for v in tail:
        if math.isnan(v):
            return math.nan
        s += v
    return s / n


def _rolling_std(buf: deque[float], window: int) -> float:
    """Sample std (ddof=1) of last `window` values.

    Returns NaN when any value is NaN, or when the window has fewer
    than 2 elements (matches polars `min_samples=2` default).
    """
    if window <= 0 or not buf:
        return math.nan
    tail = list(buf)[-window:]
    n = len(tail)
    if n < 2:
        return math.nan
    s = 0.0
    for v in tail:
        if math.isnan(v):
            return math.nan
        s += v
    mean = s / n
    var = sum((v - mean) ** 2 for v in tail) / (n - 1)
    return math.sqrt(var)


def _ewma_step(state: float | None, x: float, alpha: float) -> float:
    """y_0 = x (first call); y_t = (1-alpha)*y_{t-1} + alpha*x thereafter.

    NaN propagates after seeding: once state is NaN, output stays NaN
    even when subsequent x is finite. Matches polars
    `ewm_mean(adjust=False)` over NaN-containing columns.
    """
    if state is None:
        return float(x)
    if math.isnan(state) or math.isnan(x):
        return math.nan
    return (1.0 - alpha) * state + alpha * float(x)


def _to_float(x: Any) -> float:
    """Coerce TokenSignals attribute to float, NaN for None."""
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


@dataclass(slots=True)
class OnlineFeatureState:
    """Streaming feature state for one run.

    Usage:
        state = OnlineFeatureState(
            press="streaming_llm",
            compression_ratio=0.875,
            max_new_tokens=512,
        )
        for sig in token_signal_stream:
            row = state.update(sig)
            vec = state.feature_vector()  # 24 floats, NaN -> 0.0

    `update` returns the dict of all CHEAP_ALL features at this token.
    `feature_vector` returns the numpy-friendly fixed-order vector.
    """

    press: str
    compression_ratio: float
    max_new_tokens: int
    _token_pos: int = -1
    _state: dict[str, _RollingState] = field(default_factory=dict)
    _last_features: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._state = {t: _RollingState() for t in ROLLING_TARGETS}
        self._last_features = {c: math.nan for c in CHEAP_ALL_FEATURE_ORDER}

    def update(self, sig: TokenSignals) -> dict[str, float]:
        """Advance one token and return the row dict.

        The `sig` is the same TokenSignals produced by
        `herald.signals.extract_signals`; nulls (for `delta_h`,
        `kl_div`, `top10_jaccard` at t=0) propagate as None.
        """
        self._token_pos += 1
        token_pos = self._token_pos

        tier0 = {
            "entropy": _to_float(sig.entropy),
            "top1_prob": _to_float(sig.top1_prob),
            "top5_prob": _to_float(sig.top5_prob),
            "h_alts": _to_float(sig.h_alts),
            "kl_div": _to_float(sig.kl_div),
            "top10_jaccard": _to_float(sig.top10_jaccard),
            "tail_mass": _to_float(sig.tail_mass),
            "logit_range": _to_float(sig.logit_range),
            "eff_vocab_size": _to_float(sig.eff_vocab_size),
        }
        # delta_h is part of ROLLING_TARGETS but not exported in
        # CHEAP_ALL_FEATURE_ORDER. Track its value for state, no
        # output column. delta_h_valid=False means delta_h is not
        # meaningful (t=0 or numeric issue) -> NaN, like the offline
        # parquet stores.
        delta_h_val = (
            _to_float(sig.delta_h) if sig.delta_h_valid else math.nan
        )

        # Push into ring buffers.
        target_values = {
            "entropy": tier0["entropy"],
            "top1_prob": tier0["top1_prob"],
            "h_alts": tier0["h_alts"],
            "delta_h": delta_h_val,
            "kl_div": tier0["kl_div"],
            "top10_jaccard": tier0["top10_jaccard"],
        }
        for target, val in target_values.items():
            st = self._state[target]
            st.values.append(val)
            for hl in EWMA_HALF_LIVES:
                st.ewma[hl] = _ewma_step(st.ewma[hl], val, _ewma_alpha(hl))

        # Build the 24-feature row.
        row: dict[str, float] = dict(tier0)
        # rolling/EWMA outputs for the columns that lr_all_cheap uses.
        row["entropy_mean_8"] = _rolling_mean(
            self._state["entropy"].values, 8
        )
        row["entropy_std_8"] = _rolling_std(self._state["entropy"].values, 8)
        row["entropy_mean_32"] = _rolling_mean(
            self._state["entropy"].values, 32
        )
        ent8 = self._state["entropy"].ewma[8]
        ent32 = self._state["entropy"].ewma[32]
        row["entropy_ewma_hl8"] = math.nan if ent8 is None else ent8
        row["entropy_ewma_hl32"] = math.nan if ent32 is None else ent32
        row["top1_prob_mean_8"] = _rolling_mean(
            self._state["top1_prob"].values, 8
        )
        row["top1_prob_mean_32"] = _rolling_mean(
            self._state["top1_prob"].values, 32
        )
        tp8 = self._state["top1_prob"].ewma[8]
        row["top1_prob_ewma_hl8"] = math.nan if tp8 is None else tp8
        row["kl_div_mean_8"] = _rolling_mean(self._state["kl_div"].values, 8)
        row["h_alts_mean_8"] = _rolling_mean(self._state["h_alts"].values, 8)
        row["top10_jaccard_mean_8"] = _rolling_mean(
            self._state["top10_jaccard"].values, 8
        )
        row["top10_jaccard_mean_32"] = _rolling_mean(
            self._state["top10_jaccard"].values, 32
        )
        row["token_pos"] = float(token_pos)
        denom = max(self.max_new_tokens, 1)
        row["relative_progress"] = float(token_pos) / float(denom)
        row["compression_ratio"] = float(self.compression_ratio)

        self._last_features = row
        return row

    def feature_vector(self, fill: float = 0.0) -> list[float]:
        """Return the 24-feature vector in CHEAP_ALL_FEATURE_ORDER.

        Null/NaN entries are replaced with `fill` (default 0.0,
        matching `predictor_baselines.directional_score` behavior).
        Caller is responsible for applying the lr_all_cheap scaler
        and inverse-feature negation downstream.
        """
        out: list[float] = []
        for col in CHEAP_ALL_FEATURE_ORDER:
            v = self._last_features.get(col, math.nan)
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                out.append(fill)
            else:
                out.append(f)
        return out

    @property
    def token_pos(self) -> int:
        return self._token_pos
