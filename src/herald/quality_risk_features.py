"""Feature-roster and causality helpers for the quality-risk audit."""

import numpy as np
import pandas as pd

from herald.quality_risk_engineering import (
    ENGINEERED_NAMES,
    add_engineered_features,
)

HISTORY_BASES = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
)
WINDOWS = (8, 32)
HALF_LIVES = (8, 32)
INSTANTANEOUS_SENSORS = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
)
LABEL_COLUMNS = (
    "damage",
    "catastrophe",
    "imputed_compressed_zero",
    "lift_reference_zero",
)
V1_DIV_COLUMNS = (
    "div_band_rate_0_5",
    "div_band_rate_5_10",
    "div_band_rate_10_25",
    "div_band_rate_25_50",
)
V1_FILL_VALUE = 0.0


def history_columns() -> tuple[str, ...]:
    """Causal-history column names (must match v1's roster)."""
    names: list[str] = []
    for base in HISTORY_BASES:
        for window in WINDOWS:
            names.extend(
                (
                    f"{base}_causal_mean_{window}",
                    f"{base}_causal_std_{window}",
                )
            )
        for half_life in HALF_LIVES:
            names.append(f"{base}_causal_ewma_hl{half_life}")
    return tuple(names)


EXPECTED_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
    "fold",
    "run_length",
    "log_token_clock",
    *INSTANTANEOUS_SENSORS,
    *history_columns(),
    *ENGINEERED_NAMES,
    *LABEL_COLUMNS,
)
EXPECTED_COLUMNS_V3 = (*EXPECTED_COLUMNS, *V1_DIV_COLUMNS)
FORBIDDEN_EXACT = {
    "baseline_run_id",
    "baseline_quality_score",
    "compressed_quality_score",
    "quality_delta",
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
    "js_full",
    "kl_unc_comp_full",
    "sum_js",
    "sum_kl",
    "nll_ratio",
    "nll_ratio_flipped",
    "first_divergence_point",
    "gross_harm_final",
    "gross_help_final",
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "embedding_cosine_drop",
    "relative_progress",
    "output_length_so_far",
}
FORBIDDEN_PREFIXES = ("future_",)
RELEASED_AGGREGATE_MARKERS = ("_mean_", "_std_", "_ewma_")


def check_columns(
    columns: list[str], expected: tuple[str, ...] = EXPECTED_COLUMNS
) -> None:
    """The materialized roster is exactly expected, causal, oracle-free."""
    forbidden = [
        c
        for c in columns
        if c in FORBIDDEN_EXACT or c.startswith(FORBIDDEN_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"forbidden columns materialized: {forbidden[:5]}")
    leaked_aggregates = [
        c
        for c in columns
        if any(m in c for m in RELEASED_AGGREGATE_MARKERS)
        and "causal" not in c
    ]
    if leaked_aggregates:
        raise ValueError(
            f"released aggregate columns ingested: {leaked_aggregates[:5]}"
        )
    missing = [c for c in expected if c not in columns]
    if missing:
        raise ValueError(f"missing materialized columns: {missing[:5]}")
    unexpected = [c for c in columns if c not in expected]
    if unexpected:
        raise ValueError(f"unexpected materialized columns: {unexpected[:5]}")


def _trailing_mean_std(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    squared = np.concatenate(
        ([0.0], np.cumsum(clean * clean, dtype=np.float64))
    )
    counts = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    right = np.arange(1, len(values) + 1)
    left = np.maximum(right - window, 0)
    count = counts[right] - counts[left]
    total = cumulative[right] - cumulative[left]
    total_squared = squared[right] - squared[left]
    mean = np.divide(
        total,
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    )
    numerator = total_squared - np.divide(
        total * total,
        count,
        out=np.zeros(len(values), dtype=np.float64),
        where=count > 0,
    )
    variance = np.divide(
        np.maximum(numerator, 0.0),
        count - 1,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 1,
    )
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _ewma(values: np.ndarray, half_life: int) -> np.ndarray:
    alpha = 1.0 - np.exp(np.log(0.5) / half_life)
    output = np.full(len(values), np.nan, dtype=np.float32)
    state = np.nan
    for index, value in enumerate(values):
        if np.isfinite(value):
            state = (
                value
                if np.isnan(state)
                else alpha * value + (1.0 - alpha) * state
            )
        output[index] = state
    return output


def verify_causal_sample(
    materialized: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    tolerance: float = 2e-6,
) -> float:
    """Recompute causal history from raw sensors; return max error."""
    materialized = materialized.sort_values("token_pos", ignore_index=True)
    raw = raw.sort_values("token_pos", ignore_index=True)
    if not np.array_equal(
        materialized["token_pos"].to_numpy(), raw["token_pos"].to_numpy()
    ):
        raise ValueError("sample token identity mismatch")
    worst = 0.0
    for base in HISTORY_BASES:
        values = raw[base].to_numpy(dtype=np.float64)
        for window in WINDOWS:
            mean, std = _trailing_mean_std(values, window)
            for name, expected in (
                (f"{base}_causal_mean_{window}", mean),
                (f"{base}_causal_std_{window}", std),
            ):
                actual = materialized[name].to_numpy(dtype=np.float64)
                both_nan = np.isnan(actual) & np.isnan(expected)
                error = np.nanmax(
                    np.where(both_nan, 0.0, np.abs(actual - expected)),
                    initial=0.0,
                )
                worst = max(worst, float(error))
                if not np.allclose(
                    actual,
                    expected,
                    rtol=1e-5,
                    atol=tolerance,
                    equal_nan=True,
                ):
                    raise ValueError(f"causal recomputation mismatch: {name}")
        for half_life in HALF_LIVES:
            name = f"{base}_causal_ewma_hl{half_life}"
            expected = _ewma(values, half_life)
            actual = materialized[name].to_numpy(dtype=np.float64)
            worst = max(
                worst,
                float(
                    np.nanmax(
                        np.where(
                            np.isnan(actual) & np.isnan(expected),
                            0.0,
                            np.abs(actual - expected),
                        ),
                        initial=0.0,
                    )
                ),
            )
            if not np.allclose(
                actual, expected, rtol=1e-5, atol=tolerance, equal_nan=True
            ):
                raise ValueError(f"causal recomputation mismatch: {name}")
    if "log_token_clock" in materialized.columns:
        clock = np.log1p(
            raw["token_pos"].to_numpy(dtype=np.float64) + 1.0
        ).astype(np.float32)
        if not np.allclose(
            materialized["log_token_clock"].to_numpy(dtype=np.float64),
            clock.astype(np.float64),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError("log_token_clock recomputation mismatch")
    return worst


def verify_engineered_sample(
    materialized: pd.DataFrame,
    raw: pd.DataFrame,
) -> float:
    """Recompute engineered features from raw sensors; return max error."""
    materialized = materialized.sort_values("token_pos", ignore_index=True)
    raw = raw.sort_values("token_pos", ignore_index=True)
    if not np.array_equal(
        materialized["token_pos"].to_numpy(), raw["token_pos"].to_numpy()
    ):
        raise ValueError("sample token identity mismatch")
    recomputed = add_engineered_features(raw.copy())
    worst = 0.0
    for name in ENGINEERED_NAMES:
        actual = materialized[name].to_numpy(dtype=np.float64)
        expected = recomputed[name].to_numpy(dtype=np.float64)
        both_nan = np.isnan(actual) & np.isnan(expected)
        if not np.allclose(
            actual, expected, rtol=1e-5, atol=2e-6, equal_nan=True
        ):
            raise ValueError(f"engineered recomputation mismatch: {name}")
        error = np.nanmax(
            np.where(both_nan, 0.0, np.abs(actual - expected)), initial=0.0
        )
        worst = max(worst, float(error))
    return worst


V1_PRED_COLUMNS = (
    "pred_causal_xgb_band_rate_0_5",
    "pred_causal_xgb_band_rate_5_10",
    "pred_causal_xgb_band_rate_10_25",
    "pred_causal_xgb_band_rate_25_50",
)


def verify_divergence_sample(
    materialized: pd.DataFrame,
    v1: pd.DataFrame,
    scales: list[float],
) -> float:
    """Recompute joined divergence columns; return max error.

    v1 holds raw OOF band predictions for one run; missing bands must
    appear filled with V1_FILL_VALUE and no missingness may be stored.
    """
    if len(scales) != len(V1_PRED_COLUMNS):
        raise ValueError("v1 calibration scale roster changed")
    joined = materialized[["token_pos", *V1_DIV_COLUMNS]].merge(
        v1[["token_pos", *V1_PRED_COLUMNS]],
        on="token_pos",
        how="left",
        validate="one_to_one",
    )
    worst = 0.0
    for name, source, scale in zip(
        V1_DIV_COLUMNS, V1_PRED_COLUMNS, scales, strict=True
    ):
        actual = joined[name].to_numpy(dtype=np.float64)
        raw = joined[source].to_numpy(dtype=np.float64)
        expected = np.where(np.isnan(raw), V1_FILL_VALUE, raw * scale)
        if not np.allclose(actual, expected, rtol=1e-5, atol=2e-6):
            raise ValueError(f"divergence join mismatch: {name}")
        worst = max(worst, float(np.abs(actual - expected).max(initial=0.0)))
    return worst
