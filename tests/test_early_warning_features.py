"""Tests for src/herald/early_warning_features.py.

Behaviors under test:
- causal windows: changing future tokens does not alter past EWS values
- group isolation: features do not bleed across run_id boundary
- constant-signal edge cases produce finite or null values, not NaN
- runs shorter than W return null EWS for that window without crashing
- schema hygiene: replay/label columns never enter the deployable set
"""

import math

import polars as pl

from herald.early_warning_features import (
    DEFAULT_BASE_SIGNALS,
    DEFAULT_WINDOWS,
    NON_DEPLOYABLE_COLUMNS,
    add_ews_features,
    deployable_feature_columns,
    ews_feature_names,
    is_deployable_column,
)


def _toy_df(values: list[float], run_id: str = "r") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [run_id] * len(values),
            "token_pos": list(range(len(values))),
            "entropy": values,
            "top1_prob": [v / 10.0 for v in values],
            "h_alts": [v * 0.5 for v in values],
            "delta_h": [v - 1.0 for v in values],
            "kl_div": [v * 0.1 for v in values],
        }
    )


# ---------------------------------------------------------------
# Causal window: future does not leak into past.
# ---------------------------------------------------------------


def test_changing_future_does_not_change_past_features() -> None:
    base_vals = [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0]
    altered = list(base_vals)
    # Mutate positions 6..9 (the "future" relative to position 5).
    altered[6:] = [99.0, -42.0, 77.0, -88.0]

    df_base = _toy_df(base_vals)
    df_alt = _toy_df(altered)

    out_base = add_ews_features(df_base, windows=(4,))
    out_alt = add_ews_features(df_alt, windows=(4,))

    # Inspect EWS columns at token_pos <= 5: must agree.
    cols = [c for c in out_base.columns if c.startswith("ews_")]
    base_head = out_base.filter(pl.col("token_pos") <= 5).select(cols)
    alt_head = out_alt.filter(pl.col("token_pos") <= 5).select(cols)
    assert base_head.equals(alt_head)


# ---------------------------------------------------------------
# Group isolation: run_id boundary blocks all rolling state.
# ---------------------------------------------------------------


def test_features_do_not_bleed_across_run_id() -> None:
    a = _toy_df([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], run_id="A")
    b = _toy_df([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], run_id="B")
    combined = pl.concat([a, b], how="vertical")

    out_combined = add_ews_features(combined, windows=(4,))
    out_a_alone = add_ews_features(a, windows=(4,))

    # Run A's EWS values inside the combined frame must equal the
    # standalone A values.
    cols = [c for c in out_combined.columns if c.startswith("ews_")]
    a_in_combined = (
        out_combined.filter(pl.col("run_id") == "A")
        .sort("token_pos")
        .select(cols)
    )
    a_alone = out_a_alone.sort("token_pos").select(cols)
    assert a_in_combined.equals(a_alone)


# ---------------------------------------------------------------
# Edge cases: constant signal.
# ---------------------------------------------------------------


def test_constant_signal_produces_finite_or_null_values() -> None:
    df = _toy_df([3.0] * 16)
    out = add_ews_features(df, windows=(8,))

    # Range = 0 for any window starting >= 1 row in.
    rng = out["ews_entropy_rng_8"].to_list()
    # First W-1 rows null (min_samples=W); the rest must be 0.0.
    for v in rng[:7]:
        assert v is None
    for v in rng[7:]:
        assert v is not None
        assert math.isfinite(v)
        assert abs(v) < 1e-9

    # lag1_autocorr: variance is 0, so denominator is 0 -> null
    # (we explicitly fall back to None when den == 0).
    ac = out["ews_entropy_lag1ac_8"].to_list()
    for v in ac:
        if v is not None:
            assert math.isfinite(v)

    # Slope: x is constant, so slope is 0.
    slope = out["ews_entropy_slope_8"].to_list()
    # First W-1 rows null (min_samples=W); rest must be 0.
    for v in slope[7:]:
        assert v is not None
        assert math.isfinite(v)
        assert abs(v) < 1e-9


# ---------------------------------------------------------------
# Small run does not crash and yields nulls for big windows.
# ---------------------------------------------------------------


def test_small_run_shorter_than_window_does_not_crash() -> None:
    df = _toy_df([1.0, 2.0, 3.0])  # length 3, W = 8 below
    out = add_ews_features(df, windows=(8,))
    # All EWS columns for W=8 must be null on every row.
    for c in [
        c for c in out.columns if c.startswith("ews_") and c.endswith("_8")
    ]:
        for v in out[c].to_list():
            assert v is None


# ---------------------------------------------------------------
# Slope and skew sanity on a known monotone ramp.
# ---------------------------------------------------------------


def test_slope_on_linear_ramp_is_constant_one() -> None:
    df = _toy_df([float(i) for i in range(16)])
    out = add_ews_features(df, windows=(4,))
    slope = out["ews_entropy_slope_4"].to_list()
    # First 3 rows null (min_samples=4); the rest must be ~ 1.
    for v in slope[3:]:
        assert v is not None
        assert abs(v - 1.0) < 1e-6


# ---------------------------------------------------------------
# Schema hygiene: deployable filter excludes the right columns.
# ---------------------------------------------------------------


def test_deployable_filter_drops_replay_label_and_validator_columns() -> None:
    cols = [
        # deployable
        "entropy",
        "top1_prob",
        "compression_ratio",
        "ews_entropy_skew_8",
        "entropy_mean_8",
        "token_pos",
        "relative_progress",
        # non-deployable
        "js_full",
        "kl_unc_comp_full",
        "future_sum_js_25",
        "future_max_js_50",
        "rouge_l_drop",
        "has_looping",
        "sum_js",
        "nll_ratio_flipped",
    ]
    keep = deployable_feature_columns(cols)
    for forbidden in (
        "js_full",
        "kl_unc_comp_full",
        "future_sum_js_25",
        "future_max_js_50",
        "rouge_l_drop",
        "has_looping",
        "sum_js",
        "nll_ratio_flipped",
    ):
        assert forbidden not in keep, forbidden
        assert not is_deployable_column(forbidden), forbidden
    for kept in (
        "entropy",
        "top1_prob",
        "compression_ratio",
        "ews_entropy_skew_8",
        "entropy_mean_8",
    ):
        assert kept in keep
        assert is_deployable_column(kept)


def test_non_deployable_set_covers_all_replay_and_validator_columns() -> None:
    must_have = {
        "js_full",
        "kl_unc_comp_full",
        "rouge_l_drop",
        "has_looping",
        "has_non_termination",
        "sum_js",
        "sum_kl",
        "nll_ratio",
        "nll_ratio_flipped",
        "first_divergence_point",
    }
    assert must_have.issubset(NON_DEPLOYABLE_COLUMNS)


def test_ews_feature_names_match_added_columns() -> None:
    df = _toy_df([float(i) for i in range(64)])
    out = add_ews_features(df)
    expected = set(
        ews_feature_names(
            signals=DEFAULT_BASE_SIGNALS, windows=DEFAULT_WINDOWS
        )
    )
    actual = {c for c in out.columns if c.startswith("ews_")}
    assert expected == actual
