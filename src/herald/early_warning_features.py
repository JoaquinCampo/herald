"""Phase 2c early-warning-signal (EWS) features (CPU-only).

Augments the Phase 2 per-token dataset with rolling instability
statistics from dynamical-systems collapse theory. Hypothesis: classic
early-warning indicators (variance / skew / autocorrelation /
flickering) capture pre-collapse dynamics that the existing rolling
mean / EWMA features miss.

Spec: gold/phase-2c-early-warning-results.md (verdict doc).

Deployable input only. The base signals are cheap per-token logit
features observable at decode time:
    entropy, top1_prob, h_alts, delta_h, kl_div
where `kl_div` is the *consecutive-timestep* instability KL (Tier 1
cheap signal, computed in herald.signals), NOT the replay-based
`kl_unc_comp_full`. Replay JS / KL and the future_* labels are
forbidden as inputs and are never read by this module.

For each (base signal, window W in {8, 16, 32, 64}):
    rolling_skew_W            (third-moment shape)
    rolling_slope_W           (linear-regression slope vs token_pos)
    rolling_lag1_autocorr_W   (corr(x_t, x_{t-1}) over W)
    rolling_max_minus_min_W   (range)
    flicker_count_W           (median-crossings over W)

std / var are deliberately omitted: the Phase 2 dataset already
carries entropy_std_8/32, top1_prob_std_8/32, etc. Adding std at new
windows would be redundant explosion. flicker_rate_W = flicker_count_W
/ W is also omitted as a linear duplicate of flicker_count_W.

Causality (load-bearing):
- All temporal expressions are scoped with `.over("run_id")` so a row
  in run A never reads from run B.
- `rolling_*` uses the closed-window-ending-at-t convention: the
  feature at token_pos=t depends only on positions <= t in the same
  run.
- Slope and autocorr require the full window present and are null for
  positions where the run has fewer than W realised tokens before t.
- Flicker uses the rolling median of the same window as a causal
  reference; the count over W is itself a rolling window of crossing
  events, so it is also null until W tokens of history exist.
"""

from collections.abc import Iterable

import polars as pl

DEFAULT_BASE_SIGNALS: tuple[str, ...] = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
)

DEFAULT_WINDOWS: tuple[int, ...] = (8, 16, 32, 64)

# Columns that must NEVER be used as deployable inputs.  This is
# the schema-hygiene set audited by tests + the dataset builder.
NON_DEPLOYABLE_COLUMNS: frozenset[str] = frozenset(
    {
        # Replay-derived quantities (require uncompressed counterfactual).
        "js_full",
        "kl_unc_comp_full",
        # Run-level validators (labels, not inputs).
        "gross_harm_final",
        "gross_help_final",
        "quality_delta",
        "compressed_quality_score",
        "baseline_quality_score",
        "rouge_l_drop",
        "char_edit_ratio",
        "length_diff_ratio",
        "embedding_cosine_drop",
        "has_looping",
        "has_non_termination",
        "has_format_break",
        "has_drift",
        "sum_kl",
        "sum_js",
        "nll_ratio",
        "nll_ratio_flipped",
        "first_divergence_point",
    }
)

# Future-window labels follow the future_*_H pattern; treat any
# column starting with "future_" as a label.
LABEL_PREFIX: str = "future_"


def is_deployable_column(name: str) -> bool:
    """True iff `name` may be used as a predictor input feature."""
    if name in NON_DEPLOYABLE_COLUMNS:
        return False
    if name.startswith(LABEL_PREFIX):
        return False
    return True


# ---------------------------------------------------------------
# Per-family expression builders.
#
# Each builder takes the source column and a window size, returns
# a polars expression.  All temporal ops MUST be partitioned by
# run_id at the caller site via `.over("run_id")`.
# ---------------------------------------------------------------


def _slope_expr(col: str, w: int) -> pl.Expr:
    """OLS slope of `col` vs token_pos over the trailing W tokens.

    Closed-form via rolling sums of t, t^2, y, t*y.  Requires the
    full window (min_samples=W) to be present; partial windows
    return null.  Scoped over run_id at the caller.
    """
    t = pl.col("token_pos").cast(pl.Float64)
    y = pl.col(col).cast(pl.Float64)
    sum_y = y.rolling_sum(window_size=w, min_samples=w)
    sum_t = t.rolling_sum(window_size=w, min_samples=w)
    sum_ty = (t * y).rolling_sum(window_size=w, min_samples=w)
    sum_tt = (t * t).rolling_sum(window_size=w, min_samples=w)
    n = float(w)
    num = n * sum_ty - sum_t * sum_y
    den = n * sum_tt - sum_t * sum_t
    # den is constant for a strictly-monotone integer index of W
    # rows, but token_pos can have gaps if upstream filters skip
    # rows; we fall back to safe division.
    return pl.when(den.abs() > 0).then(num / den).otherwise(None)


def _lag1_autocorr_expr(col: str, w: int) -> pl.Expr:
    """corr(x_t, x_{t-1}) over a trailing window of W rows.

    Computed as cov(x, x_lag) / (std(x) * std(x_lag)) with rolling
    means and rolling stds.  Requires min_samples=W.  Scoped over
    run_id at the caller (and the lag itself uses .over("run_id")
    so the first row of a run does not pull from the previous one).
    """
    x = pl.col(col).cast(pl.Float64)
    lag = x.shift(1).over("run_id")
    mean_x = x.rolling_mean(window_size=w, min_samples=w)
    mean_l = lag.rolling_mean(window_size=w, min_samples=w)
    mean_xl = (x * lag).rolling_mean(window_size=w, min_samples=w)
    std_x = x.rolling_std(window_size=w, min_samples=w)
    std_l = lag.rolling_std(window_size=w, min_samples=w)
    cov = mean_xl - mean_x * mean_l
    den = std_x * std_l
    return pl.when(den.abs() > 0).then(cov / den).otherwise(None)


def _max_minus_min_expr(col: str, w: int) -> pl.Expr:
    """Rolling range over a trailing window of W rows."""
    x = pl.col(col).cast(pl.Float64)
    return x.rolling_max(window_size=w, min_samples=w) - x.rolling_min(
        window_size=w, min_samples=w
    )


def _flicker_count_expr(col: str, w: int) -> pl.Expr:
    """Count of median-crossing events in the trailing W tokens.

    Local reference is the rolling median of the same window
    (causal).  A crossing event at token t is a sign-change in
    `signal - rolling_median_W` between t-1 and t.  The flicker
    count at token t is the rolling sum of crossing indicators
    over the trailing W tokens.  Null until both the median and
    its lag are well-defined.
    """
    x = pl.col(col).cast(pl.Float64)
    med = x.rolling_median(window_size=w, min_samples=w)
    dev = x - med
    sign_curr = dev > 0
    sign_prev = dev.shift(1).over("run_id") > 0
    crossing = (sign_curr != sign_prev).cast(pl.Int32)
    # First valid row of each median has no prior median-deviation;
    # drop those crossings to None to prevent spurious flicker at
    # the warm-up boundary.
    valid = dev.is_not_null() & dev.shift(1).over("run_id").is_not_null()
    crossing_valid = pl.when(valid).then(crossing).otherwise(None)
    return crossing_valid.rolling_sum(window_size=w, min_samples=w)


# ---------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------


def ews_feature_names(
    signals: Iterable[str] = DEFAULT_BASE_SIGNALS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Canonical EWS feature column names for the given config.

    Used by the dataset builder, the audit, and the predictor
    runner so the three never disagree about the EWS feature set.
    """
    families = ("skew", "slope", "lag1ac", "rng", "flicker")
    out: list[str] = []
    for sig in signals:
        for w in windows:
            for fam in families:
                out.append(f"ews_{sig}_{fam}_{w}")
    return out


def add_ews_features(
    df: pl.DataFrame,
    signals: Iterable[str] = DEFAULT_BASE_SIGNALS,
    windows: Iterable[int] = DEFAULT_WINDOWS,
) -> pl.DataFrame:
    """Add early-warning-signal features in place.

    Input must have `run_id` and `token_pos`.  Signals that are not
    present in `df` are silently skipped (so this stays usable on
    minimal fixtures).  All temporal ops are partitioned by run_id.
    Returns a new DataFrame.
    """
    if "run_id" not in df.columns:
        raise ValueError("add_ews_features: input must have run_id")
    if "token_pos" not in df.columns:
        raise ValueError("add_ews_features: input must have token_pos")

    df = df.sort(["run_id", "token_pos"])

    exprs: list[pl.Expr] = []
    for sig in signals:
        if sig not in df.columns:
            continue
        for w in windows:
            x = pl.col(sig).cast(pl.Float64)
            exprs.append(
                x.rolling_skew(window_size=w)
                .over("run_id")
                .alias(f"ews_{sig}_skew_{w}")
            )
            exprs.append(
                _slope_expr(sig, w)
                .over("run_id")
                .alias(f"ews_{sig}_slope_{w}")
            )
            exprs.append(
                _lag1_autocorr_expr(sig, w)
                .over("run_id")
                .alias(f"ews_{sig}_lag1ac_{w}")
            )
            exprs.append(
                _max_minus_min_expr(sig, w)
                .over("run_id")
                .alias(f"ews_{sig}_rng_{w}")
            )
            exprs.append(
                _flicker_count_expr(sig, w)
                .over("run_id")
                .alias(f"ews_{sig}_flicker_{w}")
            )

    if not exprs:
        return df
    return df.with_columns(exprs)


def deployable_feature_columns(columns: Iterable[str]) -> list[str]:
    """Filter `columns` to those usable as predictor inputs.

    Drops replay-derived quantities, run-level validators, and any
    `future_*` label.  Keeps everything else (Tier 0 logit features,
    rolling/EWMA features, EWS features, position/ratio metadata).
    """
    return [c for c in columns if is_deployable_column(c)]


__all__ = [
    "DEFAULT_BASE_SIGNALS",
    "DEFAULT_WINDOWS",
    "LABEL_PREFIX",
    "NON_DEPLOYABLE_COLUMNS",
    "add_ews_features",
    "deployable_feature_columns",
    "ews_feature_names",
    "is_deployable_column",
]
