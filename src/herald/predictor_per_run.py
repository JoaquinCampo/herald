"""Per-run aggregation of Phase 2 token-level predictions, and the
user-facing validation against run_damage.parquet (Task 4 in the
Phase 2 pipeline).

The Phase 2 baselines train at the token level. To validate that
those token-level predictions correspond to *user-facing* damage,
we aggregate per-token scores up to per-run scores (max, mean,
p95, mean of top-k, count above threshold) and correlate them with
the run-level damage validators in `run_damage.parquet`:

- Spearman vs `quality_delta`
- AUROC/AUPRC vs `gross_harm_final`
- AUROC/AUPRC vs catastrophic tags (`has_looping`,
  `has_non_termination`, `has_format_break`, `has_drift`)

This is not a training objective. It is the evidence that the
intrinsic future-window label corresponds to harm a user notices.

Spec: gold/phase-2-dataset.md, gold/research-plan.md Phase 2
"User-facing validation".
"""

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def aggregate_per_run(
    df: pl.DataFrame,
    score_col: str = "score",
    top_k: int = 10,
    threshold: float = 0.5,
) -> pl.DataFrame:
    """Per (run_id) aggregates of token-level scores."""
    if "run_id" not in df.columns:
        raise ValueError("aggregate_per_run: input must have run_id")
    if score_col not in df.columns:
        raise ValueError(f"aggregate_per_run: missing {score_col!r}")
    above_col = f"n_above_{threshold:g}"
    return df.group_by("run_id").agg(
        pl.col(score_col).max().alias("max_score"),
        pl.col(score_col).mean().alias("mean_score"),
        pl.col(score_col)
        .quantile(0.95, interpolation="linear")
        .alias("p95_score"),
        pl.col(score_col)
        .top_k(top_k)
        .mean()
        .alias(f"mean_top_{top_k}_score"),
        (pl.col(score_col) >= threshold).sum().alias(above_col),
    )


def _safe_auroc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_auprc(y_true: list[int], y_score: list[float]) -> float | None:
    if len(y_true) < 2 or len(set(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def _safe_spearman(
    a: np.ndarray, b: np.ndarray
) -> tuple[float | None, float | None]:
    if a.size < 3:
        return None, None
    try:
        rho, p = spearmanr(a, b)
        if rho is None or np.isnan(rho):
            return None, None
        return float(rho), float(p)
    except Exception:
        return None, None


CONTINUOUS_VALIDATORS: tuple[str, ...] = (
    "quality_delta",
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "sum_kl",
    "sum_js",
    "nll_ratio_flipped",
    "compressed_quality_score",
)
BINARY_VALIDATORS: tuple[str, ...] = (
    "gross_harm_final",
    "gross_help_final",
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
)


def correlate_with_run_damage(
    per_run: pl.DataFrame,
    run_damage: pl.DataFrame,
    score_col: str = "max_score",
) -> dict[str, Any]:
    """For one per-run aggregate column, correlate with run_damage.

    For continuous validators: Spearman ρ. For binary validators:
    AUROC + AUPRC. Each metric drops rows where either side is null.
    `n_used_*` is recorded so the reader can spot small-sample CIs.
    """
    if score_col not in per_run.columns:
        raise ValueError(f"correlate_with_run_damage: missing {score_col!r}")

    joined = per_run.join(run_damage, on="run_id", how="inner")
    out: dict[str, Any] = {
        "score_col": score_col,
        "n_runs_joined": int(joined.height),
    }

    for col in CONTINUOUS_VALIDATORS:
        if col not in joined.columns:
            continue
        sub = joined.drop_nulls(subset=[score_col, col])
        rho, p = _safe_spearman(
            sub[score_col].to_numpy(),
            sub[col].to_numpy(),
        )
        out[f"spearman_{col}"] = round(rho, 4) if rho is not None else None
        out[f"spearman_p_{col}"] = round(p, 4) if p is not None else None
        out[f"n_used_{col}"] = int(sub.height)

    for col in BINARY_VALIDATORS:
        if col not in joined.columns:
            continue
        sub = joined.drop_nulls(subset=[score_col, col])
        if sub.is_empty():
            out[f"auroc_{col}"] = None
            out[f"auprc_{col}"] = None
            out[f"n_used_{col}"] = 0
            continue
        y = sub[col].cast(pl.Int8).to_list()
        s = sub[score_col].to_list()
        a = _safe_auroc(y, s)
        p = _safe_auprc(y, s)
        out[f"auroc_{col}"] = round(a, 4) if a is not None else None
        out[f"auprc_{col}"] = round(p, 4) if p is not None else None
        out[f"n_used_{col}"] = int(sub.height)
        out[f"n_pos_{col}"] = int(sum(y))

    return out


__all__ = [
    "BINARY_VALIDATORS",
    "CONTINUOUS_VALIDATORS",
    "aggregate_per_run",
    "correlate_with_run_damage",
]
