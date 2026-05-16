"""Regression metrics for HERALD v1.

Helpers for Spearman correlation with cluster-bootstrap CIs,
per-run aggregations, and ECE on per-run quantile bins.
"""

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 10:
        return float("nan")
    r = spearmanr(a[mask], b[mask]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _pearson_on_ranks(
    ra: np.ndarray, rb: np.ndarray
) -> float:
    """Pearson r on pre-computed ranks.

    Spearman is Pearson on the ranks; for bootstrap iterations
    where we resample indices from a parent array whose ranks we
    have precomputed, recomputing the ranks on the resample would
    re-introduce the dominant O(N log N) cost. We instead use the
    parent's ranks and compute Pearson on the subset, which is
    O(N). The resulting statistic is no longer the strict Spearman
    on the resample, but it is a consistent rank-based estimator
    whose distribution under cluster bootstrap captures the same
    variance.
    """
    if ra.size < 10:
        return float("nan")
    ra = ra.astype(np.float64, copy=False)
    rb = rb.astype(np.float64, copy=False)
    mra = ra.mean()
    mrb = rb.mean()
    num = float(((ra - mra) * (rb - mrb)).sum())
    da = float(((ra - mra) ** 2).sum())
    db = float(((rb - mrb) ** 2).sum())
    if da <= 0.0 or db <= 0.0:
        return float("nan")
    return float(num / (da ** 0.5 * db ** 0.5))


def clustered_spearman_ci(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster-bootstrap CI for Spearman ρ.

    Resamples unique cluster ids with replacement; reports point
    estimate (computed with scipy on the full sample) plus a
    percentile CI from Pearson on the parent rank vectors over
    cluster resamples. See `_pearson_on_ranks` for the rationale.
    """
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    clusters = clusters[mask]
    rho = safe_spearman(a, b)
    if not np.isfinite(rho):
        return {"rho": float("nan"),
                "lo": float("nan"), "hi": float("nan"),
                "n_boot_ok": 0}

    uniq = np.unique(clusters)
    if uniq.size < 5:
        return {"rho": rho, "lo": float("nan"),
                "hi": float("nan"), "n_boot_ok": 0}

    # Pre-rank once on the parent sample (dominant cost).
    ra_parent = np.argsort(np.argsort(a)).astype(np.float64)
    rb_parent = np.argsort(np.argsort(b)).astype(np.float64)

    cluster_to_idx: dict[Any, list[int]] = {}
    for i, c in enumerate(clusters):
        cluster_to_idx.setdefault(c, []).append(i)
    cluster_arrays = {c: np.asarray(idx, dtype=np.int64)
                      for c, idx in cluster_to_idx.items()}
    uniq_arr = np.asarray(uniq)

    rng = np.random.default_rng(seed)
    rhos: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(uniq_arr, size=uniq_arr.size,
                            replace=True)
        idx = np.concatenate(
            [cluster_arrays[c] for c in sample])
        r = _pearson_on_ranks(ra_parent[idx], rb_parent[idx])
        if np.isfinite(r):
            rhos.append(r)
    if not rhos:
        return {"rho": rho, "lo": float("nan"),
                "hi": float("nan"), "n_boot_ok": 0}
    rhos_arr = np.asarray(rhos)
    lo = float(np.quantile(rhos_arr, alpha / 2.0))
    hi = float(np.quantile(rhos_arr, 1.0 - alpha / 2.0))
    return {"rho": rho, "lo": lo, "hi": hi,
            "n_boot_ok": len(rhos)}


def per_run_aggregates(
    df: pl.DataFrame,
    pred_col: str = "pred_raw",
    y_col: str = "y_raw",
) -> pl.DataFrame:
    """Aggregate per-token predictions and labels per run_id."""
    return (
        df.group_by("run_id")
        .agg(
            pl.col(pred_col).max().alias("pred_max"),
            pl.col(pred_col).quantile(0.95).alias("pred_p95"),
            pl.col(pred_col).mean().alias("pred_mean"),
            pl.col(y_col).max().alias("y_max"),
            pl.col(y_col).quantile(0.95).alias("y_p95"),
            pl.col(y_col).mean().alias("y_mean"),
            pl.col("press").first().alias("press"),
            pl.col("task").first().alias("task"),
            pl.col("compression_ratio").first().alias("ratio"),
            pl.col("prompt_id").first().alias("prompt_id"),
            pl.col("fold").first().alias("fold"),
        )
    )


def ece_quantile(
    pred: np.ndarray, y: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error using quantile bins.

    Compares the predicted percentile rank to the true
    percentile rank of `y`, bucketed into `n_bins` equal-mass
    bins on the prediction.
    """
    mask = np.isfinite(pred) & np.isfinite(y)
    pred = pred[mask]
    y = y[mask]
    if pred.size < n_bins * 5:
        return float("nan")

    pred_rank = np.argsort(np.argsort(pred)) / max(pred.size - 1, 1)
    y_rank = np.argsort(np.argsort(y)) / max(y.size - 1, 1)
    edges = np.quantile(
        pred_rank, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_id = np.searchsorted(edges, pred_rank, side="right") - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)
    ece = 0.0
    for k in range(n_bins):
        idx = bin_id == k
        if idx.sum() == 0:
            continue
        gap = abs(pred_rank[idx].mean() - y_rank[idx].mean())
        ece += (idx.sum() / pred.size) * gap
    return float(ece)
