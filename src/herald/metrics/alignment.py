"""Alignment matrix: pairwise Spearman across metric families."""

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r):
            rs.append(r)
    arr = np.asarray(rs)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def build(final: Path, out: Path) -> None:
    """Pairwise Spearman + bootstrap CI across metric families.

    `final` here is interpreted as the metrics directory containing
    `<family>_metrics.parquet` files; the CLI passes
    `metrics/` for both args.
    """
    families: dict[str, list[str]] = {
        "trajectory": [
            "sum_kl",
            "first_divergence_point",
            "nll_ratio",
        ],
        "sequence": [
            "rouge_l",
            "edit_distance_ratio",
            "embedding_cosine",
        ],
    }
    frames = []
    for fam in families:
        path = final / f"{fam}_metrics.parquet"
        if path.exists():
            frames.append(pl.read_parquet(path))
    if not frames:
        return
    df = frames[0]
    for f in frames[1:]:
        df = df.join(f, on="run_id", how="inner")

    rows = []
    metrics = [
        c for c in df.columns if c != "run_id" and df[c].dtype.is_numeric()
    ]
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            x = df[a].to_numpy()
            y = df[b].to_numpy()
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            rho = spearmanr(x, y).statistic
            lo, hi = _bootstrap_spearman(x, y)
            rows.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "spearman": float(rho),
                    "spearman_lo": lo,
                    "spearman_hi": hi,
                }
            )
    pl.DataFrame(rows).write_parquet(out / "alignment.parquet")
