"""Phase 1 stratified alignment matrix.

Extends the Phase 0 alignment template (`herald.metrics.alignment.build`)
with:

  - Stratification by (task, press, ratio). Cells with fewer than
    `--min-cell-n` runs are skipped (recorded in a side file).
  - A "global pooled" row that aggregates everything for parity with
    the Phase 0 single-row matrix.
  - Bootstrap 95% CIs over runs (default 1000 iterations).
  - Intrinsic-to-extrinsic damage: pairs intrinsic trajectory metrics
    against the binary "outcome harm" label
    (baseline_correct AND NOT compressed_correct), reporting both
    Spearman ρ and AUROC for each (task, press, ratio).

Inputs:
  - `<root>/final/runs.parquet` (from `metrics finalize`).
  - `<root>/metrics/trajectory_metrics.parquet` (from `metrics build`).
  - `<root>/metrics/sequence_metrics.parquet` (optional; included in the
    pairwise matrix when present).

Outputs:
  - `<root>/metrics/alignment_phase1.parquet` — pairwise Spearman per
    stratum (and a `task=global, press=global, ratio=-1` row for the
    pooled matrix).
  - `<root>/metrics/intrinsic_to_outcome_phase1.parquet` — Spearman +
    AUROC of each intrinsic metric against `outcome_harm` per stratum.
  - `<root>/metrics/alignment_phase1_skipped.json` — strata that fell
    below `--min-cell-n` or had insufficient data.

The script is CPU-only and safe to run on Mac after rsync from Orion.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

INTRINSIC_METRICS: tuple[str, ...] = (
    "sum_kl",
    "sum_js",
    "nll_ratio",
    "first_divergence_point",
)
SEQUENCE_METRICS: tuple[str, ...] = (
    "rouge_l",
    "embedding_cosine",
    "edit_distance_ratio",
)


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    point = spearmanr(x, y).statistic
    if not np.isfinite(point):
        return float("nan"), float("nan"), float("nan")
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            rs.append(r)
    if not rs:
        return float(point), float("nan"), float("nan")
    arr = np.asarray(rs)
    return (
        float(point),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def _bootstrap_auroc(
    score: np.ndarray, label: np.ndarray, n: int, seed: int
) -> tuple[float, float, float]:
    if len(np.unique(label)) < 2:
        return float("nan"), float("nan"), float("nan")
    point = roc_auc_score(label, score)
    rng = np.random.default_rng(seed)
    aurocs = []
    for _ in range(n):
        idx = rng.integers(0, len(label), len(label))
        if len(np.unique(label[idx])) < 2:
            continue
        aurocs.append(roc_auc_score(label[idx], score[idx]))
    if not aurocs:
        return float(point), float("nan"), float("nan")
    arr = np.asarray(aurocs)
    return (
        float(point),
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
    )


def _load(
    root: Path,
) -> tuple[pl.DataFrame, list[str], list[str]]:
    runs_path = root / "final" / "runs.parquet"
    traj_path = root / "metrics" / "trajectory_metrics.parquet"
    seq_path = root / "metrics" / "sequence_metrics.parquet"
    if not runs_path.exists():
        sys.exit(f"missing {runs_path}")
    if not traj_path.exists():
        sys.exit(f"missing {traj_path}")

    runs = pl.read_parquet(runs_path).select(
        [
            "run_id",
            "task",
            "press",
            "compression_ratio",
            "baseline_run_id",
            "correct",
        ]
    )
    traj = pl.read_parquet(traj_path)
    intrinsic_cols = [c for c in INTRINSIC_METRICS if c in traj.columns]

    base_correct = (
        runs.filter(pl.col("press") == "none")
        .select(["run_id", "correct"])
        .rename({"run_id": "baseline_run_id", "correct": "baseline_correct"})
    )
    df = (
        runs.filter(pl.col("press") != "none")
        .join(traj, on="run_id", how="inner")
        .join(base_correct, on="baseline_run_id", how="left")
    )
    df = df.with_columns(
        (
            pl.col("baseline_correct").fill_null(False)
            & ~pl.col("correct").fill_null(False)
        )
        .cast(pl.Int8)
        .alias("outcome_harm")
    )

    seq_cols: list[str] = []
    if seq_path.exists():
        seq = pl.read_parquet(seq_path)
        seq_cols = [c for c in SEQUENCE_METRICS if c in seq.columns]
        if seq_cols:
            df = df.join(
                seq.select(["run_id", *seq_cols]),
                on="run_id",
                how="left",
            )

    return df, intrinsic_cols, seq_cols


def _strata_iter(
    df: pl.DataFrame,
) -> Iterable[tuple[tuple[str, str, float], pl.DataFrame]]:
    yield (
        ("__pooled__", "__pooled__", -1.0),
        df,
    )
    for keys, sub in df.group_by(["task"], maintain_order=True):
        (task,) = keys
        yield ((str(task), "__pooled__", -1.0), sub)
    for keys, sub in df.group_by(
        ["task", "press", "compression_ratio"], maintain_order=True
    ):
        task, press, ratio = keys
        yield ((str(task), str(press), float(ratio)), sub)


def _pair_rows(
    sub: pl.DataFrame,
    metric_cols: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for i, a in enumerate(metric_cols):
        for b in metric_cols[i + 1 :]:
            x = sub[a].to_numpy().astype(float)
            y = sub[b].to_numpy().astype(float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                continue
            rho, lo, hi = _bootstrap_spearman(
                x[mask], y[mask], n=n_boot, seed=seed
            )
            out.append(
                {
                    "metric_a": a,
                    "metric_b": b,
                    "n": int(mask.sum()),
                    "spearman": rho,
                    "spearman_lo": lo,
                    "spearman_hi": hi,
                }
            )
    return out


def _intrinsic_outcome_rows(
    sub: pl.DataFrame,
    intrinsic_cols: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    label = sub["outcome_harm"].to_numpy().astype(int)
    out: list[dict[str, object]] = []
    for col in intrinsic_cols:
        score = sub[col].to_numpy().astype(float)
        mask = np.isfinite(score)
        if mask.sum() < 5:
            continue
        rho, rho_lo, rho_hi = _bootstrap_spearman(
            score[mask], label[mask].astype(float), n=n_boot, seed=seed
        )
        auroc, au_lo, au_hi = _bootstrap_auroc(
            score[mask], label[mask], n=n_boot, seed=seed
        )
        out.append(
            {
                "metric": col,
                "n": int(mask.sum()),
                "n_pos": int(label[mask].sum()),
                "spearman_vs_outcome_harm": rho,
                "spearman_lo": rho_lo,
                "spearman_hi": rho_hi,
                "auroc_outcome_harm": auroc,
                "auroc_lo": au_lo,
                "auroc_hi": au_hi,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--min-cell-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, intrinsic_cols, seq_cols = _load(args.root)
    print(f"runs after join: {df.height}")
    print(f"intrinsic columns: {intrinsic_cols}")
    print(f"sequence columns:  {seq_cols}")

    pair_metric_cols = list(intrinsic_cols) + list(seq_cols)
    pair_rows: list[dict[str, object]] = []
    intrinsic_rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for (task, press, ratio), sub in _strata_iter(df):
        if task != "__pooled__" and sub.height < args.min_cell_n:
            skipped.append(
                {
                    "task": task,
                    "press": press,
                    "compression_ratio": ratio,
                    "n": sub.height,
                    "reason": "below_min_cell_n",
                }
            )
            continue

        pairs = _pair_rows(
            sub, pair_metric_cols, n_boot=args.n_boot, seed=args.seed
        )
        for r in pairs:
            r.update(
                {"task": task, "press": press, "compression_ratio": ratio}
            )
        pair_rows.extend(pairs)

        intr = _intrinsic_outcome_rows(
            sub, intrinsic_cols, n_boot=args.n_boot, seed=args.seed
        )
        for r in intr:
            r.update(
                {"task": task, "press": press, "compression_ratio": ratio}
            )
        intrinsic_rows.extend(intr)

    out_dir = args.root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame(pair_rows).write_parquet(
        out_dir / "alignment_phase1.parquet"
    )
    pl.DataFrame(intrinsic_rows).write_parquet(
        out_dir / "intrinsic_to_outcome_phase1.parquet"
    )
    (out_dir / "alignment_phase1_skipped.json").write_text(
        json.dumps(skipped, indent=2)
    )
    print(f"Wrote {len(pair_rows)} pairwise rows.")
    print(f"Wrote {len(intrinsic_rows)} intrinsic-vs-outcome rows.")
    print(f"Skipped strata: {len(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
