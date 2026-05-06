"""Phase 1 corrected alignment: intrinsic metrics vs final-grader labels.

Mirrors ``scripts/build_phase1_alignment.py`` but consumes the
post-hoc rescored ``run_damage.parquet`` instead of the placeholder
``correct`` field on ``runs.parquet``. The pre-registered Phase 1 bar
(AUROC ≥ 0.65 with bootstrap CI lower bound > 0.5; ≥ 3 of 4 tasks)
does NOT change; only the extrinsic label substitutes.

Outputs:

  - ``<root>/metrics/alignment_phase1.final.parquet`` — pairwise
    Spearman over intrinsic + severity columns per stratum.
  - ``<root>/metrics/intrinsic_to_outcome_phase1.final.parquet`` —
    Spearman + AUROC of each intrinsic metric against
    ``gross_harm_final`` and Spearman against the continuous
    ``quality_delta`` per stratum.

Strict-grader artifacts at ``alignment_phase1.parquet`` and
``intrinsic_to_outcome_phase1.parquet`` are not touched.
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
SEVERITY_METRICS: tuple[str, ...] = (
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "embedding_cosine_drop",
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


def _strata_iter(
    df: pl.DataFrame,
) -> Iterable[tuple[tuple[str, str, float], pl.DataFrame]]:
    yield (("__pooled__", "__pooled__", -1.0), df)
    for keys, sub in df.group_by(["task"], maintain_order=True):
        (task,) = keys
        yield ((str(task), "__pooled__", -1.0), sub)
    for keys, sub in df.group_by(
        ["task", "press", "compression_ratio"], maintain_order=True
    ):
        task, press, ratio = keys
        yield ((str(task), str(press), float(ratio)), sub)


def _present_metric_cols(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    intrinsic = [c for c in INTRINSIC_METRICS if c in df.columns]
    severity = [c for c in SEVERITY_METRICS if c in df.columns]
    return intrinsic, severity


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


def _intrinsic_vs_final_rows(
    sub: pl.DataFrame,
    intrinsic_cols: list[str],
    n_boot: int,
    seed: int,
) -> list[dict[str, object]]:
    """Spearman + AUROC vs gross_harm_final, plus Spearman vs quality_delta.

    Rows where the binary label is undefined (None) are dropped from
    the binary-label computation but kept for the continuous one if
    quality_delta is defined.
    """
    out: list[dict[str, object]] = []

    # Binary harm (drop undefined rows).
    binary = sub.filter(pl.col("gross_harm_final").is_not_null())
    binary_label = binary["gross_harm_final"].to_numpy().astype(int)

    # Continuous quality_delta (drop nulls).
    cont = sub.filter(pl.col("quality_delta").is_not_null())
    cont_delta = cont["quality_delta"].to_numpy().astype(float)

    for col in intrinsic_cols:
        # Binary AUROC + Spearman vs gross_harm_final.
        if binary.height >= 5 and col in binary.columns:
            score = binary[col].to_numpy().astype(float)
            mask = np.isfinite(score)
            if mask.sum() >= 5:
                rho, rho_lo, rho_hi = _bootstrap_spearman(
                    score[mask],
                    binary_label[mask].astype(float),
                    n=n_boot,
                    seed=seed,
                )
                auroc, au_lo, au_hi = _bootstrap_auroc(
                    score[mask], binary_label[mask], n=n_boot, seed=seed
                )
                out.append(
                    {
                        "metric": col,
                        "label": "gross_harm_final",
                        "n": int(mask.sum()),
                        "n_pos": int(binary_label[mask].sum()),
                        "spearman": rho,
                        "spearman_lo": rho_lo,
                        "spearman_hi": rho_hi,
                        "auroc": auroc,
                        "auroc_lo": au_lo,
                        "auroc_hi": au_hi,
                    }
                )

        # Continuous Spearman vs quality_delta.
        if cont.height >= 5 and col in cont.columns:
            score = cont[col].to_numpy().astype(float)
            mask = np.isfinite(score) & np.isfinite(cont_delta)
            if mask.sum() >= 5:
                rho, rho_lo, rho_hi = _bootstrap_spearman(
                    score[mask], cont_delta[mask], n=n_boot, seed=seed
                )
                out.append(
                    {
                        "metric": col,
                        "label": "quality_delta",
                        "n": int(mask.sum()),
                        "n_pos": None,
                        "spearman": rho,
                        "spearman_lo": rho_lo,
                        "spearman_hi": rho_hi,
                        "auroc": float("nan"),
                        "auroc_lo": float("nan"),
                        "auroc_hi": float("nan"),
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

    rd_path = args.root / "metrics" / "run_damage.parquet"
    if not rd_path.exists():
        sys.exit(f"missing {rd_path}; run scripts/build_run_damage.py first.")

    df = pl.read_parquet(rd_path)
    intrinsic_cols, severity_cols = _present_metric_cols(df)
    print(f"runs: {df.height}")
    print(f"intrinsic columns: {intrinsic_cols}")
    print(f"severity columns:  {severity_cols}")

    pair_metric_cols = list(intrinsic_cols) + list(severity_cols)
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

        intr = _intrinsic_vs_final_rows(
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
        out_dir / "alignment_phase1.final.parquet"
    )
    pl.DataFrame(intrinsic_rows).write_parquet(
        out_dir / "intrinsic_to_outcome_phase1.final.parquet"
    )
    (out_dir / "alignment_phase1.final.skipped.json").write_text(
        json.dumps(skipped, indent=2)
    )
    print(f"Wrote {len(pair_rows)} pairwise rows.")
    print(f"Wrote {len(intrinsic_rows)} intrinsic-vs-final rows.")
    print(f"Skipped strata: {len(skipped)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
