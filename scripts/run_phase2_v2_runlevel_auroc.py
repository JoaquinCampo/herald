"""Compute run-level AUROC on held-out runs from per-segment scores.

Goal-aligned framing per `gold/phase-4-controller-design.md` line 45:
"Run score: max segment score seen so far." The deployable controller
decision is run-level: at any point during decoding, the controller
acts based on `max segment_risk over all completed segments so far`.
The catastrophe-onset task on held-out runs is therefore naturally
evaluated at the run level, not per segment.

For each fold's held-out runs:
  - run_score = max(segment score) over the run's pre-onset segments
  - run_label = 1 iff the run is loop-first (any segment has y=1)
  - Compute AUROC + AUPRC + cluster-bootstrap CI by run_id

The lead time is K=16 tokens (one segment ahead), exactly the
controller's reaction window: it observes risk at the end of segment N
and acts on segment N+1's compression event.

Inputs: results/phase2_v2/xgb_ext_loop/scores_fold{0..4}.parquet
Output: results/phase2_v2/xgb_ext_loop/runlevel_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        vals.append(metric_fn(yt, ys))
    if not vals:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(vals)),
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("results/phase2_v2/xgb_ext_loop"),
    )
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase2_v2/xgb_ext_loop/runlevel_summary.json"
        ),
    )
    args = ap.parse_args()

    fold_summaries: list[dict[str, object]] = []
    for f in range(args.n_folds):
        path = args.scores_dir / f"scores_fold{f}.parquet"
        df = pd.read_parquet(path)
        logger.info(
            "fold {}: rows={} runs={}",
            f,
            len(df),
            df["run_id"].nunique(),
        )
        agg = df.groupby("run_id", sort=False).agg(
            run_score=("score", "max"),
            run_label=("y", "max"),
            n_seg=("seg_idx", "count"),
        )
        y_run = agg["run_label"].to_numpy(dtype=int)
        s_run = agg["run_score"].to_numpy(dtype=float)
        n_runs = len(agg)
        n_pos = int(y_run.sum())
        n_neg = int(n_runs - n_pos)
        if n_pos == 0 or n_pos == n_runs:
            logger.warning(
                "fold {} degenerate: pos={} neg={}",
                f,
                n_pos,
                n_neg,
            )
            fold_summaries.append(
                {
                    "fold": f,
                    "n_runs": int(n_runs),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "skipped": True,
                }
            )
            continue
        auroc = float(roc_auc_score(y_run, s_run))
        auprc = float(average_precision_score(y_run, s_run))
        bs_au = cluster_bootstrap_ci(
            y_run, s_run, roc_auc_score, args.n_boot
        )
        bs_pr = cluster_bootstrap_ci(
            y_run, s_run, average_precision_score, args.n_boot
        )
        logger.info(
            "fold {}: runs={} pos={} neg={}  "
            "AUROC={:.4f} (CI {:.4f}-{:.4f})  "
            "AUPRC={:.4f} (CI {:.4f}-{:.4f})",
            f,
            n_runs,
            n_pos,
            n_neg,
            auroc,
            bs_au[1],
            bs_au[2],
            auprc,
            bs_pr[1],
            bs_pr[2],
        )
        fold_summaries.append(
            {
                "fold": f,
                "n_runs": int(n_runs),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_rate": float(n_pos / n_runs),
                "auroc": auroc,
                "auroc_ci_low": bs_au[1],
                "auroc_ci_high": bs_au[2],
                "auprc": auprc,
                "auprc_ci_low": bs_pr[1],
                "auprc_ci_high": bs_pr[2],
                "skipped": False,
            }
        )

    valid = [s for s in fold_summaries if not s.get("skipped")]
    aurocs = [s["auroc"] for s in valid]
    auprcs = [s["auprc"] for s in valid]

    pooled_y: list[int] = []
    pooled_s: list[float] = []
    for f in range(args.n_folds):
        path = args.scores_dir / f"scores_fold{f}.parquet"
        df = pd.read_parquet(path)
        agg = df.groupby("run_id", sort=False).agg(
            run_score=("score", "max"),
            run_label=("y", "max"),
        )
        pooled_y.extend(agg["run_label"].astype(int).tolist())
        pooled_s.extend(agg["run_score"].astype(float).tolist())
    py = np.asarray(pooled_y, dtype=int)
    ps = np.asarray(pooled_s, dtype=float)
    pooled_auroc = float(roc_auc_score(py, ps))
    pooled_auprc = float(average_precision_score(py, ps))
    pooled_au_ci = cluster_bootstrap_ci(
        py, ps, roc_auc_score, args.n_boot
    )
    pooled_pr_ci = cluster_bootstrap_ci(
        py, ps, average_precision_score, args.n_boot
    )
    logger.info(
        "POOLED runs={} pos={} neg={}  AUROC={:.4f} (CI {:.4f}-{:.4f})"
        "  AUPRC={:.4f} (CI {:.4f}-{:.4f})",
        len(py),
        int(py.sum()),
        int((py == 0).sum()),
        pooled_auroc,
        pooled_au_ci[1],
        pooled_au_ci[2],
        pooled_auprc,
        pooled_pr_ci[1],
        pooled_pr_ci[2],
    )
    pooled = {
        "n_runs": int(len(py)),
        "n_pos": int(py.sum()),
        "n_neg": int((py == 0).sum()),
        "auroc": pooled_auroc,
        "auroc_ci_low": pooled_au_ci[1],
        "auroc_ci_high": pooled_au_ci[2],
        "auprc": pooled_auprc,
        "auprc_ci_low": pooled_pr_ci[1],
        "auprc_ci_high": pooled_pr_ci[2],
    }

    out = {
        "n_folds": len(fold_summaries),
        "n_valid_folds": len(valid),
        "n_boot": args.n_boot,
        "lead_time_tokens": 16,
        "lead_time_segments": 1,
        "rationale": (
            "K=16 tokens = controller reaction window: risk observed "
            "at end of segment N drives the compression action on "
            "segment N+1. Per gold/phase-4-controller-design.md "
            "line 41 'Segment size: K=16' and line 75 controller "
            "decision after every K=16 generated tokens."
        ),
        "scoring": (
            "Run score = max(segment score) over the run's pre-onset "
            "segments, per controller design line 45. Run label = 1 "
            "if loop-first (max y over segments), 0 if clean."
        ),
        "fold_summary": fold_summaries,
        "pooled": pooled,
        "macro_mean_auroc": float(np.mean(aurocs)) if aurocs else float("nan"),
        "macro_std_auroc": float(np.std(aurocs)) if aurocs else float("nan"),
        "macro_mean_auprc": float(np.mean(auprcs)) if auprcs else float("nan"),
        "macro_std_auprc": float(np.std(auprcs)) if auprcs else float("nan"),
        "bar_auroc": 0.96,
    }
    out["target_met_pooled"] = pooled_auroc >= 0.96
    out["target_met_macro"] = (
        out["macro_mean_auroc"] >= 0.96
        if not np.isnan(out["macro_mean_auroc"])
        else False
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info(
        "RUN-LEVEL macro AUROC={:.4f} (std {:.4f})  AUPRC={:.4f}  "
        "POOLED AUROC={:.4f}  BAR={:.4f}  POOLED_MET={}  MACRO_MET={}",
        out["macro_mean_auroc"],
        out["macro_std_auroc"],
        out["macro_mean_auprc"],
        pooled_auroc,
        0.96,
        out["target_met_pooled"],
        out["target_met_macro"],
    )
    logger.info("Wrote {}", args.output)


if __name__ == "__main__":
    main()
