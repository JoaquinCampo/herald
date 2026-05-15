"""Run-level meta-aggregator over per-segment OOF scores.

Per locked §6.5 protocol in
`gold/phase-2d-streaming-online-results.md`: train a meta-classifier
on 8 online-faithful run-level aggregates of the OOF per-segment
XGBoost scores in `scores_fold{f}.parquet`, with a no-leakage 5-fold
split that respects the base 5-fold GroupKFold by `prompt_id`.

Online-faithful aggregates (locked, 8 features) computed over the
run's pre-onset segment scores `s_1, ..., s_T`:
  max, running_mean, running_std, top3_mean, top5_mean,
  last3_mean, last5_mean, n_segs.

All are O(1) incrementally computable as new segments arrive at
deploy time; no look-ahead, no full-sequence stats.

No-leakage protocol: per-segment OOF scores in `scores_fold{f}` were
produced by base XGBoost trained on the OTHER 4 folds. For each
fold f, fit the meta-classifier on aggregates from the OTHER 4
folds and evaluate on fold f. Any other split leaks base->meta.

Model ladder (locked):
  1. Logistic regression with StandardScaler on the 8-dim vector.
  2. If LR macro AUROC < 0.96, ONE adjustment: XGBoost meta with
     max_depth <= 4, n_estimators <= 100, default lr.

Headline: macro mean run-level AUROC across 5 folds; AUPRC
alongside; cluster bootstrap CI by run_id, n_boot=2000.

Inputs : results/phase2_v2/xgb_ext_loop/scores_fold{0..4}.parquet
Output : results/phase2_v2/xgb_ext_loop/meta_aggregator_summary.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "max",
    "running_mean",
    "running_std",
    "top3_mean",
    "top5_mean",
    "last3_mean",
    "last5_mean",
    "n_segs",
]


def aggregate_run(scores: np.ndarray) -> dict[str, float]:
    """Compute the 8 locked online-faithful aggregates for one run.

    `scores` is the ordered per-segment score vector for the run's
    pre-onset segments (the parquet rows ARE pre-onset segments by
    construction of the Phase 2 onset-anchored task).
    """
    n = int(scores.shape[0])
    sorted_desc = np.sort(scores)[::-1]
    k3 = min(3, n)
    k5 = min(5, n)
    top3 = float(sorted_desc[:k3].mean())
    top5 = float(sorted_desc[:k5].mean())
    last3 = float(scores[-k3:].mean())
    last5 = float(scores[-k5:].mean())
    return {
        "max": float(scores.max()),
        "running_mean": float(scores.mean()),
        "running_std": float(scores.std(ddof=0)),
        "top3_mean": top3,
        "top5_mean": top5,
        "last3_mean": last3,
        "last5_mean": last5,
        "n_segs": float(n),
    }


def build_fold_aggregates(path: Path) -> pd.DataFrame:
    """Group per-segment scores into one row per run with aggregates."""
    df = pd.read_parquet(path)
    df = df.sort_values(["run_id", "seg_idx"], kind="stable")
    rows: list[dict[str, float | str | int]] = []
    for run_id, sub in df.groupby("run_id", sort=False):
        s = sub["score"].to_numpy(dtype=float)
        feats = aggregate_run(s)
        feats["run_id"] = str(run_id)
        feats["y"] = int(sub["y"].max())
        rows.append(feats)
    out = pd.DataFrame(rows)
    return out


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI by run_id (each row IS a run, so resample rows)."""
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
        return float("nan"), float("nan")
    return (
        float(np.percentile(vals, 2.5)),
        float(np.percentile(vals, 97.5)),
    )


def evaluate_meta(
    fold_dfs: list[pd.DataFrame],
    model_factory,
    n_boot: int,
    label: str,
) -> dict[str, object]:
    """No-leakage 5-fold meta evaluation.

    For each fold f: fit on OTHER 4 folds' aggregates, predict on f.
    Returns per-fold metrics, macro mean/std, pooled, and pooled CI.
    """
    fold_summaries: list[dict[str, object]] = []
    pooled_y: list[int] = []
    pooled_s: list[float] = []
    for f, df_f in enumerate(fold_dfs):
        train_df = pd.concat(
            [d for i, d in enumerate(fold_dfs) if i != f],
            ignore_index=True,
        )
        x_tr = train_df[FEATURES].to_numpy(dtype=float)
        y_tr = train_df["y"].to_numpy(dtype=int)
        x_ev = df_f[FEATURES].to_numpy(dtype=float)
        y_ev = df_f["y"].to_numpy(dtype=int)
        model = model_factory()
        model.fit(x_tr, y_tr)
        if hasattr(model, "predict_proba"):
            s_ev = model.predict_proba(x_ev)[:, 1]
        else:
            s_ev = model.decision_function(x_ev)
        n_runs = len(y_ev)
        n_pos = int(y_ev.sum())
        n_neg = int(n_runs - n_pos)
        auroc = float(roc_auc_score(y_ev, s_ev))
        auprc = float(average_precision_score(y_ev, s_ev))
        au_ci = cluster_bootstrap_ci(
            y_ev, s_ev, roc_auc_score, n_boot, seed=f
        )
        pr_ci = cluster_bootstrap_ci(
            y_ev,
            s_ev,
            average_precision_score,
            n_boot,
            seed=f + 1000,
        )
        logger.info(
            "[{}] fold {}: runs={} pos={} neg={}  "
            "AUROC={:.4f} (CI {:.4f}-{:.4f})  "
            "AUPRC={:.4f} (CI {:.4f}-{:.4f})",
            label,
            f,
            n_runs,
            n_pos,
            n_neg,
            auroc,
            au_ci[0],
            au_ci[1],
            auprc,
            pr_ci[0],
            pr_ci[1],
        )
        fold_summaries.append(
            {
                "fold": f,
                "n_runs": int(n_runs),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "pos_rate": float(n_pos / n_runs),
                "auroc": auroc,
                "auroc_ci_low": au_ci[0],
                "auroc_ci_high": au_ci[1],
                "auprc": auprc,
                "auprc_ci_low": pr_ci[0],
                "auprc_ci_high": pr_ci[1],
            }
        )
        pooled_y.extend(y_ev.tolist())
        pooled_s.extend(s_ev.tolist())

    aurocs = [s["auroc"] for s in fold_summaries]
    auprcs = [s["auprc"] for s in fold_summaries]
    py = np.asarray(pooled_y, dtype=int)
    ps = np.asarray(pooled_s, dtype=float)
    pooled_auroc = float(roc_auc_score(py, ps))
    pooled_auprc = float(average_precision_score(py, ps))
    pooled_au_ci = cluster_bootstrap_ci(
        py, ps, roc_auc_score, n_boot, seed=9999
    )
    pooled_pr_ci = cluster_bootstrap_ci(
        py, ps, average_precision_score, n_boot, seed=9998
    )
    macro_mean = float(np.mean(aurocs))
    macro_std = float(np.std(aurocs))
    logger.info(
        "[{}] MACRO AUROC={:.4f} (std {:.4f})  POOLED AUROC={:.4f} "
        "(CI {:.4f}-{:.4f})  POOLED AUPRC={:.4f}",
        label,
        macro_mean,
        macro_std,
        pooled_auroc,
        pooled_au_ci[0],
        pooled_au_ci[1],
        pooled_auprc,
    )
    return {
        "label": label,
        "fold_summary": fold_summaries,
        "macro_mean_auroc": macro_mean,
        "macro_std_auroc": macro_std,
        "macro_mean_auprc": float(np.mean(auprcs)),
        "macro_std_auprc": float(np.std(auprcs)),
        "pooled": {
            "n_runs": int(len(py)),
            "n_pos": int(py.sum()),
            "n_neg": int((py == 0).sum()),
            "auroc": pooled_auroc,
            "auroc_ci_low": pooled_au_ci[0],
            "auroc_ci_high": pooled_au_ci[1],
            "auprc": pooled_auprc,
            "auprc_ci_low": pooled_pr_ci[0],
            "auprc_ci_high": pooled_pr_ci[1],
        },
    }


def lr_factory():
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def xgb_factory():
    from xgboost import XGBClassifier

    return XGBClassifier(
        max_depth=4,
        n_estimators=100,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=0,
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
        "--bar-auroc",
        type=float,
        default=0.96,
        help="Locked target macro AUROC (do not change).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase2_v2/xgb_ext_loop/meta_aggregator_summary.json"
        ),
    )
    args = ap.parse_args()

    logger.info("Loading per-segment OOF scores and aggregating...")
    fold_dfs: list[pd.DataFrame] = []
    for f in range(args.n_folds):
        path = args.scores_dir / f"scores_fold{f}.parquet"
        df = build_fold_aggregates(path)
        logger.info(
            "fold {}: aggregated {} runs (pos {})",
            f,
            len(df),
            int(df["y"].sum()),
        )
        fold_dfs.append(df)

    logger.info("Step 1/2: Logistic regression meta...")
    lr_summary = evaluate_meta(fold_dfs, lr_factory, args.n_boot, "LR")

    out: dict[str, object] = {
        "n_folds": args.n_folds,
        "n_boot": args.n_boot,
        "bar_auroc": args.bar_auroc,
        "features": FEATURES,
        "no_leakage_protocol": (
            "For each fold f, meta is fit on aggregates from the "
            "OTHER 4 folds and evaluated on f. The base XGBoost "
            "scores in scores_fold{f}.parquet are themselves OOF "
            "(GroupKFold by prompt_id), so meta-fit on OTHER 4 "
            "folds never sees its evaluation runs at any stage."
        ),
        "scoring": (
            "Per-run online-faithful aggregates over pre-onset "
            "per-segment scores. 8 features locked: "
            + ", ".join(FEATURES)
            + ". Headline = macro mean AUROC across 5 folds."
        ),
        "lr_result": lr_summary,
    }

    if lr_summary["macro_mean_auroc"] < args.bar_auroc:
        logger.info(
            "LR macro AUROC {:.4f} < bar {:.2f}; running ONE "
            "ladder adjustment (XGBoost meta).",
            lr_summary["macro_mean_auroc"],
            args.bar_auroc,
        )
        xgb_summary = evaluate_meta(fold_dfs, xgb_factory, args.n_boot, "XGB")
        out["xgb_result"] = xgb_summary
        best_label = (
            "XGB"
            if xgb_summary["macro_mean_auroc"]
            > lr_summary["macro_mean_auroc"]
            else "LR"
        )
        best_macro = max(
            xgb_summary["macro_mean_auroc"],
            lr_summary["macro_mean_auroc"],
        )
    else:
        best_label = "LR"
        best_macro = lr_summary["macro_mean_auroc"]

    out["best_model"] = best_label
    out["best_macro_mean_auroc"] = float(best_macro)
    out["target_met_macro"] = bool(best_macro >= args.bar_auroc)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))

    logger.info(
        "META best={} macro AUROC={:.4f}  BAR={:.2f}  MET={}",
        best_label,
        best_macro,
        args.bar_auroc,
        out["target_met_macro"],
    )
    logger.info("Wrote {}", args.output)


if __name__ == "__main__":
    main()
