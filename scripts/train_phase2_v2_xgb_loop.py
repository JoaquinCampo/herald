"""Train and evaluate a LOOP-SPECIALIZED per-segment K=16 onset predictor.

Same protocol as `train_phase2_v2_xgb.py` (XGBoost + GroupKFold(5) by
prompt_id + cluster-bootstrap CIs) but the dataset is filtered to:

  - loop-first runs: looping_onset is the first catastrophe (loop_onset
    is set AND (nt_onset is null OR nt_onset > loop_onset))
  - clean runs: both onsets null

NT-first runs are dropped: those exist only because budget cap fires
before loop manifests, and they would pollute the "loop in next K"
label.

For the kept rows, the existing `y` (= first_onset in next K) is
exactly "loop in next K" because first_onset == looping_onset on the
kept subset.

Outputs:
  {output-dir}/{model_fold{i}.json, scores_fold{i}.parquet,
                summary.json, importance.json}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

LEAK_OR_META = {
    "run_id",
    "prompt_id",
    "press",
    "looping_onset",
    "non_termination_onset",
    "first_onset",
    "num_tokens_generated",
    "relative_progress",
    "y",
}


def cluster_bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: np.ndarray,
    metric_fn,
    n_boot: int = 200,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    uniq_g = np.unique(groups)
    vals: list[float] = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq_g, size=len(uniq_g), replace=True)
        mask = np.isin(groups, sampled)
        if y_true[mask].sum() == 0 or y_true[mask].sum() == mask.sum():
            continue
        vals.append(metric_fn(y_true[mask], y_score[mask]))
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
        "--input",
        type=Path,
        default=Path("results/phase2_v2/segments_k16_ext.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2_v2/xgb_ext_loop"),
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-budget", type=int, default=512)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading {}", args.input)
    df = pd.read_parquet(args.input)
    logger.info("rows={} cols={}", *df.shape)

    # Loop-first or clean: drop NT-first runs.
    # Run-level onsets are constant within a run; pick from first row.
    run_onsets = df.groupby("run_id", sort=False).first()
    loop = run_onsets["looping_onset"]
    nt = run_onsets["non_termination_onset"]
    loop_first = loop.notna() & (nt.isna() | (nt > loop))
    clean = loop.isna() & nt.isna()
    keep_runs = run_onsets.index[loop_first | clean]
    n_loop = int(loop_first.sum())
    n_clean = int(clean.sum())
    n_nt = int(((nt.notna()) & (loop.isna() | (loop > nt))).sum())
    logger.info(
        "Run filter: loop-first={} clean={} nt-first-dropped={}",
        n_loop,
        n_clean,
        n_nt,
    )

    df = df[df["run_id"].isin(set(keep_runs))].reset_index(drop=True)
    logger.info("After run filter: rows={}", len(df))

    df["position_in_budget"] = (
        df["seg_end_tok"].astype(float) / float(args.max_budget)
    ).clip(0.0, 1.0)

    presses = sorted(df["press"].unique())
    for p in presses:
        df[f"press_{p}"] = (df["press"] == p).astype(float)

    feature_cols = [
        c
        for c in df.columns
        if c not in LEAK_OR_META and not c.startswith("press_")
    ]
    feature_cols += [f"press_{p}" for p in presses]
    feature_cols = sorted(set(feature_cols))
    logger.info("n_features={}", len(feature_cols))

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=int)
    groups = df["prompt_id"].to_numpy()

    pos_rate = y.mean()
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0
    logger.info(
        "pos_rate={:.4%} scale_pos_weight={:.2f}", pos_rate, spw
    )

    gkf = GroupKFold(n_splits=args.n_splits)
    fold_summary: list[dict[str, object]] = []
    importances: dict[str, float] = {f: 0.0 for f in feature_cols}
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        logger.info("=== fold {} ===", fold_i)
        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            scale_pos_weight=spw,
            tree_method="hist",
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="aucpr",
            early_stopping_rounds=30,
        )
        clf.fit(
            X[tr],
            y[tr],
            eval_set=[(X[te], y[te])],
            verbose=False,
        )
        score = clf.predict_proba(X[te])[:, 1]
        auroc = roc_auc_score(y[te], score)
        auprc = average_precision_score(y[te], score)
        bs_auroc = cluster_bootstrap_ci(
            y[te], score, groups[te], roc_auc_score, args.n_boot
        )
        bs_auprc = cluster_bootstrap_ci(
            y[te],
            score,
            groups[te],
            average_precision_score,
            args.n_boot,
        )
        logger.info(
            "fold {}: AUROC={:.4f} (CI {:.4f}-{:.4f})  "
            "AUPRC={:.4f} (CI {:.4f}-{:.4f})",
            fold_i,
            auroc,
            bs_auroc[1],
            bs_auroc[2],
            auprc,
            bs_auprc[1],
            bs_auprc[2],
        )
        fold_summary.append(
            {
                "fold": fold_i,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "pos_train": int(y[tr].sum()),
                "pos_test": int(y[te].sum()),
                "auroc": float(auroc),
                "auprc": float(auprc),
                "auroc_ci_low": bs_auroc[1],
                "auroc_ci_high": bs_auroc[2],
                "auprc_ci_low": bs_auprc[1],
                "auprc_ci_high": bs_auprc[2],
                "best_iteration": int(clf.best_iteration),
            }
        )
        clf.save_model(args.output_dir / f"model_fold{fold_i}.json")
        for f, imp in zip(
            feature_cols, clf.feature_importances_, strict=True
        ):
            importances[f] += float(imp)
        pd.DataFrame(
            {
                "run_id": df.iloc[te]["run_id"].values,
                "seg_idx": df.iloc[te]["seg_idx"].values,
                "y": y[te],
                "score": score,
            }
        ).to_parquet(
            args.output_dir / f"scores_fold{fold_i}.parquet",
            index=False,
        )

    importances = {
        k: v / args.n_splits for k, v in importances.items()
    }
    auroc_vals = [f["auroc"] for f in fold_summary]
    auprc_vals = [f["auprc"] for f in fold_summary]
    overall = {
        "n_folds": len(fold_summary),
        "n_features": len(feature_cols),
        "n_loop_first_runs": n_loop,
        "n_clean_runs": n_clean,
        "n_nt_first_runs_dropped": n_nt,
        "feature_cols": feature_cols,
        "auroc_mean": float(np.mean(auroc_vals)),
        "auroc_std": float(np.std(auroc_vals)),
        "auprc_mean": float(np.mean(auprc_vals)),
        "auprc_std": float(np.std(auprc_vals)),
        "fold_summary": fold_summary,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(overall, indent=2)
    )
    (args.output_dir / "importance.json").write_text(
        json.dumps(
            dict(sorted(importances.items(), key=lambda kv: -kv[1])),
            indent=2,
        )
    )
    logger.info(
        "LOOP-ONLY AUROC mean={:.4f} std={:.4f}  "
        "AUPRC mean={:.4f} std={:.4f}",
        overall["auroc_mean"],
        overall["auroc_std"],
        overall["auprc_mean"],
        overall["auprc_std"],
    )


if __name__ == "__main__":
    main()
