"""Per-press loop-specialized diagnostic, fold 0 only.

For each press in the dataset, filter to that press, apply the same
loop-only run filter as `train_phase2_v2_xgb_loop.py` (drop NT-first
runs), build features, train XGBoost on fold 0 of GroupKFold(5),
report AUROC + AUPRC + cluster-bootstrap CI.

Headline metric: macro mean AUROC across presses. The bar is locked
in `gold/phase-2d-streaming-online-results.md` Section 6.1 BEFORE
this script is run, to avoid post-hoc gaming.

Usage:
  uv run python scripts/train_phase2_v2_xgb_loop_per_press.py \\
    --input results/phase2_v2/segments_k16_ext.parquet \\
    --output results/phase2_v2/per_press_loop_fold0.json
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


def filter_loop_only(df: pd.DataFrame) -> pd.DataFrame:
    run_onsets = df.groupby("run_id", sort=False).first()
    loop = run_onsets["looping_onset"]
    nt = run_onsets["non_termination_onset"]
    loop_first = loop.notna() & (nt.isna() | (nt > loop))
    clean = loop.isna() & nt.isna()
    keep_runs = run_onsets.index[loop_first | clean]
    return df[df["run_id"].isin(set(keep_runs))].reset_index(drop=True)


def build_features(
    df: pd.DataFrame, max_budget: int
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["position_in_budget"] = (
        df["seg_end_tok"].astype(float) / float(max_budget)
    ).clip(0.0, 1.0)
    feature_cols = sorted(
        c for c in df.columns if c not in LEAK_OR_META
    )
    return df, feature_cols


def train_press_fold0(
    df_press: pd.DataFrame,
    press: str,
    max_budget: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    n_splits: int,
    n_boot: int,
) -> dict[str, object]:
    n_runs_total = df_press["run_id"].nunique()
    df_loop = filter_loop_only(df_press)
    n_runs_kept = df_loop["run_id"].nunique()
    if df_loop.empty or df_loop["y"].sum() == 0:
        logger.warning(
            "[{}] no loop-only data or no positives, skipping",
            press,
        )
        return {
            "press": press,
            "n_runs_total": int(n_runs_total),
            "n_runs_loop_only": int(n_runs_kept),
            "skipped": True,
        }

    df_x, feats = build_features(df_loop, max_budget)
    X = df_x[feats].to_numpy(dtype=np.float32)
    y = df_x["y"].to_numpy(dtype=int)
    groups = df_x["prompt_id"].to_numpy()
    pos_rate = y.mean()
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))
    tr, te = splits[0]
    pos_test = int(y[te].sum())
    if pos_test < 5:
        logger.warning(
            "[{}] fold 0 has only {} positives; AUROC unreliable",
            press,
            pos_test,
        )

    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=spw,
        tree_method="hist",
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=30,
    )
    clf.fit(
        X[tr], y[tr], eval_set=[(X[te], y[te])], verbose=False
    )
    score = clf.predict_proba(X[te])[:, 1]
    auroc = float(roc_auc_score(y[te], score))
    auprc = float(average_precision_score(y[te], score))
    bs_auroc = cluster_bootstrap_ci(
        y[te], score, groups[te], roc_auc_score, n_boot
    )
    bs_auprc = cluster_bootstrap_ci(
        y[te],
        score,
        groups[te],
        average_precision_score,
        n_boot,
    )
    logger.info(
        "[{}] n_runs_total={} loop_only={} pos_test={} "
        "AUROC={:.4f} (CI {:.4f}-{:.4f})  "
        "AUPRC={:.4f} (CI {:.4f}-{:.4f})  best_iter={}",
        press,
        n_runs_total,
        n_runs_kept,
        pos_test,
        auroc,
        bs_auroc[1],
        bs_auroc[2],
        auprc,
        bs_auprc[1],
        bs_auprc[2],
        int(clf.best_iteration),
    )
    return {
        "press": press,
        "n_runs_total": int(n_runs_total),
        "n_runs_loop_only": int(n_runs_kept),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "pos_train": int(y[tr].sum()),
        "pos_test": pos_test,
        "auroc": auroc,
        "auroc_ci_low": bs_auroc[1],
        "auroc_ci_high": bs_auroc[2],
        "auprc": auprc,
        "auprc_ci_low": bs_auprc[1],
        "auprc_ci_high": bs_auprc[2],
        "best_iteration": int(clf.best_iteration),
        "skipped": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase2_v2/segments_k16_ext.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/phase2_v2/per_press_loop_fold0.json"
        ),
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-budget", type=int, default=512)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=200)
    args = ap.parse_args()

    logger.info("Reading {}", args.input)
    df = pd.read_parquet(args.input)
    logger.info("rows={} cols={}", *df.shape)

    presses = sorted(df["press"].unique())
    logger.info("presses ({}): {}", len(presses), presses)

    per_press: list[dict[str, object]] = []
    for p in presses:
        df_p = df[df["press"] == p].reset_index(drop=True)
        res = train_press_fold0(
            df_p,
            p,
            args.max_budget,
            args.n_estimators,
            args.max_depth,
            args.learning_rate,
            args.n_splits,
            args.n_boot,
        )
        per_press.append(res)

    aurocs = [
        r["auroc"] for r in per_press if not r.get("skipped")
    ]
    auprcs = [
        r["auprc"] for r in per_press if not r.get("skipped")
    ]
    macro_auroc = float(np.mean(aurocs)) if aurocs else float("nan")
    macro_auprc = float(np.mean(auprcs)) if auprcs else float("nan")
    std_auroc = float(np.std(aurocs)) if aurocs else float("nan")

    out = {
        "fold": 0,
        "n_splits": args.n_splits,
        "max_budget": args.max_budget,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "presses": presses,
        "per_press": per_press,
        "macro_mean_auroc": macro_auroc,
        "macro_std_auroc": std_auroc,
        "macro_mean_auprc": macro_auprc,
        "bar_auroc": 0.96,
        "target_met": macro_auroc >= 0.96
        if not np.isnan(macro_auroc)
        else False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info(
        "MACRO mean AUROC={:.4f} (std={:.4f}) AUPRC={:.4f}  "
        "BAR={:.4f}  TARGET_MET={}",
        macro_auroc,
        std_auroc,
        macro_auprc,
        0.96,
        out["target_met"],
    )
    logger.info("Wrote {}", args.output)


if __name__ == "__main__":
    main()
