"""Ablation: drop `position_in_budget` from the K=16 onset predictor.

Trains one fold (default fold 0) of the per-segment XGBoost twice
(with and without position_in_budget) on:
  - the UNIFIED dataset (segments_k16_ext.parquet, all runs)
  - the LOOP-ONLY subset (loop-first or clean runs)

Reports AUROC/AUPRC delta. The unified delta tells us how much of the
unified score is budget-cap arithmetic (NT label leak). The loop delta
should be near zero if the loop signal is genuine.

Outputs JSON to {output-path}.

Usage:
  uv run python scripts/ablate_position_in_budget.py \\
    --input results/phase2_v2/segments_k16_ext.parquet \\
    --fold 0 --output results/phase2_v2/ablation_position.json
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
    return df, feature_cols


def train_one_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    fold_i: int,
    n_splits: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
) -> dict[str, float]:
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=int)
    groups = df["prompt_id"].to_numpy()
    pos_rate = y.mean()
    spw = (1.0 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(X, y, groups))
    tr, te = splits[fold_i]

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
    return {
        "auroc": float(roc_auc_score(y[te], score)),
        "auprc": float(average_precision_score(y[te], score)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "pos_test": int(y[te].sum()),
        "best_iter": int(clf.best_iteration),
    }


def run_variant(
    df: pd.DataFrame,
    name: str,
    fold_i: int,
    n_splits: int,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    max_budget: int,
) -> dict[str, dict]:
    df_x, feats = build_features(df, max_budget)
    feats_no_pos = [f for f in feats if f != "position_in_budget"]
    logger.info(
        "[{}] fold={} n_features full={} no_pos={} n_rows={}",
        name,
        fold_i,
        len(feats),
        len(feats_no_pos),
        len(df_x),
    )
    full = train_one_fold(
        df_x,
        feats,
        fold_i,
        n_splits,
        n_estimators,
        max_depth,
        learning_rate,
    )
    logger.info(
        "[{}] FULL  AUROC={:.4f} AUPRC={:.4f}",
        name,
        full["auroc"],
        full["auprc"],
    )
    no_pos = train_one_fold(
        df_x,
        feats_no_pos,
        fold_i,
        n_splits,
        n_estimators,
        max_depth,
        learning_rate,
    )
    logger.info(
        "[{}] NOPOS AUROC={:.4f} AUPRC={:.4f}  "
        "delta_AUROC={:+.4f}  delta_AUPRC={:+.4f}",
        name,
        no_pos["auroc"],
        no_pos["auprc"],
        no_pos["auroc"] - full["auroc"],
        no_pos["auprc"] - full["auprc"],
    )
    return {
        "full": full,
        "no_position_in_budget": no_pos,
        "delta_auroc": no_pos["auroc"] - full["auroc"],
        "delta_auprc": no_pos["auprc"] - full["auprc"],
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
        default=Path("results/phase2_v2/ablation_position.json"),
    )
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--max-budget", type=int, default=512)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    args = ap.parse_args()

    logger.info("Reading {}", args.input)
    df = pd.read_parquet(args.input)
    logger.info("rows={} cols={}", *df.shape)

    unified_res = run_variant(
        df,
        "unified",
        args.fold,
        args.n_splits,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
        args.max_budget,
    )

    df_loop = filter_loop_only(df)
    logger.info("loop-only subset rows={}", len(df_loop))
    loop_res = run_variant(
        df_loop,
        "loop_only",
        args.fold,
        args.n_splits,
        args.n_estimators,
        args.max_depth,
        args.learning_rate,
        args.max_budget,
    )

    out = {
        "fold": args.fold,
        "n_splits": args.n_splits,
        "max_budget": args.max_budget,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "unified": unified_res,
        "loop_only": loop_res,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    logger.info("Wrote {}", args.output)


if __name__ == "__main__":
    main()
