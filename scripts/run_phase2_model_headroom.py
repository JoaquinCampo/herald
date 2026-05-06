"""Phase 2b Task 3: XGBoost headroom over `lr_all_cheap`.

Trains XGBoost on the same dataset / feature set / splits / labels as
the Phase 2 baseline runner, and compares head-to-head against
`lr_all_cheap`. Compares only against `lr_all_cheap`, not entropy —
the Phase 2 result already established the lr-vs-entropy delta. The
question for this task is whether a non-linear model materially beats
the linear one.

Scope (per-spec)
----------------
- Default H=25, splits prompts + presses. Add ratios + tasks if and
  only if the smaller scope shows materially > +0.02 AUROC headroom.
- xgboost is already a project dep; lightgbm is not installed and the
  spec says do not add heavy deps without justification, so we skip
  it.

Calibration
-----------
For the held-out-prompts H=25 cell (the cheapest cell to retrain on),
we also dump per-fold reliability bins (10 equal-width probability
bins) for both `lr_all_cheap` and `xgb`. ECE = mean of |obs - pred|
weighted by bin support.

Outputs
-------
- results/phase2/model_headroom/model_headroom_results.parquet
- results/phase2/model_headroom/model_headroom_summary.json
- results/phase2/model_headroom/calibration_prompts_h25.parquet
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import xgboost as xgb
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    DEFAULT_QUANTILE,
    binarize_with_threshold,
    bootstrap_auroc_ci,
    compute_train_quantile_threshold,
    iter_splits,
)


def _clean_X(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.size < 2 or len(set(y.tolist())) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _safe_auprc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.size < 2 or len(set(y.tolist())) < 2:
        return None
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return None


def _ece_and_bins(
    y: np.ndarray, s: np.ndarray, n_bins: int = 10
) -> tuple[float, list[dict[str, Any]]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    n = y.size
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (s >= lo) & (s <= hi)
        else:
            mask = (s >= lo) & (s < hi)
        if not mask.any():
            rows.append(
                {
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "n": 0,
                    "mean_pred": None,
                    "frac_pos": None,
                }
            )
            continue
        bin_n = int(mask.sum())
        mean_pred = float(s[mask].mean())
        frac_pos = float(y[mask].mean())
        rows.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n": bin_n,
                "mean_pred": mean_pred,
                "frac_pos": frac_pos,
            }
        )
        ece += (bin_n / max(n, 1)) * abs(mean_pred - frac_pos)
    return float(ece), rows


def _fit_lr(
    train_X: np.ndarray, train_y: np.ndarray, test_X: np.ndarray
) -> np.ndarray | None:
    if train_X.shape[0] < 4 or len(set(train_y.tolist())) < 2:
        return None
    train_X = _clean_X(train_X)
    test_X = _clean_X(test_X)
    sc = StandardScaler()
    Xs = sc.fit_transform(train_X)
    Xt = sc.transform(test_X)
    clf = LogisticRegression(
        max_iter=200, class_weight="balanced", solver="lbfgs"
    )
    clf.fit(Xs, train_y)
    return np.asarray(clf.predict_proba(Xt)[:, 1], dtype=float)


def _fit_xgb(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    seed: int,
    n_estimators: int,
) -> np.ndarray | None:
    if train_X.shape[0] < 4 or len(set(train_y.tolist())) < 2:
        return None
    train_X = _clean_X(train_X)
    test_X = _clean_X(test_X)
    n_pos = int((train_y == 1).sum())
    n_neg = int((train_y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    spw = max(n_neg / max(n_pos, 1), 1.0)
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=spw,
        random_state=seed,
        verbosity=0,
    )
    clf.fit(train_X, train_y)
    return np.asarray(clf.predict_proba(test_X)[:, 1], dtype=float)


def _evaluate_one_fold(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    label_col: str,
    feature_cols: list[str],
    threshold_q: float,
    drop_press_feature: bool,
    seed: int,
    n_estimators: int,
    n_boot_ci: int,
) -> dict[str, Any] | None:
    train_y_cont = train_df[label_col]
    test_y_cont = test_df[label_col]
    thr = compute_train_quantile_threshold(train_y_cont, q=threshold_q)
    train_y_bin = binarize_with_threshold(train_y_cont, thr)
    test_y_bin = binarize_with_threshold(test_y_cont, thr)
    test_mask = ~test_y_bin.is_null()
    keep_test_y = test_y_bin.filter(test_mask).cast(pl.Int64).to_numpy()
    keep_test_df = test_df.filter(test_mask)
    train_mask = ~train_y_bin.is_null()
    keep_train_df = train_df.filter(train_mask)
    keep_train_y = train_y_bin.filter(train_mask).cast(pl.Int64).to_numpy()
    if keep_test_y.size < 2 or len(set(keep_test_y.tolist())) < 2:
        return None
    feats = [f for f in feature_cols if f in train_df.columns]
    if drop_press_feature:
        # Same convention as predictor_baselines: drop ratio feature
        # only when ratio is the split itself (handled at caller).
        pass
    train_X = (
        keep_train_df.select(feats).fill_null(0.0).to_numpy().astype(float)
    )
    test_X = (
        keep_test_df.select(feats).fill_null(0.0).to_numpy().astype(float)
    )

    s_lr = _fit_lr(train_X, keep_train_y, test_X)
    s_xgb = _fit_xgb(
        train_X, keep_train_y, test_X, seed=seed, n_estimators=n_estimators
    )
    out: dict[str, Any] = {
        "n_train": int(keep_train_df.height),
        "n_test": int(keep_test_df.height),
        "threshold": thr,
        "train_pos_rate": float((keep_train_y == 1).mean()),
        "test_pos_rate": float((keep_test_y == 1).mean()),
        "groups": keep_test_df["run_id"].to_numpy(),
        "y_true": keep_test_y,
    }
    if s_lr is not None:
        out["score_lr_all_cheap"] = s_lr
        out["auroc_lr_all_cheap"] = _safe_auroc(keep_test_y, s_lr)
        out["auprc_lr_all_cheap"] = _safe_auprc(keep_test_y, s_lr)
        if n_boot_ci > 0:
            lo, hi = bootstrap_auroc_ci(
                list(keep_test_y),
                list(s_lr),
                n_boot=n_boot_ci,
                seed=seed,
            )
            out["auroc_ci_lr_lo"] = lo
            out["auroc_ci_lr_hi"] = hi
    if s_xgb is not None:
        out["score_xgb"] = s_xgb
        out["auroc_xgb"] = _safe_auroc(keep_test_y, s_xgb)
        out["auprc_xgb"] = _safe_auprc(keep_test_y, s_xgb)
        if n_boot_ci > 0:
            lo, hi = bootstrap_auroc_ci(
                list(keep_test_y),
                list(s_xgb),
                n_boot=n_boot_ci,
                seed=seed,
            )
            out["auroc_ci_xgb_lo"] = lo
            out["auroc_ci_xgb_hi"] = hi
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/model_headroom"),
    )
    ap.add_argument("--horizons", type=int, nargs="+", default=[25])
    ap.add_argument("--label-base", type=str, default="future_sum_js")
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["prompts", "presses"],
    )
    ap.add_argument("--n-prompt-folds", type=int, default=5)
    ap.add_argument("--threshold-q", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--max-train-rows", type=int, default=200_000)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--n-boot-ci", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.dataset)
    df = pl.read_parquet(args.dataset)
    logger.info("dataset rows={} cols={}", df.height, len(df.columns))
    if args.max_train_rows > 0 and df.height > args.max_train_rows:
        df = df.sample(n=args.max_train_rows, seed=args.seed, shuffle=True)
        logger.info("subsampled to {} rows", df.height)

    feature_cols = list(CHEAP_ALL_FEATURES)
    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    summary_cells: dict[str, Any] = {}

    t0 = time.time()
    for h in args.horizons:
        label = f"{args.label_base}_{h}"
        if label not in df.columns:
            logger.warning("missing label {}; skipping", label)
            continue
        for kind in args.splits:
            drop_press = kind == "presses"
            fold_aurocs_lr: list[float] = []
            fold_aurocs_xgb: list[float] = []
            fold_auprcs_lr: list[float] = []
            fold_auprcs_xgb: list[float] = []
            fold_eces_lr: list[float] = []
            fold_eces_xgb: list[float] = []
            for tr, te, fid in iter_splits(
                df,
                kind=kind,
                n_splits=args.n_prompt_folds,
                seed=args.seed,
            ):
                logger.info("fitting H={} kind={} fold={}", h, kind, fid)
                fold_feats = [c for c in feature_cols if c in df.columns]
                if drop_press:
                    # No 'press' string feature here, but match convention.
                    pass
                res = _evaluate_one_fold(
                    df[tr],
                    df[te],
                    label_col=label,
                    feature_cols=fold_feats,
                    threshold_q=args.threshold_q,
                    drop_press_feature=drop_press,
                    seed=args.seed,
                    n_estimators=args.n_estimators,
                    n_boot_ci=args.n_boot_ci,
                )
                if res is None:
                    continue
                row: dict[str, Any] = {
                    "split_kind": kind,
                    "fold_id": fid,
                    "horizon": h,
                    "label": label,
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "threshold": res["threshold"],
                    "test_pos_rate": res["test_pos_rate"],
                    "auroc_lr_all_cheap": res.get("auroc_lr_all_cheap"),
                    "auprc_lr_all_cheap": res.get("auprc_lr_all_cheap"),
                    "auroc_xgb": res.get("auroc_xgb"),
                    "auprc_xgb": res.get("auprc_xgb"),
                    "delta_xgb_minus_lr": (
                        None
                        if res.get("auroc_xgb") is None
                        or res.get("auroc_lr_all_cheap") is None
                        else round(
                            res["auroc_xgb"] - res["auroc_lr_all_cheap"],
                            4,
                        )
                    ),
                    "auroc_ci_lr_lo": res.get("auroc_ci_lr_lo"),
                    "auroc_ci_lr_hi": res.get("auroc_ci_lr_hi"),
                    "auroc_ci_xgb_lo": res.get("auroc_ci_xgb_lo"),
                    "auroc_ci_xgb_hi": res.get("auroc_ci_xgb_hi"),
                }
                rows.append(row)
                if (
                    res.get("auroc_lr_all_cheap") is not None
                    and res.get("auroc_xgb") is not None
                ):
                    fold_aurocs_lr.append(res["auroc_lr_all_cheap"])
                    fold_aurocs_xgb.append(res["auroc_xgb"])
                    fold_auprcs_lr.append(res["auprc_lr_all_cheap"])
                    fold_auprcs_xgb.append(res["auprc_xgb"])

                # Calibration on prompts H=25 only.
                if (
                    kind == "prompts"
                    and h == 25
                    and "score_lr_all_cheap" in res
                    and "score_xgb" in res
                ):
                    ece_lr, bins_lr = _ece_and_bins(
                        res["y_true"], res["score_lr_all_cheap"]
                    )
                    ece_xgb, bins_xgb = _ece_and_bins(
                        res["y_true"], res["score_xgb"]
                    )
                    fold_eces_lr.append(ece_lr)
                    fold_eces_xgb.append(ece_xgb)
                    for b in bins_lr:
                        calibration_rows.append(
                            {
                                "model": "lr_all_cheap",
                                "fold_id": fid,
                                "horizon": h,
                                **b,
                            }
                        )
                    for b in bins_xgb:
                        calibration_rows.append(
                            {
                                "model": "xgb",
                                "fold_id": fid,
                                "horizon": h,
                                **b,
                            }
                        )
            if fold_aurocs_xgb:
                summary_cells[f"{kind}::H={h}"] = {
                    "split_kind": kind,
                    "horizon": h,
                    "n_folds": len(fold_aurocs_xgb),
                    "auroc_lr_mean": round(float(np.mean(fold_aurocs_lr)), 4),
                    "auroc_xgb_mean": round(
                        float(np.mean(fold_aurocs_xgb)), 4
                    ),
                    "auprc_lr_mean": round(float(np.mean(fold_auprcs_lr)), 4),
                    "auprc_xgb_mean": round(
                        float(np.mean(fold_auprcs_xgb)), 4
                    ),
                    "auroc_delta_mean": round(
                        float(
                            np.mean(
                                [
                                    a - b
                                    for a, b in zip(
                                        fold_aurocs_xgb,
                                        fold_aurocs_lr,
                                        strict=True,
                                    )
                                ]
                            )
                        ),
                        4,
                    ),
                    "ece_lr_mean": (
                        round(float(np.mean(fold_eces_lr)), 4)
                        if fold_eces_lr
                        else None
                    ),
                    "ece_xgb_mean": (
                        round(float(np.mean(fold_eces_xgb)), 4)
                        if fold_eces_xgb
                        else None
                    ),
                }

    pl.DataFrame(rows).write_parquet(
        args.output_dir / "model_headroom_results.parquet"
    )
    pl.DataFrame(calibration_rows).write_parquet(
        args.output_dir / "calibration_prompts_h25.parquet"
    )
    summary_obj = {
        "label_base": args.label_base,
        "horizons": args.horizons,
        "splits": args.splits,
        "n_prompt_folds": args.n_prompt_folds,
        "max_train_rows": args.max_train_rows,
        "n_estimators": args.n_estimators,
        "n_boot_ci": args.n_boot_ci,
        "wall_seconds": round(time.time() - t0, 2),
        "cells": summary_cells,
    }
    (args.output_dir / "model_headroom_summary.json").write_text(
        json.dumps(summary_obj, indent=2, default=str)
    )

    print()
    print("=== XGBoost vs lr_all_cheap (mean across folds) ===")
    for k, v in sorted(summary_cells.items()):
        print(
            f"  {k:18s}  lr={v['auroc_lr_mean']}  xgb={v['auroc_xgb_mean']}  "
            f"Δ={v['auroc_delta_mean']:+.4f}  "
            f"ECE_lr={v['ece_lr_mean']}  ECE_xgb={v['ece_xgb_mean']}"
        )


if __name__ == "__main__":
    main()
