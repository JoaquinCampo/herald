"""Isotonic-calibrate the per-segment HERALD predictor.

Loads the cross-fold scores produced by `train_phase2_v2_xgb*.py`,
fits a per-fold isotonic mapping on a held-out portion of the
training side, and reports calibration metrics on the test fold.

Protocol per fold f:
  - "in-fold" = scores parquet for this fold (these are the test set
    of fold f, i.e. the held-out groups).
  - "calibration set" = pool of test-side scores from all OTHER folds
    (those segments belong to disjoint prompt_ids by GroupKFold).
  - Fit IsotonicRegression on calibration set -> apply to fold f's
    raw scores.

Outputs:
  {output-dir}/calibrated_fold{i}.parquet  (run_id, seg_idx, y, score,
                                             score_calibrated)
  {output-dir}/calibration_summary.json
  {output-dir}/isotonic_fold{i}.json (knots: x, y arrays)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def reliability_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, list[float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(
        np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1
    )
    bin_acc: list[float] = []
    bin_conf: list[float] = []
    bin_count: list[int] = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            bin_acc.append(float("nan"))
            bin_conf.append(float("nan"))
            bin_count.append(0)
            continue
        bin_acc.append(float(y_true[mask].mean()))
        bin_conf.append(float(y_prob[mask].mean()))
        bin_count.append(int(mask.sum()))
    return {
        "bin_centers": list((edges[:-1] + edges[1:]) / 2.0),
        "bin_accuracy": bin_acc,
        "bin_confidence": bin_conf,
        "bin_count": bin_count,
    }


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(
        np.digitize(y_prob, edges, right=True) - 1, 0, n_bins - 1
    )
    n = len(y_true)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += (
            mask.sum() / n
        ) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(total)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("results/phase2_v2/xgb_ext"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to scores-dir/calibrated.",
    )
    args = ap.parse_args()

    out = args.output_dir or (args.scores_dir / "calibrated")
    out.mkdir(parents=True, exist_ok=True)

    fold_files = sorted(args.scores_dir.glob("scores_fold*.parquet"))
    folds = []
    for fp in fold_files:
        i = int(fp.stem.replace("scores_fold", ""))
        df = pd.read_parquet(fp)
        df["fold"] = i
        folds.append(df)
    pooled = pd.concat(folds, ignore_index=True)
    logger.info(
        "Pooled scores: {} rows from {} folds",
        len(pooled),
        len(folds),
    )

    fold_results: list[dict[str, object]] = []
    for fold_i, fold_df in enumerate(folds):
        cal_df = pooled[pooled["fold"] != fold_i]
        iso = IsotonicRegression(
            out_of_bounds="clip", y_min=0.0, y_max=1.0
        )
        iso.fit(cal_df["score"].values, cal_df["y"].values)
        cal_scores = iso.transform(fold_df["score"].values)

        raw_brier = brier_score_loss(
            fold_df["y"].values, fold_df["score"].values
        )
        cal_brier = brier_score_loss(fold_df["y"].values, cal_scores)
        raw_ece = expected_calibration_error(
            fold_df["y"].values, fold_df["score"].values
        )
        cal_ece = expected_calibration_error(
            fold_df["y"].values, cal_scores
        )
        # Isotonic preserves AUROC; spot check.
        raw_auroc = roc_auc_score(
            fold_df["y"].values, fold_df["score"].values
        )
        cal_auroc = roc_auc_score(fold_df["y"].values, cal_scores)
        raw_auprc = average_precision_score(
            fold_df["y"].values, fold_df["score"].values
        )
        cal_auprc = average_precision_score(
            fold_df["y"].values, cal_scores
        )

        out_df = fold_df.copy()
        out_df["score_calibrated"] = cal_scores
        out_df.to_parquet(
            out / f"calibrated_fold{fold_i}.parquet", index=False
        )

        # Persist isotonic mapping for the streaming wrapper
        knot_x = iso.X_thresholds_.astype(float).tolist()
        knot_y = iso.y_thresholds_.astype(float).tolist()
        (out / f"isotonic_fold{fold_i}.json").write_text(
            json.dumps({"x": knot_x, "y": knot_y})
        )

        rel = reliability_bins(
            fold_df["y"].values, cal_scores, n_bins=10
        )
        rec = {
            "fold": fold_i,
            "n_test": int(len(fold_df)),
            "raw_brier": float(raw_brier),
            "cal_brier": float(cal_brier),
            "raw_ece": float(raw_ece),
            "cal_ece": float(cal_ece),
            "raw_auroc": float(raw_auroc),
            "cal_auroc": float(cal_auroc),
            "raw_auprc": float(raw_auprc),
            "cal_auprc": float(cal_auprc),
            "reliability": rel,
        }
        fold_results.append(rec)
        logger.info(
            "fold {}: brier {:.5f}->{:.5f}  ECE {:.4f}->{:.4f}  "
            "AUROC {:.4f}->{:.4f}  AUPRC {:.4f}->{:.4f}",
            fold_i,
            raw_brier,
            cal_brier,
            raw_ece,
            cal_ece,
            raw_auroc,
            cal_auroc,
            raw_auprc,
            cal_auprc,
        )

    summary = {
        "n_folds": len(fold_results),
        "fold_results": fold_results,
        "mean_raw_brier": float(
            np.mean([f["raw_brier"] for f in fold_results])
        ),
        "mean_cal_brier": float(
            np.mean([f["cal_brier"] for f in fold_results])
        ),
        "mean_raw_ece": float(
            np.mean([f["raw_ece"] for f in fold_results])
        ),
        "mean_cal_ece": float(
            np.mean([f["cal_ece"] for f in fold_results])
        ),
        "mean_raw_auroc": float(
            np.mean([f["raw_auroc"] for f in fold_results])
        ),
        "mean_cal_auroc": float(
            np.mean([f["cal_auroc"] for f in fold_results])
        ),
    }
    (out / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    logger.info(
        "MEAN: raw Brier={:.5f} cal Brier={:.5f}  "
        "raw ECE={:.4f} cal ECE={:.4f}  "
        "raw AUROC={:.4f} cal AUROC={:.4f}",
        summary["mean_raw_brier"],
        summary["mean_cal_brier"],
        summary["mean_raw_ece"],
        summary["mean_cal_ece"],
        summary["mean_raw_auroc"],
        summary["mean_cal_auroc"],
    )


if __name__ == "__main__":
    main()
