"""Stratify the per-segment predictor evaluation by failure mode.

Reads the fold scores from `train_phase2_v2_xgb.py` and decomposes
AUROC / AUPRC for two label sub-tasks:

  - looping-only: positives = segments where looping_onset is the
    next-K event; restrict negatives to runs without onset OR with
    only looping-onset.
  - nt-only: positives = segments where non_termination_onset is the
    next-K event; same negative restriction logic.

This shows whether the overall metric is driven by the easy
non-termination mode (predictable from position-in-budget) or the
genuinely difficult looping mode.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=Path("results/phase2_v2/xgb_ext"),
    )
    ap.add_argument(
        "--segments",
        type=Path,
        default=Path("results/phase2_v2/segments_k16_ext.parquet"),
    )
    ap.add_argument("--k", type=int, default=16)
    args = ap.parse_args()

    seg = pd.read_parquet(
        args.segments,
        columns=[
            "run_id",
            "seg_idx",
            "seg_end_tok",
            "looping_onset",
            "non_termination_onset",
        ],
    )

    fold_files = sorted(args.scores_dir.glob("scores_fold*.parquet"))
    fold_results: list[dict[str, object]] = []
    for fp in fold_files:
        fold_i = int(fp.stem.replace("scores_fold", ""))
        scores = pd.read_parquet(fp)
        merged = scores.merge(
            seg, on=["run_id", "seg_idx"], how="left"
        ).reset_index(drop=True)

        # Compute which onset (if any) the positive segment captures.
        merged["next_k_lo"] = merged["seg_end_tok"] + 1
        merged["next_k_hi"] = merged["seg_end_tok"] + args.k

        loop_in_window = (
            merged["looping_onset"].notna()
            & (merged["looping_onset"] >= merged["next_k_lo"])
            & (merged["looping_onset"] <= merged["next_k_hi"])
        )
        nt_in_window = (
            merged["non_termination_onset"].notna()
            & (merged["non_termination_onset"] >= merged["next_k_lo"])
            & (
                merged["non_termination_onset"] <= merged["next_k_hi"]
            )
        )

        # Looping-only: y=1 iff looping is the in-window event AND no
        # nt is in-window (otherwise positive captures nt). Restrict
        # negatives to runs whose RUN-LEVEL onset is not non_termination
        # only (we want the negative set to mirror the positive set).
        loop_runs = (
            merged.groupby("run_id")["looping_onset"]
            .first()
            .notna()
        )
        no_onset_runs = (
            merged.groupby("run_id")["looping_onset"]
            .first()
            .isna()
            & merged.groupby("run_id")["non_termination_onset"]
            .first()
            .isna()
        )
        eligible_runs_loop = loop_runs[
            loop_runs | no_onset_runs
        ].index.tolist()
        loop_subset = merged[
            merged["run_id"].isin(set(eligible_runs_loop))
            | merged["run_id"].isin(
                set(no_onset_runs[no_onset_runs].index)
            )
        ].copy()
        loop_subset["y_loop"] = (
            loop_in_window & ~nt_in_window
        ).astype(int)
        if (
            loop_subset["y_loop"].sum() > 0
            and (loop_subset["y_loop"] == 0).sum() > 0
        ):
            loop_auroc = roc_auc_score(
                loop_subset["y_loop"], loop_subset["score"]
            )
            loop_auprc = average_precision_score(
                loop_subset["y_loop"], loop_subset["score"]
            )
        else:
            loop_auroc = float("nan")
            loop_auprc = float("nan")

        # NT-only: positive = nt onset in window (regardless of loop),
        # negatives = nt-mode runs OR no-onset runs, segments before
        # the positive.
        nt_runs = (
            merged.groupby("run_id")["non_termination_onset"]
            .first()
            .notna()
        )
        nt_eligible = (
            nt_runs[nt_runs].index.tolist()
            + no_onset_runs[no_onset_runs].index.tolist()
        )
        nt_subset = merged[
            merged["run_id"].isin(set(nt_eligible))
        ].copy()
        nt_subset["y_nt"] = nt_in_window.loc[nt_subset.index].astype(int)
        if (
            nt_subset["y_nt"].sum() > 0
            and (nt_subset["y_nt"] == 0).sum() > 0
        ):
            nt_auroc = roc_auc_score(nt_subset["y_nt"], nt_subset["score"])
            nt_auprc = average_precision_score(
                nt_subset["y_nt"], nt_subset["score"]
            )
        else:
            nt_auroc = float("nan")
            nt_auprc = float("nan")

        all_auroc = roc_auc_score(merged["y"], merged["score"])
        all_auprc = average_precision_score(merged["y"], merged["score"])

        logger.info(
            "fold {}: ALL AUROC={:.4f} AUPRC={:.4f} | "
            "LOOP AUROC={:.4f} AUPRC={:.4f} (n_pos={}) | "
            "NT AUROC={:.4f} AUPRC={:.4f} (n_pos={})",
            fold_i,
            all_auroc,
            all_auprc,
            loop_auroc,
            loop_auprc,
            int(loop_subset["y_loop"].sum()),
            nt_auroc,
            nt_auprc,
            int(nt_subset["y_nt"].sum()),
        )
        fold_results.append(
            {
                "fold": fold_i,
                "all_auroc": float(all_auroc),
                "all_auprc": float(all_auprc),
                "loop_auroc": float(loop_auroc),
                "loop_auprc": float(loop_auprc),
                "loop_n_pos": int(loop_subset["y_loop"].sum()),
                "nt_auroc": float(nt_auroc),
                "nt_auprc": float(nt_auprc),
                "nt_n_pos": int(nt_subset["y_nt"].sum()),
            }
        )

    summary = {
        "fold_results": fold_results,
        "all_auroc_mean": float(
            np.mean([f["all_auroc"] for f in fold_results])
        ),
        "loop_auroc_mean": float(
            np.nanmean([f["loop_auroc"] for f in fold_results])
        ),
        "nt_auroc_mean": float(
            np.nanmean([f["nt_auroc"] for f in fold_results])
        ),
        "loop_auprc_mean": float(
            np.nanmean([f["loop_auprc"] for f in fold_results])
        ),
        "nt_auprc_mean": float(
            np.nanmean([f["nt_auprc"] for f in fold_results])
        ),
    }
    out_path = args.scores_dir / "by_mode_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary -> {}", out_path)


if __name__ == "__main__":
    main()
