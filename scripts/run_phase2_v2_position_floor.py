"""Position-only AUROC floor for the per-segment K=16 onset task.

For every compressed run, builds (segment_idx, segment_end_tok) decision
points and labels y=1 iff a catastrophe onset (looping or non_termination)
falls within the next K tokens. Trains a logistic regression on a
deliberately weak feature set:

  - segment_idx (token position proxy)
  - relative_progress = segment_end_tok / max_tokens
  - press_id (one-hot)
  - compression_ratio

This is the floor: any predictor that uses real per-token signals must
beat this by a meaningful margin. If the floor is already >= 0.95, the
task is trivially easy and the goal threshold is uninteresting.

Held-out by prompt_id (5-fold GroupKFold).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


def build_segments(
    runs: pd.DataFrame, onsets: pd.DataFrame, k: int
) -> pd.DataFrame:
    """Per-segment table with binary onset-in-next-K label.

    Stops emitting segments for a run after the first y=1 segment, OR
    after the last segment that fits inside `num_tokens_generated`.
    """
    onsets_min = onsets[
        ["run_id", "looping_onset", "non_termination_onset"]
    ]
    df = runs.merge(onsets_min, on="run_id", how="inner")

    rows: list[dict[str, object]] = []
    for r in df.itertuples(index=False):
        n_tok = int(r.num_tokens_generated)
        if n_tok < 1:
            continue
        loop = (
            int(r.looping_onset)
            if pd.notna(r.looping_onset)
            else None
        )
        nt = (
            int(r.non_termination_onset)
            if pd.notna(r.non_termination_onset)
            else None
        )
        first_onset = None
        if loop is not None and nt is not None:
            first_onset = min(loop, nt)
        elif loop is not None:
            first_onset = loop
        elif nt is not None:
            first_onset = nt

        max_seg_idx = (n_tok - 1) // k
        for seg_idx in range(max_seg_idx + 1):
            seg_end_tok = (seg_idx + 1) * k - 1
            if first_onset is not None and seg_end_tok >= first_onset:
                break
            if first_onset is None:
                y = 0
            else:
                lo, hi = seg_end_tok + 1, seg_end_tok + k
                y = int(lo <= first_onset <= hi)
            rows.append(
                {
                    "run_id": r.run_id,
                    "prompt_id": r.prompt_id,
                    "press": r.press,
                    "compression_ratio": float(r.compression_ratio),
                    "seg_idx": int(seg_idx),
                    "seg_end_tok": int(seg_end_tok),
                    "num_tokens_generated": int(n_tok),
                    "y": int(y),
                }
            )
            if y == 1:
                break
    out = pd.DataFrame(rows)
    out["relative_progress"] = (
        out["seg_end_tok"].astype(float)
        / out["num_tokens_generated"].astype(float)
    ).clip(0.0, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        type=Path,
        default=Path("results/phase1/final/runs.parquet"),
    )
    ap.add_argument(
        "--onsets",
        type=Path,
        default=Path("results/phase2_v2/onsets.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2_v2/position_floor"),
    )
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-splits", type=int, default=5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading {}", args.runs)
    runs = pd.read_parquet(
        args.runs,
        columns=[
            "run_id",
            "prompt_id",
            "press",
            "compression_ratio",
            "num_tokens_generated",
        ],
    )
    runs = runs[runs["press"] != "none"].reset_index(drop=True)

    logger.info("Reading {}", args.onsets)
    onsets = pd.read_parquet(args.onsets)[
        ["run_id", "looping_onset", "non_termination_onset"]
    ]

    logger.info("Building segments K={}", args.k)
    segs = build_segments(runs, onsets, args.k)
    logger.info(
        "Segments: {}, positives: {} ({:.3%})",
        len(segs),
        int(segs["y"].sum()),
        float(segs["y"].mean()),
    )

    presses = sorted(segs["press"].unique())
    for p in presses:
        segs[f"press_{p}"] = (segs["press"] == p).astype(float)

    feat_cols = (
        ["seg_idx", "relative_progress", "compression_ratio"]
        + [f"press_{p}" for p in presses]
    )

    X = segs[feat_cols].to_numpy(dtype=float)
    y = segs["y"].to_numpy(dtype=int)
    groups = segs["prompt_id"].to_numpy()

    gkf = GroupKFold(n_splits=args.n_splits)
    fold_auroc: list[float] = []
    fold_auprc: list[float] = []
    fold_pos: list[float] = []
    for fold_i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        if y[te].sum() == 0 or y[te].sum() == len(te):
            logger.warning("Fold {} degenerate, skipping", fold_i)
            continue
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        score = clf.predict_proba(X[te])[:, 1]
        auroc = roc_auc_score(y[te], score)
        auprc = average_precision_score(y[te], score)
        pos_rate = float(y[te].mean())
        logger.info(
            "fold {}: AUROC={:.4f} AUPRC={:.4f} pos_rate={:.4%}",
            fold_i,
            auroc,
            auprc,
            pos_rate,
        )
        fold_auroc.append(auroc)
        fold_auprc.append(auprc)
        fold_pos.append(pos_rate)

    summary = {
        "k": args.k,
        "n_segments": int(len(segs)),
        "n_positives": int(segs["y"].sum()),
        "pos_rate_overall": float(segs["y"].mean()),
        "feature_cols": feat_cols,
        "fold_auroc": fold_auroc,
        "fold_auprc": fold_auprc,
        "fold_pos_rate": fold_pos,
        "auroc_mean": float(np.mean(fold_auroc)) if fold_auroc else None,
        "auroc_std": float(np.std(fold_auroc)) if fold_auroc else None,
        "auprc_mean": float(np.mean(fold_auprc)) if fold_auprc else None,
        "auprc_std": float(np.std(fold_auprc)) if fold_auprc else None,
    }
    out_path = args.output_dir / "position_floor_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary -> {}", out_path)


if __name__ == "__main__":
    main()
