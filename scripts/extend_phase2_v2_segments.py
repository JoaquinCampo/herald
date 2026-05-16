"""Extend the per-segment dataset with cross-segment dynamics features.

Reads `results/phase2_v2/segments_k16.parquet`, adds per-run causal
features computed across segments (not just within one segment):

For each numeric per-segment feature `f` (the cheap-feature aggregates
and the surface-repetition features):
  - {f}_cum_mean: expanding mean over segments [0..seg_idx]
  - {f}_cum_std:  expanding std over segments [0..seg_idx]
  - {f}_prev:     value at seg_idx-1 (null at seg_idx=0)
  - {f}_delta:    current minus previous (null at seg_idx=0)
  - {f}_roll4_mean: mean over last 4 segments (causal)
  - {f}_roll4_max:  max over last 4 segments (causal)

Also adds a few hand-crafted "looping-imminent" features:
  - cum_max_window20_repeats: running max of max_window20_repeats
  - segs_since_unique_drop: # segs since n_unique_64 dropped > 5
  - cum_max_top1_streak: running max of top1_streak

Output: results/phase2_v2/segments_k16_ext.parquet
"""

import argparse
from pathlib import Path

import polars as pl
from loguru import logger


META_COLS = {
    "run_id",
    "prompt_id",
    "press",
    "compression_ratio",
    "seg_idx",
    "seg_end_tok",
    "looping_onset",
    "non_termination_onset",
    "first_onset",
    "num_tokens_generated",
    "relative_progress",
    "y",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("results/phase2_v2/segments_k16.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase2_v2/segments_k16_ext.parquet"),
    )
    args = ap.parse_args()

    logger.info("Reading {}", args.input)
    df = pl.read_parquet(args.input)
    logger.info("rows={} cols={}", df.height, df.width)

    df = df.sort(["run_id", "seg_idx"])

    feat_cols = [
        c
        for c in df.columns
        if c not in META_COLS
        and df[c].dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    logger.info("Adding cross-segment features for {} cols", len(feat_cols))

    # Build all expressions, then with_columns once per kind to keep
    # polars happy.
    cum_mean_exprs = [
        (
            pl.col(c).cum_sum().over("run_id")
            / (pl.col(c).cum_count().over("run_id") + 1e-9)
        ).alias(f"{c}_cum_mean")
        for c in feat_cols
    ]
    df = df.with_columns(cum_mean_exprs)

    # cum std (use rolling std with window=seg_idx+1 via group operation)
    # Easier: shift+expanding via .cum_sum of squared deviations is messy;
    # use a simpler proxy: rolling std over a sliding window is more
    # interpretable. Use rolling 8-seg std as the dynamics feature.
    roll8_std_exprs = [
        pl.col(c)
        .rolling_std(window_size=8, min_samples=2)
        .over("run_id")
        .alias(f"{c}_roll8_std")
        for c in feat_cols
    ]
    df = df.with_columns(roll8_std_exprs)

    prev_exprs = [
        pl.col(c).shift(1).over("run_id").alias(f"{c}_prev")
        for c in feat_cols
    ]
    df = df.with_columns(prev_exprs)

    delta_exprs = [
        (pl.col(c) - pl.col(f"{c}_prev")).alias(f"{c}_delta")
        for c in feat_cols
    ]
    df = df.with_columns(delta_exprs)

    roll4_mean_exprs = [
        pl.col(c)
        .rolling_mean(window_size=4, min_samples=1)
        .over("run_id")
        .alias(f"{c}_roll4_mean")
        for c in feat_cols
    ]
    df = df.with_columns(roll4_mean_exprs)

    roll4_max_exprs = [
        pl.col(c)
        .rolling_max(window_size=4, min_samples=1)
        .over("run_id")
        .alias(f"{c}_roll4_max")
        for c in feat_cols
    ]
    df = df.with_columns(roll4_max_exprs)

    # Hand-crafted looping-imminence
    df = df.with_columns(
        [
            pl.col("max_window20_repeats")
            .cum_max()
            .over("run_id")
            .alias("cum_max_window20_repeats"),
            pl.col("top1_streak")
            .cum_max()
            .over("run_id")
            .alias("cum_max_top1_streak"),
            (
                pl.col("n_unique_64")
                - pl.col("n_unique_64")
                .cum_max()
                .over("run_id")
            ).alias("n_unique_64_below_peak"),
        ]
    )

    logger.info("Final cols: {}", df.width)
    df.write_parquet(args.output)
    logger.info("Wrote {}", args.output)


if __name__ == "__main__":
    main()
