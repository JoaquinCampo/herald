"""Build the per-segment K=16 onset-anchored dataset.

CPU-only. For each compressed run:
  1. Partition the generated tokens into segments of K=16.
  2. Aggregate per-segment cheap logit features (mean, last, max, min,
     slope) from `results/phase1/final/tokens/`.
  3. Add surface repetition features from `runs.parquet`'s
     `generated_token_ids`: top-1 streak, unique-tokens-in-history,
     last-window-repeat-count, bigram-dup-rate.
  4. Merge with onsets and label `y = 1 iff first_onset in next K`.
  5. Drop segments where seg_end_tok >= first_onset (post-onset).

All temporal features are causal: segment ending at token t depends
only on tokens 0..t.

Output: results/phase2_v2/segments_k16.parquet
        results/phase2_v2/segments_k16_summary.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

CHEAP_FEATURES = [
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
]


def aggregate_tokens_to_segments(
    tokens_df: pl.DataFrame, k: int
) -> pl.DataFrame:
    """Per-segment aggregates of cheap features.

    seg_idx = token_pos // k
    seg_end_tok = (seg_idx + 1) * k - 1 (clipped to last token)

    For each (run_id, seg_idx) and each feature, emits:
      f_mean, f_last, f_max, f_min, f_slope
    Slope = least-squares slope of feature vs token_pos within seg.
    """
    df = tokens_df.with_columns(
        (pl.col("token_pos") // k).alias("seg_idx")
    )

    agg_exprs: list[pl.Expr] = []
    for f in CHEAP_FEATURES:
        agg_exprs.extend(
            [
                pl.col(f).mean().alias(f"{f}_mean"),
                pl.col(f).last().alias(f"{f}_last"),
                pl.col(f).max().alias(f"{f}_max"),
                pl.col(f).min().alias(f"{f}_min"),
                pl.col(f).std().alias(f"{f}_std"),
            ]
        )
    agg_exprs.append(pl.col("token_pos").max().alias("seg_end_tok"))
    agg_exprs.append(pl.col("token_pos").count().alias("seg_n_tok"))

    seg = df.group_by(["run_id", "seg_idx"]).agg(agg_exprs)

    # Slope per (run_id, seg_idx, feature): single linear-regression
    # coefficient. Compute as Cov(x, t) / Var(t) per group via polars.
    slope_exprs: list[pl.Expr] = []
    for f in CHEAP_FEATURES:
        slope_exprs.append(
            (
                (
                    (pl.col(f) - pl.col(f).mean())
                    * (pl.col("token_pos") - pl.col("token_pos").mean())
                ).sum()
                / (
                    (pl.col("token_pos") - pl.col("token_pos").mean())
                    .pow(2)
                    .sum()
                    + 1e-9
                )
            ).alias(f"{f}_slope")
        )
    slope = df.group_by(["run_id", "seg_idx"]).agg(slope_exprs)
    seg = seg.join(slope, on=["run_id", "seg_idx"], how="left")
    return seg


def surface_features_per_segment(
    token_ids: list[int], k: int, seg_end_positions: list[int]
) -> pd.DataFrame:
    """Causal surface repetition features at each seg_end position.

    For each seg_end_tok t (inclusive), looks at tokens 0..t and emits:
      top1_streak: longest consecutive run of identical ids ending at t
      n_unique_64: |{token_ids[max(0,t-63):t+1]}|
      last_k_repeat_count: # of times tokens[t-k+1:t+1] appears as a
                          subsequence in tokens[0:t-k+1] (search is
                          O(t * k); ok for t up to 512)
      bigram_dup_rate_64: among bigrams in last 64 tokens, fraction
                          that appeared as a bigram earlier
      max_window20_repeats: max # times any 20-token window appeared
                            in tokens[0..t]; mirrors detect_looping
    """
    arr = np.asarray(token_ids, dtype=np.int64)
    rows: list[dict[str, int | float]] = []
    for t in seg_end_positions:
        if t < 0 or t >= len(arr):
            rows.append(
                {
                    "top1_streak": 0,
                    "n_unique_64": 0,
                    "last_k_repeat_count": 0,
                    "bigram_dup_rate_64": 0.0,
                    "max_window20_repeats": 0,
                }
            )
            continue

        # top1_streak ending at t
        streak = 1
        for i in range(t - 1, -1, -1):
            if arr[i] == arr[t]:
                streak += 1
            else:
                break

        # last 64 tokens
        win64 = arr[max(0, t - 63) : t + 1]
        n_unique_64 = int(np.unique(win64).size)

        # bigram dup rate over last 64
        if len(win64) >= 2:
            recent_bg = list(zip(win64[:-1], win64[1:]))
            history = arr[: max(0, t - 63)]
            if len(history) >= 2:
                hist_bg = set(zip(history[:-1], history[1:]))
            else:
                hist_bg = set()
            bigram_dup = (
                sum(1 for bg in recent_bg if bg in hist_bg) / len(recent_bg)
            )
        else:
            bigram_dup = 0.0

        # last_k_repeat_count: how often last k tokens appeared as
        # subsequence earlier in arr[0:t+1-k]
        if t + 1 >= k:
            target = arr[t + 1 - k : t + 1]
            target_bytes = target.tobytes()
            history = arr[: t + 1 - k]
            count = 0
            if len(history) >= k:
                # sliding-window naive search
                for i in range(len(history) - k + 1):
                    if history[i : i + k].tobytes() == target_bytes:
                        count += 1
            last_k_repeat_count = count
        else:
            last_k_repeat_count = 0

        # max_window20_repeats over tokens[0..t]
        window_size = 20
        if t + 1 >= window_size:
            seen: dict[bytes, int] = {}
            max_rep = 0
            for i in range(t + 1 - window_size + 1):
                key = arr[i : i + window_size].tobytes()
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > max_rep:
                    max_rep = seen[key]
            max_window20_repeats = max_rep
        else:
            max_window20_repeats = 0

        rows.append(
            {
                "top1_streak": int(streak),
                "n_unique_64": int(n_unique_64),
                "last_k_repeat_count": int(last_k_repeat_count),
                "bigram_dup_rate_64": float(bigram_dup),
                "max_window20_repeats": int(max_window20_repeats),
            }
        )
    return pd.DataFrame(rows)


def build_segment_table(
    tokens_root: Path,
    runs: pd.DataFrame,
    onsets: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    """End-to-end builder. Iterates (press, ratio) partitions, builds
    per-segment aggregates, joins onsets, computes labels, returns one
    big DataFrame.
    """
    onsets_min = onsets[
        ["run_id", "looping_onset", "non_termination_onset"]
    ]
    runs_meta = runs[
        [
            "run_id",
            "prompt_id",
            "press",
            "compression_ratio",
            "num_tokens_generated",
            "generated_token_ids",
        ]
    ].merge(onsets_min, on="run_id", how="left")
    # filter compressed
    runs_meta = runs_meta[runs_meta["press"] != "none"].reset_index(
        drop=True
    )
    logger.info("Compressed runs to process: {}", len(runs_meta))

    runs_meta["first_onset"] = runs_meta.apply(
        lambda r: (
            min(
                v for v in [r["looping_onset"], r["non_termination_onset"]]
                if pd.notna(v)
            )
            if (
                pd.notna(r["looping_onset"])
                or pd.notna(r["non_termination_onset"])
            )
            else None
        ),
        axis=1,
    )

    # Read all token partitions
    presses = sorted(
        d.split("=", 1)[1]
        for d in os.listdir(tokens_root)
        if d.startswith("press=")
    )
    seg_parts: list[pl.DataFrame] = []
    for press in presses:
        press_dir = tokens_root / f"press={press}"
        ratios = sorted(
            d.split("=", 1)[1]
            for d in os.listdir(press_dir)
            if d.startswith("ratio=")
        )
        for ratio_str in ratios:
            part = press_dir / f"ratio={ratio_str}"
            logger.info(
                "Reading partition press={} ratio={}", press, ratio_str
            )
            tdf = pl.read_parquet(
                part,
                columns=["run_id", "token_pos"] + CHEAP_FEATURES,
            )
            seg = aggregate_tokens_to_segments(tdf, k)
            seg = seg.with_columns(
                pl.lit(press).alias("press"),
                pl.lit(float(ratio_str)).alias("compression_ratio"),
            )
            seg_parts.append(seg)
    seg_all = pl.concat(seg_parts).to_pandas()
    logger.info("Total segments after aggregation: {}", len(seg_all))

    # Join with run metadata (drops compressed_ratio dup with suffix)
    seg_all = seg_all.merge(
        runs_meta[
            [
                "run_id",
                "prompt_id",
                "num_tokens_generated",
                "first_onset",
                "looping_onset",
                "non_termination_onset",
                "generated_token_ids",
            ]
        ],
        on="run_id",
        how="inner",
    )

    # Drop post-onset segments and label
    keep = []
    y_vals: list[int] = []
    for r in seg_all.itertuples(index=False):
        seg_end = int(r.seg_end_tok)
        if pd.notna(r.first_onset) and seg_end >= int(r.first_onset):
            keep.append(False)
            y_vals.append(0)
            continue
        keep.append(True)
        if pd.isna(r.first_onset):
            y_vals.append(0)
        else:
            lo, hi = seg_end + 1, seg_end + k
            y_vals.append(int(lo <= int(r.first_onset) <= hi))
    seg_all["y"] = y_vals
    seg_all = seg_all[keep].reset_index(drop=True)
    logger.info(
        "After post-onset drop: {} segments, positives: {} ({:.3%})",
        len(seg_all),
        int(seg_all["y"].sum()),
        float(seg_all["y"].mean()),
    )

    # Surface repetition features per (run_id, seg_idx)
    logger.info("Computing surface repetition features...")
    surface_rows: list[pd.DataFrame] = []
    grouped = seg_all.groupby("run_id", sort=False)
    n_runs = len(grouped)
    processed = 0
    for run_id, group in grouped:
        token_ids_row = group["generated_token_ids"].iloc[0]
        if token_ids_row is None:
            continue
        token_ids = list(token_ids_row)
        seg_ends = group["seg_end_tok"].astype(int).tolist()
        feats = surface_features_per_segment(token_ids, k, seg_ends)
        feats["run_id"] = run_id
        feats["seg_idx"] = group["seg_idx"].astype(int).values
        surface_rows.append(feats)
        processed += 1
        if processed % 2000 == 0:
            logger.info("  surface progress: {}/{}", processed, n_runs)
    surface_df = pd.concat(surface_rows, ignore_index=True)

    seg_all = seg_all.merge(
        surface_df, on=["run_id", "seg_idx"], how="left"
    )
    seg_all = seg_all.drop(columns=["generated_token_ids"])
    seg_all["relative_progress"] = (
        seg_all["seg_end_tok"].astype(float)
        / seg_all["num_tokens_generated"].astype(float)
    ).clip(0.0, 1.0)
    return seg_all


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokens-root",
        type=Path,
        default=Path("results/phase1/final/tokens"),
    )
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
        "--output",
        type=Path,
        default=Path("results/phase2_v2/segments_k16.parquet"),
    )
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="If >0, subset to first N runs for a smoke test.",
    )
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs = pd.read_parquet(args.runs)
    onsets = pd.read_parquet(args.onsets)
    if args.smoke > 0:
        compressed = runs[runs["press"] != "none"].sample(
            n=args.smoke, random_state=0
        )
        runs = pd.concat(
            [runs[runs["press"] == "none"], compressed], ignore_index=True
        )
        logger.info(
            "SMOKE mode: subset to {} compressed runs", len(compressed)
        )

    table = build_segment_table(args.tokens_root, runs, onsets, args.k)
    logger.info("Final table: {} rows, {} cols", *table.shape)
    table.to_parquet(args.output, index=False)
    logger.info("Wrote {}", args.output)

    summary = {
        "n_segments": int(len(table)),
        "n_positives": int(table["y"].sum()),
        "pos_rate": float(table["y"].mean()),
        "n_features": int(table.shape[1] - 1),
        "feature_cols": [c for c in table.columns if c != "y"],
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2)
    )


if __name__ == "__main__":
    main()
