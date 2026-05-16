"""Offline calibration: segment-mean risk-score distribution.

Pipeline (CPU only, no generation rerun):

  1. load phase2_tokens.parquet
  2. filter to press == "knorm" (closest analog to decoding_knorm)
  3. compute fold0 holdout prompts via the exact same iter_splits
     as predictor_baselines (GroupKFold over prompt_id, n=5, seed=42)
  4. score every token through models/phase4_lr_all_cheap.json
  5. aggregate non-overlapping K=16 windows per (run_id), matching
     HeraldOnlineProcessor (buffer.clear() after each fire)
  6. emit segment-score distribution + quantile threshold table
     (p50/p75/p90/p95/p99) with relax/tighten action rates

Outputs to --output-dir:
  segment_scores.parquet   one row per (run_id, segment_idx)
  threshold_table.json     {threshold, quantile, relax_rate,
                            tighten_rate, mean_score_above,
                            mean_score_below, n_segments_total}
  distribution_summary.json metadata + percentile snapshot

The quantile thresholds are calibration *candidates*; the actual decision
on whether to run the GPU smoke depends on whether the offline
distribution overlaps the prior online range observed during the
3-prompt smoke (segment scores in [0.002, 0.109] under decoding_knorm).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import GroupKFold

from herald.phase4_online_features import CHEAP_ALL_FEATURE_ORDER
from herald.phase4_predictor import ExportedPredictor

PRIOR_ONLINE_MIN = 0.002
PRIOR_ONLINE_MAX = 0.109


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tokens",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--predictor",
        type=Path,
        default=Path("models/phase4_lr_all_cheap.json"),
    )
    ap.add_argument("--press", default="knorm")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=[0.50, 0.75, 0.90, 0.95, 0.99],
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4/calibration"),
    )
    return ap.parse_args(argv)


def holdout_prompt_ids(
    df: pl.DataFrame,
    n_splits: int,
    seed: int,
    fold: int,
) -> list[str]:
    """Return prompt_ids that fall in fold-`fold` of GroupKFold.

    Mirrors predictor_baselines.iter_splits("prompts"). Note: GroupKFold
    in sklearn ignores `seed` (groups are partitioned deterministically
    by group order). We pass it for documentation parity.
    """
    del seed  # GroupKFold doesn't shuffle; kept for arg parity.
    groups = df["prompt_id"].to_numpy()
    n = len(groups)
    gkf = GroupKFold(n_splits=n_splits)
    for fold_i, (_tr, te) in enumerate(
        gkf.split(np.arange(n), groups=groups)
    ):
        if fold_i != fold:
            continue
        return sorted({str(g) for g in groups[te].tolist()})
    raise ValueError(f"fold {fold} not produced by n_splits={n_splits}")


def score_tokens(
    df: pl.DataFrame, predictor: ExportedPredictor
) -> np.ndarray:
    """Score every row; matches ExportedPredictor.score semantics."""
    columns = list(CHEAP_ALL_FEATURE_ORDER)
    arr = (
        df.select(columns)
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float64)
    )
    cleaned: np.ndarray = np.nan_to_num(
        arr, nan=0.0, posinf=0.0, neginf=0.0
    )
    return predictor.score(cleaned)


def aggregate_segments(
    df: pl.DataFrame, k: int
) -> pl.DataFrame:
    """Non-overlapping K-window mean per (run_id, segment_idx).

    Matches HeraldOnlineProcessor (buffer.clear() after fire). The
    offline tokens parquet stores one row per generated token in order,
    so segment_idx = floor(token_pos / k). Drops trailing partial
    windows (the online processor only fires when the buffer is full).
    """
    counts = (
        df.group_by("run_id")
        .agg(pl.len().alias("n_tokens"))
        .with_columns(
            (pl.col("n_tokens") // k * k).alias("n_full_tokens")
        )
    )
    df2 = df.join(counts, on="run_id", how="left").filter(
        pl.col("token_pos") < pl.col("n_full_tokens")
    )
    df2 = df2.with_columns(
        (pl.col("token_pos") // k).cast(pl.Int64).alias("segment_idx")
    )
    out = (
        df2.group_by("run_id", "segment_idx")
        .agg(
            [
                pl.col("score").mean().alias("segment_score_mean"),
                pl.col("press").first().alias("press"),
                pl.col("compression_ratio").first().alias(
                    "compression_ratio"
                ),
                pl.col("prompt_id").first().alias("prompt_id"),
            ]
        )
        .sort(["run_id", "segment_idx"])
    )
    return out


def build_threshold_table(
    seg_scores: np.ndarray, quantiles: list[float]
) -> list[dict[str, Any]]:
    """Threshold candidates with relax/tighten action rates.

    Under RiskBudgetStepPolicy:
      relax_rate    = P(segment_score > T)  -> spend budget
      tighten_rate  = P(segment_score <= T) -> save budget
    """
    rows: list[dict[str, Any]] = []
    for q in quantiles:
        t = float(np.quantile(seg_scores, q))
        above = seg_scores > t
        below = ~above
        n_above = int(above.sum())
        n_below = int(below.sum())
        rows.append(
            {
                "quantile": float(q),
                "threshold": t,
                "n_segments_total": int(seg_scores.shape[0]),
                "relax_rate": float(n_above / seg_scores.shape[0]),
                "tighten_rate": float(
                    n_below / seg_scores.shape[0]
                ),
                "n_above": n_above,
                "n_below": n_below,
                "mean_score_above": (
                    float(seg_scores[above].mean()) if n_above else None
                ),
                "mean_score_below": (
                    float(seg_scores[below].mean()) if n_below else None
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.tokens.exists():
        print(f"tokens parquet missing: {args.tokens}", file=sys.stderr)
        return 2
    if not args.predictor.exists():
        print(f"predictor missing: {args.predictor}", file=sys.stderr)
        return 2

    print(f"loading predictor {args.predictor}...")
    predictor = ExportedPredictor.from_path(args.predictor)

    needed = list(CHEAP_ALL_FEATURE_ORDER) + [
        "run_id",
        "token_pos",
        "press",
        "compression_ratio",
        "prompt_id",
    ]
    needed = sorted(set(needed))
    print(f"scanning tokens, selecting {len(needed)} columns...")
    lazy = pl.scan_parquet(args.tokens).select(needed)
    lazy = lazy.filter(pl.col("press") == args.press)
    df = lazy.collect()
    print(
        f"  press={args.press} rows={df.height} "
        f"prompts={df['prompt_id'].n_unique()}"
    )

    holdout = holdout_prompt_ids(
        df, n_splits=args.n_splits, seed=args.seed, fold=args.fold
    )
    df_h = df.filter(pl.col("prompt_id").is_in(holdout))
    print(
        f"  fold{args.fold} holdout: {len(holdout)} prompts, "
        f"{df_h.height} rows"
    )

    print("scoring tokens...")
    scores = score_tokens(df_h, predictor)
    df_h = df_h.with_columns(pl.Series("score", scores))

    print(f"aggregating non-overlapping K={args.k} segments...")
    segments = aggregate_segments(df_h, k=args.k)
    print(f"  segments: {segments.height}")

    seg_score_arr = segments["segment_score_mean"].to_numpy()
    table = build_threshold_table(seg_score_arr, args.quantiles)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_snap = {
        f"p{p}": float(np.percentile(seg_score_arr, p)) for p in pcts
    }

    online_in_offline = {
        "online_min": PRIOR_ONLINE_MIN,
        "online_max": PRIOR_ONLINE_MAX,
        "offline_quantile_at_online_max": float(
            (seg_score_arr <= PRIOR_ONLINE_MAX).mean()
        ),
        "offline_quantile_at_online_min": float(
            (seg_score_arr <= PRIOR_ONLINE_MIN).mean()
        ),
    }

    summary = {
        "tokens_path": str(args.tokens),
        "predictor_path": str(args.predictor),
        "press_filter": args.press,
        "fold": args.fold,
        "n_splits": args.n_splits,
        "k": args.k,
        "n_holdout_prompts": len(holdout),
        "n_segments": int(segments.height),
        "n_runs": int(segments["run_id"].n_unique()),
        "score_min": float(seg_score_arr.min()),
        "score_max": float(seg_score_arr.max()),
        "score_mean": float(seg_score_arr.mean()),
        "score_std": float(seg_score_arr.std()),
        "percentiles": pct_snap,
        "online_overlay_3prompt_smoke": online_in_offline,
        "feature_order": list(CHEAP_ALL_FEATURE_ORDER),
        "predictor_label_threshold": predictor.threshold,
    }

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    segments.write_parquet(out_dir / "segment_scores.parquet")
    (out_dir / "threshold_table.json").write_text(
        json.dumps(table, indent=2)
    )
    (out_dir / "distribution_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("--- threshold table ---")
    for row in table:
        print(
            f"  q={row['quantile']:.2f} t={row['threshold']:.6f} "
            f"relax={row['relax_rate']:.4f} "
            f"tighten={row['tighten_rate']:.4f}"
        )
    print("--- offline / online overlay ---")
    print(
        f"  prior online range "
        f"[{PRIOR_ONLINE_MIN:.4f}, {PRIOR_ONLINE_MAX:.4f}]"
    )
    print(
        f"  offline P(score <= online_max={PRIOR_ONLINE_MAX:.3f}) "
        f"= {online_in_offline['offline_quantile_at_online_max']:.4f}"
    )
    print(
        f"  offline P(score <= online_min={PRIOR_ONLINE_MIN:.3f}) "
        f"= {online_in_offline['offline_quantile_at_online_min']:.4f}"
    )
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
