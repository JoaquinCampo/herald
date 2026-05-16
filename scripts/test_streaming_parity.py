"""Parity test: streaming predictor vs offline train_phase2_v2_xgb scores.

Picks N runs from a held-out fold, replays their tokens through
StreamingHeraldPredictor, then asserts that the per-segment scores
match the offline scores within tolerance.

Usage:
  uv run python scripts/test_streaming_parity.py \\
    --tokens-root results/phase1/final/tokens \\
    --runs results/phase1/final/runs.parquet \\
    --scores results/phase2_v2/xgb_ext/scores_fold0.parquet \\
    --model results/phase2_v2/xgb_ext/model_fold0.json \\
    --summary results/phase2_v2/xgb_ext/summary.json \\
    --n-runs 5
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from herald.config import TokenSignals
from herald.online_segment_predictor import (
    CHEAP_FEATURES,
    StreamingHeraldPredictor,
)


def stream_for_run(
    tokens_df: pd.DataFrame,
    token_ids: list[int],
) -> list[tuple[TokenSignals, int]]:
    """Build (TokenSignals, token_id) iterable in token_pos order."""
    tokens_df = tokens_df.sort_values("token_pos").reset_index(drop=True)
    pairs: list[tuple[TokenSignals, int]] = []
    for i, row in tokens_df.iterrows():
        kw: dict[str, object] = {}
        for f in CHEAP_FEATURES:
            v = row.get(f, math.nan)
            if v is None or (isinstance(v, float) and math.isnan(v)):
                kw[f] = float("nan")
            else:
                kw[f] = float(v)
        # delta_h is a NanFloat in TokenSignals; pass NaN if missing.
        sig = TokenSignals(
            entropy=kw["entropy"],
            top1_prob=kw["top1_prob"],
            top5_prob=kw["top5_prob"],
            h_alts=kw["h_alts"],
            avg_logp=kw["avg_logp"],
            delta_h=kw["delta_h"],
            delta_h_valid=not math.isnan(kw["delta_h"]),
            kl_div=kw["kl_div"],
            top10_jaccard=kw["top10_jaccard"],
            eff_vocab_size=kw["eff_vocab_size"],
            tail_mass=kw["tail_mass"],
            logit_range=kw["logit_range"],
        )
        tok = int(token_ids[i]) if i < len(token_ids) else 0
        pairs.append((sig, tok))
    return pairs


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
        "--scores",
        type=Path,
        default=Path(
            "results/phase2_v2/xgb_ext/scores_fold0.parquet"
        ),
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=Path(
            "results/phase2_v2/xgb_ext/model_fold0.json"
        ),
    )
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/phase2_v2/xgb_ext/summary.json"
        ),
    )
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--max-budget", type=int, default=512)
    ap.add_argument(
        "--tol", type=float, default=1e-4, help="Max abs score diff."
    )
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    feature_cols = summary["feature_cols"]
    logger.info("feature_cols: {}", len(feature_cols))

    scores_df = pd.read_parquet(args.scores)
    logger.info("offline scores: {} rows", len(scores_df))

    runs_meta = pd.read_parquet(args.runs)[
        [
            "run_id",
            "press",
            "compression_ratio",
            "generated_token_ids",
            "num_tokens_generated",
        ]
    ]

    # Pick N distinct run_ids from scores parquet that have token data.
    candidate = scores_df["run_id"].drop_duplicates().tolist()
    diffs: list[float] = []
    n_compared = 0
    n_runs_done = 0
    for run_id in candidate:
        if n_runs_done >= args.n_runs:
            break
        meta = runs_meta[runs_meta["run_id"] == run_id]
        if meta.empty:
            continue
        meta_row = meta.iloc[0]
        press = str(meta_row["press"])
        ratio = float(meta_row["compression_ratio"])
        tok_ids = meta_row["generated_token_ids"]
        if tok_ids is None or len(tok_ids) == 0:
            continue
        tok_ids = list(tok_ids)

        # Read this run's token parquet
        part = (
            args.tokens_root
            / f"press={press}"
            / f"ratio={ratio}"
        )
        if not part.exists():
            logger.warning("partition missing: {}", part)
            continue
        tdf = pl.read_parquet(
            part, columns=["run_id", "token_pos", *CHEAP_FEATURES]
        ).filter(pl.col("run_id") == run_id).to_pandas()
        if tdf.empty:
            logger.warning("no tokens for {}", run_id)
            continue

        pred = StreamingHeraldPredictor.from_paths(
            model_path=args.model,
            feature_cols=feature_cols,
            press=press,
            compression_ratio=ratio,
            k=args.k,
            max_budget=args.max_budget,
        )
        pred.reset()
        stream = stream_for_run(tdf, tok_ids)
        streaming_scores: dict[int, float] = {}
        for sig, tok in stream:
            r = pred.update(sig, tok)
            if r is not None:
                streaming_scores[int(r["seg_idx"])] = float(r["score"])

        # Compare against offline scores for this run
        offline = scores_df[scores_df["run_id"] == run_id]
        for _, row in offline.iterrows():
            seg_idx = int(row["seg_idx"])
            if seg_idx not in streaming_scores:
                continue  # offline filtered post-onset, streaming has all
            d = abs(streaming_scores[seg_idx] - float(row["score"]))
            diffs.append(d)
            n_compared += 1
        logger.info(
            "run {}: streaming_segs={} offline_segs={} max_diff={:.6g}",
            run_id,
            len(streaming_scores),
            len(offline),
            max(
                (
                    abs(
                        streaming_scores[int(r["seg_idx"])]
                        - float(r["score"])
                    )
                    for _, r in offline.iterrows()
                    if int(r["seg_idx"]) in streaming_scores
                ),
                default=float("nan"),
            ),
        )
        n_runs_done += 1

    if not diffs:
        logger.error("No comparisons made.")
        raise SystemExit(2)
    diffs_arr = np.asarray(diffs)
    logger.info(
        "Compared {} segments across {} runs.  "
        "max_diff={:.6g} mean_diff={:.6g} pctile99={:.6g}",
        n_compared,
        n_runs_done,
        diffs_arr.max(),
        diffs_arr.mean(),
        np.percentile(diffs_arr, 99),
    )
    if diffs_arr.max() > args.tol:
        logger.error(
            "PARITY FAIL: max diff {:.6g} > tol {:.6g}",
            diffs_arr.max(),
            args.tol,
        )
        raise SystemExit(1)
    logger.info(
        "PARITY OK (max diff <= {:.6g})", args.tol
    )


if __name__ == "__main__":
    main()
