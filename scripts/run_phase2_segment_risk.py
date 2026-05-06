"""Phase 2b Task 4: segment / run risk analysis.

The Phase 2 lead-time analysis showed that the JS-trained per-token
predictor cannot reliably anticipate looping/non-termination *onset*.
This script asks the weaker but more deployable question: do
segment- and run-level aggregates of per-token predictor risk
correspond to user-facing damage, and do they beat segment-level
entropy?

Method
------
1. Reuse the existing OOF per-token scores from
   `results/phase2/per_run/oof_predictions.parquet` (these are the
   `lr_all_cheap` predictions trained out-of-fold on the prompts split
   with `future_sum_js_25` as label — same predictor as the Phase 2
   per-run validation). Join entropy from the per-token parquet
   partitions for the same (run_id, token_pos).
2. For each K in {8, 16, 32}: bucket tokens into segments of length
   K (segment_id = token_pos // K), then compute per-segment
   aggregates (max, mean, p95, count above threshold) for both
   score sources.
3. Reduce per-run by max over segments, then correlate with
   `run_damage.parquet` validators (Spearman ρ for continuous
   severity, AUROC for binary tags).
4. Compare predictor-segment vs entropy-segment scores head-to-head
   on the same set of runs.

Outputs
-------
- results/phase2/segment_risk/segment_risk.parquet
  (one row per (run_id, K, score_source) with reduced statistics)
- results/phase2/segment_risk/segment_risk_summary.json
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_auroc(y: list[int], s: list[float]) -> float | None:
    if len(y) < 2 or len(set(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _safe_auprc(y: list[int], s: list[float]) -> float | None:
    if len(y) < 2 or len(set(y)) < 2:
        return None
    try:
        return float(average_precision_score(y, s))
    except ValueError:
        return None


def _safe_spearman(
    a: np.ndarray, b: np.ndarray
) -> tuple[float | None, float | None]:
    if a.size < 3:
        return None, None
    try:
        rho, p = spearmanr(a, b)
        if rho is None or np.isnan(rho):
            return None, None
        return float(rho), float(p)
    except Exception:
        return None, None


def _segment_aggregates(
    df: pl.DataFrame,
    score_col: str,
    K: int,
    threshold: float,
) -> pl.DataFrame:
    """Per (run_id, segment_id=token_pos//K) aggregates of `score_col`.

    Returns columns: run_id, segment_id, n_tokens, seg_max, seg_mean,
    seg_p95, seg_n_above.
    """
    out = (
        df.with_columns(
            (pl.col("token_pos") // K).alias("segment_id"),
        )
        .group_by(["run_id", "segment_id"])
        .agg(
            pl.col(score_col).len().alias("n_tokens"),
            pl.col(score_col).max().alias("seg_max"),
            pl.col(score_col).mean().alias("seg_mean"),
            pl.col(score_col)
            .quantile(0.95, interpolation="linear")
            .alias("seg_p95"),
            (pl.col(score_col) >= threshold).sum().alias("seg_n_above"),
        )
    )
    return out


def _per_run_from_segments(seg: pl.DataFrame) -> pl.DataFrame:
    """Reduce per-segment scores to per-run via max over segments
    (and a few alternative reductions for ablation)."""
    return seg.group_by("run_id").agg(
        pl.col("seg_max").max().alias("run_max_seg_max"),
        pl.col("seg_mean").max().alias("run_max_seg_mean"),
        pl.col("seg_p95").max().alias("run_max_seg_p95"),
        pl.col("seg_n_above").max().alias("run_max_seg_n_above"),
        pl.col("seg_max").mean().alias("run_mean_seg_max"),
        pl.col("seg_n_above").sum().alias("run_total_n_above"),
        pl.col("seg_max").len().alias("n_segments"),
    )


CONTINUOUS_VALIDATORS: tuple[str, ...] = (
    "quality_delta",
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "sum_kl",
    "sum_js",
    "nll_ratio_flipped",
)
BINARY_VALIDATORS: tuple[str, ...] = (
    "gross_harm_final",
    "gross_help_final",
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
)


def _correlate_run_scores(
    per_run: pl.DataFrame,
    run_damage: pl.DataFrame,
    score_cols: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    joined = per_run.join(run_damage, on="run_id", how="inner")
    out: dict[str, dict[str, Any]] = {}
    for sc in score_cols:
        if sc not in joined.columns:
            continue
        block: dict[str, Any] = {"n_runs": int(joined.height)}
        for col in CONTINUOUS_VALIDATORS:
            if col not in joined.columns:
                continue
            sub = joined.drop_nulls(subset=[sc, col])
            rho, p = _safe_spearman(sub[sc].to_numpy(), sub[col].to_numpy())
            block[f"spearman_{col}"] = (
                round(rho, 4) if rho is not None else None
            )
            block[f"n_used_{col}"] = int(sub.height)
        for col in BINARY_VALIDATORS:
            if col not in joined.columns:
                continue
            sub = joined.drop_nulls(subset=[sc, col])
            if sub.is_empty():
                block[f"auroc_{col}"] = None
                block[f"auprc_{col}"] = None
                block[f"n_used_{col}"] = 0
                continue
            y = sub[col].cast(pl.Int8).to_list()
            s = sub[sc].to_list()
            block[f"auroc_{col}"] = _safe_auroc(y, s)
            block[f"auprc_{col}"] = _safe_auprc(y, s)
            block[f"n_used_{col}"] = int(sub.height)
            block[f"n_pos_{col}"] = int(sum(y))
        out[sc] = block
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--oof-path",
        type=Path,
        default=Path("results/phase2/per_run/oof_predictions.parquet"),
        help=(
            "Per-token OOF predictions from the Phase 2 per-run "
            "validation script (lr_all_cheap, future_sum_js_25)."
        ),
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--run-damage",
        type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/segment_risk"),
    )
    ap.add_argument("--K-list", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--threshold-predictor", type=float, default=0.5)
    ap.add_argument(
        "--threshold-entropy",
        type=float,
        default=2.0,
        help="Token-level entropy threshold for n_above counts.",
    )
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.oof_path)
    oof = pl.read_parquet(args.oof_path)
    logger.info("oof rows={} cols={}", oof.height, oof.columns)
    oof = oof.drop_nulls("oof_score")

    logger.info("loading entropy from dataset")
    ds = pl.read_parquet(args.dataset).select(
        ["run_id", "token_pos", "entropy"]
    )
    joined = oof.join(ds, on=["run_id", "token_pos"], how="left")
    logger.info("joined rows={}", joined.height)

    rd = pl.read_parquet(args.run_damage)
    if "nll_ratio" in rd.columns:
        rd = rd.with_columns(
            (-pl.col("nll_ratio")).alias("nll_ratio_flipped")
        )

    all_per_run_blocks: list[pl.DataFrame] = []
    summary: dict[str, Any] = {
        "oof_path": str(args.oof_path),
        "dataset": str(args.dataset),
        "K_list": args.K_list,
        "threshold_predictor": args.threshold_predictor,
        "threshold_entropy": args.threshold_entropy,
        "validation": {},
    }
    t0 = time.time()
    for K in args.K_list:
        # Predictor segments.
        seg_pred = _segment_aggregates(
            joined,
            score_col="oof_score",
            K=K,
            threshold=args.threshold_predictor,
        )
        per_run_pred = _per_run_from_segments(seg_pred)
        per_run_pred = per_run_pred.with_columns(
            pl.lit("predictor").alias("score_source"),
            pl.lit(K).alias("K"),
        )

        # Entropy segments (skip rows with null entropy).
        seg_ent = _segment_aggregates(
            joined.drop_nulls("entropy"),
            score_col="entropy",
            K=K,
            threshold=args.threshold_entropy,
        )
        per_run_ent = _per_run_from_segments(seg_ent)
        per_run_ent = per_run_ent.with_columns(
            pl.lit("entropy").alias("score_source"),
            pl.lit(K).alias("K"),
        )

        # Validate against run_damage.
        score_cols = (
            "run_max_seg_max",
            "run_max_seg_mean",
            "run_max_seg_p95",
            "run_max_seg_n_above",
            "run_total_n_above",
        )
        block_pred = _correlate_run_scores(
            per_run_pred, rd, score_cols=score_cols
        )
        block_ent = _correlate_run_scores(
            per_run_ent, rd, score_cols=score_cols
        )
        summary["validation"][f"K={K}"] = {
            "predictor": block_pred,
            "entropy": block_ent,
        }
        all_per_run_blocks.append(per_run_pred)
        all_per_run_blocks.append(per_run_ent)
        logger.info("K={} done", K)

    full = pl.concat(all_per_run_blocks, how="vertical_relaxed")
    full.write_parquet(args.output_dir / "segment_risk.parquet")
    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary["n_segment_rows"] = int(full.height)
    (args.output_dir / "segment_risk_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print()
    print("=== Segment / run risk headline ===")
    print("(Spearman ρ, run_max_seg_max → continuous validators)")
    print(
        f"{'K':>3}  {'src':<10}  {'rouge_l_drop':>14}  "
        f"{'sum_js':>8}  {'looping_auroc':>15}  "
        f"{'nonterm_auroc':>15}"
    )
    for K in args.K_list:
        for src in ("predictor", "entropy"):
            blk = summary["validation"][f"K={K}"][src]["run_max_seg_max"]
            print(
                f"{K:>3}  {src:<10}  "
                f"{str(blk.get('spearman_rouge_l_drop')):>14}  "
                f"{str(blk.get('spearman_sum_js')):>8}  "
                f"{str(blk.get('auroc_has_looping')):>15}  "
                f"{str(blk.get('auroc_has_non_termination')):>15}"
            )


if __name__ == "__main__":
    main()
