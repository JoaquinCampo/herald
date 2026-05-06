"""Phase 2 user-facing validation (Task 4).

Trains the best cheap-feature baseline (lr_all_cheap) on held-out-
prompt folds, emits OOF per-token predictions, aggregates them per
run (max, mean, p95, mean_top_10), and correlates the per-run
scores with `run_damage.parquet` validators:

- Spearman vs continuous severity (quality_delta, rouge_l_drop,
  sum_kl, sum_js, nll_ratio_flipped, …)
- AUROC/AUPRC vs gross_harm_final
- AUROC/AUPRC vs catastrophic tags (looping, non_termination, …)

The point is to demonstrate that token-level future-window labels
correspond to user-facing damage. NOT a training objective.

Outputs:
- `results/phase2/per_run/oof_predictions.parquet`
- `results/phase2/per_run/per_run_scores.parquet`
- `results/phase2/per_run/per_run_validation.json`

Spec: gold/phase-2-dataset.md, gold/research-plan.md Phase 2
"User-facing validation".
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    binarize_with_threshold,
    compute_train_quantile_threshold,
    iter_splits,
)
from herald.predictor_per_run import (
    aggregate_per_run,
    correlate_with_run_damage,
)


def _clean_X(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_predict_lr_oof(
    df: pl.DataFrame,
    label_col: str,
    feature_cols: list[str],
    n_splits: int,
    threshold_q: float,
    seed: int,
) -> pl.DataFrame:
    """5-fold OOF predictions on held-out-prompts.

    For each fold: derive p90 threshold from train rows, binarize,
    fit LR on train, score test rows. Concatenate test-fold scores.
    """
    n = df.height
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int32)

    feat_present = [c for c in feature_cols if c in df.columns]
    logger.info("OOF features ({}): {}", len(feat_present), feat_present)

    for tr, te, fid in iter_splits(
        df, kind="prompts", n_splits=n_splits, seed=seed
    ):
        train = df[tr]
        test = df[te]
        thr = compute_train_quantile_threshold(
            train[label_col], q=threshold_q
        )
        train_y_bin = binarize_with_threshold(train[label_col], thr)
        train_mask = ~train_y_bin.is_null()
        keep_train = train.filter(train_mask)
        keep_train_y = (
            train_y_bin.filter(train_mask).cast(pl.Int64).to_numpy()
        )
        if keep_train.height < 50 or len(set(keep_train_y.tolist())) < 2:
            logger.warning(
                "fold {}: skipping (n_train={} pos={})",
                fid,
                keep_train.height,
                int((keep_train_y == 1).sum()),
            )
            continue
        train_X = _clean_X(
            keep_train.select(feat_present)
            .fill_null(0.0)
            .to_numpy()
            .astype(float)
        )
        test_X = _clean_X(
            test.select(feat_present).fill_null(0.0).to_numpy().astype(float)
        )
        scaler = StandardScaler()
        Xs = scaler.fit_transform(train_X)
        Xt = scaler.transform(test_X)
        clf = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            solver="liblinear",
        )
        clf.fit(Xs, keep_train_y)
        oof[te] = clf.predict_proba(Xt)[:, 1]
        fold_id[te] = int(fid.lstrip("fold")) if fid.startswith("fold") else 0
        logger.info(
            "fold {} done: n_train={} n_test={} thr={:.4f} pos_train={:.3f}",
            fid,
            keep_train.height,
            test.height,
            thr if thr is not None else float("nan"),
            float((keep_train_y == 1).mean()),
        )

    out = df.select(
        [
            c
            for c in (
                "run_id",
                "prompt_id",
                "task",
                "press",
                "compression_ratio",
                "token_pos",
            )
            if c in df.columns
        ]
    ).with_columns(
        pl.Series(name="oof_score", values=oof),
        pl.Series(name="oof_fold", values=fold_id),
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
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
        default=Path("results/phase2/per_run"),
    )
    ap.add_argument(
        "--label",
        type=str,
        default="future_sum_js_25",
        help="Continuous label column to threshold for OOF training.",
    )
    ap.add_argument("--threshold-q", type=float, default=0.9)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=500_000,
        help=(
            "Subsample full dataset before splitting (LR over 7M rows is "
            "intractable on CPU)."
        ),
    )
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

    t0 = time.time()
    oof = _fit_predict_lr_oof(
        df,
        label_col=args.label,
        feature_cols=feature_cols,
        n_splits=args.n_splits,
        threshold_q=args.threshold_q,
        seed=args.seed,
    )
    oof_path = args.output_dir / "oof_predictions.parquet"
    oof.write_parquet(oof_path)
    logger.info("oof predictions -> {}", oof_path)

    # Aggregate per run.
    per_run = aggregate_per_run(
        oof.drop_nulls("oof_score"),
        score_col="oof_score",
        top_k=10,
        threshold=0.5,
    )
    per_run_path = args.output_dir / "per_run_scores.parquet"
    per_run.write_parquet(per_run_path)
    logger.info("per-run scores -> {}", per_run_path)

    # Correlate against run_damage.
    rd = pl.read_parquet(args.run_damage)
    if "nll_ratio" in rd.columns:
        rd = rd.with_columns(
            (-pl.col("nll_ratio")).alias("nll_ratio_flipped")
        )

    validation_blocks: dict[str, dict] = {}
    for agg_col in (
        "max_score",
        "mean_score",
        "p95_score",
        "mean_top_10_score",
        "n_above_0.5",
    ):
        if agg_col not in per_run.columns:
            continue
        validation_blocks[agg_col] = correlate_with_run_damage(
            per_run, rd, score_col=agg_col
        )

    out_summary = {
        "label": args.label,
        "threshold_q": args.threshold_q,
        "n_splits": args.n_splits,
        "max_train_rows": args.max_train_rows,
        "feature_cols": feature_cols,
        "n_runs_with_oof": int(per_run.height),
        "wall_seconds": round(time.time() - t0, 2),
        "validation": validation_blocks,
    }
    summary_path = args.output_dir / "per_run_validation.json"
    summary_path.write_text(json.dumps(out_summary, indent=2, default=str))
    logger.info("validation summary -> {}", summary_path)

    # Headline print.
    print()
    print("=== Per-run validation headline ===")
    print(f"label = {args.label}")
    print(f"n_runs_with_oof = {per_run.height}")
    print(f"wall_seconds = {out_summary['wall_seconds']}")
    print()
    for agg, block in validation_blocks.items():
        print(f"-- score = {agg} --")
        for col in (
            "spearman_quality_delta",
            "spearman_rouge_l_drop",
            "spearman_sum_js",
            "auroc_gross_harm_final",
            "auroc_has_looping",
            "auroc_has_non_termination",
        ):
            v = block.get(col)
            n_used = block.get(f"n_used_{col.split('_', 1)[1]}", "?")
            print(f"  {col:32s} = {v}   n_used={n_used}")


if __name__ == "__main__":
    main()
