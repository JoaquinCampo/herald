"""Phase 2b Task 2: alternative label-family sweep.

Re-runs the Phase 2 baseline / per-run / lead-time evaluations for
the alternative label families `future_sum_kl` and `future_max_js`,
plus `future_sum_js` as the reference.

Why
---
The Phase 2 baseline result trained on `future_sum_js`. The per-run
validation showed strong discrimination of looping/non-termination,
but the lead-time analysis showed the JS-sum-trained predictor goes
*below* 0.5 AUROC near catastrophic onset. Looping collapses into
low-divergence repetitive output, which is what the JS-sum label
penalises *less*. The advisor flag for this experiment is that
`future_max_js` is the label most likely to fix lead time at onset
(max does not collapse when post-onset tokens go quiet — onset itself
is a JS spike).

Comparators
-----------
Baseline AUROC test: target `lr_all_cheap` vs reference
`feature::entropy_mean_8` on the new label.

Per-run validation: Spearman ρ of per-run aggregate vs continuous
damage (`rouge_l_drop`, `sum_js`, `quality_delta`, …) and AUROC vs
binary tags. Reports whether the inversion behaviour from Phase 2
(catastrophic runs scoring lower than healthy) flips under the new
label — this is the controller-score evidence.

Lead time: re-runs the Phase 2 lead-time analysis with each label.

Outputs
-------
- results/phase2/label_sweep/label_sweep_results.parquet (baseline AUROCs
  per (split, fold, baseline, label, horizon))
- results/phase2/label_sweep/label_sweep_summary.json (per-label
  baseline best-cheap, per-run validation, lead-time)
- results/phase2/label_sweep/per_run/<label>/per_run_validation.json
- results/phase2/label_sweep/lead_time/<label>/lead_time_summary.json

The lead-time scripts are invoked as subprocesses to keep code reuse
clean; the baseline + per-run pieces run in-process.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    DEFAULT_QUANTILE,
    binarize_with_threshold,
    compute_train_quantile_threshold,
    evaluate_split,
    iter_splits,
    summarize_baselines,
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
    """5-fold OOF on held-out-prompts split. Mirrors the per-run
    validation runner (kept inline to avoid script-to-script imports
    and to allow per-label feature subsets if needed)."""
    n = df.height
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_id = np.full(n, -1, dtype=np.int32)

    feat_present = [c for c in feature_cols if c in df.columns]
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
            solver="lbfgs",
        )
        clf.fit(Xs, keep_train_y)
        oof[te] = clf.predict_proba(Xt)[:, 1]
        fold_id[te] = int(fid.lstrip("fold")) if fid.startswith("fold") else 0
        logger.info(
            "  fold {} done n_train={} thr={:.4f}",
            fid,
            keep_train.height,
            thr if thr is not None else float("nan"),
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
        "--phase1-root",
        type=Path,
        default=Path("results/phase1"),
        help="Used by the lead-time subprocess.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/label_sweep"),
    )
    ap.add_argument(
        "--label-bases",
        type=str,
        nargs="+",
        default=["future_sum_js", "future_sum_kl", "future_max_js"],
    )
    ap.add_argument(
        "--horizons", type=int, nargs="+", default=[5, 10, 25, 50]
    )
    ap.add_argument(
        "--per-run-horizon",
        type=int,
        default=25,
        help="Single H used for per-run + lead-time analyses.",
    )
    ap.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["prompts", "ratios", "presses", "tasks"],
    )
    ap.add_argument("--n-prompt-folds", type=int, default=5)
    ap.add_argument("--threshold-q", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument(
        "--max-train-rows-baseline",
        type=int,
        default=200_000,
    )
    ap.add_argument(
        "--max-train-rows-per-run",
        type=int,
        default=500_000,
    )
    ap.add_argument(
        "--skip-lead-time",
        action="store_true",
        help="Skip the lead-time subprocess (faster smoke run).",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "per_run").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lead_time").mkdir(parents=True, exist_ok=True)

    logger.info("loading {}", args.dataset)
    full = pl.read_parquet(args.dataset)
    logger.info("dataset rows={} cols={}", full.height, len(full.columns))

    # Subsample once for the baseline eval (matches Phase 2 runner).
    if (
        args.max_train_rows_baseline > 0
        and full.height > args.max_train_rows_baseline
    ):
        df_baseline = full.sample(
            n=args.max_train_rows_baseline,
            seed=args.seed,
            shuffle=True,
        )
        logger.info(
            "subsampled to {} rows for baseline eval",
            df_baseline.height,
        )
    else:
        df_baseline = full

    # Subsample for per-run OOF training.
    if (
        args.max_train_rows_per_run > 0
        and full.height > args.max_train_rows_per_run
    ):
        df_per_run = full.sample(
            n=args.max_train_rows_per_run,
            seed=args.seed,
            shuffle=True,
        )
    else:
        df_per_run = full

    rd = pl.read_parquet(args.run_damage)
    if "nll_ratio" in rd.columns:
        rd = rd.with_columns(
            (-pl.col("nll_ratio")).alias("nll_ratio_flipped")
        )

    all_baseline_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "label_bases": args.label_bases,
        "horizons": args.horizons,
        "per_run_horizon": args.per_run_horizon,
        "splits": args.splits,
        "max_train_rows_baseline": args.max_train_rows_baseline,
        "max_train_rows_per_run": args.max_train_rows_per_run,
        "labels": {},
    }

    t0 = time.time()
    for base in args.label_bases:
        per_label_block: dict[str, object] = {}
        # Baseline AUROC sweep across (H, split).
        baseline_rows: list[dict[str, object]] = []
        for h in args.horizons:
            label = f"{base}_{h}"
            if label not in df_baseline.columns:
                logger.warning("label {} missing, skip", label)
                continue
            logger.info("baseline eval: {}", label)
            rows = evaluate_split(
                df_baseline,
                label_col=label,
                threshold_q=args.threshold_q,
                splits=tuple(args.splits),
                n_prompt_folds=args.n_prompt_folds,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            for r in rows:
                r["label_base"] = base
                r["horizon"] = h
            baseline_rows.extend(rows)
        all_baseline_rows.extend(baseline_rows)
        summary_for_label = summarize_baselines(baseline_rows)
        per_label_block["baseline_summary"] = summary_for_label

        # Per-run validation at canonical H.
        per_run_label = f"{base}_{args.per_run_horizon}"
        if per_run_label in df_per_run.columns:
            logger.info("per-run OOF: {}", per_run_label)
            oof = _fit_predict_lr_oof(
                df_per_run,
                label_col=per_run_label,
                feature_cols=list(CHEAP_ALL_FEATURES),
                n_splits=args.n_prompt_folds,
                threshold_q=args.threshold_q,
                seed=args.seed,
            )
            per_run_dir = args.output_dir / "per_run" / base
            per_run_dir.mkdir(parents=True, exist_ok=True)
            oof.write_parquet(per_run_dir / "oof_predictions.parquet")
            per_run = aggregate_per_run(
                oof.drop_nulls("oof_score"),
                score_col="oof_score",
                top_k=10,
                threshold=0.5,
            )
            per_run.write_parquet(per_run_dir / "per_run_scores.parquet")
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
            per_run_summary = {
                "label": per_run_label,
                "n_runs_with_oof": int(per_run.height),
                "validation": validation_blocks,
            }
            (per_run_dir / "per_run_validation.json").write_text(
                json.dumps(per_run_summary, indent=2, default=str)
            )
            per_label_block["per_run"] = per_run_summary

        # Lead-time subprocess.
        if not args.skip_lead_time:
            lt_dir = args.output_dir / "lead_time" / base
            lt_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/run_phase2_lead_time.py",
                "--root",
                str(args.phase1_root),
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(lt_dir),
                "--label",
                per_run_label,
                "--seed",
                str(args.seed),
            ]
            logger.info("lead-time: {}", " ".join(cmd))
            sub = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            (lt_dir / "stdout.log").write_text(sub.stdout)
            (lt_dir / "stderr.log").write_text(sub.stderr)
            lt_summary_path = lt_dir / "lead_time_summary.json"
            if lt_summary_path.exists():
                per_label_block["lead_time"] = json.loads(
                    lt_summary_path.read_text()
                )
            else:
                per_label_block["lead_time"] = {
                    "error": "missing summary",
                    "returncode": sub.returncode,
                }
        summary["labels"][base] = per_label_block

    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary["n_baseline_rows"] = len(all_baseline_rows)

    pl.DataFrame(all_baseline_rows).write_parquet(
        args.output_dir / "label_sweep_results.parquet"
    )
    (args.output_dir / "label_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    logger.info(
        "done: {} baseline rows in {:.1f}s",
        len(all_baseline_rows),
        time.time() - t0,
    )

    print()
    print(
        "=== Label sweep headline (lr_all_cheap delta vs best-cheap, "
        "prompts H=25) ==="
    )
    for base in args.label_bases:
        block = summary["labels"].get(base, {}).get("baseline_summary", {})
        cell = block.get(f"prompts::{base}_25", {})
        dec = cell.get("decision", {}).get("lr_all_cheap")
        print(f"  {base:18s}: {dec}")


if __name__ == "__main__":
    main()
