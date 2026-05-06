"""Phase 2 lead-time analysis (Task 5).

Compares the per-token Phase 2 predictor's lead time against
entropy's lead time, both at the same relative-position window
around catastrophic onset.

Method
------
1. Catastrophic runs = compressed runs tagged with `has_looping`
   or `has_non_termination` in run_damage.parquet.
2. Healthy controls = compressed runs with neither tag, matched
   per (task, press, compression_ratio) stratum.
3. Onset proxy:
   - looping: median fractional onset = 0.5 * num_tokens_generated
     (no per-token onset parquet; fractional proxy keeps it CPU-only)
   - non_termination: 0.75 * max_new_tokens (matches
     herald.labeling.DEFAULT_NT_ONSET_FRAC)
4. Controls receive a virtual onset = stratum-median catastrophic
   onset (clipped to control length).
5. For each relative_pos in [-window_before, +window_after]:
   - Compute AUROC(predictor_score, catastrophic vs control)
   - Compute AUROC(entropy, catastrophic vs control)
6. Lead time = earliest rel_pos < 0 where AUROC >= threshold for
   `persistence` consecutive evaluated positions.

Outputs:
- `results/phase2/lead_time/lead_time_by_feature.parquet`
- `results/phase2/lead_time/lead_time_summary.json`
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    binarize_with_threshold,
    compute_train_quantile_threshold,
)


def _clean_X(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.size < 2:
        return None
    if len(set(y.tolist())) < 2:
        return None
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return None


def _load_tokens_for_runs(
    final_dir: Path, runs: pl.DataFrame, columns: list[str]
) -> pl.DataFrame:
    """Read per-run token parquets and concat the requested columns."""
    frames: list[pl.DataFrame] = []
    keep = ["run_id", "token_pos", *columns]
    for row in runs.iter_rows(named=True):
        path = (
            final_dir
            / "tokens"
            / f"press={row['press']}"
            / f"ratio={float(row['compression_ratio']):.4f}"
            / f"{row['run_id']}.parquet"
        )
        if not path.exists():
            continue
        df = pl.read_parquet(path)
        sel = [c for c in keep if c in df.columns]
        frames.append(df.select(sel))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical_relaxed")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("results/phase1"),
        help="Phase 1 root containing final/ and metrics/.",
    )
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/lead_time"),
    )
    ap.add_argument(
        "--label",
        type=str,
        default="future_sum_js_25",
        help="Continuous label used to derive the binary thresholding.",
    )
    ap.add_argument("--threshold-q", type=float, default=0.9)
    ap.add_argument("--max-train-rows", type=int, default=400_000)
    ap.add_argument("--window-before", type=int, default=200)
    ap.add_argument("--window-after", type=int, default=50)
    ap.add_argument("--auroc-threshold", type=float, default=0.65)
    ap.add_argument("--persistence", type=int, default=5)
    ap.add_argument("--position-stride", type=int, default=2)
    ap.add_argument(
        "--max-controls-per-stratum",
        type=int,
        default=20,
        help="Cap the controls per (task,press,ratio) stratum.",
    )
    ap.add_argument(
        "--max-cats-per-stratum",
        type=int,
        default=20,
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = args.root / "final"
    rd_path = args.root / "metrics" / "run_damage.parquet"
    runs_path = final_dir / "runs.parquet"
    if not runs_path.exists() or not rd_path.exists():
        raise SystemExit("missing runs.parquet or run_damage.parquet")

    logger.info("loading run_damage and runs")
    rd = pl.read_parquet(rd_path)
    runs = pl.read_parquet(runs_path).filter(pl.col("replay_status") == "ok")

    rd_with_meta = rd.join(
        runs.select(
            [
                "run_id",
                "task",
                "press",
                "compression_ratio",
                "num_tokens_generated",
                "max_new_tokens",
            ]
        ),
        on="run_id",
        how="left",
    )

    cats = rd_with_meta.filter(
        pl.col("has_looping") | pl.col("has_non_termination")
    )
    healthy = rd_with_meta.filter(
        ~pl.col("has_looping")
        & ~pl.col("has_non_termination")
        & ~pl.col("has_format_break")
        & ~pl.col("has_drift")
    )
    logger.info("n_catastrophic={} n_healthy={}", cats.height, healthy.height)

    np.random.default_rng(args.seed)
    selected_cat: list[dict[str, Any]] = []
    selected_ctrl: list[dict[str, Any]] = []
    for (task, press, ratio), cat_sub in cats.group_by(
        ["task", "press", "compression_ratio"]
    ):
        # Sample up to N catastrophic runs.
        n_cat = min(args.max_cats_per_stratum, cat_sub.height)
        cat_pick = cat_sub.sample(n=n_cat, seed=args.seed)
        # Find healthy controls in same stratum.
        ctrl_sub = healthy.filter(
            (pl.col("task") == task)
            & (pl.col("press") == press)
            & (pl.col("compression_ratio") == ratio)
        )
        if ctrl_sub.is_empty():
            continue
        n_ctrl = min(args.max_controls_per_stratum, ctrl_sub.height)
        ctrl_pick = ctrl_sub.sample(n=n_ctrl, seed=args.seed)

        # Catastrophic onset proxy:
        #   non_termination: 0.75 * max_new_tokens
        #   looping (no nt): 0.5 * num_tokens_generated
        cat_pick = cat_pick.with_columns(
            pl.when(pl.col("has_non_termination"))
            .then((pl.col("max_new_tokens") * 0.75).cast(pl.Int64))
            .otherwise((pl.col("num_tokens_generated") * 0.5).cast(pl.Int64))
            .alias("onset_token")
        )
        # Stratum median onset for controls.
        median_onset = int(cat_pick["onset_token"].median() or 0)
        ctrl_pick = ctrl_pick.with_columns(
            pl.lit(median_onset).alias("stratum_median_onset"),
            pl.min_horizontal(
                pl.lit(median_onset),
                pl.col("num_tokens_generated") - 1,
            ).alias("onset_token"),
        )
        for r in cat_pick.iter_rows(named=True):
            selected_cat.append({**r, "label": 1})
        for r in ctrl_pick.iter_rows(named=True):
            selected_ctrl.append({**r, "label": 0})
        del task, press, ratio  # appease ruff in some configs

    if not selected_cat or not selected_ctrl:
        raise SystemExit("no matched catastrophic / control pairs")
    # infer_schema_length=None scans all rows so polars never trips
    # over a column whose first few rows happen to share a narrow
    # type while later rows widen it (intermittent on large mixed-
    # dict inputs).
    cat_df = pl.DataFrame(selected_cat, infer_schema_length=None)
    ctrl_df = pl.DataFrame(selected_ctrl, infer_schema_length=None)
    logger.info("matched: cats={} controls={}", cat_df.height, ctrl_df.height)

    # Read entropy + token_pos for selected runs.
    _load_tokens_for_runs(final_dir, cat_df, columns=["entropy"])
    _load_tokens_for_runs(final_dir, ctrl_df, columns=["entropy"])

    # Train predictor on a held-out 400k random subsample of the
    # full Phase 2 dataset (runs disjoint from selected cats/ctrls).
    logger.info("loading dataset for predictor training")
    ds = pl.read_parquet(args.dataset)
    selected_run_ids = set(
        cat_df["run_id"].to_list() + ctrl_df["run_id"].to_list()
    )
    train_pool = ds.filter(~pl.col("run_id").is_in(list(selected_run_ids)))
    if args.max_train_rows > 0 and train_pool.height > args.max_train_rows:
        train_pool = train_pool.sample(
            n=args.max_train_rows, seed=args.seed, shuffle=True
        )
    logger.info("predictor train pool: {} rows", train_pool.height)

    feat = [c for c in CHEAP_ALL_FEATURES if c in ds.columns]
    label_col = args.label
    thr = compute_train_quantile_threshold(
        train_pool[label_col], q=args.threshold_q
    )
    train_y = binarize_with_threshold(train_pool[label_col], thr)
    train_mask = ~train_y.is_null()
    keep_train = train_pool.filter(train_mask)
    keep_train_y = train_y.filter(train_mask).cast(pl.Int64).to_numpy()
    train_X = _clean_X(
        keep_train.select(feat).fill_null(0.0).to_numpy().astype(float)
    )
    scaler = StandardScaler()
    Xs = scaler.fit_transform(train_X)
    clf = LogisticRegression(
        max_iter=200, class_weight="balanced", solver="liblinear"
    )
    clf.fit(Xs, keep_train_y)
    logger.info(
        "trained LR on {} rows (pos_rate={:.3f}, thr={:.4f})",
        keep_train.height,
        float((keep_train_y == 1).mean()),
        thr,
    )

    # Score selected cats + ctrls.
    selected_full = ds.filter(pl.col("run_id").is_in(list(selected_run_ids)))
    sel_X = _clean_X(
        selected_full.select(feat).fill_null(0.0).to_numpy().astype(float)
    )
    sel_score = clf.predict_proba(scaler.transform(sel_X))[:, 1]
    selected_full = selected_full.with_columns(
        pl.Series(name="predictor_score", values=sel_score)
    )

    # Build the long lead-time frame for each feature.
    onset_map = pl.concat(
        [
            cat_df.select(["run_id", "onset_token"]).with_columns(
                pl.lit(1).alias("label")
            ),
            ctrl_df.select(["run_id", "onset_token"]).with_columns(
                pl.lit(0).alias("label")
            ),
        ]
    )

    long = selected_full.join(onset_map, on="run_id", how="inner")
    long = long.with_columns(
        (pl.col("token_pos") - pl.col("onset_token")).alias("relative_pos")
    )
    long = long.filter(
        pl.col("relative_pos").is_between(
            -args.window_before, args.window_after
        )
    )
    if args.position_stride > 1:
        long = long.filter(
            (pl.col("relative_pos") % args.position_stride) == 0
        )

    # Per (feature, relative_pos): AUROC of feature value, label.
    rows: list[dict[str, Any]] = []
    for feat_name in ("entropy", "predictor_score"):
        if feat_name not in long.columns:
            continue
        for rp, sub in long.group_by("relative_pos", maintain_order=True):
            y = sub["label"].to_numpy().astype(np.int8)
            s = sub[feat_name].to_numpy().astype(np.float64)
            # entropy: higher = more uncertain = more dangerous (no flip).
            auc = _safe_auroc(y, s)
            rows.append(
                {
                    "feature": feat_name,
                    "relative_pos": int(rp[0]),
                    "n_pos": int((y == 1).sum()),
                    "n_neg": int((y == 0).sum()),
                    "auroc": auc,
                }
            )
    by_feat = pl.DataFrame(rows)
    by_feat.write_parquet(args.output_dir / "lead_time_by_feature.parquet")

    # Lead time = earliest negative rel_pos with persistence consecutive
    # evaluated positions where AUROC >= threshold.
    leads: dict[str, int | None] = {}
    for f in by_feat["feature"].unique().to_list():
        sub = by_feat.filter(pl.col("feature") == f).sort("relative_pos")
        rps = sub["relative_pos"].to_numpy()
        aurocs = sub["auroc"].to_numpy()
        passes = np.array(
            [
                a is not None
                and not np.isnan(a)
                and float(a) >= args.auroc_threshold
                and rps[i] <= 0
                for i, a in enumerate(aurocs)
            ]
        )
        lead: int | None = None
        for i in range(len(rps) - args.persistence + 1):
            if (
                passes[i : i + args.persistence].all()
                and rps[i + args.persistence - 1] <= 0
            ):
                lead = int(-rps[i])
                break
        leads[f] = lead

    summary = {
        "label": args.label,
        "threshold_q": args.threshold_q,
        "n_catastrophic_runs": int(cat_df.height),
        "n_control_runs": int(ctrl_df.height),
        "n_aligned_rows": int(long.height),
        "window_before": args.window_before,
        "window_after": args.window_after,
        "position_stride": args.position_stride,
        "auroc_threshold": args.auroc_threshold,
        "persistence": args.persistence,
        "lead_time_tokens": leads,
        "by_feature_path": str(
            args.output_dir / "lead_time_by_feature.parquet"
        ),
    }
    summary_path = args.output_dir / "lead_time_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary -> {}", summary_path)

    print()
    print("=== Lead-time headline ===")
    print(f"label = {args.label}")
    print(f"n_cat = {cat_df.height}, n_ctrl = {ctrl_df.height}")
    print(f"window = [{-args.window_before}, +{args.window_after}]")
    print(
        f"AUROC threshold = {args.auroc_threshold}, "
        f"persistence = {args.persistence}"
    )
    print()
    print("Lead time (tokens before onset, larger = better):")
    for f, lt in leads.items():
        print(f"  {f:18s}: {lt}")


if __name__ == "__main__":
    main()
