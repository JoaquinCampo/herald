"""HERALD v1 smoke test.

Per-token regression of `future_sum_js_25` on a 500k-row sample
from `results/phase2/dataset/phase2_tokens.parquet`.

Aims to answer one question before scaling up:

  - Per-token Spearman vs `future_sum_js_25` on OOF predictions.
  - Per-run aggregate Spearman vs `rouge_l_drop` and `sum_js`.

If per-run lands in the 0.75-0.80 band on this small subset, the
0.85 headline bar is plausibly reachable with full data + tuning.
If it lands at 0.73 (the prior binary-baseline ceiling), the
substrate or aggregation is the bottleneck, not the model.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from herald.predictor_baselines import CHEAP_ALL_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    parser.add_argument(
        "--run-damage-path",
        type=Path,
        default=Path("results/phase1/metrics/run_damage.parquet"),
    )
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--sample-rows", type=int, default=500_000)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/phase3_smoke/herald_v1_smoke.json"),
    )
    args = parser.parse_args()

    label = f"future_sum_js_{args.horizon}"
    feats = [f for f in CHEAP_ALL_FEATURES]
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos", label]
        + [c for c in feats if c not in {"token_pos", "compression_ratio"}]
        + ["compression_ratio"]
    )
    keep = list(dict.fromkeys(keep))

    print(f"[load] tokens path: {args.tokens_path}")
    t0 = time.time()
    full = (
        pl.scan_parquet(args.tokens_path)
        .select(keep)
        .filter(pl.col(label).is_not_null())
        .collect(streaming=True)
    )
    print(
        f"[load] {full.height:,} non-null rows, {full.width} cols "
        f"in {time.time() - t0:.1f}s"
    )

    rng = np.random.default_rng(args.seed)
    if full.height > args.sample_rows:
        idx = rng.choice(full.height, size=args.sample_rows, replace=False)
        df = full[idx]
    else:
        df = full
    del full
    print(f"[sample] {df.height:,} rows for smoke")

    df = df.drop_nulls(subset=feats + [label])
    print(f"[drop_nulls] {df.height:,} rows after feature-null drop")

    y = np.log1p(df[label].to_numpy().astype(np.float64))
    y_raw = df[label].to_numpy().astype(np.float64)
    X = df.select(feats).to_numpy().astype(np.float32)
    groups = df["prompt_id"].to_numpy()
    run_ids = df["run_id"].to_numpy()

    print(f"[train] features={len(feats)}  X.shape={X.shape}")
    print(f"[train] unique prompts={len(np.unique(groups))}")
    print(f"[train] unique runs={len(np.unique(run_ids))}")

    oof = np.full(X.shape[0], np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=args.n_folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        t0 = time.time()
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=args.seed + fold,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        rho_fold = spearmanr(oof[te], y_raw[te]).statistic
        print(
            f"[fold {fold}] n_tr={len(tr):>7,} n_te={len(te):>7,} "
            f"ρ_token={rho_fold:.4f}  ({time.time() - t0:.1f}s)"
        )

    assert not np.any(np.isnan(oof)), "OOF has NaNs after GroupKFold"

    rho_token = spearmanr(oof, y_raw).statistic
    print(f"\n[token] Spearman(pred, future_sum_js_{args.horizon}) "
          f"= {rho_token:.4f}")

    per_run = (
        pl.DataFrame({"run_id": run_ids, "pred": oof, "y": y_raw})
        .group_by("run_id")
        .agg(
            pl.col("pred").max().alias("pred_max"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("pred").quantile(0.95).alias("pred_p95"),
            pl.col("y").max().alias("y_max"),
            pl.col("y").mean().alias("y_mean"),
        )
    )
    print(f"[per-run] {per_run.height:,} runs in smoke sample")

    rd = pl.read_parquet(args.run_damage_path).select(
        ["run_id", "rouge_l_drop", "sum_js", "sum_kl"]
    )
    joined = per_run.join(rd, on="run_id", how="inner")
    print(f"[per-run] {joined.height:,} runs joined with run_damage")

    def rho(a: str, b: str) -> float:
        sub = joined.drop_nulls(subset=[a, b])
        if sub.height < 10:
            return float("nan")
        return float(
            spearmanr(sub[a].to_numpy(), sub[b].to_numpy()).statistic
        )

    results: dict[str, float] = {
        "n_rows": int(df.height),
        "n_runs_smoke": int(joined.height),
        "horizon": args.horizon,
        "rho_token_vs_label": float(rho_token),
    }
    for agg in ("pred_max", "pred_mean", "pred_p95"):
        for tgt in ("rouge_l_drop", "sum_js", "sum_kl", "y_max"):
            key = f"rho_run_{agg}_vs_{tgt}"
            results[key] = rho(agg, tgt)
            print(f"[per-run] {key:<40s} = {results[key]:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import json
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
