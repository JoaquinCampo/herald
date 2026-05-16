"""HERALD v1 smoke v2: prompt-stratified, full per-run coverage.

Same regression task as v1, but samples by `prompt_id` and keeps all
tokens for each sampled prompt. This gives unbiased per-run
aggregations (max over a full run, not a thin slice).

Also evaluates H in {5, 10, 25, 50} side-by-side and reports both
log1p and quantile targets.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from herald.predictor_baselines import CHEAP_ALL_FEATURES


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def _train_eval(
    df: pl.DataFrame,
    label_col: str,
    feats: list[str],
    n_folds: int,
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
) -> dict:
    sub = df.drop_nulls(subset=feats + [label_col])
    y_raw = sub[label_col].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)
    X = sub.select(feats).to_numpy().astype(np.float32)
    groups = sub["prompt_id"].to_numpy()
    run_ids = sub["run_id"].to_numpy()
    presses = sub["press"].to_numpy()
    ratios = sub["compression_ratio"].to_numpy()
    tasks = sub["task"].to_numpy()

    print(f"[{label_col}] X={X.shape} prompts={len(np.unique(groups))} "
          f"runs={len(np.unique(run_ids))}")

    oof = np.full(X.shape[0], np.nan)
    gkf = GroupKFold(n_splits=n_folds)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        t0 = time.time()
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed + fold,
        )
        model.fit(X[tr], y[tr])
        oof[te] = model.predict(X[te])
        print(f"  [{label_col} fold {fold}] {time.time()-t0:.1f}s")

    rho_token = _spearman(oof, y_raw)

    per_run = (
        pl.DataFrame(
            {"run_id": run_ids, "pred": oof, "y": y_raw,
             "press": presses, "ratio": ratios, "task": tasks}
        )
        .group_by("run_id")
        .agg(
            pl.col("pred").max().alias("pred_max"),
            pl.col("pred").quantile(0.95).alias("pred_p95"),
            pl.col("pred").mean().alias("pred_mean"),
            pl.col("y").max().alias("y_max"),
            pl.col("y").quantile(0.95).alias("y_p95"),
            pl.col("press").first().alias("press"),
            pl.col("ratio").first().alias("ratio"),
            pl.col("task").first().alias("task"),
        )
    )

    rd = pl.read_parquet(
        "results/phase1/metrics/run_damage.parquet"
    ).select([
        "run_id", "rouge_l_drop", "sum_js", "sum_kl",
        "char_edit_ratio", "embedding_cosine_drop",
    ])
    joined = per_run.join(rd, on="run_id", how="inner")

    out: dict[str, float] = {
        "label": label_col,
        "n_rows": int(sub.height),
        "n_runs": int(joined.height),
        "rho_token_vs_label": rho_token,
    }
    for agg in ("pred_max", "pred_p95", "pred_mean"):
        for tgt in (
            "rouge_l_drop", "sum_js", "sum_kl",
            "char_edit_ratio", "embedding_cosine_drop",
            "y_max",
        ):
            s = joined.drop_nulls(subset=[agg, tgt])
            if s.height < 10:
                out[f"rho_{agg}_vs_{tgt}"] = float("nan")
                continue
            out[f"rho_{agg}_vs_{tgt}"] = _spearman(
                s[agg].to_numpy(), s[tgt].to_numpy()
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[5, 10, 25, 50])
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/phase3_smoke/herald_v1_smoke_v2.json"))
    args = parser.parse_args()

    feats = list(CHEAP_ALL_FEATURES)
    labels = [f"future_sum_js_{h}" for h in args.horizons]
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio"]
        + [c for c in feats if c not in {"token_pos", "compression_ratio"}]
        + labels
    )
    keep = list(dict.fromkeys(keep))

    t0 = time.time()
    full = (
        pl.scan_parquet("results/phase2/dataset/phase2_tokens.parquet")
        .select(keep)
        .collect()
    )
    print(f"[load] {full.height:,} rows × {full.width} cols "
          f"in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    all_prompts = full["prompt_id"].unique().to_numpy()
    sampled = rng.choice(
        all_prompts, size=min(args.n_prompts, len(all_prompts)),
        replace=False,
    )
    df = full.filter(pl.col("prompt_id").is_in(sampled.tolist()))
    print(f"[sample] {len(sampled):,} prompts → {df.height:,} rows")

    results = []
    for h in args.horizons:
        label = f"future_sum_js_{h}"
        out = _train_eval(
            df=df,
            label_col=label,
            feats=feats,
            n_folds=args.n_folds,
            seed=args.seed,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
        )
        out["horizon"] = h
        print(f"\n=== H={h} ===")
        for k, v in out.items():
            if isinstance(v, float):
                print(f"  {k:<40s} = {v:.4f}")
        results.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
