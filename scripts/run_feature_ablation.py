"""Feature-class ablation for HERALD v1.

Trains the same HGB regressor with feature subsets to attribute
per-token Spearman to feature classes:

  - tier0_only: 9 Tier-0 token-level features only.
  - tier0_plus_rolling: + rolling/EWMA features.
  - tier0_plus_position: Tier 0 + token_pos + relative_progress.
  - tier0_plus_ratio: Tier 0 + compression_ratio.
  - position_only: only token_pos + relative_progress.
  - ratio_only: only compression_ratio.
  - all: full CHEAP_ALL_FEATURE_ORDER.

Reports per-token Spearman on a single GroupKFold(prompt_id) fold.
This is a fast, comparable attribution; absolute numbers may differ
slightly from the 5-fold OOF results.
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

from herald.predictor_baselines import (
    CHEAP_ALL_FEATURES,
    CHEAP_TIER0_FEATURES,
)

ROLLING_FEATURES = tuple(
    c for c in CHEAP_ALL_FEATURES
    if c not in CHEAP_TIER0_FEATURES
    and c not in {"token_pos", "relative_progress", "compression_ratio"}
)
POSITION_FEATURES = ("token_pos", "relative_progress")
RATIO_FEATURES = ("compression_ratio",)


FEATURE_SUBSETS: dict[str, tuple[str, ...]] = {
    "tier0_only": CHEAP_TIER0_FEATURES,
    "tier0_plus_rolling": CHEAP_TIER0_FEATURES + ROLLING_FEATURES,
    "tier0_plus_position": CHEAP_TIER0_FEATURES + POSITION_FEATURES,
    "tier0_plus_ratio": CHEAP_TIER0_FEATURES + RATIO_FEATURES,
    "position_only": POSITION_FEATURES,
    "ratio_only": RATIO_FEATURES,
    "position_plus_ratio": POSITION_FEATURES + RATIO_FEATURES,
    "all_cheap": CHEAP_ALL_FEATURES,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens-path", type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--sample-rows", type=int, default=1_500_000)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/phase3/feature_ablation.json"))
    args = parser.parse_args()

    label = f"future_sum_js_{args.horizon}"
    all_needed = list(dict.fromkeys(list(CHEAP_ALL_FEATURES)))
    keep = (
        ["run_id", "prompt_id", "task", "press", "token_pos",
         "compression_ratio", label]
        + [c for c in all_needed
           if c not in {"token_pos", "compression_ratio"}]
    )
    keep = list(dict.fromkeys(keep))

    print("[load] reading tokens...")
    t0 = time.time()
    full = (
        pl.scan_parquet(args.tokens_path)
        .select(keep)
        .filter(pl.col(label).is_not_null())
        .collect()
    )
    full = full.drop_nulls(subset=list(CHEAP_ALL_FEATURES) + [label])
    print(f"[load] {full.height:,} rows in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    if full.height > args.sample_rows:
        idx = rng.choice(
            full.height, size=args.sample_rows, replace=False)
        df = full[idx]
    else:
        df = full
    print(f"[sample] {df.height:,} rows")

    groups = df["prompt_id"].to_numpy()
    gkf = GroupKFold(n_splits=5)
    fold0 = next(iter(gkf.split(df, df, groups=groups)))
    tr, te = fold0

    y_raw = df[label].to_numpy().astype(np.float64)
    y = np.log1p(y_raw)

    results: list[dict] = []
    for name, feats in FEATURE_SUBSETS.items():
        feats = tuple(dict.fromkeys(feats))
        X = df.select(list(feats)).to_numpy().astype(np.float32)
        t0 = time.time()
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=args.max_iter,
            max_depth=8,
            min_samples_leaf=200,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=args.seed,
        )
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rho = float(spearmanr(pred, y_raw[te]).statistic)
        elapsed = time.time() - t0
        results.append({
            "subset": name,
            "n_features": len(feats),
            "features": list(feats),
            "rho_token": rho,
            "elapsed_seconds": elapsed,
        })
        print(f"[{name:<22s}] {len(feats):>2d} feats  "
              f"ρ={rho:.4f}  ({elapsed:.1f}s)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[done] wrote {args.out}")


if __name__ == "__main__":
    main()
