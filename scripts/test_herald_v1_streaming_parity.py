"""Parity test: StreamingHeraldRegressor vs batched OOF prediction.

For a held-out run, replay tokens through `StreamingHeraldRegressor`
and assert that the per-token predictions match the batched OOF
predictions from the same trained model within tolerance.

Usage:
    uv run python scripts/test_herald_v1_streaming_parity.py \
        --tokens-path results/phase2/dataset/phase2_tokens.parquet \
        --model-path models/herald_v1_h25.pkl \
        --n-runs 3
"""

import argparse
import math
import pickle
from pathlib import Path

import numpy as np
import polars as pl

from herald.config import TokenSignals
from herald.herald_v1_streaming import StreamingHeraldRegressor
from herald.phase4_online_features import CHEAP_ALL_FEATURE_ORDER

# Token-level fields needed to reconstruct TokenSignals.
SIGNAL_FIELDS = (
    "entropy", "top1_prob", "top5_prob", "h_alts",
    "avg_logp", "delta_h", "delta_h_valid", "kl_div",
    "top10_jaccard", "eff_vocab_size", "tail_mass",
    "logit_range", "js_full", "kl_unc_comp_full",
)


def _to_signal(row: dict) -> TokenSignals:
    kw: dict = {}
    for f in SIGNAL_FIELDS:
        v = row.get(f)
        if f == "delta_h_valid":
            kw[f] = bool(v) if v is not None else False
            continue
        if v is None or (isinstance(v, float) and math.isnan(v)):
            kw[f] = float("nan")
        else:
            kw[f] = float(v)
    return TokenSignals(**kw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-path", type=Path,
                        default=Path(
                            "results/phase2/dataset/phase2_tokens.parquet"))
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=5e-4)
    args = parser.parse_args()

    with open(args.model_path, "rb") as fp:
        model = pickle.load(fp)

    needed_cols = [
        "run_id", "token_pos", "press", "compression_ratio",
        *SIGNAL_FIELDS,
        *[c for c in CHEAP_ALL_FEATURE_ORDER
          if c not in {"token_pos", "compression_ratio"}],
    ]
    needed_cols = list(dict.fromkeys(needed_cols))

    df = pl.scan_parquet(args.tokens_path).select(needed_cols).collect()
    runs = df["run_id"].unique().to_list()
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(
        np.asarray(runs), size=min(args.n_runs, len(runs)),
        replace=False,
    )

    failures = 0
    for run_id in sample:
        sub = (df.filter(pl.col("run_id") == run_id)
                 .sort("token_pos"))
        press = sub["press"][0]
        ratio = float(sub["compression_ratio"][0])

        streamer = StreamingHeraldRegressor(
            model=model, press=press,
            compression_ratio=ratio,
            max_new_tokens=args.max_new_tokens,
        )

        streamed_preds = np.empty(sub.height, dtype=np.float64)
        for i, row in enumerate(sub.iter_rows(named=True)):
            sig = _to_signal(row)
            streamed_preds[i] = streamer.step(sig)

        feats = list(CHEAP_ALL_FEATURE_ORDER)
        X = sub.select(feats).to_numpy().astype(np.float32)
        batched_log1p = model.predict(X)
        batched_pred = np.expm1(batched_log1p)

        diff = np.abs(streamed_preds - batched_pred)
        max_diff = float(np.nanmax(diff))
        ok = max_diff < args.tol
        print(f"run={run_id} n={sub.height:>5d} "
              f"max_abs_diff={max_diff:.6f} {'OK' if ok else 'FAIL'}")
        if not ok:
            failures += 1
            worst = int(np.nanargmax(diff))
            print(f"  worst at token_pos={worst}: "
                  f"stream={streamed_preds[worst]:.4f} "
                  f"batch={batched_pred[worst]:.4f}")

    if failures:
        raise SystemExit(
            f"{failures} of {len(sample)} runs failed parity")
    print(f"\n[parity OK] {len(sample)} runs all under "
          f"tol={args.tol}")


if __name__ == "__main__":
    main()
