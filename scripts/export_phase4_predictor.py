"""Freeze the Phase 2 lr_all_cheap predictor for the Phase 4 controller.

Reads `phase2_tokens.parquet`, fits lr_all_cheap on
`future_sum_js_25` (training-fold p90 binary), and writes a portable
JSON to `models/phase4_lr_all_cheap.json`. CPU-only.

Existing Phase 1/2/4 artifacts are NOT modified — the script only
writes to its `--output` path. Default path is fresh and gitignored.

Example:
    uv run python scripts/export_phase4_predictor.py \\
        --dataset results/phase2/dataset/phase2_tokens.parquet \\
        --output models/phase4_lr_all_cheap.json
"""

import argparse
import sys
from pathlib import Path

import polars as pl

from herald.phase4_predictor import (
    DEFAULT_HORIZON,
    DEFAULT_LABEL_BASE,
    DEFAULT_QUANTILE,
    DEFAULT_SEED,
    fit_lr_all_cheap,
    save_predictor,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("models/phase4_lr_all_cheap.json"),
    )
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    ap.add_argument("--label-base", default=DEFAULT_LABEL_BASE)
    ap.add_argument("--quantile", type=float, default=DEFAULT_QUANTILE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.dataset.exists():
        print(f"dataset not found: {args.dataset}", file=sys.stderr)
        return 2

    print(f"loading {args.dataset}...")
    df = pl.read_parquet(args.dataset)
    print(f"  rows={df.height} cols={len(df.columns)}")

    state = fit_lr_all_cheap(
        df,
        horizon=args.horizon,
        label_base=args.label_base,
        quantile=args.quantile,
        seed=args.seed,
    )
    save_predictor(state, args.output)
    print(
        f"wrote {args.output} | "
        f"label={state['label']['column']} "
        f"threshold={state['label']['threshold']:.4f} "
        f"n_pos/n_neg={state['training']['n_pos']}/"
        f"{state['training']['n_neg']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
