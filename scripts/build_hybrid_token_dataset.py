"""Build token-level compressed-stream predictor data."""

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from herald.hybrid_token_dataset import build_hybrid_token_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_dir",
        type=Path,
        nargs="?",
        default=Path("results/sweep"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/predictor/hybrid_tokens.parquet"),
    )
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--max-hybrids-per-task", type=int, default=None)
    args = ap.parse_args()

    rows, summary = build_hybrid_token_dataset(
        args.results_dir,
        models=args.models,
        tasks=args.tasks,
        max_hybrids_per_task=args.max_hybrids_per_task,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.out)  # type: ignore[no-untyped-call]
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary["output_path"] = str(args.out)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
