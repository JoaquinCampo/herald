# pyright: reportMissingImports=false

"""Build the v2 switch-level predictor dataset.

Reads ``results/sweep`` by default and writes:

- ``results/predictor/switch_dataset.parquet``
- ``results/predictor/switch_dataset_summary.json``
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from herald.sweep_provenance import bind_table_to_sweep_config
from herald.switch_dataset import build_switch_dataset, rows_to_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_dir", type=Path, nargs="?", default=Path("results/sweep")
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/predictor/switch_dataset.parquet"),
    )
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--max-rows-per-task", type=int, default=None)
    ap.add_argument("--sweep-config", type=Path, default=None)
    args = ap.parse_args()
    sweep_config = args.sweep_config or args.results_dir / "config.json"

    rows, summary = build_switch_dataset(
        args.results_dir,
        models=args.models,
        tasks=args.tasks,
        max_rows_per_task=args.max_rows_per_task,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = bind_table_to_sweep_config(rows_to_table(rows), sweep_config)
    pq.write_table(table, args.out)  # type: ignore[no-untyped-call]
    summary_path = args.out.with_name(args.out.stem + "_summary.json")
    summary["output_path"] = str(args.out)
    summary["sweep_config"] = str(sweep_config)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
