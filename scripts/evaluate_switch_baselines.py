"""Evaluate train-only baselines for switch-level damage prediction."""

import argparse
import json
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from herald.switch_baselines import baseline_report, evaluate_baselines


def main() -> None:
    """Run leave-one-compressor-out baseline evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("results/predictor/switch_dataset.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/baselines"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-group-fraction", type=float, default=0.25)
    parser.add_argument("--position-bucket-size", type=int, default=16)
    parser.add_argument(
        "--compressor",
        action="append",
        dest="compressors",
        help="Restrict to a compressor. May be passed multiple times.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    summary = evaluate_baselines(
        rows,
        compressors=args.compressors,
        seed=args.seed,
        test_group_fraction=args.test_group_fraction,
        position_bucket_size=args.position_bucket_size,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "baseline_summary.json"
    md_path = args.out_dir / "baseline_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(baseline_report(summary))

    print(f"rows={summary['n_rows']}")
    print(f"splits={len(summary['splits'])}")
    print(f"summary -> {json_path}")
    print(f"report -> {md_path}")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load parquet rows without requiring pyarrow type stubs."""
    parquet = import_module("pyarrow.parquet")
    read_table = cast(Any, parquet.read_table)
    table = read_table(path)
    to_pylist = cast(Any, table.to_pylist)
    rows = to_pylist()
    if not isinstance(rows, list):
        raise TypeError("expected parquet table rows to be a list")
    return cast(list[dict[str, Any]], rows)


if __name__ == "__main__":
    main()
