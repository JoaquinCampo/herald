"""Evaluate train-only baselines for switch-level damage prediction."""

import argparse
import json
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from herald.switch_baselines import (
    baseline_report,
    compare_to_baseline_lock,
    evaluate_baselines,
    make_baseline_lock,
)


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
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/implementation/switch_baseline_lock.json"),
    )
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--check-lock", action="store_true")
    parser.add_argument("--lock-tolerance", type=float, default=None)
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
        bootstrap_resamples=args.bootstrap_resamples,
        command=sys.argv,
    )
    if args.write_lock:
        tolerance = (
            1e-9 if args.lock_tolerance is None else args.lock_tolerance
        )
        lock = make_baseline_lock(summary, tolerance=tolerance)
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(lock, indent=2, sort_keys=True))

    if args.check_lock:
        lock = _load_lock(args.lock)
        comparison = compare_to_baseline_lock(
            summary, lock, tolerance=args.lock_tolerance
        )
        summary["lock_comparison"] = comparison

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "baseline_summary.json"
    md_path = args.out_dir / "baseline_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(baseline_report(summary))

    print(f"rows={summary['n_rows']}")
    print(f"splits={len(summary['splits'])}")
    print(f"summary -> {json_path}")
    print(f"report -> {md_path}")
    if args.write_lock:
        print(f"lock -> {args.lock}")
    if args.check_lock:
        passed = summary["lock_comparison"]["passed"]
        print(f"lock_check={passed}")
        if not passed:
            raise SystemExit(1)


def _load_lock(path: Path) -> dict[str, Any]:
    """Load a baseline lock with a clear CLI error."""
    try:
        text = path.read_text()
        value = json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(
            f"could not load baseline lock {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SystemExit(f"baseline lock must be a JSON object: {path}")
    return value


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
