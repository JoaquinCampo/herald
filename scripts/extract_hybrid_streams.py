# pyright: reportMissingImports=false

"""Extract verified grace-window streams from a completed sweep."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

sys.path.insert(0, "src")

from herald.hybrid_streams import (  # noqa: E402
    extract_hybrid_streams,
    save_hybrid_streams,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parquet-aligned hybrid feature blocks"
    )
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-block-tokens", type=int, default=16)
    parser.add_argument("--task", default="ifeval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        table = pq.read_table(args.parquet)  # type: ignore[no-untyped-call]
        rows_value = table.to_pylist()
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"could not read switch parquet {args.parquet}"
        ) from error
    if not isinstance(rows_value, list):
        raise RuntimeError("parquet reader did not return rows")
    rows = cast(list[dict[str, Any]], rows_value)
    rows = [row for row in rows if row.get("task") == args.task]
    if not rows:
        raise ValueError(f"no rows found for task {args.task!r}")
    streams = extract_hybrid_streams(
        rows,
        args.results_dir,
        max_block_tokens=args.max_block_tokens,
    )
    save_hybrid_streams(args.out, streams)
    print(
        json.dumps(
            {
                "event": "hybrid_streams_saved",
                "out": str(args.out),
                "n_rows": len(rows),
                "task": args.task,
                "max_block_tokens": args.max_block_tokens,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
