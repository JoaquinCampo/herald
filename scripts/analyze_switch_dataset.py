"""Classical EDA for the v2 switch-level predictor dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from herald.switch_analysis import build_eda_summary, markdown_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("results/predictor/switch_dataset.parquet"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/eda"),
    )
    ap.add_argument("--top-n-features", type=int, default=20)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = pq.read_table(args.dataset).to_pylist()
    summary = build_eda_summary(rows, top_n_features=args.top_n_features)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "eda_summary.json"
    md_path = args.out_dir / "eda_report.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str))
    md_path.write_text(markdown_report(summary))
    print(f"rows={summary['inventory']['n_rows']}")
    print(f"summary -> {json_path}")
    print(f"report -> {md_path}")


if __name__ == "__main__":
    main()
