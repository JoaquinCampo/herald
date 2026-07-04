"""Render a self-contained HTML report for switch-level EDA."""

import argparse
import json
from pathlib import Path

from herald.switch_visual_report import render_visual_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("results/predictor/eda/eda_summary.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/predictor/eda/visual_report.html"),
    )
    args = ap.parse_args()

    summary = json.loads(args.summary.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_visual_report(summary))
    print(f"visual report -> {args.out}")


if __name__ == "__main__":
    main()
