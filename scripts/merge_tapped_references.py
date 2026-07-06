"""Merge tap-widened references with the legacy hybrid grid.

For each model/task, takes references from a tapped regeneration
run (validated token-identical to the legacy greedy continuations)
and symlinks the legacy hybrids next to them, producing a merged
results tree the switch-dataset builder can consume directly.
"""

import argparse
import json
from pathlib import Path

from herald.switch_dataset import merge_tapped_references


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old_dir", type=Path)
    ap.add_argument("new_dir", type=Path)
    ap.add_argument("merged_dir", type=Path)
    ap.add_argument("--models", nargs="+", default=["llama"])
    ap.add_argument(
        "--tasks", nargs="+", default=["gsm8k", "humaneval", "ifeval"]
    )
    args = ap.parse_args()

    reports: dict[str, dict] = {}
    for model in args.models:
        for task in args.tasks:
            old_task = args.old_dir / model / task
            new_task = args.new_dir / model / task
            if not old_task.is_dir() or not new_task.is_dir():
                print(f"skip {model}/{task}: missing input dir")
                continue
            report = merge_tapped_references(
                old_task, new_task, args.merged_dir / model / task
            )
            reports[f"{model}/{task}"] = report
            print(
                f"{model}/{task}: matched={report['matched']} "
                f"prefix_salvaged={report['prefix_salvaged']} "
                f"dropped={report['dropped']} "
                f"missing_in_new={report['missing_in_new']}"
            )
            if report["salvaged_ids"]:
                print(f"  salvaged: {report['salvaged_ids'][:10]}")

    out = args.merged_dir / "merge_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=2))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
