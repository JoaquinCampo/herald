"""Rebuild a unified phase1_sweep_summary.json from per-cell summaries.

The Phase 1 sweep wrote a top-level `phase1_sweep_summary.json` with
every cell's outcome. The longbench rerun (43 cells) overwrote that
file; the per-cell `cell_summary.json` files for all 172 cells survived
intact under `<root>/<task>/<press>/ratio=<r:.4f>/cell_summary.json`.

This script walks those 172 files and rebuilds a unified outcomes list.
Run config fields (model, seed, num_prompts, etc.) are inherited from a
reference summary file (the preserved longbench-rerun copy by default).

Usage:
    python scripts/rebuild_phase1_summary.py \\
        --root results/phase1 \\
        --reference \\
            results/phase1/phase1_sweep_summary.longbench-rerun.json \\
        --output results/phase1/phase1_sweep_summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_cell_summary(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cell_files = sorted(args.root.glob("*/*/ratio=*/cell_summary.json"))
    if not cell_files:
        print(
            f"No cell_summary.json found under {args.root}",
            file=sys.stderr,
        )
        return 1
    print(f"Found {len(cell_files)} cell summaries under {args.root}")

    outcomes: list[dict[str, Any]] = []
    for f in cell_files:
        outcomes.append(load_cell_summary(f))

    # Sort: task -> press -> ratio. baseline (none@0) first within task.
    def sort_key(o: dict[str, Any]) -> tuple[str, int, str, float]:
        is_baseline = 0 if o["press"] == "none" else 1
        return (
            str(o["task"]),
            is_baseline,
            str(o["press"]),
            float(o["compression_ratio"]),
        )

    outcomes.sort(key=sort_key)

    ref = json.loads(args.reference.read_text())
    unified: dict[str, Any] = {
        "model": ref.get("model"),
        "max_new_tokens": ref.get("max_new_tokens"),
        "num_prompts": ref.get("num_prompts"),
        "seed": ref.get("seed"),
        "longbench_subtask": ref.get("longbench_subtask"),
        # total_wall_clock_seconds is the sum of per-cell wall clock; not
        # the same as the original launcher's wall clock, but the per-
        # cell time is what's preserved on disk.
        "total_wall_clock_seconds": sum(
            float(o.get("cell_wall_clock_seconds", 0.0)) for o in outcomes
        ),
        "n_cells_completed": len(outcomes),
        "n_cells_planned": len(outcomes),
        "abort_reason": None,
        "rebuilt_from_cell_summaries": True,
        "outcomes": outcomes,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(unified, indent=2))
    print(f"Wrote unified summary -> {args.output}")

    by_task: dict[str, dict[str, int]] = {}
    for o in outcomes:
        t = str(o["task"])
        d = by_task.setdefault(
            t,
            {"n_cells": 0, "n_ok": 0, "n_failed": 0, "n_skipped": 0,
             "n_attempted": 0},
        )
        d["n_cells"] += 1
        d["n_ok"] += int(o.get("n_ok", 0))
        d["n_failed"] += int(o.get("n_failed", 0))
        d["n_skipped"] += int(o.get("n_skipped", 0))
        d["n_attempted"] += int(o.get("n_attempted", 0))

    print()
    print(f"{'task':<20} {'cells':>6} {'attempted':>10} {'ok':>8} "
          f"{'failed':>8} {'skipped':>8}")
    print("-" * 64)
    grand: dict[str, int] = {
        "n_cells": 0, "n_attempted": 0, "n_ok": 0, "n_failed": 0,
        "n_skipped": 0,
    }
    for t in sorted(by_task):
        d = by_task[t]
        for k in grand:
            grand[k] += d[k]
        print(f"{t:<20} {d['n_cells']:>6} {d['n_attempted']:>10} "
              f"{d['n_ok']:>8} {d['n_failed']:>8} {d['n_skipped']:>8}")
    print("-" * 64)
    print(f"{'TOTAL':<20} {grand['n_cells']:>6} {grand['n_attempted']:>10} "
          f"{grand['n_ok']:>8} {grand['n_failed']:>8} "
          f"{grand['n_skipped']:>8}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
