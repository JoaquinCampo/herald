"""Evaluate live runs against Herald's executable north-star contract."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from herald.deployment_contract import (  # noqa: E402
    DeploymentContract,
    evaluate_deployment,
    measurement_from_live_records,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-dir", default="results/live_controller")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quality-margin", type=float, default=0.01)
    parser.add_argument("--max-slowdown", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=30)
    parser.add_argument("--bootstrap-resamples", type=int, default=2_000)
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    live_dir = Path(args.live_dir)
    output = (
        Path(args.output)
        if args.output is not None
        else live_dir / "deployment_report.json"
    )
    contract = DeploymentContract(
        quality_noninferiority_margin=args.quality_margin,
        max_end_to_end_slowdown=args.max_slowdown,
        min_pairs=args.min_pairs,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    baselines = {
        str(row["prompt_id"]): row
        for row in load_jsonl(live_dir / "baseline.jsonl")
    }
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for episode in load_jsonl(live_dir / "episodes.jsonl"):
        key = (str(episode["compressor"]), float(episode["ratio"]))
        groups[key].append(episode)

    reports: dict[str, dict[str, Any]] = {}
    feasible: list[tuple[str, float]] = []
    for (compressor, ratio), episodes in sorted(groups.items()):
        measurements = [
            measurement_from_live_records(ep, baselines[str(ep["prompt_id"])])
            for ep in episodes
        ]
        evaluation = evaluate_deployment(measurements, contract=contract)
        name = f"{compressor}|{ratio:.4f}"
        reports[name] = evaluation.as_dict()
        if evaluation.feasible:
            if evaluation.peak_kv_savings is None:
                raise AssertionError("feasible report lacks KV savings")
            feasible.append((name, evaluation.peak_kv_savings.lower))
        memory = (
            "unverified"
            if evaluation.peak_kv_savings is None
            else f"{evaluation.peak_kv_savings.mean:.1%}"
        )
        print(
            f"{name}: feasible={evaluation.feasible} "
            f"quality_upper={evaluation.quality_damage.upper:.3%} "
            f"slowdown_upper={evaluation.end_to_end_slowdown.upper:.1%} "
            f"peak_kv_savings={memory} "
            f"failures={','.join(evaluation.failures) or 'none'}"
        )

    feasible.sort(key=lambda item: item[1], reverse=True)
    result = {
        "schema_version": 1,
        "contract": {
            "quality_noninferiority_margin": (
                contract.quality_noninferiority_margin
            ),
            "max_end_to_end_slowdown": contract.max_end_to_end_slowdown,
            "confidence": contract.confidence,
            "min_pairs": contract.min_pairs,
            "bootstrap_resamples": contract.bootstrap_resamples,
        },
        "overall_pass": bool(feasible),
        "best_feasible": feasible[0][0] if feasible else None,
        "groups": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    if not feasible and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
