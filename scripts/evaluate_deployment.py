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
from herald.deployment_evidence import (  # noqa: E402
    candidate_id,
    load_live_run_manifest,
    verify_live_run,
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
    parser.add_argument("--target-compressor", required=True)
    parser.add_argument("--target-ratio", type=float, required=True)
    parser.add_argument("--target-sustain-interval", type=int, default=None)
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
    manifest = load_live_run_manifest(live_dir / "run_manifest.json")
    baseline_rows = load_jsonl(live_dir / "baseline.jsonl")
    episode_rows = load_jsonl(live_dir / "episodes.jsonl")
    evidence_errors = verify_live_run(
        manifest,
        baseline_rows,
        episode_rows,
        require_complete=True,
    )
    if evidence_errors:
        raise RuntimeError(
            "invalid live evidence: " + "; ".join(evidence_errors)
        )
    baselines = {str(row["prompt_id"]): row for row in baseline_rows}
    target_id = candidate_id(
        args.target_compressor,
        args.target_ratio,
        args.target_sustain_interval,
    )
    run_config = manifest["run_config"]
    if not isinstance(run_config, dict):
        raise RuntimeError("live evidence manifest has invalid run_config")
    candidate_ids = run_config.get("candidate_ids")
    if not isinstance(candidate_ids, list) or target_id not in candidate_ids:
        raise RuntimeError(
            f"target candidate is not bound by run manifest: {target_id}"
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episode_rows:
        groups[str(episode["candidate_id"])].append(episode)
    if target_id not in groups:
        raise RuntimeError(f"target candidate has no episodes: {target_id}")

    reports: dict[str, dict[str, Any]] = {}
    target_evaluation = None
    for name, episodes in sorted(groups.items()):
        measurements = [
            measurement_from_live_records(ep, baselines[str(ep["prompt_id"])])
            for ep in episodes
        ]
        evaluation = evaluate_deployment(measurements, contract=contract)
        reports[name] = evaluation.as_dict()
        if name == target_id:
            target_evaluation = evaluation
        memory = (
            "unverified"
            if evaluation.peak_kv_savings is None
            else f"{evaluation.peak_kv_savings.mean:.1%}"
        )
        print(
            f"{name}: feasible={evaluation.feasible} "
            f"quality_upper={evaluation.quality_damage.upper:.3%} "
            f"major_damage_upper={evaluation.major_damage_rate.upper:.1%} "
            f"slowdown_upper={evaluation.end_to_end_slowdown.upper:.1%} "
            f"peak_kv_savings={memory} "
            f"failures={','.join(evaluation.failures) or 'none'}"
        )
    if target_evaluation is None:
        raise AssertionError("target candidate report was not produced")

    result = {
        "schema_version": 1,
        "contract": {
            "quality_noninferiority_margin": (
                contract.quality_noninferiority_margin
            ),
            "major_damage_threshold": contract.major_damage_threshold,
            "max_major_damage_rate": contract.max_major_damage_rate,
            "max_end_to_end_slowdown": contract.max_end_to_end_slowdown,
            "confidence": contract.confidence,
            "min_pairs": contract.min_pairs,
            "bootstrap_resamples": contract.bootstrap_resamples,
        },
        "target_candidate": target_id,
        "overall_pass": target_evaluation.feasible,
        "best_feasible": target_id if target_evaluation.feasible else None,
        "groups": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}")
    if not target_evaluation.feasible and not args.report_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
