import json
import subprocess
import sys
from pathlib import Path

from herald.deployment_evidence import (
    END_TO_END_RETAINED_KV_CACHE,
    candidate_id,
    initialize_live_run,
)

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_deployment.py"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _baseline(run_id: str) -> dict[str, object]:
    return {
        "prompt_id": "p0",
        "run_id": run_id,
        "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
        "q_ref_live": 1.0,
        "wall_s": 10.0,
        "ref_len": 100,
        "peak_kv_cache_bytes": 1_000,
    }


def _episode(
    run_id: str,
    compressor: str,
    *,
    candidate_wall_s: float,
) -> dict[str, object]:
    current_candidate_id = candidate_id(compressor, 0.25, None)
    return {
        "key": f"{current_candidate_id}|p0",
        "prompt_id": "p0",
        "compressor": compressor,
        "ratio": 0.25,
        "sustain_interval": None,
        "candidate_id": current_candidate_id,
        "run_id": run_id,
        "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
        "commit_s": 20,
        "n_new_ids": 80,
        "q_live": 1.0,
        "total_wall_s": candidate_wall_s,
        "peak_kv_cache_bytes": 500,
    }


def _evaluate(
    live_dir: Path,
    target_compressor: str,
) -> dict[str, object]:
    output = live_dir / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--live-dir",
            str(live_dir),
            "--output",
            str(output),
            "--target-compressor",
            target_compressor,
            "--target-ratio",
            "0.25",
            "--min-pairs",
            "1",
            "--bootstrap-resamples",
            "10",
            "--report-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "wrote" in result.stdout
    report = json.loads(output.read_text())
    if not isinstance(report, dict):
        raise AssertionError("deployment evaluator did not write an object")
    return report


def test_deployment_report_is_specific_to_the_requested_candidate(
    tmp_path: Path,
) -> None:
    live_dir = tmp_path / "live"
    expected = candidate_id("expected_attention_stats", 0.25, None)
    knorm = candidate_id("knorm", 0.25, None)
    manifest = initialize_live_run(
        live_dir,
        {
            "task": "ifeval",
            "prompt_ids": ["p0"],
            "candidate_ids": [expected, knorm],
            "candidate_prompt_ids": {
                expected: ["p0"],
                knorm: ["p0"],
            },
        },
        resume=False,
    )
    run_id = str(manifest["run_id"])
    _write_jsonl(live_dir / "baseline.jsonl", [_baseline(run_id)])
    _write_jsonl(
        live_dir / "episodes.jsonl",
        [
            _episode(
                run_id,
                "expected_attention_stats",
                candidate_wall_s=11.0,
            ),
            _episode(run_id, "knorm", candidate_wall_s=10.0),
        ],
    )

    report = _evaluate(live_dir, "expected_attention_stats")

    assert report["target_candidate"] == expected
    assert report["overall_pass"] is False
    assert report["best_feasible"] is None
    groups = report["groups"]
    assert isinstance(groups, dict)
    knorm_report = groups[knorm]
    assert isinstance(knorm_report, dict)
    assert knorm_report["feasible"] is True
