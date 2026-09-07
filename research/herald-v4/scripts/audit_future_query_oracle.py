#!/usr/bin/env python3
"""Independent integrity and metric audit for study 047 future-query oracle."""
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN_PATH = ROOT / "results/future-query-oracle-v1/run.json"
PRIOR_RUN_PATH = ROOT / "results/value-level-evaluation/run.json"
PRIOR_DIR = ROOT / "results/value-level-evaluation"
SUMMARY_PATH = ROOT / "results/future-query-oracle-summary.json"
EXPECTED_SOURCE = "09364027b0fdbb2aff4a74d84e45ea6d90ee4b201a7291e3667715c5a931c2b4"
EXPECTED_IDS = {f"value-level-v1-evaluation-{i:03d}" for i in range(64)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_average(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    x, y = rank_average(a), rank_average(b)
    return float(np.corrcoef(x, y)[0, 1])


def independent_metrics(oracle: list[list[float]], pending: list[float], losses: list[list[float]]) -> dict[str, Any]:
    oracle_array = np.asarray(oracle, dtype=float)
    pending_array = np.asarray(pending, dtype=float)
    loss_array = np.asarray(losses, dtype=float)
    prompt_oracle = oracle_array.mean(axis=1)
    task_losses = loss_array.mean(axis=1)
    prompt_scores: list[float] = []
    eligible = 0
    tied_oracle = 0
    mixed = np.ptp(loss_array, axis=1) != 0
    for row, use in enumerate(mixed):
        if not use:
            continue
        row_score: list[float] = []
        for i in range(4):
            for j in range(i + 1, 4):
                truth = loss_array[row, i] - loss_array[row, j]
                if truth == 0:
                    continue
                eligible += 1
                guessed = oracle_array[row, i] - oracle_array[row, j]
                if guessed == 0:
                    tied_oracle += 1
                    row_score.append(0.5)
                else:
                    row_score.append(float((truth * guessed) > 0))
        prompt_scores.append(float(np.mean(row_score)))
    within = float(np.mean(prompt_scores)) if prompt_scores else None
    return {
        "n": int(loss_array.shape[0]),
        "mixed_prompts": int(mixed.sum()),
        "within_prompt_concordance": within,
        "prompt_spearman": spearman(prompt_oracle.tolist(), task_losses.tolist()),
        "pending_query_spearman": spearman(pending_array.tolist(), task_losses.tolist()),
        "eligible_value_pairs": eligible,
        "tied_oracle_value_pairs": tied_oracle,
        "prompt_oracle": prompt_oracle.tolist(),
        "task_losses": task_losses.tolist(),
        "oracle": oracle_array.tolist(),
        "pending": pending_array.tolist(),
    }


def main() -> int:
    run = json.loads(RUN_PATH.read_text())
    prior_run = json.loads(PRIOR_RUN_PATH.read_text())
    summary = json.loads(SUMMARY_PATH.read_text())
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    checks["run_completed_no_failures"] = run.get("status") == "completed" and not run.get("failures")
    checks["run_has_exact_64_completed_prompts"] = len(run.get("prompts", [])) == 64 and {x.get("id") for x in run["prompts"]} == EXPECTED_IDS and all(x.get("status") == "completed" for x in run["prompts"])
    checks["prior_run_completed_no_failures"] = prior_run.get("status") == "completed" and not prior_run.get("failures")
    checks["prior_run_manifest_hash_exact"] = run.get("manifest_sha256") == prior_run.get("manifest_sha256")
    checks["prior_run_sha_exact"] = run.get("prior_run_sha256") == sha(PRIOR_RUN_PATH)
    checks["runtime_identity_exact"] = run.get("runtime") == prior_run.get("runtime")
    details["runtime"] = run.get("runtime")

    source_hashes = run.get("source_hashes", {})
    source_by_name = {Path(path).name: digest for path, digest in source_hashes.items()}
    expected_shared = {
        "engine.py": "4071a4fcdb2bae4d3d7f3f05b1560d54dd2f2143d675e4e2e79a912ae67dbb52",
        "run_pair_pilot.py": "27045a21b32139ab870bcfb379a6c62f66b28242f1acc302274a8fb42b0d7941",
        "value_group_adapter.py": "6b2e329f1ed51c7878c0ad58d3fbbcbf2797e3445c07b6c2f0f2a256612bfd43",
        "measure_future_query_oracle.py": EXPECTED_SOURCE,
    }
    checks["source_hashes_exact"] = all(source_by_name.get(name) == digest for name, digest in expected_shared.items())
    local_source_paths = {
        "measure_future_query_oracle.py": ROOT / "scripts/measure_future_query_oracle.py",
        "run_pair_pilot.py": ROOT / "scripts/run_pair_pilot.py",
        "value_group_adapter.py": ROOT / "scripts/value_group_adapter.py",
        "engine.py": ROOT.parent / "herald-v3/src/herald_v3/engineering/engine.py",
    }
    checks["local_source_hashes_match_run"] = all(path.is_file() and sha(path) == source_by_name.get(name) for name, path in local_source_paths.items())
    details["source_hashes"] = source_by_name

    prior_records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in PRIOR_DIR.glob("*.json"):
        if path.name == "run.json" or ".features." in path.name:
            continue
        record = json.loads(path.read_text())
        rid = record.get("manifest_row", {}).get("id")
        if rid:
            prior_records[rid] = (path, record)
    checks["prior_population_exact_64"] = set(prior_records) == EXPECTED_IDS

    current_records: dict[str, tuple[Path, dict[str, Any]]] = {}
    for item in run.get("prompts", []):
        path = RUN_PATH.parent / item["path"]
        record = json.loads(path.read_text())
        current_records[item["id"]] = (path, record)
    checks["current_population_exact_64"] = set(current_records) == EXPECTED_IDS

    measurement_checks: list[bool] = []
    prompt_identity_checks: list[bool] = []
    physical_checks: list[bool] = []
    shape_checks: list[bool] = []
    hash_checks: list[bool] = []
    for rid in sorted(EXPECTED_IDS):
        path, record = current_records[rid]
        prior_path, prior = prior_records[rid]
        record_checks = record.get("checks", {})
        measurement_checks.append(record.get("status") == "completed" and bool(record_checks) and all(bool(v) for k, v in record_checks.items() if k != "prior_path_sha256"))
        prompt_identity_checks.append(record.get("manifest_row") == prior.get("manifest_row") and record.get("prompt_token_ids") == prior.get("prompt_token_ids"))
        ref = record.get("branches", {}).get("reference", {})
        act = record.get("branches", {}).get("action", {})
        old_ref = prior.get("branches", {}).get("reference", {})
        old_act = prior.get("branches", {}).get("action", {})
        prompt_identity_checks.append(ref.get("continuation", {}).get("token_ids") == old_ref.get("continuation", {}).get("token_ids"))
        prompt_identity_checks.append(act.get("continuation", {}).get("token_ids") == old_act.get("continuation", {}).get("token_ids"))
        prompt_identity_checks.append(ref.get("continuation", {}).get("termination_reason") == old_ref.get("continuation", {}).get("termination_reason"))
        prompt_identity_checks.append(act.get("continuation", {}).get("termination_reason") == old_act.get("continuation", {}).get("termination_reason"))
        prompt_identity_checks.append(record_checks.get("prior_path_sha256") == sha(prior_path))

        compression = record.get("observation", {}).get("native_mask", {})
        lengths_before = compression.get("before_lengths", [])
        lengths_after = compression.get("after_lengths", [])
        expected_after = compression.get("expected_after_lengths", [])
        physical_checks.append(
            len(lengths_before) == 28 and len(lengths_after) == 28 and len(expected_after) == 28
            and len(set(lengths_before)) == 1 and len(set(lengths_after)) == 1
            and lengths_after == expected_after and compression.get("before_bytes") == lengths_before[0] * 28 * 4 * 2 * 128 * 2
            and compression.get("after_bytes") == lengths_after[0] * 28 * 4 * 2 * 128 * 2
            and compression.get("physical_effect_exact") is True
            and compression.get("kept_index_hash") == act.get("compression", {}).get("kept_index_hash")
        )
        masks = np.load(path.with_name(path.stem + ".masks.npz"))["kept"]
        obs = record.get("observation", {})
        recon = obs.get("reconstruction", {})
        vals = obs.get("adapter", {}).get("values", [])
        shape_checks.append(
            masks.shape == (28, 4, len(compression.get("after_lengths", [])) and compression["after_lengths"][0]) and np.issubdtype(masks.dtype, np.integer)
            and len(vals) == 4 and all(len(v.get("query_positions", [])) == 7 for v in vals)
            and recon.get("query_count") == 29 and recon.get("gqa_groups") == 7
            and recon.get("finite") is True and float(recon.get("max_relative", 1.0)) <= 0.005
            and len(obs.get("per_value_oracle", [])) == 4 and np.isfinite(obs["per_value_oracle"]).all()
            and np.isfinite(float(obs.get("pending_query_error")))
        )
        feature_path = path.with_name(path.stem + ".features.json")
        hash_checks.append(sha(feature_path) == record.get("feature_sha256") and sha(path.with_name(path.stem + ".masks.npz")) == record.get("masks_sha256"))

    checks["all_record_measurement_checks_true"] = all(measurement_checks)
    checks["all_prompt_reference_action_identity_checks_true"] = all(prompt_identity_checks)
    checks["all_physical_mask_checks_true"] = all(physical_checks)
    checks["all_shape_finite_reconstruction_checks_true"] = all(shape_checks)
    checks["all_feature_and_mask_hashes_true"] = all(hash_checks)

    oracle = [current_records[rid][1]["observation"]["per_value_oracle"] for rid in sorted(EXPECTED_IDS)]
    pending = [current_records[rid][1]["observation"]["pending_query_error"] for rid in sorted(EXPECTED_IDS)]
    losses = [prior_records[rid][1]["per_value_signed_losses"] for rid in sorted(EXPECTED_IDS)]
    metrics = independent_metrics(oracle, pending, losses)
    metric_checks = {
        "metrics_n_exact": metrics["n"] == summary["n"] == 64,
        "metrics_mixed_exact": metrics["mixed_prompts"] == summary["mixed_prompts"] == 42,
        "concordance_reproduced": abs(metrics["within_prompt_concordance"] - summary["within_prompt_concordance"]) < 1e-12,
        "prompt_spearman_reproduced": abs(metrics["prompt_spearman"] - summary["prompt_spearman"]) < 1e-12,
        "pending_spearman_reproduced": abs(metrics["pending_query_spearman"] - summary["pending_query_spearman"]) < 1e-12,
        "frozen_concordance_gate_false": metrics["within_prompt_concordance"] < 0.81 and summary["concordance_gate"] is False,
        "frozen_spearman_gate_false": metrics["prompt_spearman"] < 0.70 and summary["spearman_gate"] is False,
        "frozen_overall_gate_false": summary["passes"] is False,
    }
    checks.update(metric_checks)
    details["independent_metrics"] = metrics
    details["summary_metrics"] = {key: summary.get(key) for key in ("n", "mixed_prompts", "within_prompt_concordance", "prompt_spearman", "pending_query_spearman", "eligible_value_pairs", "tied_oracle_value_pairs", "concordance_gate", "spearman_gate", "passes")}
    result = {
        "schema_version": "future-query-oracle.independent-audit.v1",
        "source_sha256": sha(Path(__file__)),
        "run_sha256": sha(RUN_PATH),
        "prior_run_sha256": sha(PRIOR_RUN_PATH),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "integrity_checks_pass": all(checks[key] for key in checks if "gate" not in key and key != "frozen_overall_gate_false"),
        "scientific_gate_pass": summary["passes"],
        "details": details,
    }
    out = ROOT / "results/future-query-oracle-independent-audit.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"all_checks_pass": result["all_checks_pass"], "integrity_checks_pass": result["integrity_checks_pass"], "scientific_gate_pass": result["scientific_gate_pass"], "failed_checks": [key for key, value in checks.items() if not value]}, sort_keys=True))
    return 0 if result["integrity_checks_pass"] and not result["scientific_gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
