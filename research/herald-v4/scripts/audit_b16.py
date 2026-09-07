#!/usr/bin/env python3
"""Independent arithmetic and artifact audit for the frozen B16 slice."""

import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/ruler-ea-dev-v1/manifest.json"
FIRST = ROOT / "results/b16-v1-first"
REST = ROOT / "results/b16-v1-rest"
ANALYSIS = ROOT / "results/b16-model/summary.json"
OUTPUT = ROOT / "results/b16-owner-audit.json"
EXPECTED_IDS = {
    f"ruler-ea-dev-v1-niah_single_2-{index:03d}" for index in range(12)
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def official_scorer():
    eval_path = ROOT / "vendor/ruler/scripts/eval/evaluate.py"
    constants_path = ROOT / "vendor/ruler/scripts/eval/synthetic/constants.py"
    tree = ast.parse(eval_path.read_text(encoding="utf-8"), filename=str(eval_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "postprocess_pred"
    )
    namespace = {"re": __import__("re")}
    code = compile(ast.Module(body=[function], type_ignores=[]), str(eval_path), "exec")
    exec(code, namespace)
    spec = importlib.util.spec_from_file_location("ruler_constants_audit", constants_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load official RULER constants")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def score(text, answers):
        processed = namespace["postprocess_pred"](text, {})
        value = module.string_match_all([processed], [answers])
        return {
            "score": float(value),
            "score_fraction": float(value) / 100.0,
            "postprocessed": processed,
            "pass_vector": [answer.lower() in processed.lower() for answer in answers],
        }

    return score


def index_digest(masks):
    digest = hashlib.sha256()
    counts = []
    for layer in masks:
        digest.update(len(layer).to_bytes(4, "little"))
        layer_counts = []
        for head in layer:
            layer_counts.append(len(head))
            digest.update(len(head).to_bytes(8, "little"))
            for index in head:
                digest.update(int(index).to_bytes(8, "little", signed=False))
        counts.append(layer_counts)
    return digest.hexdigest(), counts


def close(a, b, tolerance=2e-6):
    return bool(math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance))


def reconstruct_feature(tensor_path, masks):
    data = torch.load(tensor_path, map_location="cpu", weights_only=True)
    probabilities = data["probabilities"]
    value_norm = data["value_norm"]
    saved_salience = data["salience"]
    if len(probabilities) != len(value_norm) or len(probabilities) != len(saved_salience):
        raise AssertionError("layer counts differ in persisted tensors")
    recomputed = []
    probability_sum_error = 0.0
    for probs, norms, salience in zip(probabilities, value_norm, saved_salience, strict=True):
        if probs.ndim != 2 or norms.ndim != 2 or salience.ndim != 2:
            raise AssertionError("unexpected persisted tensor rank")
        probability_sum_error = max(
            probability_sum_error, float((probs.sum(-1) - 1.0).abs().max())
        )
        kv_heads, length = norms.shape
        query_heads = probs.shape[0]
        if query_heads % kv_heads:
            raise AssertionError("invalid GQA grouping")
        groups = query_heads // kv_heads
        raw = probs * norms.repeat_interleave(groups, dim=0)
        expected = raw.view(kv_heads, groups, length).amax(dim=1)
        expected = expected / expected.sum(-1, keepdim=True)
        if tuple(expected.shape) != tuple(salience.shape):
            raise AssertionError("salience shape mismatch")
        recomputed.append(expected)
    max_salience_error = max(
        float((actual - expected).abs().max())
        for actual, expected in zip(saved_salience, recomputed, strict=True)
    )
    masses = []
    for layer_index, salience in enumerate(recomputed):
        layer_values = []
        for head_index, row in enumerate(salience):
            kept = set(int(x) for x in masks[layer_index][head_index])
            evicted = [index for index in range(row.shape[-1]) if index not in kept]
            layer_values.append(float(row[evicted].sum()))
        masses.append(layer_values)
    z = float(np.asarray(masses, dtype=float).mean())
    persisted_mass = data["mass_per_head"]
    return {
        "probability_sum_max_abs_error": probability_sum_error,
        "salience_reconstruction_max_abs_error": max_salience_error,
        "mass_per_head": masses,
        "persisted_mass_per_head_max_abs_error": max(
            abs(float(a) - float(b))
            for layer_a, layer_b in zip(masses, persisted_mass, strict=True)
            for a, b in zip(layer_a, layer_b, strict=True)
        ),
        "z": z,
        "tensor_keys": sorted(data),
        "pending_token_id": int(data["pending_token_id"]),
        "shape": {
            "layers": len(probabilities),
            "query_heads": int(probabilities[0].shape[0]),
            "kv_heads": int(value_norm[0].shape[0]),
            "cache_length": int(value_norm[0].shape[-1]),
        },
    }


def closed_form_loo(rows):
    y = np.asarray([float(row["signed_loss"]) for row in rows], dtype=float)
    x = np.asarray([float(row["z"]) for row in rows], dtype=float)
    baseline = np.asarray([(y.sum() - y[i]) / (len(y) - 1) for i in range(len(y))])
    prediction = []
    for test_index in range(len(y)):
        train = np.asarray([i for i in range(len(y)) if i != test_index])
        mean_x = x[train].mean()
        scale_x = x[train].std()
        z_train = (x[train] - mean_x) / scale_x
        z_test = (x[test_index] - mean_x) / scale_x
        mean_y = y[train].mean()
        coefficient = np.sum(z_train * (y[train] - mean_y)) / (np.sum(z_train**2) + 1.0)
        prediction.append(float(mean_y + coefficient * z_test))
    prediction = np.asarray(prediction)
    baseline_mse = float(np.mean((baseline - y) ** 2))
    feature_mse = float(np.mean((prediction - y) ** 2))
    abs_baseline = np.abs(baseline - y)
    abs_feature = np.abs(prediction - y)
    return {
        "baseline": baseline.tolist(),
        "prediction": prediction.tolist(),
        "baseline_mse": baseline_mse,
        "feature_mse": feature_mse,
        "gain_fraction": 1.0 - feature_mse / baseline_mse,
        "model_wins": int(np.sum(abs_feature < abs_baseline)),
        "baseline_wins": int(np.sum(abs_feature > abs_baseline)),
        "ties": int(np.sum(abs_feature == abs_baseline)),
        "auc": auc(x, y > 0),
    }


def auc(values, positive):
    values = np.asarray(values, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    positive_values = values[positive]
    negative_values = values[~positive]
    comparisons = [
        1.0 if value > other else 0.5 if value == other else 0.0
        for value in positive_values
        for other in negative_values
    ]
    return float(np.mean(comparisons))


def collect_records():
    manifest_payload = load_json(MANIFEST)
    manifest_rows = manifest_payload["prompts"] if isinstance(manifest_payload, dict) else manifest_payload
    manifest = {row["id"]: row for row in manifest_rows}
    records = {}
    for directory in (FIRST, REST):
        run = load_json(directory / "run.json")
        for item in run.get("prompts", []):
            record_path = directory / item["path"]
            record = load_json(record_path)
            prompt_id = record["manifest_row"]["id"]
            if prompt_id in records:
                raise AssertionError(f"duplicate record {prompt_id}")
            tensor_path = directory / record["raw_tensor_file"]
            records[prompt_id] = (record_path, record, tensor_path, run)
    if set(records) != EXPECTED_IDS:
        raise AssertionError(f"record IDs differ: {sorted(set(records) ^ EXPECTED_IDS)}")
    return manifest, records


def audit():
    score = official_scorer()
    manifest, records = collect_records()
    rows = []
    case_checks = {}
    numeric_diagnostics = {}
    for prompt_id in sorted(records):
        record_path, record, tensor_path, run = records[prompt_id]
        row = manifest[prompt_id]
        checks = record.get("checks", {})
        if record.get("status") != "completed":
            raise AssertionError(f"incomplete record {prompt_id}")
        if not checks.get("all_checks_pass") or not all(checks.values()):
            raise AssertionError(f"collector controls failed for {prompt_id}")
        boundary = record["b16"]
        prompt_length = int(record["prompt_length"])
        expected_length = prompt_length + 15
        semantic = {
            "b16_generated_count_16": len(boundary["generated_ids"]) == 16,
            "b16_cache_length_prompt_plus_15": boundary["cache_lengths"] == [expected_length] * len(boundary["cache_lengths"]),
            "b16_logical_position_prompt_plus_15": boundary["logical_position"] == expected_length,
            "all_current_arms_share_16_prefix": all(
                arm["continuation"]["token_ids"][:16] == boundary["generated_ids"]
                for arm in (record["reference"], record["noop"], record["action"])
            ),
        }
        compression = record["action"]["compression"]
        masks = compression["kept_index_counts_per_layer_head"]
        mask_cardinality = {
            "all_mask_counts_equal_expected": all(
                count == int(length * 0.9)
                for layer, length in zip(masks, compression["before_lengths"], strict=True)
                for count in layer
            ),
            "all_mask_counts_equal_after_lengths": all(
                count == length
                for layer, length in zip(masks, compression["after_lengths"], strict=True)
                for count in layer
            ),
            "mask_layer_count_matches_cache": len(masks) == len(boundary["cache_lengths"]),
            "mask_head_count_4": all(len(layer) == 4 for layer in masks),
        }
        raw_masks = torch.load(tensor_path, map_location="cpu", weights_only=True)["native_masks"].tolist()
        native_hash, native_counts = index_digest(raw_masks)
        mask_cardinality.update({
            "feature_mask_matches_raw_mask": record["feature"]["native_masks"] == raw_masks,
            "raw_mask_counts_match_compression": native_counts == masks,
            "raw_mask_hash_matches_compression": native_hash == compression["kept_index_hash"],
            "raw_mask_indices_unique_in_range": all(
                len(head) == len(set(head)) and all(0 <= int(index) < compression["before_lengths"][layer_index] for index in head)
                for layer_index, layer in enumerate(raw_masks)
                for head in layer
            ),
        })
        reconstructed = reconstruct_feature(tensor_path, raw_masks)
        feature = record["feature"]
        feature_checks = {
            "probabilities_normalized": reconstructed["probability_sum_max_abs_error"] < 2e-5,
            "salience_reconstructed": reconstructed["salience_reconstruction_max_abs_error"] < 2e-5,
            "mass_per_head_matches_record": all(
                close(a, b) for layer_a, layer_b in zip(reconstructed["mass_per_head"], feature["mass_per_head"], strict=True) for a, b in zip(layer_a, layer_b, strict=True)
            ),
            "z_matches_record": close(reconstructed["z"], feature["z"]),
        }
        scored = {}
        for arm_name in ("reference", "noop", "action"):
            arm = record[arm_name]
            score_value = score(arm["continuation"]["text"], row["answers"])
            stored = arm["score"]
            scored[arm_name] = {
                "stored_score_fraction": stored["score_fraction"],
                "recomputed_score_fraction": score_value["score_fraction"],
                "score_fraction_equal": close(stored["score_fraction"], score_value["score_fraction"], 1e-12),
                "pass_vector_equal": stored["pass_vector"] == score_value["pass_vector"],
                "postprocessed_equal": stored["postprocessed_prediction"] == score_value["postprocessed"],
            }
        recomputed_loss = (
            scored["reference"]["recomputed_score_fraction"]
            - scored["action"]["recomputed_score_fraction"]
        )
        numeric_diagnostics[prompt_id] = {
            "probability_sum_max_abs_error": reconstructed["probability_sum_max_abs_error"],
            "salience_reconstruction_max_abs_error": reconstructed["salience_reconstruction_max_abs_error"],
            "z_abs_error": abs(reconstructed["z"] - float(feature["z"])),
            "mass_per_head_max_abs_error": max(
                abs(a - b)
                for layer_a, layer_b in zip(reconstructed["mass_per_head"], feature["mass_per_head"], strict=True)
                for a, b in zip(layer_a, layer_b, strict=True)
            ),
            "persisted_mass_per_head_max_abs_error": reconstructed["persisted_mass_per_head_max_abs_error"],
            "max_score_fraction_abs_error": max(
                abs(value["stored_score_fraction"] - value["recomputed_score_fraction"])
                for value in scored.values()
            ),
            "signed_loss_abs_error": abs(float(record["signed_loss"]) - recomputed_loss),
            "compression_after_bytes_delta": int(compression["after_bytes"] - compression["expected_after_bytes"]),
        }
        old_dir = FIRST if prompt_id.endswith("-000") else REST
        old_record = load_json(old_dir.parent / ("ea-dev-v1-first" if old_dir == FIRST else "ea-dev-v1-rest") / record_path.name)
        old_ref = old_record["reference"]
        current_ref = record["reference"]
        old_score = score(old_ref["text"], row["answers"])
        historical = {
            "historical_record_id_exact": old_record["manifest_row"]["id"] == prompt_id,
            "historical_reference_path_shared_boundary": old_ref.get("reference_path") == "shared_boundary",
            "reference_token_ids_exact": current_ref["continuation"]["token_ids"] == old_ref["token_ids"],
            "reference_termination_exact": current_ref["continuation"]["termination_reason"] == old_ref["termination_reason"],
            "reference_score_exact": close(current_ref["score"]["score_fraction"], old_score["score_fraction"], 1e-12),
        }
        horizon = {
            "reference_final_cache_length": current_ref["continuation"]["final_cache_lengths"][0],
            "action_final_cache_length": record["action"]["continuation"]["final_cache_lengths"][0],
            "reference_length_equation": current_ref["continuation"]["final_cache_lengths"] == [expected_length + len(current_ref["continuation"]["token_ids"]) - 16] * len(current_ref["continuation"]["final_cache_lengths"]),
            "action_length_equation": record["action"]["continuation"]["final_cache_lengths"] == [compression["after_lengths"][0] + len(record["action"]["continuation"]["token_ids"]) - 16] * len(record["action"]["continuation"]["final_cache_lengths"]),
        }
        all_case_checks = {**semantic, **mask_cardinality, **feature_checks, **historical, **horizon}
        for arm_name, values in scored.items():
            all_case_checks.update({f"{arm_name}_{key}": value for key, value in values.items() if key.endswith("equal") or key in ("score_fraction_equal", "pass_vector_equal", "postprocessed_equal")})
        case_checks[prompt_id] = all_case_checks
        rows.append({"prompt_id": prompt_id, "signed_loss": float(record["signed_loss"]), "z": float(feature["z"])})

    arithmetic = closed_form_loo(rows)
    analysis = load_json(ANALYSIS)
    summary_comparison = {
        "baseline_mse_equal": close(arithmetic["baseline_mse"], analysis["baseline"]["mse"], 1e-12),
        "feature_mse_equal": close(arithmetic["feature_mse"], analysis["feature"]["mse"], 1e-12),
        "gain_equal": close(arithmetic["gain_fraction"], analysis["gain_fraction"], 1e-12),
        "wins_equal": arithmetic["model_wins"] == analysis["prompt_wins"]["model_wins"],
        "auc_equal": close(arithmetic["auc"], analysis["raw_positive_auc"], 1e-12),
    }
    output = {
        "scope": "Independent audit of exposed 12-prompt B16 results",
        "records": len(rows),
        "positive_labels": sum(row["signed_loss"] > 0 for row in rows),
        "zero_labels": sum(row["signed_loss"] == 0 for row in rows),
        "case_checks": case_checks,
        "numeric_diagnostics": {
            "max_probability_sum_abs_error": max(item["probability_sum_max_abs_error"] for item in numeric_diagnostics.values()),
            "max_salience_reconstruction_abs_error": max(item["salience_reconstruction_max_abs_error"] for item in numeric_diagnostics.values()),
            "max_z_abs_error": max(item["z_abs_error"] for item in numeric_diagnostics.values()),
            "max_mass_per_head_abs_error": max(item["mass_per_head_max_abs_error"] for item in numeric_diagnostics.values()),
            "max_persisted_mass_per_head_abs_error": max(item["persisted_mass_per_head_max_abs_error"] for item in numeric_diagnostics.values()),
            "max_score_fraction_abs_error": max(item["max_score_fraction_abs_error"] for item in numeric_diagnostics.values()),
            "max_signed_loss_abs_error": max(item["signed_loss_abs_error"] for item in numeric_diagnostics.values()),
            "compression_after_bytes_delta_values": sorted({item["compression_after_bytes_delta"] for item in numeric_diagnostics.values()}),
        },
        "arithmetic": arithmetic,
        "analyzer_comparison": summary_comparison,
        "all_case_checks_pass": all(all(values.values()) for values in case_checks.values()),
        "all_arithmetic_checks_pass": all(summary_comparison.values()),
        "verdict": "audit_passes_controls_and_arithmetic; predictor_gates_fail",
        "limitations": [
            "The audit verifies persisted tensors and continuations; it does not rerun the 7B GPU inference locally.",
            "The exposed 12 prompts remain development data, so this is not confirmation evidence or a generalization claim.",
        ],
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: output[k] for k in ("records", "positive_labels", "zero_labels", "all_case_checks_pass", "all_arithmetic_checks_pass", "verdict")}, indent=2))


if __name__ == "__main__":
    audit()
