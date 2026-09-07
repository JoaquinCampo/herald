#!/usr/bin/env python3
"""Collect frozen per-value signed losses and native Knorm retention features."""

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import diagnose_needle_rescue as rescue  # noqa: E402
import run_pair_pilot as runner  # noqa: E402
from score_ruler_pilot import score_prediction  # noqa: E402
from value_group_adapter import locate_values  # noqa: E402

ROOT = SCRIPT_DIR.parent
FRAC = 0.05
FEATURE_NAMES = ["log_prompt_length", "occurrence_order", "midpoint_normalized", "global_removed_fraction",
                 "value_mean_missing", "prompt_mean_missing"] + [f"head_missing_{i}" for i in range(112)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compact(engine, evidence, boundary):
    return runner.compact_compression(engine, evidence, boundary)


def prior_record(root, row_id):
    found = []
    for path in Path(root).expanduser().resolve().glob("**/*.json"):
        if path.name == "run.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("manifest_row", {}).get("id") == row_id:
            found.append((path, record))
    if len(found) != 1:
        raise ValueError(f"expected one prior record for {row_id}, found {len(found)}")
    return found[0]


def prior_arm(old, name):
    if name == "reference" and isinstance(old.get("reference"), dict):
        return old["reference"]
    key = "knorm:0.0" if name == "reference" else "knorm:0.05"
    arm = old.get("arms", {}).get(key)
    if arm is None:
        raise ValueError(f"prior record has no {name} continuation")
    return arm


def predict(bundle, features):
    if bundle is None:
        return {}
    values = np.asarray(features, dtype=float)
    predictions = {}
    for name, fitted in bundle["models"].items():
        predictions[name] = ([float(fitted["constant"])] * len(features)
                             if "constant" in fitted else fitted["estimator"].predict(values[:, fitted["columns"]]).tolist())
    return predictions


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("manifest", "model", "engine-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32")
    args = parser.parse_args(argv)
    import torch
    import transformers

    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if args.ids:
        wanted = set(args.ids)
        if not wanted <= {row["id"] for row in rows}:
            raise ValueError("requested id is absent from manifest")
        rows = [row for row in rows if row["id"] in wanted]
    if not rows:
        raise ValueError("manifest selection is empty")
    output = runner.ensure_output_dir(args.output)
    torch.manual_seed(0)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(args.model, device, dtype, transformers)
    bundle = joblib.load(args.models) if args.models else None
    if bundle is not None:
        import sklearn
        if bundle.get("feature_count") != 118 or bundle.get("sklearn_version") != sklearn.__version__:
            raise ValueError("value-level model bundle schema mismatch")
    source_paths = [Path(__file__), Path(locate_values.__code__.co_filename), Path(runner.__file__),
                    ROOT / "scripts" / "diagnose_needle_rescue.py", ROOT / "scripts" / "score_ruler_pilot.py", engine_path]
    run = {"schema_version": "value_level_probe.v1", "status": "running", "manifest_sha256": sha(manifest_path),
           "source_hashes": {str(path.resolve()): sha(path) for path in source_paths},
           "runtime": runner.model_runtime_identity(engine, model, args.model, transformers, torch),
           "config": {"removal_fraction": FRAC, "feature_count": 118, "feature_names": FEATURE_NAMES, "seed": 0},
           "models_sha256": sha(args.models) if args.models else None, "prompts": [], "failures": []}
    runner.write_json(output / "run.json", run)
    eos = runner.eos_ids(model, tokenizer)
    for index, row in enumerate(rows):
        path = output / runner.safe_filename(index, row["id"])
        record = {"schema_version": "value_level_probe.v1", "status": "running", "manifest_row": row}
        try:
            ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
            token_identity = runner.token_identity_checks(tokenizer, row, ids, row["prompt"])
            prefill_started = time.perf_counter()
            boundary, source = runner.build_last_prompt_boundary(engine, model, ids)
            runner.sync(engine, device)
            prefill_seconds = time.perf_counter() - prefill_started
            boundary_fp, source_fp = engine.cache_fingerprint(boundary.cache), engine.cache_fingerprint(source)
            state = {"boundary_fingerprint": boundary_fp, "source_fingerprint": source_fp,
                     "boundary_source_equal": engine.cache_tensors_equal(boundary.cache, source),
                     "boundary_source_disjoint": engine.cache_storage_independent(boundary.cache, source)}
            if not all(state.values()):
                raise RuntimeError("B0 source and boundary cache controls failed")
            observation_started = time.perf_counter()
            adapter_started = time.perf_counter()
            located = locate_values(row["prompt"], tokenizer)
            adapter_seconds = time.perf_counter() - adapter_started
            if not located["found"] or len(located["values"]) != 4:
                raise ValueError(f"value adapter failed: {located.get('reason')}")
            runner.sync(engine, device)
            candidate_started = time.perf_counter()
            candidate_cache = engine.clone_cache(boundary.cache)
            candidate = engine.compress_knorm(candidate_cache, FRAC)
            masks = rescue.json_indices(candidate.kept_indices)
            runner.sync(engine, device)
            candidate_seconds = time.perf_counter() - candidate_started
            action_compact = compact(engine, candidate, boundary)
            del candidate_cache
            heads = [set(head) for layer in masks for head in layer]
            value_positions = [set(item["value_positions"]) for item in located["values"]]
            if any(not positions for positions in value_positions):
                raise ValueError("adapter returned an empty value span")
            missing = [[len(positions - head) / len(positions) for head in heads] for positions in value_positions]
            prompt_mean = float(np.mean(missing))
            global_removed = float(1.0 - action_compact["after_bytes"] / action_compact["before_bytes"])
            features = []
            for j, item in enumerate(located["values"]):
                features.append([float(math.log(int(ids.shape[1]))), float(j / 3.0), float(item["position_midpoint_normalized"]),
                                 global_removed, float(np.mean(missing[j])), prompt_mean, *missing[j]])
            if any(len(values) != 6 + len(heads) for values in features) or not np.isfinite(features).all():
                raise ValueError("value-level features have invalid shape or values")
            prediction_started = time.perf_counter()
            predictions = predict(bundle, features)
            prediction_seconds = time.perf_counter() - prediction_started
            observation = {"id": row["id"], "adapter": located, "features": features,
                           "predictions": predictions, "head_count": len(heads), "feature_names": FEATURE_NAMES[:6 + len(heads)],
                           "native_masks": {"action": action_compact}, "state": state,
                           "cost": {"prefill_seconds_synchronized": prefill_seconds, "observation_wall_seconds": time.perf_counter() - observation_started,
                                    "adapter_seconds": adapter_seconds, "candidate_clone_compress_transfer_seconds": candidate_seconds,
                                    "prediction_seconds": prediction_seconds}}
            feature_path = path.with_name(path.stem + ".features.json")
            mask_path = path.with_name(path.stem + ".masks.npz")
            runner.write_json(feature_path, observation)
            np.savez_compressed(mask_path, kept=np.asarray(masks, dtype=np.int32))
            feature_sha, mask_sha = sha(feature_path), sha(mask_path)
            record.update({"prompt_token_ids": ids[0].tolist(), "token_identity": token_identity, "boundary": boundary.to_dict(),
                           "observation": observation, "features_sha256": feature_sha, "masks_sha256": mask_sha, "branches": {}})
            arms = {}
            for name, selected in (("reference", 0.0), ("noop", 0.0), ("action", FRAC)):
                arm = engine.continue_from_boundary(model, boundary, max_new_tokens=row.get("max_new_tokens", 128), eos_ids=eos,
                                                     action=engine.ActionSpec("knorm", selected))
                text = tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True)
                arms[name] = arm
                record["branches"][name] = {"continuation": runner.continuation_dict(arm.continuation, text),
                                             "score": score_prediction(text, row["answers"]), "compression": compact(engine, arm.compression, boundary)}
                if engine.cache_fingerprint(boundary.cache) != boundary_fp or engine.cache_fingerprint(source) != source_fp:
                    raise RuntimeError("source or boundary cache changed during branch")
            values = [item["value_text"] for item in located["values"]]
            if len(set(values)) != 4 or set(values) != set(row["answers"]):
                raise ValueError("located values do not map exactly to official answers")
            ref_text = record["branches"]["reference"]["continuation"]["text"]
            act_text = record["branches"]["action"]["continuation"]["text"]
            per_value = [{"value": value,
                          "reference_hit": score_prediction(ref_text, [value])["score_fraction"],
                          "action_hit": score_prediction(act_text, [value])["score_fraction"]} for value in values]
            per_losses = [item["reference_hit"] - item["action_hit"] for item in per_value]
            task_loss = record["branches"]["reference"]["score"]["score_fraction"] - record["branches"]["action"]["score"]["score_fraction"]
            checks = {"boundary_source_equal": state["boundary_source_equal"], "boundary_source_disjoint": state["boundary_source_disjoint"],
                      "source_unchanged": engine.cache_fingerprint(source) == source_fp and engine.cache_fingerprint(boundary.cache) == boundary_fp,
                      "noop_ids_exact": list(arms["reference"].continuation.token_ids) == list(arms["noop"].continuation.token_ids),
                      "noop_termination_exact": arms["reference"].continuation.termination_reason == arms["noop"].continuation.termination_reason,
                      "candidate_mask_exact": action_compact["kept_index_hash"] == record["branches"]["action"]["compression"]["kept_index_hash"],
                      "physical_effect_exact": action_compact["physical_effect_exact"] and action_compact["strictly_reduced_for_nonzero_action"],
                      "features_precede_outcomes_unchanged": sha(feature_path) == feature_sha and sha(mask_path) == mask_sha,
                      "signed_loss_aggregation_exact": abs(float(np.mean(per_losses)) - task_loss) <= 1e-12}
            if args.prior:
                old_path, old = prior_record(args.prior, row["id"])
                old_ref, old_act = prior_arm(old, "reference"), prior_arm(old, "action")
                old_ref_ids = old_ref.get("token_ids", old_ref.get("continuation", {}).get("token_ids", []))
                old_act_ids = old_act.get("token_ids", old_act.get("continuation", {}).get("token_ids", []))
                old_ref_term = old_ref.get("termination_reason", old_ref.get("continuation", {}).get("termination_reason"))
                old_act_term = old_act.get("termination_reason", old_act.get("continuation", {}).get("termination_reason"))
                old_comp = old_act.get("compression", {})
                checks.update({"prior_reference_unforced_exact": list(arms["reference"].continuation.token_ids) == list(old_ref_ids),
                               "prior_action_unforced_exact": list(arms["action"].continuation.token_ids) == list(old_act_ids),
                               "prior_reference_termination_exact": arms["reference"].continuation.termination_reason == old_ref_term,
                               "prior_action_termination_exact": arms["action"].continuation.termination_reason == old_act_term,
                               "prior_action_mask_exact": action_compact["kept_index_hash"] == old_comp.get("kept_index_hash"), "prior_path_sha256": sha(old_path)})
            checks["all_checks_pass"] = all(value for key, value in checks.items() if key != "prior_path_sha256")
            if not checks["all_checks_pass"]:
                raise RuntimeError("value-level gate failed: " + json.dumps(checks))
            record.update({"status": "completed", "checks": checks, "per_value": per_value,
                           "per_value_signed_losses": per_losses, "signed_loss": task_loss})
            status = "completed"
        except Exception as error:
            record.update({"status": "failed", "failure": {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}})
            run["failures"].append({"id": row["id"], "error": str(error)})
            status = "failed"
        runner.write_json(path, record)
        run["prompts"].append({"id": row["id"], "path": path.name, "status": status})
        runner.write_json(output / "run.json", run)
        print(row["id"], status, flush=True)
        if status == "failed":
            break
    run["status"] = "failed" if run["failures"] else "completed"
    runner.write_json(output / "run.json", run)
    return int(bool(run["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
