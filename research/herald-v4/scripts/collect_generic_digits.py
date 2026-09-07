#!/usr/bin/env python3
"""Collect the frozen generic seven-digit continuation feature and outcomes."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import measure_digit_copy as digit  # noqa: E402
import run_pair_pilot as runner  # noqa: E402
from score_ruler_pilot import score_prediction  # noqa: E402

ROOT = SCRIPT_DIR.parent
PREFIX = "For an independent check, the next seven digits after 3.1415926535 are "
DIGITS = "8979323"
FRAC = 0.10
FEATURE_NAMES = ["z", "reference_nll", "js", "log_prompt_length", "removed_fraction"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixed_spec(tokenizer):
    encoded = tokenizer(PREFIX + DIGITS, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if ids and isinstance(ids[0], list):
        ids, offsets = ids[0], offsets[0]
    positions = [i for i, (start, end) in enumerate(offsets) if end > len(PREFIX)]
    if positions != list(range(len(ids) - 7, len(ids))) or len(positions) != 7:
        raise ValueError("fixed digits are not seven contiguous native tokens")
    for j, index in enumerate(positions):
        if offsets[index] != (len(PREFIX) + j, len(PREFIX) + j + 1):
            raise ValueError("fixed digit offsets do not reconstruct the prefix")
        if tokenizer.decode([int(ids[index])], skip_special_tokens=False).strip() != DIGITS[j]:
            raise ValueError("fixed digit token does not decode to its native digit")
    prefix_ids = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    if [int(i) for i in ids[: positions[0]]] != [int(i) for i in prefix_ids]:
        raise ValueError("fixed prefix tokenization is not stable")
    return {"prefix": PREFIX, "digits": DIGITS, "prefix_ids": [int(i) for i in prefix_ids],
            "digit_ids": [int(ids[i]) for i in positions]}


def prior_record(root, row_id):
    matches = []
    for path in Path(root).expanduser().resolve().glob("**/*.json"):
        if path.name == "run.json":
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("manifest_row", {}).get("id") == row_id:
            matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(f"expected one prior record for {row_id}, found {len(matches)}")
    return matches[0]


def old_branch(record, name):
    branch = record.get("branches", {}).get(name)
    if branch is None:
        branch = record.get("arms", {}).get("knorm:0.1" if name == "action" else "knorm:0.0")
    if branch is None:
        raise ValueError(f"prior record has no {name} branch")
    return branch


def compact(engine, compression, boundary):
    return runner.compact_compression(engine, compression, boundary)


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("manifest", "model", "engine-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--verify-sequential", action="store_true")
    args = parser.parse_args(argv)
    import torch
    import transformers

    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if args.ids:
        selected = set(args.ids)
        if not selected <= {row["id"] for row in rows}:
            raise ValueError("requested id is absent from manifest")
        rows = [row for row in rows if row["id"] in selected]
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
        if bundle.get("sklearn_version") != sklearn.__version__ or bundle.get("feature_count") != 5:
            raise ValueError("model bundle schema does not match five generic features")
    source_paths = [Path(__file__), Path(digit.__file__), Path(runner.__file__),
                    ROOT / "scripts" / "score_ruler_pilot.py", engine_path]
    run = {"schema_version": "generic_digit_probe.v1", "status": "running",
           "manifest_sha256": sha(manifest_path),
           "source_hashes": {str(path.resolve()): sha(path) for path in source_paths},
           "runtime": runner.model_runtime_identity(engine, model, args.model, transformers, torch),
           "config": {"prefix": PREFIX, "digits": DIGITS, "removal_fraction": FRAC,
                      "feature_names": FEATURE_NAMES, "seed": 0,
                      "verify_sequential": bool(args.verify_sequential)},
           "models_sha256": sha(args.models) if args.models else None, "prompts": [], "failures": []}
    runner.write_json(output / "run.json", run)
    eos = runner.eos_ids(model, tokenizer)
    for index, row in enumerate(rows):
        path = output / runner.safe_filename(index, row["id"])
        record = {"schema_version": "generic_digit_probe.v1", "status": "running", "manifest_row": row}
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
            if not state["boundary_source_equal"] or not state["boundary_source_disjoint"]:
                raise RuntimeError("B0 source and boundary cache controls failed")
            observation_started = time.perf_counter()
            spec = fixed_spec(tokenizer)
            pending = torch.tensor([[boundary.pending_token_id]], dtype=ids.dtype)
            sequence = torch.tensor([pending[0].tolist() + spec["prefix_ids"] + spec["digit_ids"][:-1]], dtype=ids.dtype)
            zero, action = engine.ActionSpec("knorm", 0.0), engine.ActionSpec("knorm", FRAC)
            ref_logits, ref_comp, ref_cost = digit.forward(model, engine, boundary, torch, sequence, zero)
            act_logits, act_comp, act_cost = digit.forward(model, engine, boundary, torch, sequence, action)
            probe = engine.full_vocabulary_js(ref_logits[0], act_logits[0], action)
            prefix_n = len(spec["prefix_ids"])
            ref_metrics = digit.digit_metrics(ref_logits[prefix_n:], spec["digit_ids"], torch, eos)
            act_metrics = digit.digit_metrics(act_logits[prefix_n:], spec["digit_ids"], torch, eos)
            z = float(np.mean(np.asarray(ref_metrics["logprobs"]) - np.asarray(act_metrics["logprobs"])))
            reference_nll = float(-np.mean(ref_metrics["logprobs"]))
            action_compact = compact(engine, act_comp, boundary)
            ref_compact = compact(engine, ref_comp, boundary)
            reduction_seconds = float(probe.reduction_seconds)
            feature_started = time.perf_counter()
            features = [z, reference_nll, float(probe.js_divergence), float(np.log(int(ids.shape[1]))),
                        float(action_compact["before_bytes"] and 1.0 - action_compact["after_bytes"] / action_compact["before_bytes"])]
            if not np.isfinite(features).all():
                raise ValueError("generic features are nonfinite")
            predictions = {}
            prediction_started = time.perf_counter()
            if bundle is not None:
                vector = np.asarray(features, dtype=float)
                for name, fitted in bundle["models"].items():
                    predictions[name] = float(fitted["constant"]) if "constant" in fitted else float(fitted["estimator"].predict(vector[np.asarray(fitted["columns"], dtype=int)].reshape(1, -1))[0])
            prediction_seconds = time.perf_counter() - prediction_started
            feature_seconds = time.perf_counter() - feature_started
            vectors = {"reference_digit_logprobs": ref_metrics["logprobs"], "action_digit_logprobs": act_metrics["logprobs"],
                       "pending_reference_top_ids": ref_logits[0].topk(min(8, ref_logits.shape[-1])).indices.tolist(),
                       "pending_action_top_ids": act_logits[0].topk(min(8, act_logits.shape[-1])).indices.tolist()}
            observation = {"id": row["id"], "spec": spec, "features": features,
                           "feature_names": FEATURE_NAMES, "predictions": predictions,
                           "vectors": vectors, "reference": ref_metrics, "action": act_metrics,
                           "probe": probe.to_dict(), "native_masks": {"reference": ref_compact, "action": action_compact},
                           "state": state,
                           "cost": {"prefill_seconds_synchronized": prefill_seconds,
                                    "observation_wall_seconds": time.perf_counter() - observation_started,
                                    "reference_clone_seconds": ref_cost["clone_seconds"], "action_clone_seconds": act_cost["clone_seconds"],
                                    "reference_compression_seconds": ref_cost["compression_seconds"], "action_compression_seconds": act_cost["compression_seconds"],
                                    "reference_compression_validation_seconds": ref_cost["compression_validation_seconds"],
                                    "action_compression_validation_seconds": act_cost["compression_validation_seconds"],
                                    "reference_forward_seconds": ref_cost["forward_seconds"], "action_forward_seconds": act_cost["forward_seconds"],
                                    "reference_transfer_seconds": ref_cost["transfer_seconds"], "action_transfer_seconds": act_cost["transfer_seconds"],
                                    "probe_clone_seconds": ref_cost["clone_seconds"] + act_cost["clone_seconds"],
                                    "probe_forward_seconds": ref_cost["forward_seconds"] + act_cost["forward_seconds"],
                                    "probe_compression_seconds": ref_cost["compression_seconds"] + act_cost["compression_seconds"],
                                    "probe_transfer_seconds": ref_cost["transfer_seconds"] + act_cost["transfer_seconds"],
                                    "probe_reduction_seconds": reduction_seconds, "feature_seconds": feature_seconds,
                                    "prediction_seconds": prediction_seconds}}
            feature_path = path.with_name(path.stem + ".features.json")
            vector_path = path.with_name(path.stem + ".vectors.npz")
            runner.write_json(feature_path, observation)
            np.savez_compressed(vector_path, reference_digit_logprobs=np.asarray(vectors["reference_digit_logprobs"], dtype=np.float32),
                                action_digit_logprobs=np.asarray(vectors["action_digit_logprobs"], dtype=np.float32),
                                pending_reference_logits=ref_logits[0].numpy().astype(np.float32),
                                pending_action_logits=act_logits[0].numpy().astype(np.float32),
                                pending_reference_top_ids=np.asarray(vectors["pending_reference_top_ids"], dtype=np.int64),
                                pending_action_top_ids=np.asarray(vectors["pending_action_top_ids"], dtype=np.int64))
            feature_sha = sha(feature_path)
            record.update({"prompt_token_ids": ids[0].tolist(), "token_identity": token_identity,
                      "boundary": boundary.to_dict(), "observation": observation,
                      "features_sha256": feature_sha, "vectors_sha256": sha(vector_path), "branches": {}})
            arms = {}
            for name, selected_action in (("reference", zero), ("noop", zero), ("action", action)):
                arm = engine.continue_from_boundary(model, boundary, max_new_tokens=row.get("max_new_tokens", 128), eos_ids=eos, action=selected_action)
                text = tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True)
                arms[name] = arm
                record["branches"][name] = {"continuation": runner.continuation_dict(arm.continuation, text),
                                             "score": score_prediction(text, row["answers"]),
                                             "compression": compact(engine, arm.compression, boundary)}
            checks = {"boundary_source_equal": state["boundary_source_equal"], "boundary_source_disjoint": state["boundary_source_disjoint"],
                      "source_unchanged": engine.cache_fingerprint(source) == source_fp and engine.cache_fingerprint(boundary.cache) == boundary_fp,
                      "noop_ids_exact": arms["reference"].continuation.token_ids == arms["noop"].continuation.token_ids,
                      "noop_termination_exact": arms["reference"].continuation.termination_reason == arms["noop"].continuation.termination_reason,
                      "candidate_mask_exact": action_compact["kept_index_hash"] == record["branches"]["action"]["compression"]["kept_index_hash"],
                      "physical_effect_exact": action_compact["physical_effect_exact"] and action_compact["strictly_reduced_for_nonzero_action"],
                      "features_precede_outcomes_unchanged": sha(feature_path) == feature_sha and sha(vector_path) == record["vectors_sha256"],
                      "finite_probe": bool(probe.finite) and all(np.isfinite(v) for v in features)}
            if index == 0 and (device.type == "cpu" or args.verify_sequential):
                seq_ref, seq_ref_comp = digit.sequential(model, engine, boundary, torch, sequence, zero)
                seq_act, seq_act_comp = digit.sequential(model, engine, boundary, torch, sequence, action)
                checks["batched_sequential_reference"] = bool(torch.allclose(ref_logits, seq_ref, atol=1e-5, rtol=1e-5))
                checks["batched_sequential_action"] = bool(torch.allclose(act_logits, seq_act, atol=1e-5, rtol=1e-5))
                checks["sequential_mask_reference"] = runner.index_digest(seq_ref_comp.kept_indices)[0] == ref_compact["kept_index_hash"]
                checks["sequential_mask_action"] = runner.index_digest(seq_act_comp.kept_indices)[0] == action_compact["kept_index_hash"]
            if args.prior:
                old_path, old = prior_record(args.prior, row["id"])
                old_ref, old_act = old_branch(old, "reference"), old_branch(old, "action")
                checks.update({"prior_reference_unforced_exact": list(arms["reference"].continuation.token_ids) == old_ref["continuation"]["token_ids"],
                               "prior_action_unforced_exact": list(arms["action"].continuation.token_ids) == old_act["continuation"]["token_ids"],
                               "prior_reference_termination_exact": arms["reference"].continuation.termination_reason == old_ref["continuation"]["termination_reason"],
                               "prior_action_termination_exact": arms["action"].continuation.termination_reason == old_act["continuation"]["termination_reason"],
                               "prior_action_mask_exact": action_compact["kept_index_hash"] == old_act["compression"]["kept_index_hash"],
                               "prior_path_sha256": sha(old_path)})
            checks["all_checks_pass"] = all(value for key, value in checks.items() if key != "prior_path_sha256")
            if not checks["all_checks_pass"]:
                raise RuntimeError("generic digit gate failed: " + json.dumps(checks))
            record.update({"status": "completed", "checks": checks,
                           "signed_loss": record["branches"]["reference"]["score"]["score_fraction"] - record["branches"]["action"]["score"]["score_fraction"]})
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
