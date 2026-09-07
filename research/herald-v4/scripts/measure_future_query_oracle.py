#!/usr/bin/env python3
"""Measure the frozen future-query attention oracle on stored value-level cases."""

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as runner  # noqa: E402
from score_ruler_pilot import score_prediction  # noqa: E402
from value_group_adapter import locate_values  # noqa: E402

ROOT = SCRIPT_DIR.parent
FRAC = 0.05
EPSILON = 1e-12
REQUIRED_PRIOR_SOURCE_NAMES = (
    "engine.py",
    "run_pair_pilot.py",
    "value_group_adapter.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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


def semantic_source_hashes(source_hashes):
    """Index the shared implementation files without binding to checkout paths."""
    result = {}
    for path, digest in source_hashes.items():
        name = Path(path).name
        if name not in REQUIRED_PRIOR_SOURCE_NAMES:
            continue
        if name in result and result[name] != digest:
            raise ValueError(f"duplicate semantic source identity for {name}")
        result[name] = digest
    return result


def validate_prior_run(root, runtime, source_hashes):
    run_path = Path(root).expanduser().resolve() / "run.json"
    if not run_path.is_file():
        raise ValueError(f"prior run manifest is missing: {run_path}")
    try:
        prior_run = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"prior run manifest is not valid JSON: {run_path}") from error
    if prior_run.get("status") != "completed" or prior_run.get("failures"):
        raise ValueError("prior run did not complete cleanly")
    if prior_run.get("runtime") != runtime:
        raise ValueError("prior run runtime identity does not match current runtime")
    current_sources = semantic_source_hashes(source_hashes)
    prior_sources = semantic_source_hashes(prior_run.get("source_hashes", {}))
    if set(prior_sources) != set(REQUIRED_PRIOR_SOURCE_NAMES):
        raise ValueError("prior run is missing required shared source hashes")
    if any(prior_sources[name] != current_sources.get(name) for name in REQUIRED_PRIOR_SOURCE_NAMES):
        raise ValueError("prior run shared source hashes do not match current sources")
    prompts = prior_run.get("prompts")
    if not isinstance(prompts, list) or any(item.get("status") != "completed" for item in prompts):
        raise ValueError("prior run contains an incomplete prompt")
    return run_path


def validate_prior_record(record):
    if record.get("status") != "completed":
        raise ValueError("prior record is not completed")
    checks = record.get("checks")
    if not isinstance(checks, dict) or not checks or not all(checks.values()):
        raise ValueError("prior record checks are not all true")
    action = branch(record, "action")
    compression = action.get("compression")
    if not isinstance(compression, dict):
        raise ValueError("prior record has no action compression")
    action_spec = compression.get("action", {})
    if action_spec.get("name") != "knorm" or float(action_spec.get("removal_fraction", -1.0)) != FRAC:
        raise ValueError("prior record action is not native Knorm .05")
    before_lengths = compression.get("before_lengths")
    after_lengths = compression.get("after_lengths")
    expected_lengths = compression.get("expected_after_lengths")
    if not before_lengths or after_lengths != expected_lengths:
        raise ValueError("prior native physical lengths are invalid")
    if compression.get("after_bytes") != compression.get("expected_after_bytes"):
        raise ValueError("prior native physical byte effect is invalid")
    if not compression.get("physical_effect_exact"):
        raise ValueError("prior native physical effect check is false")
    if not compression.get("kept_index_hash"):
        raise ValueError("prior native action mask hash is missing")
    if checks.get("physical_effect_exact") is not True or checks.get("candidate_mask_exact") is not True:
        raise ValueError("prior native physical or mask check is false")


def branch(record, name):
    value = record.get("branches", {}).get(name)
    if value is None:
        raise ValueError(f"prior record has no {name} branch")
    return value


def ids_of(value):
    return [int(x) for x in value["continuation"]["token_ids"]]


def is_real_study047(row):
    return str(row.get("id", "")).startswith("value-level-v1-evaluation-")


def validate_query_shape(row, mapped, query_positions, layer_count):
    if not is_real_study047(row):
        return
    if layer_count != 28:
        raise ValueError(f"study047 requires 28 layers, found {layer_count}")
    if len(mapped) != 4:
        raise ValueError(f"study047 requires four mapped values, found {len(mapped)}")
    if any(len(set(item["query_positions"])) != 7 for item in mapped):
        raise ValueError("study047 requires seven unique query positions per value")
    if len(set(query_positions)) != 29 or query_positions[0] != 0:
        raise ValueError("study047 requires one pending query and 28 unique value queries")


def map_reference_values(tokenizer, values, reference_ids):
    """Map each prompt value to one exact contiguous seven-token reference span."""
    mapped = []
    for item in values:
        encoded = tokenizer(item["value_text"], add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        token_ids = [int(x) for x in encoded]
        if len(token_ids) != 7:
            raise ValueError("official value is not seven native tokens")
        matches = [
            start for start in range(len(reference_ids) - 6)
            if reference_ids[start : start + 7] == token_ids
        ]
        if len(matches) != 1:
            raise ValueError(f"value {item['value_text']} has {len(matches)} reference occurrences")
        start = matches[0]
        mapped.append({**item, "reference_token_ids": token_ids,
                       "continuation_positions": list(range(start, start + 7)),
                       "query_positions": list(range(start, start + 7))})
    used = [p for item in mapped for p in item["continuation_positions"]]
    if len(set(used)) != 28:
        raise ValueError("reference value mappings overlap")
    return mapped


def capture_teacher_forced(model, engine, boundary, torch, modeling_qwen2, input_ids, query_positions):
    modules = [layer.self_attn for layer in model.model.layers]
    captured = [{"q": None, "cos": None, "sin": None, "o": None} for _ in modules]
    handles = []

    def pre_hook(index):
        def hook(module, args, kwargs):
            embeddings = kwargs.get("position_embeddings")
            if embeddings is None and len(args) >= 2:
                embeddings = args[1]
            if embeddings is None:
                raise RuntimeError("native Qwen position embeddings were not supplied")
            captured[index]["cos"] = embeddings[0][:, query_positions].detach().clone()
            captured[index]["sin"] = embeddings[1][:, query_positions].detach().clone()
        return hook

    def projection_hook(index):
        def hook(module, args, output):
            captured[index]["q"] = output[:, query_positions].detach().clone()
        return hook

    def output_hook(index):
        def hook(module, args, output):
            captured[index]["o"] = output[:, query_positions].detach().clone()
        return hook

    for index, module in enumerate(modules):
        handles += [module.register_forward_pre_hook(pre_hook(index), with_kwargs=True),
                    module.q_proj.register_forward_hook(projection_hook(index)),
                    module.o_proj.register_forward_hook(output_hook(index))]
    device = engine._model_device(model)
    n = int(input_ids.shape[1])
    positions = torch.arange(boundary.logical_position, boundary.logical_position + n, device=device, dtype=torch.long)
    cache = engine.clone_cache(boundary.cache)
    mask = torch.ones((1, int(cache.get_seq_length()) + n), device=device, dtype=torch.long)
    runner.sync(engine, device)
    started = time.perf_counter()
    try:
        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), attention_mask=mask,
                           position_ids=positions.unsqueeze(0), cache_position=positions,
                           past_key_values=cache, use_cache=True, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    runner.sync(engine, device)
    elapsed = time.perf_counter() - started
    for index, item in enumerate(captured):
        if any(item[key] is None for key in ("q", "cos", "sin", "o")):
            raise RuntimeError(f"teacher-forced hooks missed layer {index}")
    return output.past_key_values, captured, elapsed


def attention_oracle(model, engine, torch, modeling_qwen2, boundary, full_cache, captured,
                     kept_indices, query_positions, value_query_positions):
    import torch.nn.functional as F

    def require_finite(tensor, name):
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"nonfinite {name} tensor")

    prefix_length = int(boundary.cache.get_seq_length())
    sequence_length = int(full_cache.get_seq_length()) - prefix_length
    query_positions = list(query_positions)
    query_to_local = {position: index for index, position in enumerate(query_positions)}
    target_local = [[query_to_local[position] for position in positions] for positions in value_query_positions]
    layer_errors = [[] for _ in value_query_positions]
    pending_errors = []
    drift_abs, drift_rel = 0.0, 0.0
    gqa_ok, causal_ok, finite_ok = True, True, True
    for layer_index, module in enumerate([layer.self_attn for layer in model.model.layers]):
        item = captured[layer_index]
        head_dim = int(module.head_dim)
        query_heads = int(item["q"].shape[-1]) // head_dim
        kv_heads = int(full_cache.layers[layer_index].keys.shape[1])
        if query_heads % kv_heads:
            raise ValueError("query heads are not divisible by KV heads")
        groups = query_heads // kv_heads
        q = item["q"].view(1, len(query_positions), query_heads, head_dim).transpose(1, 2)
        require_finite(q, f"layer {layer_index} raw q")
        require_finite(item["cos"], f"layer {layer_index} cos")
        require_finite(item["sin"], f"layer {layer_index} sin")
        dummy = torch.zeros_like(q)
        q, _ = modeling_qwen2.apply_rotary_pos_emb(q, dummy, item["cos"], item["sin"])
        require_finite(q, f"layer {layer_index} rotated q")
        layer_cache = full_cache.layers[layer_index]
        keys = layer_cache.keys[0].float()
        vals = layer_cache.values[0].float()
        require_finite(keys, f"layer {layer_index} raw K")
        require_finite(vals, f"layer {layer_index} raw V")
        keys = keys.repeat_interleave(groups, dim=0)
        vals = vals.repeat_interleave(groups, dim=0)
        require_finite(keys, f"layer {layer_index} expanded K")
        require_finite(vals, f"layer {layer_index} expanded V")
        total_length = keys.shape[1]
        full_mask = torch.zeros((len(query_positions), total_length), dtype=torch.bool, device=keys.device)
        full_mask[:, :prefix_length] = True
        for row, position in enumerate(query_positions):
            post_end = prefix_length + position + 1
            full_mask[row, prefix_length:post_end] = True
            causal_ok = causal_ok and post_end <= total_length and post_end > prefix_length
        masked_mask = torch.zeros((query_heads, len(query_positions), total_length), dtype=torch.bool, device=keys.device)
        for head in range(query_heads):
            retained = torch.tensor(kept_indices[layer_index][head // groups], dtype=torch.long, device=keys.device)
            masked_mask[head, :, retained] = True
        masked_mask[:, :, prefix_length:] = full_mask[:, prefix_length:].unsqueeze(0)
        q = q.float()
        full_attn = F.scaled_dot_product_attention(q, keys.unsqueeze(0), vals.unsqueeze(0),
                                                    attn_mask=full_mask.unsqueeze(0).unsqueeze(0),
                                                    dropout_p=0.0, scale=float(module.scaling))
        masked_attn = F.scaled_dot_product_attention(q, keys.unsqueeze(0), vals.unsqueeze(0),
                                                      attn_mask=masked_mask.unsqueeze(0),
                                                      dropout_p=0.0, scale=float(module.scaling))
        require_finite(full_attn, f"layer {layer_index} full attention")
        require_finite(masked_attn, f"layer {layer_index} masked attention")
        full_flat = full_attn.transpose(1, 2).reshape(1, len(query_positions), -1)
        masked_flat = masked_attn.transpose(1, 2).reshape(1, len(query_positions), -1)
        require_finite(full_flat, f"layer {layer_index} full projection input")
        require_finite(masked_flat, f"layer {layer_index} masked projection input")
        weight = module.o_proj.weight.float()
        bias = module.o_proj.bias.float() if module.o_proj.bias is not None else None
        require_finite(weight, f"layer {layer_index} o_proj weight")
        if bias is not None:
            require_finite(bias, f"layer {layer_index} o_proj bias")
        full_out = F.linear(full_flat, weight, bias)
        masked_out = F.linear(masked_flat, weight, bias)
        native_out = item["o"].float()
        require_finite(full_out, f"layer {layer_index} full projection")
        require_finite(masked_out, f"layer {layer_index} masked projection")
        require_finite(native_out, f"layer {layer_index} native projection")
        abs_drift = (full_out - native_out).abs().max().item()
        rel_drift = ((full_out - native_out).norm(dim=-1) / (native_out.norm(dim=-1) + EPSILON)).max().item()
        drift_abs, drift_rel = max(drift_abs, float(abs_drift)), max(drift_rel, float(rel_drift))
        for local in range(len(query_positions)):
            error = float((full_out[0, local] - masked_out[0, local]).norm().item() /
                          (full_out[0, local].norm().item() + EPSILON))
            finite_ok = bool(finite_ok and bool(np.isfinite(error)))
            if not np.isfinite(error):
                raise ValueError(f"nonfinite layer {layer_index} attention error")
            if local == query_to_local[0]:
                pending_errors.append(error)
            for value_index, locals_for_value in enumerate(target_local):
                if local in locals_for_value:
                    layer_errors[value_index].append(error)
    per_value = [float(np.mean(errors)) for errors in layer_errors]
    return per_value, float(np.mean(pending_errors)), {"max_abs": drift_abs, "max_relative": drift_rel,
        "query_count": len(query_positions), "gqa_groups": groups, "finite": finite_ok,
        "causal_self_included": causal_ok}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("manifest", "model", "engine-root", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--reconstruction-relative-tolerance", type=float, default=None)
    args = parser.parse_args(argv)
    import torch
    import transformers
    from transformers.models.qwen2 import modeling_qwen2

    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if args.ids:
        wanted = set(args.ids)
        if not wanted <= {row["id"] for row in rows}:
            raise ValueError("requested id is absent from manifest")
        rows = [row for row in rows if row["id"] in wanted]
    output = runner.ensure_output_dir(args.output)
    torch.manual_seed(0)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(args.model, device, dtype, transformers)
    tolerance = args.reconstruction_relative_tolerance or (1e-5 if dtype == torch.float32 else 5e-3)
    source_paths = [Path(__file__), Path(runner.__file__), Path(locate_values.__code__.co_filename), engine_path]
    source_hashes = {str(path.resolve()): sha(path) for path in source_paths}
    runtime = runner.model_runtime_identity(engine, model, args.model, transformers, torch)
    prior_run_path = validate_prior_run(args.prior, runtime, source_hashes)
    run = {"schema_version": "future_query_oracle.v1", "status": "running", "manifest_sha256": sha(manifest_path),
           "source_hashes": source_hashes, "runtime": runtime,
           "prior_run_sha256": sha(prior_run_path),
           "config": {"removal_fraction": FRAC, "epsilon": EPSILON, "reconstruction_relative_tolerance": tolerance},
           "prompts": [], "failures": []}
    runner.write_json(output / "run.json", run)
    eos = runner.eos_ids(model, tokenizer)
    for index, row in enumerate(rows):
        path = output / runner.safe_filename(index, row["id"])
        record = {"schema_version": "future_query_oracle.v1", "status": "running", "manifest_row": row}
        try:
            ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
            boundary, source = runner.build_last_prompt_boundary(engine, model, ids)
            prompt_token_ids = [int(item) for item in ids[0].tolist()]
            boundary_fp, source_fp = engine.cache_fingerprint(boundary.cache), engine.cache_fingerprint(source)
            if not engine.cache_tensors_equal(boundary.cache, source) or not engine.cache_storage_independent(boundary.cache, source):
                raise RuntimeError("B0 source and boundary cache controls failed")
            prior_path, prior = prior_record(args.prior, row["id"])
            validate_prior_record(prior)
            prior_prompt_ids = prior.get("prompt_token_ids")
            if prior_prompt_ids is not None and [int(item) for item in prior_prompt_ids] != prompt_token_ids:
                raise ValueError("full prompt token IDs do not match prior record")
            reference_ids = ids_of(branch(prior, "reference"))
            prompt_adapter = locate_values(row["prompt"], tokenizer)
            if not prompt_adapter["found"] or len(prompt_adapter["values"]) != 4:
                raise ValueError("prompt value adapter did not return four values")
            mapped = map_reference_values(tokenizer, prompt_adapter["values"], reference_ids)
            query_positions = sorted({0, *(position for item in mapped for position in item["query_positions"])})
            validate_query_shape(row, mapped, query_positions, len(model.model.layers))
            input_ids = torch.tensor([[boundary.pending_token_id] + reference_ids], dtype=ids.dtype)
            candidate_cache = engine.clone_cache(boundary.cache)
            candidate = engine.compress_knorm(candidate_cache, FRAC)
            candidate_compact = runner.compact_compression(engine, candidate, boundary)
            kept_indices = candidate.kept_indices
            del candidate_cache
            full_cache, captured, forward_seconds = capture_teacher_forced(model, engine, boundary, torch, modeling_qwen2, input_ids, query_positions)
            per_value, pending_error, reconstruction = attention_oracle(model, engine, torch, modeling_qwen2, boundary, full_cache, captured, kept_indices,
                                                                        query_positions, [item["query_positions"] for item in mapped])
            checks = {"finite_oracle": reconstruction["finite"], "gqa_alignment": reconstruction["gqa_groups"] >= 1,
                      "causal_self_included": reconstruction["causal_self_included"], "reconstruction_within_tolerance": reconstruction["max_relative"] <= tolerance,
                      "source_boundary_unchanged": engine.cache_fingerprint(boundary.cache) == boundary_fp and engine.cache_fingerprint(source) == source_fp,
                      "candidate_physical_effect_exact": candidate_compact["physical_effect_exact"] and candidate_compact["strictly_reduced_for_nonzero_action"]}
            observation = {"adapter": {"key": prompt_adapter["key"], "values": mapped}, "value_texts": [item["value_text"] for item in mapped],
                           "per_value_oracle": per_value, "pending_query_error": pending_error, "reconstruction": reconstruction,
                           "native_mask": candidate_compact, "cost": {"teacher_forced_forward_seconds": forward_seconds}}
            feature_path = path.with_name(path.stem + ".features.json")
            mask_path = path.with_name(path.stem + ".masks.npz")
            runner.write_json(feature_path, observation)
            np.savez_compressed(mask_path, kept=np.asarray(kept_indices, dtype=np.int32))
            record.update({"observation": observation, "feature_sha256": sha(feature_path), "masks_sha256": sha(mask_path),
                           "prompt_token_ids": prompt_token_ids,
                           "boundary": boundary.to_dict(), "branches": {}})
            arms = {}
            for name, fraction in (("reference", 0.0), ("noop", 0.0), ("action", FRAC)):
                arm = engine.continue_from_boundary(model, boundary, max_new_tokens=row.get("max_new_tokens", 128), eos_ids=eos,
                                                     action=engine.ActionSpec("knorm", fraction))
                text = tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True)
                arms[name] = arm
                record["branches"][name] = {"continuation": runner.continuation_dict(arm.continuation, text),
                                             "score": score_prediction(text, row["answers"]), "compression": runner.compact_compression(engine, arm.compression, boundary)}
            checks.update({"noop_ids_exact": list(arms["reference"].continuation.token_ids) == list(arms["noop"].continuation.token_ids),
                           "noop_termination_exact": arms["reference"].continuation.termination_reason == arms["noop"].continuation.termination_reason,
                           "candidate_mask_exact": candidate_compact["kept_index_hash"] == record["branches"]["action"]["compression"]["kept_index_hash"],
                           "features_precede_labels": sha(feature_path) == record["feature_sha256"] and sha(mask_path) == record["masks_sha256"],
                           "source_boundary_after_branches": engine.cache_fingerprint(boundary.cache) == boundary_fp and engine.cache_fingerprint(source) == source_fp})
            if prior is not None:
                old_ref, old_act = branch(prior, "reference"), branch(prior, "action")
                checks.update({"prior_reference_exact": list(arms["reference"].continuation.token_ids) == ids_of(old_ref),
                               "prior_action_exact": list(arms["action"].continuation.token_ids) == ids_of(old_act),
                               "prior_reference_termination_exact": arms["reference"].continuation.termination_reason == old_ref["continuation"]["termination_reason"],
                       "prior_action_termination_exact": arms["action"].continuation.termination_reason == old_act["continuation"]["termination_reason"],
                       "prior_mask_exact": candidate_compact["kept_index_hash"] == old_act["compression"]["kept_index_hash"],
                       "prior_prompt_token_ids_exact": prior_prompt_ids is None or prompt_token_ids == [int(item) for item in prior_prompt_ids],
                       "prior_path_sha256": sha(prior_path)})
            checks["all_checks_pass"] = all(value for key, value in checks.items() if key != "prior_path_sha256")
            if not checks["all_checks_pass"]:
                raise RuntimeError("future-query oracle gate failed: " + json.dumps(checks))
            record.update({"status": "completed", "checks": checks, "signed_loss": record["branches"]["reference"]["score"]["score_fraction"] - record["branches"]["action"]["score"]["score_fraction"]})
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
