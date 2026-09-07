#!/usr/bin/env python3
"""Collect the frozen B16 unforced-mass feature and paired signed loss."""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_needle_rescue as rescue  # noqa: E402
import measure_retrieval_probe as retrieval  # noqa: E402
import run_pair_pilot as runner  # noqa: E402
from score_ruler_pilot import score_prediction  # noqa: E402

DECISION_TOKENS = 16
FRACTION = 0.10
SEED = 0
SCHEMA = "b16_loss.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_arm(tokenizer, continuation, answers):
    text = tokenizer.decode(list(continuation.token_ids), skip_special_tokens=True)
    return runner.continuation_dict(continuation, text), score_prediction(text, answers)


def b16_boundary(engine, model, b0, torch, eos):
    """Advance independent B0 clones, returning a B16 boundary and audit."""
    def advance():
        cache = engine.clone_cache(b0.cache)
        runner.sync(engine, engine._model_device(model))
        engine._restore_rng(b0.rng_state, engine._model_device(model))
        started = time.perf_counter()
        continuation = engine._continue_cache(
            model, b0, cache, max_new_tokens=DECISION_TOKENS, eos_ids=eos,
            first_logits_observer=None,
        )
        return cache, continuation, time.perf_counter() - started

    first_cache, first, first_seconds = advance()
    if first.termination_reason != "token_budget" or len(first.token_ids) != DECISION_TOKENS:
        return None, {"eligible": False, "reason": "eos_at_or_before_b16", "generated_ids": list(first.token_ids)}
    second_cache, second, second_seconds = advance()
    if second.termination_reason != "token_budget" or len(second.token_ids) != DECISION_TOKENS:
        raise RuntimeError("independent B16 repeat became ineligible")
    exact_ids = first.token_ids == second.token_ids
    exact_cache = engine.cache_tensors_equal(first_cache, second_cache)
    disjoint = engine.cache_storage_independent(first_cache, second_cache)
    if not (exact_ids and exact_cache and disjoint):
        raise RuntimeError("B16 prefix/cache repeat gate failed")
    device = engine._model_device(model)
    rng = engine._capture_rng(device)
    model_fp, tensor_count = engine._model_runtime_state_fingerprint(model)
    expected_length = int(b0.prompt_ids.shape[1]) + DECISION_TOKENS - 1
    lengths = engine.cache_lengths(first_cache)
    if any(length != expected_length for length in lengths):
        raise RuntimeError(f"B16 cache lengths {lengths} != {expected_length}")
    mask = torch.ones((1, expected_length), device=device, dtype=b0.attention_mask.dtype)
    pending = int(first.token_ids[-1])
    fingerprint = engine.decoder_state_fingerprint(
        b0.prompt_ids, tuple(first.token_ids), first_cache, expected_length,
        attention_mask=mask, pending_token_id=pending,
        rng_fingerprint=rng.fingerprint, model_state_fingerprint=model_fp,
    )
    boundary = dataclasses.replace(
        b0, generated_ids=tuple(first.token_ids), cache=first_cache,
        pending_token_id=pending, logical_position=expected_length,
        attention_mask=mask.detach().clone(), rng_state=rng,
        state_fingerprint=fingerprint, cache_lengths=lengths,
        cache_bytes=engine.cache_nbytes(first_cache),
        model_state_fingerprint=model_fp, model_tensor_count=tensor_count,
    )
    return boundary, {
        "eligible": True, "generated_ids": list(first.token_ids),
        "logical_position": expected_length, "cache_lengths": list(lengths),
        "first_seconds": first_seconds, "repeat_seconds": second_seconds,
        "exact_generated_ids": exact_ids, "exact_cache": exact_cache,
        "independent_cache_clones": disjoint,
    }


def run_case(args, row, index, engine, model, tokenizer, torch, modeling_qwen2, eos, output, prior_dirs):
    path = output / runner.safe_filename(index, row["id"])
    tensor_path = output / f"{path.stem}.pt"
    record = {"schema_version": SCHEMA, "status": "running", "manifest_row": row, "prompt_index": index}
    runner.write_json(path, record)
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    b0, source = runner.build_last_prompt_boundary(engine, model, prompt_ids)
    source_fp = engine.cache_fingerprint(source)
    b0_fp = engine.cache_fingerprint(b0.cache)
    boundary, audit = b16_boundary(engine, model, b0, torch, eos)
    record.update({"prompt_token_ids": [int(x) for x in prompt_ids[0].tolist()], "prompt_length": int(prompt_ids.shape[1]), "max_new_tokens": int(row.get("max_new_tokens", 128)), "b0": {"state": b0.to_dict(), "source_fingerprint": source_fp}, "b16": audit})
    if boundary is None:
        record.update({"status": "ineligible", "reason": audit["reason"]})
        runner.write_json(path, record)
        return path, "ineligible", record
    b16_fp = engine.cache_fingerprint(boundary.cache)
    native, keep_counts = rescue.native_masks(engine, boundary, FRACTION)
    feature_cache = engine.clone_cache(boundary.cache)
    pending_tensor = torch.tensor([[boundary.pending_token_id]], dtype=prompt_ids.dtype)
    unforced_logits, q_rotated, probe_cost = retrieval.probe(model, engine, boundary, torch, modeling_qwen2, pending_tensor, feature_cache)
    plain_logits, _, _, _, _ = engine._probe_logits(model, boundary, None)
    if not torch.equal(unforced_logits, plain_logits):
        raise RuntimeError("B16 instrumented/plain logits differ")
    measured = retrieval.features(boundary, q_rotated, torch)
    z, mass_per_head, reduction_seconds = retrieval.scalar_z(measured[2], native, torch)
    feature_ok = engine.cache_fingerprint(boundary.cache) == b16_fp and engine.cache_storage_independent(boundary.cache, feature_cache)
    if not feature_ok:
        raise RuntimeError("feature probe mutated source boundary")
    torch.save({"probabilities": measured[0], "value_norm": measured[1], "salience": measured[2], "native_masks": torch.tensor(native, dtype=torch.long), "mass_per_head": mass_per_head, "pending_token_id": boundary.pending_token_id}, tensor_path)
    cap = int(row.get("max_new_tokens", 128))
    ref_boundary = dataclasses.replace(boundary, cache=engine.clone_cache(boundary.cache))
    noop_boundary = dataclasses.replace(boundary, cache=engine.clone_cache(boundary.cache))
    action_boundary = dataclasses.replace(boundary, cache=engine.clone_cache(boundary.cache))
    ref_arm = engine.continue_from_boundary(model, ref_boundary, max_new_tokens=cap, eos_ids=eos, action=engine.ActionSpec("knorm", 0.0))
    noop_arm = engine.continue_from_boundary(model, noop_boundary, max_new_tokens=cap, eos_ids=eos, action=engine.ActionSpec("knorm", 0.0))
    action_arm = engine.continue_from_boundary(model, action_boundary, max_new_tokens=cap, eos_ids=eos, action=engine.ActionSpec("knorm", FRACTION))
    if args.skip_prior_check:
        old_cache = engine.clone_cache(b0.cache)
        engine._restore_rng(b0.rng_state, engine._model_device(model))
        old_uninterrupted = engine._continue_cache(model, b0, old_cache, max_new_tokens=cap, eos_ids=eos, first_logits_observer=None)
        old_ref = {"token_ids": list(old_uninterrupted.token_ids), "termination_reason": old_uninterrupted.termination_reason}
        old_path = Path("same_b0_uninterrupted_cpu_reference")
    else:
        old_path, old = retrieval.find_prior(prior_dirs, row["id"])
        old_ref = old["reference"]
    ref_cont, ref_score = score_arm(tokenizer, ref_arm.continuation, row["answers"])
    noop_cont, noop_score = score_arm(tokenizer, noop_arm.continuation, row["answers"])
    action_cont, action_score = score_arm(tokenizer, action_arm.continuation, row["answers"])
    action_compression = runner.compact_compression(engine, action_arm.compression, action_boundary)
    native_hash, _ = runner.index_digest(native)
    checks = {"source_boundary_unchanged": engine.cache_fingerprint(b0.cache) == b0_fp and engine.cache_fingerprint(source) == source_fp, "noop_matches_b16_reference": noop_arm.continuation.token_ids == ref_arm.continuation.token_ids and noop_arm.continuation.termination_reason == ref_arm.continuation.termination_reason, "old_reference_ids_exact": list(ref_arm.continuation.token_ids) == old_ref["token_ids"], "old_reference_termination_exact": ref_arm.continuation.termination_reason == old_ref["termination_reason"], "shared_prefix_preserved": all(tuple(item.token_ids[:DECISION_TOKENS]) == boundary.generated_ids for item in (ref_arm.continuation, noop_arm.continuation, action_arm.continuation)), "feature_instrumented_plain_exact": bool(torch.equal(unforced_logits, plain_logits)), "feature_source_unchanged": feature_ok, "native_mask_exact": action_compression["kept_index_hash"] == native_hash, "action_physical_effect_exact": bool(action_compression["physical_effect_exact"] and action_compression["strictly_reduced_for_nonzero_action"]), "action_source_unchanged": engine.cache_fingerprint(boundary.cache) == b16_fp}
    checks["all_checks_pass"] = all(checks.values())
    if not checks["all_checks_pass"]:
        raise RuntimeError("B16 validation gate failed: " + json.dumps(checks))
    record.update({"status": "completed", "feature": {"z": z, "mass_per_head": mass_per_head, "native_masks": native}, "cost": {"ordinary_b16_advance_seconds": audit["first_seconds"], "repeat_advance_verification_seconds": audit["repeat_seconds"], "probe_details": probe_cost, "probe_seconds": probe_cost["probe_seconds"], "feature_seconds": measured[4], "feature_transfer_bytes": measured[5], "reduction_seconds": reduction_seconds}, "reference": {"continuation": ref_cont, "score": ref_score}, "noop": {"continuation": noop_cont, "score": noop_score}, "action": {"continuation": action_cont, "score": action_score, "compression": action_compression}, "signed_loss": ref_score["score_fraction"] - action_score["score_fraction"], "q_ref": ref_score["score_fraction"], "q_action": action_score["score_fraction"], "checks": checks, "raw_tensor_file": tensor_path.name, "old_reference_path": str(old_path)})
    runner.write_json(path, record)
    return path, "completed", record


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--model", required=True); p.add_argument("--engine-root", required=True); p.add_argument("--output", required=True); p.add_argument("--ids", nargs="+"); p.add_argument("--prior-results", nargs="+", default=["results/ea-dev-v1-first", "results/ea-dev-v1-rest"]); p.add_argument("--skip-prior-check", action="store_true"); p.add_argument("--device", choices=("cpu", "cuda"), default="cpu"); p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = p.parse_args(argv)
    output = runner.ensure_output_dir(args.output); engine, engine_root, engine_path = runner.load_engine(args.engine_root); manifest_path, rows = runner.load_manifest(args.manifest); selected = [r for r in rows if not args.ids or r["id"] in args.ids]
    import torch; import transformers
    from transformers.models.qwen2 import modeling_qwen2
    torch.manual_seed(SEED); device = runner.choose_device(args.device, torch); dtype = runner.choose_dtype(args.dtype, device, torch); model, tokenizer = runner.load_model_and_tokenizer(args.model, device, dtype, transformers); eos = runner.eos_ids(model, tokenizer)
    run = {"schema_version": SCHEMA, "status": "running", "started_at_utc": datetime.now(UTC).isoformat(), "manifest_path": str(manifest_path), "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()), "engine_path": str(engine_path), "engine_sha256": sha256_file(engine_path), "script_sha256": sha256_file(Path(__file__).resolve()), "model": runner.model_runtime_identity(engine, model, args.model, transformers, torch), "config": {"decision_tokens": DECISION_TOKENS, "removal_fraction": FRACTION, "total_cap": 128, "feature": "unforced_pending_token16_full_cache_gqa_maxattn_vnorm_mean_removed_mass"}, "prompts": [], "ineligible": [], "failures": []}
    runner.write_json(output / "run.json", run)
    for index, row in enumerate(selected):
        try:
            path, status, record = run_case(args, row, index, engine, model, tokenizer, torch, modeling_qwen2, eos, output, args.prior_results)
        except Exception as exc:
            path = output / runner.safe_filename(index, row["id"]); status = "failed"; record = {"schema_version": SCHEMA, "status": status, "manifest_row": row, "failure": {"type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}}; runner.write_json(path, record); run["failures"].append({"id": row["id"], "error": str(exc)})
        run["prompts"].append({"id": row["id"], "path": path.name, "status": status})
        if status == "ineligible": run["ineligible"].append({"id": row["id"], "reason": record.get("reason")})
        runner.write_json(output / "run.json", run)
    run["status"] = "completed" if not run["failures"] else "failed"; runner.write_json(output / "run.json", run); return 0 if run["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
