#!/usr/bin/env python3
"""Prove the ExpectedAttentionPress boundary action on one tiny CPU case.

This is an experiment-local technical proof. It keeps the plain B0 boundary as
the reference, collects native ExpectedAttentionPress scores in an independent
prefill without mutating that cache, gathers a clone before the pending token,
and verifies continuation, cache, timing, and determinism controls.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as runner  # noqa: E402

DEFAULT_MODEL = Path("/private/tmp/herald-v4-ea-proof/model")
DEFAULT_MANIFEST = Path("/private/tmp/herald-v4-ea-proof/manifest.json")
DEFAULT_ENGINE_ROOT = Path(
    "/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/src"
)
DEFAULT_OUTPUT = Path(
    "/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v4/results/ea-boundary-cpu"
)
REMOVAL_FRACTION = 0.10
N_SINK = 4
N_FUTURE_POSITIONS = 512
USE_COVARIANCE = True
USE_VNORM = True
EPSILON = 0.0
SEED = 0
MAX_NEW_TOKENS = 4
SCHEMA_VERSION = "ea_boundary_cpu.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_engine(engine_root):
    root = Path(engine_root).expanduser().resolve()
    engine_path = root / "herald_v3" / "engineering" / "engine.py"
    if not engine_path.is_file():
        raise FileNotFoundError(f"engine.py not found below {root}")
    sys.path.insert(0, str(root))
    module = __import__("herald_v3.engineering.engine", fromlist=["engine"])
    return module, root, engine_path


def load_row(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must contain a non-empty prompt list")
    row = rows[0]
    if not isinstance(row, dict) or not row.get("prompt"):
        raise ValueError("first manifest row needs a prompt")
    return path, row


def load_model(model_path, torch, transformers):
    device = torch.device("cpu")
    dtype = torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)
    model.eval()
    return model, tokenizer, device, dtype


def render_prompt(tokenizer, prompt):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer must provide a chat template")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered.ndim == 1:
        rendered = rendered.unsqueeze(0)
    if rendered.ndim != 2 or rendered.shape[0] != 1:
        raise ValueError("chat template must produce batch-1 token IDs")
    return rendered


def sync(device):
    if device.type == "cuda":
        import torch

        torch.cuda.synchronize(device)
    elif device.type == "mps":
        import torch

        torch.mps.synchronize()


def timed(device, function):
    sync(device)
    started = time.perf_counter()
    value = function()
    sync(device)
    return value, time.perf_counter() - started


def build_boundary(engine, model, prompt_ids):
    return runner.build_last_prompt_boundary(engine, model, prompt_ids)


def cache_tensors(engine, cache):
    return engine._cache_tensors(cache)


def cache_summary(engine, cache):
    return {
        "lengths": list(engine.cache_lengths(cache)),
        "bytes": engine.cache_nbytes(cache),
        "fingerprint": engine.cache_fingerprint(cache),
    }


def clone_cache(engine, cache):
    return engine.clone_cache(cache)


def install_rotary(model):
    rotary = model.model.rotary_emb
    previous = []
    for layer in model.model.layers:
        module = layer.self_attn
        had_value = hasattr(module, "rotary_emb")
        old_value = getattr(module, "rotary_emb", None)
        previous.append((module, had_value, old_value))
        if not had_value or old_value is not rotary:
            module.rotary_emb = rotary
    return previous


def restore_rotary(previous):
    for module, had_value, old_value in previous:
        if had_value:
            module.rotary_emb = old_value
        else:
            delattr(module, "rotary_emb")


def mask_digest(indices_by_layer):
    digest = hashlib.sha256()
    for layer_indices in indices_by_layer:
        digest.update(len(layer_indices).to_bytes(4, "little"))
        for head_indices in layer_indices:
            digest.update(len(head_indices).to_bytes(8, "little"))
            for position in head_indices:
                digest.update(int(position).to_bytes(8, "little"))
    return digest.hexdigest()


def score_stats(scores):
    values = scores.detach().float()
    if not torch_isfinite(values):
        raise RuntimeError("ExpectedAttentionPress returned nonfinite scores")
    return {
        "shape": list(values.shape),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "mean": float(values.mean().item()),
        "finite": True,
    }


def torch_isfinite(value):
    return bool(value.isfinite().all().item())


def instrument_prefill(engine, model, prompt_ids, press, device):
    scores_by_layer = []
    hook_seconds = 0.0
    hooks = []

    def make_hook(layer_index):
        def hook(module, args, kwargs, output):
            nonlocal hook_seconds
            cache = kwargs["past_key_values"]
            layer_cache = cache.layers[module.layer_idx]
            keys = layer_cache.keys
            values = layer_cache.values
            sync(device)
            started = time.perf_counter()
            scores = press.score(
                module,
                kwargs["hidden_states"],
                keys,
                values,
                None,
                kwargs,
            )
            sync(device)
            hook_seconds += time.perf_counter() - started
            scores_by_layer.append((layer_index, scores.detach().clone()))
            return None

        return hook

    press.post_init_from_model(model)
    previous_rotary = install_rotary(model)
    for index, layer in enumerate(model.model.layers):
        hooks.append(
            layer.self_attn.register_forward_hook(
                make_hook(index), with_kwargs=True
            )
        )
    try:
        boundary_source, wall_seconds = timed(
            device, lambda: build_boundary(engine, model, prompt_ids)
        )
    finally:
        for handle in hooks:
            handle.remove()
        restore_rotary(previous_rotary)
    scores_by_layer.sort(key=lambda item: item[0])
    if not scores_by_layer:
        raise RuntimeError("instrumented prefill collected no scores")
    return boundary_source, scores_by_layer, hook_seconds, wall_seconds


def direct_indices(scores, removal_fraction):
    length = int(scores.shape[-1])
    kept = int(length * (1.0 - removal_fraction))
    if kept < 1 or kept >= length:
        raise ValueError(
            f"removal fraction gives invalid keep count {kept} for {length}"
        )
    if not torch_isfinite(scores):
        raise RuntimeError("cannot select from nonfinite scores")
    return scores.topk(kept, dim=-1).indices


def apply_ea_action(engine, base_cache, scores_by_layer, removal_fraction):
    action_cache = clone_cache(engine, base_cache)
    kept_indices = []
    expected_layers = len(action_cache.layers)
    if len(scores_by_layer) != expected_layers:
        raise RuntimeError(
            "score layer count does not match cache layer count"
        )
    for layer_index, scores in scores_by_layer:
        if layer_index >= expected_layers:
            raise RuntimeError("score layer index exceeds cache layer count")
        layer = action_cache.layers[layer_index]
        indices = direct_indices(scores, removal_fraction)
        expanded = indices.unsqueeze(-1).expand(
            -1, -1, -1, layer.keys.shape[-1]
        )
        layer.keys = layer.keys.gather(2, expanded).contiguous()
        layer.values = layer.values.gather(2, expanded).contiguous()
        kept_indices.append(
            [
                [int(value) for value in row]
                for row in indices[0].detach().cpu().tolist()
            ]
        )
    return action_cache, kept_indices


def direct_gather_matches(
    engine, base_cache, action_cache, scores_by_layer, removal_fraction
):
    checks = []
    for layer_index, scores in scores_by_layer:
        base_layer = base_cache.layers[layer_index]
        action_layer = action_cache.layers[layer_index]
        indices = direct_indices(scores, removal_fraction)
        expanded = indices.unsqueeze(-1).expand(
            -1, -1, -1, base_layer.keys.shape[-1]
        )
        expected_keys = base_layer.keys.gather(2, expanded).contiguous()
        expected_values = base_layer.values.gather(2, expanded).contiguous()
        checks.append(
            {
                "layer": layer_index,
                "indices_shape": list(indices.shape),
                "keys_equal": bool(action_layer.keys.equal(expected_keys)),
                "values_equal": bool(
                    action_layer.values.equal(expected_values)
                ),
            }
        )
    return checks


def continuation(engine, model, boundary, cache, cap, eos):
    return engine._continue_cache(
        model,
        boundary,
        cache,
        max_new_tokens=cap,
        eos_ids=eos,
        first_logits_observer=None,
    )


def continuation_record(tokenizer, result):
    return {
        "token_ids": list(result.token_ids),
        "termination_reason": result.termination_reason,
        "text": tokenizer.decode(
            list(result.token_ids), skip_special_tokens=True
        ),
        "forward_seconds": result.forward_seconds,
        "first_forward_seconds": result.first_forward_seconds,
        "final_cache_lengths": list(result.final_cache_lengths),
        "final_cache_bytes": result.final_cache_bytes,
        "final_cache_fingerprint": result.final_cache_fingerprint,
    }


def source_hashes():
    from kvpress.presses import (
        base_press,
        expected_attention_press,
        scorer_press,
    )

    paths = {
        "expected_attention_press": Path(
            inspect.getsourcefile(expected_attention_press)
        ),
        "scorer_press": Path(inspect.getsourcefile(scorer_press)),
        "base_press": Path(inspect.getsourcefile(base_press)),
    }
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def run(args):
    import torch
    import transformers
    from kvpress import ExpectedAttentionPress

    if args.device != "cpu":
        raise ValueError("this technical proof is CPU-only")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    partial_path = output / "partial.json"
    partial = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "stages": [],
    }
    write_json(partial_path, partial)

    engine, engine_root, engine_path = load_engine(args.engine_root)
    manifest_path, row = load_row(args.manifest)
    model, tokenizer, device, dtype = load_model(
        args.model, torch, transformers
    )
    torch.manual_seed(SEED)
    prompt_ids = render_prompt(tokenizer, row["prompt"])
    prompt_length = int(prompt_ids.shape[1])
    if prompt_length < 2:
        raise ValueError("prompt must contain at least two tokens")
    token_ids = [int(value) for value in prompt_ids[0].tolist()]
    partial["stages"].append(
        {
            "name": "tokenization",
            "prompt_length": prompt_length,
            "token_ids": token_ids,
        }
    )
    write_json(partial_path, partial)

    press = ExpectedAttentionPress(
        compression_ratio=0.0,
        n_future_positions=N_FUTURE_POSITIONS,
        n_sink=N_SINK,
        use_covariance=USE_COVARIANCE,
        use_vnorm=USE_VNORM,
        epsilon=EPSILON,
    )
    ratio_zero_keys = torch.zeros((1, 1, 3, 8), dtype=dtype)
    ratio_zero_values = torch.ones_like(ratio_zero_keys)
    original_score = press.score

    def score_must_not_run(*unused_args, **unused_kwargs):
        raise AssertionError("native ratio-zero compress unexpectedly scored")

    press.score = score_must_not_run
    try:
        passthrough_keys, passthrough_values = press.compress(
            None,
            None,
            ratio_zero_keys,
            ratio_zero_values,
            None,
            {},
        )
    finally:
        press.score = original_score
    ratio_zero_compress_passthrough = (
        passthrough_keys is ratio_zero_keys
        and passthrough_values is ratio_zero_values
    )
    plain_result, plain_seconds = timed(
        device, lambda: build_boundary(engine, model, prompt_ids)
    )
    plain_boundary, plain_source = plain_result
    plain_summary = cache_summary(engine, plain_boundary.cache)
    plain_source_summary = cache_summary(engine, plain_source)
    partial["stages"].append(
        {
            "name": "plain_prefill",
            "seconds": plain_seconds,
            "boundary": plain_summary,
        }
    )
    write_json(partial_path, partial)

    instrumented_result = instrument_prefill(
        engine, model, prompt_ids, press, device
    )
    inst_boundary, scores_by_layer, hook_seconds, inst_seconds = (
        instrumented_result
    )
    inst_boundary, inst_source = inst_boundary
    inst_summary = cache_summary(engine, inst_boundary.cache)
    inst_source_summary = cache_summary(engine, inst_source)
    partial["stages"].append(
        {
            "name": "instrumented_prefill",
            "seconds": inst_seconds,
            "hook_seconds": hook_seconds,
            "boundary": inst_summary,
        }
    )
    write_json(partial_path, partial)

    deterministic_result = instrument_prefill(
        engine, model, prompt_ids, press, device
    )
    det_boundary_pair, det_scores, det_hook_seconds, det_seconds = (
        deterministic_result
    )
    det_boundary, det_source = det_boundary_pair

    cache_equal = engine.cache_tensors_equal(
        plain_boundary.cache, inst_boundary.cache
    )
    source_equal = engine.cache_tensors_equal(plain_source, inst_source)
    boundary_disjoint = engine.cache_storage_independent(
        plain_boundary.cache, inst_boundary.cache
    )
    source_disjoint = engine.cache_storage_independent(
        plain_source, plain_boundary.cache
    ) and engine.cache_storage_independent(inst_source, inst_boundary.cache)
    score_layer_count = len(scores_by_layer) == len(model.model.layers)
    expected_heads = int(model.config.num_key_value_heads)
    score_shape_checks = all(
        tuple(score.shape[:2]) == (1, expected_heads)
        for _, score in scores_by_layer
    )
    score_finite = all(torch_isfinite(score) for _, score in scores_by_layer)
    score_records = []
    for layer_index, scores in scores_by_layer:
        score_records.append({"layer": layer_index, **score_stats(scores)})

    (action_cache, kept_indices), action_transform_seconds = timed(
        device,
        lambda: apply_ea_action(
            engine, plain_boundary.cache, scores_by_layer, REMOVAL_FRACTION
        ),
    )
    gather_checks = direct_gather_matches(
        engine,
        plain_boundary.cache,
        action_cache,
        scores_by_layer,
        REMOVAL_FRACTION,
    )
    gather_equal = all(
        item["keys_equal"] and item["values_equal"] for item in gather_checks
    )
    mask_hash = mask_digest(kept_indices)
    det_indices = []
    for _, scores in det_scores:
        indices = direct_indices(scores, REMOVAL_FRACTION)
        det_indices.append(
            [
                [int(value) for value in row]
                for row in indices[0].detach().cpu().tolist()
            ]
        )
    det_mask_hash = mask_digest(det_indices)
    deterministic_masks = (
        kept_indices == det_indices and mask_hash == det_mask_hash
    )
    sink_positions = set(range(N_SINK))
    sink_retained = all(
        sink_positions.issubset(set(head))
        for layer in kept_indices
        for head in layer
    )
    action_lengths = list(engine.cache_lengths(action_cache))
    action_bytes = engine.cache_nbytes(action_cache)
    expected_keep = int(
        plain_summary["lengths"][0] * (1.0 - REMOVAL_FRACTION)
    )
    expected_bytes = sum(
        (tensor.numel() // int(tensor.shape[-2]))
        * expected_keep
        * tensor.element_size()
        for tensor in cache_tensors(engine, plain_boundary.cache)
    )
    physical_effect = (
        action_lengths == [expected_keep] * len(action_lengths)
        and action_bytes == expected_bytes
        and action_bytes < plain_summary["bytes"]
    )

    eos_value = getattr(tokenizer, "eos_token_id", None)
    if eos_value is None:
        eos_value = getattr(model.config, "eos_token_id", None)
    if isinstance(eos_value, int):
        eos = {eos_value}
    elif eos_value is None:
        eos = set()
    else:
        eos = {int(value) for value in eos_value}

    plain_before = engine.cache_fingerprint(plain_boundary.cache)
    inst_before = engine.cache_fingerprint(inst_boundary.cache)
    plain_source_before = engine.cache_fingerprint(plain_source)
    inst_source_before = engine.cache_fingerprint(inst_source)
    uninterrupted, uninterrupted_seconds = timed(
        device,
        lambda: engine._greedy_from_prompt(
            model, prompt_ids, max_new_tokens=args.max_new_tokens, eos_ids=eos
        ),
    )
    plain_noop_cache = clone_cache(engine, plain_boundary.cache)
    inst_noop_cache = clone_cache(engine, inst_boundary.cache)
    plain_noop, plain_noop_seconds = timed(
        device,
        lambda: continuation(
            engine,
            model,
            plain_boundary,
            plain_noop_cache,
            args.max_new_tokens,
            eos,
        ),
    )
    inst_noop, inst_noop_seconds = timed(
        device,
        lambda: continuation(
            engine,
            model,
            inst_boundary,
            inst_noop_cache,
            args.max_new_tokens,
            eos,
        ),
    )
    action_continuation, action_seconds = timed(
        device,
        lambda: continuation(
            engine,
            model,
            plain_boundary,
            action_cache,
            args.max_new_tokens,
            eos,
        ),
    )
    no_op_exact = (
        uninterrupted.token_ids == plain_noop.token_ids == inst_noop.token_ids
        and uninterrupted.termination_reason
        == plain_noop.termination_reason
        == inst_noop.termination_reason
    )
    source_unchanged = (
        engine.cache_fingerprint(plain_source) == plain_source_before
        and engine.cache_fingerprint(inst_source) == inst_source_before
        and engine.cache_fingerprint(plain_boundary.cache) == plain_before
        and engine.cache_fingerprint(inst_boundary.cache) == inst_before
    )
    action_disjoint = engine.cache_storage_independent(
        plain_boundary.cache, action_cache
    )

    gates = {
        "plain_instrumented_boundary_exact": cache_equal and source_equal,
        "boundary_and_source_storage_disjoint": boundary_disjoint
        and source_disjoint
        and action_disjoint,
        "native_scores_have_expected_layers_heads_and_are_finite": (
            score_layer_count
            and score_shape_checks
            and score_finite
            and ratio_zero_compress_passthrough
        ),
        "native_mask_reproducible": deterministic_masks,
        "native_gather_exact": gather_equal,
        "sink_positions_retained": sink_retained,
        "physical_cache_effect_exact": physical_effect,
        "no_op_continuation_exact": no_op_exact,
        "source_and_reference_boundaries_unchanged": source_unchanged,
        "acted_clone_continues": len(action_continuation.token_ids) > 0,
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if all(gates.values()) else "failed",
        "technical_proof_only": True,
        "claim_boundary": (
            "CPU tiny-case engineering proof; no predictor or quality claim"
        ),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "row_id": row.get("id"),
        },
        "prompt": {
            "text": row["prompt"],
            "token_ids": token_ids,
            "length": prompt_length,
        },
        "config": {
            "removal_fraction": REMOVAL_FRACTION,
            "n_sink": N_SINK,
            "n_future_positions": N_FUTURE_POSITIONS,
            "use_covariance": USE_COVARIANCE,
            "use_vnorm": USE_VNORM,
            "epsilon": EPSILON,
            "max_new_tokens": args.max_new_tokens,
            "seed": SEED,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "kvpress": importlib.metadata.version("kvpress"),
            "device": str(device),
            "dtype": str(dtype),
        },
        "engine": {
            "root": str(engine_root),
            "path": str(engine_path),
            "sha256": sha256_file(engine_path),
        },
        "kvpress_sources": source_hashes(),
        "prefill": {
            "plain": {
                "seconds": plain_seconds,
                **plain_summary,
                "source": plain_source_summary,
            },
            "instrumented": {
                "seconds": inst_seconds,
                "hook_seconds": hook_seconds,
                **inst_summary,
                "source": inst_source_summary,
            },
            "deterministic_repeat": {
                "seconds": det_seconds,
                "hook_seconds": det_hook_seconds,
                "boundary_equal": engine.cache_tensors_equal(
                    inst_boundary.cache, det_boundary.cache
                ),
                "source_equal": engine.cache_tensors_equal(
                    inst_source, det_source
                ),
            },
            "two_full_prefills": True,
        },
        "scores": {
            "layers": score_records,
            "layer_count": len(scores_by_layer),
            "expected_kv_heads": expected_heads,
            "ratio_zero_compress_passthrough": (
                ratio_zero_compress_passthrough
            ),
        },
        "action": {
            "mask_hash": mask_hash,
            "repeat_mask_hash": det_mask_hash,
            "kept_indices": kept_indices,
            "kept_count_per_layer_head": [
                [len(head) for head in layer] for layer in kept_indices
            ],
            "logical_position_before_pending": (
                plain_boundary.logical_position
            ),
            "cache": {
                "lengths": action_lengths,
                "bytes": action_bytes,
                "fingerprint": engine.cache_fingerprint(action_cache),
            },
            "expected_keep": expected_keep,
            "expected_bytes": expected_bytes,
            "direct_gather_checks": gather_checks,
        },
        "continuations": {
            "uninterrupted": {
                "wall_seconds": uninterrupted_seconds,
                **continuation_record(tokenizer, uninterrupted),
            },
            "plain_noop": {
                "wall_seconds": plain_noop_seconds,
                **continuation_record(tokenizer, plain_noop),
            },
            "instrumented_noop": {
                "wall_seconds": inst_noop_seconds,
                **continuation_record(tokenizer, inst_noop),
            },
            "ea_action": {
                "wall_seconds": action_seconds,
                **continuation_record(tokenizer, action_continuation),
            },
            "no_op_exact": no_op_exact,
        },
        "cache_controls": {
            "plain_instrumented_boundary_equal": cache_equal,
            "plain_instrumented_source_equal": source_equal,
            "boundary_storage_disjoint": boundary_disjoint,
            "source_storage_disjoint": source_disjoint,
            "action_storage_disjoint": action_disjoint,
            "source_and_boundaries_unchanged": source_unchanged,
            "plain_boundary_before": plain_before,
            "plain_boundary_after": engine.cache_fingerprint(
                plain_boundary.cache
            ),
            "instrumented_boundary_before": inst_before,
            "instrumented_boundary_after": engine.cache_fingerprint(
                inst_boundary.cache
            ),
            "plain_source_before": plain_source_before,
            "plain_source_after": engine.cache_fingerprint(plain_source),
            "instrumented_source_before": inst_source_before,
            "instrumented_source_after": engine.cache_fingerprint(
                inst_source
            ),
        },
        "timings": {
            "plain_prefill_seconds": plain_seconds,
            "instrumented_prefill_seconds": inst_seconds,
            "instrumented_hook_seconds": hook_seconds,
            "repeat_instrumented_prefill_seconds": det_seconds,
            "action_clone_and_gather_seconds": action_transform_seconds,
        },
        "gates": gates,
    }
    write_json(output / "0000-tiny-ea-boundary.json", record)
    partial["status"] = record["status"]
    partial["gates"] = gates
    write_json(partial_path, partial)
    write_json(
        output / "run.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": record["status"],
            "record": "0000-tiny-ea-boundary.json",
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "finished_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    if not all(gates.values()):
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"EA CPU acceptance gates failed: {failed}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--engine-root", default=str(DEFAULT_ENGINE_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_json(
            output / "failure.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        print(
            f"EA CPU proof failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
