#!/usr/bin/env python3
"""Measure ExpectedAttentionPress features on one existing development row."""

import argparse
import hashlib
import importlib.metadata
import inspect
import math
import sys
import time
import traceback
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as pilot  # noqa: E402


PROMPT_ID = "ruler-pilot-v1-cwe-002"
ACTIONS = (0.25, 0.5, 0.75)
SINK = 4
FUTURE = 128
CAP = 120
SEED = 0


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_features(
    engine, scores, knorm_scores, keys, values, layer_index, fraction
):
    length = int(keys.shape[-2])
    keep = int(length * (1.0 - fraction))
    if not torch.isfinite(scores).all():
        raise ValueError("ExpectedAttentionPress scores contain nonfinite values")
    if not torch.isfinite(knorm_scores).all():
        raise ValueError("Knorm scores contain nonfinite values")
    knorm_native = keys.norm(dim=-1)
    direct_indices = (-knorm_native).topk(keep, dim=-1).indices
    indices = knorm_scores.topk(keep, dim=-1).indices
    if not torch.equal(indices, direct_indices):
        raise AssertionError("KnormPress mask differs from direct key-norm topk")
    knorm = knorm_native.float()
    vnorm = values.norm(dim=-1).float()
    indices = (-knorm).topk(keep, dim=-1).indices
    removed = torch.ones_like(knorm, dtype=torch.bool)
    removed.scatter_(2, indices, False)
    scores = scores.float()
    positions = torch.arange(length, device=keys.device).view(1, 1, -1)
    non_sink = positions >= SINK
    summaries = []
    for head in range(int(keys.shape[1])):
        mask = removed[0, head]
        ns_mask = mask & non_sink[0, 0]
        non_sink_count = int(non_sink[0, 0].sum().item())
        removed_count = int(mask.sum().item())
        ns_removed_count = int(ns_mask.sum().item())
        if non_sink_count <= 0:
            raise ValueError("cache length must exceed the sink count")
        score_ns = scores[0, head, SINK:]
        score_total = float(score_ns.sum().item())
        score_removed = float(score_ns[ns_mask[SINK:]].sum().item())
        removed_indices = positions[0, 0, mask].float()
        if removed_count:
            position_mean = float((removed_indices / max(length - 1, 1)).mean().item())
            last_start = max(0, length - 128)
            last_fraction = float((removed_indices >= last_start).float().mean().item())
            sink_fraction = float((removed_indices < SINK).float().mean().item())
        else:
            position_mean = last_fraction = sink_fraction = 0.0
        knorm_total = float(knorm[0, head].sum().item())
        vnorm_total = float(vnorm[0, head].sum().item())
        if not math.isfinite(score_total) or score_total <= 0:
            raise ValueError("non-sink ExpectedAttentionPress score total is invalid")
        if not math.isfinite(knorm_total) or knorm_total <= 0:
            raise ValueError("Knorm denominator is invalid")
        if not math.isfinite(vnorm_total) or vnorm_total <= 0:
            raise ValueError("V-norm denominator is invalid")
        summaries.append(
            {
                "layer": layer_index,
                "kv_head": head,
                "removal_fraction": fraction,
                "cache_length": length,
                "removed_count": removed_count,
                "non_sink_removed_count": ns_removed_count,
                "ea_removed_mass_non_sink": score_removed / score_total
                if score_total > 0
                else 0.0,
                "ea_removed_fraction_non_sink": ns_removed_count / non_sink_count,
                "ea_removed_excess_non_sink": (
                    score_removed / score_total - ns_removed_count / non_sink_count
                    if score_total > 0
                    else 0.0
                ),
                "ea_removed_mass_non_sink_raw": score_removed,
                "removed_position_mean_normalized": position_mean,
                "removed_last128_fraction": last_fraction,
                "removed_sink_fraction": sink_fraction,
                "removed_knorm_mass": float(knorm[0, head, mask].sum().item())
                / knorm_total
                if knorm_total > 0
                else 0.0,
                "removed_knorm_mass_raw": float(knorm[0, head, mask].sum().item()),
                "removed_vnorm_mass": float(vnorm[0, head, mask].sum().item())
                / vnorm_total
                if vnorm_total > 0
                else 0.0,
                "removed_vnorm_mass_raw": float(vnorm[0, head, mask].sum().item()),
            }
        )
    return summaries


def instrument_prefill(engine, model, prompt_ids, press, device):
    summaries = {str(fraction): [] for fraction in ACTIONS}
    hook_seconds = 0.0
    hooks = []
    prior_rotary = []
    from kvpress import KnormPress

    knorm_press = KnormPress(compression_ratio=0.0)
    layers = model.model.layers

    def make_hook(layer_index):
        def hook(module, args, kwargs, output):
            nonlocal hook_seconds
            hidden_states = kwargs["hidden_states"]
            cache = kwargs["past_key_values"]
            layer_cache = cache.layers[module.layer_idx]
            keys = layer_cache.keys
            values = layer_cache.values
            pilot.sync(engine, device)
            started = time.perf_counter()
            scores = press.score(module, hidden_states, keys, values, None, kwargs)
            knorm_scores = knorm_press.score(
                module, hidden_states, keys, values, None, kwargs
            )
            for fraction in ACTIONS:
                summaries[str(fraction)].extend(
                    action_features(
                        engine,
                        scores,
                        knorm_scores,
                        keys,
                        values,
                        layer_index,
                        fraction,
                    )
                )
            pilot.sync(engine, device)
            hook_seconds += time.perf_counter() - started
            return None

        return hook

    press.post_init_from_model(model)
    try:
        rotary = model.model.rotary_emb
        for layer_index, layer in enumerate(layers):
            module = layer.self_attn
            had_rotary = hasattr(module, "rotary_emb")
            old_rotary = getattr(module, "rotary_emb", None)
            prior_rotary.append((module, had_rotary, old_rotary))
            if not had_rotary or old_rotary is not rotary:
                module.rotary_emb = rotary
            hooks.append(
                module.register_forward_hook(
                    make_hook(layer_index), with_kwargs=True
                )
            )
        (boundary, source_cache), wall_seconds = pilot.timed_call(
            engine,
            device,
            lambda: pilot.build_last_prompt_boundary(engine, model, prompt_ids),
        )
    finally:
        for handle in hooks:
            handle.remove()
        for module, had_rotary, old_rotary in prior_rotary:
            if had_rotary:
                module.rotary_emb = old_rotary
            else:
                delattr(module, "rotary_emb")
    return boundary, source_cache, summaries, hook_seconds, wall_seconds


def prefill_with_memory(engine, device, function):
    pilot.sync(engine, device)
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    value, wall_seconds = pilot.timed_call(engine, device, function)
    peak = int(torch.cuda.max_memory_allocated(device))
    return value, wall_seconds, baseline, peak


def summarize(values):
    if not values:
        return {}
    keys = tuple(key for key, value in values[0].items() if isinstance(value, (int, float)))
    return {
        key: {
            "mean": sum(float(item[key]) for item in values) / len(values),
            "min": min(float(item[key]) for item in values),
            "max": max(float(item[key]) for item in values),
        }
        for key in keys
    }


def run(args):
    output = pilot.ensure_output_dir(args.output)
    partial_path = output / "partial.json"
    partial = {
        "schema_version": "expected_attention_measurement.v1",
        "status": "running",
        "prompt_id": PROMPT_ID,
        "stages": [],
    }
    pilot.write_json(partial_path, partial)
    engine, engine_root, engine_path = pilot.load_engine(args.engine_root)
    manifest_path, rows = pilot.load_manifest(args.manifest)
    row = next((item for item in rows if item["id"] == PROMPT_ID), None)
    if row is None or row.get("max_new_tokens") != CAP:
        raise ValueError(f"manifest must contain {PROMPT_ID} with max_new_tokens={CAP}")

    import torch
    import transformers
    from kvpress import ExpectedAttentionPress
    from kvpress.presses import expected_attention_press, knorm_press, scorer_press

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = pilot.choose_device("cuda", torch)
    model, tokenizer = pilot.load_model_and_tokenizer(
        args.model, device, torch.bfloat16, transformers
    )
    prompt_ids = pilot.tokenize_chat_prompt(tokenizer, row["prompt"])
    token_identity = pilot.token_identity_checks(tokenizer, row, prompt_ids, row["prompt"])
    partial["stages"].append({"name": "tokenization", "token_identity": token_identity})
    pilot.write_json(partial_path, partial)
    press = ExpectedAttentionPress(
        compression_ratio=0.0,
        n_sink=SINK,
        n_future_positions=FUTURE,
        use_covariance=False,
        use_vnorm=True,
        epsilon=0.0,
    )
    plain_result, plain_wall, plain_baseline, plain_peak = prefill_with_memory(
        engine,
        device,
        lambda: pilot.build_last_prompt_boundary(engine, model, prompt_ids),
    )
    plain_boundary, plain_source = plain_result
    partial["stages"].append(
        {
            "name": "plain_prefill",
            "cache_lengths": list(plain_boundary.cache_lengths),
            "cache_fingerprint": engine.cache_fingerprint(plain_boundary.cache),
            "wall_seconds_synchronized": plain_wall,
        }
    )
    pilot.write_json(partial_path, partial)
    instrumented_result, inst_wall, inst_baseline, inst_peak = prefill_with_memory(
        engine,
        device,
        lambda: instrument_prefill(engine, model, prompt_ids, press, device),
    )
    inst_boundary, inst_source, summaries, hook_seconds, _ = instrumented_result
    partial["stages"].append(
        {
            "name": "instrumented_prefill",
            "head_counts": {action: len(items) for action, items in summaries.items()},
            "hook_seconds_synchronized": hook_seconds,
            "cache_fingerprint": engine.cache_fingerprint(inst_boundary.cache),
        }
    )
    pilot.write_json(partial_path, partial)
    cache_equal = engine.cache_tensors_equal(plain_boundary.cache, inst_boundary.cache)
    source_equal = engine.cache_tensors_equal(plain_source, inst_source)
    cache_disjoint = engine.cache_storage_independent(
        plain_boundary.cache, inst_boundary.cache
    )
    plain_source_before = engine.cache_fingerprint(plain_source)
    inst_source_before = engine.cache_fingerprint(inst_source)
    plain_boundary_before = engine.cache_fingerprint(plain_boundary.cache)
    inst_boundary_before = engine.cache_fingerprint(inst_boundary.cache)
    eos = pilot.eos_ids(model, tokenizer)

    def continue_noop(boundary):
        arm = engine.continue_from_boundary(
            model,
            boundary,
            max_new_tokens=CAP,
            eos_ids=eos,
            action=engine.ActionSpec("knorm", 0.0),
        )
        return arm.continuation

    plain_noop = continue_noop(plain_boundary)
    plain_source_after = engine.cache_fingerprint(plain_source)
    plain_boundary_after = engine.cache_fingerprint(plain_boundary.cache)
    inst_noop = continue_noop(inst_boundary)
    inst_source_after = engine.cache_fingerprint(inst_source)
    inst_boundary_after = engine.cache_fingerprint(inst_boundary.cache)
    partial["stages"].append(
        {
            "name": "no_op_continuations",
            "plain_token_count": len(plain_noop.token_ids),
            "instrumented_token_count": len(inst_noop.token_ids),
            "token_ids_equal": plain_noop.token_ids == inst_noop.token_ids,
        }
    )
    pilot.write_json(partial_path, partial)
    source_unchanged = {
        "plain_source": plain_source_before == plain_source_after,
        "instrumented_source": inst_source_before == inst_source_after,
        "plain_boundary": plain_boundary_before == plain_boundary_after,
        "instrumented_boundary": inst_boundary_before == inst_boundary_after,
    }
    output_equal = plain_noop.token_ids == inst_noop.token_ids
    termination_equal = plain_noop.termination_reason == inst_noop.termination_reason
    finite = all(
        value == value and abs(float(value)) != float("inf")
        for action in summaries.values()
        for item in action
        for value in item.values()
        if isinstance(value, (int, float))
    )
    checks = {
        "cache_equal": cache_equal,
        "source_equal": source_equal,
        "boundary_copies_disjoint": cache_disjoint,
        "source_unchanged": all(source_unchanged.values()),
        "noop_token_ids_equal": output_equal,
        "noop_termination_equal": termination_equal,
        "scores_finite": finite,
        "expected_head_count": all(len(items) == len(model.model.layers) * model.config.num_key_value_heads for items in summaries.values()),
    }
    checks["all_checks_pass"] = all(checks.values())
    source_files = [
        Path(inspect.getsourcefile(expected_attention_press)),
        Path(inspect.getsourcefile(scorer_press)),
        Path(inspect.getsourcefile(knorm_press)),
    ]
    source_hashes = {str(path): file_sha256(path) for path in source_files}
    record = {
        "schema_version": "expected_attention_measurement.v1",
        "status": "completed" if checks["all_checks_pass"] else "failed",
        "prompt_id": PROMPT_ID,
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "token_identity": token_identity,
        "seed": SEED,
        "parameters": {
            "actions": list(ACTIONS),
            "n_sink": SINK,
            "n_future_positions": FUTURE,
            "use_covariance": False,
            "use_vnorm": True,
            "epsilon": 0.0,
            "max_new_tokens": CAP,
        },
        "engine": {
            "root": str(engine_root),
            "path": str(engine_path),
            "sha256": file_sha256(engine_path),
        },
        "runtime": {
            "model": pilot.model_runtime_identity(
                engine, model, args.model, transformers, torch
            ),
            "device": str(device),
            "dtype": str(torch.bfloat16),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "kvpress": importlib.metadata.version("kvpress"),
            "kvpress_source_hashes": source_hashes,
        },
        "prefill": {
            "plain_seconds_synchronized": plain_wall,
            "instrumented_seconds_synchronized": inst_wall,
            "summed_hook_seconds_synchronized": hook_seconds,
            "plain_memory_baseline_bytes": plain_baseline,
            "plain_peak_allocated_bytes": plain_peak,
            "instrumented_memory_baseline_bytes": inst_baseline,
            "instrumented_peak_allocated_bytes": inst_peak,
            "instrumented_peak_increment_bytes": inst_peak - inst_baseline,
        },
        "cache_controls": {
            "plain_cache_fingerprint": plain_boundary_before,
            "instrumented_cache_fingerprint": inst_boundary_before,
            "plain_source_fingerprint": plain_source_before,
            "instrumented_source_fingerprint": inst_source_before,
            "cache_equal": cache_equal,
            "source_equal": source_equal,
            "boundary_copies_disjoint": cache_disjoint,
            "source_unchanged": source_unchanged,
        },
        "features": {
            str(action): {
                "head_count": len(items),
                "summary": summarize(items),
                "heads": items,
            }
            for action, items in summaries.items()
        },
        "no_op_transparency": {
            "plain": pilot.continuation_dict(
                plain_noop, tokenizer.decode(list(plain_noop.token_ids), skip_special_tokens=True)
            ),
            "instrumented": pilot.continuation_dict(
                inst_noop, tokenizer.decode(list(inst_noop.token_ids), skip_special_tokens=True)
            ),
            "token_ids_equal": output_equal,
            "termination_equal": termination_equal,
        },
        "checks": checks,
    }
    pilot.write_json(output / "measurement.json", record)
    partial["status"] = record["status"]
    partial["checks"] = checks
    pilot.write_json(partial_path, partial)
    return 0 if checks["all_checks_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        if output.is_dir():
            pilot.write_json(
                output / "failure.json",
                {
                    "schema_version": "expected_attention_measurement.v1",
                    "status": "failed",
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        print(f"measurement failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
