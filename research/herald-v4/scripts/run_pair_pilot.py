#!/usr/bin/env python3
"""Run a small paired-continuation KV-cache engineering pilot.

The pilot deliberately uses the last prompt token as the pending token.  The
cache is prefetched for ``prompt_ids[:, :-1]`` and each arm then processes the
last prompt token before greedily decoding the continuation.  This is a
variation assay only.  It does not score answers or fit a predictor.
"""

import argparse
import hashlib
import importlib
import json
import os
import platform
import resource
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path


DEFAULT_ACTIONS = (0.0, 0.25, 0.5, 0.75)
SCHEMA_VERSION = "pair_pilot.v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def load_engine(engine_root):
    root = Path(engine_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"engine root is not a directory: {root}")
    engine_path = root / "herald_v3" / "engineering" / "engine.py"
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"engine root does not contain herald_v3/engineering/engine.py: {root}"
        )
    sys.path.insert(0, str(root))
    module = importlib.import_module("herald_v3.engineering.engine")
    return module, root, engine_path


def load_manifest(path):
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    rows = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list or an object with prompts")
    validated = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"manifest row {index} must be an object")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError(f"manifest row {index} needs a non-empty string id")
        if not isinstance(row.get("prompt"), str) or not row["prompt"]:
            raise ValueError(
                f"manifest row {index} needs a non-empty string prompt"
            )
        if "answers" not in row:
            raise ValueError(f"manifest row {index} needs answers metadata")
        cap = row.get("max_new_tokens")
        if cap is not None and (not isinstance(cap, int) or cap < 1):
            raise ValueError(
                f"manifest row {index} max_new_tokens must be a positive integer"
            )
        validated.append(dict(row))
    return manifest_path, validated


def choose_device(requested, torch):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return device


def choose_dtype(name, device, torch):
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"unsupported dtype: {name}") from error


def load_model_and_tokenizer(model_path, device, dtype, transformers):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    kwargs = {
        "local_files_only": True,
        "dtype": dtype,
        "attn_implementation": "sdpa"
        if device.type == "cuda"
        else "sdpa",
    }
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer


def eos_ids(model, tokenizer):
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        value = getattr(model.config, "eos_token_id", None)
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    return {int(item) for item in value}


def process_rss():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return {
        "value": value,
        "unit": "bytes" if sys.platform == "darwin" else "kilobytes",
    }


def sync(engine, device):
    engine._sync_device(device)


def timed_call(engine, device, function):
    sync(engine, device)
    started = time.perf_counter()
    value = function()
    sync(engine, device)
    return value, time.perf_counter() - started


def build_last_prompt_boundary(engine, model, prompt_ids):
    """Build a boundary with a full prompt and one pending prompt token."""
    engine._validate_model_and_input(model, prompt_ids)
    if int(prompt_ids.shape[1]) < 2:
        raise ValueError("last-prompt boundary requires at least two prompt tokens")
    device = engine._model_device(model)
    full_prompt = prompt_ids.to(device)
    prefix = full_prompt[:, :-1]
    prefix_mask = model_input_ones(prefix)
    with engine.torch.no_grad():
        output = model(
            input_ids=prefix,
            attention_mask=prefix_mask,
            use_cache=True,
            return_dict=True,
        )
    source_cache = output.past_key_values
    engine._require_dynamic_cache(source_cache)
    engine._require_full_attention_cache(source_cache)
    boundary_cache = engine.clone_cache(source_cache)
    pending_token_id = int(full_prompt[0, -1].item())
    logical_position = int(full_prompt.shape[1]) - 1
    attention_mask = model_input_ones(prefix)
    rng = engine._capture_rng(device)
    model_fingerprint, model_tensor_count = engine._model_runtime_state_fingerprint(
        model
    )
    state_fingerprint = engine.decoder_state_fingerprint(
        full_prompt,
        (),
        boundary_cache,
        logical_position,
        attention_mask=attention_mask,
        pending_token_id=pending_token_id,
        rng_fingerprint=rng.fingerprint,
        model_state_fingerprint=model_fingerprint,
    )
    boundary = engine.BoundaryState(
        prompt_ids=full_prompt.detach().clone(),
        generated_ids=(),
        cache=boundary_cache,
        pending_token_id=pending_token_id,
        logical_position=logical_position,
        attention_mask=attention_mask.detach().clone(),
        rng_state=rng,
        state_fingerprint=state_fingerprint,
        cache_lengths=engine.cache_lengths(boundary_cache),
        cache_bytes=engine.cache_nbytes(boundary_cache),
        model_state_fingerprint=model_fingerprint,
        model_tensor_count=model_tensor_count,
    )
    return boundary, source_cache


def model_input_ones(value):
    return value.new_ones(value.shape)


def index_digest(kept_indices):
    digest = hashlib.sha256()
    counts = []
    for layer in kept_indices:
        layer_counts = []
        digest.update(len(layer).to_bytes(4, "little"))
        for head in layer:
            layer_counts.append(len(head))
            digest.update(len(head).to_bytes(8, "little"))
            for index in head:
                digest.update(int(index).to_bytes(8, "little", signed=False))
        counts.append(layer_counts)
    return digest.hexdigest(), counts


def compact_compression(engine, compression, boundary):
    index_hash, retained_counts = index_digest(compression.kept_indices)
    expected_lengths = tuple(
        int(length * (1.0 - compression.action.removal_fraction))
        for length in compression.before_lengths
    )
    expected_bytes = 0
    tensors = engine._cache_tensors(boundary.cache)
    for tensor in tensors:
        length = int(tensor.shape[-2])
        kept = int(length * (1.0 - compression.action.removal_fraction))
        expected_bytes += (tensor.numel() // length) * kept * tensor.element_size()
    physical_ok = (
        compression.after_lengths == expected_lengths
        and compression.after_bytes == expected_bytes
    )
    strictly_reduced = (
        compression.action.removal_fraction == 0.0
        or (
            all(
                after < before
                for before, after in zip(
                    compression.before_lengths,
                    compression.after_lengths,
                    strict=True,
                )
            )
            and compression.after_bytes < compression.before_bytes
        )
    )
    return {
        "action": compression.action.to_dict(),
        "before_lengths": list(compression.before_lengths),
        "after_lengths": list(compression.after_lengths),
        "expected_after_lengths": list(expected_lengths),
        "before_bytes": compression.before_bytes,
        "after_bytes": compression.after_bytes,
        "expected_after_bytes": expected_bytes,
        "retained_byte_fraction": (
            compression.after_bytes / compression.before_bytes
            if compression.before_bytes
            else None
        ),
        "before_fingerprint": compression.before_fingerprint,
        "after_fingerprint": compression.after_fingerprint,
        "kept_index_hash": index_hash,
        "kept_index_counts_per_layer_head": retained_counts,
        "compression_seconds": compression.compression_seconds,
        "validation_seconds": compression.validation_seconds,
        "physical_effect_exact": physical_ok,
        "strictly_reduced_for_nonzero_action": strictly_reduced,
    }


def continuation_dict(continuation, text):
    return {
        "token_ids": list(continuation.token_ids),
        "token_count": len(continuation.token_ids),
        "text": text,
        "termination_reason": continuation.termination_reason,
        "forward_seconds": continuation.forward_seconds,
        "first_forward_seconds": continuation.first_forward_seconds,
        "final_cache_lengths": list(continuation.final_cache_lengths),
        "final_cache_bytes": continuation.final_cache_bytes,
        "final_cache_fingerprint": continuation.final_cache_fingerprint,
        "validation_seconds": continuation.validation_seconds,
    }


def chat_messages(prompt, assistant_prefix=None):
    messages = [{"role": "user", "content": prompt}]
    if assistant_prefix is not None:
        if not isinstance(assistant_prefix, str):
            raise ValueError("assistant_prefix must be a string when supplied")
        messages.append({"role": "assistant", "content": assistant_prefix})
    return messages


def render_chat_prompt(tokenizer, prompt, assistant_prefix=None, tokenize=True):
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer must provide a Qwen chat template")
    kwargs = {"tokenize": tokenize}
    if tokenize:
        kwargs["return_tensors"] = "pt"
    if assistant_prefix is None:
        kwargs["add_generation_prompt"] = True
    else:
        kwargs["continue_final_message"] = True
    return tokenizer.apply_chat_template(chat_messages(prompt, assistant_prefix), **kwargs)


def tokenize_chat_prompt(tokenizer, prompt, assistant_prefix=None):
    rendered = render_chat_prompt(tokenizer, prompt, assistant_prefix, tokenize=True)
    if isinstance(rendered, dict):
        rendered = rendered["input_ids"]
    if rendered.ndim == 1:
        rendered = rendered.unsqueeze(0)
    if rendered.ndim != 2 or rendered.shape[0] != 1:
        raise ValueError("chat template must return one batch of token IDs")
    return rendered


def token_identity_checks(tokenizer, row, prompt_ids, prompt, assistant_prefix=None):
    token_ids = [int(item) for item in prompt_ids[0].tolist()]
    actual_count = len(token_ids)
    token_counts = row.get("token_counts")
    count_checks = {}
    if isinstance(token_counts, dict):
        expected = token_counts.get("qwen_chat_prompt_tokens")
        if expected is None:
            expected = token_counts.get("rendered_prompt_tokens")
        if expected is not None:
            count_checks["expected"] = int(expected)
            count_checks["actual"] = actual_count
            count_checks["matches"] = actual_count == int(expected)
            if not count_checks["matches"]:
                raise ValueError(
                    f"chat token count {actual_count} != manifest {expected}"
                )
    hashes = {}
    candidates = {
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "raw_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "token_ids_sha256": hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    rendered_text = render_chat_prompt(tokenizer, prompt, assistant_prefix, tokenize=False)
    candidates["rendered_prompt_sha256"] = hashlib.sha256(
        rendered_text.encode("utf-8")
    ).hexdigest()
    candidates["qwen_chat_prompt_sha256"] = candidates["rendered_prompt_sha256"]
    if assistant_prefix is not None:
        candidates["assistant_prefix_sha256"] = hashlib.sha256(
            assistant_prefix.encode("utf-8")
        ).hexdigest()
    for key in (
        "prompt_sha256",
        "raw_prompt_sha256",
        "token_ids_sha256",
        "rendered_prompt_sha256",
        "qwen_chat_prompt_sha256",
    ):
        expected = row.get(key)
        if expected is None and isinstance(token_counts, dict):
            expected = token_counts.get(key)
        if expected is not None:
            actual = candidates[key]
            hashes[key] = {"expected": expected, "actual": actual, "matches": expected == actual}
            if expected != actual:
                raise ValueError(f"{key} mismatch")
    expected_prefix_hash = row.get("assistant_prefix_sha256")
    if expected_prefix_hash is None and isinstance(token_counts, dict):
        expected_prefix_hash = token_counts.get("assistant_prefix_sha256")
    if expected_prefix_hash is not None:
        if assistant_prefix is None:
            raise ValueError("manifest supplies assistant_prefix_sha256 without assistant_prefix")
        actual_prefix_hash = candidates["assistant_prefix_sha256"]
        hashes["assistant_prefix_sha256"] = {
            "expected": expected_prefix_hash,
            "actual": actual_prefix_hash,
            "matches": expected_prefix_hash == actual_prefix_hash,
        }
        if expected_prefix_hash != actual_prefix_hash:
            raise ValueError("assistant_prefix_sha256 mismatch")
    return {
        "actual_token_count": actual_count,
        "assistant_prefix": assistant_prefix,
        "chat_template_mode": "continue_final_message" if assistant_prefix is not None else "add_generation_prompt",
        "count_check": count_checks,
        "sha256_checks": hashes,
        "token_ids_sha256": candidates["token_ids_sha256"],
        "rendered_prompt_sha256": candidates["rendered_prompt_sha256"],
    }


def model_runtime_identity(engine, model, model_path, transformers, torch):
    signature = engine.model_signature(model)
    return {
        "model_path": str(Path(model_path).expanduser().resolve()),
        "model_signature": signature,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }


def safe_filename(index, row_id):
    clean = "".join(
        char if char.isalnum() or char in ("-", "_") else "_" for char in row_id
    )
    return f"{index:04d}-{clean[:80] or 'prompt'}.json"


def ensure_output_dir(path):
    target = Path(path).expanduser().resolve()
    if target.exists():
        if not target.is_dir():
            raise FileExistsError(f"output path is not a directory: {target}")
        if any(target.iterdir()):
            raise FileExistsError(
                f"output directory must be exclusive and empty: {target}"
            )
    else:
        target.mkdir(parents=True)
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON prompt manifest")
    parser.add_argument(
        "--model", required=True, help="local pinned Transformers model snapshot"
    )
    parser.add_argument(
        "--engine-root",
        required=True,
        help="v3 src directory containing herald_v3/engineering/engine.py",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--reference-path",
        choices=("full_prefill", "shared_boundary"),
        default="full_prefill",
        help="reference continuation protocol",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument(
        "--actions",
        type=float,
        nargs="+",
        default=list(DEFAULT_ACTIONS),
        help="Knorm removal fractions, default: 0 .25 .5 .75",
    )
    return parser.parse_args(argv)


def run(args):
    if args.max_new_tokens < 1 or args.max_prompts < 1:
        raise ValueError("max-new-tokens and max-prompts must be positive")
    actions = []
    for fraction in args.actions:
        if not 0.0 <= fraction < 1.0:
            raise ValueError("action removal fractions must be in [0, 1)")
        if fraction not in actions:
            actions.append(fraction)
    if 0.0 not in actions:
        actions.insert(0, 0.0)

    engine, engine_root, engine_path = load_engine(args.engine_root)
    manifest_path, rows = load_manifest(args.manifest)
    rows = rows[: args.max_prompts]
    output_default = (
        Path("outputs")
        / f"pair-pilot-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_dir = ensure_output_dir(args.output_dir or output_default)
    run_started = datetime.now(UTC).isoformat()
    import torch
    import transformers

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(args.device, torch)
    dtype = choose_dtype(args.dtype, device, torch)
    if device.type == "cuda" and dtype != torch.bfloat16:
        raise ValueError("CUDA pilot runs require bfloat16 for the pinned assay")
    model, tokenizer = load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    model_identity = model_runtime_identity(
        engine, model, args.model, transformers, torch
    )
    runtime_identity = {
        **model_identity,
        "requested_device": args.device,
        "resolved_device": str(device),
        "requested_dtype": args.dtype,
        "resolved_dtype": str(dtype),
        "seed": seed,
        "attention_implementation": str(
            getattr(model.config, "_attn_implementation", "")
        ),
    }
    engine_identity = {
        "engine_root": str(engine_root),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "pilot_source_path": str(Path(__file__).resolve()),
        "pilot_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    manifest_bytes = manifest_path.read_bytes()
    run_record = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "started_at_utc": run_started,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "engine": engine_identity,
        "runtime": runtime_identity,
        "config": {
            "actions": [engine.ActionSpec("knorm", value).to_dict() for value in actions],
            "default_max_new_tokens": args.max_new_tokens,
            "max_prompts": args.max_prompts,
            "selected_prompt_count": len(rows),
            "reference_path": args.reference_path,
            "boundary": {
                "prefill": "prompt_ids[:, :-1]",
                "pending_token": "prompt_ids[:, -1]",
                "generated_ids": [],
                "logical_position": "prompt_length - 1",
                "assistant_prefix": "optional row field rendered with continue_final_message",
            },
        },
        "prompts": [],
        "failures": [],
    }
    write_json(output_dir / "run.json", run_record)

    eos = eos_ids(model, tokenizer)
    for index, row in enumerate(rows):
        record = {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "prompt_index": index,
            "manifest_row": row,
            "prompt": row["prompt"],
            "answers": row["answers"],
            "task": row.get("task"),
            "model": runtime_identity,
            "engine": engine_identity,
            "actions": [
                engine.ActionSpec("knorm", value).to_dict() for value in actions
            ],
            "reference_path": args.reference_path,
        }
        prompt_path = output_dir / safe_filename(index, row["id"])
        try:
            cap = row.get("max_new_tokens", args.max_new_tokens)
            assistant_prefix = row.get("assistant_prefix")
            prompt_ids = tokenize_chat_prompt(tokenizer, row["prompt"], assistant_prefix)
            record["prompt_token_ids"] = [int(item) for item in prompt_ids[0].tolist()]
            record["prompt_length"] = int(prompt_ids.shape[1])
            record["max_new_tokens"] = cap
            token_checks = token_identity_checks(
                tokenizer, row, prompt_ids, row["prompt"], assistant_prefix
            )
            (boundary, source_cache), boundary_seconds = timed_call(
                engine,
                device,
                lambda: build_last_prompt_boundary(engine, model, prompt_ids),
            )
            source_before = engine.cache_fingerprint(source_cache)
            boundary_cache_before = engine.cache_fingerprint(boundary.cache)
            prefill_clone_controls = {
                "equal": engine.cache_tensors_equal(source_cache, boundary.cache),
                "disjoint": engine.cache_storage_independent(
                    source_cache, boundary.cache
                ),
            }
            clone_a = engine.clone_cache(boundary.cache)
            clone_b = engine.clone_cache(boundary.cache)
            clone_controls = {
                "equal_to_source": engine.cache_tensors_equal(
                    boundary.cache, clone_a
                )
                and engine.cache_tensors_equal(boundary.cache, clone_b),
                "a_b_equal": engine.cache_tensors_equal(clone_a, clone_b),
                "source_a_disjoint": engine.cache_storage_independent(
                    boundary.cache, clone_a
                ),
                "source_b_disjoint": engine.cache_storage_independent(
                    boundary.cache, clone_b
                ),
                "a_b_disjoint": engine.cache_storage_independent(clone_a, clone_b),
            }
            record["token_identity"] = token_checks
            record["boundary"] = {
                **boundary.to_dict(),
                "source_cache_fingerprint": source_before,
                "boundary_cache_fingerprint": boundary_cache_before,
                "source_cache_lengths": list(engine.cache_lengths(source_cache)),
                "source_cache_bytes": engine.cache_nbytes(source_cache),
                "prefill_seconds_synchronized": boundary_seconds,
                "prefill_clone_controls": prefill_clone_controls,
                "clone_controls": clone_controls,
            }
            if not all(prefill_clone_controls.values()) or not all(
                clone_controls.values()
            ):
                raise RuntimeError("cache clone controls failed before arm collection")
            partition_audit, partition_seconds = timed_call(
                engine,
                device,
                lambda: engine._greedy_from_prompt(
                    model, prompt_ids, max_new_tokens=cap, eos_ids=eos
                ),
            )
            partition_text = tokenizer.decode(
                list(partition_audit.token_ids), skip_special_tokens=True
            )
            reference_source_unchanged = True
            reference_seconds = partition_seconds
            if args.reference_path == "shared_boundary":
                reference_boundary_before = engine.cache_fingerprint(boundary.cache)
                reference_prefill_before = engine.cache_fingerprint(source_cache)
                reference_arm, reference_seconds = timed_call(
                    engine,
                    device,
                    lambda: engine.continue_from_boundary(
                        model,
                        boundary,
                        max_new_tokens=cap,
                        eos_ids=eos,
                        action=engine.ActionSpec("knorm", 0.0),
                    ),
                )
                reference = reference_arm.continuation
                reference_source_unchanged = (
                    engine.cache_fingerprint(boundary.cache)
                    == reference_boundary_before
                    and engine.cache_fingerprint(source_cache)
                    == reference_prefill_before
                )
            else:
                reference = partition_audit
            reference_text = tokenizer.decode(
                list(reference.token_ids), skip_special_tokens=True
            )
            record["memory"] = {"process_maxrss_before": process_rss()}
            record["reference"] = {
                **continuation_dict(reference, reference_text),
                "reference_path": args.reference_path,
                "wall_seconds_synchronized": reference_seconds,
                "process_maxrss_after": process_rss(),
            }
            record["partition_audit"] = {
                **continuation_dict(partition_audit, partition_text),
                "wall_seconds_synchronized": partition_seconds,
                "matches_reference": partition_audit.token_ids == reference.token_ids,
                "termination_matches_reference": partition_audit.termination_reason
                == reference.termination_reason,
                "source_cache_unchanged": reference_source_unchanged,
            }
            record["arms"] = {}
            first_noop_passed = None
            for action_index, fraction in enumerate(actions):
                action = engine.ActionSpec("knorm", fraction)
                before_arm_source = engine.cache_fingerprint(boundary.cache)
                arm, arm_seconds = timed_call(
                    engine,
                    device,
                    lambda action=action: engine.continue_from_boundary(
                        model,
                        boundary,
                        max_new_tokens=cap,
                        eos_ids=eos,
                        action=action,
                    ),
                )
                after_arm_source = engine.cache_fingerprint(boundary.cache)
                after_prefill_source = engine.cache_fingerprint(source_cache)
                arm_text = tokenizer.decode(
                    list(arm.continuation.token_ids), skip_special_tokens=True
                )
                source_unchanged = (
                    before_arm_source == after_arm_source == boundary_cache_before
                    and after_prefill_source == source_before
                )
                parity = arm.continuation.token_ids == reference.token_ids
                physical = compact_compression(engine, arm.compression, boundary)
                arm_record = {
                    "action": action.to_dict(),
                    "compression": physical,
                    "continuation": continuation_dict(arm.continuation, arm_text),
                    "score_ready": {
                        "text": arm_text,
                        "token_ids": list(arm.continuation.token_ids),
                    },
                    "wall_seconds_synchronized": arm_seconds,
                    "state_copy_seconds": arm.state_copy_seconds,
                    "device_memory_baseline_bytes": arm.device_memory_baseline_bytes,
                    "device_peak_allocated_bytes": arm.device_peak_allocated_bytes,
                    "process_maxrss_after": process_rss(),
                    "boundary_cache_fingerprint_before": before_arm_source,
                    "boundary_cache_fingerprint_after": after_arm_source,
                    "prefill_source_cache_fingerprint_before": source_before,
                    "prefill_source_cache_fingerprint_after": after_prefill_source,
                    "source_cache_unchanged": source_unchanged,
                    "matches_reference": parity,
                    "matches_full_prefill_reference": (
                        arm.continuation.token_ids == partition_audit.token_ids
                    ),
                }
                record["arms"][action.action_id] = arm_record
                action_valid = (
                    source_unchanged
                    and physical["physical_effect_exact"]
                    and physical["strictly_reduced_for_nonzero_action"]
                )
                if fraction == 0.0:
                    first_noop_passed = bool(
                        parity and action_valid and reference_source_unchanged
                    )
                    record["checks"] = {
                        "noop_matches_reference": parity,
                        "noop_source_unchanged": source_unchanged,
                        "noop_physical_effect_exact": physical["physical_effect_exact"],
                        "clone_controls_all_pass": all(clone_controls.values())
                        and all(prefill_clone_controls.values()),
                        "reference_source_unchanged": reference_source_unchanged,
                    }
                    if not first_noop_passed:
                        raise RuntimeError(
                            "first no-op gate failed; see reference and knorm:0 arm"
                        )
                if not action_valid:
                    raise RuntimeError(
                        f"action gate failed for {action.action_id}; see saved arm"
                    )
            record["status"] = "completed"
            record["checks"]["all_action_sources_unchanged"] = all(
                arm["source_cache_unchanged"] for arm in record["arms"].values()
            )
            record["memory"]["process_maxrss_after"] = process_rss()
        except Exception as error:
            record["status"] = "failed"
            record["failure"] = {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            run_record["failures"].append(
                {
                    "prompt_index": index,
                    "id": row.get("id"),
                    "prompt_token_ids": record.get("prompt_token_ids", []),
                    "error": str(error),
                }
            )
        write_json(prompt_path, record)
        run_record["prompts"].append(
            {
                "prompt_index": index,
                "id": row.get("id"),
                "status": record["status"],
                "path": prompt_path.name,
            }
        )
        run_record["completed_prompt_count"] = sum(
            item["status"] == "completed" for item in run_record["prompts"]
        )
        run_record["failed_prompt_count"] = len(run_record["failures"])
        write_json(output_dir / "run.json", run_record)
        if index == 0 and record["status"] == "failed":
            break

    run_record["status"] = "completed" if not run_record["failures"] else "failed"
    run_record["finished_at_utc"] = datetime.now(UTC).isoformat()
    write_json(output_dir / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        print(f"pair pilot failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
