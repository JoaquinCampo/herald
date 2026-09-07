#!/usr/bin/env python3
"""Measure forced seven-digit copy probabilities at the frozen B0 boundary."""

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pair_pilot as runner

PREFIX = "The special magic number for KEY mentioned in the provided text is "
QUESTION = re.compile(
    r"(?:What|what) is the special magic number for (?P<key>.+?) mentioned in the provided text\?"
)
FACT = re.compile(
    r"One of the special magic numbers for (?P<key>[^:\n]+?) is:\s*(?P<value>[0-9]+)\."
)
FRAC = 0.10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flat(value):
    value = value.tolist() if hasattr(value, "tolist") else value
    return value[0] if value and isinstance(value[0], list) else value


def forced_spec(prompt, tokenizer):
    """Derive both forced strings from prompt text only."""
    last = prompt.splitlines()[-1].strip()
    query = QUESTION.fullmatch(last)
    if query is None:
        raise ValueError("final line is not the frozen query")
    context = prompt[: len(prompt) - len(prompt.splitlines()[-1])]
    values = {m.group("key").strip(): m.group("value") for m in FACT.finditer(context)}
    key = query.group("key").strip()
    if key not in values or len(values) < 2:
        raise ValueError("query key or control key is absent")
    control_key = sorted(k for k in values if k != key)[0]
    if (
        len(values[key]) != 7
        or len(values[control_key]) != 7
        or values[key] == values[control_key]
    ):
        raise ValueError("forced values must be distinct seven-digit strings")

    prefix = PREFIX.replace("KEY", key)

    def encode(value):
        text = prefix + value
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offsets = flat(encoded["input_ids"]), flat(encoded["offset_mapping"])
        positions = [
            i
            for i, (start, end) in enumerate(offsets)
            if end > len(prefix) and start < len(text)
        ]
        if len(positions) != 7 or positions != list(
            range(positions[0], positions[0] + 7)
        ):
            raise ValueError("value is not exactly seven native digit tokens")
        digits = [int(ids[i]) for i in positions]
        decoded = [
            tokenizer.decode([i], skip_special_tokens=False).strip() for i in digits
        ]
        if decoded != list(value) or any(
            offsets[i] != (len(prefix) + j, len(prefix) + j + 1)
            for j, i in enumerate(positions)
        ):
            raise ValueError("digit token offsets do not reconstruct the value")
        prefix_ids = flat(tokenizer(prefix, add_special_tokens=False)["input_ids"])
        if prefix_ids != [int(i) for i in ids[: positions[0]]]:
            raise ValueError("prefix tokenization does not match value offsets")
        return prefix_ids, digits

    correct_prefix, correct_digits = encode(values[key])
    control_prefix, control_digits = encode(values[control_key])
    if correct_prefix != control_prefix:
        raise AssertionError("fixed response prefix tokenization differs")
    return {
        "query_key": key,
        "control_key": control_key,
        "correct_value": values[key],
        "control_value": values[control_key],
        "prefix": prefix,
        "prefix_ids": correct_prefix,
        "correct_digit_ids": correct_digits,
        "control_digit_ids": control_digits,
    }


def forward(model, engine, boundary, torch, input_ids, action):
    started = time.perf_counter()
    cache = engine.clone_cache(boundary.cache)
    clone_seconds = time.perf_counter() - started
    compression = engine.compress_knorm(cache, action.removal_fraction)
    device = engine._model_device(model)
    n = int(input_ids.shape[1])
    pos = torch.arange(
        boundary.logical_position, boundary.logical_position + n, device=device
    )
    past = int(cache.get_seq_length())
    # A two-dimensional mask is unsafe after compression: HF can let every
    # forced query attend to every forced key.  Give Qwen an explicit causal
    # mask in physical cache coordinates while retaining logical RoPE positions.
    mask = torch.full(
        (1, 1, n, past + n),
        torch.finfo(model.dtype).min,
        device=device,
        dtype=model.dtype,
    )
    mask[:, :, :, :past] = 0
    for j in range(n):
        mask[:, :, j, past : past + j + 1] = 0
    runner.sync(engine, device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model(
            input_ids=input_ids.to(device),
            attention_mask=mask,
            position_ids=pos.unsqueeze(0),
            cache_position=pos,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
    runner.sync(engine, device)
    forward_seconds = time.perf_counter() - started
    transfer_started = time.perf_counter()
    logits = out.logits.detach().float().cpu()[0]
    transfer_seconds = time.perf_counter() - transfer_started
    return (
        logits,
        compression,
        {
            "clone_seconds": clone_seconds,
            "compression_seconds": compression.compression_seconds,
            "compression_validation_seconds": compression.validation_seconds,
            "forward_seconds": forward_seconds,
            "transfer_seconds": transfer_seconds,
            "transferred_bytes": int(out.logits.numel() * out.logits.element_size()),
        },
    )


def sequential(model, engine, boundary, torch, input_ids, action):
    cache = engine.clone_cache(boundary.cache)
    compression = engine.compress_knorm(cache, action.removal_fraction)
    device = engine._model_device(model)
    rows = []
    for j in range(int(input_ids.shape[1])):
        token = input_ids[:, j : j + 1].to(device)
        pos = torch.tensor([boundary.logical_position + j], device=device)
        mask = torch.ones(
            (1, cache.get_seq_length() + 1), device=device, dtype=torch.long
        )
        with torch.no_grad():
            out = model(
                input_ids=token,
                attention_mask=mask,
                position_ids=pos.unsqueeze(0),
                cache_position=pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
        cache = out.past_key_values
        rows.append(out.logits[:, -1].detach().float().cpu()[0])
    return torch.stack(rows), compression


def digit_metrics(logits, digit_ids, torch, eos_ids):
    if logits.shape[0] < 7 or not bool(torch.isfinite(logits).all()):
        raise ValueError("forced logits are nonfinite or incomplete")
    lp = torch.log_softmax(logits[-7:], dim=-1)
    values, margins, eos_margins = [], [], []
    for row, target in zip(lp, digit_ids, strict=True):
        raw = logits[-7 + len(values)]
        other = raw.clone()
        other[target] = -torch.inf
        values.append(float(row[target]))
        margins.append(float(raw[target] - other.max()))
        eos_margins.append(
            float(raw[target] - max(raw[i] for i in eos_ids)) if eos_ids else None
        )
    return {
        "logprobs": values,
        "correct_minus_best_other_logit_margin": margins,
        "digit_minus_eos_logit_margin": eos_margins,
    }


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


def one(row, args, engine, model, tokenizer, torch):
    spec = forced_spec(row["prompt"], tokenizer)
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    boundary, source = runner.build_last_prompt_boundary(engine, model, prompt_ids)
    boundary_fp, source_fp = (
        engine.cache_fingerprint(boundary.cache),
        engine.cache_fingerprint(source),
    )
    pending = torch.tensor([[boundary.pending_token_id]], dtype=prompt_ids.dtype)
    sequence = torch.tensor(
        [pending[0].tolist() + spec["prefix_ids"] + spec["correct_digit_ids"][:-1]],
        dtype=prompt_ids.dtype,
    )
    control_sequence = torch.tensor(
        [pending[0].tolist() + spec["prefix_ids"] + spec["control_digit_ids"][:-1]],
        dtype=prompt_ids.dtype,
    )
    zero = engine.ActionSpec("knorm", 0.0)
    action = engine.ActionSpec("knorm", FRAC)
    old_path, old = prior_record(args.prior, row["id"])
    prior_ref = (
        old.get("branches", {})
        .get("reference", {})
        .get("continuation", {})
        .get("token_ids", old.get("reference", {}).get("token_ids", []))
    )
    unforced = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=row.get("max_new_tokens", 128),
        eos_ids=runner.eos_ids(model, tokenizer),
        action=zero,
    )
    unforced_action = engine.continue_from_boundary(model, boundary, max_new_tokens=row.get("max_new_tokens", 128), eos_ids=runner.eos_ids(model, tokenizer), action=action)
    ref, ref_comp, ref_cost = forward(model, engine, boundary, torch, sequence, zero)
    act, act_comp, act_cost = forward(model, engine, boundary, torch, sequence, action)
    cref, cref_comp, cref_cost = forward(
        model, engine, boundary, torch, control_sequence, zero
    )
    cact, cact_comp, cact_cost = forward(
        model, engine, boundary, torch, control_sequence, action
    )
    eos = sorted(runner.eos_ids(model, tokenizer))
    prefix_n = len(spec["prefix_ids"])
    correct = {
        "reference": digit_metrics(
            ref[prefix_n:], spec["correct_digit_ids"], torch, eos
        ),
        "action": digit_metrics(act[prefix_n:], spec["correct_digit_ids"], torch, eos),
    }
    control = {
        "reference": digit_metrics(
            cref[prefix_n:], spec["control_digit_ids"], torch, eos
        ),
        "action": digit_metrics(cact[prefix_n:], spec["control_digit_ids"], torch, eos),
    }
    if args.device == "cpu":
        sref, _ = sequential(model, engine, boundary, torch, sequence, zero)
        sact, _ = sequential(model, engine, boundary, torch, sequence, action)
        if not bool(torch.allclose(ref, sref, atol=1e-5, rtol=1e-5)) or not bool(
            torch.allclose(act, sact, atol=1e-5, rtol=1e-5)
        ):
            raise RuntimeError("batched and sequential teacher forcing disagree")
        sequential_check = True
    else:
        sequential_check = None
    ref_compact, act_compact = (
        runner.compact_compression(engine, ref_comp, boundary),
        runner.compact_compression(engine, act_comp, boundary),
    )
    unforced_ids = list(unforced.continuation.token_ids)
    prior_action_cont = old["branches"]["action"]["continuation"]
    prior_ref_cont = old["branches"]["reference"]["continuation"]
    old_action = old.get("branches", {}).get("action", {}).get("compression")
    if old_action is None:
        old_action = old.get("arms", {}).get("knorm:0.1", {}).get("compression")
    if old_action is None:
        raise ValueError("prior record has no native knorm:0.1 compression")
    checks = {
        "source_boundary_unchanged": engine.cache_fingerprint(boundary.cache)
        == boundary_fp
        and engine.cache_fingerprint(source) == source_fp,
        "source_boundary_equal_disjoint": engine.cache_tensors_equal(
            boundary.cache, source
        )
        and engine.cache_storage_independent(boundary.cache, source),
        "action_physical_effect": act_compact["physical_effect_exact"]
        and act_compact["strictly_reduced_for_nonzero_action"],
        "prior_native_mask_hash_match": act_compact["kept_index_hash"]
        == old_action["kept_index_hash"],
        "prior_first_token_match": bool(prior_ref)
        and int(unforced_ids[0]) == int(prior_ref[0]),
        "unforced_prior_continuation_exact": unforced_ids == list(prior_ref),
        "unforced_action_prior_exact": list(unforced_action.continuation.token_ids) == prior_action_cont["token_ids"],
        "reference_termination_exact": unforced.continuation.termination_reason == prior_ref_cont["termination_reason"],
        "action_termination_exact": unforced_action.continuation.termination_reason == prior_action_cont["termination_reason"],
        "batched_first_reference_exact": int(ref[0].argmax()) == prior_ref[0],
        "batched_first_action_exact": int(act[0].argmax()) == prior_action_cont["token_ids"][0],
        "control_masks_exact": runner.index_digest(cref_comp.kept_indices)[0] == ref_compact["kept_index_hash"] and runner.index_digest(cact_comp.kept_indices)[0] == act_compact["kept_index_hash"],
        "batched_sequential_teacher_force": sequential_check,
        "finite": all(bool(torch.isfinite(x).all()) for x in (ref, act, cref, cact)),
    }
    checks["all_checks_pass"] = all(v for v in checks.values() if v is not None)
    if not checks["all_checks_pass"]:
        raise RuntimeError("digit-copy gate failed: " + json.dumps(checks))
    z = (
        sum(
            a - b
            for a, b in zip(
                correct["reference"]["logprobs"],
                correct["action"]["logprobs"],
                strict=True,
            )
        )
        / 7
    )
    zc = (
        sum(
            a - b
            for a, b in zip(
                control["reference"]["logprobs"],
                control["action"]["logprobs"],
                strict=True,
            )
        )
        / 7
    )
    return {
        "schema_version": "digit_copy_probe.v1",
        "status": "completed",
        "spec": spec,
        "boundary": boundary.to_dict(),
        "native_masks": {
            "reference": ref_compact["kept_index_hash"],
            "action": act_compact["kept_index_hash"],
            "control_reference": runner.compact_compression(
                engine, cref_comp, boundary
            )["kept_index_hash"],
            "control_action": runner.compact_compression(engine, cact_comp, boundary)[
                "kept_index_hash"
            ],
        },
        "correct": correct,
        "control": control,
        "z": z,
        "z_control": zc,
        "reference_nll": -sum(correct["reference"]["logprobs"]) / 7,
        "cost": {
            "correct_reference": ref_cost,
            "correct_action": act_cost,
            "control_reference": cref_cost,
            "control_action": cact_cost,
        },
        "checks": checks,
        "prior": {
            "path": str(old_path), "sha256": sha(old_path),
            "reference_first_token_ids": list(prior_ref[:7]),
        },
        "compression": {"reference": ref_compact, "action": act_compact},
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--engine-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    args = p.parse_args(argv)
    output = runner.ensure_output_dir(args.output)
    engine, _, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if args.ids:
        available = {r["id"] for r in rows}
        if not set(args.ids) <= available:
            raise ValueError("requested id is absent from manifest")
        rows = [r for r in rows if r["id"] in set(args.ids)]
    if not rows:
        raise ValueError("manifest selection is empty")
    import torch
    import transformers

    torch.manual_seed(0)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    run = {
        "schema_version": "digit_copy_probe.v1",
        "status": "running",
        "manifest_sha256": sha(manifest_path),
        "source_hashes": {
            str(Path(__file__).resolve()): sha(Path(__file__).resolve()),
            str(engine_path): sha(engine_path),
            str(Path(runner.__file__).resolve()): sha(Path(runner.__file__).resolve()),
        },
        "runtime": runner.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "config": {
            "removal_fraction": FRAC,
            "response_prefix_template": PREFIX,
            "device": str(device),
            "dtype": str(dtype),
        },
        "prompts": [],
        "failures": [],
    }
    runner.write_json(output / "run.json", run)
    for i, row in enumerate(rows):
        path = output / runner.safe_filename(i, row["id"])
        try:
            observation = one(row, args, engine, model, tokenizer, torch)
            feature_path = path.with_name(path.stem + ".features.json")
            runner.write_json(feature_path, observation)
            _, old = prior_record(args.prior, row["id"])
            record = {
                "schema_version": "digit_copy_probe.v1",
                "status": "completed",
                "manifest_row": row,
                "observation": observation,
                "features_sha256": sha(feature_path),
                "signed_loss": old.get("signed_loss"),
            }
            if record["signed_loss"] is None:
                raise ValueError("prior record has no signed_loss label")
            if record["features_sha256"] != sha(feature_path):
                raise RuntimeError("feature file changed before label join")
            runner.write_json(path, record)
            status = "completed"
        except Exception as error:
            runner.write_json(
                path,
                {
                    "schema_version": "digit_copy_probe.v1",
                    "status": "failed",
                    "manifest_row": row,
                    "failure": {
                        "type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                },
            )
            run["failures"].append({"id": row["id"], "error": str(error)})
            status = "failed"
        run["prompts"].append({"id": row["id"], "path": path.name, "status": status})
        runner.write_json(output / "run.json", run)
        if status == "failed":
            break
    run["status"] = "failed" if run["failures"] else "completed"
    runner.write_json(output / "run.json", run)
    return int(bool(run["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
