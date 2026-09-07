#!/usr/bin/env python3
"""Run a bounded, oracle-only needle-span rescue diagnostic."""

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as runner  # noqa: E402
from score_ruler_pilot import (  # noqa: E402
    OFFICIAL_CONSTANTS,
    OFFICIAL_EVAL,
    score_prediction,
)

TARGET_IDS = (
    "ruler-ea-dev-v1-niah_single_2-000",
    "ruler-ea-dev-v1-niah_single_2-004",
    "ruler-ea-dev-v1-niah_single_2-011",
    "ruler-ea-dev-v1-niah_single_2-001",
)
REMOVAL_FRACTION = 0.10
SINK = 4
SEED = 0


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_indices(indices):
    return [
        [[int(value) for value in head] for head in layer]
        for layer in indices
    ]


def span_retention_fractions(kept_indices, target_positions):
    target = set(target_positions)
    if not target:
        raise ValueError("needle span must contain at least one token")
    return [
        [
            sum(index in target for index in head) / len(target)
            for head in layer
        ]
        for layer in kept_indices
    ]


def score_branch(tokenizer, continuation, answers):
    text = tokenizer.decode(
        list(continuation.token_ids), skip_special_tokens=True
    )
    return {
        "continuation": runner.continuation_dict(continuation, text),
        "score": score_prediction(text, answers),
    }


def find_prior_record(prior_dirs, row_id):
    matches = []
    for directory in prior_dirs:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(
                f"prior result directory is missing: {root}"
            )
        for path in root.glob("*.json"):
            if path.name == "run.json":
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if record.get("manifest_row", {}).get("id") == row_id:
                matches.append((path, record))
    if len(matches) != 1:
        raise ValueError(
            f"expected one prior record for {row_id}, found {len(matches)}"
        )
    return matches[0]


def locate_span(tokenizer, prompt_ids, prompt, answer):
    sentence_pattern = re.compile(
        r"One of the special magic numbers for [^:\n]+ is: "
        + re.escape(answer)
        + r"\."
    )
    matches = list(sentence_pattern.finditer(prompt))
    if len(matches) != 1 or prompt.count(answer) != 1:
        raise ValueError(
            "needle sentence and answer must each occur exactly once"
        )
    sentence = matches[0].group(0)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_start = rendered.find(sentence)
    if rendered_start < 0 or rendered.count(sentence) != 1:
        raise ValueError(
            "needle sentence is not uniquely present in rendered chat"
        )
    rendered_end = rendered_start + len(sentence)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    fast_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    native_ids = [int(value) for value in prompt_ids[0].tolist()]
    if fast_ids != native_ids:
        raise ValueError(
            "fast rendered tokenization differs from native chat template"
        )
    positions = []
    for index, (start, end) in enumerate(
        encoded["offset_mapping"][0].tolist()
    ):
        if end > rendered_start and start < rendered_end:
            positions.append(index)
    if not positions:
        raise ValueError("needle sentence has no token offsets")
    prefix_length = len(native_ids) - 1
    if any(index >= prefix_length for index in positions):
        raise ValueError(
            "needle span reaches the pending token and is not in prefix"
        )
    return {
        "answer": answer,
        "sentence": sentence,
        "raw_start": matches[0].start(),
        "raw_end": matches[0].end(),
        "rendered_start": rendered_start,
        "rendered_end": rendered_end,
        "token_positions": positions,
        "token_ids": [native_ids[index] for index in positions],
        "prefix_length": prefix_length,
        "chat_tokenization_ids_equal": True,
    }


def native_masks(engine, boundary, fraction):
    masks = []
    keep_counts = []
    for layer in boundary.cache.layers:
        keys = layer.keys
        length = int(keys.shape[-2])
        keep = int(length * (1.0 - fraction))
        if keep <= 0:
            raise ValueError("action would retain no cache entries")
        scores = -keys.norm(dim=-1)
        indices = scores.topk(keep, dim=-1).indices[0].tolist()
        masks.append([[int(value) for value in head] for head in indices])
        keep_counts.append(keep)
    return masks, keep_counts


def build_swap_masks(boundary, baseline_masks, target_positions, mode, sink):
    target = set(target_positions)
    output = []
    audit = []
    for layer_index, (layer, layer_masks) in enumerate(
        zip(boundary.cache.layers, baseline_masks, strict=True)
    ):
        keys = layer.keys
        scores = (-keys.norm(dim=-1))[0]
        scores_cpu = scores.detach().to(device="cpu").tolist()
        output_layer = []
        audit_layer = []
        for head_index, base in enumerate(layer_masks):
            base_set = set(base)
            missing = sorted(target - base_set)
            k = len(missing)
            victim_candidates = [
                value
                for value in base
                if value not in target and value >= sink
            ]
            victims = sorted(
                victim_candidates,
                key=lambda value: (
                    float(scores_cpu[head_index][value]),
                    value,
                ),
            )[:k]
            if len(victims) != k:
                raise ValueError("not enough non-needle retained victims")
            excluded = set(base)
            candidates = [
                value
                for value in range(int(keys.shape[-2]))
                if value not in excluded
                and value not in target
                and value >= sink
            ]
            if mode == "oracle_span_rescue":
                additions = missing
            elif mode == "matched_control":
                additions = sorted(
                    candidates,
                    key=lambda value: (
                        min(
                            abs(value - target_value)
                            for target_value in target
                        ),
                        value,
                    ),
                )[:k]
            else:
                raise ValueError(f"unsupported swap mode: {mode}")
            if len(additions) != k:
                raise ValueError(
                    "not enough non-needle, non-sink control candidates"
                )
            victim_slots = sorted(base.index(value) for value in victims)
            replacements = sorted(additions)
            branch = list(base)
            for slot, replacement in zip(
                victim_slots, replacements, strict=True
            ):
                branch[slot] = replacement
            if len(set(branch)) != len(branch):
                raise AssertionError(
                    "custom mask contains duplicate cache positions"
                )
            if any(
                branch[index] != base[index]
                for index in range(len(base))
                if index not in victim_slots
            ):
                raise AssertionError(
                    "custom mask reordered an unaffected cache slot"
                )
            output_layer.append(branch)
            audit_layer.append(
                {
                    "layer": layer_index,
                    "kv_head": head_index,
                    "baseline_indices": list(base),
                    "branch_indices": list(branch),
                    "target_positions": sorted(target),
                    "missing_target_indices": missing,
                    "k": k,
                    "victim_indices": victims,
                    "victim_slots": victim_slots,
                    "replacement_indices_sorted": replacements,
                    "span_retention_fraction": sum(
                        value in target for value in branch
                    )
                    / len(target),
                    "control_candidate_distances": {
                        str(value): min(
                            abs(value - target_value)
                            for target_value in target
                        )
                        for value in replacements
                    },
                    "victim_scores": {
                        str(value): float(scores_cpu[head_index][value])
                        for value in victims
                    },
                }
            )
        output.append(output_layer)
        audit.append(audit_layer)
    return output, audit


def apply_masks(engine, cache, masks, torch):
    for layer, layer_masks in zip(cache.layers, masks, strict=True):
        keys = layer.keys
        values = layer.values
        indices = torch.tensor(
            layer_masks, dtype=torch.long, device=keys.device
        )
        indices = indices.unsqueeze(0)
        gather_indices = indices.unsqueeze(-1).expand(
            -1, -1, -1, keys.shape[-1]
        )
        layer.keys = keys.gather(2, gather_indices).contiguous()
        layer.values = values.gather(2, gather_indices).contiguous()
    lengths = engine.cache_lengths(cache)
    if len(set(lengths)) != 1:
        raise AssertionError("custom cache layers have unequal lengths")


def custom_branch(engine, model, boundary, masks, eos, cap, torch):
    engine._restore_rng(boundary.rng_state, engine._model_device(model))
    cache = engine.clone_cache(boundary.cache)
    clone_equal = engine.cache_tensors_equal(cache, boundary.cache)
    clone_disjoint = engine.cache_storage_independent(boundary.cache, cache)
    apply_masks(engine, cache, masks, torch)
    masked_lengths = list(engine.cache_lengths(cache))
    masked_bytes = engine.cache_nbytes(cache)
    masked_fingerprint = engine.cache_fingerprint(cache)
    started = time.perf_counter()
    continuation = engine._continue_cache(
        model,
        boundary,
        cache,
        max_new_tokens=cap,
        eos_ids=eos,
        first_logits_observer=None,
    )
    engine._sync_device(engine._model_device(model))
    return continuation, {
        "clone_equal_before_mask": clone_equal,
        "clone_disjoint_before_mask": clone_disjoint,
        "cache_lengths": masked_lengths,
        "cache_bytes": masked_bytes,
        "cache_fingerprint_before_continuation": masked_fingerprint,
        "continuation_seconds": time.perf_counter() - started,
    }


def run_prompt(
    args, row, engine, model, tokenizer, torch, output_path, prior_dirs
):
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    span = locate_span(
        tokenizer, prompt_ids, row["prompt"], row["answers"][0]
    )
    boundary, source_cache = runner.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    boundary_before = engine.cache_fingerprint(boundary.cache)
    source_before = engine.cache_fingerprint(source_cache)
    if not engine.cache_tensors_equal(boundary.cache, source_cache):
        raise RuntimeError("boundary and source cache tensors differ")
    if not engine.cache_storage_independent(boundary.cache, source_cache):
        raise RuntimeError("boundary and source cache storage is aliased")
    masks, keep_counts = native_masks(engine, boundary, REMOVAL_FRACTION)
    eos = runner.eos_ids(model, tokenizer)
    cap = row.get("max_new_tokens", 120)
    prior = None
    prior_checks = {"enabled": not args.skip_prior_check}
    if not args.skip_prior_check:
        _, prior = find_prior_record(prior_dirs, row["id"])

    reference_arm = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    noop_arm = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    standard_arm = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", REMOVAL_FRACTION),
    )
    standard_repeat_arm = engine.continue_from_boundary(
        model,
        boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", REMOVAL_FRACTION),
    )
    standard_masks = json_indices(standard_arm.compression.kept_indices)
    standard_repeat_masks = json_indices(
        standard_repeat_arm.compression.kept_indices
    )
    if prior is not None:
        prior_checks.update(
            {
                "reference_token_ids_match": (
                    reference_arm.continuation.token_ids
                )
                == tuple(prior["reference"]["token_ids"]),
                "standard_token_ids_match": (
                    standard_arm.continuation.token_ids
                )
                == tuple(
                    prior["arms"]["knorm:0.1"]["continuation"]["token_ids"]
                ),
            }
        )

    controls = {
        "span_all_in_prefix": all(
            index < span["prefix_length"] for index in span["token_positions"]
        ),
        "span_disjoint_sink": all(
            index >= SINK for index in span["token_positions"]
        ),
        "reference_noop_tokens_equal": reference_arm.continuation.token_ids
        == noop_arm.continuation.token_ids,
        "reference_noop_termination_equal": (
            reference_arm.continuation.termination_reason
        )
        == noop_arm.continuation.termination_reason,
        "standard_repeat_tokens_equal": standard_arm.continuation.token_ids
        == standard_repeat_arm.continuation.token_ids,
        "standard_repeat_termination_equal": (
            standard_arm.continuation.termination_reason
        )
        == standard_repeat_arm.continuation.termination_reason,
        "standard_repeat_masks_equal": standard_masks
        == standard_repeat_masks,
        "standard_repeat_final_cache_equal": (
            standard_arm.continuation.final_cache_fingerprint
        )
        == standard_repeat_arm.continuation.final_cache_fingerprint,
        "reference_noop_final_cache_equal": (
            reference_arm.continuation.final_cache_fingerprint
        )
        == noop_arm.continuation.final_cache_fingerprint,
        "native_baseline_indices_equal": standard_masks == masks,
        "boundary_source_unchanged": engine.cache_fingerprint(boundary.cache)
        == boundary_before
        and engine.cache_fingerprint(source_cache) == source_before,
        **{
            key: value
            for key, value in prior_checks.items()
            if key != "enabled"
        },
    }

    branches = {
        "reference_shared_boundary": score_branch(
            tokenizer, reference_arm.continuation, row["answers"]
        ),
        "noop_shared_boundary": score_branch(
            tokenizer, noop_arm.continuation, row["answers"]
        ),
        "standard_knorm_0.10": {
            **score_branch(
                tokenizer, standard_arm.continuation, row["answers"]
            ),
            "branch_label": "standard_knorm_0.10",
            "compression": standard_arm.compression.to_dict(),
            "kept_indices": standard_masks,
            "span_retention_fractions": span_retention_fractions(
                standard_masks, span["token_positions"]
            ),
        },
        "standard_knorm_0.10_repeat": {
            **score_branch(
                tokenizer, standard_repeat_arm.continuation, row["answers"]
            ),
            "branch_label": "standard_knorm_0.10_repeat",
            "kept_indices": standard_repeat_masks,
            "span_retention_fractions": span_retention_fractions(
                standard_repeat_masks, span["token_positions"]
            ),
        },
    }
    branches["reference_shared_boundary"]["branch_label"] = (
        "reference_shared_boundary"
    )
    branches["noop_shared_boundary"]["branch_label"] = "noop_shared_boundary"
    partial = {
        "schema_version": "needle_rescue_diagnostic.v1",
        "status": "running",
        "manifest_row": row,
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "prompt_length": int(prompt_ids.shape[1]),
        "span": span,
        "boundary": {
            "cache_lengths": list(boundary.cache_lengths),
            "cache_bytes": boundary.cache_bytes,
            "boundary_cache_fingerprint": boundary_before,
            "source_cache_fingerprint": source_before,
            "source_boundary_equal": True,
            "source_boundary_disjoint": True,
        },
        "action": {"name": "knorm", "removal_fraction": REMOVAL_FRACTION},
        "native_baseline_masks": masks,
        "keep_counts": keep_counts,
        "branches": branches,
        "prior_checks": prior_checks,
        "checks": controls,
    }
    runner.write_json(output_path, partial)
    if not all(controls.values()):
        raise RuntimeError("baseline needle rescue gate failed")

    native_replay, native_replay_meta = custom_branch(
        engine, model, boundary, masks, eos, cap, torch
    )
    branches["native_mask_replay"] = {
        **score_branch(tokenizer, native_replay, row["answers"]),
        "branch_label": "native_mask_replay",
        "mask_semantics": "validation replay of native Knorm indices",
        "kept_indices": masks,
        "span_retention_fractions": span_retention_fractions(
            masks, span["token_positions"]
        ),
        "branch_meta": native_replay_meta,
    }
    controls.update(
        {
            "native_mask_replay_tokens_equal": native_replay.token_ids
            == standard_arm.continuation.token_ids,
            "native_mask_replay_termination_equal": (
                native_replay.termination_reason
            )
            == standard_arm.continuation.termination_reason,
            "native_mask_replay_final_cache_equal": (
                native_replay.final_cache_fingerprint
            )
            == standard_arm.continuation.final_cache_fingerprint,
            "native_mask_replay_source_unchanged": engine.cache_fingerprint(
                boundary.cache
            )
            == boundary_before
            and engine.cache_fingerprint(source_cache) == source_before,
            "native_mask_replay_clone_controls": native_replay_meta[
                "clone_equal_before_mask"
            ]
            and native_replay_meta["clone_disjoint_before_mask"],
        }
    )
    runner.write_json(output_path, partial)
    if not all(
        controls[key]
        for key in controls
        if key.startswith("native_mask_replay_")
    ):
        raise RuntimeError("native mask replay did not match standard Knorm")

    target_positions = span["token_positions"]
    for mode in ("oracle_span_rescue", "matched_control"):
        branch_masks, audit = build_swap_masks(
            boundary, masks, target_positions, mode, SINK
        )
        continuation, branch_meta = custom_branch(
            engine, model, boundary, branch_masks, eos, cap, torch
        )
        branches[mode] = {
            **score_branch(tokenizer, continuation, row["answers"]),
            "branch_label": mode,
            "mask_semantics": (
                "fixed-cardinality swap over native Knorm baseline"
            ),
            "kept_indices": branch_masks,
            "span_retention_fractions": span_retention_fractions(
                branch_masks, span["token_positions"]
            ),
            "swap_audit": audit,
            "branch_meta": branch_meta,
        }
        controls[f"{mode}_cardinality_exact"] = all(
            len(head) == keep
            for layer, keep in zip(branch_masks, keep_counts, strict=True)
            for head in layer
        )
        controls[f"{mode}_clone_controls"] = all(
            layer_meta["clone_equal_before_mask"]
            and layer_meta["clone_disjoint_before_mask"]
            for layer_meta in [branch_meta]
        )
        controls[f"{mode}_source_unchanged"] = (
            engine.cache_fingerprint(boundary.cache) == boundary_before
            and engine.cache_fingerprint(source_cache) == source_before
        )
    oracle_audit = branches["oracle_span_rescue"]["swap_audit"]
    control_audit = branches["matched_control"]["swap_audit"]
    reference_fraction = branches["reference_shared_boundary"]["score"][
        "score_fraction"
    ]
    for branch in branches.values():
        branch["signed_loss_vs_reference"] = (
            reference_fraction - branch["score"]["score_fraction"]
        )
    controls["oracle_control_victim_sets_equal"] = all(
        oracle_head["victim_indices"] == control_head["victim_indices"]
        for oracle_layer, control_layer in zip(
            oracle_audit, control_audit, strict=True
        )
        for oracle_head, control_head in zip(
            oracle_layer, control_layer, strict=True
        )
    )
    controls["oracle_control_k_equal"] = all(
        oracle_head["k"] == control_head["k"]
        for oracle_layer, control_layer in zip(
            oracle_audit, control_audit, strict=True
        )
        for oracle_head, control_head in zip(
            oracle_layer, control_layer, strict=True
        )
    )
    controls["all_checks_pass"] = all(controls.values())
    runner.write_json(output_path, partial)
    if not controls["all_checks_pass"]:
        raise RuntimeError("needle rescue diagnostic gate failed")
    return {
        "schema_version": "needle_rescue_diagnostic.v1",
        "status": "completed",
        "manifest_row": row,
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "prompt_length": int(prompt_ids.shape[1]),
        "span": span,
        "boundary": {
            "cache_lengths": list(boundary.cache_lengths),
            "cache_bytes": boundary.cache_bytes,
            "boundary_cache_fingerprint": boundary_before,
            "source_cache_fingerprint": source_before,
            "source_boundary_equal": True,
            "source_boundary_disjoint": True,
        },
        "action": {"name": "knorm", "removal_fraction": REMOVAL_FRACTION},
        "native_baseline_masks": masks,
        "keep_counts": keep_counts,
        "branches": branches,
        "prior_checks": prior_checks,
        "checks": controls,
    }


def run(args):
    output = runner.ensure_output_dir(args.output)
    engine, engine_root, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    requested_ids = tuple(args.ids)
    by_id = {row["id"]: row for row in rows}
    if (
        len(requested_ids) != len(set(requested_ids))
        or len(by_id) != len(rows)
        or any(row_id not in by_id for row_id in requested_ids)
    ):
        raise ValueError(
            "manifest does not contain every requested diagnostic ID"
        )
    selected_rows = [by_id[row_id] for row_id in requested_ids]
    import torch
    import transformers

    torch.manual_seed(SEED)
    device = runner.choose_device(args.device, torch)
    dtype = runner.choose_dtype(args.dtype, device, torch)
    model, tokenizer = runner.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    run_record = {
        "schema_version": "needle_rescue_diagnostic.v1",
        "status": "running",
        "ids": list(requested_ids),
        "selection_rule": (
            "fixed IDs chosen from generator answer-position metadata "
            "after development exposure"
        ),
        "action": {"name": "knorm", "removal_fraction": REMOVAL_FRACTION},
        "seed": SEED,
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "official_evaluate_sha256": sha256_file(OFFICIAL_EVAL),
        "official_constants_sha256": sha256_file(OFFICIAL_CONSTANTS),
        "model": runner.model_runtime_identity(
            engine, model, args.model, transformers, torch
        ),
        "device": str(device),
        "dtype": str(dtype),
        "prompts": [],
        "failures": [],
    }
    runner.write_json(output / "run.json", run_record)
    prior_dirs = args.prior_results
    for index, row in enumerate(selected_rows):
        prompt_path = output / runner.safe_filename(index, row["id"])
        try:
            record = run_prompt(
                args,
                row,
                engine,
                model,
                tokenizer,
                torch,
                prompt_path,
                prior_dirs,
            )
            runner.write_json(prompt_path, record)
            status = "completed"
        except Exception as error:
            status = "failed"
            record = {}
            if prompt_path.exists():
                record = json.loads(prompt_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "schema_version": "needle_rescue_diagnostic.v1",
                    "status": "failed",
                    "manifest_row": row,
                    "failure": {
                        "type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                }
            )
            runner.write_json(prompt_path, record)
            run_record["failures"].append(
                {"id": row["id"], **record["failure"]}
            )
        run_record["prompts"].append(
            {"id": row["id"], "path": prompt_path.name, "status": status}
        )
        runner.write_json(output / "run.json", run_record)
    run_record["status"] = (
        "completed" if not run_record["failures"] else "failed"
    )
    runner.write_json(output / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ids", nargs="+", default=list(TARGET_IDS))
    parser.add_argument(
        "--prior-results",
        nargs="+",
        default=["results/ea-dev-v1-first", "results/ea-dev-v1-rest"],
    )
    parser.add_argument("--skip-prior-check", action="store_true")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        if output.is_dir():
            runner.write_json(
                output / "failure.json",
                {
                    "status": "failed",
                    "type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        print(
            f"needle rescue failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
