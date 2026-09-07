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
    "ruler-ea-dev-v1-niah_single_2-002",
    "ruler-ea-dev-v1-niah_single_2-003",
    "ruler-ea-dev-v1-niah_single_2-004",
    "ruler-ea-dev-v1-niah_single_2-005",
    "ruler-ea-dev-v1-niah_single_2-008",
    "ruler-ea-dev-v1-niah_single_2-010",
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
    answer_start = rendered_start + sentence.find(answer)
    answer_end = answer_start + len(answer)
    value_positions = []
    for index, (start, end) in enumerate(
        encoded["offset_mapping"][0].tolist()
    ):
        if end > rendered_start and start < rendered_end:
            positions.append(index)
        if end > answer_start and start < answer_end:
            value_positions.append(index)
    if not positions:
        raise ValueError("needle sentence has no token offsets")
    prefix_length = len(native_ids) - 1
    if any(index >= prefix_length for index in positions):
        raise ValueError(
            "needle span reaches the pending token and is not in prefix"
        )
    value_positions = sorted(set(value_positions))
    value_positions = [index for index in value_positions if index in positions]
    if not value_positions:
        raise ValueError("answer value has no token offsets")
    context_positions = [index for index in positions if index not in value_positions]
    if set(value_positions) | set(context_positions) != set(positions):
        raise AssertionError("value and context partitions do not cover sentence")
    return {
        "answer": answer,
        "sentence": sentence,
        "raw_start": matches[0].start(),
        "raw_end": matches[0].end(),
        "rendered_start": rendered_start,
        "rendered_end": rendered_end,
        "token_positions": positions,
        "token_ids": [native_ids[index] for index in positions],
        "value_raw_start": matches[0].start() + sentence.find(answer),
        "value_raw_end": matches[0].start() + sentence.find(answer) + len(answer),
        "value_rendered_start": answer_start,
        "value_rendered_end": answer_end,
        "value_token_positions": value_positions,
        "context_token_positions": context_positions,
        "partition_union_exact": set(value_positions) | set(context_positions)
        == set(positions),
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


def build_partition_masks(
    boundary, baseline_masks, sentence_positions, partition_positions,
    prior_audit, mode, sink
):
    sentence = set(sentence_positions)
    target = set(partition_positions)
    if mode not in ("value_rescue", "context_rescue", "matched_control"):
        raise ValueError(f"unsupported partition mode: {mode}")
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
            old_head = prior_audit[layer_index][head_index]
            old_victims = [int(value) for value in old_head["victim_indices"]]
            victims = old_victims[:k]
            if len(victims) != k:
                raise ValueError("old full-sentence rescue has too few victims")
            if any(
                value not in base or value in sentence or value < sink
                for value in victims
            ):
                raise AssertionError("old victim ordering violates exclusions")
            excluded = set(base)
            candidates = [
                value
                for value in range(int(keys.shape[-2]))
                if value not in excluded
                and value not in sentence
                and value >= sink
            ]
            if mode in ("value_rescue", "context_rescue"):
                additions = missing
            elif mode == "matched_control":
                additions = sorted(
                    candidates,
                    key=lambda value: (
                        min(
                            abs(value - target_value)
                            for target_value in target
                        ) if target else 0,
                        value,
                    ),
                )[:k]
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
                    "partition": mode,
                    "full_sentence_positions": sorted(sentence),
                    "partition_positions": sorted(target),
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
                    "old_fullsentence_victim_indices": old_victims,
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


def _same_old_branch(current, prior, key):
    old = prior["branches"][key]
    return {
        "token_ids": current["continuation"]["token_ids"]
        == old["continuation"]["token_ids"],
        "termination": current["continuation"]["termination_reason"]
        == old["continuation"]["termination_reason"],
        "score": current["score"] == old["score"],
    }


def _mask_checks(masks, base, keep_counts, target, sentence, audits, require_target):
    checks = {
        "cardinality_exact": all(
            len(head) == keep
            for layer, keep in zip(masks, keep_counts, strict=True)
            for head in layer
        ),
        "unique_indices": all(
            len(head) == len(set(head)) for layer in masks for head in layer
        ),
        "unchanged_slots": all(
            branch[index] == original[index]
            for branch_layer, base_layer, audit_layer in zip(
                masks, base, audits, strict=True
            )
            for branch, original, audit in zip(
                branch_layer, base_layer, audit_layer, strict=True
            )
            for index in range(len(original))
            if index not in set(audit["victim_slots"])
        ),
    }
    if require_target:
        checks["target_fully_retained"] = all(
            set(target) <= set(head) for layer in masks for head in layer
        )
    else:
        checks["target_membership_unchanged"] = all(
            set(original) & set(target) == set(branch) & set(target)
            for branch_layer, base_layer in zip(masks, base, strict=True)
            for branch, original in zip(branch_layer, base_layer, strict=True)
        )
    if not require_target:
        checks["control_additions_outside_sentence"] = all(
            all(value not in set(sentence) for value in audit["replacement_indices_sorted"])
            for layer in audits for audit in layer
        )
    return checks


def run_prompt(
    args, row, engine, model, tokenizer, torch, output_path, prior_dirs
):
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    span = locate_span(tokenizer, prompt_ids, row["prompt"], row["answers"][0])
    prior_path, prior = find_prior_record(prior_dirs, row["id"])
    boundary, source_cache = runner.build_last_prompt_boundary(engine, model, prompt_ids)
    boundary_before = engine.cache_fingerprint(boundary.cache)
    source_before = engine.cache_fingerprint(source_cache)
    if not engine.cache_tensors_equal(boundary.cache, source_cache):
        raise RuntimeError("boundary and source cache tensors differ")
    if not engine.cache_storage_independent(boundary.cache, source_cache):
        raise RuntimeError("boundary and source cache storage is aliased")
    masks, keep_counts = native_masks(engine, boundary, REMOVAL_FRACTION)
    old_masks = prior["native_baseline_masks"]
    old_full = prior["branches"]["oracle_span_rescue"]
    sentence_positions = span["token_positions"]
    eos = runner.eos_ids(model, tokenizer)
    cap = row.get("max_new_tokens", 128)

    reference_arm = engine.continue_from_boundary(
        model, boundary, max_new_tokens=cap, eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0)
    )
    standard_arm = engine.continue_from_boundary(
        model, boundary, max_new_tokens=cap, eos_ids=eos,
        action=engine.ActionSpec("knorm", REMOVAL_FRACTION)
    )
    standard_masks = json_indices(standard_arm.compression.kept_indices)
    branches = {
        "reference_shared_boundary": {
            **score_branch(tokenizer, reference_arm.continuation, row["answers"]),
            "branch_label": "reference_shared_boundary",
        },
        "standard_knorm_0.10": {
            **score_branch(tokenizer, standard_arm.continuation, row["answers"]),
            "branch_label": "standard_knorm_0.10",
            "compression": standard_arm.compression.to_dict(),
            "kept_indices": standard_masks,
        },
    }
    prior_checks = {"enabled": True, "prior_path": str(prior_path)}
    prior_checks["prompt_token_ids_match"] = [int(x) for x in prompt_ids[0].tolist()] == prior["prompt_token_ids"]
    prior_checks["native_masks_match"] = masks == old_masks
    prior_checks["standard_masks_match_native"] = standard_masks == masks
    prior_checks["reference_replay"] = _same_old_branch(
        branches["reference_shared_boundary"], prior, "reference_shared_boundary"
    )
    prior_checks["standard_replay"] = _same_old_branch(
        branches["standard_knorm_0.10"], prior, "standard_knorm_0.10"
    )
    partial = {
        "schema_version": "span_partition_diagnostic.v1",
        "status": "running",
        "manifest_row": row,
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "prompt_length": int(prompt_ids.shape[1]),
        "span": span,
        "partition": {
            "value_token_positions": span["value_token_positions"],
            "context_token_positions": span["context_token_positions"],
            "union_exact": span["partition_union_exact"],
        },
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
        "checks": {},
    }
    runner.write_json(output_path, partial)
    if not prior_checks["prompt_token_ids_match"] or not prior_checks["native_masks_match"] or not prior_checks["standard_masks_match_native"]:
        raise RuntimeError("exact prior/native baseline replay failed")
    if not all(all(values.values()) for key, values in prior_checks.items() if isinstance(values, dict)):
        raise RuntimeError("exact reference or standard replay failed")

    full_cont, full_meta = custom_branch(engine, model, boundary, old_full["kept_indices"], eos, cap, torch)
    branches["oracle_span_rescue"] = {
        **score_branch(tokenizer, full_cont, row["answers"]),
        "branch_label": "oracle_span_rescue",
        "kept_indices": old_full["kept_indices"],
        "swap_audit": old_full["swap_audit"],
        "branch_meta": full_meta,
    }
    prior_checks["full_oracle_replay"] = _same_old_branch(
        branches["oracle_span_rescue"], prior, "oracle_span_rescue"
    )
    if not all(prior_checks["full_oracle_replay"].values()):
        raise RuntimeError("exact full-sentence oracle replay failed")

    partition_specs = (
        ("value_rescue", "value_control", span["value_token_positions"]),
        ("context_rescue", "context_control", span["context_token_positions"]),
    )
    branch_checks = {}
    for rescue_name, control_name, target in partition_specs:
        rescue_masks, rescue_audit = build_partition_masks(
            boundary, masks, sentence_positions, target,
            old_full["swap_audit"], rescue_name, SINK
        )
        control_masks, control_audit = build_partition_masks(
            boundary, masks, sentence_positions, target,
            old_full["swap_audit"], "matched_control", SINK
        )
        rescue_cont, rescue_meta = custom_branch(engine, model, boundary, rescue_masks, eos, cap, torch)
        control_cont, control_meta = custom_branch(engine, model, boundary, control_masks, eos, cap, torch)
        for name, continuation, branch_masks, audit, meta in (
            (rescue_name, rescue_cont, rescue_masks, rescue_audit, rescue_meta),
            (control_name, control_cont, control_masks, control_audit, control_meta),
        ):
            branches[name] = {
                **score_branch(tokenizer, continuation, row["answers"]),
                "branch_label": name,
                "mask_semantics": "fixed-cardinality partition swap over native Knorm baseline",
                "kept_indices": branch_masks,
                "partition_token_positions": sorted(target),
                "full_sentence_token_positions": sorted(sentence_positions),
                "swap_audit": audit,
                "branch_meta": meta,
            }
        pair_checks = {
            rescue_name: _mask_checks(
                rescue_masks, masks, keep_counts, target, sentence_positions,
                rescue_audit, True
            ),
            control_name: _mask_checks(
                control_masks, masks, keep_counts, target, sentence_positions,
                control_audit, False
            ),
            "victim_indices_equal": all(
                a["victim_indices"] == b["victim_indices"]
                for al, bl in zip(rescue_audit, control_audit, strict=True)
                for a, b in zip(al, bl, strict=True)
            ),
            "victim_slots_equal": all(
                a["victim_slots"] == b["victim_slots"]
                for al, bl in zip(rescue_audit, control_audit, strict=True)
                for a, b in zip(al, bl, strict=True)
            ),
            "swap_counts_equal": all(
                a["k"] == b["k"]
                for al, bl in zip(rescue_audit, control_audit, strict=True)
                for a, b in zip(al, bl, strict=True)
            ),
            "source_cache_unchanged": engine.cache_fingerprint(boundary.cache) == boundary_before
            and engine.cache_fingerprint(source_cache) == source_before,
            "rescue_clone_controls": rescue_meta["clone_equal_before_mask"] and rescue_meta["clone_disjoint_before_mask"],
            "control_clone_controls": control_meta["clone_equal_before_mask"] and control_meta["clone_disjoint_before_mask"],
            "physical_lengths_equal": rescue_meta["cache_lengths"] == control_meta["cache_lengths"] == list(keep_counts),
            "physical_bytes_equal": rescue_meta["cache_bytes"] == control_meta["cache_bytes"],
        }
        branch_checks.update({f"{rescue_name}_{k}": v for k, v in pair_checks[rescue_name].items()})
        branch_checks.update({f"{control_name}_{k}": v for k, v in pair_checks[control_name].items()})
        branch_checks.update({f"{rescue_name}_{k}": v for k, v in pair_checks.items() if k not in (rescue_name, control_name)})

    reference_fraction = branches["reference_shared_boundary"]["score"]["score_fraction"]
    for branch in branches.values():
        branch["signed_loss_vs_reference"] = reference_fraction - branch["score"]["score_fraction"]
    checks = {
        "span_all_in_prefix": all(index < span["prefix_length"] for index in sentence_positions),
        "span_disjoint_sink": all(index >= SINK for index in sentence_positions),
        "partition_union_exact": set(span["value_token_positions"]) | set(span["context_token_positions"]) == set(sentence_positions),
        "boundary_source_unchanged": engine.cache_fingerprint(boundary.cache) == boundary_before and engine.cache_fingerprint(source_cache) == source_before,
        "full_oracle_clone_controls": full_meta["clone_equal_before_mask"] and full_meta["clone_disjoint_before_mask"],
        "full_oracle_physical_lengths": full_meta["cache_lengths"] == list(keep_counts),
        "full_oracle_physical_bytes": full_meta["cache_bytes"] == branches["value_rescue"]["branch_meta"]["cache_bytes"],
        **branch_checks,
    }
    checks["all_checks_pass"] = all(checks.values()) and all(
        all(values.values()) for values in prior_checks.values() if isinstance(values, dict)
    )
    partial.update({"status": "completed", "branches": branches, "prior_checks": prior_checks, "checks": checks})
    runner.write_json(output_path, partial)
    if not checks["all_checks_pass"]:
        raise RuntimeError("span partition diagnostic gate failed: " + json.dumps({k: v for k, v in checks.items() if not v}, sort_keys=True))
    return partial


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
        "schema_version": "span_partition_diagnostic.v1",
        "status": "running",
        "ids": list(requested_ids),
        "selection_rule": "eight known failed B0.10 cases plus healthy001 guard",
        "action": {"name": "knorm", "removal_fraction": REMOVAL_FRACTION},
        "seed": SEED,
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()),
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "diagnostic_source_sha256": sha256_file(Path(__file__).resolve()),
        "prior_rescue_dirs": [str(Path(item).expanduser().resolve()) for item in args.prior_rescue],
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
    prior_dirs = args.prior_rescue
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
                    "schema_version": "span_partition_diagnostic.v1",
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
        "--prior-rescue",
        nargs="+",
        default=[
            "results/needle-rescue-v1-first",
            "results/needle-rescue-v1-rest",
            "results/needle-rescue-all12-additional",
        ],
    )
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
            f"span partition failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
