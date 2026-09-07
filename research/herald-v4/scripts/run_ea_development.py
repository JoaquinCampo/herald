#!/usr/bin/env python3
"""Collect the locked EA feature and paired continuations for development rows."""

import argparse
import math
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import measure_ea as ea  # noqa: E402
import run_pair_pilot as runner  # noqa: E402


ACTIONS = (0.05, 0.10, 0.20)
CAP_DEFAULT = 120
SEED = 0


def action_record(tokenizer, arm, action):
    text = tokenizer.decode(list(arm.continuation.token_ids), skip_special_tokens=True)
    return {
        "action": action.to_dict(),
        "continuation": runner.continuation_dict(arm.continuation, text),
        "score_ready": {
            "text": text,
            "token_ids": list(arm.continuation.token_ids),
        },
    }


def run_prompt(
    *,
    index,
    row,
    engine,
    model,
    tokenizer,
    device,
    eos,
    press,
    output_path,
):
    ea.ACTIONS = ACTIONS
    record = {
        "schema_version": "ea_development.v1",
        "status": "running",
        "prompt_index": index,
        "manifest_row": row,
        "reference_path": "shared_boundary",
        "actions": [engine.ActionSpec("knorm", value).to_dict() for value in ACTIONS],
    }
    runner.write_json(output_path, record)
    cap = row.get("max_new_tokens", CAP_DEFAULT)
    if not isinstance(cap, int) or cap < 1:
        raise ValueError("max_new_tokens must be positive")
    prompt_ids = runner.tokenize_chat_prompt(tokenizer, row["prompt"])
    record["prompt_token_ids"] = [int(item) for item in prompt_ids[0].tolist()]
    record["prompt_length"] = int(prompt_ids.shape[1])
    record["max_new_tokens"] = cap
    record["token_identity"] = runner.token_identity_checks(
        tokenizer, row, prompt_ids, row["prompt"]
    )
    runner.write_json(output_path, record)

    plain_result, plain_wall, plain_baseline, plain_peak = ea.prefill_with_memory(
        engine,
        device,
        lambda: runner.build_last_prompt_boundary(engine, model, prompt_ids),
    )
    plain_boundary, plain_source = plain_result
    record["plain_prefill"] = {
        "wall_seconds_synchronized": plain_wall,
        "memory_baseline_bytes": plain_baseline,
        "peak_allocated_bytes": plain_peak,
        "cache_lengths": list(plain_boundary.cache_lengths),
        "cache_bytes": plain_boundary.cache_bytes,
        "cache_fingerprint": engine.cache_fingerprint(plain_boundary.cache),
        "source_fingerprint": engine.cache_fingerprint(plain_source),
    }
    runner.write_json(output_path, record)

    instrumented_result, inst_wall, inst_baseline, inst_peak = ea.prefill_with_memory(
        engine,
        device,
        lambda: ea.instrument_prefill(engine, model, prompt_ids, press, device),
    )
    inst_boundary, inst_source, features, hook_seconds, _ = instrumented_result
    record["instrumented_prefill"] = {
        "wall_seconds_synchronized": inst_wall,
        "summed_hook_seconds_synchronized": hook_seconds,
        "memory_baseline_bytes": inst_baseline,
        "peak_allocated_bytes": inst_peak,
        "peak_increment_bytes": inst_peak - inst_baseline,
        "cache_lengths": list(inst_boundary.cache_lengths),
        "cache_bytes": inst_boundary.cache_bytes,
        "cache_fingerprint": engine.cache_fingerprint(inst_boundary.cache),
        "source_fingerprint": engine.cache_fingerprint(inst_source),
    }
    record["features"] = {
        action: {
            "head_count": len(items),
            "heads": items,
            "summary": ea.summarize(items),
        }
        for action, items in features.items()
    }
    expected_feature_heads = len(model.model.layers) * int(
        model.config.num_key_value_heads
    )
    feature_checks = {
        action: {
            "head_count": len(items),
            "expected_head_count": expected_feature_heads,
            "head_count_exact": len(items) == expected_feature_heads,
            "all_numeric_values_finite": all(
                math.isfinite(float(value))
                for item in items
                for value in item.values()
                if isinstance(value, (int, float))
            ),
        }
        for action, items in features.items()
    }
    record["feature_checks"] = feature_checks
    if not all(
        check["head_count_exact"] and check["all_numeric_values_finite"]
        for check in feature_checks.values()
    ):
        raise RuntimeError("EA feature head-count or finite-value gate failed")
    runner.write_json(output_path, record)

    plain_cache_before = engine.cache_fingerprint(plain_boundary.cache)
    inst_cache_before = engine.cache_fingerprint(inst_boundary.cache)
    plain_source_before = engine.cache_fingerprint(plain_source)
    inst_source_before = engine.cache_fingerprint(inst_source)
    controls = {
        "boundary_cache_equal": engine.cache_tensors_equal(
            plain_boundary.cache, inst_boundary.cache
        ),
        "source_cache_equal": engine.cache_tensors_equal(plain_source, inst_source),
        "boundary_copies_disjoint": engine.cache_storage_independent(
            plain_boundary.cache, inst_boundary.cache
        ),
    }
    if not all(controls.values()):
        raise RuntimeError("plain and instrumented split boundaries differ")

    reference_arm = engine.continue_from_boundary(
        model,
        plain_boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    reference = reference_arm.continuation
    noop_arm = engine.continue_from_boundary(
        model,
        inst_boundary,
        max_new_tokens=cap,
        eos_ids=eos,
        action=engine.ActionSpec("knorm", 0.0),
    )
    noop = noop_arm.continuation
    controls["reference_source_unchanged"] = (
        engine.cache_fingerprint(plain_boundary.cache) == plain_cache_before
        and engine.cache_fingerprint(plain_source) == plain_source_before
    )
    controls["noop_source_unchanged"] = (
        engine.cache_fingerprint(inst_boundary.cache) == inst_cache_before
        and engine.cache_fingerprint(inst_source) == inst_source_before
    )
    controls["noop_matches_reference"] = (
        noop.token_ids == reference.token_ids
        and noop.termination_reason == reference.termination_reason
    )
    if not all(controls.values()):
        raise RuntimeError("shared-boundary reference/no-op gate failed")

    reference_text = tokenizer.decode(list(reference.token_ids), skip_special_tokens=True)
    record["reference"] = {
        **runner.continuation_dict(reference, reference_text),
        "reference_path": "shared_boundary",
    }
    record["checks"] = controls
    record["arms"] = {}
    noop_action = engine.ActionSpec("knorm", 0.0)
    noop_item = action_record(tokenizer, noop_arm, noop_action)
    noop_compression = runner.compact_compression(
        engine, noop_arm.compression, inst_boundary
    )
    noop_item.update(
        {
            "compression": noop_compression,
            "source_cache_unchanged": controls["noop_source_unchanged"],
            "matches_reference": controls["noop_matches_reference"],
            "boundary_cache_fingerprint_before": inst_cache_before,
            "boundary_cache_fingerprint_after": inst_cache_before,
            "source_cache_fingerprint_before": inst_source_before,
            "source_cache_fingerprint_after": inst_source_before,
        }
    )
    record["arms"][noop_action.action_id] = noop_item
    runner.write_json(output_path, record)

    for fraction in ACTIONS:
        action = engine.ActionSpec("knorm", fraction)
        before_boundary = engine.cache_fingerprint(inst_boundary.cache)
        before_source = engine.cache_fingerprint(inst_source)
        arm = engine.continue_from_boundary(
            model,
            inst_boundary,
            max_new_tokens=cap,
            eos_ids=eos,
            action=action,
        )
        after_boundary = engine.cache_fingerprint(inst_boundary.cache)
        after_source = engine.cache_fingerprint(inst_source)
        compression = runner.compact_compression(engine, arm.compression, inst_boundary)
        source_ok = before_boundary == after_boundary == inst_cache_before and after_source == before_source == inst_source_before
        action_ok = source_ok and compression["physical_effect_exact"] and compression["strictly_reduced_for_nonzero_action"]
        item = action_record(tokenizer, arm, action)
        item.update(
            {
                "compression": compression,
                "source_cache_unchanged": source_ok,
                "boundary_cache_fingerprint_before": before_boundary,
                "boundary_cache_fingerprint_after": after_boundary,
                "source_cache_fingerprint_before": before_source,
                "source_cache_fingerprint_after": after_source,
            }
        )
        record["arms"][action.action_id] = item
        runner.write_json(output_path, record)
        if not action_ok:
            raise RuntimeError(f"action gate failed for {action.action_id}")

    record["status"] = "completed"
    record["checks"]["all_actions_valid"] = True
    runner.write_json(output_path, record)
    return record


def run(args):
    output = runner.ensure_output_dir(args.output)
    engine, engine_root, engine_path = runner.load_engine(args.engine_root)
    manifest_path, rows = runner.load_manifest(args.manifest)
    if not rows:
        raise ValueError("manifest is empty")
    import torch
    import transformers
    from kvpress import ExpectedAttentionPress

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = runner.choose_device("cuda", torch)
    model, tokenizer = runner.load_model_and_tokenizer(
        args.model, device, torch.bfloat16, transformers
    )
    ea.ACTIONS = ACTIONS
    press = ExpectedAttentionPress(
        compression_ratio=0.0,
        n_sink=ea.SINK,
        n_future_positions=ea.FUTURE,
        use_covariance=False,
        use_vnorm=True,
        epsilon=0.0,
    )
    eos = runner.eos_ids(model, tokenizer)
    run_record = {
        "schema_version": "ea_development.v1",
        "status": "running",
        "reference_path": "shared_boundary",
        "actions": list(ACTIONS),
        "seed": SEED,
        "manifest_path": str(manifest_path),
        "manifest_sha256": runner.sha256_bytes(manifest_path.read_bytes()),
        "engine_sha256": runner.sha256_file(engine_path),
        "collector_source_sha256": runner.sha256_file(Path(__file__).resolve()),
        "runner_source_sha256": runner.sha256_file(SCRIPT_DIR / "run_pair_pilot.py"),
        "measurement_source_sha256": runner.sha256_file(SCRIPT_DIR / "measure_ea.py"),
        "model": runner.model_runtime_identity(engine, model, args.model, transformers, torch),
        "prompts": [],
        "failures": [],
    }
    runner.write_json(output / "run.json", run_record)
    for index, row in enumerate(rows):
        prompt_path = output / runner.safe_filename(index, row["id"])
        try:
            run_prompt(
                index=index,
                row=row,
                engine=engine,
                model=model,
                tokenizer=tokenizer,
                device=device,
                eos=eos,
                press=press,
                output_path=prompt_path,
            )
            status = "completed"
        except Exception as error:
            status = "failed"
            failure = {
                "type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            current = {}
            if prompt_path.exists():
                current = __import__("json").loads(prompt_path.read_text())
            current.update({"status": "failed", "failure": failure})
            runner.write_json(prompt_path, current)
            run_record["failures"].append({"id": row["id"], **failure})
        run_record["prompts"].append(
            {"id": row["id"], "path": prompt_path.name, "status": status}
        )
        runner.write_json(output / "run.json", run_record)
    run_record["status"] = "completed" if not run_record["failures"] else "failed"
    runner.write_json(output / "run.json", run_record)
    return 0 if run_record["status"] == "completed" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        if output.is_dir():
            runner.write_json(
                output / "failure.json",
                {"status": "failed", "type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()},
            )
        print(f"EA development failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
