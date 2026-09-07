#!/usr/bin/env python3
"""Diagnose full-prefill versus last-token split continuation divergence."""

import argparse
import sys
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import run_pair_pilot as pilot  # noqa: E402


PROMPT_ID = "ruler-pilot-v1-cwe-002"
CAP = 120
SEED = 0


def divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right, strict=False)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def continuation_record(tokenizer, continuation):
    token_ids = list(continuation.token_ids)
    return {
        "token_ids": token_ids,
        "token_count": len(token_ids),
        "text": tokenizer.decode(token_ids, skip_special_tokens=True),
        "termination_reason": continuation.termination_reason,
        "forward_seconds": continuation.forward_seconds,
        "first_forward_seconds": continuation.first_forward_seconds,
        "final_cache_lengths": list(continuation.final_cache_lengths),
        "final_cache_bytes": continuation.final_cache_bytes,
        "final_cache_fingerprint": continuation.final_cache_fingerprint,
        "validation_seconds": continuation.validation_seconds,
    }


def run(args):
    output = pilot.ensure_output_dir(args.output)
    engine, engine_root, engine_path = pilot.load_engine(args.engine_root)
    manifest_path, rows = pilot.load_manifest(args.manifest)
    row = next((item for item in rows if item["id"] == PROMPT_ID), None)
    if row is None:
        raise ValueError(f"manifest lacks required row {PROMPT_ID}")
    if row.get("max_new_tokens") != CAP:
        raise ValueError(f"{PROMPT_ID} max_new_tokens must be {CAP}")

    import torch
    import transformers

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = pilot.choose_device(args.device, torch)
    dtype = pilot.choose_dtype("bfloat16", device, torch)
    if device.type != "cuda":
        raise ValueError("diagnostic requires CUDA for the pinned BF16 SDPA runtime")
    model, tokenizer = pilot.load_model_and_tokenizer(
        args.model, device, dtype, transformers
    )
    prompt_ids = pilot.tokenize_chat_prompt(tokenizer, row["prompt"])
    token_checks = pilot.token_identity_checks(
        tokenizer, row, prompt_ids, row["prompt"]
    )
    boundary, source_cache = pilot.build_last_prompt_boundary(
        engine, model, prompt_ids
    )
    source_before = engine.cache_fingerprint(boundary.cache)
    source_prefill_before = engine.cache_fingerprint(source_cache)
    eos = pilot.eos_ids(model, tokenizer)
    action = engine.ActionSpec("knorm", 0.0)

    full = []
    split = []
    source_checks = []
    partial_path = output / "partial.json"
    partial = {
        "schema_version": "prefill_partition_diagnostic.v1",
        "status": "running",
        "prompt_id": PROMPT_ID,
        "arms": [],
        "source_checks": [],
    }
    pilot.write_json(partial_path, partial)
    for arm in ("full_1", "full_2", "split_1", "split_2"):
        before = engine.cache_fingerprint(boundary.cache)
        if arm.startswith("full"):
            engine._restore_rng(boundary.rng_state, device)
            result, elapsed = pilot.timed_call(
                engine,
                device,
                lambda: engine._greedy_from_prompt(
                    model, prompt_ids, max_new_tokens=CAP, eos_ids=eos
                ),
            )
        else:
            result, elapsed = pilot.timed_call(
                engine,
                device,
                lambda: engine.continue_from_boundary(
                    model,
                    boundary,
                    max_new_tokens=CAP,
                    eos_ids=eos,
                    action=action,
                ),
            )
        after = engine.cache_fingerprint(boundary.cache)
        source_prefill_after = engine.cache_fingerprint(source_cache)
        source_ok = (
            before == after == source_before
            and source_prefill_after == source_prefill_before
        )
        source_checks.append(
            {
                "arm": arm,
                "before": before,
                "after": after,
                "prefill_source_before": source_prefill_before,
                "prefill_source_after": source_prefill_after,
                "unchanged": source_ok,
            }
        )
        continuation = result.continuation if arm.startswith("split") else result
        item = continuation_record(tokenizer, continuation)
        item["wall_seconds_synchronized"] = elapsed
        if arm.startswith("full"):
            full.append(item)
        else:
            split.append(item)
        partial["arms"].append({"arm": arm, "continuation": item})
        partial["source_checks"].append(source_checks[-1])
        pilot.write_json(partial_path, partial)

    checks = {
        "source_cache_unchanged_each_arm": all(
            item["unchanged"] for item in source_checks
        ),
        "full_repeat_tokens_equal": full[0]["token_ids"] == full[1]["token_ids"],
        "full_repeat_termination_equal": full[0]["termination_reason"]
        == full[1]["termination_reason"],
        "split_repeat_tokens_equal": split[0]["token_ids"] == split[1]["token_ids"],
        "split_repeat_termination_equal": split[0]["termination_reason"]
        == split[1]["termination_reason"],
        "full_split_termination_equal": all(
            item["termination_reason"] == split[0]["termination_reason"]
            for item in full
        ),
    }
    checks["all_assertions_pass"] = all(checks.values())
    record = {
        "schema_version": "prefill_partition_diagnostic.v1",
        "status": "completed" if checks["all_assertions_pass"] else "failed",
        "prompt_id": PROMPT_ID,
        "prompt": row["prompt"],
        "answers": row["answers"],
        "task": row.get("task"),
        "prompt_token_ids": [int(item) for item in prompt_ids[0].tolist()],
        "token_identity": token_checks,
        "max_new_tokens": CAP,
        "seed": SEED,
        "engine": {
            "root": str(engine_root),
            "path": str(engine_path),
            "sha256": pilot.sha256_file(engine_path),
        },
        "runtime": {
            "model_path": str(Path(args.model).expanduser().resolve()),
            "device": str(device),
            "dtype": str(dtype),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "attention_implementation": str(
                getattr(model.config, "_attn_implementation", "")
            ),
        },
        "boundary": boundary.to_dict(),
        "source_cache_fingerprint": source_before,
        "full_prefill": full,
        "split_prefix_noop": split,
        "source_checks": source_checks,
        "first_divergence_indices": {
            "full_1_vs_split_1": divergence(full[0]["token_ids"], split[0]["token_ids"]),
            "full_2_vs_split_2": divergence(full[1]["token_ids"], split[1]["token_ids"]),
        },
        "checks": checks,
    }
    pilot.write_json(output / "diagnostic.json", record)
    partial["status"] = record["status"]
    partial["checks"] = checks
    pilot.write_json(partial_path, partial)
    return 0 if checks["all_assertions_pass"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda",))
    args = parser.parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        output = Path(args.output).expanduser().resolve()
        if output.is_dir():
            pilot.write_json(
                output / "failure.json",
                {
                    "schema_version": "prefill_partition_diagnostic.v1",
                    "status": "failed",
                    "error": str(error),
                    "type": type(error).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        print(f"diagnostic failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
