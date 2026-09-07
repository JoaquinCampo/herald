#!/usr/bin/env python3
"""Measure paired GPU memory modes for one engineering prompt."""

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import gc
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from herald_v3.engineering import engine, runner
from herald_v3.engineering.prompts import load_prompt_manifest

ACTION = engine.ActionSpec("knorm", 0.5)
DEFAULT_MAX_NEW_TOKENS = 1024


def measure_memory_modes(
    model: Any,
    input_ids: torch.Tensor,
    *,
    eos_ids: frozenset[int],
    max_new_tokens: int,
    seed: int = 0,
    require_cuda: bool = True,
) -> dict[str, object]:
    """Measure the two exact same-action memory modes."""
    if max_new_tokens <= 32:
        raise ValueError("max_new_tokens must exceed the 32-token boundary")
    device = torch.device(next(model.parameters()).device)
    if require_cuda and device.type != "cuda":
        raise RuntimeError("memory diagnostic requires a CUDA model")
    model.eval()

    _set_seed(seed)
    direct_boundary = engine.build_boundary(model, input_ids, eos_ids=eos_ids)
    direct_start_fingerprint = direct_boundary.state_fingerprint
    direct_model_fingerprint = direct_boundary.model_state_fingerprint
    direct_expected_model_tensor_count = direct_boundary.model_tensor_count
    direct_rng_fingerprint = direct_boundary.rng_state.fingerprint
    direct_baseline = _begin_peak(device)
    engine._sync_device(device)
    started = time.perf_counter()
    direct_compression = engine.compress_knorm(
        direct_boundary.cache, ACTION.removal_fraction
    )
    direct_continuation = engine._continue_cache(
        model,
        direct_boundary,
        direct_boundary.cache,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        first_logits_observer=None,
    )
    engine._sync_device(device)
    direct_seconds = time.perf_counter() - started
    direct_peak = _finish_peak(device)
    model_after_direct, direct_observed_model_tensor_count = (
        engine._model_runtime_state_fingerprint(model)
    )
    rng_after_direct = engine._capture_rng(device).fingerprint
    direct_record = {
        "mode": "live_no_reference_no_probe",
        "reference_cache_preserved": False,
        "probe_enabled": False,
        "baseline_allocated_bytes": direct_baseline,
        "peak_allocated_bytes": direct_peak,
        "peak_increment_bytes": _difference(direct_peak, direct_baseline),
        "wall_seconds": direct_seconds,
        "state_copy_seconds": 0.0,
        "compression": direct_compression.to_dict(),
        "continuation": direct_continuation.to_dict(),
        "residual_gpu_state": (
            "model, input tensor, and the single mutated live cache"
        ),
    }

    del direct_boundary
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    engine._sync_device(device)

    _set_seed(seed)
    preserved_boundary = engine.build_boundary(
        model, input_ids, eos_ids=eos_ids
    )
    preserved_start_fingerprint = preserved_boundary.state_fingerprint
    engine._sync_device(device)
    started = time.perf_counter()
    (
        reference_logits,
        reference_forward_seconds,
        reference_clone_seconds,
        reference_baseline,
        reference_peak,
    ) = engine._probe_logits(model, preserved_boundary, None)
    preserved_arm = engine.continue_from_boundary(
        model,
        preserved_boundary,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        action=ACTION,
        reference_logits=reference_logits,
        enable_probe=True,
    )
    engine._sync_device(device)
    preserved_seconds = time.perf_counter() - started
    preserved_peak = _maximum(
        reference_peak, preserved_arm.device_peak_allocated_bytes
    )
    (
        preserved_state_after,
        model_after_preserved,
        rng_after_preserved,
        preserved_model_tensor_count,
        source_validation_seconds,
    ) = engine._inspect_boundary_state(model, preserved_boundary)
    preserved_record = {
        "mode": "preserved_reference_with_probe",
        "reference_cache_preserved": True,
        "probe_enabled": True,
        "baseline_allocated_bytes": reference_baseline,
        "peak_allocated_bytes": preserved_peak,
        "peak_increment_bytes": _difference(
            preserved_peak, reference_baseline
        ),
        "wall_seconds": preserved_seconds,
        "reference_sandbox": {
            "clone_seconds": reference_clone_seconds,
            "forward_seconds": reference_forward_seconds,
            "baseline_allocated_bytes": reference_baseline,
            "peak_allocated_bytes": reference_peak,
        },
        "action_phase": preserved_arm.to_dict(),
        "source_validation_seconds_outside_peak": source_validation_seconds,
        "residual_gpu_state": (
            "model, input tensor, preserved boundary cache, and reference "
            "logits during the action phase"
        ),
    }

    probe = preserved_arm.probe
    assert probe is not None
    start_state_equal = (
        direct_start_fingerprint == preserved_start_fingerprint
    )
    action_equal = (
        direct_compression.before_fingerprint
        == preserved_arm.compression.before_fingerprint
        and direct_compression.after_fingerprint
        == preserved_arm.compression.after_fingerprint
        and direct_compression.kept_indices
        == preserved_arm.compression.kept_indices
    )
    continuation_equal = _continuations_equal(
        direct_continuation, preserved_arm.continuation
    )
    direct_runtime_unchanged = (
        model_after_direct == direct_model_fingerprint
        and direct_observed_model_tensor_count
        == direct_expected_model_tensor_count
        and rng_after_direct == direct_rng_fingerprint
    )
    preserved_source_unchanged = (
        preserved_state_after == preserved_start_fingerprint
        and model_after_preserved
        == preserved_boundary.model_state_fingerprint
        and preserved_model_tensor_count
        == preserved_boundary.model_tensor_count
        and rng_after_preserved == preserved_boundary.rng_state.fingerprint
    )
    checks = {
        "identical_start_state": start_state_equal,
        "identical_action_transform": action_equal,
        "exact_continuation_parity": continuation_equal,
        "direct_model_and_rng_unchanged": direct_runtime_unchanged,
        "preserved_source_model_rng_unchanged": preserved_source_unchanged,
        "probe_finite": probe.finite,
    }
    return {
        "schema_version": "herald_v3.engineering_memory.v1",
        "passed": all(checks.values()),
        "configuration": {
            "action": ACTION.to_dict(),
            "decision_tokens": 32,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "mode_order": [
                "live_no_reference_no_probe",
                "preserved_reference_with_probe",
            ],
        },
        "checks": checks,
        "modes": {
            "live_no_reference_no_probe": direct_record,
            "preserved_reference_with_probe": preserved_record,
        },
        "comparison": {
            "baseline_allocated_difference_bytes": _difference(
                reference_baseline, direct_baseline
            ),
            "peak_allocated_difference_bytes": _difference(
                preserved_peak, direct_peak
            ),
            "combined_feature_and_preservation_cost": True,
        },
        "limitations": [
            "Peak allocated excludes CUDA reserved memory.",
            "The two-mode difference combines reference preservation and "
            "probe acquisition; existing engine evidence supplies the "
            "probe-on versus probe-off comparison with reference preserved.",
            "Mode order is fixed and recorded; repeat runs are timing "
            "diagnostics, not independent research samples.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompts",
        default=str(PROJECT_ROOT / "data/engineering-prompts.json"),
    )
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("CUDA is required before loading the model")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    prompt_ids = [args.prompt_id] if args.prompt_id is not None else None
    manifest = load_prompt_manifest(
        args.prompts,
        limit=1,
        prompt_ids=prompt_ids,
    )
    prompt = manifest.prompts[0]
    model, tokenizer = runner.load_offline_model(args.model)
    input_ids = runner.tokenize_prompt(tokenizer, prompt)
    eos_ids = runner._eos_ids(model, tokenizer)
    measurement = measure_memory_modes(
        model,
        input_ids,
        eos_ids=eos_ids,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    measurement.update(
        {
            "prompt": prompt.to_dict(),
            "tokenization": {
                "input_ids": [
                    int(item) for item in input_ids.detach().cpu()[0].tolist()
                ],
                "input_length": int(input_ids.shape[1]),
                "chat_template_verified": True,
            },
            "model": engine.model_signature(model),
            "environment": runner._environment_manifest(
                model,
                tokenizer,
                {
                    "prompt_manifest_path": str(Path(args.prompts).resolve()),
                    "prompt_manifest_sha256": manifest.fingerprint,
                },
                args.seed,
            ),
            "source_manifest": {
                "engineering_package": runner._source_manifest(),
                "diagnostic_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": runner._file_sha256(Path(__file__)),
                },
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner._write_json(args.output, measurement)
    print(
        json.dumps(
            {
                "passed": measurement["passed"],
                "output": str(args.output),
                "checks": measurement["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if measurement["passed"] is True else 1


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _begin_peak(device: torch.device) -> int | None:
    engine._sync_device(device)
    if device.type != "cuda":
        return None
    baseline = int(torch.cuda.memory_allocated(device))
    torch.cuda.reset_peak_memory_stats(device)
    return baseline


def _finish_peak(device: torch.device) -> int | None:
    engine._sync_device(device)
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _difference(left: int | None, right: int | None) -> int | None:
    if left is None or right is None:
        return None
    return left - right


def _maximum(left: int | None, right: int | None) -> int | None:
    values = [value for value in (left, right) if value is not None]
    return max(values) if values else None


def _continuations_equal(
    left: engine.ContinuationResult,
    right: engine.ContinuationResult,
) -> bool:
    return bool(
        left.token_ids == right.token_ids
        and left.termination_reason == right.termination_reason
        and left.final_cache_lengths == right.final_cache_lengths
        and left.final_cache_bytes == right.final_cache_bytes
        and left.final_cache_fingerprint == right.final_cache_fingerprint
    )


if __name__ == "__main__":
    raise SystemExit(main())
