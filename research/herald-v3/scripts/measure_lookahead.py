#!/usr/bin/env python3
"""Measure the bounded reference-prefix lookahead for one prompt."""

# The script is executable without an installed editable package.
# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

from herald_v3.engineering import engine
from herald_v3.engineering import lookahead
from herald_v3.engineering import runner
from herald_v3.engineering.prompts import load_prompt_manifest

DEFAULT_PROMPTS = PROJECT_ROOT / "data/engineering-prompts.json"
DEFAULT_MAX_STEPS = lookahead.MAX_LOOKAHEAD_STEPS
SHORT_CONTINUATION_TOKENS = 4
PARITY_TOLERANCE = 1e-6


def measure_prompt(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    *,
    removal_fraction: float = 0.5,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
) -> dict[str, object]:
    """Build one boundary and measure one action without collecting output."""
    _set_seed(seed)
    input_ids = runner.tokenize_prompt(tokenizer, prompt)
    eos_ids = runner._eos_ids(model, tokenizer)
    boundary = engine.build_boundary(model, input_ids, eos_ids=eos_ids)
    action = engine.ActionSpec("knorm", removal_fraction)
    continuation_before = _short_continuation(
        model,
        boundary,
        action,
        eos_ids=eos_ids,
        max_new_tokens=len(boundary.generated_ids)
        + SHORT_CONTINUATION_TOKENS,
    )
    result = lookahead.run_lookahead(
        model,
        boundary,
        action,
        eos_ids=eos_ids,
        max_steps=max_steps,
    )
    step0_parity = _step_zero_parity(model, boundary, result)
    continuation_after = _short_continuation(
        model,
        boundary,
        action,
        eos_ids=eos_ids,
        max_new_tokens=len(boundary.generated_ids)
        + SHORT_CONTINUATION_TOKENS,
    )
    noop = lookahead.run_lookahead(
        model,
        boundary,
        engine.ActionSpec("knorm", 0.0),
        eos_ids=eos_ids,
        max_steps=max_steps,
    )
    diagnostics = {
        "step0_parity": step0_parity,
        "continuation_before_probe": continuation_before,
        "continuation_after_probe": continuation_after,
        "continuation_transparency": {
            "passed": (
                continuation_before["token_ids"]
                == continuation_after["token_ids"]
                and continuation_before["sha256"]
                == continuation_after["sha256"]
                and continuation_before["termination_reason"]
                == continuation_after["termination_reason"]
                and continuation_before["final_cache_fingerprint"]
                == continuation_after["final_cache_fingerprint"]
                and continuation_before["source_preserved"]
                and continuation_after["source_preserved"]
            ),
            "token_ids_equal": continuation_before["token_ids"]
            == continuation_after["token_ids"],
            "sha256_equal": continuation_before["sha256"]
            == continuation_after["sha256"],
            "termination_equal": continuation_before["termination_reason"]
            == continuation_after["termination_reason"],
            "final_cache_fingerprint_equal": continuation_before[
                "final_cache_fingerprint"
            ]
            == continuation_after["final_cache_fingerprint"],
            "source_preserved_before": continuation_before[
                "source_preserved"
            ],
            "source_preserved_after": continuation_after["source_preserved"],
        },
        "timing_seconds": {
            "continuation_before_probe": continuation_before["wall_seconds"],
            "step0_parity": step0_parity["wall_seconds"],
            "continuation_after_probe": continuation_after["wall_seconds"],
            "total": cast(float, continuation_before["wall_seconds"])
            + cast(float, step0_parity["wall_seconds"])
            + cast(float, continuation_after["wall_seconds"]),
        },
    }
    checks = {
        "lookahead": result.passed,
        "noop_control": noop.passed,
        "step0_parity": step0_parity["passed"],
        "continuation_transparency": diagnostics["continuation_transparency"][
            "passed"
        ],
    }
    return {
        "schema_version": "herald_v3.engineering.lookahead_measurement.v1",
        "passed": all(checks.values()),
        "checks": checks,
        "prompt": prompt.to_dict(),
        "tokenization": {
            "input_ids": [
                int(item) for item in input_ids.detach().cpu()[0].tolist()
            ],
            "input_length": int(input_ids.shape[1]),
            "chat_template_verified": True,
        },
        "configuration": {
            "decision_tokens": 32,
            "first_output_index": lookahead.FIRST_OUTPUT_INDEX,
            "max_steps": max_steps,
            "action": action.to_dict(),
            "seed": seed,
            "eos_ids": sorted(eos_ids),
            "probe_prefix": "reference_greedy_tokens_shared_by_all_arms",
            "stored_outcomes": False,
        },
        "lookahead": result.to_dict(),
        "noop_control": noop.to_dict(),
        "diagnostics": diagnostics,
        "limitations": [
            "The probe follows a synthetic reference greedy prefix.",
            "No generated answer, score, or future outcome is collected.",
            "Timing includes source-state validation and cleanup.",
            "CUDA peak reports allocated bytes, not reserved bytes.",
            "Transparency uses a four-token action continuation control.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--removal-fraction", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if not 1 <= args.max_steps <= DEFAULT_MAX_STEPS:
        parser.error(f"--max-steps must be in [1, {DEFAULT_MAX_STEPS}]")

    prompt_ids = [args.prompt_id] if args.prompt_id is not None else None
    manifest = load_prompt_manifest(
        args.prompts,
        limit=1,
        prompt_ids=prompt_ids,
    )
    prompt = manifest.prompts[0]
    model, tokenizer = runner.load_offline_model(args.model)
    measurement = measure_prompt(
        model,
        tokenizer,
        prompt,
        removal_fraction=args.removal_fraction,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    measurement.update(
        {
            "model": engine.model_signature(model),
            "environment": runner._environment_manifest(
                model,
                tokenizer,
                {
                    "prompt_manifest_path": str(args.prompts.resolve()),
                    "prompt_manifest_sha256": manifest.fingerprint,
                },
                args.seed,
            ),
            "source_manifest": {
                "engineering_package": runner._source_manifest(),
                "measurement_script": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": runner._file_sha256(Path(__file__)),
                },
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    runner._write_json(args.output, measurement)
    summary = measurement["lookahead"]
    noop = measurement["noop_control"]
    assert isinstance(summary, dict)
    assert isinstance(noop, dict)
    print(
        json.dumps(
            {
                "passed": measurement["passed"],
                "output": str(args.output),
                "realized_steps": summary["realized_steps"],
                "checks": {
                    "action": summary["checks"],
                    "noop_control": noop["checks"],
                    "measurement": measurement["checks"],
                },
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


def _short_continuation(
    model: Any,
    boundary: engine.BoundaryState,
    action: engine.ActionSpec,
    *,
    eos_ids: set[int] | frozenset[int],
    max_new_tokens: int,
) -> dict[str, object]:
    """Run a short control and retain IDs plus a digest, never its answer."""
    device = engine._model_device(model)
    caller_rng = engine._capture_rng(device)
    source_state = boundary.state_fingerprint
    started = time.perf_counter()
    try:
        arm = engine.continue_from_boundary(
            model,
            boundary,
            max_new_tokens=max_new_tokens,
            eos_ids=eos_ids,
            action=action,
            enable_probe=False,
        )
        new_ids = arm.continuation.token_ids[len(boundary.generated_ids) :]
        token_bytes = json.dumps(list(new_ids), separators=(",", ":")).encode(
            "ascii"
        )
    finally:
        engine._restore_rng(caller_rng, device)
        state_after, _, _, _, validation_seconds = (
            engine._inspect_boundary_state(model, boundary)
        )
        engine._sync_device(device)
    return {
        "token_ids": list(new_ids),
        "token_count": len(new_ids),
        "sha256": hashlib.sha256(token_bytes).hexdigest(),
        "termination_reason": arm.continuation.termination_reason,
        "final_cache_fingerprint": arm.continuation.final_cache_fingerprint,
        "final_cache_lengths": list(arm.continuation.final_cache_lengths),
        "max_new_tokens": max_new_tokens,
        "source_preserved": state_after == source_state,
        "validation_seconds": validation_seconds,
        "wall_seconds": time.perf_counter() - started,
    }


def _step_zero_parity(
    model: Any,
    boundary: engine.BoundaryState,
    measured: lookahead.LookaheadResult,
) -> dict[str, object]:
    """Compare the old one-step probe with lookahead step zero."""
    device = engine._model_device(model)
    caller_rng = engine._capture_rng(device)
    started = time.perf_counter()
    try:
        original = engine.probe_action(model, boundary, measured.action)
    finally:
        engine._restore_rng(caller_rng, device)
        engine._sync_device(device)
    if not measured.steps:
        return {
            "passed": False,
            "checks": {"step_zero_present": False},
            "wall_seconds": time.perf_counter() - started,
        }
    observed = measured.steps[0].probe
    checks: dict[str, bool] = {
        "step_zero_present": measured.steps[0].output_index == 32,
        "action": observed.action.to_dict() == original.action.to_dict(),
        "finite": observed.finite == original.finite,
        "argmax_match": observed.argmax_match == original.argmax_match,
        "reference_argmax": observed.reference_argmax
        == original.reference_argmax,
        "action_argmax": observed.action_argmax == original.action_argmax,
    }
    for field_name in (
        "js_divergence",
        "reference_entropy",
        "action_entropy",
        "reference_top2_margin",
        "action_top2_margin",
        "reference_probability_sum",
        "action_probability_sum",
        "max_probability_difference",
        "max_logit_difference",
    ):
        observed_value = float(getattr(observed, field_name))
        original_value = float(getattr(original, field_name))
        checks[field_name] = abs(observed_value - original_value) <= (
            PARITY_TOLERANCE
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "original_probe": original.to_dict(),
        "lookahead_step_zero": observed.to_dict(),
        "tolerance": PARITY_TOLERANCE,
        "wall_seconds": time.perf_counter() - started,
    }


if __name__ == "__main__":
    raise SystemExit(main())
