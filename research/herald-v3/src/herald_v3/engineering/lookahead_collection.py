"""Outcome-free, resumable collection of paired H8 lookahead probes.

The collector builds the already accepted token-32 boundary and calls
``run_lookahead`` for the two locked Knorm actions plus one no-op control. It
never continues an answer, scores text, or reads a completed outcome.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from herald_v3.engineering import engine, lookahead, runner
from herald_v3.engineering.prompts import EngineeringPrompt, PromptManifest

SCHEMA_VERSION = "herald_v3.lookahead_collection.v1"
RECORD_SCHEMA_VERSION = "herald_v3.lookahead_collection_record.v1"
LOCK_SCHEMA_VERSION = "herald_v3.lookahead_protocol_lock.v1"
LOCK_STATUS = "frozen_implementation_pending_owner_collection_approval"
SEAL_SCHEMA_VERSION = "herald_v3.lookahead_prediction_seal.v1"
OUTCOME_COLLECTION_SCHEMA = "herald_v3.lookahead_outcomes.v1"
TRAINING_REUSE_SCHEMA_VERSION = "herald_v3.lookahead_training_reuse.v1"
TOKENIZER_ASSET_NAMES = {
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
LOCK_NAME = "collection-lock.json"
CHECKPOINT_NAME = "checkpoint.json"
RECORD_DIRECTORY = ""
ACTION_RATIOS = (0.25, 0.5)
ACTION_IDS = tuple(f"knorm:{ratio:.6g}" for ratio in ACTION_RATIOS)
NOOP_ID = "knorm:0"
TRAIN_PROMPT_COUNT = 120
TEST_PROMPT_COUNT = 76
ACTION_ROW_COUNT = 2
SEED = 0
DECISION_TOKENS = 32
MAX_STEPS = lookahead.MAX_LOOKAHEAD_STEPS
MAX_NEW_TOKENS = 1024


class LookaheadCollectionError(RuntimeError):
    """Raised when a locked collection cannot be started or resumed."""


def collect_lookahead(
    model: Any,
    tokenizer: Any,
    manifest: PromptManifest,
    manifest_path: str | Path,
    lock_path: str | Path,
    output: str | Path,
    *,
    phase: str,
    train_reuse: str | Path | None = None,
    expected_prompt_count: int | None = None,
) -> dict[str, object]:
    """Collect a complete locked train or test manifest with safe resume.

    ``expected_prompt_count`` is an internal fixture escape hatch for tiny
    CPU tests. The CLI never supplies it, so production train and test runs
    enforce exactly 120 and 76 prompts respectively.
    """
    if phase not in {"train", "test"}:
        raise ValueError("phase must be 'train' or 'test'")
    expected_count = (
        _expected_count(phase)
        if expected_prompt_count is None
        else expected_prompt_count
    )
    if expected_count <= 0:
        raise ValueError("expected_prompt_count must be positive")
    manifest_file = Path(manifest_path).resolve()
    protocol_lock_file = Path(lock_path).resolve()
    output_path = Path(output).resolve()
    manifest_bytes = _read_bytes(manifest_file)
    protocol_lock_bytes = _read_bytes(protocol_lock_file)
    protocol_lock = _read_object(protocol_lock_bytes, protocol_lock_file)
    _validate_protocol_lock(
        protocol_lock,
        manifest,
        manifest_file,
        phase,
        expected_count,
        strict=expected_prompt_count is None,
    )
    _validate_manifest(manifest, phase, expected_count)
    _validate_manifest_file(manifest_file, manifest)
    train_reuse_file = (
        Path(train_reuse).resolve() if train_reuse is not None else None
    )
    _validate_training_reuse_lock(
        protocol_lock,
        protocol_lock_file,
        train_reuse_file,
        phase,
        strict=expected_prompt_count is None,
    )
    reuse = _load_train_reuse(
        train_reuse_file, manifest, phase, expected_count
    )
    torch = runner._torch()
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    observed_model = engine.model_signature(model)
    eos_ids = runner._eos_ids(model, tokenizer)
    _validate_runtime_lock(
        protocol_lock,
        observed_model,
        eos_ids,
        tokenizer_name=_tokenizer_name(tokenizer),
        tokenizer=tokenizer,
        strict=expected_prompt_count is None,
    )
    expected_lock = _collection_lock(
        phase=phase,
        manifest=manifest,
        manifest_file=manifest_file,
        manifest_bytes=manifest_bytes,
        protocol_lock_file=protocol_lock_file,
        protocol_lock_bytes=protocol_lock_bytes,
        train_reuse_file=train_reuse_file,
        model=observed_model,
        tokenizer=tokenizer,
        eos_ids=eos_ids,
        output_path=output_path,
        expected_count=expected_count,
    )
    lock_file, checkpoint_file, checkpoint = _prepare_output(
        output_path, expected_lock
    )
    lock_sha256 = _sha256(lock_file.read_bytes())
    ledger = _recover_ledger(
        output_path,
        checkpoint,
        manifest,
        lock_sha256,
        expected_count,
    )
    processed_ids = {cast(str, item["prompt_id"]) for item in ledger}
    for prompt in manifest.prompts:
        if prompt.prompt_id in processed_ids:
            continue
        record_path = (
            output_path / RECORD_DIRECTORY / prompt.prompt_id / "record.json"
        )
        try:
            record = _collect_prompt(
                model,
                tokenizer,
                prompt,
                manifest,
                phase=phase,
                eos_ids=eos_ids,
                reuse=reuse,
                manifest_file_sha256=_sha256(manifest_bytes),
                model_signature=observed_model,
                tokenizer_name=_tokenizer_name(tokenizer),
            )
            record_path.parent.mkdir(parents=True, exist_ok=False)
            status = record["status"]
            if status == "ineligible_early_eos":
                early_eos = record.get("early_eos")
                if not isinstance(early_eos, Mapping):
                    raise LookaheadCollectionError(
                        "early-EOS record is missing its reason"
                    )
                entry = {
                    "prompt_id": prompt.prompt_id,
                    "status": status,
                    "reason": early_eos.get("reason"),
                    "position": early_eos.get("position"),
                    "source_hash": prompt.prompt_text_sha256,
                    "action_row_count": 0,
                }
                record_path.parent.rmdir()
            else:
                _write_json(record_path, record)
                record_hash = _sha256(record_path.read_bytes())
                entry = {
                    "prompt_id": prompt.prompt_id,
                    "status": status,
                    "record": str(record_path.relative_to(output_path)),
                    "record_sha256": record_hash,
                    "source_hash": prompt.prompt_text_sha256,
                    "action_row_count": len(
                        cast(list[object], record["action_rows"])
                    ),
                }
            ledger.append(entry)
            _write_checkpoint(
                checkpoint_file,
                _summary(
                    "running",
                    phase,
                    manifest,
                    ledger,
                    expected_count,
                    lock_sha256,
                    current_prompt_id=prompt.prompt_id,
                ),
            )
        except Exception as error:
            failure = _summary(
                "failed",
                phase,
                manifest,
                ledger,
                expected_count,
                lock_sha256,
                current_prompt_id=prompt.prompt_id,
            )
            failure["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "prompt_id": prompt.prompt_id,
            }
            _write_checkpoint(checkpoint_file, failure)
            if isinstance(error, LookaheadCollectionError):
                raise
            raise LookaheadCollectionError(
                f"lookahead collection failed at {prompt.prompt_id}: {error}"
            ) from error
    if len(ledger) != expected_count:
        status = "insufficient_coverage"
    else:
        status = "completed"
    final = _summary(
        status,
        phase,
        manifest,
        ledger,
        expected_count,
        lock_sha256,
        current_prompt_id=None,
    )
    _write_checkpoint(checkpoint_file, final)
    return final


def _collect_prompt(
    model: Any,
    tokenizer: Any,
    prompt: EngineeringPrompt,
    manifest: PromptManifest,
    *,
    phase: str,
    eos_ids: set[int] | frozenset[int],
    reuse: Mapping[str, object] | None,
    manifest_file_sha256: str,
    model_signature: Mapping[str, object],
    tokenizer_name: str,
) -> dict[str, object]:
    started = time.perf_counter()
    input_ids = runner.tokenize_prompt(tokenizer, prompt)
    try:
        boundary = engine.build_boundary(
            model,
            input_ids,
            decision_tokens=DECISION_TOKENS,
            eos_ids=eos_ids,
        )
    except engine.EarlyBoundaryTermination as error:
        if phase == "train":
            raise LookaheadCollectionError(
                f"training prompt ended before boundary: {prompt.prompt_id}"
            ) from error
        return {
            "schema_version": RECORD_SCHEMA_VERSION,
            "phase": phase,
            "prompt_id": prompt.prompt_id,
            "status": "ineligible_early_eos",
            "prompt": prompt.to_dict(),
            "prompt_manifest_sha256": manifest.fingerprint,
            "prompt_manifest_file_sha256": manifest_file_sha256,
            "provenance": _prompt_provenance(prompt, manifest),
            "configuration": _configuration(eos_ids),
            "boundary": None,
            "lookahead": None,
            "action_rows": [],
            "early_eos": {
                "reason": error.reason,
                "generated_token_ids": list(error.generated_ids),
                "position": len(error.generated_ids) - 1,
            },
            "checks": {
                "pre_boundary_eos_recorded": True,
                "outcomes_collected": False,
            },
            "collection_seconds": time.perf_counter() - started,
            "outcomes_collected": False,
        }

    if phase == "train":
        if reuse is None:
            raise LookaheadCollectionError(
                "train collection requires provenance-matched train reuse"
            )
        _validate_reuse_row(
            reuse,
            prompt,
            manifest,
            boundary,
            model_signature=model_signature,
            tokenizer_name=tokenizer_name,
            eos_ids=eos_ids,
        )

    actions = {
        f"knorm:{ratio:.6g}": lookahead.run_lookahead(
            model,
            boundary,
            engine.ActionSpec("knorm", ratio),
            eos_ids=eos_ids,
            max_steps=MAX_STEPS,
        )
        for ratio in ACTION_RATIOS
    }
    noop = lookahead.run_lookahead(
        model,
        boundary,
        engine.ActionSpec("knorm", 0.0),
        eos_ids=eos_ids,
        max_steps=MAX_STEPS,
    )
    pair = _validate_pair(actions, noop)
    serialized_actions = {
        action_id: _serialized_action(actions[action_id])
        for action_id in ACTION_IDS
    }
    action_rows = [
        _action_row(action_id, actions[action_id], boundary)
        for action_id in ACTION_IDS
    ]
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "phase": phase,
        "prompt_id": prompt.prompt_id,
        "status": "eligible",
        "prompt": prompt.to_dict(),
        "prompt_manifest_sha256": manifest.fingerprint,
        "prompt_manifest_file_sha256": manifest_file_sha256,
        "provenance": _prompt_provenance(prompt, manifest),
        "configuration": _configuration(eos_ids),
        "boundary": boundary.to_dict(),
        "boundary_cache_fingerprint": engine.cache_fingerprint(
            boundary.cache
        ),
        "lookahead": {
            "actions": serialized_actions,
            "noop_controls": {NOOP_ID: noop.to_dict()},
            "shared_reference": pair["shared_reference"],
        },
        "actions": serialized_actions,
        "action_rows": action_rows,
        "checks": pair["checks"],
        "collection_seconds": time.perf_counter() - started,
        "outcomes_collected": False,
    }


def _validate_pair(
    actions: Mapping[str, lookahead.LookaheadResult],
    noop: lookahead.LookaheadResult,
) -> dict[str, object]:
    if set(actions) != set(ACTION_IDS):
        raise LookaheadCollectionError("both locked action rows are required")
    if not noop.passed:
        raise LookaheadCollectionError("no-op lookahead control failed")
    if any(not result.passed for result in actions.values()):
        raise LookaheadCollectionError("a Knorm lookahead probe failed")

    def trajectory(
        result: lookahead.LookaheadResult,
    ) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                step.output_index,
                step.input_token_id,
                step.input_position,
                step.reference_argmax,
                step.reference_argmax_is_eos,
            )
            for step in result.steps
        )

    reference = trajectory(noop)
    same_as_noop = all(
        trajectory(result) == reference for result in actions.values()
    )
    same_length = (
        len(
            {
                len(noop.steps),
                *(len(result.steps) for result in actions.values()),
            }
        )
        == 1
    )
    realized_steps = len(noop.steps)
    l1_valid = 1 <= realized_steps <= MAX_STEPS
    reference_eos = reference[-1][4] if reference else None
    reference_eos_equal = all(
        trajectory(result) and trajectory(result)[-1][4] == reference_eos
        for result in actions.values()
    )
    source_fingerprints = {
        noop.boundary_state_fingerprint_before,
        *(
            result.boundary_state_fingerprint_before
            for result in actions.values()
        ),
    }
    if (
        not same_as_noop
        or not same_length
        or not l1_valid
        or not reference_eos_equal
        or len(source_fingerprints) != 1
    ):
        raise LookaheadCollectionError(
            "paired actions do not share the reference trajectory and L"
        )
    shared_reference = {
        "output_indices": [item[0] for item in reference],
        "forced_input_token_ids": [item[1] for item in reference],
        "input_positions": [item[2] for item in reference],
        "reference_argmax_token_ids": [item[3] for item in reference],
        "reference_argmax_is_eos": [item[4] for item in reference],
        "L": realized_steps,
        "reference_eos_output_index": (
            reference[-1][0] if reference and reference[-1][4] else None
        ),
        "reference_state_fingerprint": noop.boundary_state_fingerprint_before,
        "reference_probe_sha256": _sha256_text(
            json.dumps(reference, separators=(",", ":"))
        ),
    }
    return {
        "shared_reference": shared_reference,
        "checks": {
            "actions_complete": True,
            "action_lookahead_passed": True,
            "noop_control_passed": True,
            "reference_trajectory_equal": same_as_noop,
            "reference_L_equal": same_length,
            "reference_EOS_equal": reference_eos_equal,
            "reference_state_equal": len(source_fingerprints) == 1,
            "L1_valid": l1_valid,
            "finite_features": all(
                step.probe.finite
                for result in actions.values()
                for step in result.steps
            ),
            "outcomes_collected": False,
        },
    }


def _action_row(
    action_id: str,
    result: lookahead.LookaheadResult,
    boundary: engine.BoundaryState,
) -> dict[str, object]:
    if not result.steps:
        raise LookaheadCollectionError(
            f"empty lookahead result for {action_id}"
        )
    first = result.steps[0].probe
    delayed = [step.probe.js_divergence for step in result.steps[1:]]
    return {
        "action_id": action_id,
        "removal_fraction": result.action.removal_fraction,
        "prompt_token_count": float(boundary.prompt_ids.shape[1]),
        "decision_index32": 32.0,
        "pre_action_cache_size": float(boundary.cache_bytes),
        "immediate_js": first.js_divergence,
        "js_divergence": first.js_divergence,
        "mean_delayed_js": sum(delayed) / len(delayed) if delayed else 0.0,
        "L": len(result.steps),
        "has_delayed": int(len(result.steps) > 1),
        "reference_entropy": first.reference_entropy,
        "reference_top2_margin": first.reference_top2_margin,
        "action_reference_entropy_delta": (
            first.action_entropy - first.reference_entropy
        ),
        "action_reference_margin_delta": (
            first.action_top2_margin - first.reference_top2_margin
        ),
        "argmax_match": first.argmax_match,
        "reference_argmax": first.reference_argmax,
        "action_argmax": first.action_argmax,
        "reference_state_fingerprint": (
            result.boundary_state_fingerprint_before
        ),
        "reference_probe_sha256": _sha256_text(
            json.dumps(
                {
                    "output_indices": [
                        step.output_index for step in result.steps
                    ],
                    "reference_argmax": [
                        step.reference_argmax for step in result.steps
                    ],
                    "forced_inputs": [
                        step.input_token_id for step in result.steps
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ),
    }


def _serialized_action(
    result: lookahead.LookaheadResult,
) -> dict[str, object]:
    """Add the derived H8 fields consumed by the fixed evaluator."""
    payload = result.to_dict()
    delayed = [step.probe.js_divergence for step in result.steps[1:]]
    eos_positions = [
        index
        for index, step in enumerate(result.steps)
        if step.reference_argmax_is_eos
    ]
    state = result.boundary_state_fingerprint_before
    reference_eos_position = eos_positions[0] if eos_positions else None
    reference_hash = _sha256_text(
        json.dumps(
            {
                "forced_input_token_ids": [
                    step.input_token_id for step in result.steps
                ],
                "reference_argmax_token_ids": [
                    step.reference_argmax for step in result.steps
                ],
                "reference_eos_position": reference_eos_position,
                "reference_state_fingerprint": state,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    payload["mean_delayed_js"] = (
        sum(delayed) / len(delayed) if delayed else 0.0
    )
    payload["reference_eos_position"] = reference_eos_position
    payload["reference_probe_hash"] = reference_hash
    return payload


def _configuration(eos_ids: set[int] | frozenset[int]) -> dict[str, object]:
    return {
        "mode": "probe_only",
        "decision_tokens": DECISION_TOKENS,
        "max_lookahead_steps": MAX_STEPS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "actions": [
            {"name": "knorm", "removal_fraction": ratio}
            for ratio in ACTION_RATIOS
        ],
        "noop_control": {"name": "knorm", "removal_fraction": 0.0},
        "seed": SEED,
        "eos_ids": sorted(eos_ids),
        "stored_outcomes": False,
        "stored_scores": False,
        "stored_full_answers": False,
    }


def _prompt_provenance(
    prompt: EngineeringPrompt, manifest: PromptManifest
) -> dict[str, object]:
    return {
        "prompt_id": prompt.prompt_id,
        "official_key": prompt.key,
        "prompt_text_utf8_sha256": prompt.prompt_text_sha256,
        "user_prompt_utf8_sha256": prompt.user_prompt_sha256,
        "manifest_source_sha256": manifest.source_sha256,
        "official_dataset_sha256": manifest.official_dataset_sha256,
        "fold": prompt.fold,
        "fold_hash": prompt.fold_hash,
        "split_hash": prompt.split_hash,
    }


def _validate_protocol_lock(
    lock: Mapping[str, object],
    manifest: PromptManifest,
    manifest_file: Path,
    phase: str,
    expected_count: int,
    *,
    strict: bool = True,
) -> None:
    if strict:
        _validate_lock_shape(lock)
    entry = _manifest_lock_entry(lock, phase)
    if strict and entry.get("path") != manifest_file.name:
        raise LookaheadCollectionError(
            f"{phase} manifest path does not match protocol lock"
        )
    ids = entry.get("prompt_ids")
    if not isinstance(ids, list) or ids != [
        p.prompt_id for p in manifest.prompts
    ]:
        raise LookaheadCollectionError(
            f"{phase} manifest IDs do not match protocol lock"
        )
    if len(ids) != expected_count:
        raise LookaheadCollectionError(
            f"{phase} manifest must contain exactly {expected_count} IDs"
        )
    expected_file_hash = entry.get("file_sha256")
    if strict and not isinstance(expected_file_hash, str):
        raise LookaheadCollectionError(
            f"{phase} manifest file hash is missing from lock"
        )
    if isinstance(expected_file_hash, str) and (
        expected_file_hash != _sha256(manifest_file.read_bytes())
    ):
        raise LookaheadCollectionError(
            f"{phase} manifest hash does not match lock"
        )
    expected_manifest_hash = entry.get("manifest_sha256")
    if strict and not isinstance(expected_manifest_hash, str):
        raise LookaheadCollectionError(
            f"{phase} manifest fingerprint is missing from lock"
        )
    if isinstance(expected_manifest_hash, str) and (
        expected_manifest_hash != manifest.fingerprint
    ):
        raise LookaheadCollectionError(
            f"{phase} manifest fingerprint does not match lock"
        )
    config = _lock_configuration(lock)
    if (
        not isinstance(config.get("seed"), int)
        or isinstance(config["seed"], bool)
        or config["seed"] != SEED
    ):
        raise LookaheadCollectionError("protocol lock seed is not zero")
    if config.get("decision_tokens") != DECISION_TOKENS:
        raise LookaheadCollectionError(
            "protocol lock boundary is not 32 tokens"
        )
    actions = config.get("actions")
    if not _configuration_actions_match(actions):
        raise LookaheadCollectionError(
            "protocol lock actions are not Knorm .25/.50"
        )
    steps = config.get("max_lookahead_steps")
    if steps != MAX_STEPS:
        raise LookaheadCollectionError(
            "protocol lock lookahead cap is not eight"
        )
    if lock.get("outcomes_collected") is True:
        raise LookaheadCollectionError(
            "protocol lock already contains outcomes"
        )


def _validate_lock_shape(lock: Mapping[str, object]) -> None:
    """Validate the frozen lock before loading model or reuse data."""
    required = {
        "schema_version",
        "status",
        "manifests",
        "training_reuse",
        "boundary",
        "actions",
        "configuration",
        "features",
        "estimator",
        "evaluation",
        "model",
        "tokenizer",
        "generation_environment",
        "software",
        "source_manifest",
        "scorer_provenance",
        "prediction_seal_schema",
        "outcome_collection_schema",
        "outcomes_collected",
        "proposal_sha256",
        "roster_derivation_sha256",
    }
    if set(lock) != required:
        raise LookaheadCollectionError("protocol lock keys are not exact")
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise LookaheadCollectionError("protocol lock schema is invalid")
    if lock.get("status") != LOCK_STATUS:
        raise LookaheadCollectionError("protocol lock status is invalid")
    if lock.get("outcomes_collected") is not False:
        raise LookaheadCollectionError(
            "protocol lock already contains outcomes"
        )

    manifests = lock["manifests"]
    if not isinstance(manifests, Mapping) or set(manifests) != {
        "train",
        "test",
    }:
        raise LookaheadCollectionError(
            "protocol lock manifests are not exact"
        )
    for name in ("train", "test"):
        entry = manifests[name]
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "file_sha256",
            "manifest_sha256",
            "prompt_ids",
        }:
            raise LookaheadCollectionError(
                f"protocol lock {name} manifest entry is not exact"
            )
        if (
            not isinstance(entry["path"], str)
            or not entry["path"]
            or not isinstance(entry["file_sha256"], str)
            or not _is_sha256(entry["file_sha256"])
            or not isinstance(entry["manifest_sha256"], str)
            or not _is_sha256(entry["manifest_sha256"])
            or not isinstance(entry["prompt_ids"], list)
            or not all(isinstance(item, str) for item in entry["prompt_ids"])
        ):
            raise LookaheadCollectionError(
                f"protocol lock {name} manifest entry is malformed"
            )

    reuse = lock["training_reuse"]
    if (
        not isinstance(reuse, Mapping)
        or set(reuse) != {"path", "sha256"}
        or not isinstance(reuse["path"], str)
        or not reuse["path"]
        or Path(reuse["path"]).is_absolute()
        or ".." in Path(reuse["path"]).parts
        or not isinstance(reuse["sha256"], str)
        or not _is_sha256(reuse["sha256"])
    ):
        raise LookaheadCollectionError(
            "protocol lock training reuse reference is malformed"
        )

    boundary = lock["boundary"]
    if (
        not isinstance(boundary, Mapping)
        or set(boundary)
        != {
            "committed_output_tokens",
            "first_affected_prediction_output_index",
            "pending_token_output_index",
        }
        or boundary != {
            "committed_output_tokens": 32,
            "first_affected_prediction_output_index": 32,
            "pending_token_output_index": 31,
        }
    ):
        raise LookaheadCollectionError("protocol lock boundary is not exact")

    if lock["actions"] != list(ACTION_IDS):
        raise LookaheadCollectionError("protocol lock actions are not exact")
    configuration = lock["configuration"]
    if (
        not isinstance(configuration, Mapping)
        or set(configuration)
        != {
            "max_new_tokens",
            "decision_tokens",
            "actions",
            "seed",
            "eos_ids",
            "decode_skip_special_tokens",
            "max_lookahead_steps",
        }
        or isinstance(configuration["max_new_tokens"], bool)
        or configuration["max_new_tokens"] != MAX_NEW_TOKENS
        or isinstance(configuration["decision_tokens"], bool)
        or configuration["decision_tokens"] != DECISION_TOKENS
        or isinstance(configuration["seed"], bool)
        or configuration["seed"] != SEED
        or configuration["decode_skip_special_tokens"] is not True
        or isinstance(configuration["max_lookahead_steps"], bool)
        or configuration["max_lookahead_steps"] != MAX_STEPS
        or not isinstance(configuration["eos_ids"], list)
        or not all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in configuration["eos_ids"]
        )
        or not _configuration_actions_match(configuration["actions"])
    ):
        raise LookaheadCollectionError(
            "protocol lock configuration is not exact"
        )
    if not isinstance(lock["features"], Mapping):
        raise LookaheadCollectionError("protocol lock features are malformed")
    if not isinstance(lock["estimator"], Mapping):
        raise LookaheadCollectionError("protocol lock estimator is malformed")
    if not isinstance(lock["evaluation"], Mapping):
        raise LookaheadCollectionError(
            "protocol lock evaluation is malformed"
        )
    tokenizer = lock["tokenizer"]
    if not isinstance(tokenizer, Mapping) or set(tokenizer) != {
        "name_or_path",
        "revision",
        "chat_template_sha256",
        "files_sha256",
    } or not isinstance(tokenizer["name_or_path"], str) or not tokenizer[
        "name_or_path"
    ] or not isinstance(tokenizer["revision"], str) or not tokenizer[
        "revision"
    ] or not isinstance(tokenizer["chat_template_sha256"], str) or not (
        _is_sha256(tokenizer["chat_template_sha256"])
    ):
        raise LookaheadCollectionError("protocol lock tokenizer is malformed")
    tokenizer_files = tokenizer["files_sha256"]
    if (
        not isinstance(tokenizer_files, Mapping)
        or not tokenizer_files
        or set(tokenizer_files) != TOKENIZER_ASSET_NAMES
        or any(
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(value, str)
            or not _is_sha256(value)
            for name, value in tokenizer_files.items()
        )
    ):
        raise LookaheadCollectionError(
            "protocol lock tokenizer assets are malformed"
        )
    generation_environment = lock["generation_environment"]
    if not isinstance(generation_environment, Mapping):
        raise LookaheadCollectionError(
            "protocol lock generation environment is malformed"
        )
    expected_environment_keys = {
        "attention_backend",
        "cuda_available",
        "cuda_version",
        "cwd",
        "device",
        "dtype",
        "packages",
        "platform",
        "python",
        "python_executable",
        "resources",
        "seed",
        "tokenizer_name_or_path",
        "torch_version",
    }
    if set(generation_environment) != expected_environment_keys:
        raise LookaheadCollectionError(
            "protocol lock generation environment is not exact"
        )
    if not isinstance(lock["model"], Mapping) or not lock["model"]:
        raise LookaheadCollectionError("protocol lock model is malformed")
    if lock["prediction_seal_schema"] != SEAL_SCHEMA_VERSION:
        raise LookaheadCollectionError("protocol lock seal schema is invalid")
    if lock["outcome_collection_schema"] != OUTCOME_COLLECTION_SCHEMA:
        raise LookaheadCollectionError(
            "protocol lock outcome schema is invalid"
        )
    for key in ("proposal_sha256", "roster_derivation_sha256"):
        value = lock[key]
        if not isinstance(value, str) or not _is_sha256(value):
            raise LookaheadCollectionError(
                f"protocol lock {key} is malformed"
            )
    _validate_source_manifest(lock["source_manifest"])
    _validate_scorer_provenance(lock["scorer_provenance"])


def _validate_source_manifest(value: object) -> None:
    if not isinstance(value, Mapping) or not value:
        raise LookaheadCollectionError(
            "protocol source manifest is malformed"
        )
    root = Path(__file__).resolve().parents[3]
    expected_paths = {
        str(path.relative_to(root))
        for path in sorted((root / "src/herald_v3/engineering").rglob("*.py"))
    }
    for name in (
        "build_lookahead_manifests.py",
        "freeze_lookahead.py",
        "collect_lookahead.py",
        "evaluate_lookahead.py",
        "collect_lookahead_outcomes.py",
        "verify_official_scores.py",
    ):
        path = root / "scripts" / name
        if path.is_file():
            expected_paths.add(str(path.relative_to(root)))
    if set(value) != expected_paths:
        raise LookaheadCollectionError(
            "protocol source manifest paths do not match runtime sources"
        )
    for name, expected in value.items():
        if (
            not isinstance(name, str)
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not isinstance(expected, str)
            or not _is_sha256(expected)
        ):
            raise LookaheadCollectionError(
                "protocol source manifest entry is malformed"
            )
        path = root / name
        if not path.is_file() or _sha256(path.read_bytes()) != expected:
            raise LookaheadCollectionError(
                f"protocol source hash mismatch: {name}"
            )


def _validate_scorer_provenance(value: object) -> None:
    if not isinstance(value, Mapping):
        raise LookaheadCollectionError(
            "protocol scorer provenance is malformed"
        )
    required = {"repository", "commit", "files"}
    if set(value) != required:
        raise LookaheadCollectionError(
            "protocol scorer provenance is not exact"
        )
    if (
        not isinstance(value["repository"], str)
        or not value["repository"]
        or not isinstance(value["commit"], str)
        or not value["commit"]
        or not isinstance(value["files"], Mapping)
        or not value["files"]
    ):
        raise LookaheadCollectionError(
            "protocol scorer provenance is malformed"
        )
    for name, entry in value["files"].items():
        if not isinstance(name, str) or not isinstance(entry, Mapping):
            raise LookaheadCollectionError(
                "protocol scorer provenance file is malformed"
            )
        if set(entry) != {"url", "sha256", "bytes"}:
            raise LookaheadCollectionError(
                "protocol scorer provenance file is not exact"
            )
        if (
            not isinstance(entry["url"], str)
            or not entry["url"]
            or not isinstance(entry["sha256"], str)
            or not _is_sha256(entry["sha256"])
            or isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] <= 0
        ):
            raise LookaheadCollectionError(
                "protocol scorer provenance file is malformed"
            )
    source_path = (
        Path(__file__).resolve().parents[3]
        / "results/engineering/official-scorer/source-manifest.json"
    )
    if source_path.is_file():
        actual = _read_object(source_path.read_bytes(), source_path)
        if dict(value) != actual:
            raise LookaheadCollectionError(
                "protocol scorer provenance differs from official source"
            )


def _validate_software(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "generation",
        "evaluation",
    }:
        raise LookaheadCollectionError("protocol software is not exact")
    for section_name in ("generation", "evaluation"):
        section = value[section_name]
        if not isinstance(section, Mapping) or not section:
            raise LookaheadCollectionError(
                f"protocol {section_name} software is malformed"
            )
        for package, expected in section.items():
            if not isinstance(package, str) or not package:
                raise LookaheadCollectionError(
                    "protocol software package name is malformed"
                )
            if not isinstance(expected, str) or not expected:
                raise LookaheadCollectionError(
                    f"protocol software version is missing: {package}"
                )
            if section_name == "evaluation":
                continue
            try:
                observed = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError as error:
                raise LookaheadCollectionError(
                    f"required software package is missing: {package}"
                ) from error
            if observed != expected:
                raise LookaheadCollectionError(
                    f"software version mismatch for {package}"
                )


def _validate_runtime_lock(
    lock: Mapping[str, object],
    observed_model: Mapping[str, object],
    eos_ids: set[int] | frozenset[int],
    *,
    tokenizer_name: str,
    tokenizer: Any,
    strict: bool = True,
) -> None:
    config = _lock_configuration(lock)
    locked_eos = config.get("eos_ids")
    if strict and locked_eos != sorted(eos_ids):
        raise LookaheadCollectionError(
            "runtime EOS IDs do not match protocol lock"
        )
    locked_model = lock.get("model")
    if strict:
        if not isinstance(locked_model, Mapping) or (
            dict(locked_model) != dict(observed_model)
        ):
            raise LookaheadCollectionError(
                "runtime model identity does not match protocol lock"
            )
    elif isinstance(locked_model, Mapping):
        for key, expected in locked_model.items():
            if key in observed_model and expected != observed_model[key]:
                raise LookaheadCollectionError(
                    f"model field {key} does not match protocol lock"
                )
    locked_tokenizer = lock.get("tokenizer")
    if strict:
        if (
            not isinstance(locked_tokenizer, Mapping)
            or locked_tokenizer.get("name_or_path") != tokenizer_name
        ):
            raise LookaheadCollectionError(
                "runtime tokenizer identity does not match protocol lock"
            )
    elif isinstance(locked_tokenizer, Mapping):
        expected_name = locked_tokenizer.get("name_or_path")
        if expected_name is not None and expected_name != tokenizer_name:
            raise LookaheadCollectionError(
                "tokenizer name does not match protocol lock"
            )
    if strict:
        _validate_software(lock["software"])
        _validate_generation_environment(
            lock["generation_environment"],
            observed_model,
            tokenizer,
            tokenizer_name,
            lock["software"],
        )
        _validate_tokenizer_assets(lock["tokenizer"], tokenizer)


def _validate_generation_environment(
    value: object,
    observed_model: Mapping[str, object],
    tokenizer: Any,
    tokenizer_name: str,
    software: object,
) -> None:
    if not isinstance(value, Mapping):
        raise LookaheadCollectionError(
            "protocol generation environment is malformed"
        )
    expected = {
        "dtype": observed_model.get("dtype"),
        "device": observed_model.get("device"),
        "attention_backend": observed_model.get("attention_implementation"),
        "tokenizer_name_or_path": tokenizer_name,
        "seed": SEED,
    }
    if not isinstance(software, Mapping):
        raise LookaheadCollectionError("protocol software is malformed")
    if value.get("packages") != software.get("generation"):
        raise LookaheadCollectionError(
            "generation environment packages do not match protocol lock"
        )
    for key, actual in expected.items():
        if value.get(key) != actual:
            raise LookaheadCollectionError(
                f"generation environment {key} does not match runtime"
            )
    torch = runner._torch()
    if value.get("torch_version") != torch.__version__:
        raise LookaheadCollectionError(
            "generation environment torch version does not match runtime"
        )
    cuda_available = bool(torch.cuda.is_available())
    if value.get("cuda_available") is not cuda_available:
        raise LookaheadCollectionError(
            "generation environment CUDA availability does not match runtime"
        )
    if value.get("cuda_version") != torch.version.cuda:
        raise LookaheadCollectionError(
            "generation environment CUDA version does not match runtime"
        )


def _validate_tokenizer_assets(value: object, tokenizer: Any) -> None:
    if not isinstance(value, Mapping):
        raise LookaheadCollectionError("protocol tokenizer is malformed")
    tokenizer_name = _tokenizer_name(tokenizer)
    if value.get("name_or_path") != tokenizer_name:
        raise LookaheadCollectionError(
            "runtime tokenizer name does not match protocol lock"
        )
    if value.get("revision") != _tokenizer_revision(tokenizer):
        raise LookaheadCollectionError(
            "runtime tokenizer revision does not match protocol lock"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    expected_chat_hash = value.get("chat_template_sha256")
    if not isinstance(chat_template, str) or not isinstance(
        expected_chat_hash, str
    ) or _sha256_text(chat_template) != expected_chat_hash:
        raise LookaheadCollectionError(
            "runtime tokenizer chat template does not match protocol lock"
        )
    files = value.get("files_sha256")
    if not isinstance(files, Mapping):
        raise LookaheadCollectionError(
            "protocol tokenizer assets are malformed"
        )
    tokenizer_root = Path(tokenizer_name)
    for filename, expected in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise LookaheadCollectionError(
                "protocol tokenizer asset name is unsafe"
            )
        path = tokenizer_root / filename
        if (
            not isinstance(expected, str)
            or not path.is_file()
            or _sha256(path.read_bytes()) != expected
        ):
            raise LookaheadCollectionError(
                f"runtime tokenizer asset hash mismatch: {filename}"
            )


def _tokenizer_revision(tokenizer: Any) -> str:
    name = _tokenizer_name(tokenizer)
    for part in reversed(Path(name).parts):
        if re.fullmatch(r"[0-9a-f]{40}", part):
            return part
    revision = getattr(tokenizer, "revision", None)
    if isinstance(revision, str) and revision:
        return revision
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if isinstance(init_kwargs, Mapping):
        revision = init_kwargs.get("revision")
        if isinstance(revision, str) and revision:
            return revision
    return ""


def _validate_manifest(
    manifest: PromptManifest, phase: str, expected_count: int
) -> None:
    if len(manifest.prompts) != expected_count:
        raise LookaheadCollectionError(
            f"{phase} manifest contains {len(manifest.prompts)} prompts, "
            f"expected {expected_count}"
        )
    ids = [prompt.prompt_id for prompt in manifest.prompts]
    if len(ids) != len(set(ids)):
        raise LookaheadCollectionError(
            "manifest contains duplicate prompt IDs"
        )


def _validate_manifest_file(
    manifest_file: Path, manifest: PromptManifest
) -> None:
    document = _read_object(manifest_file.read_bytes(), manifest_file)
    if document.get("manifest_sha256") != manifest.fingerprint:
        raise LookaheadCollectionError("manifest file fingerprint is invalid")
    prompts = document.get("prompts")
    ids = [prompt.prompt_id for prompt in manifest.prompts]
    if (
        not isinstance(prompts, list)
        or not all(isinstance(item, Mapping) for item in prompts)
        or [item["prompt_id"] for item in prompts] != ids
        ):
        raise LookaheadCollectionError(
            "manifest file IDs do not match object"
        )


def _validate_training_reuse_lock(
    lock: Mapping[str, object],
    protocol_lock_file: Path,
    supplied_path: Path | None,
    phase: str,
    *,
    strict: bool,
) -> None:
    """Check the immutable reuse bytes before parsing them as JSON."""
    if not strict:
        return
    reference = lock["training_reuse"]
    if not isinstance(reference, Mapping):
        raise LookaheadCollectionError(
            "protocol lock training reuse reference is malformed"
        )
    locked_path = Path(cast(str, reference["path"]))
    expected_path = (protocol_lock_file.parent / locked_path).resolve()
    if phase == "train":
        if supplied_path is None:
            raise LookaheadCollectionError(
                "train collection requires --train-reuse"
            )
        actual_path = supplied_path.resolve()
        if actual_path != expected_path:
            raise LookaheadCollectionError(
                "training reuse path does not match protocol lock"
            )
    else:
        actual_path = expected_path
    raw = _read_bytes(actual_path)
    if _sha256(raw) != reference["sha256"]:
        raise LookaheadCollectionError(
            "training reuse hash does not match protocol lock"
        )


def _load_train_reuse(
    path: Path | None,
    manifest: PromptManifest,
    phase: str,
    expected_count: int,
) -> Mapping[str, object] | None:
    if phase != "train":
        if path is not None:
            raise LookaheadCollectionError(
                "test collection must not accept training reuse"
            )
        return None
    if path is None:
        raise LookaheadCollectionError(
            "train collection requires --train-reuse"
        )
    document = _read_object(_read_bytes(path), path)
    if document.get("schema_version") != TRAINING_REUSE_SCHEMA_VERSION:
        raise LookaheadCollectionError("training reuse schema is invalid")
    if (
        document.get("labels_embedded") is not False
        or document.get("outcomes_read", False)
        or document.get("scores_read", False)
    ):
        raise LookaheadCollectionError(
            "training reuse is outcome-contaminated"
        )
    ids = document.get(
        "training_prompt_ids",
        document.get("ordered_prompt_ids", document.get("prompt_ids")),
    )
    expected = [prompt.prompt_id for prompt in manifest.prompts]
    if ids != expected or len(expected) != expected_count:
        raise LookaheadCollectionError(
            "training reuse IDs do not match manifest order"
        )
    if document.get("expected_prompt_count") != expected_count:
        raise LookaheadCollectionError(
            "training reuse prompt count is invalid"
        )
    if (
        document.get("expected_action_rows")
        != expected_count * ACTION_ROW_COUNT
    ):
        raise LookaheadCollectionError(
            "training reuse action row count is invalid"
        )
    rows = document.get("rows", document.get("by_prompt"))
    if isinstance(rows, list):
        row_ids = [
            item.get("prompt_id")
            for item in rows
            if isinstance(item, Mapping)
        ]
        if row_ids != expected:
            raise LookaheadCollectionError(
                "training reuse rows are not an ordered complete list"
            )
    elif isinstance(rows, Mapping):
        if list(rows) != expected:
            raise LookaheadCollectionError(
                "training reuse rows are not an ordered complete map"
            )
    else:
        raise LookaheadCollectionError("training reuse rows are malformed")
    return document


def _validate_reuse_row(
    reuse: Mapping[str, object],
    prompt: EngineeringPrompt,
    manifest: PromptManifest,
    boundary: engine.BoundaryState,
    *,
    model_signature: Mapping[str, object],
    tokenizer_name: str,
    eos_ids: set[int] | frozenset[int],
) -> None:
    rows = reuse.get("rows", reuse.get("by_prompt"))
    entry = _reuse_entry(rows, prompt.prompt_id)
    if not isinstance(entry, Mapping):
        raise LookaheadCollectionError(
            f"missing training reuse row {prompt.prompt_id}"
        )
    if entry.get("prompt") != prompt.to_dict():
        raise LookaheadCollectionError(
            "training reuse prompt provenance mismatch for "
            f"{prompt.prompt_id}"
        )
    reused_boundary = entry.get("boundary")
    if not isinstance(reused_boundary, Mapping):
        raise LookaheadCollectionError(
            f"training reuse boundary is missing for {prompt.prompt_id}"
        )
    if _stable_boundary(reused_boundary) != _stable_boundary(
        boundary.to_dict()
    ):
        raise LookaheadCollectionError(
            f"training reuse boundary mismatch for {prompt.prompt_id}"
        )
    expected_cache_fingerprint = entry.get("boundary_cache_fingerprint")
    if (
        not isinstance(expected_cache_fingerprint, str)
        or expected_cache_fingerprint
        != engine.cache_fingerprint(boundary.cache)
    ):
        raise LookaheadCollectionError(
            "training reuse boundary cache fingerprint mismatch for "
            f"{prompt.prompt_id}"
        )
    tokenization = entry.get("tokenization")
    if not isinstance(tokenization, Mapping):
        raise LookaheadCollectionError(
            f"training reuse tokenization is missing for {prompt.prompt_id}"
        )
    if tokenization.get("input_ids") != boundary.prompt_ids[0].tolist():
        raise LookaheadCollectionError(
            f"training reuse tokenization mismatch for {prompt.prompt_id}"
        )
    _validate_reuse_artifacts(reuse, entry, prompt)
    configuration = entry.get("configuration")
    if not isinstance(configuration, Mapping):
        raise LookaheadCollectionError(
            f"training reuse configuration is missing for {prompt.prompt_id}"
        )
    if configuration.get("decision_tokens") != DECISION_TOKENS:
        raise LookaheadCollectionError(
            "training reuse boundary configuration mismatch for "
            f"{prompt.prompt_id}"
        )
    if not _actions_match(configuration.get("actions")):
        raise LookaheadCollectionError(
            "training reuse action configuration mismatch for "
            f"{prompt.prompt_id}"
        )
    if configuration.get("seed") != SEED:
        raise LookaheadCollectionError(
            f"training reuse seed mismatch for {prompt.prompt_id}"
        )
    if configuration.get("eos_ids") != sorted(eos_ids):
        raise LookaheadCollectionError(
            f"training reuse EOS IDs mismatch for {prompt.prompt_id}"
        )
    if configuration.get("max_new_tokens") != MAX_NEW_TOKENS:
        raise LookaheadCollectionError(
            "training reuse generation budget mismatch for "
            f"{prompt.prompt_id}"
        )
    if entry.get("model") != dict(model_signature):
        raise LookaheadCollectionError(
            f"training reuse model provenance mismatch for {prompt.prompt_id}"
        )
    environment = entry.get("environment")
    if not isinstance(environment, Mapping):
        raise LookaheadCollectionError(
            f"training reuse environment is missing for {prompt.prompt_id}"
        )
    if environment.get("tokenizer_name_or_path") != tokenizer_name:
        raise LookaheadCollectionError(
            "training reuse tokenizer provenance mismatch for "
            f"{prompt.prompt_id}"
        )
    source_manifest = entry.get("source_manifest")
    if not isinstance(source_manifest, Mapping):
        raise LookaheadCollectionError(
            "training reuse source manifest is missing for "
            f"{prompt.prompt_id}"
        )
    expected_source = source_manifest.get("sha256")
    if not isinstance(expected_source, Mapping):
        raise LookaheadCollectionError(
            f"training reuse source hashes are missing for {prompt.prompt_id}"
        )
    if any(
        not isinstance(value, str) or not _source_hash_matches(name, value)
        for name, value in expected_source.items()
    ):
        raise LookaheadCollectionError(
            f"training reuse source hash mismatch for {prompt.prompt_id}"
        )
    if entry.get("prompt_id") != prompt.prompt_id:
        raise LookaheadCollectionError(
            f"training reuse prompt ID mismatch for {prompt.prompt_id}"
        )


def _validate_reuse_artifacts(
    document: Mapping[str, object],
    entry: Mapping[str, object],
    prompt: EngineeringPrompt,
) -> None:
    directory = entry.get("artifact_directory")
    if not isinstance(directory, str):
        raise LookaheadCollectionError(
            "training reuse artifact directory missing for "
            f"{prompt.prompt_id}"
        )
    path = Path(directory)
    root_relative = document.get("collection_root_relative_to_project")
    candidates = [path] if path.is_absolute() else []
    if isinstance(root_relative, str):
        project_root = Path(__file__).resolve().parents[3]
        candidates.append(project_root / root_relative / path)
    candidates.extend(
        (Path.cwd() / path, Path(__file__).resolve().parents[3] / path)
    )
    artifact_dir = next(
        (candidate for candidate in candidates if candidate.is_dir()), None
    )
    if artifact_dir is None:
        raise LookaheadCollectionError(
            "training reuse artifact directory is missing for "
            f"{prompt.prompt_id}"
        )
    artifact_index = artifact_dir / "artifacts.json"
    expected_index = entry.get("artifacts_sha256")
    if not isinstance(expected_index, str) or not artifact_index.is_file():
        raise LookaheadCollectionError(
            "training reuse artifacts_sha256 evidence is missing for "
            f"{prompt.prompt_id}"
        )
    if _sha256(artifact_index.read_bytes()) != expected_index:
        raise LookaheadCollectionError(
            f"training reuse artifacts_sha256 mismatch for {prompt.prompt_id}"
        )
    files = entry.get("files_sha256")
    if not isinstance(files, Mapping):
        raise LookaheadCollectionError(
            f"training reuse file hashes are missing for {prompt.prompt_id}"
        )
    for filename, expected in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise LookaheadCollectionError(
                "training reuse artifact filename is unsafe for "
                f"{prompt.prompt_id}"
            )
        actual_path = artifact_dir / filename
        if not isinstance(expected, str) or not actual_path.is_file():
            raise LookaheadCollectionError(
                "training reuse artifact hash is missing for "
                f"{prompt.prompt_id}"
            )
        if _sha256(actual_path.read_bytes()) != expected:
            raise LookaheadCollectionError(
                "training reuse artifact hash mismatch for "
                f"{prompt.prompt_id}"
            )


def _stable_boundary(value: Mapping[str, object]) -> dict[str, object]:
    return {
        key: item
        for key, item in value.items()
        if key
        not in {
            "validation_seconds",
            "model_state_fingerprint",
            "state_fingerprint",
        }
    }


def _source_hash_matches(name: object, expected: str) -> bool:
    if not isinstance(name, str):
        return False
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        return False
    source = Path(__file__).resolve().parent / relative
    return source.is_file() and _sha256(source.read_bytes()) == expected


def _reuse_entry(rows: object, prompt_id: str) -> Mapping[str, object] | None:
    if isinstance(rows, Mapping):
        entry = rows.get(prompt_id)
        return entry if isinstance(entry, Mapping) else None
    if isinstance(rows, list):
        for item in rows:
            if (
                isinstance(item, Mapping)
                and item.get("prompt_id") == prompt_id
            ):
                return item
    return None


def _collection_lock(
    *,
    phase: str,
    manifest: PromptManifest,
    manifest_file: Path,
    manifest_bytes: bytes,
    protocol_lock_file: Path,
    protocol_lock_bytes: bytes,
    train_reuse_file: Path | None,
    model: Mapping[str, object],
    tokenizer: Any,
    eos_ids: set[int] | frozenset[int],
    output_path: Path,
    expected_count: int,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "mode": "probe_only",
        "manifest": {
            "path": str(manifest_file),
            "file_sha256": _sha256(manifest_bytes),
            "manifest_sha256": manifest.fingerprint,
            "prompt_ids": [prompt.prompt_id for prompt in manifest.prompts],
            "expected_prompt_count": expected_count,
        },
        "protocol_lock": {
            "path": str(protocol_lock_file),
            "sha256": _sha256(protocol_lock_bytes),
        },
        "train_reuse": (
            {
                "path": str(train_reuse_file),
                "sha256": _sha256(train_reuse_file.read_bytes()),
            }
            if train_reuse_file is not None
            else None
        ),
        "model": dict(model),
        "tokenizer": {"name_or_path": _tokenizer_name(tokenizer)},
        "configuration": _configuration(eos_ids),
        "output": str(output_path),
        "collector": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve().read_bytes()),
        },
        "outcomes_collected": False,
    }


def _prepare_output(
    output: Path, expected_lock: Mapping[str, object]
) -> tuple[Path, Path, dict[str, object]]:
    if output.exists() and not output.is_dir():
        raise LookaheadCollectionError(
            f"collection output is not a directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / LOCK_NAME
    checkpoint_path = output / CHECKPOINT_NAME
    entries = list(output.iterdir())
    if lock_path.exists():
        actual = _read_object(lock_path.read_bytes(), lock_path)
        if actual != expected_lock:
            raise LookaheadCollectionError(
                "collection lock does not match current inputs"
            )
        if not checkpoint_path.is_file():
            raise LookaheadCollectionError("collection checkpoint is missing")
        checkpoint = _read_object(
            checkpoint_path.read_bytes(), checkpoint_path
        )
        return lock_path, checkpoint_path, checkpoint
    if entries:
        raise LookaheadCollectionError(
            "nonempty collection output has no lock and cannot be overwritten"
        )
    _write_json(lock_path, expected_lock)
    lock_sha256 = _sha256(lock_path.read_bytes())
    manifest_lock = expected_lock["manifest"]
    if not isinstance(manifest_lock, Mapping):
        raise LookaheadCollectionError(
            "collection lock manifest is malformed"
        )
    initial = {
        "schema_version": SCHEMA_VERSION,
        "phase": expected_lock["phase"],
        "status": "starting",
        "lock_sha256": lock_sha256,
        "manifest_sha256": manifest_lock["manifest_sha256"],
        "prompt_ids": manifest_lock["prompt_ids"],
        "processed": 0,
        "eligible_prompts": 0,
        "early_eos_prompts": 0,
        "action_row_count": 0,
        "ledger": [],
    }
    _write_json(checkpoint_path, initial)
    return lock_path, checkpoint_path, initial


def _recover_ledger(
    output: Path,
    checkpoint: Mapping[str, object],
    manifest: PromptManifest,
    lock_sha256: str,
    expected_count: int,
) -> list[dict[str, object]]:
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise LookaheadCollectionError(
            "collection checkpoint schema is invalid"
        )
    if checkpoint.get("lock_sha256") != lock_sha256:
        raise LookaheadCollectionError(
            "collection checkpoint lock hash mismatches"
        )
    ledger = checkpoint.get("ledger")
    if not isinstance(ledger, list):
        raise LookaheadCollectionError(
            "collection checkpoint ledger is malformed"
        )
    expected_ids = [prompt.prompt_id for prompt in manifest.prompts]
    if len(ledger) > expected_count:
        raise LookaheadCollectionError(
            "collection checkpoint exceeds manifest"
        )
    recovered: list[dict[str, object]] = []
    for index, item in enumerate(ledger):
        if not isinstance(item, Mapping):
            raise LookaheadCollectionError(
                "collection ledger entry is malformed"
            )
        if item.get("prompt_id") != expected_ids[index]:
            raise LookaheadCollectionError(
                "collection ledger is not an ordered manifest prefix"
            )
        expected_source_hash = manifest.prompts[index].prompt_text_sha256
        if item.get("source_hash") != expected_source_hash:
            raise LookaheadCollectionError(
                "collection ledger source hash mismatch for "
                f"{expected_ids[index]}"
            )
        status = item.get("status")
        if status == "ineligible_early_eos":
            if item.get("action_row_count") != 0:
                raise LookaheadCollectionError(
                    "early-EOS ledger unexpectedly has action rows"
                )
            if not isinstance(item.get("reason"), str) or not isinstance(
                item.get("position"), int
            ):
                raise LookaheadCollectionError(
                    "early-EOS ledger entry is incomplete"
                )
            recovered.append(dict(item))
            continue
        if status != "eligible":
            raise LookaheadCollectionError(
                "collection ledger status is invalid"
            )
        record_rel = item.get("record")
        expected_hash = item.get("record_sha256")
        if not isinstance(record_rel, str) or not isinstance(
            expected_hash, str
        ):
            raise LookaheadCollectionError(
                "collection ledger record hash is missing"
            )
        record_path = _safe_record_path(output, record_rel)
        if (
            not record_path.is_file()
            or _sha256(record_path.read_bytes()) != expected_hash
        ):
            raise LookaheadCollectionError(
                f"collection record hash mismatch for {item['prompt_id']}"
            )
        record = _read_object(record_path.read_bytes(), record_path)
        if record.get("prompt_id") != item["prompt_id"]:
            raise LookaheadCollectionError(
                "collection record prompt ID mismatch"
            )
        if record.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise LookaheadCollectionError(
                "collection record schema is invalid"
            )
        if record.get("phase") != checkpoint.get("phase"):
            raise LookaheadCollectionError("collection record phase mismatch")
        if record.get("outcomes_collected") is not False:
            raise LookaheadCollectionError(
                "collection record contains outcome state"
            )
        if record.get("status") not in {
            "eligible",
            "ineligible_early_eos",
        }:
            raise LookaheadCollectionError(
                "collection record status is invalid"
            )
        if (
            record.get("status") == "eligible"
            and item.get("action_row_count") != 2
        ):
            raise LookaheadCollectionError(
                "eligible collection record is not a complete pair"
            )
        if (
            record.get("status") == "ineligible_early_eos"
            and item.get("action_row_count") != 0
        ):
            raise LookaheadCollectionError(
                "early-EOS record unexpectedly has action rows"
            )
        recovered.append(dict(item))
    return recovered


def _safe_record_path(output: Path, relative: str) -> Path:
    candidate = (output / relative).resolve()
    try:
        candidate.relative_to(output.resolve())
    except ValueError as error:
        raise LookaheadCollectionError(
            "collection ledger record path escapes output"
        ) from error
    return candidate


def _summary(
    status: str,
    phase: str,
    manifest: PromptManifest,
    ledger: Sequence[Mapping[str, object]],
    expected_count: int,
    lock_sha256: str,
    *,
    current_prompt_id: str | None,
) -> dict[str, object]:
    eligible = sum(item.get("status") == "eligible" for item in ledger)
    early = sum(
        item.get("status") == "ineligible_early_eos" for item in ledger
    )
    action_rows = sum(
        cast(int, item.get("action_row_count", 0)) for item in ledger
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "status": status,
        "passed": status == "completed" and len(ledger) == expected_count,
        "lock_sha256": lock_sha256,
        "manifest_sha256": manifest.fingerprint,
        "expected_prompt_count": expected_count,
        "prompt_ids": [prompt.prompt_id for prompt in manifest.prompts],
        "processed": len(ledger),
        "eligible_prompts": eligible,
        "early_eos_prompts": early,
        "action_row_count": action_rows,
        "current_prompt_id": current_prompt_id,
        "ledger": [dict(item) for item in ledger],
        "outcomes_collected": False,
    }


def _manifest_lock_entry(
    lock: Mapping[str, object], phase: str
) -> Mapping[str, object]:
    manifests = lock.get("manifests")
    if isinstance(manifests, Mapping):
        value = manifests.get(phase)
        if isinstance(value, Mapping):
            return cast(Mapping[str, object], value)
    raise LookaheadCollectionError(
        "protocol lock has no phase manifest entry"
    )


def _lock_configuration(lock: Mapping[str, object]) -> Mapping[str, object]:
    value = lock.get("configuration")
    if isinstance(value, Mapping):
        return value
    return {}


def _actions_match(value: object) -> bool:
    if not isinstance(value, list):
        return False
    observed = []
    for item in value:
        if not isinstance(item, Mapping):
            return False
        if item.get("name") != "knorm":
            return False
        observed.append(item.get("removal_fraction"))
    return observed == list(ACTION_RATIOS)


def _configuration_actions_match(value: object) -> bool:
    if not isinstance(value, list) or len(value) != len(ACTION_RATIOS):
        return False
    for item, ratio in zip(value, ACTION_RATIOS, strict=True):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"name", "removal_fraction"}
            or item.get("name") != "knorm"
            or item.get("removal_fraction") != ratio
        ):
            return False
    return True


def _expected_count(phase: str) -> int:
    return TRAIN_PROMPT_COUNT if phase == "train" else TEST_PROMPT_COUNT


def _tokenizer_name(tokenizer: Any) -> str:
    return str(getattr(tokenizer, "name_or_path", ""))


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise LookaheadCollectionError(
            f"required source is missing: {path}"
        ) from error


def _read_object(raw: bytes, path: Path) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LookaheadCollectionError(
            f"invalid JSON source: {path}"
        ) from error
    if not isinstance(value, dict):
        raise LookaheadCollectionError(
            f"JSON source is not an object: {path}"
        )
    return value


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    _write_json(path, payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))


def _is_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
