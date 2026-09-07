#!/usr/bin/env python3
"""Collect a resumable frozen-manifest HERALD pilot."""

# The imports intentionally follow the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3.engineering import runner
from herald_v3.engineering.engine import model_signature, to_builtin
from herald_v3.engineering.prompts import (
    EngineeringPrompt,
    PromptManifest,
    load_prompt_manifest,
)
from herald_v3.engineering.scoring import check_ifeval_resources

MAX_ROSTER = 160
DEFAULT_TARGET_ELIGIBLE = 120
DEFAULT_MAX_NEW_TOKENS = 1024
LOCK_NAME = "experiment-lock.json"
CHECKPOINT_NAME = "checkpoint.json"
REQUIRED_PROMPT_ARTIFACTS = {
    "prompt-manifest.json",
    "run.json",
    "run.log",
}


class PilotStateError(RuntimeError):
    """Existing pilot state is incomplete or does not match the lock."""


class PilotRunError(RuntimeError):
    """A new prompt run failed and collection must stop."""


def run_pilot(
    model: Any,
    tokenizer: Any,
    manifest: PromptManifest,
    manifest_path: str | Path,
    output: str | Path,
    *,
    model_path: str,
    target_eligible: int = DEFAULT_TARGET_ELIGIBLE,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    seed: int = 0,
) -> dict[str, object]:
    """Collect eligible prompts in manifest order with verified resume."""
    _validate_configuration(manifest, target_eligible, max_new_tokens)
    frozen_manifest_path = Path(manifest_path).resolve()
    _validate_frozen_manifest(frozen_manifest_path, manifest)
    output_path = Path(output).resolve()
    _emit(
        "pilot_started",
        output=str(output_path),
        roster_count=len(manifest.prompts),
        target_eligible=target_eligible,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    lock = _lock_payload(
        model,
        tokenizer,
        manifest,
        frozen_manifest_path,
        model_path=model_path,
        target_eligible=target_eligible,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    lock_path, checkpoint_path, prior_checkpoint = _prepare_output(
        output_path, lock
    )
    lock_sha256 = runner._file_sha256(lock_path)
    ledger = _recover_ledger(
        output_path,
        manifest,
        lock,
        target_eligible=target_eligible,
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    _validate_checkpoint_prefix(prior_checkpoint, ledger, checkpoint_path)
    accepted = _count_status(ledger, "accepted")

    for prompt in manifest.prompts[len(ledger) :]:
        if accepted >= target_eligible:
            break
        prompt_output = output_path / prompt.prompt_id
        before = _summary(
            "running",
            manifest,
            ledger,
            target_eligible,
            lock_sha256,
            current_prompt_id=prompt.prompt_id,
            phase="before_prompt",
        )
        _durable_write_json(checkpoint_path, before)
        _emit(
            "pilot_progress",
            phase="before_prompt",
            prompt_id=prompt.prompt_id,
            processed=len(ledger),
            accepted_eligible=accepted,
            target_eligible=target_eligible,
        )
        one_prompt = _one_prompt_manifest(manifest, prompt)
        try:
            try:
                prompt_output.mkdir()
            except FileExistsError as error:
                raise PilotStateError(
                    f"prompt output appeared before launch: {prompt_output}"
                ) from error
            runner.run_engineering(
                model,
                tokenizer,
                one_prompt,
                prompt_output,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            entry = _verify_prompt_output(
                prompt_output,
                prompt,
                one_prompt,
                lock,
                max_new_tokens=max_new_tokens,
                seed=seed,
            )
            _sync_prompt_artifacts(prompt_output)
        except Exception as error:
            failure = _summary(
                "failed",
                manifest,
                ledger,
                target_eligible,
                lock_sha256,
                current_prompt_id=prompt.prompt_id,
                phase="after_prompt",
            )
            failure["error"] = {
                "type": type(error).__name__,
                "message": str(error),
                "prompt_output": str(prompt_output),
            }
            _durable_write_json(checkpoint_path, failure)
            if isinstance(error, PilotStateError):
                raise
            raise PilotRunError(
                f"prompt run failed at {prompt_output}: {error}"
            ) from error
        ledger.append(entry)
        accepted = _count_status(ledger, "accepted")
        after = _summary(
            "running",
            manifest,
            ledger,
            target_eligible,
            lock_sha256,
            current_prompt_id=prompt.prompt_id,
            phase="after_prompt",
        )
        _durable_write_json(checkpoint_path, after)
        _emit(
            "pilot_progress",
            phase="after_prompt",
            prompt_id=prompt.prompt_id,
            prompt_status=entry["status"],
            processed=len(ledger),
            accepted_eligible=accepted,
            ineligible_early_eos=_count_status(
                ledger, "ineligible_early_eos"
            ),
            target_eligible=target_eligible,
        )

    status = (
        "completed"
        if accepted >= target_eligible
        else "insufficient_coverage"
    )
    final = _summary(
        status,
        manifest,
        ledger,
        target_eligible,
        lock_sha256,
        current_prompt_id=None,
        phase="finished",
    )
    _durable_write_json(checkpoint_path, final)
    _emit(
        "pilot_finished",
        status=status,
        processed=len(ledger),
        accepted_eligible=accepted,
        ineligible_early_eos=_count_status(ledger, "ineligible_early_eos"),
        target_eligible=target_eligible,
        checkpoint=str(checkpoint_path),
    )
    return final


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-eligible", type=int, default=DEFAULT_TARGET_ELIGIBLE
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if not 1 <= args.target_eligible <= MAX_ROSTER:
        parser.error(f"--target-eligible must be in [1, {MAX_ROSTER}]")
    if args.max_new_tokens <= 32:
        parser.error("--max-new-tokens must exceed 32")
    if not runner._torch().cuda.is_available():
        parser.error("CUDA is required before loading the model")
    _emit(
        "pilot_starting",
        phase="before_model_load",
        model=args.model,
        prompts=str(Path(args.prompts).resolve()),
        output=str(Path(args.output).resolve()),
        target_eligible=args.target_eligible,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    manifest = load_prompt_manifest(args.prompts, limit=MAX_ROSTER + 1)
    if len(manifest.prompts) > MAX_ROSTER:
        parser.error(f"prompt roster exceeds {MAX_ROSTER}")
    model, tokenizer = runner.load_offline_model(args.model)
    try:
        summary = run_pilot(
            model,
            tokenizer,
            manifest,
            args.prompts,
            args.output,
            model_path=args.model,
            target_eligible=args.target_eligible,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    except (PilotStateError, PilotRunError) as error:
        _emit(
            "pilot_failed",
            error_type=type(error).__name__,
            message=str(error),
        )
        return 1
    return 0 if summary["status"] == "completed" else 1


def _validate_configuration(
    manifest: PromptManifest,
    target_eligible: int,
    max_new_tokens: int,
) -> None:
    if not 1 <= target_eligible <= MAX_ROSTER:
        raise ValueError(f"target_eligible must be in [1, {MAX_ROSTER}]")
    if max_new_tokens <= 32:
        raise ValueError("max_new_tokens must exceed 32")
    if len(manifest.prompts) > MAX_ROSTER:
        raise ValueError(f"prompt roster exceeds {MAX_ROSTER}")
    prompt_ids = [prompt.prompt_id for prompt in manifest.prompts]
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt roster contains duplicate IDs")


def _validate_frozen_manifest(path: Path, manifest: PromptManifest) -> None:
    document = _read_mapping(path)
    expected = manifest.to_dict()
    if document != expected:
        raise PilotStateError(
            f"frozen prompt manifest does not match loaded roster: {path}"
        )


def _lock_payload(
    model: Any,
    tokenizer: Any,
    manifest: PromptManifest,
    manifest_path: Path,
    *,
    model_path: str,
    target_eligible: int,
    max_new_tokens: int,
    seed: int,
) -> dict[str, object]:
    script_path = Path(__file__).resolve()
    resources = check_ifeval_resources()
    return {
        "schema_version": "herald_v3.pilot_lock.v1",
        "model_argument": model_path,
        "model": to_builtin(model_signature(model)),
        "prompt_manifest": {
            "path": str(manifest_path),
            "file_sha256": runner._file_sha256(manifest_path),
            "manifest_sha256": manifest.fingerprint,
            "prompt_ids": [prompt.prompt_id for prompt in manifest.prompts],
        },
        "configuration": {
            "target_eligible": target_eligible,
            "max_new_tokens": max_new_tokens,
            "decision_tokens": 32,
            "actions": [
                {"name": "knorm", "removal_fraction": ratio}
                for ratio in runner.DEFAULT_RATIOS
            ],
            "seed": seed,
            "eos_ids": sorted(runner._eos_ids(model, tokenizer)),
        },
        "source_manifest": runner._source_manifest(),
        "environment": runner._environment_manifest(
            model, tokenizer, resources, seed
        ),
        "wrapper": {
            "path": str(script_path),
            "sha256": runner._file_sha256(script_path),
        },
    }


def _prepare_output(
    output: Path, expected_lock: dict[str, object]
) -> tuple[Path, Path, dict[str, object]]:
    if output.exists() and not output.is_dir():
        raise PilotStateError(f"pilot output is not a directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / LOCK_NAME
    checkpoint_path = output / CHECKPOINT_NAME
    entries = list(output.iterdir())
    if lock_path.exists():
        actual_lock = _read_mapping(lock_path)
        if actual_lock != expected_lock:
            raise PilotStateError(
                f"experiment lock does not match current configuration: "
                f"{lock_path}"
            )
        if not checkpoint_path.is_file():
            raise PilotStateError(
                f"resume checkpoint is missing: {checkpoint_path}"
            )
        checkpoint = _read_mapping(checkpoint_path)
        _validate_checkpoint(
            checkpoint,
            expected_lock,
            runner._file_sha256(lock_path),
            checkpoint_path,
        )
        return lock_path, checkpoint_path, checkpoint
    if entries:
        raise PilotStateError(
            f"nonempty pilot output has no experiment lock: {output}"
        )
    _durable_write_json(lock_path, expected_lock)
    lock_sha256 = runner._file_sha256(lock_path)
    initial = {
        "schema_version": "herald_v3.pilot_checkpoint.v1",
        "status": "starting",
        "phase": "before_roster",
        "experiment_lock_sha256": lock_sha256,
        "target_eligible": _lock_configuration(expected_lock)[
            "target_eligible"
        ],
        "accepted_eligible": 0,
        "ineligible_early_eos": 0,
        "processed": 0,
        "roster_count": len(_lock_prompt_ids(expected_lock)),
        "current_prompt_id": None,
        "ledger": [],
        "early_eos_ledger": [],
        "passed": False,
    }
    _durable_write_json(checkpoint_path, initial)
    return lock_path, checkpoint_path, initial


def _recover_ledger(
    output: Path,
    manifest: PromptManifest,
    lock: dict[str, object],
    *,
    target_eligible: int,
    max_new_tokens: int,
    seed: int,
) -> list[dict[str, object]]:
    allowed_files = {LOCK_NAME, CHECKPOINT_NAME}
    prompt_ids = {prompt.prompt_id for prompt in manifest.prompts}
    for child in output.iterdir():
        if child.name in allowed_files:
            continue
        if not child.is_dir() or child.name not in prompt_ids:
            raise PilotStateError(f"unexpected pilot output entry: {child}")
    ledger: list[dict[str, object]] = []
    missing_seen = False
    accepted = 0
    for prompt in manifest.prompts:
        prompt_output = output / prompt.prompt_id
        if not prompt_output.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise PilotStateError(
                f"out-of-order prompt output cannot be resumed: "
                f"{prompt_output}"
            )
        if accepted >= target_eligible:
            raise PilotStateError(
                f"prompt output exists beyond the locked target: "
                f"{prompt_output}"
            )
        one_prompt = _one_prompt_manifest(manifest, prompt)
        entry = _verify_prompt_output(
            prompt_output,
            prompt,
            one_prompt,
            lock,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        ledger.append(entry)
        accepted = _count_status(ledger, "accepted")
        _emit(
            "pilot_progress",
            phase="resume_verified",
            prompt_id=prompt.prompt_id,
            prompt_status=entry["status"],
            processed=len(ledger),
            accepted_eligible=accepted,
            target_eligible=target_eligible,
        )
    return ledger


def _verify_prompt_output(
    output: Path,
    prompt: EngineeringPrompt,
    one_prompt: PromptManifest,
    lock: dict[str, object],
    *,
    max_new_tokens: int,
    seed: int,
) -> dict[str, object]:
    if not output.is_dir():
        raise PilotStateError(f"prompt output is not a directory: {output}")
    artifacts_path = output / "artifacts.json"
    artifacts = _read_mapping(artifacts_path)
    hashes = _require_mapping(artifacts.get("sha256"), artifacts_path)
    if artifacts.get("schema_version") != 1 or set(hashes) != (
        REQUIRED_PROMPT_ARTIFACTS
    ):
        raise PilotStateError(f"artifact index is malformed: {output}")
    for name in REQUIRED_PROMPT_ARTIFACTS:
        artifact_path = output / name
        expected_hash = hashes.get(name)
        if (
            not isinstance(expected_hash, str)
            or not artifact_path.is_file()
            or runner._file_sha256(artifact_path) != expected_hash
        ):
            raise PilotStateError(
                f"artifact hash verification failed: {output}"
            )
    expected_manifest = one_prompt.to_dict()
    prompt_manifest_path = output / "prompt-manifest.json"
    if _read_mapping(prompt_manifest_path) != expected_manifest:
        raise PilotStateError(f"prompt manifest mismatch: {output}")
    run_path = output / "run.json"
    run = _read_mapping(run_path)
    if run.get("prompt_manifest") != expected_manifest:
        raise PilotStateError(f"embedded prompt manifest mismatch: {output}")
    if run.get("source_manifest") != lock.get("source_manifest"):
        raise PilotStateError(f"engineering source mismatch: {output}")
    if run.get("environment") != lock.get("environment"):
        raise PilotStateError(f"runtime environment mismatch: {output}")
    if run.get("model") != lock.get("model"):
        raise PilotStateError(f"model signature mismatch: {output}")
    configuration = _require_mapping(run.get("configuration"), run_path)
    expected_configuration = _lock_configuration(lock)
    for key in (
        "max_new_tokens",
        "decision_tokens",
        "actions",
        "seed",
        "eos_ids",
    ):
        if configuration.get(key) != expected_configuration.get(key):
            raise PilotStateError(
                f"prompt configuration mismatch for {key}: {output}"
            )
    if max_new_tokens != configuration.get("max_new_tokens"):
        raise PilotStateError(f"prompt budget mismatch: {output}")
    if seed != configuration.get("seed"):
        raise PilotStateError(f"prompt seed mismatch: {output}")
    results = run.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise PilotStateError(f"prompt result count is not one: {output}")
    row = _require_mapping(results[0], run_path)
    if row.get("prompt") != prompt.to_dict() or row.get("seed") != seed:
        raise PilotStateError(f"prompt row provenance mismatch: {output}")
    status = row.get("status")
    acceptance = _require_mapping(row.get("acceptance"), run_path)
    if status == "accepted":
        _verify_accepted_controls(run, row, acceptance, output)
        return {
            "prompt_id": prompt.prompt_id,
            "status": "accepted",
            "artifact_directory": str(output),
            "artifacts_sha256": runner._file_sha256(artifacts_path),
        }
    if status == "ineligible":
        eligibility = _require_mapping(
            acceptance.get("eligibility"), run_path
        )
        reason = eligibility.get("reason")
        if (
            eligibility.get("eligible") is not False
            or reason != "eos_at_or_before_pending_boundary"
            or "scores" in row
            or "outputs" in row
            or run.get("stopped_on_failure") is not False
        ):
            raise PilotStateError(
                f"ineligible prompt evidence is malformed: {output}"
            )
        return {
            "prompt_id": prompt.prompt_id,
            "status": "ineligible_early_eos",
            "reason": reason,
            "generated_token_ids": eligibility.get("generated_token_ids", []),
            "artifact_directory": str(output),
            "artifacts_sha256": runner._file_sha256(artifacts_path),
        }
    raise PilotStateError(
        f"prompt artifact has non-resumable status {status!r}: {output}"
    )


def _verify_accepted_controls(
    run: Mapping[str, object],
    row: Mapping[str, object],
    acceptance: Mapping[str, object],
    output: Path,
) -> None:
    outputs = _require_mapping(row.get("outputs"), output / "run.json")
    scores = _require_mapping(row.get("scores"), output / "run.json")
    action_ids = {f"knorm:{ratio:.6g}" for ratio in runner.DEFAULT_RATIOS}
    output_actions = _require_mapping(
        outputs.get("actions"), output / "run.json"
    )
    reverse_actions = _require_mapping(
        outputs.get("reverse_actions"), output / "run.json"
    )
    score_actions = _require_mapping(
        scores.get("actions"), output / "run.json"
    )
    noop_outputs = outputs.get("noop_forks")
    noop_scores = scores.get("noop_forks")
    gates = acceptance.get("gates")
    if (
        run.get("status") != "completed"
        or run.get("passed") is not True
        or run.get("stopped_on_failure") is not False
        or acceptance.get("passed") is not True
        or not isinstance(gates, list)
        or not gates
        or not all(
            isinstance(gate, Mapping) and gate.get("passed") is True
            for gate in gates
        )
        or "uninterrupted" not in outputs
        or not isinstance(noop_outputs, list)
        or len(noop_outputs) != 2
        or not isinstance(noop_scores, list)
        or len(noop_scores) != 2
        or set(output_actions) != action_ids
        or set(reverse_actions) != action_ids
        or set(score_actions) != action_ids
        or "reference" not in scores
    ):
        raise PilotStateError(
            f"accepted controls or gates are incomplete: {output}"
        )


def _one_prompt_manifest(
    manifest: PromptManifest, prompt: EngineeringPrompt
) -> PromptManifest:
    return PromptManifest(
        source_path=manifest.source_path,
        source_sha256=manifest.source_sha256,
        source_bytes=manifest.source_bytes,
        official_dataset_path=manifest.official_dataset_path,
        official_dataset_sha256=manifest.official_dataset_sha256,
        official_dataset_bytes=manifest.official_dataset_bytes,
        official_dataset_rows=manifest.official_dataset_rows,
        selection_rule=manifest.selection_rule,
        prompts=(prompt,),
    )


def _summary(
    status: str,
    manifest: PromptManifest,
    ledger: Sequence[dict[str, object]],
    target_eligible: int,
    lock_sha256: str,
    *,
    current_prompt_id: str | None,
    phase: str,
) -> dict[str, object]:
    accepted = _count_status(ledger, "accepted")
    early_eos = [
        dict(entry)
        for entry in ledger
        if entry.get("status") == "ineligible_early_eos"
    ]
    return {
        "schema_version": "herald_v3.pilot_checkpoint.v1",
        "status": status,
        "phase": phase,
        "passed": status == "completed" and accepted >= target_eligible,
        "experiment_lock_sha256": lock_sha256,
        "target_eligible": target_eligible,
        "accepted_eligible": accepted,
        "ineligible_early_eos": len(early_eos),
        "processed": len(ledger),
        "roster_count": len(manifest.prompts),
        "current_prompt_id": current_prompt_id,
        "ledger": list(ledger),
        "early_eos_ledger": early_eos,
    }


def _count_status(ledger: Sequence[Mapping[str, object]], status: str) -> int:
    return sum(entry.get("status") == status for entry in ledger)


def _lock_configuration(
    lock: Mapping[str, object],
) -> Mapping[str, object]:
    return _require_mapping(lock.get("configuration"), Path(LOCK_NAME))


def _lock_prompt_ids(lock: Mapping[str, object]) -> list[object]:
    manifest = _require_mapping(lock.get("prompt_manifest"), Path(LOCK_NAME))
    prompt_ids = manifest.get("prompt_ids")
    if not isinstance(prompt_ids, list):
        raise PilotStateError(f"lock prompt IDs are malformed: {LOCK_NAME}")
    return prompt_ids


def _validate_checkpoint(
    checkpoint: Mapping[str, object],
    lock: Mapping[str, object],
    lock_sha256: str,
    path: Path,
) -> None:
    configuration = _lock_configuration(lock)
    ledger = checkpoint.get("ledger")
    early_eos = checkpoint.get("early_eos_ledger")
    if not isinstance(ledger, list) or not all(
        isinstance(entry, Mapping) for entry in ledger
    ):
        raise PilotStateError(
            f"resume checkpoint ledger is malformed: {path}"
        )
    expected_early_eos = [
        entry
        for entry in ledger
        if isinstance(entry, Mapping)
        and entry.get("status") == "ineligible_early_eos"
    ]
    expected_accepted = sum(
        isinstance(entry, Mapping) and entry.get("status") == "accepted"
        for entry in ledger
    )
    valid_statuses = {
        "starting",
        "running",
        "completed",
        "insufficient_coverage",
        "failed",
    }
    valid_phases = {
        "before_roster",
        "before_prompt",
        "after_prompt",
        "finished",
    }
    if (
        checkpoint.get("schema_version") != "herald_v3.pilot_checkpoint.v1"
        or checkpoint.get("experiment_lock_sha256") != lock_sha256
        or checkpoint.get("target_eligible")
        != configuration.get("target_eligible")
        or checkpoint.get("roster_count") != len(_lock_prompt_ids(lock))
        or checkpoint.get("processed") != len(ledger)
        or checkpoint.get("accepted_eligible") != expected_accepted
        or checkpoint.get("ineligible_early_eos") != len(expected_early_eos)
        or early_eos != expected_early_eos
        or checkpoint.get("status") not in valid_statuses
        or checkpoint.get("phase") not in valid_phases
    ):
        raise PilotStateError(
            f"resume checkpoint does not match experiment lock: {path}"
        )


def _validate_checkpoint_prefix(
    checkpoint: Mapping[str, object],
    recovered_ledger: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    prior = checkpoint.get("ledger")
    if not isinstance(prior, list) or (
        len(prior) > len(recovered_ledger)
        or prior != list(recovered_ledger[: len(prior)])
    ):
        raise PilotStateError(
            f"resume checkpoint disagrees with prompt artifacts: {path}"
        )


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PilotStateError(f"cannot read JSON evidence: {path}") from error
    if not isinstance(value, dict):
        raise PilotStateError(f"JSON evidence is not an object: {path}")
    return value


def _require_mapping(value: object, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PilotStateError(f"evidence object is malformed: {path}")
    return value


def _durable_write_json(path: Path, payload: Mapping[str, object]) -> None:
    text = (
        json.dumps(
            to_builtin(dict(payload)),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _sync_prompt_artifacts(output: Path) -> None:
    for name in (*sorted(REQUIRED_PROMPT_ARTIFACTS), "artifacts.json"):
        file_descriptor = os.open(output / name, os.O_RDONLY)
        try:
            os.fsync(file_descriptor)
        finally:
            os.close(file_descriptor)
    directory_fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _emit(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
