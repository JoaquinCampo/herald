"""Small offline runner for the eight-prompt engineering acceptance slice."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from herald_v3.engineering.engine import (
    ArmResult,
    model_signature,
    run_acceptance,
    to_builtin,
)
from herald_v3.engineering.prompts import (
    EngineeringPrompt,
    PromptManifest,
    load_prompt_manifest,
    write_prompt_manifest,
)
from herald_v3.engineering.scoring import (
    check_ifeval_resources,
    score_ifeval_gold,
    score_pair,
)

DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_LIMIT = 8
DEFAULT_RATIOS = (0.25, 0.5)


def tokenize_prompt(tokenizer: Any, prompt: EngineeringPrompt) -> Any:
    """Apply and verify the model chat template for one source prompt."""
    messages = [dict(message) for message in prompt.messages]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered != prompt.prompt_text:
        raise ValueError(
            "chat template output differs from saved prompt "
            f"{prompt.prompt_id}"
        )
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    templated_ids = _input_ids(templated)
    direct = tokenizer(
        prompt.prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    direct_ids = _input_ids(direct)
    if (
        templated_ids.shape != direct_ids.shape
        or not (templated_ids == direct_ids).all()
    ):
        raise ValueError(
            "chat template token IDs differ from source prompt "
            f"{prompt.prompt_id}"
        )
    return templated_ids


def run_engineering(
    model: Any,
    tokenizer: Any,
    manifest: PromptManifest,
    output: str | Path,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ratios: tuple[float, ...] = DEFAULT_RATIOS,
    seed: int = 0,
) -> dict[str, object]:
    """Run every selected prompt and retain failures in evidence."""
    if max_new_tokens <= 32:
        raise ValueError("max_new_tokens must exceed the 32-token boundary")
    resources = check_ifeval_resources()
    paths = _output_paths(Path(output))
    write_prompt_manifest(manifest, paths.prompt_manifest)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model.eval()
    eos_ids = _eos_ids(model, tokenizer)
    rows: list[dict[str, object]] = []
    log_lines = _start_log_lines(
        len(manifest.prompts), max_new_tokens, ratios, seed
    )
    _write_lines(paths.run_log, log_lines)
    evidence_base: dict[str, object] = {
        "schema_version": 1,
        "prompt_manifest": manifest.to_dict(),
        "configuration": {
            "max_new_tokens": max_new_tokens,
            "decision_tokens": 32,
            "actions": [
                {"name": "knorm", "removal_fraction": ratio}
                for ratio in ratios
            ],
            "seed": seed,
            "eos_ids": sorted(eos_ids),
            "decode_skip_special_tokens": True,
        },
        "model": to_builtin(model_signature(model)),
        "environment": _environment_manifest(
            model, tokenizer, resources, seed
        ),
        "source_manifest": _source_manifest(),
    }
    _write_checkpoint(paths.run_json, evidence_base, rows, stopped=False)
    stopped_on_failure = False
    for prompt in manifest.prompts:
        try:
            row = _run_prompt(
                model,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
                ratios=ratios,
                seed=seed,
            )
        except Exception as error:
            row = {
                "prompt": prompt.to_dict(),
                "status": "failed",
                "error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            }
        rows.append(row)
        acceptance = row.get("acceptance")
        log_lines.append(
            json.dumps(
                {
                    "event": "prompt",
                    "prompt_id": prompt.prompt_id,
                    "status": row.get("status"),
                    "acceptance_passed": acceptance.get("passed")
                    if isinstance(acceptance, dict)
                    else None,
                },
                sort_keys=True,
            )
        )
        _write_lines(paths.run_log, log_lines)
        row_failed = row.get("status") in {
            "failed",
            "eligible_with_gate_failures",
        }
        _write_checkpoint(
            paths.run_json, evidence_base, rows, stopped=row_failed
        )
        if row_failed:
            stopped_on_failure = True
            break

    passed = _overall_passed(rows)
    evidence = dict(evidence_base)
    evidence["status"] = "completed" if passed else "completed_with_failures"
    evidence["passed"] = passed
    evidence["stopped_on_failure"] = stopped_on_failure
    evidence["results"] = rows
    _write_json(paths.run_json, evidence)
    log_lines.append(
        json.dumps(
            {
                "event": "finish",
                "status": evidence["status"],
                "run_json": str(paths.run_json),
            },
            sort_keys=True,
        )
    )
    _write_lines(paths.run_log, log_lines)
    artifact_hashes = {
        str(path.name): _file_sha256(path)
        for path in (paths.prompt_manifest, paths.run_json, paths.run_log)
    }
    paths.artifacts.write_text(
        json.dumps(
            {"schema_version": 1, "sha256": artifact_hashes},
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``scripts/run_engineering.py``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="local model ID or snapshot"
    )
    parser.add_argument(
        "--prompts",
        required=True,
        help="saved prompt manifest or exposed inputs.json",
    )
    parser.add_argument(
        "--output", required=True, help="evidence output directory"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="total generated tokens, including the common 32-token prefix",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument(
        "--ifeval-arrow",
        default=None,
        help="Arrow cache path when --prompts points to exposed inputs.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.limit <= 0:
        parser.error("--limit must be positive")
    manifest = load_prompt_manifest(
        args.prompts,
        arrow_path=args.ifeval_arrow,
        limit=args.limit,
    )
    model, tokenizer = load_offline_model(args.model)
    evidence = run_engineering(
        model,
        tokenizer,
        manifest,
        args.output,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    return 0 if evidence.get("passed") is True else 1


def load_offline_model(model_path: str | Path) -> tuple[Any, Any]:
    """Load a local Transformers checkpoint without network access."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "the offline engineering runner requires torch and transformers"
        ) from error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        str(model_path), local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.to(device)  # type: ignore[arg-type]
    model.eval()  # type: ignore[no-untyped-call]
    return model, tokenizer


def _run_prompt(
    model: Any,
    tokenizer: Any,
    prompt: EngineeringPrompt,
    *,
    max_new_tokens: int,
    eos_ids: frozenset[int],
    ratios: tuple[float, ...],
    seed: int,
) -> dict[str, object]:
    input_ids = tokenize_prompt(tokenizer, prompt)
    result = run_acceptance(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        eos_ids=eos_ids,
        ratios=ratios,
    )
    result_dict = result.to_dict()
    row: dict[str, object] = {
        "prompt": prompt.to_dict(),
        "status": "ineligible"
        if not result.eligibility.eligible
        else "eligible",
        "seed": seed,
        "tokenization": {
            "input_ids": _input_id_list(input_ids),
            "input_length": int(input_ids.shape[1]),
            "chat_template_verified": True,
        },
        "acceptance": result_dict,
    }
    if result.uninterrupted is None:
        return row
    gold = {
        "prompt": prompt.user_prompt,
        "instruction_id_list": prompt.instruction_id_list,
        "kwargs": prompt.kwargs,
    }
    reference_text = _decode(tokenizer, result.uninterrupted.token_ids)
    outputs: dict[str, object] = {
        "uninterrupted": _continuation_output(
            tokenizer, result.uninterrupted
        ),
        "noop_forks": [
            _arm_output(tokenizer, arm) for arm in result.noop_forks
        ],
        "actions": {
            arm.action.action_id: _arm_output(tokenizer, arm)
            for arm in result.action_arms
        },
        "reverse_actions": {
            arm.action.action_id: _arm_output(tokenizer, arm)
            for arm in result.reverse_action_arms
        },
    }
    scores: dict[str, object] = {
        "reference": score_ifeval_gold(reference_text, gold).to_dict(),
        "noop_forks": [],
        "actions": {},
    }
    noop_scores = cast_list(scores["noop_forks"])
    for arm in result.noop_forks:
        pair = score_pair(
            reference_text,
            _decode(tokenizer, arm.continuation.token_ids),
            gold,
        )
        noop_scores.append(pair.to_dict())
    action_scores = cast_dict(scores["actions"])
    for arm in result.action_arms:
        pair = score_pair(
            reference_text,
            _decode(tokenizer, arm.continuation.token_ids),
            gold,
        )
        action_scores[arm.action.action_id] = pair.to_dict()
    row["outputs"] = outputs
    row["scores"] = scores
    row["status"] = (
        "accepted" if result.passed else "eligible_with_gate_failures"
    )
    return row


def _continuation_output(
    tokenizer: Any, continuation: Any
) -> dict[str, object]:
    return {
        "token_ids": list(continuation.token_ids),
        "text": _decode(tokenizer, continuation.token_ids),
        "termination_reason": continuation.termination_reason,
    }


def _arm_output(tokenizer: Any, arm: ArmResult) -> dict[str, object]:
    result = _continuation_output(tokenizer, arm.continuation)
    result["action"] = arm.action.to_dict()
    result["probe_enabled"] = arm.probe_enabled
    return result


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(
        tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    )


def _input_ids(value: Any) -> Any:
    if hasattr(value, "input_ids"):
        value = value.input_ids
    elif isinstance(value, dict):
        value = value["input_ids"]
    if not hasattr(value, "ndim") or value.ndim != 2 or value.shape[0] != 1:
        raise ValueError("tokenizer must return batch-1 input IDs")
    return value


def _input_id_list(value: Any) -> list[int]:
    return [int(item) for item in value.detach().cpu()[0].tolist()]


def _torch() -> Any:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("the engineering runner requires torch") from error
    return torch


def _eos_ids(model: Any, tokenizer: Any) -> frozenset[int]:
    """Return the union of tokenizer and generation-config EOS IDs."""
    values = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(
            getattr(model, "generation_config", None), "eos_token_id", None
        ),
    ]
    result: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, int) and not isinstance(value, bool):
            result.add(value)
            continue
        if isinstance(value, (list, tuple, set, frozenset)):
            candidates = value
        else:
            candidates = (value,)
        for candidate in candidates:
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                result.add(candidate)
    return frozenset(result)


def _start_log_lines(
    prompt_count: int,
    max_new_tokens: int,
    ratios: tuple[float, ...],
    seed: int,
) -> list[str]:
    return [
        json.dumps(
            {
                "event": "start",
                "prompt_count": prompt_count,
                "max_new_tokens": max_new_tokens,
                "ratios": list(ratios),
                "seed": seed,
            },
            sort_keys=True,
        )
    ]


def _overall_passed(rows: Sequence[dict[str, object]]) -> bool:
    """Require passed eligible cases; record early EOS as coverage."""
    return any(row.get("status") == "accepted" for row in rows) and all(
        row.get("status") in {"accepted", "ineligible"} for row in rows
    )


def _write_checkpoint(
    path: Path,
    base: dict[str, object],
    rows: Sequence[dict[str, object]],
    *,
    stopped: bool,
) -> None:
    payload = dict(base)
    payload["status"] = "running"
    payload["passed"] = False
    payload["stopped_on_failure"] = stopped
    payload["results"] = list(rows)
    _write_json(path, payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    text = (
        json.dumps(
            to_builtin(payload), sort_keys=True, ensure_ascii=False, indent=2
        )
        + "\n"
    )
    _write_text(path, text)


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    _write_text(path, "\n".join(lines) + "\n")


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


class _OutputPaths:
    def __init__(
        self,
        run_json: Path,
        run_log: Path,
        prompt_manifest: Path,
        artifacts: Path,
    ) -> None:
        self.run_json = run_json
        self.run_log = run_log
        self.prompt_manifest = prompt_manifest
        self.artifacts = artifacts


def _output_paths(output: Path) -> _OutputPaths:
    if output.suffix.lower() == ".json":
        output.parent.mkdir(parents=True, exist_ok=True)
        stem = output.with_suffix("")
        return _OutputPaths(
            output,
            stem.with_suffix(".log"),
            stem.with_name(stem.name + "-prompt-manifest.json"),
            stem.with_name(stem.name + "-artifacts.json"),
        )
    output.mkdir(parents=True, exist_ok=True)
    return _OutputPaths(
        output / "run.json",
        output / "run.log",
        output / "prompt-manifest.json",
        output / "artifacts.json",
    )


def _source_manifest() -> dict[str, object]:
    package = Path(__file__).resolve().parent
    files = {
        str(path.relative_to(package)): _file_sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    return {"root": str(package), "sha256": files}


def _environment_manifest(
    model: Any,
    tokenizer: Any,
    resources: dict[str, object],
    seed: int,
) -> dict[str, object]:
    torch = _torch()
    config = getattr(model, "config", None)
    parameter = next(model.parameters())
    packages: dict[str, str | None] = {}
    for name in (
        "torch",
        "transformers",
        "kvpress",
        "datasets",
        "nltk",
        "langdetect",
        "immutabledict",
        "absl-py",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
        "seed": seed,
        "packages": packages,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "attention_backend": getattr(config, "_attn_implementation", None),
        "tokenizer_name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "resources": resources,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cast_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("evidence field is not a list")
    return value


def cast_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError("evidence field is not an object")
    return value
