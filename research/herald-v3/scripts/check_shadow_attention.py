#!/usr/bin/env python3
# mypy: disable-error-code=import-not-found
# mypy: follow_imports=skip
"""Preflight or run the frozen one-case SDPA shadow-attention slice.

Owner workflow after the three files are frozen::

    uv run python scripts/check_shadow_attention.py --prepare-seal \
        results/shadow-attention-seal-v1.json --model /path/to/snapshot \
        --tokenizer /path/to/tokenizer

Then use the printed seal SHA-256 for both a CPU preflight and the one
production run.  Every output is exclusive, and a production run never
overwrites an older failed result.
"""

# ruff: noqa: E402, I001

import argparse
import json
import os
import sys
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import discover_retrieval_heads as legacy_cli
from herald_v3 import retrieval_discovery, shadow_attention


SEAL_SCHEMA_VERSION = "herald_v3.shadow_attention_execution_seal.v1"
RESULT_SCHEMA_VERSION = "herald_v3.shadow_attention.v1"
INTEGRITY_SCHEMA_VERSION = "herald_v3.shadow_attention_integrity.v1"
PINNED_MODEL_ID = legacy_cli.PINNED_MODEL_ID
PINNED_MODEL_REVISION = legacy_cli.PINNED_MODEL_REVISION
PINNED_TOKENIZER_REVISION = legacy_cli.PINNED_TOKENIZER_REVISION
TOKENIZER_IDENTITY_DEFAULT = legacy_cli.TOKENIZER_IDENTITY_DEFAULT
MODEL_IDENTITY_DEFAULT = legacy_cli.MODEL_IDENTITY_DEFAULT
PINNED_PACKAGES = legacy_cli.PINNED_PACKAGES
REQUIRED_MODEL_FILES = legacy_cli.REQUIRED_MODEL_FILES
MAX_NEW_TOKENS = shadow_attention.MAX_NEW_TOKENS
SEED = shadow_attention.SEED
EXPECTED_ARCHITECTURE = dict(legacy_cli.EXPECTED_ARCHITECTURE)


class CliIntegrityError(RuntimeError):
    """Raised when a seal, input, or immutable output contract is invalid."""


def _shadow_runtime_contract() -> dict[str, object]:
    """Describe the exact runtime accepted by this SDPA-only slice."""
    import torch

    device_name: str | None = None
    total_memory_bytes = 0
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(device)
        total_memory_bytes = int(
            torch.cuda.get_device_properties(device).total_memory
        )
    return {
        "device_type": "cuda",
        "dtype": "bfloat16",
        "device_name": device_name,
        "total_memory_bytes": total_memory_bytes,
        "seed": SEED,
        "attention_backends": ["sdpa"],
    }


def _shadow_protocol() -> dict[str, object]:
    return {
        "case_id": shadow_attention.EXPECTED_CASE_ID,
        "case_grid_sha256": shadow_attention.EXPECTED_CASE_GRID_SHA256,
        "time_cap_seconds": shadow_attention.TIME_CAP_SECONDS,
        "continuations": 2,
        "diagnostic": "postrotary_qk_shadow_only",
        "prefill_diagnostic": False,
        "head_selection": False,
        "head_labels": False,
        "ifeval": False,
        "fit": False,
    }


def _verify_shadow_contract_metadata(seal: Mapping[str, object]) -> None:
    """Require exact scope metadata before any model or tokenizer load."""
    if seal.get("seed") != SEED:
        raise CliIntegrityError("shadow seed contract differs")
    if seal.get("attention_backend") != "sdpa":
        raise CliIntegrityError("shadow backend contract differs")
    greedy = _mapping(seal, "greedy")
    if greedy != {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS}:
        raise CliIntegrityError("shadow greedy contract differs")
    runtime = _mapping(seal, "runtime")
    if runtime != _shadow_runtime_contract():
        raise CliIntegrityError("shadow runtime contract differs")
    model = _mapping(seal, "model")
    architecture = _mapping(model, "architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise CliIntegrityError("shadow model architecture differs")
    protocol = _mapping(seal, "protocol")
    if protocol != _shadow_protocol():
        raise CliIntegrityError("shadow protocol contract differs")


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    if args.prepare_seal is not None:
        try:
            _validate_preparation_args(args)
            seal = _prepare_execution_seal(
                args.prepare_seal,
                args.model,
                args.tokenizer,
                args.source_root,
                args.tokenizer_identity,
                args.model_identity,
            )
            print(
                json.dumps(
                    {"seal": seal, "sha256": _sha256_file(args.prepare_seal)},
                    sort_keys=True,
                )
            )
            return 0
        except Exception as error:
            print(
                json.dumps(
                    _failure_payload(error, "seal_preparation"),
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 1

    try:
        _validate_run_args(args)
        run_dir = _reserve_run_dir(args.output)
    except Exception as error:
        print(
            json.dumps(
                _failure_payload(error, "run_reservation"), sort_keys=True
            ),
            file=sys.stderr,
        )
        return 1

    context: dict[str, object] = {
        "case_id": shadow_attention.EXPECTED_CASE_ID,
        "case_prompt_token_ids": None,
        "unwrapped_token_ids": None,
        "wrapped_token_ids": None,
        "first_divergence": None,
        "execution_seal_sha256": None,
        "preflight_sha256": None,
    }
    try:
        result = _execute(args, run_dir, context)
        print(json.dumps(result, sort_keys=True))
        return (
            0
            if result.get("status") in {"completed", "preflight_passed"}
            else 1
        )
    except Exception as error:
        result = _preserve_failure(args.output, run_dir, context, error)
        print(json.dumps(result, sort_keys=True), file=sys.stderr)
        return 1


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=retrieval_discovery.DEFAULT_SOURCE_ROOT,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--execution-seal", "--seal", dest="execution_seal", type=Path
    )
    parser.add_argument(
        "--execution-seal-sha256",
        "--expected-seal-sha256",
        dest="execution_seal_sha256",
    )
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--prepare-seal", type=Path, metavar="PATH")
    parser.add_argument(
        "--tokenizer-identity", type=Path, default=TOKENIZER_IDENTITY_DEFAULT
    )
    parser.add_argument(
        "--model-identity", type=Path, default=MODEL_IDENTITY_DEFAULT
    )
    return parser


def _validate_preparation_args(args: argparse.Namespace) -> None:
    required = {
        "--model": args.model,
        "--tokenizer": args.tokenizer,
        "--source-root": args.source_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise CliIntegrityError(
            "seal preparation requires " + ", ".join(missing)
        )
    if args.output is not None or args.execution_seal is not None:
        raise CliIntegrityError(
            "seal preparation does not accept --output or --execution-seal"
        )


def _validate_run_args(args: argparse.Namespace) -> None:
    required = {
        "--model": args.model,
        "--tokenizer": args.tokenizer,
        "--output": args.output,
        "--execution-seal": args.execution_seal,
        "--execution-seal-sha256": args.execution_seal_sha256,
    }
    missing = [
        name for name, value in required.items() if value in (None, "")
    ]
    if missing:
        raise CliIntegrityError("run requires " + ", ".join(missing))


def _prepare_execution_seal(
    output: Path,
    model_path: Path,
    tokenizer_path: Path,
    source_root: Path,
    tokenizer_identity_path: Path,
    model_identity_path: Path,
) -> dict[str, object]:
    if output.exists():
        raise CliIntegrityError(f"refusing existing seal: {output}")
    source_manifest = retrieval_discovery.verify_source_manifest(source_root)
    source_manifest_path = source_root.resolve().parent / "manifest.json"
    tokenizer_identity = _read_json_object(tokenizer_identity_path)
    tokenizer_files = legacy_cli._verify_tokenizer_files(
        tokenizer_path, tokenizer_identity, expected=None
    )
    chat_template_sha256 = legacy_cli._chat_template_hash_from_config(
        tokenizer_path
    )
    if chat_template_sha256 != tokenizer_identity.get("chat_template_sha256"):
        raise CliIntegrityError(
            "tokenizer chat template does not match identity"
        )
    model_config, model_files = legacy_cli._verify_model_files(
        model_path, model_identity_path, expected=None
    )
    legacy_cli._verify_architecture(model_config)
    if model_path.resolve().name != PINNED_MODEL_REVISION:
        raise CliIntegrityError(
            "model path is not the pinned snapshot revision"
        )
    if tokenizer_identity.get("revision") != PINNED_TOKENIZER_REVISION:
        raise CliIntegrityError("tokenizer identity revision is not pinned")
    if model_config.get("_name_or_path") not in (None, PINNED_MODEL_ID):
        raise CliIntegrityError("model config identity is not pinned")
    legacy_cli._assert_frozen_module()
    seal: dict[str, object] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "repository": retrieval_discovery.REPOSITORY,
        "commit": retrieval_discovery.PINNED_COMMIT,
        "source_manifest": {
            "path": legacy_cli._relative_path(source_manifest_path),
            "sha256": _sha256_file(source_manifest_path),
            "schema_version": source_manifest["schema_version"],
        },
        "code": {
            "shadow_module": {
                "path": "src/herald_v3/shadow_attention.py",
                "sha256": _sha256_file(
                    PROJECT_ROOT / "src/herald_v3/shadow_attention.py"
                ),
            },
            "shadow_cli": {
                "path": "scripts/check_shadow_attention.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "discovery_module": {
                "path": "src/herald_v3/retrieval_discovery.py",
                "sha256": _sha256_file(
                    PROJECT_ROOT / "src/herald_v3/retrieval_discovery.py"
                ),
            },
            "discovery_cli": {
                "path": "scripts/discover_retrieval_heads.py",
                "sha256": _sha256_file(
                    PROJECT_ROOT / "scripts/discover_retrieval_heads.py"
                ),
            },
        },
        "tokenizer": {
            "revision": PINNED_TOKENIZER_REVISION,
            "identity_sha256": _sha256_file(tokenizer_identity_path),
            "files_sha256": tokenizer_files,
            "chat_template_sha256": chat_template_sha256,
        },
        "model": {
            "model_id": PINNED_MODEL_ID,
            "revision": PINNED_MODEL_REVISION,
            "identity_sha256": _sha256_file(model_identity_path),
            "config_sha256": model_files["config.json"],
            "weight_index_sha256": model_files[
                "model.safetensors.index.json"
            ],
            "weight_files_sha256": {
                name: digest
                for name, digest in model_files.items()
                if name not in REQUIRED_MODEL_FILES
            },
            "architecture": EXPECTED_ARCHITECTURE,
        },
        "versions": legacy_cli._runtime_versions(),
        "runtime": _shadow_runtime_contract(),
        "seed": SEED,
        "greedy": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "attention_backend": "sdpa",
        "protocol": _shadow_protocol(),
    }
    legacy_cli._write_json_exclusive(output, seal)
    return seal


def _verify_execution_seal(
    path: Path,
    expected_sha256: str,
    model_path: Path,
    tokenizer_path: Path,
    source_root: Path,
    tokenizer_identity_path: Path,
    model_identity_path: Path,
) -> tuple[dict[str, object], str]:
    raw = _read_bytes(path)
    actual = _sha256(raw)
    if actual != expected_sha256:
        raise CliIntegrityError("execution seal SHA-256 does not match")
    seal = _parse_json_object(raw, "execution seal")
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise CliIntegrityError("execution seal schema is invalid")
    if (
        seal.get("repository") != retrieval_discovery.REPOSITORY
        or seal.get("commit") != retrieval_discovery.PINNED_COMMIT
    ):
        raise CliIntegrityError(
            "execution seal upstream identity is not pinned"
        )
    source = _mapping(seal, "source_manifest")
    source_path = source_root.resolve().parent / "manifest.json"
    if source.get("path") != legacy_cli._relative_path(
        source_path
    ) or source.get("sha256") != _sha256_file(source_path):
        raise CliIntegrityError("source manifest identity differs from seal")
    source_manifest = retrieval_discovery.verify_source_manifest(source_root)
    if source.get("schema_version") != source_manifest.get("schema_version"):
        raise CliIntegrityError("source manifest schema differs from seal")
    code = _mapping(seal, "code")
    for name, path_value in (
        ("shadow_module", PROJECT_ROOT / "src/herald_v3/shadow_attention.py"),
        ("shadow_cli", Path(__file__).resolve()),
        (
            "discovery_module",
            PROJECT_ROOT / "src/herald_v3/retrieval_discovery.py",
        ),
        (
            "discovery_cli",
            PROJECT_ROOT / "scripts/discover_retrieval_heads.py",
        ),
    ):
        entry = _mapping(code, name)
        if entry.get("sha256") != _sha256_file(path_value):
            raise CliIntegrityError(f"sealed {name} hash differs")
    tokenizer_identity = _read_json_object(tokenizer_identity_path)
    tokenizer = _mapping(seal, "tokenizer")
    if tokenizer.get("identity_sha256") != _sha256_file(
        tokenizer_identity_path
    ):
        raise CliIntegrityError("tokenizer identity hash differs from seal")
    tokenizer_files = legacy_cli._verify_tokenizer_files(
        tokenizer_path,
        tokenizer_identity,
        expected=_string_mapping(tokenizer, "files_sha256"),
    )
    if (
        tokenizer_files != _string_mapping(tokenizer, "files_sha256")
        or tokenizer.get("revision") != PINNED_TOKENIZER_REVISION
    ):
        raise CliIntegrityError(
            "tokenizer files or revision differ from seal"
        )
    if tokenizer.get(
        "chat_template_sha256"
    ) != legacy_cli._chat_template_hash_from_config(tokenizer_path):
        raise CliIntegrityError("tokenizer chat template differs from seal")
    model = _mapping(seal, "model")
    if (
        model.get("model_id") != PINNED_MODEL_ID
        or model.get("revision") != PINNED_MODEL_REVISION
    ):
        raise CliIntegrityError("model identity is not pinned")
    if (
        model.get("identity_sha256") != _sha256_file(model_identity_path)
        or model_path.resolve().name != PINNED_MODEL_REVISION
    ):
        raise CliIntegrityError("model path or identity differs from seal")
    config, model_files = legacy_cli._verify_model_files(
        model_path, model_identity_path, expected=model
    )
    legacy_cli._verify_architecture(config)
    if config.get("_name_or_path") not in (None, PINNED_MODEL_ID):
        raise CliIntegrityError("model config identity differs from seal")
    if model_files.get("config.json") != model.get(
        "config_sha256"
    ) or model_files.get("model.safetensors.index.json") != model.get(
        "weight_index_sha256"
    ):
        raise CliIntegrityError("model required file hashes differ from seal")
    weights = {
        name: digest
        for name, digest in model_files.items()
        if name not in REQUIRED_MODEL_FILES
    }
    if weights != _string_mapping(model, "weight_files_sha256"):
        raise CliIntegrityError("model weight hashes differ from seal")
    legacy_cli._verify_versions(seal)
    _verify_shadow_contract_metadata(seal)
    return seal, actual


def _execute(
    args: argparse.Namespace, run_dir: Path, context: dict[str, object]
) -> dict[str, object]:
    assert (
        args.model is not None
        and args.tokenizer is not None
        and args.output is not None
    )
    assert (
        args.execution_seal is not None
        and args.execution_seal_sha256 is not None
    )
    if not args.cpu_preflight and (
        args.tokenizer_identity.resolve()
        != TOKENIZER_IDENTITY_DEFAULT.resolve()
        or args.model_identity.resolve() != MODEL_IDENTITY_DEFAULT.resolve()
    ):
        raise CliIntegrityError(
            "production runs require pinned identity evidence paths"
        )
    seal, seal_sha = _verify_execution_seal(
        args.execution_seal,
        args.execution_seal_sha256,
        args.model,
        args.tokenizer,
        args.source_root,
        args.tokenizer_identity,
        args.model_identity,
    )
    context["execution_seal_sha256"] = seal_sha
    _set_offline_environment()
    tokenizer = _load_tokenizer(args.tokenizer)
    _verify_loaded_tokenizer(tokenizer, seal)
    case, cases, grid_sha = shadow_attention.select_fixed_case(
        tokenizer, args.source_root, require_pinned_grid=True
    )
    context["case_id"] = case.case_id
    context["case_prompt_token_ids"] = list(case.prompt_ids)
    preflight = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "passed",
        "mode": "cpu_preflight" if args.cpu_preflight else "production",
        "repository": retrieval_discovery.REPOSITORY,
        "commit": retrieval_discovery.PINNED_COMMIT,
        "execution_seal_sha256": seal_sha,
        "source_manifest_sha256": _sha256_file(
            args.source_root.resolve().parent / "manifest.json"
        ),
        "shadow_module_sha256": _sha256_file(
            PROJECT_ROOT / "src/herald_v3/shadow_attention.py"
        ),
        "shadow_cli_sha256": _sha256_file(Path(__file__).resolve()),
        "model_path": str(args.model.resolve()),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "seed": SEED,
        "case_count": len(cases),
        "case_grid_sha256": grid_sha,
        "selected_case": case.to_dict(),
        "source_manifest": retrieval_discovery.verify_source_manifest(
            args.source_root
        ),
        "runtime_contract": dict(_mapping(seal, "runtime")),
    }
    preflight_path = run_dir / "shadow-preflight.json"
    legacy_cli._write_json_exclusive(preflight_path, preflight)
    preflight_sha = _sha256_file(preflight_path)
    context["preflight_sha256"] = preflight_sha
    if args.cpu_preflight:
        result: dict[str, object] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "status": "preflight_passed",
            "passed": True,
            "mode": "cpu_preflight",
            "case_id": case.case_id,
            "case_grid_sha256": grid_sha,
            "execution_seal_sha256": seal_sha,
            "preflight_sha256": preflight_sha,
        }
        return _write_result_chain(run_dir, args.output, result, context)
    runtime = _require_cuda_bfloat16(seal)
    model = _load_model(args.model)
    _verify_loaded_model(model, seal)
    pair = shadow_attention.run_paired_continuations(
        model,
        tokenizer,
        case,
        runtime=shadow_attention.ShadowRuntime(
            deadline_seconds=shadow_attention.TIME_CAP_SECONDS
        ),
        strict_contract=True,
        state=context,
    )
    result = pair.to_dict()
    result["schema_version"] = RESULT_SCHEMA_VERSION
    result["mode"] = "production"
    result["runtime"] = runtime
    result["execution_seal_sha256"] = seal_sha
    result["preflight_sha256"] = preflight_sha
    return _write_result_chain(run_dir, args.output, result, context)


def _write_result_chain(
    run_dir: Path,
    output: Path,
    result: Mapping[str, object],
    context: MutableMapping[str, object],
) -> dict[str, object]:
    final = dict(result)
    final["integrity"] = {
        "execution_seal_sha256": context.get("execution_seal_sha256"),
        "preflight_sha256": context.get("preflight_sha256"),
    }
    legacy_cli._write_json_exclusive(output, final)
    result_sha = _sha256_file(output)
    integrity = {
        "schema_version": INTEGRITY_SCHEMA_VERSION,
        "execution_seal_sha256": context.get("execution_seal_sha256"),
        "preflight_sha256": context.get("preflight_sha256"),
        "result_sha256": result_sha,
    }
    legacy_cli._write_json_exclusive(
        run_dir / "shadow-integrity.json", integrity
    )
    return final


def _preserve_failure(
    output: Path | None,
    run_dir: Path,
    context: Mapping[str, object],
    error: Exception,
) -> dict[str, object]:
    assert output is not None
    failure = _failure_payload(error, "execution")
    for name in (
        "case_id",
        "case_prompt_token_ids",
        "unwrapped_token_ids",
        "wrapped_token_ids",
        "first_divergence",
        "diagnostics",
    ):
        failure[name] = context.get(name)
    failure["integrity"] = {
        "execution_seal_sha256": context.get("execution_seal_sha256"),
        "preflight_sha256": context.get("preflight_sha256"),
    }
    legacy_cli._write_json_exclusive(output, failure)
    result_sha = _sha256_file(output)
    legacy_cli._write_json_exclusive(
        run_dir / "shadow-integrity.json",
        {
            "schema_version": INTEGRITY_SCHEMA_VERSION,
            **cast(dict[str, object], failure["integrity"]),
            "result_sha256": result_sha,
        },
    )
    return failure


def _load_tokenizer(path: Path) -> Any:
    return legacy_cli._load_tokenizer(path)


def _load_model(path: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    _set_offline_environment()
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.to(torch.device("cuda"))
    model.eval()
    return model


def _verify_loaded_tokenizer(
    tokenizer: Any, seal: Mapping[str, object]
) -> None:
    expected = _mapping(seal, "tokenizer").get("chat_template_sha256")
    observed = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(observed, str)
        or legacy_cli._sha256_text(observed) != expected
    ):
        raise CliIntegrityError(
            "loaded tokenizer chat template differs from seal"
        )


def _verify_loaded_model(model: Any, seal: Mapping[str, object]) -> None:
    import torch

    try:
        device = next(model.parameters()).device
        dtypes = {parameter.dtype for parameter in model.parameters()}
    except (AttributeError, StopIteration) as error:
        raise CliIntegrityError("loaded model has no parameters") from error
    if device.type != "cuda" or dtypes != {torch.bfloat16}:
        raise CliIntegrityError("loaded model is not CUDA BF16")
    if (
        getattr(getattr(model, "config", None), "_attn_implementation", None)
        != "sdpa"
    ):
        raise CliIntegrityError("loaded model attention backend is not SDPA")
    retrieval_discovery.verify_model_gqa_mapping(model)
    expected_name = _mapping(seal, "runtime").get("device_name")
    if (
        isinstance(expected_name, str)
        and torch.cuda.get_device_name(0) != expected_name
    ):
        raise CliIntegrityError(
            "loaded model device differs from execution seal"
        )


def _require_cuda_bfloat16(seal: Mapping[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise CliIntegrityError("production shadow slice requires CUDA BF16")
    expected = _mapping(seal, "runtime")
    name = torch.cuda.get_device_name(torch.cuda.current_device())
    if (
        isinstance(expected.get("device_name"), str)
        and name != expected["device_name"]
    ):
        raise CliIntegrityError(
            "CUDA device name differs from execution seal"
        )
    total = int(torch.cuda.get_device_properties(0).total_memory)
    if isinstance(expected.get("total_memory_bytes"), int) and total < cast(
        int, expected["total_memory_bytes"]
    ):
        raise CliIntegrityError("CUDA device memory is below execution seal")
    return {
        "device_type": "cuda",
        "dtype": "bfloat16",
        "device_name": name,
        "total_memory_bytes": total,
        "torch_version": legacy_cli._runtime_versions()["torch"],
        "transformers_version": legacy_cli._runtime_versions()[
            "transformers"
        ],
        "seed": SEED,
        "attention_backends": ["sdpa"],
    }


def _set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _reserve_run_dir(output: Path | None) -> Path:
    if output is None:
        raise CliIntegrityError("--output is required")
    resolved = output.resolve()
    if resolved.exists():
        raise CliIntegrityError(f"refusing existing result: {resolved}")
    run_dir = resolved.parent
    if run_dir.exists():
        raise CliIntegrityError(f"refusing existing run directory: {run_dir}")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    return run_dir


def _failure_payload(error: Exception, stage: str) -> dict[str, object]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "failed",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CliIntegrityError(
            f"cannot read required file: {path}"
        ) from error


def _parse_json_object(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CliIntegrityError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CliIntegrityError(f"{description} must be an object")
    return cast(dict[str, object], value)


def _read_json_object(path: Path) -> dict[str, object]:
    return _parse_json_object(_read_bytes(path), str(path))


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    nested = value.get(name)
    if not isinstance(nested, Mapping):
        raise CliIntegrityError(f"seal field is not an object: {name}")
    return nested


def _string_mapping(value: Mapping[str, object], name: str) -> dict[str, str]:
    nested = _mapping(value, name)
    if not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in nested.items()
    ):
        raise CliIntegrityError(f"seal field is not a string map: {name}")
    return {
        key: item
        for key, item in nested.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _sha256(raw: bytes) -> str:
    return legacy_cli._sha256(raw)


def _sha256_file(path: Path) -> str:
    return legacy_cli._sha256_file(path)


if __name__ == "__main__":
    raise SystemExit(main())
