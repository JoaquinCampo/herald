#!/usr/bin/env python3
# mypy: disable-error-code=import-not-found
"""Run or preflight the bounded synthetic retrieval-head discovery assay.

The owner creates a seal after the code freeze with, for example::

    uv run python scripts/discover_retrieval_heads.py --prepare-seal \
        results/retrieval-discovery-seal.json \
        --model /path/to/Qwen2.5-7B-Instruct/snapshots/a09a354... \
        --tokenizer /path/to/tokenizer \
        --source-root data/retrieval-discovery-v1/source

The resulting seal digest is then supplied to ``--execution-seal-sha256``
for both ``--cpu-preflight`` and the production run.
"""

# ruff: noqa: E402, I001

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3 import retrieval_discovery


SEAL_SCHEMA_VERSION = "herald_v3.retrieval_discovery_execution_seal.v1"
SELECTED_CELLS_SCHEMA_VERSION = (
    "herald_v3.retrieval_discovery_selected_cells.v1"
)
FROZEN_MODULE_SHA256 = (
    "61b1ff4a57d4777aded3cd4ce4e491a4350c92685a6f3ca4e2329457304865b8"
)
PINNED_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PINNED_MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PINNED_TOKENIZER_REVISION = PINNED_MODEL_REVISION
TOKENIZER_IDENTITY_DEFAULT = (
    PROJECT_ROOT / "data" / "lookahead-v1" / "tokenizer-identity.json"
)
MODEL_IDENTITY_DEFAULT = (
    PROJECT_ROOT / "results" / "retrieval-model-identity-orion.sha256"
)
PINNED_PACKAGES = ("torch", "transformers", "tokenizers", "numpy")
REQUIRED_MODEL_FILES = frozenset(
    {"config.json", "model.safetensors.index.json"}
)
EXPECTED_ARCHITECTURE = {
    "model_type": "qwen2",
    "num_hidden_layers": retrieval_discovery.EXPECTED_LAYERS,
    "num_attention_heads": retrieval_discovery.EXPECTED_QUERY_HEADS,
    "num_key_value_heads": retrieval_discovery.EXPECTED_KV_HEADS,
    "num_key_value_groups": retrieval_discovery.EXPECTED_GQA_GROUPS,
}


class CliIntegrityError(RuntimeError):
    """Raised when an execution identity or output contract is invalid."""


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
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
            digest = _sha256_file(args.prepare_seal)
            print(
                json.dumps({"seal": seal, "sha256": digest}, sort_keys=True)
            )
            return 0
        except Exception as error:
            payload = _failure_payload(error, "seal_preparation")
            print(json.dumps(payload, sort_keys=True), file=sys.stderr)
            return 1

    try:
        _validate_run_args(args)
        run_dir = _reserve_run_dir(args.output)
    except Exception as error:
        payload = _failure_payload(error, "run_reservation")
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1

    context: dict[str, str | None] = {
        "execution_seal_sha256": None,
        "preflight_sha256": None,
        "selected_cells_sha256": None,
        "result_sha256": None,
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
        "--execution-seal",
        "--seal",
        dest="execution_seal",
        type=Path,
    )
    parser.add_argument(
        "--execution-seal-sha256",
        "--expected-seal-sha256",
        dest="execution_seal_sha256",
    )
    parser.add_argument(
        "--cpu-preflight",
        action="store_true",
        help="verify the seal and all inputs without loading model weights",
    )
    parser.add_argument(
        "--prepare-seal",
        type=Path,
        metavar="PATH",
        help="create a new seal from frozen local artifacts",
    )
    parser.add_argument(
        "--tokenizer-identity",
        type=Path,
        default=TOKENIZER_IDENTITY_DEFAULT,
    )
    parser.add_argument(
        "--model-identity",
        type=Path,
        default=MODEL_IDENTITY_DEFAULT,
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


def _execute(
    args: argparse.Namespace,
    run_dir: Path,
    context: dict[str, str | None],
) -> dict[str, object]:
    assert args.model is not None
    assert args.tokenizer is not None
    assert args.output is not None
    assert args.execution_seal is not None
    assert args.execution_seal_sha256 is not None
    if not args.cpu_preflight and (
        args.tokenizer_identity.resolve()
        != TOKENIZER_IDENTITY_DEFAULT.resolve()
        or args.model_identity.resolve() != MODEL_IDENTITY_DEFAULT.resolve()
    ):
        raise CliIntegrityError(
            "production runs require the pinned tokenizer and model "
            "identity evidence"
        )
    seal, seal_sha256 = _verify_execution_seal(
        args.execution_seal,
        args.execution_seal_sha256,
        args.model,
        args.tokenizer,
        args.source_root,
        args.tokenizer_identity,
        args.model_identity,
    )
    context["execution_seal_sha256"] = seal_sha256
    source_manifest = retrieval_discovery.verify_source_manifest(
        args.source_root
    )
    _set_offline_environment()
    tokenizer = _load_tokenizer(args.tokenizer)
    _verify_loaded_tokenizer(tokenizer, seal)
    cases = retrieval_discovery.build_cases(tokenizer, args.source_root)
    preflight = _preflight_payload(
        seal,
        seal_sha256,
        args,
        source_manifest,
        cases,
        tokenizer,
    )
    preflight_path = run_dir / "discovery-preflight.json"
    _write_json_exclusive(preflight_path, preflight)
    preflight_sha256 = _sha256_file(preflight_path)
    context["preflight_sha256"] = preflight_sha256

    if args.cpu_preflight:
        result: dict[str, object] = {
            "schema_version": retrieval_discovery.RESULT_SCHEMA_VERSION,
            "status": "preflight_passed",
            "passed": True,
            "mode": "cpu_preflight",
            "repository": retrieval_discovery.REPOSITORY,
            "commit": retrieval_discovery.PINNED_COMMIT,
            "case_count": len(cases),
            "source_manifest": source_manifest,
            "verified_architecture": EXPECTED_ARCHITECTURE,
        }
    else:
        runtime_observation = _require_cuda_bfloat16(seal)
        model = _load_model(args.model)
        _verify_loaded_model(model, seal)
        result = retrieval_discovery.run_discovery(
            model,
            tokenizer,
            cases,
            source_manifest=source_manifest,
        )
        result["runtime"] = runtime_observation
    return _write_result_chain(
        run_dir,
        args.output,
        result,
        seal_sha256,
        preflight_sha256,
        context,
    )


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
    tokenizer_files = _verify_tokenizer_files(
        tokenizer_path, tokenizer_identity, expected=None
    )
    chat_template_sha256 = _chat_template_hash_from_config(tokenizer_path)
    if chat_template_sha256 != tokenizer_identity.get("chat_template_sha256"):
        raise CliIntegrityError(
            "tokenizer chat template does not match identity"
        )
    model_config, model_files = _verify_model_files(
        model_path, model_identity_path, expected=None
    )
    _verify_architecture(model_config)
    if model_path.resolve().name != PINNED_MODEL_REVISION:
        raise CliIntegrityError(
            "model path is not the pinned snapshot revision"
        )
    if tokenizer_identity.get("revision") != PINNED_TOKENIZER_REVISION:
        raise CliIntegrityError("tokenizer identity revision is not pinned")
    if model_config.get("_name_or_path") not in (None, PINNED_MODEL_ID):
        raise CliIntegrityError("model config identity is not pinned")
    _assert_frozen_module()
    versions = _runtime_versions()
    runtime = _runtime_contract()
    seal: dict[str, object] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "repository": retrieval_discovery.REPOSITORY,
        "commit": retrieval_discovery.PINNED_COMMIT,
        "source_manifest": {
            "path": _relative_path(source_manifest_path),
            "sha256": _sha256_file(source_manifest_path),
            "schema_version": source_manifest["schema_version"],
        },
        "code": {
            "module": {
                "path": "src/herald_v3/retrieval_discovery.py",
                "sha256": FROZEN_MODULE_SHA256,
            },
            "cli": {
                "path": "scripts/discover_retrieval_heads.py",
                "sha256": _sha256_file(Path(__file__).resolve()),
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
        "versions": versions,
        "runtime": runtime,
        "seed": retrieval_discovery.SEED,
        "greedy": {"do_sample": False, "max_new_tokens": 50},
        "attention_backends": ["eager", "sdpa"],
    }
    _write_json_exclusive(output, seal)
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
    actual_sha256 = _sha256(raw)
    if actual_sha256 != expected_sha256:
        raise CliIntegrityError("execution seal SHA-256 does not match")
    value = _parse_json_object(raw, "execution seal")
    if value.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise CliIntegrityError("execution seal schema is invalid")
    if value.get("repository") != retrieval_discovery.REPOSITORY:
        raise CliIntegrityError("execution seal repository is not pinned")
    if value.get("commit") != retrieval_discovery.PINNED_COMMIT:
        raise CliIntegrityError("execution seal commit is not pinned")
    source = _mapping(value, "source_manifest")
    source_path = source_root.resolve().parent / "manifest.json"
    if source.get("path") != _relative_path(source_path):
        raise CliIntegrityError("source manifest path differs from seal")
    if source.get("sha256") != _sha256_file(source_path):
        raise CliIntegrityError("source manifest SHA-256 does not match seal")
    source_manifest = retrieval_discovery.verify_source_manifest(source_root)
    if source.get("schema_version") != source_manifest.get("schema_version"):
        raise CliIntegrityError("source manifest schema does not match seal")
    code = _mapping(value, "code")
    module = _mapping(code, "module")
    cli = _mapping(code, "cli")
    if module.get("sha256") != FROZEN_MODULE_SHA256:
        raise CliIntegrityError("frozen discovery module hash is not pinned")
    if _sha256_file(
        PROJECT_ROOT / "src/herald_v3/retrieval_discovery.py"
    ) != module.get("sha256"):
        raise CliIntegrityError("discovery module hash does not match seal")
    if _sha256_file(Path(__file__).resolve()) != cli.get("sha256"):
        raise CliIntegrityError("discovery CLI hash does not match seal")
    tokenizer_identity = _read_json_object(tokenizer_identity_path)
    tokenizer = _mapping(value, "tokenizer")
    if tokenizer.get("identity_sha256") != _sha256_file(
        tokenizer_identity_path
    ):
        raise CliIntegrityError("tokenizer identity hash does not match seal")
    if tokenizer.get("revision") != PINNED_TOKENIZER_REVISION:
        raise CliIntegrityError("tokenizer revision is not pinned")
    if tokenizer_identity.get("revision") != tokenizer.get("revision"):
        raise CliIntegrityError(
            "tokenizer identity revision differs from seal"
        )
    tokenizer_files_expected = _string_mapping(tokenizer, "files_sha256")
    tokenizer_files = _verify_tokenizer_files(
        tokenizer_path, tokenizer_identity, tokenizer_files_expected
    )
    if tokenizer.get(
        "chat_template_sha256"
    ) != _chat_template_hash_from_config(tokenizer_path):
        raise CliIntegrityError(
            "tokenizer chat template hash does not match seal"
        )
    if tokenizer.get("chat_template_sha256") != tokenizer_identity.get(
        "chat_template_sha256"
    ):
        raise CliIntegrityError(
            "tokenizer chat template hash differs from identity"
        )
    model = _mapping(value, "model")
    if model.get("model_id") != PINNED_MODEL_ID:
        raise CliIntegrityError("model ID is not pinned")
    if model.get("revision") != PINNED_MODEL_REVISION:
        raise CliIntegrityError("model revision is not pinned")
    if model_path.resolve().name != PINNED_MODEL_REVISION:
        raise CliIntegrityError(
            "model path is not the pinned snapshot revision"
        )
    if model.get("identity_sha256") != _sha256_file(model_identity_path):
        raise CliIntegrityError("model identity hash does not match seal")
    config, model_files = _verify_model_files(
        model_path, model_identity_path, expected=model
    )
    _verify_architecture(config)
    if config.get("_name_or_path") not in (None, PINNED_MODEL_ID):
        raise CliIntegrityError("model config identity is not pinned")
    architecture = _mapping(model, "architecture")
    if architecture != EXPECTED_ARCHITECTURE:
        raise CliIntegrityError("model architecture in seal is not exact")
    if model_files.get("config.json") != model.get("config_sha256"):
        raise CliIntegrityError("model config hash differs from seal")
    if model_files.get("model.safetensors.index.json") != model.get(
        "weight_index_sha256"
    ):
        raise CliIntegrityError("model weight index hash differs from seal")
    weight_expected = _string_mapping(model, "weight_files_sha256")
    observed_weights = {
        name: digest
        for name, digest in model_files.items()
        if name not in REQUIRED_MODEL_FILES
    }
    if observed_weights != weight_expected:
        raise CliIntegrityError("model weight hashes differ from seal")
    _verify_versions(value)
    _verify_runtime_contract(value)
    _verify_protocol_contract(value)
    if tokenizer_files != tokenizer_files_expected:
        raise CliIntegrityError("tokenizer files differ from seal")
    return value, actual_sha256


def _verify_model_files(
    model_path: Path,
    model_identity_path: Path,
    *,
    expected: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, str]]:
    identity = _read_hash_file(model_identity_path)
    if not REQUIRED_MODEL_FILES.issubset(identity):
        raise CliIntegrityError("model identity is missing required files")
    observed: dict[str, str] = {}
    for name, expected_hash in identity.items():
        observed_hash = _sha256_file(model_path / name)
        if observed_hash != expected_hash:
            raise CliIntegrityError(f"model file hash mismatch: {name}")
        observed[name] = observed_hash
    config_raw = _read_bytes(model_path / "config.json")
    config = _parse_json_object(config_raw, "model config")
    if expected is not None:
        expected_hashes = _string_mapping(expected, "weight_files_sha256")
        for name, digest in expected_hashes.items():
            if observed.get(name) != digest:
                raise CliIntegrityError(f"model weight hash mismatch: {name}")
    index = _parse_json_object(
        _read_bytes(model_path / "model.safetensors.index.json"),
        "model weight index",
    )
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise CliIntegrityError("model weight index has no weight_map")
    referenced = {str(value) for value in weight_map.values()}
    expected_weight_names = set(observed) - set(REQUIRED_MODEL_FILES)
    if referenced != expected_weight_names:
        raise CliIntegrityError(
            "model weight index does not cover pinned shards"
        )
    return config, observed


def _verify_tokenizer_files(
    tokenizer_path: Path,
    identity: Mapping[str, object],
    expected: Mapping[str, str] | None,
) -> dict[str, str]:
    identity_files = _string_mapping(identity, "files_sha256")
    required = {
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if set(identity_files) != required:
        raise CliIntegrityError("tokenizer identity files are not exact")
    if expected is not None and dict(expected) != identity_files:
        raise CliIntegrityError("tokenizer identity hashes differ from seal")
    observed = {
        name: _sha256_file(tokenizer_path / name) for name in identity_files
    }
    if observed != identity_files:
        raise CliIntegrityError("tokenizer file hash mismatch")
    return observed


def _chat_template_hash_from_config(tokenizer_path: Path) -> str:
    config = _read_json_object(tokenizer_path / "tokenizer_config.json")
    template = config.get("chat_template")
    if not isinstance(template, str):
        raise CliIntegrityError("tokenizer chat template is not a string")
    return _sha256_text(template)


def _verify_architecture(config: Mapping[str, object]) -> None:
    observed = {
        "model_type": config.get("model_type"),
        "num_hidden_layers": config.get("num_hidden_layers"),
        "num_attention_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get(
            "num_key_value_heads", config.get("num_attention_heads")
        ),
        "num_key_value_groups": _integer(config.get("num_attention_heads"))
        // _integer(
            config.get(
                "num_key_value_heads", config.get("num_attention_heads")
            )
        ),
    }
    if observed != EXPECTED_ARCHITECTURE:
        raise CliIntegrityError(
            "model must be exactly Qwen2 with 28 layers, 28 query heads, "
            "four KV heads, and seven GQA groups"
        )


def _verify_versions(seal: Mapping[str, object]) -> None:
    expected = _string_mapping(seal, "versions")
    if set(expected) != set(PINNED_PACKAGES) | {"python"}:
        raise CliIntegrityError("execution seal version pins are incomplete")
    observed = _runtime_versions()
    if observed != expected:
        raise CliIntegrityError("runtime package versions differ from seal")


def _verify_runtime_contract(seal: Mapping[str, object]) -> None:
    runtime = _mapping(seal, "runtime")
    if runtime.get("device_type") != "cuda":
        raise CliIntegrityError("runtime device type must be CUDA")
    if runtime.get("dtype") != "bfloat16":
        raise CliIntegrityError("runtime dtype must be BF16")
    if runtime.get("seed") != retrieval_discovery.SEED:
        raise CliIntegrityError("runtime seed differs from frozen seed")
    if runtime.get("attention_backends") != ["eager", "sdpa"]:
        raise CliIntegrityError("runtime attention backends are not pinned")
    if "device_name" not in runtime or "total_memory_bytes" not in runtime:
        raise CliIntegrityError("runtime device identity is incomplete")


def _verify_protocol_contract(seal: Mapping[str, object]) -> None:
    if seal.get("seed") != retrieval_discovery.SEED:
        raise CliIntegrityError("seal seed differs from frozen seed")
    greedy = _mapping(seal, "greedy")
    if greedy != {"do_sample": False, "max_new_tokens": 50}:
        raise CliIntegrityError("greedy generation contract is not pinned")
    if seal.get("attention_backends") != ["eager", "sdpa"]:
        raise CliIntegrityError("seal attention backends are not pinned")


def _verify_loaded_tokenizer(
    tokenizer: Any, seal: Mapping[str, object]
) -> None:
    expected = _mapping(seal, "tokenizer").get("chat_template_sha256")
    observed = getattr(tokenizer, "chat_template", None)
    if not isinstance(observed, str) or _sha256_text(observed) != expected:
        raise CliIntegrityError(
            "loaded tokenizer chat template differs from seal"
        )


def _verify_loaded_model(model: Any, seal: Mapping[str, object]) -> None:
    try:
        device = next(model.parameters()).device
        dtypes = {parameter.dtype for parameter in model.parameters()}
    except (AttributeError, StopIteration) as error:
        raise CliIntegrityError("loaded model has no parameters") from error
    import torch

    if device.type != "cuda" or dtypes != {torch.bfloat16}:
        raise CliIntegrityError("loaded model is not CUDA BF16")
    retrieval_discovery.verify_model_gqa_mapping(model)
    runtime = _mapping(seal, "runtime")
    expected_name = runtime.get("device_name")
    if (
        isinstance(expected_name, str)
        and torch.cuda.get_device_name(0) != expected_name
    ):
        raise CliIntegrityError(
            "loaded model device differs from execution seal"
        )


def _require_cuda_bfloat16(seal: Mapping[str, object]) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise CliIntegrityError(
            "CUDA is required for the production discovery assay"
        )
    if not torch.cuda.is_bf16_supported():
        raise CliIntegrityError("CUDA BF16 is not supported")
    runtime = _mapping(seal, "runtime")
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    expected_name = runtime.get("device_name")
    if isinstance(expected_name, str) and device_name != expected_name:
        raise CliIntegrityError(
            "CUDA device name differs from execution seal"
        )
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    expected_memory = runtime.get("total_memory_bytes")
    if (
        isinstance(expected_memory, int)
        and expected_memory > 0
        and total_memory < expected_memory
    ):
        raise CliIntegrityError("CUDA device memory is below execution seal")
    return {
        "device_type": "cuda",
        "dtype": "bfloat16",
        "device_name": device_name,
        "total_memory_bytes": total_memory,
        "torch_version": _runtime_versions()["torch"],
        "transformers_version": _runtime_versions()["transformers"],
        "seed": retrieval_discovery.SEED,
        "attention_backends": ["eager", "sdpa"],
    }


def _load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return cast(
        Any,
        AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            str(path), local_files_only=True
        ),
    )


def _set_offline_environment() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def _load_model(path: Path) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    _set_offline_environment()
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(path),
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.to(torch.device("cuda"))
    model.eval()
    return model


def _preflight_payload(
    seal: Mapping[str, object],
    seal_sha256: str,
    args: argparse.Namespace,
    source_manifest: Mapping[str, object],
    cases: Sequence[retrieval_discovery.RetrievalCase],
    tokenizer: Any,
) -> dict[str, object]:
    assert args.model is not None
    assert args.tokenizer is not None
    return {
        "schema_version": retrieval_discovery.RESULT_SCHEMA_VERSION,
        "status": "passed",
        "mode": "cpu_preflight" if args.cpu_preflight else "production",
        "repository": retrieval_discovery.REPOSITORY,
        "commit": retrieval_discovery.PINNED_COMMIT,
        "execution_seal_sha256": seal_sha256,
        "source_manifest_sha256": _sha256_file(
            args.source_root.resolve().parent / "manifest.json"
        ),
        "module_sha256": FROZEN_MODULE_SHA256,
        "cli_sha256": _sha256_file(Path(__file__).resolve()),
        "model_path": str(args.model.resolve()),
        "tokenizer_path": str(args.tokenizer.resolve()),
        "seed": retrieval_discovery.SEED,
        "case_count": len(cases),
        "cases": [case.to_dict() for case in cases],
        "source_manifest": dict(source_manifest),
        "architecture": EXPECTED_ARCHITECTURE,
        "tokenizer_chat_template_sha256": _sha256_text(
            cast(str, tokenizer.chat_template)
        ),
        "runtime_contract": dict(_mapping(seal, "runtime")),
    }


def _write_result_chain(
    run_dir: Path,
    output: Path,
    result: Mapping[str, object],
    seal_sha256: str,
    preflight_sha256: str,
    context: dict[str, str | None],
) -> dict[str, object]:
    selected_cells = result.get("selected_cells", [])
    selected_payload = {
        "schema_version": SELECTED_CELLS_SCHEMA_VERSION,
        "execution_seal_sha256": seal_sha256,
        "preflight_sha256": preflight_sha256,
        "selected_cells": selected_cells,
    }
    selected_path = run_dir / "selected-cells.json"
    _write_json_exclusive(selected_path, selected_payload)
    selected_sha256 = _sha256_file(selected_path)
    context["selected_cells_sha256"] = selected_sha256
    final_result = dict(result)
    final_result["integrity"] = {
        "execution_seal_sha256": seal_sha256,
        "preflight_sha256": preflight_sha256,
        "selected_cells_sha256": selected_sha256,
    }
    _write_json_exclusive(output, final_result)
    result_sha256 = _sha256_file(output)
    context["result_sha256"] = result_sha256
    integrity_payload = {
        "schema_version": "herald_v3.retrieval_discovery_integrity.v1",
        "execution_seal_sha256": seal_sha256,
        "preflight_sha256": preflight_sha256,
        "result_sha256": result_sha256,
        "selected_cells_sha256": selected_sha256,
    }
    _write_json_exclusive(
        run_dir / "discovery-integrity.json", integrity_payload
    )
    return final_result


def _preserve_failure(
    output: Path,
    run_dir: Path,
    context: Mapping[str, str | None],
    error: Exception,
) -> dict[str, object]:
    selected_path = run_dir / "selected-cells.json"
    selected_sha256 = context.get("selected_cells_sha256")
    if not selected_path.exists():
        selected_payload: dict[str, object] = {
            "schema_version": SELECTED_CELLS_SCHEMA_VERSION,
            "execution_seal_sha256": context.get("execution_seal_sha256"),
            "preflight_sha256": context.get("preflight_sha256"),
            "selected_cells": [],
            "status": "failed",
        }
        _write_json_exclusive(selected_path, selected_payload)
        selected_sha256 = _sha256_file(selected_path)
    failure = _failure_payload(error, "execution")
    integrity = {
        "execution_seal_sha256": context.get("execution_seal_sha256"),
        "preflight_sha256": context.get("preflight_sha256"),
        "selected_cells_sha256": selected_sha256,
    }
    failure["integrity"] = integrity
    _write_json_exclusive(output, failure)
    result_sha256 = _sha256_file(output)
    _write_json_exclusive(
        run_dir / "discovery-integrity.json",
        {
            "schema_version": "herald_v3.retrieval_discovery_integrity.v1",
            **integrity,
            "result_sha256": result_sha256,
        },
    )
    return failure


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
    preflight = run_dir / "discovery-preflight.json"
    if preflight.exists():
        raise CliIntegrityError(f"refusing existing preflight: {preflight}")
    return run_dir


def _write_json_exclusive(path: Path, value: object) -> None:
    raw = (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _failure_payload(error: Exception, stage: str) -> dict[str, object]:
    return {
        "schema_version": retrieval_discovery.RESULT_SCHEMA_VERSION,
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


def _read_json_object(path: Path) -> dict[str, object]:
    return _parse_json_object(_read_bytes(path), str(path))


def _parse_json_object(raw: bytes, description: str) -> dict[str, object]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CliIntegrityError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise CliIntegrityError(f"{description} must be an object")
    return cast(dict[str, object], value)


def _read_hash_file(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in _read_bytes(path).decode("utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2 or len(fields[0]) != 64:
            raise CliIntegrityError(f"malformed model identity line: {line}")
        hashes[fields[1]] = fields[0]
    return hashes


def _mapping(value: Mapping[str, object], name: str) -> Mapping[str, object]:
    nested = value.get(name)
    if not isinstance(nested, Mapping):
        raise CliIntegrityError(f"seal field is not an object: {name}")
    return nested


def _string_mapping(value: Mapping[str, object], name: str) -> dict[str, str]:
    nested = _mapping(value, name)
    result: dict[str, str] = {}
    for key, item in nested.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise CliIntegrityError(f"seal field is not a string map: {name}")
        result[key] = item
    return result


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CliIntegrityError("model architecture field is not an integer")
    return value


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in PINNED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise CliIntegrityError(
                f"required package is missing: {package}"
            ) from error
    return versions


def _runtime_contract() -> dict[str, object]:
    import torch

    name: str | None = None
    total_memory = 0
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        total_memory = int(
            torch.cuda.get_device_properties(index).total_memory
        )
    return {
        "device_type": "cuda",
        "dtype": "bfloat16",
        "device_name": name,
        "total_memory_bytes": total_memory,
        "seed": retrieval_discovery.SEED,
        "attention_backends": ["eager", "sdpa"],
    }


def _assert_frozen_module() -> None:
    observed = _sha256_file(
        PROJECT_ROOT / "src/herald_v3/retrieval_discovery.py"
    )
    if observed != FROZEN_MODULE_SHA256:
        raise CliIntegrityError("frozen discovery module has changed")


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256(_read_bytes(path))


def _sha256_text(value: str) -> str:
    return _sha256(value.encode("utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
