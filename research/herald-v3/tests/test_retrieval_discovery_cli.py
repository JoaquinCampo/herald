import hashlib
import json
import runpy
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "data" / "retrieval-discovery-v1" / "source"
CLI_PATH = ROOT / "scripts" / "discover_retrieval_heads.py"
CLI: dict[str, Any] = runpy.run_path(str(CLI_PATH), run_name="cli_test")
CLI_GLOBALS: dict[str, Any] = cast(dict[str, Any], CLI["main"].__globals__)


class CharacterTokenizer:
    chat_template = "fixture chat template"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_: object,
    ) -> dict[str, object]:
        del add_special_tokens
        result: dict[str, object] = {
            "input_ids": [ord(character) + 1 for character in text]
        }
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_tensors: str | None = None,
    ) -> object:
        rendered = f"<user>\n{messages[0]['content']}\n</user>"
        if add_generation_prompt:
            rendered += "\n<assistant>\n"
        if not tokenize:
            return rendered
        encoded = self(rendered)
        ids = cast(list[int], encoded["input_ids"])
        if return_tensors == "pt":
            import torch

            return {"input_ids": torch.tensor([ids])}
        return {"input_ids": ids}

    def decode(
        self,
        token_ids: list[int] | tuple[int, ...],
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr((token_id - 1) % 128) for token_id in token_ids)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    revision = CLI["PINNED_MODEL_REVISION"]
    model = tmp_path / revision
    tokenizer = tmp_path / "tokenizer"
    model.mkdir()
    tokenizer.mkdir()
    template = CharacterTokenizer.chat_template
    tokenizer_files = {
        "merges.txt": b"fixture merges\n",
        "tokenizer.json": b'{"fixture":true}\n',
        "tokenizer_config.json": json.dumps(
            {"chat_template": template}, sort_keys=True
        ).encode("utf-8")
        + b"\n",
        "vocab.json": b'{"<fixture>":0}\n',
    }
    for name, raw in tokenizer_files.items():
        (tokenizer / name).write_bytes(raw)
    tokenizer_identity = tmp_path / "tokenizer-identity.json"
    _write_json(
        tokenizer_identity,
        {
            "revision": revision,
            "chat_template_sha256": _sha256(template.encode("utf-8")),
            "files_sha256": {
                name: _sha256(raw) for name, raw in tokenizer_files.items()
            },
        },
    )

    config = {
        "_name_or_path": CLI["PINNED_MODEL_ID"],
        "model_type": "qwen2",
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
    }
    _write_json(model / "config.json", config)
    shard_names = [
        f"model-0000{index}-of-00004.safetensors" for index in range(1, 5)
    ]
    for index, name in enumerate(shard_names):
        (model / name).write_bytes(f"fixture shard {index}\n".encode())
    _write_json(
        model / "model.safetensors.index.json",
        {
            "weight_map": {
                f"parameter_{index}": name
                for index, name in enumerate(shard_names)
            }
        },
    )
    model_identity = tmp_path / "model-identity.sha256"
    model_names = [
        "config.json",
        "model.safetensors.index.json",
        *shard_names,
    ]
    model_identity.write_text(
        "".join(
            f"{_sha256((model / name).read_bytes())}  {name}\n"
            for name in model_names
        ),
        encoding="utf-8",
    )
    seal = tmp_path / "execution-seal.json"
    main = cast(Callable[[Sequence[str] | None], int], CLI["main"])
    assert (
        main(
            [
                "--prepare-seal",
                str(seal),
                "--model",
                str(model),
                "--tokenizer",
                str(tokenizer),
                "--source-root",
                str(SOURCE_ROOT),
                "--tokenizer-identity",
                str(tokenizer_identity),
                "--model-identity",
                str(model_identity),
            ]
        )
        == 0
    )
    return model, tokenizer, tokenizer_identity, model_identity, seal


def _run_args(
    model: Path,
    tokenizer: Path,
    seal: Path,
    output: Path,
    tokenizer_identity: Path,
    model_identity: Path,
    *extra: str,
) -> list[str]:
    digest = CLI["_sha256_file"](seal)
    return [
        "--model",
        str(model),
        "--tokenizer",
        str(tokenizer),
        "--source-root",
        str(SOURCE_ROOT),
        "--output",
        str(output),
        "--execution-seal",
        str(seal),
        "--execution-seal-sha256",
        digest,
        "--tokenizer-identity",
        str(tokenizer_identity),
        "--model-identity",
        str(model_identity),
        *extra,
    ]


@pytest.fixture(autouse=True)
def fixture_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        CLI_GLOBALS, "_load_tokenizer", lambda path: CharacterTokenizer()
    )


def test_cpu_preflight_verifies_seal_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, tokenizer, tokenizer_identity, model_identity, seal = _fixture(
        tmp_path
    )
    monkeypatch.setitem(
        CLI_GLOBALS,
        "_load_model",
        lambda path: pytest.fail("CPU preflight loaded model weights"),
    )
    output = tmp_path / "run-cpu" / "result.json"
    main = cast(Callable[[Sequence[str] | None], int], CLI["main"])

    assert (
        main(
            _run_args(
                model,
                tokenizer,
                seal,
                output,
                tokenizer_identity,
                model_identity,
                "--cpu-preflight",
            )
        )
        == 0
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    preflight = json.loads(
        (output.parent / "discovery-preflight.json").read_text(
            encoding="utf-8"
        )
    )
    integrity = json.loads(
        (output.parent / "discovery-integrity.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["status"] == "preflight_passed"
    assert preflight["case_count"] == 24
    assert result["integrity"]["execution_seal_sha256"] == CLI[
        "_sha256_file"
    ](seal)
    assert integrity["result_sha256"] == CLI["_sha256_file"](output)
    assert "--allow-cpu" not in CLI["_argument_parser"]().format_help()


def test_tampered_seal_is_preserved_as_failure(
    tmp_path: Path,
) -> None:
    model, tokenizer, tokenizer_identity, model_identity, seal = _fixture(
        tmp_path
    )
    tampered = tmp_path / "tampered-seal.json"
    value = json.loads(seal.read_text(encoding="utf-8"))
    value["commit"] = "0" * 40
    _write_json(tampered, value)
    output = tmp_path / "run-tampered" / "result.json"
    main = cast(Callable[[Sequence[str] | None], int], CLI["main"])

    assert (
        main(
            _run_args(
                model,
                tokenizer,
                tampered,
                output,
                tokenizer_identity,
                model_identity,
                "--cpu-preflight",
            )
        )
        == 1
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["error_type"] == "CliIntegrityError"
    assert (output.parent / "selected-cells.json").exists()
    assert (output.parent / "discovery-integrity.json").exists()


def test_existing_run_directory_is_refused_without_overwrite(
    tmp_path: Path,
) -> None:
    model, tokenizer, tokenizer_identity, model_identity, seal = _fixture(
        tmp_path
    )
    output = tmp_path / "run-once" / "result.json"
    main = cast(Callable[[Sequence[str] | None], int], CLI["main"])
    args = _run_args(
        model,
        tokenizer,
        seal,
        output,
        tokenizer_identity,
        model_identity,
        "--cpu-preflight",
    )
    assert main(args) == 0
    original = output.read_bytes()
    assert main(args) == 1
    assert output.read_bytes() == original


def test_failure_after_seal_verification_is_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, tokenizer, tokenizer_identity, model_identity, seal = _fixture(
        tmp_path
    )

    def fail_loader(path: Path) -> Any:
        del path
        raise RuntimeError("faithful fixture loader failure")

    monkeypatch.setitem(CLI_GLOBALS, "_load_tokenizer", fail_loader)
    output = tmp_path / "run-failure" / "result.json"
    main = cast(Callable[[Sequence[str] | None], int], CLI["main"])
    args = _run_args(
        model,
        tokenizer,
        seal,
        output,
        tokenizer_identity,
        model_identity,
        "--cpu-preflight",
    )

    assert main(args) == 1
    original = output.read_bytes()
    result = json.loads(original)
    assert result["status"] == "failed"
    assert result["error"] == "faithful fixture loader failure"
    assert main(args) == 1
    assert output.read_bytes() == original


def test_wrong_architecture_is_rejected() -> None:
    verify = cast(
        Callable[[dict[str, object]], None], CLI["_verify_architecture"]
    )
    with pytest.raises(Exception, match="exactly Qwen2"):
        verify(
            {
                "model_type": "qwen2",
                "num_hidden_layers": 27,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
            }
        )
