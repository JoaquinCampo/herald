"""Resumable pilot collection tests over the real tiny engine pipeline."""

# The pilot entry point is an executable script, not an installed module.
# ruff: noqa: E402, I001

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_pilot as pilot
from tests.test_engine import _input_ids, _tiny_model
from herald_v3.engineering import runner
from herald_v3.engineering.prompts import (
    EngineeringPrompt,
    PromptManifest,
    write_prompt_manifest,
)


class _TinyTokenizer:
    eos_token_id: int | None = None
    name_or_path = "fixture-tokenizer"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
    ) -> object:
        del add_generation_prompt, return_tensors
        rendered = (
            "<|im_start|>user\n"
            + messages[0]["content"]
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        return _input_ids().clone() if tokenize else rendered

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_tensors: str
    ) -> object:
        del text, add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=_input_ids().clone())

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        del token_ids, kwargs
        return "fixture response"


class _Score:
    def to_dict(self) -> dict[str, object]:
        return {
            "loose": 1.0,
            "strict": 1.0,
            "loose_pass": [True],
            "strict_pass": [True],
            "instruction_count": 1,
        }


class _PairScore:
    def to_dict(self) -> dict[str, object]:
        return {
            "reference": _Score().to_dict(),
            "action": _Score().to_dict(),
            "d_loose": 0.0,
            "d_strict": 0.0,
        }


def _manifest(count: int) -> PromptManifest:
    prompts: list[EngineeringPrompt] = []
    for index in range(count):
        user_prompt = f"fixture {index}"
        prompt_text = (
            f"<|im_start|>user\n{user_prompt}"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        prompts.append(
            EngineeringPrompt(
                prompt_id=f"ifeval_{100 + index}",
                key=100 + index,
                fold=index,
                fold_hash=f"fold-{index}",
                split_hash=f"split-{index}",
                prompt_text=prompt_text,
                user_prompt=user_prompt,
                messages=({"role": "user", "content": user_prompt},),
                instruction_id_list=("punctuation:no_comma",),
                kwargs=({},),
                prompt_text_sha256=pilot._sha256_text(prompt_text),
                user_prompt_sha256=pilot._sha256_text(user_prompt),
            )
        )
    return PromptManifest(
        source_path="fixture-source",
        source_sha256="source-sha",
        source_bytes=1,
        official_dataset_path="fixture-arrow",
        official_dataset_sha256="arrow-sha",
        official_dataset_bytes=1,
        official_dataset_rows=count,
        selection_rule="fixture manifest order",
        prompts=tuple(prompts),
    )


def _patch_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pilot, "check_ifeval_resources", lambda: {"fixture": True}
    )
    monkeypatch.setattr(
        runner, "check_ifeval_resources", lambda: {"fixture": True}
    )
    monkeypatch.setattr(
        runner, "score_ifeval_gold", lambda *args, **kwargs: _Score()
    )
    monkeypatch.setattr(
        runner, "score_pair", lambda *args, **kwargs: _PairScore()
    )


def _write_manifest(tmp_path: Path, manifest: PromptManifest) -> Path:
    path = tmp_path / "frozen-prompts.json"
    write_prompt_manifest(manifest, path)
    return path


def test_real_tiny_pipeline_stops_at_target_and_resumes_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_scoring(monkeypatch)
    manifest = _manifest(2)
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "pilot"
    model = _tiny_model()
    tokenizer = _TinyTokenizer()

    first = pilot.run_pilot(
        model,
        tokenizer,
        manifest,
        manifest_path,
        output,
        model_path="fixture-model",
        target_eligible=1,
        max_new_tokens=34,
        seed=7,
    )

    assert first["status"] == "completed"
    assert first["accepted_eligible"] == 1
    assert (output / "ifeval_100" / "artifacts.json").exists()
    assert not (output / "ifeval_101").exists()

    def unexpected_run(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        raise AssertionError("verified prompt was rerun")

    monkeypatch.setattr(runner, "run_engineering", unexpected_run)
    resumed = pilot.run_pilot(
        model,
        tokenizer,
        manifest,
        manifest_path,
        output,
        model_path="fixture-model",
        target_eligible=1,
        max_new_tokens=34,
        seed=7,
    )
    assert resumed["status"] == "completed"
    assert resumed["accepted_eligible"] == 1


def test_resume_rejects_corrupt_artifact_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_scoring(monkeypatch)
    manifest = _manifest(1)
    manifest_path = _write_manifest(tmp_path, manifest)
    output = tmp_path / "pilot"
    model = _tiny_model()
    tokenizer = _TinyTokenizer()
    pilot.run_pilot(
        model,
        tokenizer,
        manifest,
        manifest_path,
        output,
        model_path="fixture-model",
        target_eligible=1,
        max_new_tokens=34,
        seed=7,
    )
    run_log = output / "ifeval_100" / "run.log"
    run_log.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(pilot.PilotStateError, match=str(run_log.parent)):
        pilot.run_pilot(
            model,
            tokenizer,
            manifest,
            manifest_path,
            output,
            model_path="fixture-model",
            target_eligible=1,
            max_new_tokens=34,
            seed=7,
        )

    assert run_log.read_text(encoding="utf-8") == "corrupt\n"


def test_exhaustion_records_early_eos_without_damage_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_scoring(monkeypatch)
    manifest = _manifest(2)
    manifest_path = _write_manifest(tmp_path, manifest)
    model = _tiny_model()
    with torch.no_grad():
        first_token = int(
            model(input_ids=_input_ids()).logits[:, -1, :].argmax().item()
        )
    model.generation_config.eos_token_id = first_token

    summary = pilot.run_pilot(
        model,
        _TinyTokenizer(),
        manifest,
        manifest_path,
        tmp_path / "pilot",
        model_path="fixture-model",
        target_eligible=1,
        max_new_tokens=34,
        seed=7,
    )

    assert summary["status"] == "insufficient_coverage"
    assert summary["accepted_eligible"] == 0
    assert summary["ineligible_early_eos"] == 2
    ledger = summary["ledger"]
    assert isinstance(ledger, list)
    assert all(item["status"] == "ineligible_early_eos" for item in ledger)
    for prompt_id in ("ifeval_100", "ifeval_101"):
        run = json.loads(
            (tmp_path / "pilot" / prompt_id / "run.json").read_text()
        )
        row = run["results"][0]
        assert row["status"] == "ineligible"
        assert "scores" not in row


def test_cli_rejects_no_cuda_before_model_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    load_called = False

    class _Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    def unexpected_load(path: str) -> tuple[Any, Any]:
        del path
        nonlocal load_called
        load_called = True
        raise AssertionError("model load must not happen")

    monkeypatch.setattr(runner, "_torch", lambda: SimpleNamespace(cuda=_Cuda))
    monkeypatch.setattr(runner, "load_offline_model", unexpected_load)
    with pytest.raises(SystemExit, match="2"):
        pilot.main(
            [
                "--model",
                "fixture-model",
                "--prompts",
                str(tmp_path / "unused.json"),
                "--output",
                str(tmp_path / "pilot"),
                "--target-eligible",
                "1",
                "--max-new-tokens",
                "40",
            ]
        )

    assert not load_called
