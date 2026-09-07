"""Runner persistence and chat-template verification tests."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from herald_v3.engineering import runner
from herald_v3.engineering.prompts import EngineeringPrompt, PromptManifest


class _Tokenizer:
    eos_token_id = 99
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
        return torch.tensor([[1, 2]]) if tokenize else rendered

    def __call__(
        self, text: str, *, add_special_tokens: bool, return_tensors: str
    ) -> object:
        del text, add_special_tokens, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(
            _name_or_path="fixture-model",
            _commit_hash="fixture-revision",
            vocab_size=4,
            hidden_size=2,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            _attn_implementation="sdpa",
        )


def _manifest() -> PromptManifest:
    prompt_text = (
        "<|im_start|>user\nfixture<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt = EngineeringPrompt(
        prompt_id="ifeval_100",
        key=100,
        fold=0,
        fold_hash="fold",
        split_hash="split",
        prompt_text=prompt_text,
        user_prompt="fixture",
        messages=({"role": "user", "content": "fixture"},),
        instruction_id_list=("punctuation:no_comma",),
        kwargs=({},),
        prompt_text_sha256=hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest(),
        user_prompt_sha256=hashlib.sha256(b"fixture").hexdigest(),
    )
    return PromptManifest(
        source_path="fixture",
        source_sha256="source",
        source_bytes=1,
        official_dataset_path="fixture",
        official_dataset_sha256="arrow",
        official_dataset_bytes=1,
        official_dataset_rows=1,
        selection_rule="fixture",
        prompts=(prompt,),
    )


def test_tokenize_prompt_checks_chat_template_and_source_ids() -> None:
    ids = runner.tokenize_prompt(_Tokenizer(), _manifest().prompts[0])

    assert ids.tolist() == [[1, 2]]


def test_run_engineering_persists_manifest_logs_and_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_run_prompt(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        return {"status": "accepted", "acceptance": {"passed": True}}

    monkeypatch.setattr(runner, "_run_prompt", fake_run_prompt)
    evidence = runner.run_engineering(
        _Model(),
        _Tokenizer(),
        _manifest(),
        tmp_path / "evidence",
        max_new_tokens=40,
    )

    output = tmp_path / "evidence"
    assert evidence["status"] == "completed"
    assert (output / "run.json").exists()
    assert (output / "run.log").exists()
    assert (output / "prompt-manifest.json").exists()
    assert (output / "artifacts.json").exists()


def test_eos_ids_include_generation_config_values() -> None:
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=[100, 101])
    )
    tokenizer = SimpleNamespace(eos_token_id=99)

    assert runner._eos_ids(model, tokenizer) == frozenset({99, 100, 101})


def test_runner_stops_after_failure_and_returns_failed_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = _manifest().prompts[0]
    second = EngineeringPrompt(
        prompt_id="ifeval_101",
        key=101,
        fold=1,
        fold_hash="fold-1",
        split_hash="split-1",
        prompt_text=first.prompt_text,
        user_prompt=first.user_prompt,
        messages=first.messages,
        instruction_id_list=first.instruction_id_list,
        kwargs=first.kwargs,
        prompt_text_sha256=first.prompt_text_sha256,
        user_prompt_sha256=first.user_prompt_sha256,
    )
    manifest = PromptManifest(
        source_path="fixture",
        source_sha256="source",
        source_bytes=1,
        official_dataset_path="fixture",
        official_dataset_sha256="arrow",
        official_dataset_bytes=1,
        official_dataset_rows=2,
        selection_rule="fixture",
        prompts=(first, second),
    )
    calls = 0

    def fake_run_prompt(*args: object, **kwargs: object) -> dict[str, object]:
        del args, kwargs
        nonlocal calls
        calls += 1
        return {
            "status": "eligible_with_gate_failures",
            "acceptance": {"passed": False},
        }

    monkeypatch.setattr(runner, "_run_prompt", fake_run_prompt)
    evidence = runner.run_engineering(
        _Model(),
        _Tokenizer(),
        manifest,
        tmp_path / "failure",
        max_new_tokens=40,
    )

    assert calls == 1
    assert evidence["passed"] is False
    assert evidence["stopped_on_failure"] is True
    assert evidence["status"] == "completed_with_failures"
    results = evidence["results"]
    assert isinstance(results, list)
    assert len(results) == 1
    log = (tmp_path / "failure" / "run.log").read_text(encoding="utf-8")
    assert '"event": "finish"' in log


def test_early_eos_is_coverage_not_an_acceptance_failure() -> None:
    accepted: dict[str, object] = {"status": "accepted"}
    ineligible: dict[str, object] = {"status": "ineligible"}
    assert runner._overall_passed([accepted, ineligible])
    assert not runner._overall_passed([ineligible])
    assert not runner._overall_passed([])
    assert not runner._overall_passed([accepted, {"status": "failed"}])
    assert not runner._overall_passed(
        [accepted, {"status": "eligible_with_gate_failures"}]
    )
