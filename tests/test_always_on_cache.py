# pyright: reportMissingImports=false, reportOperatorIssue=false

from typing import cast

import pytest
import torch

import herald.generate as G
from herald.generate import (
    LoadedModel,
    generate_always_on_press,
    generate_always_on_streaming,
    generate_always_on_sustained,
    generate_baseline,
    load_model,
)
from herald.presses import get_press
from herald.tasks import PromptRecord

pytestmark = pytest.mark.model
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def lm() -> LoadedModel:
    return load_model(
        "llama",
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
        model_id=TINY,
    )


def test_always_on_streaming_reduces_real_retained_peak(
    lm: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = PromptRecord(
        task="ifeval",
        prompt_id="always-on",
        messages=[{"role": "user", "content": "alpha beta gamma " * 96}],
        gold={"instruction_id_list": [], "kwargs": []},
    )

    def direct_input_ids(
        loaded: LoadedModel, prompt: PromptRecord
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            loaded.tokenizer(
                prompt.messages[-1]["content"], return_tensors="pt"
            ).input_ids[0],
        )

    monkeypatch.setattr(G, "build_input_ids", direct_input_ids)
    baseline = generate_baseline(lm, record, 32)
    candidate = generate_always_on_streaming(
        lm,
        record,
        32,
        ratio=0.5,
        sustain_interval=4,
    )

    assert candidate.gen_ids
    assert candidate.peak_kv_cache_bytes > 0
    assert candidate.peak_kv_cache_bytes < baseline.peak_kv_cache_bytes


def test_generic_always_on_snapkv_runs_real_generation(
    lm: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = PromptRecord(
        task="ifeval",
        prompt_id="snapkv-always-on",
        messages=[{"role": "user", "content": "alpha beta gamma " * 96}],
        gold={"instruction_id_list": [], "kwargs": []},
    )

    def direct_input_ids(
        loaded: LoadedModel, prompt: PromptRecord
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            loaded.tokenizer(
                prompt.messages[-1]["content"], return_tensors="pt"
            ).input_ids[0],
        )

    monkeypatch.setattr(G, "build_input_ids", direct_input_ids)
    candidate = generate_always_on_press(
        lm,
        record,
        16,
        press=get_press("snapkv", 0.25),
    )

    assert candidate.gen_ids
    assert candidate.peak_kv_cache_bytes > 0


def test_generic_always_on_sustained_press_runs_real_generation(
    lm: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = PromptRecord(
        task="ifeval",
        prompt_id="knorm-sustained",
        messages=[{"role": "user", "content": "alpha beta gamma " * 96}],
        gold={"instruction_id_list": [], "kwargs": []},
    )

    def direct_input_ids(
        loaded: LoadedModel, prompt: PromptRecord
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            loaded.tokenizer(
                prompt.messages[-1]["content"], return_tensors="pt"
            ).input_ids[0],
        )

    monkeypatch.setattr(G, "build_input_ids", direct_input_ids)
    candidate = generate_always_on_sustained(
        lm,
        record,
        16,
        press=get_press("knorm", 0.25),
        ratio=0.25,
        sustain_interval=4,
    )

    assert candidate.gen_ids
    assert candidate.peak_kv_cache_bytes > 0
