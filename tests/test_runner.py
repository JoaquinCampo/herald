"""End-to-end sweep pipeline on the tiny model: generate, score, store,
resume. Validates references + hybrids + scoring + storage wiring and
idempotent resume (re-running adds no work).
"""

from pathlib import Path
from typing import cast

import pytest
import torch

import herald.generate as G
import herald.runner as R
from herald import storage
from herald.config import Config, TaskSpec
from herald.generate import LoadedModel, load_model
from herald.tasks import PromptRecord

pytestmark = pytest.mark.model

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
LONG = " ".join(["alpha beta gamma delta epsilon zeta"] * 16)


@pytest.fixture(scope="module")
def lm() -> LoadedModel:
    return load_model(
        "llama",
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
        model_id=TINY,
    )


@pytest.fixture(autouse=True)
def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    def build(lm: LoadedModel, record: PromptRecord) -> torch.Tensor:
        text = record.messages[-1]["content"]
        ids = lm.tokenizer(text, return_tensors="pt").input_ids[0]
        return cast(torch.Tensor, ids)

    monkeypatch.setattr(G, "build_input_ids", build)
    # Small generation budget so the tiny model test is fast.
    monkeypatch.setattr(
        R,
        "TASKS",
        {
            "gsm8k": TaskSpec(
                name="gsm8k",
                hf_path="openai/gsm8k",
                hf_subset="main",
                split="test",
                max_new_tokens=24,
            )
        },
    )


def _records() -> list[PromptRecord]:
    return [
        PromptRecord(
            task="gsm8k",
            prompt_id=f"gsm8k-{i}",
            messages=[{"role": "user", "content": LONG + f" item {i}"}],
            gold={"answer": str(i)},
        )
        for i in range(3)
    ]


def _count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0


def test_pipeline_and_resume(lm: LoadedModel, tmp_path: Path) -> None:
    recs = _records()
    config = Config(
        models=["llama"],
        tasks=["gsm8k"],
        compressors=["streaming_llm", "random"],
        ratios=[0.5],
        switch_stride=16,
        prompts_per_task=3,
        ref_batch_size=2,
        hybrid_batch_size=1,
        results_dir=tmp_path,
    )

    R._run_references(lm, "gsm8k", recs, config)
    done = storage.reference_done(tmp_path, "llama", "gsm8k")
    assert done == {"gsm8k-0", "gsm8k-1", "gsm8k-2"}

    R._run_hybrids(lm, "gsm8k", recs, config)
    shard = (
        tmp_path
        / "llama"
        / "gsm8k"
        / "hybrids"
        / "streaming_llm__0.5000.jsonl"
    )
    n_stream = _count_lines(shard)
    assert n_stream > 0
    done_pairs = storage.hybrid_done(
        tmp_path, "llama", "gsm8k", "streaming_llm", 0.5
    )
    assert done_pairs == set(
        storage.hybrid_done(tmp_path, "llama", "gsm8k", "streaming_llm", 0.5)
    )
    # Every (prompt, s) with s < run_length should be present.
    assert all(s in (0, 16) for _, s in done_pairs)

    # Resume: re-run adds nothing (idempotent skip).
    R._run_references(lm, "gsm8k", recs, config)
    R._run_hybrids(lm, "gsm8k", recs, config)
    assert _count_lines(shard) == n_stream
