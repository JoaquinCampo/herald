"""Correctness invariants for the generation mechanism.

Marked `model`: loads a tiny random Llama on CPU. The model's outputs
are gibberish (random weights); only the mechanism is under test. Real
chat templating and the score-based presses on real-length prompts are
validated on the GPU in Phase 1.
"""

from typing import cast

import numpy as np
import pytest
import torch

import herald.generate as G
from herald.features import FEATURE_NAMES
from herald.generate import (
    LoadedModel,
    generate_hybrids,
    generate_reference,
    load_model,
    switch_positions,
)
from herald.presses import get_press
from herald.tasks import PromptRecord

pytestmark = pytest.mark.model

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
M = 24
# Long enough that SnapKV's 64-token observation window is satisfied.
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
def _direct_tokenize(monkeypatch: pytest.MonkeyPatch) -> None:
    # The tiny tokenizer has no chat template; tokenize the last message
    # content directly. Real chat templating is exercised on real models.
    def build(lm: LoadedModel, record: PromptRecord) -> torch.Tensor:
        text = record.messages[-1]["content"]
        ids = lm.tokenizer(text, return_tensors="pt").input_ids[0]
        return cast(torch.Tensor, ids)

    monkeypatch.setattr(G, "build_input_ids", build)


def _rec(content: str, pid: str) -> PromptRecord:
    return PromptRecord(
        task="gsm8k",
        prompt_id=pid,
        messages=[{"role": "user", "content": content}],
        gold={"answer": "0"},
    )


def test_determinism(lm: LoadedModel) -> None:
    recs = [_rec(LONG, "p0"), _rec("short prompt here", "p1")]
    a = generate_reference(lm, recs, M)
    b = generate_reference(lm, recs, M)
    for x, y in zip(a, b, strict=True):
        assert x.gen_ids == y.gen_ids
        assert np.allclose(x.features, y.features, equal_nan=True)


def test_scores_equal_logits(lm: LoadedModel) -> None:
    ids = lm.tokenizer(LONG, return_tensors="pt").input_ids
    with torch.no_grad():
        out = lm.model.generate(  # type: ignore[operator]
            ids,
            generation_config=lm.gen_config,
            max_new_tokens=8,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=True,
        )
    assert all(
        torch.equal(sc, lg)
        for sc, lg in zip(out.scores, out.logits, strict=True)
    )


def test_feature_shape(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    assert ref.features.shape == (len(ref.gen_ids), len(FEATURE_NAMES))
    assert len(ref.gen_ids) > 0


@pytest.mark.parametrize(
    "press_name",
    ["streaming_llm", "knorm", "expected_attention", "random", "snapkv"],
)
def test_s0_equals_fully_compressed(lm: LoadedModel, press_name: str) -> None:
    # s=0 hybrid (prompt-only cache, compressed) must reproduce an
    # independent fully-compressed run from the same prompt.
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    press = get_press(press_name, 0.5)
    hyb = generate_hybrids(lm, [(ref, 0)], press_name, 0.5, press, M, seed=0)[
        0
    ]

    seq = torch.tensor(ref.prompt_input_ids, dtype=torch.long)
    press2 = get_press(press_name, 0.5)
    torch.manual_seed(0)
    with torch.no_grad(), press2(lm.model):
        out = lm.model.generate(  # type: ignore[operator]
            seq.unsqueeze(0),
            generation_config=lm.gen_config,
            max_new_tokens=M,
            return_dict_in_generate=True,
        )
    indep = out.sequences[0, len(ref.prompt_input_ids) :].tolist()
    trimmed: list[int] = []
    for tok in indep:
        trimmed.append(tok)
        if tok in lm.eos_ids:
            break
    assert hyb.new_ids == trimmed


def test_prefix_identity(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    positions = [s for s in switch_positions(len(ref.gen_ids), 16) if s > 0]
    assert positions, "run too short to test a positive switch position"
    s = positions[0]
    press = get_press("streaming_llm", 0.5)
    hyb = generate_hybrids(lm, [(ref, s)], "streaming_llm", 0.5, press, M)[0]
    full = ref.gen_ids[:s] + hyb.new_ids
    assert full[:s] == ref.gen_ids[:s]


def test_hybrid_feature_shape(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    press = get_press("streaming_llm", 0.5)
    hyb = generate_hybrids(lm, [(ref, 0)], "streaming_llm", 0.5, press, M)[0]
    assert hyb.features.shape == (len(hyb.new_ids), len(FEATURE_NAMES))


def test_teacher_forced_feature_crosscheck(lm: LoadedModel) -> None:
    # Recomputing features by teacher-forcing the realized sequence must
    # match the inline features (causality + correctness). Approximate,
    # not byte-exact: prefill and incremental decode differ in fp.
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    prompt_len = len(ref.prompt_input_ids)
    full = torch.tensor(
        [ref.prompt_input_ids + ref.gen_ids], dtype=torch.long
    )
    with torch.no_grad():
        logits = lm.model(full).logits[0]  # (L_total, vocab)

    ent_idx = FEATURE_NAMES.index("entropy")
    for t in range(len(ref.gen_ids)):
        row_logits = logits[prompt_len + t - 1].float()
        # Greedy consistency: teacher-forced argmax equals the token.
        assert int(row_logits.argmax()) == ref.gen_ids[t]
        logp = torch.log_softmax(row_logits, dim=-1)
        entropy = float(-(logp.exp() * logp).sum())
        assert abs(entropy - float(ref.features[t, ent_idx])) < 1e-2
