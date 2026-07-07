"""Live grace-window controller invariants on a tiny CPU model.

The controller must be numerically indistinguishable from the sweep
machinery that produced the recorded dataset:

- segment-chained reference decoding == one-call generate_reference;
- a committed attempt at s == generate_hybrids at the same s;
- a never-commit episode outputs exactly the reference;
- the alarm sees the float16-quantized post-switch stream the recorded
  blocks stored.

Marked `model`: loads a tiny random Llama on CPU; only the mechanism
is under test.
"""

from typing import Any, cast

import numpy as np
import pytest
import torch

import herald.generate as G
import herald.live_controller as LC
from herald.features import FEATURE_NAMES
from herald.generate import (
    LoadedModel,
    generate_hybrids,
    generate_reference,
    load_model,
)
from herald.grace_window import assemble_alarm_row
from herald.presses import get_press
from herald.tasks import PromptRecord

pytestmark = pytest.mark.model

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
M = 40
STRIDE = 16
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
    def build(lm: LoadedModel, record: PromptRecord) -> torch.Tensor:
        text = record.messages[-1]["content"]
        ids = lm.tokenizer(text, return_tensors="pt").input_ids[0]
        return cast(torch.Tensor, ids)

    monkeypatch.setattr(G, "build_input_ids", build)
    monkeypatch.setattr(LC, "build_input_ids", build)


def _rec(content: str, pid: str) -> PromptRecord:
    return PromptRecord(
        task="ifeval",
        prompt_id=pid,
        messages=[{"role": "user", "content": content}],
        gold={"instruction_id_list": [], "kwargs": []},
    )


class ScriptedAlarm:
    """Alarm stub: scripted commit decisions, captures assembled rows."""

    def __init__(self, decisions: list[bool], k: int = 2) -> None:
        self.k = k
        self._decisions = decisions
        self.rows: list[dict[str, Any]] = []
        self.theta = 0.5

    def score(self, row: dict[str, Any]) -> float:
        self.rows.append(row)
        commit = self._decisions[len(self.rows) - 1]
        return 0.0 if commit else 1.0

    def commits(self, score: float) -> bool:
        return score <= self.theta


class RowParityAlarm(ScriptedAlarm):
    """Alarm stub that pins live rows to batch assembly rows."""

    def __init__(
        self,
        decisions: list[bool],
        expected: list[dict[str, Any]],
        k: int = 2,
    ) -> None:
        super().__init__(decisions, k=k)
        self._expected = expected

    def score(self, row: dict[str, Any]) -> float:
        expected = self._expected[len(self.rows)]
        assert row == expected
        return super().score(row)


def _run(
    lm: LoadedModel, alarm: ScriptedAlarm, pid: str = "p0"
) -> LC.LiveEpisode:
    return LC.run_episode(
        lm,
        _rec(LONG, pid),
        lambda: get_press("streaming_llm", 0.5),
        alarm,
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
    )


def test_never_commit_reproduces_reference(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    n_grid = len(range(0, len(ref.gen_ids), STRIDE))
    alarm = ScriptedAlarm([False] * n_grid)
    ep = _run(lm, alarm)
    assert ep.commit_s is None
    assert ep.ref_ids == ref.gen_ids
    assert ep.text == ref.text
    assert [a.s for a in ep.attempts] == list(
        range(0, len(ref.gen_ids), STRIDE)
    )
    assert all(not a.committed for a in ep.attempts)
    # every revert observed exactly k tokens
    assert all(a.n_new_tokens == alarm.k for a in ep.attempts)


def test_chained_reference_features_match_one_call(
    lm: LoadedModel,
) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    alarm = ScriptedAlarm([False] * 10)
    ep = _run(lm, alarm)
    # the alarm rows carry feat__position == s, proving row alignment
    for a, row in zip(ep.attempts, alarm.rows, strict=True):
        assert row["feat__position"] == float(a.s)
    # pre-switch entropy at row s must match the recorded reference
    ent = FEATURE_NAMES.index("entropy")
    for a, row in zip(ep.attempts, alarm.rows, strict=True):
        want = float(ref.features[a.s, ent])
        assert row["feat__entropy"] == pytest.approx(want, rel=1e-4)


def test_commit_at_s0_reproduces_hybrid(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    press = get_press("streaming_llm", 0.5)
    hyb = generate_hybrids(lm, [(ref, 0)], "streaming_llm", 0.5, press, M)[0]
    alarm = ScriptedAlarm([True])
    ep = _run(lm, alarm)
    assert ep.commit_s == 0
    assert len(ep.attempts) == 1
    assert ep.attempts[0].committed
    assert ep.new_ids == hyb.new_ids
    assert ep.text == hyb.text


def test_commit_at_later_s_reproduces_hybrid(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    assert len(ref.gen_ids) > STRIDE, "run too short for a late switch"
    press = get_press("streaming_llm", 0.5)
    hyb = generate_hybrids(
        lm, [(ref, STRIDE)], "streaming_llm", 0.5, press, M
    )[0]
    alarm = ScriptedAlarm([False, True])
    ep = _run(lm, alarm)
    assert ep.commit_s == STRIDE
    assert ep.new_ids == hyb.new_ids
    assert ep.text == hyb.text
    assert [a.s for a in ep.attempts] == [0, STRIDE]


def test_alarm_sees_quantized_hybrid_stream(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    press = get_press("streaming_llm", 0.5)
    hyb = generate_hybrids(
        lm, [(ref, STRIDE)], "streaming_llm", 0.5, press, M
    )[0]
    alarm = ScriptedAlarm([False, True])
    ep = _run(lm, alarm)
    assert ep.commit_s == STRIDE
    row = alarm.rows[1]
    ent = FEATURE_NAMES.index("entropy")
    for step, stat in ((0, "step0"),):
        want = float(
            np.float32(np.float16(np.float32(hyb.features[step, ent])))
        )
        assert row[f"hyb__{stat}_entropy_k2"] == pytest.approx(
            want, rel=1e-6
        ), stat


def test_live_rows_match_original_batch_assembly(lm: LoadedModel) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    press0 = get_press("streaming_llm", 0.5)
    press1 = get_press("streaming_llm", 0.5)
    hyb0 = generate_hybrids(lm, [(ref, 0)], "streaming_llm", 0.5, press0, M)[
        0
    ]
    hyb1 = generate_hybrids(
        lm, [(ref, STRIDE)], "streaming_llm", 0.5, press1, M
    )[0]
    expected = [
        assemble_alarm_row(
            ref_raw=ref.features,
            s=0,
            block=hyb0.features[:2],
            ratio=0.5,
            k=2,
        ),
        assemble_alarm_row(
            ref_raw=ref.features,
            s=STRIDE,
            block=hyb1.features[:2],
            ratio=0.5,
            k=2,
        ),
    ]
    alarm = RowParityAlarm([False, True], expected)
    ep = _run(lm, alarm)
    assert [a.s for a in ep.attempts] == [0, STRIDE]
    assert [a.score for a in ep.attempts] == [1.0, 0.0]
    assert [a.committed for a in ep.attempts] == [False, True]


def test_attempt_budget_matches_sweep_cap(lm: LoadedModel) -> None:
    # A committed attempt at s may generate at most M - s tokens, the
    # same budget the recorded hybrids had.
    alarm = ScriptedAlarm([False, True])
    ep = _run(lm, alarm)
    assert ep.commit_s == STRIDE
    assert len(ep.new_ids) <= M - STRIDE
