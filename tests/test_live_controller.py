# pyright: reportMissingImports=false, reportOperatorIssue=false, reportPrivateImportUsage=false

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
from kvpress import ExpectedAttentionStatsPress

import herald.generate as G
import herald.live_controller as LC
from herald.expected_attention_stats import collect_query_moments
from herald.features import FEATURE_NAMES
from herald.generate import (
    LoadedModel,
    generate_baseline,
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


def _as_float(value: Any, *, source: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise AssertionError(f"{source} must be numeric") from error


def _expected_attention_stats_press(
    lm: LoadedModel,
) -> ExpectedAttentionStatsPress:
    config = lm.model.config
    dtype = next(lm.model.parameters()).dtype
    press = ExpectedAttentionStatsPress(
        compression_ratio=0.5,
        use_covariance=False,
    )
    query_count = (
        config.num_hidden_layers
        * config.num_attention_heads
        * config.head_dim
    )
    press.mu = (
        torch.arange(query_count, dtype=dtype).reshape(
            config.num_hidden_layers,
            config.num_attention_heads,
            config.head_dim,
        )
        / query_count
    )
    press.cov = torch.zeros(
        config.num_hidden_layers,
        config.num_attention_heads,
        config.head_dim,
        config.head_dim,
        dtype=dtype,
    )
    return press


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


class ScriptedGate:
    """Gate stub: scripted attempt decisions, captures pre-switch rows."""

    def __init__(self, decisions: list[bool]) -> None:
        self._decisions = decisions
        self.rows: list[dict[str, Any]] = []
        self.tau = 0.5

    def score(self, row: dict[str, Any]) -> float:
        self.rows.append(row)
        attempt = self._decisions[len(self.rows) - 1]
        return 1.0 if attempt else 0.0

    def attempts(self, score: float) -> bool:
        return score >= self.tau


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
    assert ep.peak_kv_cache_bytes > 0
    assert all(a.peak_kv_cache_bytes > 0 for a in ep.attempts)


def test_plain_baseline_matches_reference_and_reports_kv_bytes(
    lm: LoadedModel,
) -> None:
    record = _rec(LONG, "p-baseline")
    [ref] = generate_reference(lm, [record], M)
    baseline = generate_baseline(lm, record, M)

    assert baseline.gen_ids == ref.gen_ids
    assert baseline.text == ref.text
    assert baseline.peak_kv_cache_bytes > 0


@pytest.mark.parametrize("compressor", ["streaming_llm", "knorm"])
def test_score_cache_fork_matches_kvpress_prefill(
    lm: LoadedModel,
    compressor: str,
) -> None:
    record = _rec(LONG, f"p-fork-{compressor}")
    prompt_ids = G.build_input_ids(lm, record).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)

    with torch.no_grad():
        plain = lm.model.generate(  # type: ignore[operator]
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            generation_config=lm.gen_config,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
    press = get_press(compressor, 0.5)
    with torch.no_grad(), press(lm.model):
        compressed = lm.model.generate(  # type: ignore[operator]
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            generation_config=lm.gen_config,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )

    forked = LC._fork_score_cache(lm.model, plain.past_key_values, press)

    for original_layer, forked_layer, compressed_layer in zip(
        plain.past_key_values.layers,
        forked.layers,
        compressed.past_key_values.layers,
        strict=True,
    ):
        assert torch.equal(forked_layer.keys, compressed_layer.keys)
        assert torch.equal(forked_layer.values, compressed_layer.values)
        assert forked_layer.keys.data_ptr() != original_layer.keys.data_ptr()
        assert (
            forked_layer.values.data_ptr() != original_layer.values.data_ptr()
        )


def test_expected_attention_stats_collection_uses_all_layers(
    lm: LoadedModel,
) -> None:
    token_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    mu, cov, count = collect_query_moments(
        lm.model,
        [token_ids],
        n_sink=1,
    )

    config = lm.model.config
    expected_mu_shape = (
        config.num_hidden_layers,
        config.num_attention_heads,
        config.head_dim,
    )
    expected_cov_shape = (*expected_mu_shape, config.head_dim)
    assert count == 3
    assert tuple(mu.shape) == expected_mu_shape
    assert tuple(cov.shape) == expected_cov_shape
    assert torch.isfinite(mu).all()
    assert torch.isfinite(cov).all()


def test_expected_attention_stats_cache_fork_matches_prefill(
    lm: LoadedModel,
) -> None:
    record = _rec(LONG, "p-fork-expected-attention-stats")
    prompt_ids = G.build_input_ids(lm, record).unsqueeze(0)
    attention_mask = torch.ones_like(prompt_ids)
    press = _expected_attention_stats_press(lm)

    with torch.no_grad():
        plain = lm.model.generate(  # type: ignore[operator]
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            generation_config=lm.gen_config,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )
    with torch.no_grad(), press(lm.model):
        compressed = lm.model.generate(  # type: ignore[operator]
            input_ids=prompt_ids,
            attention_mask=attention_mask,
            generation_config=lm.gen_config,
            max_new_tokens=1,
            return_dict_in_generate=True,
        )

    forked = LC._fork_score_cache(lm.model, plain.past_key_values, press)

    for original_layer, forked_layer, compressed_layer in zip(
        plain.past_key_values.layers,
        forked.layers,
        compressed.past_key_values.layers,
        strict=True,
    ):
        assert torch.equal(forked_layer.keys, compressed_layer.keys)
        assert torch.equal(forked_layer.values, compressed_layer.values)
        assert forked_layer.keys.data_ptr() != original_layer.keys.data_ptr()
        assert (
            forked_layer.values.data_ptr() != original_layer.values.data_ptr()
        )


def test_expected_attention_stats_direct_attempt_matches_hybrid(
    lm: LoadedModel,
) -> None:
    record = _rec(LONG, "p-expected-attention-stats-direct")
    [ref] = generate_reference(lm, [record], M)
    hybrid = generate_hybrids(
        lm,
        [(ref, 0)],
        "expected_attention_stats",
        0.5,
        _expected_attention_stats_press(lm),
        M,
    )[0]

    episode = LC.run_episode(
        lm,
        record,
        lambda: _expected_attention_stats_press(lm),
        ScriptedAlarm([True]),
        compressor="expected_attention_stats",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
    )

    assert episode.commit_s == 0
    assert episode.new_ids == hybrid.new_ids
    assert episode.text == hybrid.text
    assert episode.attempts[0].recomputed_prefill_tokens == 0


def test_expected_attention_stats_late_direct_attempt_matches_hybrid(
    lm: LoadedModel,
) -> None:
    record = _rec(LONG, "p-expected-attention-stats-late")
    [ref] = generate_reference(lm, [record], M)
    hybrid = generate_hybrids(
        lm,
        [(ref, STRIDE)],
        "expected_attention_stats",
        0.5,
        _expected_attention_stats_press(lm),
        M,
    )[0]

    episode = LC.run_episode(
        lm,
        record,
        lambda: _expected_attention_stats_press(lm),
        ScriptedAlarm([False, True]),
        compressor="expected_attention_stats",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
    )

    assert episode.commit_s == STRIDE
    assert episode.new_ids == hybrid.new_ids
    assert episode.text == hybrid.text
    assert episode.attempts[-1].recomputed_prefill_tokens == 0


def test_chained_reference_features_match_one_call(
    lm: LoadedModel,
) -> None:
    [ref] = generate_reference(lm, [_rec(LONG, "p0")], M)
    alarm = ScriptedAlarm([False] * 10)
    ep = _run(lm, alarm)
    # the alarm rows carry feat__position == s, proving row alignment
    for a, row in zip(ep.attempts, alarm.rows, strict=True):
        assert row["feat__position"] == _as_float(
            a.s,
            source="attempt position",
        )
    # pre-switch entropy at row s must match the recorded reference
    ent = FEATURE_NAMES.index("entropy")
    for a, row in zip(ep.attempts, alarm.rows, strict=True):
        want = _as_float(
            ref.features[a.s, ent],
            source="reference entropy",
        )
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


@pytest.mark.parametrize("compressor", ["streaming_llm", "knorm"])
def test_direct_cache_attempt_matches_hybrid_without_reprefill(
    lm: LoadedModel,
    compressor: str,
) -> None:
    record = _rec(LONG, f"p-direct-{compressor}")
    [ref] = generate_reference(lm, [record], M)
    hybrid = generate_hybrids(
        lm,
        [(ref, STRIDE)],
        compressor,
        0.5,
        get_press(compressor, 0.5),
        M,
    )[0]
    episode = LC.run_episode(
        lm,
        record,
        lambda: get_press(compressor, 0.5),
        ScriptedAlarm([False, True]),
        compressor=compressor,
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
    )

    assert episode.commit_s == STRIDE
    assert episode.new_ids == hybrid.new_ids
    assert episode.text == hybrid.text
    assert all(a.recomputed_prefill_tokens == 0 for a in episode.attempts)


def test_sustained_ratio_reduces_retained_kv_peak(lm: LoadedModel) -> None:
    record = _rec(LONG, "p-sustained")
    plain = LC.run_episode(
        lm,
        record,
        lambda: get_press("streaming_llm", 0.5),
        ScriptedAlarm([True]),
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
    )
    sustained = LC.run_episode(
        lm,
        record,
        lambda: get_press("streaming_llm", 0.5),
        ScriptedAlarm([True]),
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
        sustain_interval=4,
    )

    assert sustained.attempts[0].peak_kv_cache_bytes < (
        plain.attempts[0].peak_kv_cache_bytes
    )


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


def test_gate_skip_s0_then_allow_s16_reproduces_late_commit(
    lm: LoadedModel,
) -> None:
    ungated = _run(lm, ScriptedAlarm([False, True]), pid="p-late")
    gated = LC.run_episode(
        lm,
        _rec(LONG, "p-late"),
        lambda: get_press("streaming_llm", 0.5),
        ScriptedAlarm([True]),
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
        gate=ScriptedGate([False, True]),
    )
    assert gated.commit_s == STRIDE
    assert gated.new_ids == ungated.new_ids
    assert gated.text == ungated.text
    assert [skip.s for skip in gated.skips] == [0]
    assert [a.s for a in gated.attempts] == [STRIDE]
    assert [a.gate_score for a in gated.attempts] == [1.0]


def test_gate_allow_all_reproduces_no_gate_episode(
    lm: LoadedModel,
) -> None:
    no_gate = _run(lm, ScriptedAlarm([False, True]), pid="p-allow")
    gate = ScriptedGate([True, True])
    gated = LC.run_episode(
        lm,
        _rec(LONG, "p-allow"),
        lambda: get_press("streaming_llm", 0.5),
        ScriptedAlarm([False, True]),
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
        gate=gate,
    )
    assert gated.commit_s == no_gate.commit_s
    assert gated.ref_ids == no_gate.ref_ids
    assert gated.new_ids == no_gate.new_ids
    assert gated.text == no_gate.text
    assert gated.skips == []
    gated_attempts = [
        (a.s, a.score, a.committed, a.n_new_tokens, a.block_len)
        for a in gated.attempts
    ]
    no_gate_attempts = [
        (a.s, a.score, a.committed, a.n_new_tokens, a.block_len)
        for a in no_gate.attempts
    ]
    assert gated_attempts == no_gate_attempts


def test_gate_sees_same_preswitch_features_as_alarm(
    lm: LoadedModel,
) -> None:
    alarm = ScriptedAlarm([False, True])
    gate = ScriptedGate([True, True])
    ep = LC.run_episode(
        lm,
        _rec(LONG, "p-feat"),
        lambda: get_press("streaming_llm", 0.5),
        alarm,
        compressor="streaming_llm",
        ratio=0.5,
        max_new_tokens=M,
        stride=STRIDE,
        gate=gate,
    )
    assert [a.s for a in ep.attempts] == [0, STRIDE]
    assert len(gate.rows) == len(alarm.rows)
    for gate_row, alarm_row in zip(gate.rows, alarm.rows, strict=True):
        assert gate_row["task"] == alarm_row["task"]
        assert gate_row["ratio"] == alarm_row["ratio"]
        for name, value in alarm_row.items():
            if name.startswith("feat__"):
                assert gate_row[name] == value


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
        want = _as_float(
            np.float32(np.float16(np.float32(hyb.features[step, ent]))),
            source="quantized hybrid entropy",
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
