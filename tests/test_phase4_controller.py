"""CPU-only tests for the Phase 4 controller smoke.

Covers:
- FixedBudgetPolicy: returns the same budget every segment.
- RiskBudgetStepPolicy: ±1 step transitions, clamped at grid edges.
- RandomMatchedBudgetPolicy: same multiset as input, seeded shuffle,
  reproducible across constructions with the same seed.
- HeraldOnlineProcessor:
  - emits one SegmentEntry every K calls,
  - mutates `press.target_size` to the policy's chosen value,
  - records (current_budget, next_budget, decision) consistently,
  - is deterministic given the same predictor + signals.
- attach_cache_size_per_segment: aggregates events to per-segment.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from herald.config import TokenSignals
from herald.phase4_controller import (
    DEFAULT_BUDGETS,
    DEFAULT_K,
    ControllerRun,
    FixedBudgetPolicy,
    HeraldOnlineProcessor,
    RandomMatchedBudgetPolicy,
    RiskBudgetStepPolicy,
    SegmentEntry,
    attach_cache_size_per_segment,
)
from herald.phase4_feasibility import CompressionEvent
from herald.phase4_online_features import OnlineFeatureState

# --- Policy tests ---------------------------------------------------


def test_fixed_budget_policy_constant() -> None:
    p = FixedBudgetPolicy(budget=128, name="fixed_128")
    assert p.initial_budget() == 128
    for s in range(5):
        assert p.choose(s, 0.99, last_budget=64) == 128


def test_risk_budget_step_relax_when_above() -> None:
    p = RiskBudgetStepPolicy(
        budgets=(64, 128, 256), threshold=0.5, start_index=0
    )
    assert p.initial_budget() == 64
    # high risk -> relax (move up)
    assert p.choose(0, 0.9, 64) == 128
    assert p.choose(1, 0.9, 128) == 256
    # already at top -> stays
    assert p.choose(2, 0.9, 256) == 256


def test_risk_budget_step_tighten_when_below() -> None:
    p = RiskBudgetStepPolicy(
        budgets=(64, 128, 256), threshold=0.5, start_index=2
    )
    assert p.initial_budget() == 256
    assert p.choose(0, 0.1, 256) == 128
    assert p.choose(1, 0.1, 128) == 64
    assert p.choose(2, 0.1, 64) == 64  # clamped at bottom


def test_risk_budget_step_threshold_inclusive_below() -> None:
    # at threshold -> tighten (relax requires score > threshold)
    p = RiskBudgetStepPolicy(
        budgets=(64, 128, 256), threshold=0.5, start_index=1
    )
    assert p.choose(0, 0.5, 128) == 64


def test_risk_budget_step_validates() -> None:
    with pytest.raises(ValueError, match="budgets"):
        RiskBudgetStepPolicy(budgets=(), threshold=0.5)
    with pytest.raises(ValueError, match="threshold"):
        RiskBudgetStepPolicy(threshold=1.5)
    with pytest.raises(ValueError, match="start_index"):
        RiskBudgetStepPolicy(start_index=99)


def test_random_matched_preserves_multiset() -> None:
    actions = (64, 64, 128, 128, 256, 256)
    p = RandomMatchedBudgetPolicy(actions=actions, seed=42)
    out = [p.initial_budget()]
    for s in range(len(actions) - 1):
        out.append(p.choose(s, 0.0, out[-1]))
    assert sorted(out) == sorted(actions)


def test_random_matched_is_seeded() -> None:
    actions = (64, 128, 256, 64, 128, 256, 64, 128)
    a = RandomMatchedBudgetPolicy(actions=actions, seed=7)
    b = RandomMatchedBudgetPolicy(actions=actions, seed=7)
    c = RandomMatchedBudgetPolicy(actions=actions, seed=8)
    a_seq = [a.initial_budget()] + [
        a.choose(i, 0.0, 0) for i in range(len(actions) - 1)
    ]
    b_seq = [b.initial_budget()] + [
        b.choose(i, 0.0, 0) for i in range(len(actions) - 1)
    ]
    c_seq = [c.initial_budget()] + [
        c.choose(i, 0.0, 0) for i in range(len(actions) - 1)
    ]
    assert a_seq == b_seq
    assert a_seq != c_seq  # vanishingly unlikely collision


def test_random_matched_rejects_empty() -> None:
    with pytest.raises(ValueError, match="actions"):
        RandomMatchedBudgetPolicy(actions=())


# --- HeraldOnlineProcessor tests ------------------------------------


@dataclass
class _MockPredictor:
    """Returns a fixed score sequence for score_one calls."""

    scores: list[float]
    _idx: int = 0

    def score_one(self, row: dict) -> float:
        v = self.scores[self._idx % len(self.scores)]
        self._idx += 1
        return v


class _MockPress:
    """Just exposes a mutable target_size attribute."""

    def __init__(self, target_size: int) -> None:
        self.target_size = target_size


def _fake_signal(_logits, prev=None):  # noqa: ANN001
    """Deterministic mock for herald.signals.extract_signals.

    Returns a TokenSignals with non-NaN Tier 0 fields and a phony
    "previous state" carry. The processor only cares that update()
    accepts what comes back.
    """
    sig = TokenSignals(
        entropy=1.0,
        top1_prob=0.5,
        top5_prob=0.7,
        top5_logprobs=[],
        h_alts=0.5,
        avg_logp=-1.0,
        delta_h=float("nan"),
        delta_h_valid=False,
        kl_div=float("nan"),
        top10_jaccard=float("nan"),
        eff_vocab_size=2.7,
        tail_mass=0.01,
        logit_range=10.0,
    )
    return sig, "state"


def _make_processor(
    scores: list[float],
    policy: object,
    initial_budget: int,
    k: int = 4,
    threshold: float = 0.5,
):
    press = _MockPress(target_size=initial_budget)
    online = OnlineFeatureState(
        press="decoding_knorm",
        compression_ratio=0.0,
        max_new_tokens=64,
    )
    proc = HeraldOnlineProcessor(
        predictor=_MockPredictor(scores=scores),
        online_state=online,
        policy=policy,
        press=press,
        k=k,
        threshold=threshold,
        signal_extractor=_fake_signal,
    )
    return proc, press


def _drive(proc: HeraldOnlineProcessor, n_calls: int) -> None:
    # `scores` argument shape is (batch=1, vocab); we pass anything
    # iterable since _fake_signal ignores it.
    fake_scores = np.zeros((1, 16))
    for _ in range(n_calls):
        proc(input_ids=None, scores=fake_scores)


def test_processor_fires_one_segment_per_k_calls() -> None:
    proc, press = _make_processor(
        scores=[0.9] * 100,
        policy=FixedBudgetPolicy(budget=128, name="fixed_128"),
        initial_budget=128,
        k=4,
    )
    _drive(proc, 12)
    assert len(proc.segments) == 3  # 12 / 4
    assert proc.token_idx == 12
    # Fixed policy didn't change the budget.
    assert press.target_size == 128
    for s in proc.segments:
        assert s.next_budget == 128
        assert s.current_budget == 128
        assert s.decision == "keep"


def test_processor_mutates_press_for_relax() -> None:
    proc, press = _make_processor(
        scores=[0.9] * 100,
        policy=RiskBudgetStepPolicy(
            budgets=(64, 128, 256), threshold=0.5, start_index=0
        ),
        initial_budget=64,
        k=4,
    )
    _drive(proc, 8)  # two segments, both high-risk
    assert len(proc.segments) == 2
    assert proc.segments[0].current_budget == 64
    assert proc.segments[0].next_budget == 128
    assert proc.segments[0].decision == "relax"
    assert proc.segments[1].current_budget == 128
    assert proc.segments[1].next_budget == 256
    assert proc.segments[1].decision == "relax"
    assert press.target_size == 256


def test_processor_mutates_press_for_tighten() -> None:
    proc, press = _make_processor(
        scores=[0.0] * 100,
        policy=RiskBudgetStepPolicy(
            budgets=(64, 128, 256), threshold=0.5, start_index=2
        ),
        initial_budget=256,
        k=4,
    )
    _drive(proc, 8)
    assert proc.segments[0].decision == "tighten"
    assert proc.segments[0].next_budget == 128
    assert proc.segments[1].decision == "tighten"
    assert proc.segments[1].next_budget == 64
    assert press.target_size == 64


def test_processor_segment_score_is_mean_of_buffer() -> None:
    proc, _ = _make_processor(
        scores=[0.0, 0.5, 1.0, 0.5],  # mean = 0.5
        policy=FixedBudgetPolicy(budget=128, name="fixed_128"),
        initial_budget=128,
        k=4,
    )
    _drive(proc, 4)
    assert proc.segments[0].segment_score_mean == pytest.approx(0.5)


def test_processor_segment_score_max_tracked() -> None:
    proc, _ = _make_processor(
        scores=[0.1, 0.9, 0.2, 0.4, 0.0, 0.0, 0.5, 0.7],
        policy=FixedBudgetPolicy(budget=128, name="fixed_128"),
        initial_budget=128,
        k=4,
    )
    _drive(proc, 8)
    assert proc.segments[0].segment_score_max == pytest.approx(0.9)
    assert proc.segments[1].segment_score_max == pytest.approx(0.7)


def test_processor_buffer_clears_between_segments() -> None:
    proc, _ = _make_processor(
        scores=[1.0] * 4 + [0.0] * 4,
        policy=FixedBudgetPolicy(budget=128, name="fixed_128"),
        initial_budget=128,
        k=4,
    )
    _drive(proc, 8)
    assert proc.segments[0].segment_score_mean == pytest.approx(1.0)
    assert proc.segments[1].segment_score_mean == pytest.approx(0.0)


def test_processor_deterministic_two_runs() -> None:
    """Same predictor + same signal stream + same policy -> identical
    segment log."""
    runs = []
    for _ in range(2):
        proc, _ = _make_processor(
            scores=[0.6, 0.4, 0.7, 0.3, 0.2, 0.9, 0.1, 0.8],
            policy=RiskBudgetStepPolicy(
                budgets=(64, 128, 256), threshold=0.5, start_index=1
            ),
            initial_budget=128,
            k=4,
        )
        _drive(proc, 8)
        runs.append(
            [
                (
                    s.segment_idx,
                    s.segment_score_mean,
                    s.current_budget,
                    s.next_budget,
                    s.decision,
                )
                for s in proc.segments
            ]
        )
    assert runs[0] == runs[1]


def test_processor_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        HeraldOnlineProcessor(
            predictor=SimpleNamespace(score_one=lambda r: 0.0),
            online_state=OnlineFeatureState(
                press="x", compression_ratio=0.0, max_new_tokens=10
            ),
            policy=FixedBudgetPolicy(budget=64, name="fixed_64"),
            press=_MockPress(64),
            k=0,
        )


# --- Cache attribution ----------------------------------------------


def _evt(layer: int, before: int, after: int) -> CompressionEvent:
    return CompressionEvent(
        step_idx=-1,
        layer_idx=layer,
        event_type="decode",
        retained_cache_len_before=before,
        retained_cache_len_after=after,
        target_size=64,
        threshold=None,
        wall_clock_seconds=0.0001,
    )


def _segment(idx: int) -> SegmentEntry:
    return SegmentEntry(
        segment_idx=idx,
        tokens_completed=(idx + 1) * 16,
        segment_score_mean=0.5,
        threshold=0.5,
        current_budget=64,
        next_budget=64,
        decision="keep",
    )


def test_attach_cache_size_per_segment_picks_layer_max() -> None:
    # 2 layers, 3 fires each -> 3 segments. Per-segment cache_size
    # is the max retained_after across layers for that fire ordinal.
    events = [
        _evt(0, 80, 64),
        _evt(1, 80, 64),  # segment 0
        _evt(0, 80, 64),
        _evt(1, 80, 56),  # segment 1
        _evt(0, 80, 70),
        _evt(1, 80, 64),  # segment 2
    ]
    run = ControllerRun(
        run_id="r",
        prompt_id="p",
        policy_name="herald_risk_budget_step",
        initial_budget=64,
        generated_text="",
        num_tokens_generated=48,
        stop_reason="max_tokens",
        wall_clock_seconds=1.0,
        wall_clock_per_token=0.02,
        peak_memory_mb=10.0,
        segments=[_segment(0), _segment(1), _segment(2)],
        events=events,
        compression_event_count=6,
        decode_event_count=6,
        total_evicted_tokens=0,
        predicted_answer=None,
        correct=None,
        ground_truth="",
    )
    rows = attach_cache_size_per_segment(run, k=16)
    assert len(rows) == 3
    assert rows[0]["cache_size_observed"] == 64
    assert rows[1]["cache_size_observed"] == 64  # max(64, 56)
    assert rows[2]["cache_size_observed"] == 70
    # New fields: start/end positions and per-segment evictions.
    assert rows[0]["start_token_pos"] == 0
    assert rows[1]["start_token_pos"] == 16
    assert rows[2]["start_token_pos"] == 32
    assert rows[0]["end_token_pos"] == 16
    assert rows[2]["end_token_pos"] == 48
    # Segment 0: layer0=80-64=16, layer1=80-64=16 -> 32.
    assert rows[0]["evicted_tokens_in_segment"] == 32
    # Segment 1: 16 + 24 = 40.
    assert rows[1]["evicted_tokens_in_segment"] == 40
    # Segment 2: 10 + 16 = 26.
    assert rows[2]["evicted_tokens_in_segment"] == 26
    # Sum-of-segments equals the run-level total_evicted definition.
    sum_seg_evictions = sum(r["evicted_tokens_in_segment"] for r in rows)
    sum_run_evictions = sum(
        max(0, e.retained_cache_len_before - e.retained_cache_len_after)
        for e in events
    )
    assert sum_seg_evictions == sum_run_evictions


# --- Sanity on the exported constants ------------------------------


def test_default_constants() -> None:
    assert DEFAULT_K == 16
    assert DEFAULT_BUDGETS == (64, 128, 256)
