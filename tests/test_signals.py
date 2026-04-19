"""Tests for herald.signals — per-token signal extraction."""

import math

import pytest
import torch

from herald.config import TokenSignals
from herald.signals import (
    StepState,
    compute_lookback_ratios,
    extract_signals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _uniform_logits(vocab_size: int = 100) -> torch.Tensor:
    """Uniform distribution — maximum entropy."""
    return torch.zeros(vocab_size)


def _peaked_logits(vocab_size: int = 100, peak: float = 20.0) -> torch.Tensor:
    """One dominant token — low entropy."""
    logits = torch.zeros(vocab_size)
    logits[0] = peak
    return logits


def _two_token_logits() -> torch.Tensor:
    """50/50 between two tokens — entropy = ln(2)."""
    logits = torch.zeros(2)
    return logits


# ---------------------------------------------------------------------------
# Return type and structure
# ---------------------------------------------------------------------------


class TestReturnStructure:
    def test_returns_tuple_of_signals_and_state(self):
        result = extract_signals(_uniform_logits())
        assert isinstance(result, tuple)
        assert len(result) == 2
        signals, state = result
        assert isinstance(signals, TokenSignals)
        assert isinstance(state, StepState)

    def test_all_signal_fields_present(self):
        signals, _ = extract_signals(_uniform_logits())
        expected_fields = {
            "entropy",
            "top1_prob",
            "top5_prob",
            "top5_logprobs",
            "h_alts",
            "avg_logp",
            "delta_h",
            "delta_h_valid",
            "kl_div",
            "top10_jaccard",
            "eff_vocab_size",
            "tail_mass",
            "logit_range",
            "lookback_ratio",
        }
        actual_fields = set(signals.model_fields.keys())
        assert expected_fields == actual_fields

    def test_state_has_required_fields(self):
        _, state = extract_signals(_uniform_logits())
        assert isinstance(state.entropy, float)
        assert isinstance(state.log_probs, torch.Tensor)
        assert isinstance(state.top10_ids, frozenset)


# ---------------------------------------------------------------------------
# First token (no prev state)
# ---------------------------------------------------------------------------


class TestFirstToken:
    def test_temporal_features_are_nan(self):
        signals, _ = extract_signals(_uniform_logits())
        assert math.isnan(signals.delta_h)
        assert math.isnan(signals.kl_div)
        assert math.isnan(signals.top10_jaccard)
        assert signals.delta_h_valid is False

    def test_non_temporal_features_are_populated(self):
        signals, _ = extract_signals(_uniform_logits())
        assert signals.entropy > 0
        assert signals.top1_prob > 0
        assert signals.top5_prob > 0
        assert len(signals.top5_logprobs) == 5
        assert signals.eff_vocab_size > 0


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------


class TestEntropy:
    def test_uniform_distribution_max_entropy(self):
        vocab_size = 100
        signals, _ = extract_signals(_uniform_logits(vocab_size))
        expected = math.log(vocab_size)
        assert signals.entropy == pytest.approx(expected, abs=0.01)

    def test_peaked_distribution_low_entropy(self):
        signals_uniform, _ = extract_signals(_uniform_logits())
        signals_peaked, _ = extract_signals(_peaked_logits())
        assert signals_peaked.entropy < signals_uniform.entropy

    def test_two_token_uniform(self):
        signals, _ = extract_signals(_two_token_logits())
        assert signals.entropy == pytest.approx(math.log(2), abs=0.001)


# ---------------------------------------------------------------------------
# Top-k probabilities
# ---------------------------------------------------------------------------


class TestTopK:
    def test_top1_prob_high_for_peaked(self):
        signals, _ = extract_signals(_peaked_logits())
        assert signals.top1_prob > 0.99

    def test_top1_prob_uniform(self):
        vocab_size = 100
        signals, _ = extract_signals(_uniform_logits(vocab_size))
        assert signals.top1_prob == pytest.approx(1.0 / vocab_size, abs=0.001)

    def test_top5_prob_geq_top1(self):
        signals, _ = extract_signals(_uniform_logits())
        assert signals.top5_prob >= signals.top1_prob

    def test_top5_logprobs_length(self):
        signals, _ = extract_signals(_uniform_logits())
        assert len(signals.top5_logprobs) == 5

    def test_top5_logprobs_sorted_descending(self):
        signals, _ = extract_signals(_uniform_logits())
        for i in range(len(signals.top5_logprobs) - 1):
            assert signals.top5_logprobs[i] >= signals.top5_logprobs[i + 1]

    def test_small_vocab_fewer_than_5(self):
        logits = torch.zeros(3)
        signals, _ = extract_signals(logits)
        assert len(signals.top5_logprobs) == 3


# ---------------------------------------------------------------------------
# H_alts (competitor entropy)
# ---------------------------------------------------------------------------


class TestHAlts:
    def test_peaked_low_h_alts(self):
        """When one token dominates, competitors still have some entropy."""
        signals, _ = extract_signals(_peaked_logits())
        # h_alts should be positive (competitors are uniform-ish)
        assert signals.h_alts > 0

    def test_h_alts_zero_when_single_token(self):
        """If top1 has all the mass, h_alts = 0."""
        logits = torch.tensor([-1000.0, 1000.0])
        signals, _ = extract_signals(logits)
        assert signals.h_alts == pytest.approx(0.0, abs=0.01)

    def test_h_alts_nonnegative(self):
        signals, _ = extract_signals(_uniform_logits())
        assert signals.h_alts >= 0


# ---------------------------------------------------------------------------
# Effective vocabulary size
# ---------------------------------------------------------------------------


class TestEffVocabSize:
    def test_uniform_equals_vocab_size(self):
        vocab_size = 100
        signals, _ = extract_signals(_uniform_logits(vocab_size))
        assert signals.eff_vocab_size == pytest.approx(vocab_size, rel=0.01)

    def test_peaked_near_one(self):
        signals, _ = extract_signals(_peaked_logits(peak=50.0))
        assert signals.eff_vocab_size < 5


# ---------------------------------------------------------------------------
# Tail mass
# ---------------------------------------------------------------------------


class TestTailMass:
    def test_small_vocab_zero_tail(self):
        """With vocab_size <= 20, all mass is in top-20 → tail = 0."""
        signals, _ = extract_signals(torch.zeros(10))
        assert signals.tail_mass == pytest.approx(0.0, abs=0.001)

    def test_uniform_large_vocab_has_tail(self):
        signals, _ = extract_signals(_uniform_logits(1000))
        # 980/1000 of the mass is outside top-20
        assert signals.tail_mass == pytest.approx(0.98, abs=0.01)


# ---------------------------------------------------------------------------
# Logit range
# ---------------------------------------------------------------------------


class TestLogitRange:
    def test_uniform_zero_range(self):
        signals, _ = extract_signals(_uniform_logits())
        assert signals.logit_range == pytest.approx(0.0, abs=0.001)

    def test_peaked_positive_range(self):
        signals, _ = extract_signals(_peaked_logits())
        assert signals.logit_range > 0


# ---------------------------------------------------------------------------
# Temporal features (with prev state)
# ---------------------------------------------------------------------------


class TestTemporalFeatures:
    def test_delta_h_computed(self):
        _, state1 = extract_signals(_uniform_logits())
        signals2, _ = extract_signals(_peaked_logits(), prev=state1)
        assert not math.isnan(signals2.delta_h)
        assert signals2.delta_h_valid is True
        # Peaked has lower entropy → delta_h should be negative
        assert signals2.delta_h < 0

    def test_delta_h_zero_same_logits(self):
        logits = _uniform_logits()
        _, state1 = extract_signals(logits)
        signals2, _ = extract_signals(logits, prev=state1)
        assert signals2.delta_h == pytest.approx(0.0, abs=0.001)

    def test_kl_div_nonnegative(self):
        _, state1 = extract_signals(_uniform_logits())
        signals2, _ = extract_signals(_peaked_logits(), prev=state1)
        assert not math.isnan(signals2.kl_div)
        assert signals2.kl_div >= 0

    def test_kl_div_zero_same_distribution(self):
        logits = _uniform_logits()
        _, state1 = extract_signals(logits)
        signals2, _ = extract_signals(logits, prev=state1)
        assert signals2.kl_div == pytest.approx(0.0, abs=0.001)

    def test_kl_div_large_for_different_distributions(self):
        _, state1 = extract_signals(_uniform_logits())
        signals2, _ = extract_signals(_peaked_logits(peak=50.0), prev=state1)
        assert not math.isnan(signals2.kl_div)
        assert signals2.kl_div > 1.0

    def test_jaccard_one_for_identical(self):
        logits = _uniform_logits()
        _, state1 = extract_signals(logits)
        signals2, _ = extract_signals(logits, prev=state1)
        assert signals2.top10_jaccard == pytest.approx(1.0)

    def test_jaccard_low_for_different(self):
        # Two distributions with completely different top-10
        logits_a = torch.zeros(100)
        logits_a[:10] = 20.0  # tokens 0-9 dominate
        logits_b = torch.zeros(100)
        logits_b[50:60] = 20.0  # tokens 50-59 dominate
        _, state_a = extract_signals(logits_a)
        signals_b, _ = extract_signals(logits_b, prev=state_a)
        assert not math.isnan(signals_b.top10_jaccard)
        assert signals_b.top10_jaccard < 0.1

    def test_jaccard_between_zero_and_one(self):
        _, state1 = extract_signals(_uniform_logits())
        signals2, _ = extract_signals(_peaked_logits(), prev=state1)
        assert not math.isnan(signals2.top10_jaccard)
        assert 0.0 <= signals2.top10_jaccard <= 1.0


# ---------------------------------------------------------------------------
# State chaining
# ---------------------------------------------------------------------------


class TestStateChaining:
    def test_three_step_chain(self):
        """Simulate 3 decoding steps, verify state threads correctly."""
        logits_seq = [_uniform_logits(), _peaked_logits(), _uniform_logits()]
        state = None
        all_signals = []
        for logits in logits_seq:
            signals, state = extract_signals(logits, prev=state)
            all_signals.append(signals)

        # First token: temporal features are NaN
        assert math.isnan(all_signals[0].delta_h)
        assert math.isnan(all_signals[0].kl_div)
        assert math.isnan(all_signals[0].top10_jaccard)

        # Second and third: temporal features present
        for sig in all_signals[1:]:
            assert not math.isnan(sig.delta_h)
            assert not math.isnan(sig.kl_div)
            assert not math.isnan(sig.top10_jaccard)
            assert sig.delta_h_valid is True

    def test_state_log_probs_detached(self):
        """State log_probs should not carry gradients."""
        logits = _uniform_logits().requires_grad_(True)
        _, state = extract_signals(logits)
        assert not state.log_probs.requires_grad


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_token_vocab(self):
        logits = torch.tensor([5.0])
        signals, state = extract_signals(logits)
        assert signals.entropy == pytest.approx(0.0, abs=0.001)
        assert signals.top1_prob == pytest.approx(1.0, abs=0.001)
        assert signals.eff_vocab_size == pytest.approx(1.0, abs=0.1)
        assert len(signals.top5_logprobs) == 1

    def test_float16_input(self):
        logits = _uniform_logits().half()
        signals, _ = extract_signals(logits)
        assert signals.entropy > 0

    def test_large_logits_no_overflow(self):
        logits = torch.randn(50000) * 100
        signals, _ = extract_signals(logits)
        assert math.isfinite(signals.entropy)
        assert math.isfinite(signals.logit_range)

    def test_negative_logits(self):
        logits = torch.randn(100) - 10.0
        signals, _ = extract_signals(logits)
        assert signals.entropy > 0
        assert signals.top1_prob > 0


# ---------------------------------------------------------------------------
# compute_lookback_ratios
# ---------------------------------------------------------------------------


class TestLookbackRatios:
    def test_all_attention_to_context_returns_one(self):
        # 1 layer, 1 head; attention concentrated on context
        # input_len=2, generated 1 token (k=3, last is the new token)
        attn = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # all on ctx
        out = compute_lookback_ratios(((attn,),), input_len=2)
        assert out == [1.0]

    def test_all_attention_to_generated_returns_zero(self):
        # New token attends entirely to a previously generated
        # token (position 2 is the prior generated, position 3
        # is itself with 0 weight)
        attn = torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]])
        out = compute_lookback_ratios(((attn,),), input_len=2)
        assert out == [0.0]

    def test_mixed_attention_returns_proportion(self):
        # Half on context, half on generated
        attn = torch.tensor([[[[0.25, 0.25, 0.5]]]])
        out = compute_lookback_ratios(((attn,),), input_len=2)
        assert abs(out[0] - 0.5) < 1e-4

    def test_averages_across_heads_and_layers(self):
        # 2 heads, 2 layers; head A all-ctx, head B all-gen.
        # Per-layer ratio = (1.0 + 0.0) / 2 = 0.5; both layers
        # equal → final = 0.5
        head_a = torch.tensor([[1.0, 0.0, 0.0]])  # (q=1, k=3)
        head_b = torch.tensor([[0.0, 0.0, 1.0]])
        layer = torch.stack([head_a, head_b]).unsqueeze(0)  # (1,2,1,3)
        out = compute_lookback_ratios(((layer, layer),), input_len=2)
        assert abs(out[0] - 0.5) < 1e-4

    def test_multiple_steps_returns_list(self):
        a1 = torch.tensor([[[[1.0, 0.0]]]])  # input_len=2
        a2 = torch.tensor([[[[0.5, 0.5, 0.0]]]])  # next step
        out = compute_lookback_ratios(((a1,), (a2,)), input_len=2)
        assert len(out) == 2
