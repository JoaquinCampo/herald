"""Tests for herald.config — Pydantic models."""

import math
from pathlib import Path

import pytest

from herald.config import ExperimentConfig, RunResult, TokenSignals

# ---------------------------------------------------------------------------
# ExperimentConfig
# ---------------------------------------------------------------------------


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.model_name == "Qwen/Qwen2.5-7B-Instruct"
        assert cfg.press_name == "streaming_llm"
        assert cfg.compression_ratio == 0.875
        assert cfg.max_new_tokens == 512
        assert cfg.num_prompts == 10
        assert cfg.seed == 42
        assert cfg.device == "auto"
        assert cfg.output_dir == Path("results")
        assert cfg.prompt_timeout_seconds == 300.0

    def test_custom_values(self):
        cfg = ExperimentConfig(
            model_name="meta-llama/Llama-2-7b",
            press_name="snapkv",
            compression_ratio=0.5,
            num_prompts=100,
            seed=123,
            device="cpu",
            output_dir=Path("/tmp/out"),
        )
        assert cfg.model_name == "meta-llama/Llama-2-7b"
        assert cfg.press_name == "snapkv"
        assert cfg.compression_ratio == 0.5
        assert cfg.num_prompts == 100
        assert cfg.seed == 123

    def test_resolve_device_explicit(self):
        cfg = ExperimentConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_resolve_device_explicit_cuda(self):
        cfg = ExperimentConfig(device="cuda")
        assert cfg.resolve_device() == "cuda"


# ---------------------------------------------------------------------------
# TokenSignals
# ---------------------------------------------------------------------------


class TestTokenSignals:
    def test_defaults(self):
        sig = TokenSignals(entropy=1.0, top1_prob=0.5, top5_prob=0.9)
        assert sig.top5_logprobs == []
        assert sig.h_alts == 0.0
        assert sig.avg_logp == 0.0
        assert math.isnan(sig.delta_h)
        assert sig.delta_h_valid is False
        assert math.isnan(sig.kl_div)
        assert math.isnan(sig.top10_jaccard)
        assert sig.eff_vocab_size == 0.0
        assert sig.tail_mass == 0.0
        assert sig.logit_range == 0.0

    def test_full_construction(self):
        sig = TokenSignals(
            entropy=2.5,
            top1_prob=0.3,
            top5_prob=0.8,
            top5_logprobs=[-0.5, -1.0, -1.5, -2.0, -2.5],
            h_alts=1.2,
            avg_logp=-5.0,
            delta_h=-0.3,
            delta_h_valid=True,
            kl_div=0.1,
            top10_jaccard=0.7,
            eff_vocab_size=12.18,
            tail_mass=0.05,
            logit_range=15.0,
        )
        assert sig.entropy == 2.5
        assert sig.delta_h == -0.3
        assert sig.delta_h_valid is True

    def test_serialization_roundtrip(self):
        sig = TokenSignals(
            entropy=1.5,
            top1_prob=0.6,
            top5_prob=0.95,
            delta_h=0.1,
            delta_h_valid=True,
        )
        json_str = sig.model_dump_json()
        restored = TokenSignals.model_validate_json(json_str)
        # NaN fields need special comparison (NaN != NaN)
        for field in TokenSignals.model_fields:
            orig = getattr(sig, field)
            rest = getattr(restored, field)
            if isinstance(orig, float) and math.isnan(orig):
                assert math.isnan(rest), field
            else:
                assert orig == rest, field


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class TestRunResult:
    @pytest.fixture()
    def minimal_result(self) -> RunResult:
        return RunResult(
            prompt_id="gsm8k_0",
            prompt_text="Solve: 2+2",
            model="test-model",
            press="none",
            compression_ratio=0.0,
            seed=42,
            generated_text="#### 4",
            ground_truth="4",
            predicted_answer="4",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=10,
            cache_size_after_prefill=None,
            signals=[],
        )

    def test_construction(self, minimal_result: RunResult):
        assert minimal_result.prompt_id == "gsm8k_0"
        assert minimal_result.correct is True
        assert minimal_result.catastrophes == []
        assert minimal_result.catastrophe_onsets == {}

    def test_with_catastrophes(self):
        result = RunResult(
            prompt_id="gsm8k_1",
            prompt_text="Solve: 3+3",
            model="test-model",
            press="streaming_llm",
            compression_ratio=0.875,
            seed=42,
            generated_text="loop loop loop",
            ground_truth="6",
            predicted_answer=None,
            correct=False,
            stop_reason="max_tokens",
            catastrophes=["looping", "non_termination"],
            num_tokens_generated=512,
            cache_size_after_prefill=100,
            catastrophe_onsets={"looping": 50, "non_termination": 511},
            signals=[],
        )
        assert "looping" in result.catastrophes
        assert result.catastrophe_onsets["looping"] == 50

    def test_serialization_roundtrip(self, minimal_result: RunResult):
        json_str = minimal_result.model_dump_json()
        restored = RunResult.model_validate_json(json_str)
        assert restored.prompt_id == minimal_result.prompt_id
        assert restored.correct == minimal_result.correct

    def test_with_signals(self):
        sig = TokenSignals(entropy=1.0, top1_prob=0.5, top5_prob=0.9)
        result = RunResult(
            prompt_id="gsm8k_2",
            prompt_text="Solve: 1+1",
            model="test-model",
            press="none",
            compression_ratio=0.0,
            seed=42,
            generated_text="#### 2",
            ground_truth="2",
            predicted_answer="2",
            correct=True,
            stop_reason="eos",
            catastrophes=[],
            num_tokens_generated=5,
            cache_size_after_prefill=None,
            signals=[sig],
        )
        assert len(result.signals) == 1
        assert result.signals[0].entropy == 1.0
