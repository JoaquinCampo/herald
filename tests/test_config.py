"""Tests for herald.config — Pydantic models."""

import hashlib
import math
from pathlib import Path

import pytest

from herald.config import (
    ExperimentConfig,
    GenerationArtifact,
    RunResult,
    TokenSignals,
    compute_prompt_hash,
    make_run_id,
)

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


def _minimal_run_result(**overrides: object) -> RunResult:
    """Build a minimal valid RunResult for testing."""
    prompt_text = "Solve: 2+2"
    prompt_id = "gsm8k_0"
    press = "none"
    compression_ratio = 0.0
    seed = 42
    kwargs: dict[str, object] = dict(
        run_id=make_run_id(prompt_id, press, compression_ratio, seed),
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        prompt_hash=compute_prompt_hash(prompt_text),
        model="test-model",
        press=press,
        compression_ratio=compression_ratio,
        seed=seed,
        baseline_run_id=make_run_id(prompt_id, "none", 0.0, seed),
        generated_text="#### 4",
        ground_truth="4",
        predicted_answer="4",
        correct=True,
        stop_reason="eos",
        catastrophes=[],
        num_tokens_generated=10,
        signals=[],
    )
    kwargs.update(overrides)
    return RunResult(**kwargs)  # type: ignore[arg-type]


class TestRunResult:
    @pytest.fixture()
    def minimal_result(self) -> RunResult:
        return _minimal_run_result()

    def test_construction(self, minimal_result: RunResult):
        assert minimal_result.prompt_id == "gsm8k_0"
        assert minimal_result.correct is True
        assert minimal_result.catastrophes == []
        assert minimal_result.catastrophe_onsets == {}

    def test_with_catastrophes(self):
        result = _minimal_run_result(
            prompt_id="gsm8k_1",
            prompt_text="Solve: 3+3",
            prompt_hash=compute_prompt_hash("Solve: 3+3"),
            run_id=make_run_id("gsm8k_1", "streaming_llm", 0.875, 42),
            press="streaming_llm",
            compression_ratio=0.875,
            baseline_run_id=make_run_id("gsm8k_1", "none", 0.0, 42),
            generated_text="loop loop loop",
            ground_truth="6",
            predicted_answer=None,
            correct=False,
            stop_reason="max_tokens",
            catastrophes=["looping", "non_termination"],
            num_tokens_generated=512,
            catastrophe_onsets={"looping": 50, "non_termination": 511},
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
        result = _minimal_run_result(
            prompt_id="gsm8k_2",
            prompt_text="Solve: 1+1",
            prompt_hash=compute_prompt_hash("Solve: 1+1"),
            run_id=make_run_id("gsm8k_2", "none", 0.0, 42),
            baseline_run_id=make_run_id("gsm8k_2", "none", 0.0, 42),
            generated_text="#### 2",
            ground_truth="2",
            predicted_answer="2",
            num_tokens_generated=5,
            signals=[sig],
        )
        assert len(result.signals) == 1
        assert result.signals[0].entropy == 1.0


# ---------------------------------------------------------------------------
# compute_prompt_hash and make_run_id helpers
# ---------------------------------------------------------------------------


def test_compute_prompt_hash_is_sha256_hex():
    h = compute_prompt_hash("hello world")
    assert h == hashlib.sha256(b"hello world").hexdigest()
    assert len(h) == 64


def test_make_run_id_deterministic():
    a = make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.875, seed=42
    )
    b = make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.875, seed=42
    )
    assert a == b
    assert a != make_run_id(
        prompt_id="p1", press="snapkv", compression_ratio=0.5, seed=42
    )


def test_run_result_baseline_self_link_default():
    rr = RunResult(
        run_id="r1",
        prompt_id="p1",
        prompt_text="Q",
        prompt_hash=compute_prompt_hash("Q"),
        model="x",
        press="none",
        compression_ratio=0.0,
        seed=42,
        max_new_tokens=10,
        decoding_config={"do_sample": False},
        task="gsm8k",
        baseline_run_id="r1",
        generated_text="A",
        generated_token_ids=[1, 2, 3],
        ground_truth="A",
        predicted_answer="A",
        correct=True,
        stop_reason="eos",
        catastrophes=[],
        num_tokens_generated=3,
        signals=[],
        replay_status="ok",
        herald_git_sha="abc",
    )
    assert rr.baseline_run_id == rr.run_id


def test_generation_artifact_holds_required_fields():
    import torch

    art = GenerationArtifact(
        run_id="r1",
        input_ids=torch.zeros(1, 5, dtype=torch.long),
        input_len=5,
        generated_token_ids=[10, 11],
        compressed_scores=[torch.zeros(100), torch.zeros(100)],
    )
    assert art.input_len == 5
    assert len(art.compressed_scores) == 2
