"""Tests for herald.experiment — pure functions (no GPU required)."""

import json
from pathlib import Path

import pytest

from herald.config import ExperimentConfig, RunResult, TokenSignals
from herald.experiment import (
    _checkpoint_path,
    _clear_checkpoint,
    _load_checkpoint,
    _append_checkpoint,
    build_sweep_configs,
    result_exists,
    summarize,
    SWEEP_METHODS,
    SWEEP_RATIOS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(
    prompt_id: str = "gsm8k_0",
    correct: bool | None = True,
    catastrophes: list[str] | None = None,
    num_tokens: int = 50,
) -> RunResult:
    return RunResult(
        prompt_id=prompt_id,
        prompt_text="Solve: 2+2",
        model="test-model",
        press="none",
        compression_ratio=0.0,
        seed=42,
        generated_text="#### 4",
        ground_truth="4",
        predicted_answer="4",
        correct=correct,
        stop_reason="eos",
        catastrophes=catastrophes or [],
        num_tokens_generated=num_tokens,
        cache_size_after_prefill=None,
        signals=[],
    )


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


class TestSummarize:
    def test_empty_results(self):
        s = summarize([])
        assert s == {"total": 0}

    def test_single_correct(self):
        s = summarize([_make_result(correct=True)])
        assert s["total"] == 1
        assert s["correct"] == 1
        assert s["accuracy"] == 1.0
        assert s["catastrophic_failure_rate"] == 0.0

    def test_single_wrong(self):
        s = summarize([_make_result(correct=False)])
        assert s["correct"] == 0
        assert s["accuracy"] == 0.0

    def test_catastrophe_counting(self):
        results = [
            _make_result("p0", correct=False, catastrophes=["looping", "non_termination"]),
            _make_result("p1", correct=True, catastrophes=[]),
            _make_result("p2", correct=False, catastrophes=["looping"]),
        ]
        s = summarize(results)
        assert s["total"] == 3
        assert s["correct"] == 1
        assert s["catastrophic_failure_rate"] == pytest.approx(2 / 3, rel=0.01)
        assert s["catastrophe_counts"]["looping"] == 2
        assert s["catastrophe_counts"]["non_termination"] == 1

    def test_avg_tokens(self):
        results = [
            _make_result("p0", num_tokens=100),
            _make_result("p1", num_tokens=200),
        ]
        s = summarize(results)
        assert s["avg_tokens"] == 150.0

    def test_none_correct_not_counted(self):
        # correct=None should not count as correct
        s = summarize([_make_result(correct=None)])
        assert s["correct"] == 0


# ---------------------------------------------------------------------------
# _checkpoint_path
# ---------------------------------------------------------------------------


class TestCheckpointPath:
    def test_path_format(self):
        cfg = ExperimentConfig(
            model_name="org/MyModel",
            press_name="snapkv",
            compression_ratio=0.5,
            num_prompts=100,
            output_dir=Path("/tmp/results"),
        )
        path = _checkpoint_path(cfg)
        assert path == Path("/tmp/results/snapkv/MyModel_0.500_100p.ckpt.jsonl")

    def test_different_configs_different_paths(self):
        cfg1 = ExperimentConfig(compression_ratio=0.5)
        cfg2 = ExperimentConfig(compression_ratio=0.75)
        assert _checkpoint_path(cfg1) != _checkpoint_path(cfg2)


# ---------------------------------------------------------------------------
# Checkpoint round-trip
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:
    def test_load_empty(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        results = _load_checkpoint(cfg)
        assert results == []

    def test_append_and_load(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        r1 = _make_result("p0")
        r2 = _make_result("p1")
        _append_checkpoint(r1, cfg)
        _append_checkpoint(r2, cfg)
        loaded = _load_checkpoint(cfg)
        assert len(loaded) == 2
        assert loaded[0].prompt_id == "p0"
        assert loaded[1].prompt_id == "p1"

    def test_clear_checkpoint(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        _append_checkpoint(_make_result(), cfg)
        assert _checkpoint_path(cfg).exists()
        _clear_checkpoint(cfg)
        assert not _checkpoint_path(cfg).exists()

    def test_clear_nonexistent_is_noop(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        _clear_checkpoint(cfg)  # should not raise


# ---------------------------------------------------------------------------
# build_sweep_configs
# ---------------------------------------------------------------------------


class TestBuildSweepConfigs:
    def test_total_configs(self):
        configs = build_sweep_configs()
        # ratio=0.0 gives 1 config (press="none"), each other ratio gives len(SWEEP_METHODS)
        nonzero_ratios = [r for r in SWEEP_RATIOS if r > 0]
        expected = 1 + len(nonzero_ratios) * len(SWEEP_METHODS)
        assert len(configs) == expected

    def test_baseline_has_press_none(self):
        configs = build_sweep_configs()
        baseline = [c for c in configs if c.compression_ratio == 0.0]
        assert len(baseline) == 1
        assert baseline[0].press_name == "none"

    def test_all_methods_present(self):
        configs = build_sweep_configs()
        methods = {c.press_name for c in configs if c.compression_ratio > 0}
        assert methods == set(SWEEP_METHODS)

    def test_custom_params_propagated(self):
        configs = build_sweep_configs(
            num_prompts=200, seed=99, model_name="org/OtherModel"
        )
        for cfg in configs:
            assert cfg.num_prompts == 200
            assert cfg.seed == 99
            assert cfg.model_name == "org/OtherModel"


# ---------------------------------------------------------------------------
# result_exists
# ---------------------------------------------------------------------------


class TestResultExists:
    def test_no_file(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        assert result_exists(cfg) is False

    def test_file_exists(self, tmp_path: Path):
        cfg = ExperimentConfig(output_dir=tmp_path)
        # Create the expected result file
        out_dir = tmp_path / cfg.press_name
        out_dir.mkdir(parents=True)
        model_short = cfg.model_name.split("/")[-1]
        ratio_str = f"{cfg.compression_ratio:.3f}"
        filename = f"{model_short}_{ratio_str}_{cfg.num_prompts}p.json"
        (out_dir / filename).write_text("{}")
        assert result_exists(cfg) is True
