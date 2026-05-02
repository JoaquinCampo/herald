"""CostWatchdog: aborts a cell when wall_clock_per_token drifts above
the predicted budget for `consecutive_breaches` consecutive prompts."""

import math

from herald.config import (
    RunResult,
    compute_prompt_hash,
    make_run_id,
)
from herald.experiment import CostWatchdog


def _result(wpt: float, prompt_id: str = "p0") -> RunResult:
    pt = "x"
    return RunResult(
        run_id=make_run_id(prompt_id, "none", 0.0, 42),
        prompt_id=prompt_id,
        prompt_text=pt,
        prompt_hash=compute_prompt_hash(pt),
        model="m",
        press="none",
        compression_ratio=0.0,
        seed=42,
        baseline_run_id=make_run_id(prompt_id, "none", 0.0, 42),
        generated_text="",
        ground_truth="",
        predicted_answer=None,
        correct=None,
        stop_reason="eos",
        catastrophes=[],
        num_tokens_generated=10,
        signals=[],
        wall_clock_per_token=wpt,
    )


class TestCostWatchdog:
    def test_no_breach_under_threshold(self) -> None:
        wd = CostWatchdog(predicted_s_per_token=0.10, tolerance=1.25)
        for _ in range(5):
            assert wd.observe(_result(0.10)) is False
        assert wd._streak == 0

    def test_isolated_spike_does_not_trigger(self) -> None:
        wd = CostWatchdog(
            predicted_s_per_token=0.10,
            tolerance=1.25,
            consecutive_breaches=3,
        )
        assert wd.observe(_result(0.50)) is False
        assert wd.observe(_result(0.10)) is False
        assert wd._streak == 0

    def test_consecutive_breaches_triggers(self) -> None:
        wd = CostWatchdog(
            predicted_s_per_token=0.10,
            tolerance=1.25,
            consecutive_breaches=3,
        )
        assert wd.observe(_result(0.50)) is False
        assert wd.observe(_result(0.50)) is False
        assert wd.observe(_result(0.50)) is True

    def test_nan_wall_clock_ignored(self) -> None:
        wd = CostWatchdog(predicted_s_per_token=0.10)
        assert wd.observe(_result(float("nan"))) is False
        assert wd.history == []

    def test_history_accumulates_finite_values(self) -> None:
        wd = CostWatchdog(predicted_s_per_token=0.10)
        wd.observe(_result(0.10))
        wd.observe(_result(0.12))
        wd.observe(_result(float("nan")))
        wd.observe(_result(0.09))
        assert wd.history == [0.10, 0.12, 0.09]
        assert all(math.isfinite(x) for x in wd.history)
