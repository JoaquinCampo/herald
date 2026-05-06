"""Phase 1 Block 3 sweep: planning, dry-run, watchdog-trip simulation.

The sweep script (`scripts/phase1_sweep.py`) is the orchestrator;
these tests exercise the lightweight pieces that do not require a
GPU model load. The end-to-end smoke is run separately on Orion as
part of the Block 3 acceptance gates.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from herald.cost_budget import (
    CellBudget,
    CostBudget,
    dump_cost_budget,
)
from herald.experiment import CostWatchdog

REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP_SCRIPT = REPO_ROOT / "scripts" / "phase1_sweep.py"


def _budget_with_qasper(path: Path, *, all_tasks: bool = True) -> Path:
    cells = {
        ("gsm8k", "streaming_llm", 0.5): CellBudget(
            "gsm8k", "streaming_llm", 0.5, 0.014, 327.0, 50
        ),
        ("gsm8k", "streaming_llm", 0.875): CellBudget(
            "gsm8k", "streaming_llm", 0.875, 0.014, 226.0, 50
        ),
        ("gsm8k", "streaming_llm", 0.9375): CellBudget(
            "gsm8k", "streaming_llm", 0.9375, 0.014, 426.0, 50
        ),
        ("longbench_single", "streaming_llm", 0.5): CellBudget(
            "longbench_single", "streaming_llm", 0.5, 0.019, 61.0, 50
        ),
    }
    fallbacks: dict[str, CellBudget] = {
        "gsm8k": CellBudget("gsm8k", "*", -1.0, 0.020, 300.0, 50),
        "longbench_single": CellBudget(
            "longbench_single", "*", -1.0, 0.025, 60.0, 50
        ),
    }
    if all_tasks:
        fallbacks["humaneval"] = CellBudget(
            "humaneval", "*", -1.0, 0.014, 70.0, 50
        )
        fallbacks["ifeval"] = CellBudget(
            "ifeval", "*", -1.0, 0.014, 280.0, 50
        )
    budget = CostBudget(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=512,
        cells=cells,
        task_fallbacks=fallbacks,
    )
    dump_cost_budget(budget, path)
    raw = json.loads(path.read_text())
    raw["longbench_subtask"] = "qasper"
    path.write_text(json.dumps(raw, indent=2))
    return path


class TestDryRun:
    def test_dry_run_full_block3_grid_prints_172_cells(
        self, tmp_path: Path
    ) -> None:
        # Hard requirement from the Block 3 launch packet: 4 tasks ×
        # 6 presses × 7 ratios + 4 baselines = 172 cells, 34 400 runs.
        budget_path = _budget_with_qasper(tmp_path / "budget.json")
        result = subprocess.run(
            [
                sys.executable,
                str(SWEEP_SCRIPT),
                "--tasks",
                "gsm8k,humaneval,ifeval,longbench_single",
                "--presses",
                "streaming_llm,snapkv,knorm,expected_attention,tova,random",
                "--ratios",
                "0.25,0.375,0.5,0.75,0.875,0.9375,0.96875",
                "--include-baseline",
                "--num-prompts",
                "200",
                "--cost-budget",
                str(budget_path),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        out = result.stdout
        assert "172 cells" in out
        assert "34400 runs" in out
        assert "qasper" in out
        # narrativeqa should not surface anywhere in the plan output
        assert "narrativeqa" not in out

    def test_dry_run_logs_nearest_ratio_for_unprofiled_ratio(
        self, tmp_path: Path
    ) -> None:
        budget_path = _budget_with_qasper(tmp_path / "budget.json")
        result = subprocess.run(
            [
                sys.executable,
                str(SWEEP_SCRIPT),
                "--tasks",
                "gsm8k",
                "--presses",
                "streaming_llm",
                # 0.25 is not in the budget; nearest is 0.5
                "--ratios",
                "0.25",
                "--cost-budget",
                str(budget_path),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        # Substitution log goes to loguru -> stderr
        assert "nearest_ratio substitution" in result.stderr
        assert "nearest_ratio" in result.stdout

    def test_dry_run_uses_task_fallback_for_unprofiled_press(
        self, tmp_path: Path
    ) -> None:
        budget_path = _budget_with_qasper(tmp_path / "budget.json")
        result = subprocess.run(
            [
                sys.executable,
                str(SWEEP_SCRIPT),
                "--tasks",
                "gsm8k",
                "--presses",
                "snapkv",
                "--ratios",
                "0.5",
                "--cost-budget",
                str(budget_path),
                "--dry-run",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert "task_fallback substitution" in result.stderr
        assert "task_fallback" in result.stdout

    def test_dry_run_hard_fails_on_missing_task(self, tmp_path: Path) -> None:
        # Budget without humaneval/ifeval coverage; planning humaneval
        # must surface as a non-zero exit + an explicit error.
        budget_path = _budget_with_qasper(
            tmp_path / "budget.json", all_tasks=False
        )
        result = subprocess.run(
            [
                sys.executable,
                str(SWEEP_SCRIPT),
                "--tasks",
                "humaneval",
                "--presses",
                "streaming_llm",
                "--ratios",
                "0.5",
                "--cost-budget",
                str(budget_path),
                "--dry-run",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "BudgetTaskNotFound" in result.stdout


class TestWatchdogForceTrip:
    """The sweep script wires CostWatchdog with a configurable
    min_observations gate. A simulated stream of overshooting wpts
    must trip the watchdog at the documented prompt count."""

    def test_simulated_drift_trips_at_expected_prompt(self) -> None:
        wd = CostWatchdog(
            predicted_s_per_token=0.020,
            tolerance=1.25,
            consecutive_breaches=3,
            min_observations_before_trip=3,
        )
        # First 2 prompts are clean: do not trip.
        assert wd.observe_wpt(0.018) is False
        assert wd.observe_wpt(0.018) is False
        # Then 3 consecutive overshooting prompts: trips on the third.
        assert wd.observe_wpt(0.030) is False
        assert wd.observe_wpt(0.030) is False
        assert wd.observe_wpt(0.030) is True

    def test_min_observations_blocks_first_3_prompts(self) -> None:
        # Same overshoot pattern but only 3 total observations:
        # min_observations gate blocks the trip.
        wd = CostWatchdog(
            predicted_s_per_token=0.020,
            tolerance=1.25,
            consecutive_breaches=3,
            min_observations_before_trip=5,
        )
        assert wd.observe_wpt(0.030) is False
        assert wd.observe_wpt(0.030) is False
        assert wd.observe_wpt(0.030) is False
        assert wd.observe_wpt(0.030) is False
        # 5th: gate met, streak met -> trip
        assert wd.observe_wpt(0.030) is True


@pytest.fixture
def smoke_sweep_layout(tmp_path: Path) -> tuple[Path, Path]:
    budget = _budget_with_qasper(tmp_path / "budget.json")
    out = tmp_path / "results"
    return budget, out
