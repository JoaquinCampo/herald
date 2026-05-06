"""Round-trip + resolve tests for the Phase 1 cost-budget contract."""

from pathlib import Path

import pytest

from herald.cost_budget import (
    BudgetTaskNotFound,
    CellBudget,
    CostBudget,
    dump_cost_budget,
    load_cost_budget,
)


def _budget(task_fallback: bool = False) -> CostBudget:
    cells = {
        ("gsm8k", "streaming_llm", 0.5): CellBudget(
            task="gsm8k",
            press="streaming_llm",
            compression_ratio=0.5,
            predicted_s_per_token=0.024,
            predicted_n_tokens=312.0,
            n_profile_runs=50,
        ),
        ("gsm8k", "streaming_llm", 0.875): CellBudget(
            task="gsm8k",
            press="streaming_llm",
            compression_ratio=0.875,
            predicted_s_per_token=0.026,
            predicted_n_tokens=240.0,
            n_profile_runs=50,
        ),
        ("humaneval", "snapkv", 0.875): CellBudget(
            task="humaneval",
            press="snapkv",
            compression_ratio=0.875,
            predicted_s_per_token=0.031,
            predicted_n_tokens=180.0,
            n_profile_runs=50,
        ),
    }
    fallbacks: dict[str, CellBudget] = {}
    if task_fallback:
        fallbacks["humaneval"] = CellBudget(
            task="humaneval",
            press="*",
            compression_ratio=-1.0,
            predicted_s_per_token=0.040,
            predicted_n_tokens=200.0,
            n_profile_runs=50,
        )
    return CostBudget(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=512,
        cells=cells,
        task_fallbacks=fallbacks,
    )


class TestRoundTrip:
    def test_dump_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "phase-1-cost-budget.json"
        budget = _budget()
        dump_cost_budget(budget, path)
        loaded = load_cost_budget(path)
        assert loaded.model == budget.model
        assert loaded.max_new_tokens == budget.max_new_tokens
        assert set(loaded.cells.keys()) == set(budget.cells.keys())
        for k, v in budget.cells.items():
            assert loaded.cells[k] == v

    def test_lookup_quantizes_ratio(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        dump_cost_budget(_budget(), path)
        loaded = load_cost_budget(path)
        # tiny float drift on the lookup ratio still hits the cell.
        cell = loaded.lookup("gsm8k", "streaming_llm", 0.5000001)
        assert cell is not None
        assert cell.predicted_s_per_token == 0.024

    def test_missing_cell_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        dump_cost_budget(_budget(), path)
        loaded = load_cost_budget(path)
        assert loaded.lookup("gsm8k", "snapkv", 0.5) is None

    def test_dump_and_load_with_task_fallbacks(self, tmp_path: Path) -> None:
        path = tmp_path / "b.json"
        budget = _budget(task_fallback=True)
        dump_cost_budget(budget, path)
        loaded = load_cost_budget(path)
        assert "humaneval" in loaded.task_fallbacks
        fb = loaded.task_fallbacks["humaneval"]
        assert fb.predicted_s_per_token == 0.040
        assert fb.compression_ratio == -1.0


class TestResolve:
    def test_exact_match(self) -> None:
        b = _budget()
        m = b.resolve("gsm8k", "streaming_llm", 0.5)
        assert m.match_kind == "exact"
        assert m.cell.predicted_s_per_token == 0.024

    def test_nearest_ratio_picks_closest(self) -> None:
        b = _budget()
        # 0.7 is closer to 0.875 than to 0.5
        m = b.resolve("gsm8k", "streaming_llm", 0.7)
        assert m.match_kind == "nearest_ratio"
        assert m.cell.compression_ratio == 0.875
        assert m.requested_ratio == 0.7

    def test_nearest_ratio_tie_break_prefers_lower(self) -> None:
        b = _budget()
        # 0.6875 is equidistant from 0.5 and 0.875; tie -> lower ratio
        m = b.resolve("gsm8k", "streaming_llm", 0.6875)
        assert m.match_kind == "nearest_ratio"
        assert m.cell.compression_ratio == 0.5

    def test_task_fallback_when_press_absent(self) -> None:
        b = _budget(task_fallback=True)
        # No (humaneval, knorm, *) cell exists. With fallback recorded
        # for humaneval, resolve substitutes it.
        m = b.resolve("humaneval", "knorm", 0.5)
        assert m.match_kind == "task_fallback"
        assert m.cell.predicted_s_per_token == 0.040

    def test_press_absent_no_fallback_raises(self) -> None:
        b = _budget(task_fallback=False)
        with pytest.raises(BudgetTaskNotFound):
            b.resolve("humaneval", "knorm", 0.5)

    def test_missing_task_raises(self) -> None:
        b = _budget()
        with pytest.raises(BudgetTaskNotFound):
            b.resolve("ifeval", "streaming_llm", 0.5)

    def test_logs_nearest_ratio_substitution(self) -> None:
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        sink_id = logger.add(sink, level="INFO", format="{message}")
        try:
            _budget().resolve("gsm8k", "streaming_llm", 0.7)
        finally:
            logger.remove(sink_id)
        assert "nearest_ratio substitution" in sink.getvalue()

    def test_logs_task_fallback_substitution(self) -> None:
        from io import StringIO

        from loguru import logger

        sink = StringIO()
        sink_id = logger.add(sink, level="INFO", format="{message}")
        try:
            _budget(task_fallback=True).resolve("humaneval", "knorm", 0.5)
        finally:
            logger.remove(sink_id)
        assert "task_fallback substitution" in sink.getvalue()
