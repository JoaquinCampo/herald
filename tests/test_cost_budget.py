"""Round-trip tests for the Phase 1 cost-budget contract."""

from pathlib import Path

from herald.cost_budget import (
    CellBudget,
    CostBudget,
    dump_cost_budget,
    load_cost_budget,
)


def _budget() -> CostBudget:
    cells = {
        ("gsm8k", "streaming_llm", 0.5): CellBudget(
            task="gsm8k",
            press="streaming_llm",
            compression_ratio=0.5,
            predicted_s_per_token=0.024,
            predicted_n_tokens=312.0,
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
    return CostBudget(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_new_tokens=512,
        cells=cells,
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
