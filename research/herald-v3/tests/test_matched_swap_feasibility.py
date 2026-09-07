"""Tests for deterministic, budget-matched structural swap feasibility."""

import json
import sys
from pathlib import Path

import pytest

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_matched_swap_feasibility as audit  # noqa: E402


def test_matching_maximizes_cardinality_then_distance_and_lex_order() -> None:
    pairs, maximum = audit._match_pairs([2, 4, 8], [1, 3, 7], caliper=4)

    assert maximum == 3
    assert pairs == [(2, 1), (4, 3)]


def test_matching_returns_zero_when_caliper_has_no_edge() -> None:
    pairs, maximum = audit._match_pairs([2], [10], caliper=1)

    assert maximum == 0
    assert pairs == []


def test_generated_positions_are_excluded_from_control_and_donor_pools() -> (
    None
):
    cell = {
        "layer": 0,
        "kv_head": 0,
        "evicted_user_positions": [2],
        "retained_non_user_positions": [0, 1, 5, 6],
    }

    result = audit._audit_cell(
        cell,
        "ifeval_1",
        5,
        7,
        2,
        3,
        3,
        Path("fixture.json"),
    )

    assert result["retained_prompt_non_user_donor_positions"] == [0, 1]
    assert result["excluded_retained_generated_positions"] == [5, 6]
    assert all(pair["donor_position"] < 5 for pair in result["pairs"])
    assert all(pair["control_position"] < 5 for pair in result["pairs"])


def test_donor_assignment_is_one_to_one_and_midpoint_deterministic() -> None:
    result = audit._assign_donors([(2, 4), (7, 9)], [0, 3, 8])

    assert [item["donor_position"] for item in result] == [3, 8]
    assert len({item["donor_position"] for item in result}) == 2


def test_real_upper_bound_output_has_all_112_cells_and_positive_gate() -> (
    None
):
    output = (
        ROOT / "results/retrieval-feasibility/matched-swap-upper-bound.json"
    )
    if not output.is_file():
        pytest.skip("matched swap output is unavailable")
    document = json.loads(output.read_text(encoding="utf-8"))

    assert document["status"] == "completed"
    assert document["interpretation"]["result_is_optimistic_upper_bound"]
    assert document["interpretation"]["uses_scores_or_labels"] is False
    assert document["summary"]["prompt_count"] == 69
    assert document["summary"]["possible_prompt_count"] == 69
    assert document["summary"]["all_112_cells_considered_per_prompt"]
    assert all(
        len(record["cells_considered"]) == 112
        and record["total_swap_count"] >= 8
        and record["selected_cell_count"] >= 4
        for record in document["records"]
    )
