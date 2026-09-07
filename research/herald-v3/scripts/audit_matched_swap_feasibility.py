#!/usr/bin/env python3
"""Audit label-blind matched swap feasibility for the fixed 0.25 arm.

The audit consumes the exact user-span report and saved Knorm indices.  It
uses only prompt positions for the control and donor pools, so generated
context cannot make a prompt-action cell look feasible.  Every layer and KV
head is considered as an optimistic upper bound.  No retrieval head is
selected and no outcome or score field is read.
"""

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import hashlib
import itertools
import json
import math
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3.engineering import runner


SCHEMA_VERSION = "herald_v3.matched_swap_feasibility.v1"
SPAN_AUDIT_SCHEMA = "herald_v3.user_span_feasibility.v1"
ACTION_ID = "knorm:0.25"
PROMPT_COUNT = 69
HEAD_COUNT = 112
LAYER_COUNT = 28
KV_HEAD_COUNT = 4
TOP_HEAD_COUNT = 8
MIN_SWAP_COUNT = 8
MIN_SELECTED_CELLS = 4
MAX_PAIRS_PER_CELL = 2
CALIPER_FRACTION = 0.10


class MatchedSwapFeasibilityError(RuntimeError):
    """Raised when the structural inputs cannot support this audit."""


def audit_matched_swap_feasibility(
    span_audit_path: str | Path,
    proposal_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Run the fixed-caliper matched swap feasibility audit."""
    span_file = Path(span_audit_path).resolve()
    proposal_file = Path(proposal_path).resolve()
    output_file = Path(output_path).resolve()
    span_audit = _read_mapping(span_file)
    _validate_span_audit(span_audit, span_file)
    if not proposal_file.is_file():
        raise MatchedSwapFeasibilityError(
            f"diagnostic proposal is missing: {proposal_file}"
        )
    records = _mapping_list(span_audit.get("records"), "records", span_file)
    by_action: dict[str, list[dict[str, object]]] = {ACTION_ID: []}
    for record in records:
        result = _audit_record(record, span_file)
        by_action[ACTION_ID].append(result)
    summary = _summarize(by_action[ACTION_ID])
    try:
        import scipy
    except ImportError as error:
        raise MatchedSwapFeasibilityError(
            "scipy is required for maximum bipartite matching"
        ) from error
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": {
            "action_id": ACTION_ID,
            "ratio": 0.25,
            "target": "evicted_user_content_tokens",
            "control_pool": "prompt_non_user_positions_only",
            "donor_pool": "prompt_non_user_positions_retained_by_knorm_only",
            "generated_context_allowed_as_donor": False,
            "record_count": len(records),
            "heads_considered_per_record": HEAD_COUNT,
            "layers": LAYER_COUNT,
            "kv_heads_per_layer": KV_HEAD_COUNT,
            "top_heads_for_upper_bound": TOP_HEAD_COUNT,
            "caliper_fraction": CALIPER_FRACTION,
            "minimum_total_swaps": MIN_SWAP_COUNT,
            "minimum_selected_cells": MIN_SELECTED_CELLS,
            "maximum_pairs_per_cell": MAX_PAIRS_PER_CELL,
        },
        "provenance": {
            "span_audit": str(span_file),
            "span_audit_sha256": _sha256(span_file),
            "proposal": str(proposal_file),
            "proposal_sha256": _sha256(proposal_file),
            "matching_primitive": (
                "scipy.sparse.csgraph.maximum_bipartite_matching"
            ),
            "scipy_version": scipy.__version__,
            "python": platform.python_version(),
            "runner_source_sha256": _sha256(Path(runner.__file__).resolve()),
        },
        "interpretation": {
            "head_selection": "none",
            "result_is_optimistic_upper_bound": True,
            "uses_scores_or_labels": False,
            "uses_retrieval_discovery": False,
            "selected_cells_are_only_hypothetical_top8": True,
        },
        "summary": summary,
        "records": sorted(
            by_action[ACTION_ID],
            key=lambda item: cast(str, item["prompt_id"]),
        ),
    }
    _write_json(output_file, result)
    return result


def _validate_span_audit(document: Mapping[str, object], path: Path) -> None:
    if document.get("schema_version") != SPAN_AUDIT_SCHEMA:
        raise MatchedSwapFeasibilityError(
            f"span audit schema differs: {path}"
        )
    if document.get("status") != "completed":
        raise MatchedSwapFeasibilityError(
            f"span audit is not complete: {path}"
        )
    scope = _mapping(document.get("scope"), "scope", path)
    if scope.get("target") != "user_content_tokens":
        raise MatchedSwapFeasibilityError(
            "span audit target is not user content tokens"
        )
    if scope.get("raw_record_count") != PROMPT_COUNT:
        raise MatchedSwapFeasibilityError(
            f"span audit has {scope.get('raw_record_count')} records, "
            f"expected {PROMPT_COUNT}"
        )


def _audit_record(
    record: Mapping[str, object], span_file: Path
) -> dict[str, object]:
    prompt_id = record.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise MatchedSwapFeasibilityError(
            f"record prompt ID is malformed: {span_file}"
        )
    prompt_length = _positive_int(
        record.get("prompt_length"), "prompt_length", span_file
    )
    cache_length = _positive_int(
        record.get("boundary_cache_length"),
        "boundary_cache_length",
        span_file,
    )
    if cache_length <= prompt_length:
        raise MatchedSwapFeasibilityError(
            f"cache does not contain generated context: {prompt_id}"
        )
    span = _mapping(record.get("user_span"), "user_span", span_file)
    user_start = _nonnegative_int(
        span.get("content_token_start"),
        "user_span.content_token_start",
        span_file,
    )
    user_end = _positive_int(
        span.get("content_token_end"),
        "user_span.content_token_end",
        span_file,
    )
    if user_start >= user_end or user_end > prompt_length:
        raise MatchedSwapFeasibilityError(
            f"user span is outside prompt: {prompt_id}"
        )
    action_container = _mapping(record.get("actions"), "actions", span_file)
    action = _mapping(action_container.get(ACTION_ID), ACTION_ID, span_file)
    cells = _mapping_list(
        action.get("head_records"), "head_records", span_file
    )
    if len(cells) != HEAD_COUNT:
        raise MatchedSwapFeasibilityError(
            f"{prompt_id} has {len(cells)} heads, expected {HEAD_COUNT}"
        )
    caliper = math.ceil(CALIPER_FRACTION * cache_length)
    audited_cells: list[dict[str, object]] = []
    seen_cells: set[tuple[int, int]] = set()
    for cell in cells:
        audited_cells.append(
            _audit_cell(
                cell,
                prompt_id,
                prompt_length,
                cache_length,
                user_start,
                user_end,
                caliper,
                span_file,
            )
        )
        key = (
            cast(int, audited_cells[-1]["layer"]),
            cast(int, audited_cells[-1]["kv_head"]),
        )
        if key in seen_cells:
            raise MatchedSwapFeasibilityError(
                f"duplicate layer/KV head cell: {prompt_id}/{key}"
            )
        seen_cells.add(key)
    expected_cells = {
        (layer, kv_head)
        for layer in range(LAYER_COUNT)
        for kv_head in range(KV_HEAD_COUNT)
    }
    if seen_cells != expected_cells:
        raise MatchedSwapFeasibilityError(
            f"layer/KV head roster differs from 28x4 grid: {prompt_id}"
        )
    selected = sorted(
        audited_cells,
        key=lambda cell: (
            -cast(int, cell["swap_count"]),
            cast(int, cell["layer"]),
            cast(int, cell["kv_head"]),
        ),
    )[:TOP_HEAD_COUNT]
    for cell in audited_cells:
        cell["selected_for_upper_bound"] = cell in selected
    total_swaps = sum(cast(int, cell["swap_count"]) for cell in selected)
    selected_cell_count = sum(
        cast(int, cell["swap_count"]) > 0 for cell in selected
    )
    eligible = (
        total_swaps >= MIN_SWAP_COUNT
        and selected_cell_count >= MIN_SELECTED_CELLS
    )
    return {
        "prompt_id": prompt_id,
        "prompt_length": prompt_length,
        "boundary_cache_length": cache_length,
        "caliper": caliper,
        "cells_considered": audited_cells,
        "selected_cells": selected,
        "selected_cell_count": selected_cell_count,
        "total_swap_count": total_swaps,
        "eligible_under_ideal_top8": eligible,
        "exclusion_reason": None
        if eligible
        else _exclusion_reason(total_swaps, selected_cell_count),
    }


def _audit_cell(
    cell: Mapping[str, object],
    prompt_id: str,
    prompt_length: int,
    cache_length: int,
    user_start: int,
    user_end: int,
    caliper: int,
    span_file: Path,
) -> dict[str, object]:
    layer = _nonnegative_int(cell.get("layer"), "layer", span_file)
    kv_head = _nonnegative_int(cell.get("kv_head"), "kv_head", span_file)
    evicted_user = _integer_list(
        cell.get("evicted_user_positions"),
        "evicted_user_positions",
        span_file,
    )
    retained_non_user = _integer_list(
        cell.get("retained_non_user_positions"),
        "retained_non_user_positions",
        span_file,
    )
    user_positions = set(range(user_start, user_end))
    if any(position not in user_positions for position in evicted_user):
        raise MatchedSwapFeasibilityError(
            "evicted user position is outside span: "
            f"{prompt_id}/{layer}/{kv_head}"
        )
    if len(set(evicted_user)) != len(evicted_user):
        raise MatchedSwapFeasibilityError(
            f"evicted user positions repeat: {prompt_id}/{layer}/{kv_head}"
        )
    if any(position in user_positions for position in retained_non_user):
        raise MatchedSwapFeasibilityError(
            f"retained donor pool contains user position: "
            f"{prompt_id}/{layer}/{kv_head}"
        )
    if any(
        position < 0 or position >= cache_length
        for position in retained_non_user
    ):
        raise MatchedSwapFeasibilityError(
            "retained donor position is outside cache: "
            f"{prompt_id}/{layer}/{kv_head}"
        )
    if len(set(retained_non_user)) != len(retained_non_user):
        raise MatchedSwapFeasibilityError(
            f"retained donor positions repeat: {prompt_id}/{layer}/{kv_head}"
        )
    prompt_non_user = set(range(prompt_length)) - user_positions
    prompt_donors = sorted(
        position for position in retained_non_user if position < prompt_length
    )
    generated_donors_excluded = sorted(
        position
        for position in retained_non_user
        if position >= prompt_length
    )
    donor_set = set(prompt_donors)
    control_evicted = sorted(prompt_non_user - donor_set)
    pairs, maximum_cardinality = _match_pairs(
        sorted(evicted_user), control_evicted, caliper
    )
    pair_records = _assign_donors(pairs, prompt_donors)
    if len(pair_records) != min(MAX_PAIRS_PER_CELL, maximum_cardinality):
        raise MatchedSwapFeasibilityError(
            f"matching cardinality changed during pair selection: "
            f"{prompt_id}/{layer}/{kv_head}"
        )
    return {
        "layer": layer,
        "kv_head": kv_head,
        "caliper": caliper,
        "evicted_user_positions": sorted(evicted_user),
        "control_evicted_prompt_count": len(control_evicted),
        "retained_prompt_non_user_donor_positions": prompt_donors,
        "excluded_retained_generated_positions": generated_donors_excluded,
        "maximum_matching_cardinality": maximum_cardinality,
        "pair_count": len(pair_records),
        "swap_count": len(pair_records),
        "pair_shortfall_reason": (
            "no_caliper_valid_pair"
            if maximum_cardinality == 0
            else "maximum_matching_below_two_pair_cap"
            if maximum_cardinality < MAX_PAIRS_PER_CELL
            else None
        ),
        "pairs": pair_records,
    }


def _match_pairs(
    user_positions: list[int],
    control_positions: list[int],
    caliper: int,
) -> tuple[list[tuple[int, int]], int]:
    edges = [
        (user, control)
        for user in user_positions
        for control in control_positions
        if abs(user - control) <= caliper
    ]
    maximum_cardinality = _maximum_matching_cardinality(
        user_positions, control_positions, edges
    )
    target = min(MAX_PAIRS_PER_CELL, maximum_cardinality)
    if target == 0:
        return [], maximum_cardinality
    if target == 1:
        return [
            min(edges, key=lambda pair: (abs(pair[0] - pair[1]), pair))
        ], maximum_cardinality
    best_key: tuple[int, tuple[tuple[int, int], ...]] | None = None
    best_pairs: list[tuple[int, int]] = []
    for first, second in itertools.combinations(edges, 2):
        if first[0] == second[0] or first[1] == second[1]:
            continue
        pairs = tuple(sorted((first, second)))
        key = (
            sum(abs(user - control) for user, control in pairs),
            pairs,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_pairs = list(pairs)
    if len(best_pairs) != 2:
        raise MatchedSwapFeasibilityError(
            "maximum matching reported two pairs but pair selection "
            "found fewer"
        )
    return best_pairs, maximum_cardinality


def _maximum_matching_cardinality(
    user_positions: list[int],
    control_positions: list[int],
    edges: list[tuple[int, int]],
) -> int:
    if not user_positions or not control_positions or not edges:
        return 0
    try:
        import numpy as np
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import maximum_bipartite_matching
    except ImportError as error:
        raise MatchedSwapFeasibilityError(
            "numpy and scipy are required for maximum bipartite matching"
        ) from error
    user_index = {
        position: index for index, position in enumerate(user_positions)
    }
    control_index = {
        position: index for index, position in enumerate(control_positions)
    }
    rows = [user_index[user] for user, _ in edges]
    columns = [control_index[control] for _, control in edges]
    graph = csr_matrix(
        (np.ones(len(edges), dtype=np.int8), (rows, columns)),
        shape=(len(user_positions), len(control_positions)),
    )
    matching = maximum_bipartite_matching(graph, perm_type="column")
    return int(sum(int(index >= 0) for index in matching))


def _assign_donors(
    pairs: list[tuple[int, int]], donors: list[int]
) -> list[dict[str, int]]:
    available = set(donors)
    result: list[dict[str, int]] = []
    for user, control in sorted(pairs):
        midpoint = int(round((user + control) / 2))
        if not available:
            raise MatchedSwapFeasibilityError(
                "matched pair donor pool is empty"
            )
        donor = min(
            available,
            key=lambda position: (abs(position - midpoint), position),
        )
        available.remove(donor)
        result.append(
            {
                "user_position": user,
                "control_position": control,
                "source_distance": abs(user - control),
                "rounded_midpoint": midpoint,
                "donor_position": donor,
                "donor_midpoint_distance": abs(donor - midpoint),
            }
        )
    return result


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    selected_cells = [
        cell
        for record in records
        for cell in cast(list[object], record["selected_cells"])
        if isinstance(cell, Mapping)
    ]
    pairs = [
        pair
        for cell in selected_cells
        for pair in cast(list[object], cell["pairs"])
        if isinstance(pair, Mapping)
    ]
    total_swaps = [
        cast(int, record["total_swap_count"]) for record in records
    ]
    selected_counts = [
        cast(int, record["selected_cell_count"]) for record in records
    ]
    eligible = [
        cast(bool, record["eligible_under_ideal_top8"]) for record in records
    ]
    source_distances = [cast(int, pair["source_distance"]) for pair in pairs]
    donor_distances = [
        cast(int, pair["donor_midpoint_distance"]) for pair in pairs
    ]
    cell_reasons = [
        cell.get("pair_shortfall_reason")
        for record in records
        for cell in cast(list[object], record["cells_considered"])
        if isinstance(cell, Mapping)
    ]
    return {
        "prompt_count": len(records),
        "possible_prompt_count": sum(eligible),
        "impossible_prompt_count": sum(not item for item in eligible),
        "all_112_cells_considered_per_prompt": all(
            len(cast(list[object], record["cells_considered"])) == HEAD_COUNT
            for record in records
        ),
        "eligibility_rule": (
            f"at least {MIN_SWAP_COUNT} swaps across at least "
            f"{MIN_SELECTED_CELLS} hypothetical top-{TOP_HEAD_COUNT} cells"
        ),
        "total_swap_count": _integer_summary(total_swaps),
        "selected_cell_count": _integer_summary(selected_counts),
        "pair_shortfall_cells": {
            "no_caliper_valid_pair": cell_reasons.count(
                "no_caliper_valid_pair"
            ),
            "maximum_matching_below_two_pair_cap": cell_reasons.count(
                "maximum_matching_below_two_pair_cap"
            ),
        },
        "source_distance": _integer_summary(source_distances),
        "donor_midpoint_distance": _integer_summary(donor_distances),
        "action": {
            ACTION_ID: {
                "possible_prompt_count": sum(eligible),
                "all_prompts_possible_under_ideal_selection": all(eligible),
                "total_selected_pairs": len(pairs),
            }
        },
    }


def _integer_summary(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def _exclusion_reason(total_swaps: int, selected_cells: int) -> str:
    if total_swaps < MIN_SWAP_COUNT and selected_cells < MIN_SELECTED_CELLS:
        return "below_swap_and_selected_cell_floors"
    if total_swaps < MIN_SWAP_COUNT:
        return "below_swap_floor"
    return "below_selected_cell_floor"


def _mapping_list(
    value: object, name: str, path: Path
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise MatchedSwapFeasibilityError(f"{name} is malformed: {path}")
    return [cast(Mapping[str, object], item) for item in value]


def _mapping(value: object, name: str, path: Path) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MatchedSwapFeasibilityError(f"{name} is malformed: {path}")
    return cast(Mapping[str, object], value)


def _integer_list(value: object, name: str, path: Path) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise MatchedSwapFeasibilityError(f"{name} is malformed: {path}")
    return [int(item) for item in value]


def _positive_int(value: object, name: str, path: Path) -> int:
    result = _nonnegative_int(value, name, path)
    if result <= 0:
        raise MatchedSwapFeasibilityError(f"{name} must be positive: {path}")
    return result


def _nonnegative_int(value: object, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MatchedSwapFeasibilityError(f"{name} is malformed: {path}")
    return int(value)


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MatchedSwapFeasibilityError(
            f"cannot read JSON: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise MatchedSwapFeasibilityError(f"JSON object required: {path}")
    return {str(key): item for key, item in value.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--span-audit", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_matched_swap_feasibility(
            args.span_audit,
            args.proposal,
            args.output,
        )
    except (
        MatchedSwapFeasibilityError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {"status": result["status"], "output": str(args.output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
