#!/usr/bin/env python3
"""Audit whether saved Knorm indices permit a user-span rescue diagnostic.

This is a CPU-only structural audit.  It re-tokenizes each saved prompt with
the pinned chat template, checks the complete token sequence against the
saved boundary, and counts the user-content entries evicted by each saved
Knorm arm.  The donor pool contains retained cache entries outside the user
content.  A one-for-one swap can therefore restore at most
``min(evicted_user, retained_non_user)`` entries.

The audit never loads a model, scores a response, or interprets output text.
"""

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3.engineering.prompts import (
    EngineeringPrompt,
    load_prompt_manifest,
)


SCHEMA_VERSION = "herald_v3.user_span_feasibility.v1"
EXPECTED_TOKENIZER_ASSETS = {
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
ACTION_IDS = ("knorm:0.25", "knorm:0.5")
DECISION_TOKENS = 32
RAW_RECORD_COUNT = 69
TEST_PROMPT_COUNT = 76
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MESSAGE_RE = re.compile(
    r"<\|im_start\|>(?P<role>system|user|assistant)\n"
    r"(?P<content>.*?)<\|im_end\|>",
    re.DOTALL,
)


class FeasibilityAuditError(RuntimeError):
    """Raised when the saved evidence cannot support an exact audit."""


def audit_user_span_feasibility(
    tokenizer_path: str | Path,
    tokenizer_identity_path: str | Path,
    test_manifest_path: str | Path,
    raw_root: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Run the exact tokenizer and saved-index feasibility audit."""
    tokenizer_dir = Path(tokenizer_path).resolve()
    identity_file = Path(tokenizer_identity_path).resolve()
    manifest_file = Path(test_manifest_path).resolve()
    raw_directory = Path(raw_root).resolve()
    output_file = Path(output_path).resolve()

    identity = _read_mapping(identity_file)
    asset_hashes = _verify_tokenizer_assets(tokenizer_dir, identity)
    tokenizer = _load_tokenizer(tokenizer_dir)
    tokenizer_metadata = _verify_tokenizer_identity(tokenizer, identity)
    manifest = load_prompt_manifest(manifest_file, limit=TEST_PROMPT_COUNT)
    if len(manifest.prompts) != TEST_PROMPT_COUNT:
        raise FeasibilityAuditError(
            f"test manifest has {len(manifest.prompts)} prompts, expected "
            f"{TEST_PROMPT_COUNT}"
        )
    raw_files = _raw_run_files(raw_directory)
    if len(raw_files) != RAW_RECORD_COUNT:
        raise FeasibilityAuditError(
            f"raw root has {len(raw_files)} run.json files, expected "
            f"{RAW_RECORD_COUNT}"
        )
    prompt_by_id = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_file in raw_files:
        record = _audit_raw_run(
            raw_file,
            tokenizer,
            prompt_by_id,
        )
        prompt_id = cast(str, record["prompt_id"])
        if prompt_id in seen:
            raise FeasibilityAuditError(
                f"duplicate raw record for prompt {prompt_id}"
            )
        seen.add(prompt_id)
        records.append(record)
    missing = sorted(set(prompt_by_id) - seen)
    unexpected = sorted(seen - set(prompt_by_id))
    if unexpected or len(missing) != TEST_PROMPT_COUNT - RAW_RECORD_COUNT:
        raise FeasibilityAuditError(
            f"raw prompt coverage mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    records.sort(key=lambda item: cast(str, item["prompt_id"]))
    summary = _summarize(records)
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "scope": {
            "test_manifest": str(manifest_file),
            "test_manifest_sha256": _sha256(manifest_file),
            "test_prompt_count": len(manifest.prompts),
            "raw_root": str(raw_directory),
            "raw_record_count": len(records),
            "unmaterialized_prompt_ids": missing,
            "target": "user_content_tokens",
            "decision_tokens": DECISION_TOKENS,
        },
        "tokenizer": {
            "source_directory": str(tokenizer_dir),
            "identity_file": str(identity_file),
            "identity_sha256": _sha256(identity_file),
            "name_or_path": identity["name_or_path"],
            "revision": identity["revision"],
            "chat_template_sha256": tokenizer_metadata[
                "chat_template_sha256"
            ],
            "files_sha256": asset_hashes,
            "assets_verified": True,
            "chat_template_verified": True,
        },
        "acceptance": {
            "exact_prompt_rendering": True,
            "exact_full_token_ids": True,
            "exact_offset_round_trip": True,
            "all_raw_records_verified": len(records) == RAW_RECORD_COUNT,
            "all_user_restorations_feasible": summary[
                "all_user_restorations_feasible"
            ],
            "bounded_equal_budget_swaps_available": summary[
                "bounded_equal_budget_swaps_available"
            ],
        },
        "summary": summary,
        "records": records,
    }
    _write_json(output_file, result)
    return result


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise FeasibilityAuditError(
            "transformers is required to load the pinned tokenizer"
        ) from error
    try:
        return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    except Exception as error:
        raise FeasibilityAuditError(
            f"cannot load pinned tokenizer from {path}: {error}"
        ) from error


def _verify_tokenizer_assets(
    tokenizer_dir: Path, identity: Mapping[str, object]
) -> dict[str, str]:
    expected = identity.get("files_sha256")
    if (
        not isinstance(expected, Mapping)
        or set(expected) != EXPECTED_TOKENIZER_ASSETS
    ):
        raise FeasibilityAuditError(
            "tokenizer identity must list exactly the four pinned assets"
        )
    observed: dict[str, str] = {}
    for name in sorted(EXPECTED_TOKENIZER_ASSETS):
        digest = expected[name]
        if (
            not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise FeasibilityAuditError(
                f"tokenizer identity hash is malformed for {name}"
            )
        path = tokenizer_dir / name
        if not path.is_file():
            raise FeasibilityAuditError(
                f"pinned tokenizer asset is missing: {path}"
            )
        observed[name] = _sha256(path)
        if observed[name] != digest:
            raise FeasibilityAuditError(
                f"pinned tokenizer asset hash differs for {name}"
            )
    return observed


def _verify_tokenizer_identity(
    tokenizer: Any, identity: Mapping[str, object]
) -> dict[str, str]:
    expected_template = identity.get("chat_template_sha256")
    expected_revision = identity.get("revision")
    expected_name = identity.get("name_or_path")
    if (
        not isinstance(expected_template, str)
        or _SHA256_RE.fullmatch(expected_template) is None
        or not isinstance(expected_revision, str)
        or not expected_revision
        or not isinstance(expected_name, str)
        or not expected_name
    ):
        raise FeasibilityAuditError(
            "tokenizer identity metadata is malformed"
        )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        raise FeasibilityAuditError(
            "pinned tokenizer has no string chat template"
        )
    observed_template = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if observed_template != expected_template:
        raise FeasibilityAuditError(
            "chat template hash differs from identity"
        )
    return {
        "chat_template_sha256": observed_template,
        "revision": expected_revision,
        "name_or_path": expected_name,
    }


def _raw_run_files(raw_root: Path) -> list[Path]:
    if not raw_root.is_dir():
        raise FeasibilityAuditError(
            f"raw root is not a directory: {raw_root}"
        )
    files = sorted(raw_root.glob("*/run.json"))
    if any(not path.is_file() for path in files):
        raise FeasibilityAuditError("raw run path is not a regular file")
    return files


def _audit_raw_run(
    raw_file: Path,
    tokenizer: Any,
    prompt_by_id: Mapping[str, EngineeringPrompt],
) -> dict[str, object]:
    document = _read_mapping(raw_file)
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise FeasibilityAuditError(
            f"raw result shape is invalid: {raw_file}"
        )
    result = _mapping(results[0], "result", raw_file)
    prompt_value = result.get("prompt")
    if not isinstance(prompt_value, Mapping):
        raise FeasibilityAuditError(f"raw prompt is missing: {raw_file}")
    prompt_id = prompt_value.get("prompt_id")
    if not isinstance(prompt_id, str) or prompt_id not in prompt_by_id:
        raise FeasibilityAuditError(
            f"raw prompt ID is outside the test manifest: {raw_file}"
        )
    prompt = prompt_by_id[prompt_id]
    if result.get("status") != "accepted":
        raise FeasibilityAuditError(f"raw run is not accepted: {raw_file}")
    if prompt_value != prompt.to_dict():
        raise FeasibilityAuditError(
            f"raw prompt provenance differs from manifest: {prompt_id}"
        )
    tokenization = _mapping(
        result.get("tokenization"), "tokenization", raw_file
    )
    saved_ids = _integer_list(
        tokenization.get("input_ids"), "tokenization.input_ids", raw_file
    )
    if tokenization.get("chat_template_verified") is not True:
        raise FeasibilityAuditError(
            f"raw chat-template check is missing: {prompt_id}"
        )
    if tokenization.get("input_length") != len(saved_ids):
        raise FeasibilityAuditError(
            f"raw tokenization length differs: {prompt_id}"
        )
    acceptance = _mapping(result.get("acceptance"), "acceptance", raw_file)
    boundary = _mapping(acceptance.get("boundary"), "boundary", raw_file)
    boundary_ids = _integer_list(
        boundary.get("prompt_token_ids"),
        "boundary.prompt_token_ids",
        raw_file,
    )
    if boundary_ids != saved_ids:
        raise FeasibilityAuditError(
            f"saved boundary and tokenization IDs differ: {prompt_id}"
        )
    span = _verify_prompt_tokens(tokenizer, prompt, saved_ids)
    _validate_boundary(boundary, len(saved_ids), raw_file)
    actions = _mapping_list(
        acceptance.get("action_arms"), "action_arms", raw_file
    )
    action_ids: list[str] = []
    for action in actions:
        spec = action.get("action")
        if isinstance(spec, Mapping) and isinstance(
            spec.get("action_id"), str
        ):
            action_ids.append(spec["action_id"])
    if set(action_ids) != set(ACTION_IDS) or len(action_ids) != len(
        ACTION_IDS
    ):
        raise FeasibilityAuditError(
            f"action arms are incomplete: {prompt_id}"
        )
    action_records: dict[str, object] = {}
    for action in actions:
        action_id = _action_id(action, raw_file)
        if action_id in action_records:
            raise FeasibilityAuditError(
                f"duplicate action arm: {prompt_id}/{action_id}"
            )
        action_records[action_id] = _audit_action(
            action,
            action_id,
            boundary,
            span,
            prompt_id,
            raw_file,
        )
    return {
        "prompt_id": prompt_id,
        "raw_run": str(raw_file),
        "raw_run_sha256": _sha256(raw_file),
        "prompt_length": len(saved_ids),
        "boundary_cache_length": boundary["cache_lengths"][0],
        "user_span": span,
        "actions": action_records,
    }


def _verify_prompt_tokens(
    tokenizer: Any, prompt: EngineeringPrompt, saved_ids: list[int]
) -> dict[str, object]:
    messages = [dict(message) for message in prompt.messages]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered != prompt.prompt_text:
        raise FeasibilityAuditError(
            f"chat-template rendering differs: {prompt.prompt_id}"
        )
    templated = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    templated_ids = _encoded_ids(templated, "templated IDs", prompt.prompt_id)
    if templated_ids != saved_ids:
        raise FeasibilityAuditError(
            f"templated IDs differ from saved boundary: {prompt.prompt_id}"
        )
    direct = tokenizer(
        prompt.prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    )
    direct_ids = _encoded_ids(direct, "direct IDs", prompt.prompt_id)
    if direct_ids != saved_ids:
        raise FeasibilityAuditError(
            f"direct IDs differ from saved boundary: {prompt.prompt_id}"
        )
    encoded = tokenizer(
        prompt.prompt_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offset_ids = _encoded_ids(encoded, "offset IDs", prompt.prompt_id)
    if offset_ids != saved_ids:
        raise FeasibilityAuditError(
            f"offset IDs differ from saved boundary: {prompt.prompt_id}"
        )
    offsets = _encoded_offsets(encoded, prompt.prompt_id)
    if len(offsets) != len(saved_ids):
        raise FeasibilityAuditError(
            f"offset count differs from saved boundary: {prompt.prompt_id}"
        )
    matches = [
        match
        for match in _MESSAGE_RE.finditer(prompt.prompt_text)
        if match.group("role") == "user"
    ]
    if len(matches) != 1 or matches[0].group("content") != prompt.user_prompt:
        raise FeasibilityAuditError(
            f"user message cannot be identified exactly: {prompt.prompt_id}"
        )
    message = matches[0]
    content_start, content_end = message.span("content")
    message_start, message_end = message.span()
    content_indices = _indices_for_span(
        offsets,
        content_start,
        content_end,
        prompt.prompt_id,
    )
    message_indices = _indices_for_span(
        offsets,
        message_start,
        message_end,
        prompt.prompt_id,
    )
    if not content_indices or not message_indices:
        raise FeasibilityAuditError(
            f"user token span is empty: {prompt.prompt_id}"
        )
    _verify_offset_round_trip(
        prompt.prompt_text,
        offsets,
        content_indices,
        content_start,
        content_end,
        prompt.prompt_id,
    )
    _verify_offset_round_trip(
        prompt.prompt_text,
        offsets,
        message_indices,
        message_start,
        message_end,
        prompt.prompt_id,
    )
    return {
        "content_token_start": content_indices[0],
        "content_token_end": content_indices[-1] + 1,
        "content_token_count": len(content_indices),
        "content_char_start": content_start,
        "content_char_end": content_end,
        "content_token_ids_sha256": _hash_ints(
            saved_ids[content_indices[0] : content_indices[-1] + 1]
        ),
        "message_token_start": message_indices[0],
        "message_token_end": message_indices[-1] + 1,
        "message_token_count": len(message_indices),
        "message_char_start": message_start,
        "message_char_end": message_end,
    }


def _indices_for_span(
    offsets: list[tuple[int, int]],
    start: int,
    end: int,
    prompt_id: str,
) -> list[int]:
    overlapping: list[int] = []
    for index, (token_start, token_end) in enumerate(offsets):
        if token_end <= start or token_start >= end:
            continue
        if token_start < start or token_end > end or token_end <= token_start:
            raise FeasibilityAuditError(
                f"token crosses user span boundary: {prompt_id}"
            )
        overlapping.append(index)
    if not overlapping or overlapping != list(
        range(overlapping[0], overlapping[-1] + 1)
    ):
        raise FeasibilityAuditError(
            f"user span token indices are not contiguous: {prompt_id}"
        )
    return overlapping


def _verify_offset_round_trip(
    text: str,
    offsets: list[tuple[int, int]],
    indices: list[int],
    start: int,
    end: int,
    prompt_id: str,
) -> None:
    if offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
        raise FeasibilityAuditError(
            f"token offsets do not cover exact span: {prompt_id}"
        )
    pieces = [
        text[offsets[index][0] : offsets[index][1]] for index in indices
    ]
    if "".join(pieces) != text[start:end]:
        raise FeasibilityAuditError(
            f"token offsets do not round-trip exact span: {prompt_id}"
        )


def _validate_boundary(
    boundary: Mapping[str, object], prompt_length: int, raw_file: Path
) -> None:
    if boundary.get("prompt_length") != prompt_length:
        raise FeasibilityAuditError(
            f"boundary prompt length differs: {raw_file}"
        )
    generated = _integer_list(
        boundary.get("generated_token_ids"),
        "boundary.generated_token_ids",
        raw_file,
    )
    if (
        boundary.get("generated_count") != DECISION_TOKENS
        or len(generated) != DECISION_TOKENS
    ):
        raise FeasibilityAuditError(
            f"boundary token count differs: {raw_file}"
        )
    if (
        boundary.get("pending_generated_index") != DECISION_TOKENS - 1
        or boundary.get("pending_token_id") != generated[-1]
    ):
        raise FeasibilityAuditError(
            f"boundary pending token differs: {raw_file}"
        )
    if (
        boundary.get("logical_position")
        != prompt_length + DECISION_TOKENS - 1
    ):
        raise FeasibilityAuditError(
            f"boundary logical position differs: {raw_file}"
        )
    cache_lengths = _integer_list(
        boundary.get("cache_lengths"), "boundary.cache_lengths", raw_file
    )
    expected_length = prompt_length + DECISION_TOKENS - 1
    if not cache_lengths or any(
        length != expected_length for length in cache_lengths
    ):
        raise FeasibilityAuditError(
            f"boundary cache lengths differ: {raw_file}"
        )


def _audit_action(
    action: Mapping[str, object],
    action_id: str,
    boundary: Mapping[str, object],
    span: Mapping[str, object],
    prompt_id: str,
    raw_file: Path,
) -> dict[str, object]:
    compression = _mapping(action.get("compression"), "compression", raw_file)
    kept_layers = compression.get("kept_indices")
    if not isinstance(kept_layers, list) or not kept_layers:
        raise FeasibilityAuditError(
            f"kept indices are missing: {prompt_id}/{action_id}"
        )
    cache_lengths = _integer_list(
        boundary["cache_lengths"], "boundary.cache_lengths", raw_file
    )
    before_lengths = _integer_list(
        compression.get("before_lengths"),
        "compression.before_lengths",
        raw_file,
    )
    after_lengths = _integer_list(
        compression.get("after_lengths"),
        "compression.after_lengths",
        raw_file,
    )
    if (
        before_lengths != cache_lengths
        or len(after_lengths) != len(cache_lengths)
        or len(kept_layers) != len(cache_lengths)
    ):
        raise FeasibilityAuditError(
            f"action layer lengths differ: {prompt_id}/{action_id}"
        )
    user_start = cast(int, span["content_token_start"])
    user_end = cast(int, span["content_token_end"])
    user_positions = set(range(user_start, user_end))
    layers: list[dict[str, object]] = []
    for layer_index, (heads_value, cache_length, after_length) in enumerate(
        zip(kept_layers, cache_lengths, after_lengths, strict=True)
    ):
        if not isinstance(heads_value, list) or not heads_value:
            raise FeasibilityAuditError(
                "action heads are malformed: "
                f"{prompt_id}/{action_id}/{layer_index}"
            )
        for head_index, kept_value in enumerate(heads_value):
            kept = _strict_integer_list(
                kept_value,
                f"kept_indices[{layer_index}][{head_index}]",
                raw_file,
            )
            if (
                len(kept) != after_length
                or len(set(kept)) != len(kept)
                or any(
                    position < 0 or position >= cache_length
                    for position in kept
                )
            ):
                raise FeasibilityAuditError(
                    "kept positions are invalid: "
                    f"{prompt_id}/{action_id}/{layer_index}/{head_index}"
                )
            kept_set = set(kept)
            evicted_user = sorted(user_positions - kept_set)
            retained_non_user = sorted(
                (set(range(cache_length)) - user_positions) & kept_set
            )
            donor_count = len(retained_non_user)
            evicted_count = len(evicted_user)
            max_swaps = min(evicted_count, donor_count)
            layers.append(
                {
                    "layer": layer_index,
                    "kv_head": head_index,
                    "retained_count": len(kept),
                    "evicted_user_positions": evicted_user,
                    "evicted_user_count": evicted_count,
                    "retained_non_user_positions": retained_non_user,
                    "retained_non_user_count": donor_count,
                    "max_equal_budget_swaps": max_swaps,
                    "all_user_restoration_swap_count": evicted_count,
                    "all_user_restoration_feasible": evicted_count
                    <= donor_count,
                    "equal_budget_swap_feasible": max_swaps > 0,
                }
            )
    if not layers:
        raise FeasibilityAuditError(
            f"action has no head records: {prompt_id}/{action_id}"
        )
    evicted = [cast(int, row["evicted_user_count"]) for row in layers]
    donors = [cast(int, row["retained_non_user_count"]) for row in layers]
    swaps = [cast(int, row["max_equal_budget_swaps"]) for row in layers]
    return {
        "action_id": action_id,
        "layer_count": len(kept_layers),
        "kv_head_count": len(kept_layers[0])
        if isinstance(kept_layers[0], list)
        else 0,
        "head_records": layers,
        "all_user_restoration_feasible": all(
            cast(bool, row["all_user_restoration_feasible"]) for row in layers
        ),
        "equal_budget_swap_feasible": any(swaps),
        "evicted_user_count": _range_summary(evicted),
        "retained_non_user_count": _range_summary(donors),
        "max_equal_budget_swaps": _range_summary(swaps),
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    actions: dict[str, dict[str, object]] = {}
    for action_id in ACTION_IDS:
        cells: list[Mapping[str, object]] = []
        for record in records:
            record_actions = cast(Mapping[str, object], record["actions"])
            action = record_actions.get(action_id)
            if not isinstance(action, Mapping):
                raise FeasibilityAuditError(
                    f"summary action is missing: {action_id}"
                )
            cells.extend(
                row
                for row in cast(list[object], action["head_records"])
                if isinstance(row, Mapping)
            )
        if not cells:
            raise FeasibilityAuditError(f"summary has no cells: {action_id}")
        evicted = [cast(int, row["evicted_user_count"]) for row in cells]
        donors = [cast(int, row["retained_non_user_count"]) for row in cells]
        swaps = [cast(int, row["max_equal_budget_swaps"]) for row in cells]
        feasible = [
            cast(bool, row["all_user_restoration_feasible"]) for row in cells
        ]
        actions[action_id] = {
            "record_count": len(records),
            "head_record_count": len(cells),
            "all_user_restoration_feasible": all(feasible),
            "all_user_restoration_infeasible_cells": sum(
                not item for item in feasible
            ),
            "positive_equal_budget_swap_cells": sum(
                value > 0 for value in swaps
            ),
            "evicted_user_count": _range_summary(evicted),
            "retained_non_user_count": _range_summary(donors),
            "max_equal_budget_swaps": _range_summary(swaps),
        }
    return {
        "record_count": len(records),
        "action_count": len(actions),
        "actions": actions,
        "all_user_restorations_feasible": all(
            cast(bool, action["all_user_restoration_feasible"])
            for action in actions.values()
        ),
        "bounded_equal_budget_swaps_available": all(
            cast(int, action["positive_equal_budget_swap_cells"]) > 0
            for action in actions.values()
        ),
    }


def _range_summary(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def _action_id(action: Mapping[str, object], raw_file: Path) -> str:
    spec = _mapping(action.get("action"), "action", raw_file)
    value = spec.get("action_id")
    if not isinstance(value, str) or value not in ACTION_IDS:
        raise FeasibilityAuditError(f"unknown action ID: {raw_file}")
    return value


def _mapping_list(
    value: object, name: str, raw_file: Path
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not all(
        isinstance(item, Mapping) for item in value
    ):
        raise FeasibilityAuditError(f"{name} is malformed: {raw_file}")
    return [cast(Mapping[str, object], item) for item in value]


def _mapping(
    value: object, name: str, raw_file: Path
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FeasibilityAuditError(f"{name} is malformed: {raw_file}")
    return cast(Mapping[str, object], value)


def _integer_list(value: object, name: str, raw_file: Path) -> list[int]:
    if not isinstance(value, list):
        raise FeasibilityAuditError(f"{name} is malformed: {raw_file}")
    return _strict_integer_list(value, name, raw_file)


def _strict_integer_list(
    value: object, name: str, raw_file: Path
) -> list[int]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise FeasibilityAuditError(f"{name} is malformed: {raw_file}")
    return [int(item) for item in value]


def _encoded_ids(value: object, name: str, prompt_id: str) -> list[int]:
    if isinstance(value, Mapping):
        ids = value.get("input_ids")
    else:
        ids = getattr(value, "input_ids", None)
        if ids is None and hasattr(value, "tolist"):
            ids = value
    try:
        raw = ids.tolist() if hasattr(ids, "tolist") else ids
    except Exception as error:
        raise FeasibilityAuditError(
            f"{name} cannot be converted: {prompt_id}"
        ) from error
    while (
        isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list)
    ):
        raw = raw[0]
    if not isinstance(raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in raw
    ):
        raise FeasibilityAuditError(f"{name} is malformed: {prompt_id}")
    return [int(item) for item in raw]


def _encoded_offsets(value: object, prompt_id: str) -> list[tuple[int, int]]:
    if isinstance(value, Mapping):
        offsets = value.get("offset_mapping")
    else:
        offsets = getattr(value, "offset_mapping", None)
    raw = offsets.tolist() if hasattr(offsets, "tolist") else offsets
    while (
        isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list)
    ):
        raw = raw[0]
    if not isinstance(raw, list):
        raise FeasibilityAuditError(
            f"offset mapping is malformed: {prompt_id}"
        )
    result: list[tuple[int, int]] = []
    for item in raw:
        if not isinstance(item, list | tuple) or len(item) != 2:
            raise FeasibilityAuditError(
                f"offset mapping is malformed: {prompt_id}"
            )
        start, end = item
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
        ):
            raise FeasibilityAuditError(
                f"offset mapping is malformed: {prompt_id}"
            )
        result.append((int(start), int(end)))
    return result


def _read_mapping(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FeasibilityAuditError(f"cannot read JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise FeasibilityAuditError(f"JSON object required: {path}")
    return {str(key): item for key, item in value.items()}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_ints(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--tokenizer-identity", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = audit_user_span_feasibility(
            args.tokenizer,
            args.tokenizer_identity,
            args.test_manifest,
            args.raw_root,
            args.output,
        )
    except (FeasibilityAuditError, OSError, TypeError, ValueError) as error:
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
