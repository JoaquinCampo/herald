#!/usr/bin/env python3
"""Locate the queried numeric value in a known-schema NIAH prompt.

This is a measurement adapter for the frozen value-head study.  It accepts
only prompt text and a tokenizer, and never consults answers or metadata.
"""

import re
from typing import Any


_QUESTION = re.compile(
    r"\b(?:What|what)\s+is\s+the\s+special\s+magic\s+number\s+for\s+"
    r"(?P<key>.+?)\s+mentioned\s+in\s+the\s+provided\s+text\?\s*$"
)
_FACT = re.compile(
    r"One\s+of\s+the\s+special\s+magic\s+numbers\s+for\s+"
    r"(?P<key>[^:\n]+?)\s+is:\s*(?P<value>[0-9]+)\."
)
_LINE = re.compile(r"[^\n]+")


def _ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError("tokenizer returned an unexpected ID container")
    return [int(item) for item in value]


def _offsets(value: Any) -> list[tuple[int, int]]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)):
        raise ValueError("tokenizer returned an unexpected offset container")
    return [(int(item[0]), int(item[1])) for item in value]


def _miss(reason: str, prompt_length: int = 0) -> dict[str, Any]:
    return {
        "found": False,
        "reason": reason,
        "value_positions": [],
        "prompt_length": int(prompt_length),
        "value_text": "",
        "position_midpoint_normalized": 0.0,
    }


def locate_value(prompt: str, tokenizer: Any) -> dict[str, Any]:
    """Return native chat-token positions for the queried value, or an explicit miss."""
    try:
        native = _ids(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    except Exception as exc:  # keep adapter misses observable to collectors
        return _miss(f"native_tokenization_error:{type(exc).__name__}")
    prompt_length = len(native)
    lines = list(_LINE.finditer(prompt))
    if not lines:
        return _miss("missing_final_question", prompt_length)
    final = lines[-1]
    question = final.group(0).strip()
    context = prompt[: final.start()]
    query = _QUESTION.fullmatch(question)
    if query is None:
        return _miss("unrecognized_final_question", prompt_length)
    key = query.group("key").strip()
    if not key or not re.search(rf"(?<!\w){re.escape(key)}(?!\w)", question):
        return _miss("invalid_queried_key", prompt_length)
    candidates = []
    for match in _FACT.finditer(context):
        fact_key = match.group("key").strip()
        if fact_key == key and re.search(
            rf"(?<!\w){re.escape(key)}(?!\w)", match.group("key")
        ):
            candidates.append(match)
    if len(candidates) != 1:
        return _miss(
            "missing_fact" if not candidates else "ambiguous_fact",
            prompt_length,
        )
    match = candidates[0]
    value = match.group("value")
    value_start, value_end = match.span("value")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    rendered_prompt_start = rendered.find(prompt)
    if rendered_prompt_start < 0 or rendered.count(prompt) != 1:
        return _miss("prompt_not_unique_in_chat_template", prompt_length)
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    explicit = _ids(encoded["input_ids"])
    offsets = _offsets(encoded["offset_mapping"])
    if explicit != native or len(offsets) != prompt_length:
        return _miss("chat_offset_tokenization_mismatch", prompt_length)
    rendered_value_start = rendered_prompt_start + value_start
    rendered_value_end = rendered_prompt_start + value_end
    positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > rendered_value_start and start < rendered_value_end
    ]
    if not positions:
        return _miss("value_has_no_token_offsets", prompt_length)
    if any(index >= prompt_length - 1 for index in positions):
        return _miss("value_reaches_generation_prompt", prompt_length)
    midpoint = sum(positions) / len(positions)
    return {
        "found": True,
        "reason": "ok",
        "value_positions": positions,
        "prompt_length": prompt_length,
        "value_text": value,
        "position_midpoint_normalized": midpoint / max(1, prompt_length - 2),
    }
