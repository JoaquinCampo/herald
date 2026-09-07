#!/usr/bin/env python3
"""Locate four native seven-digit values for one known-schema prompt key."""

import re


QUESTION = re.compile(
    r"\b(?:What|what)\s+are\s+all\s+the\s+special\s+magic\s+numbers\s+for\s+"
    r"(?P<key>.+?)\s+mentioned\s+in\s+the\s+provided\s+text\?\s*$"
)
FACT = re.compile(
    r"One\s+of\s+the\s+special\s+magic\s+numbers\s+for\s+"
    r"(?P<key>[^:\n]+?)\s+is:\s*(?P<value>[0-9]+)\."
)


def _flat(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return value


def _miss(reason, prompt_length=0):
    return {"found": False, "reason": reason, "prompt_length": int(prompt_length), "key": "", "values": []}


def locate_values(prompt, tokenizer):
    """Return four occurrence-ordered native spans without consulting answers."""
    try:
        native = [int(x) for x in _flat(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
        ))]
    except Exception as exc:
        return _miss(f"native_tokenization_error:{type(exc).__name__}")
    length = len(native)
    lines = list(re.finditer(r"[^\n]+", prompt))
    if not lines:
        return _miss("missing_final_question", length)
    final = lines[-1]
    query = QUESTION.fullmatch(final.group(0).strip())
    if query is None:
        return _miss("unrecognized_final_question", length)
    key = query.group("key").strip()
    matches = [m for m in FACT.finditer(prompt[: final.start()]) if m.group("key").strip() == key]
    if len(matches) != 4 or any(len(m.group("value")) != 7 for m in matches):
        return _miss("expected_four_seven_digit_facts", length)
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    start = rendered.find(prompt)
    if start < 0 or rendered.count(prompt) != 1:
        return _miss("prompt_not_unique_in_chat_template", length)
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    explicit = [int(x) for x in _flat(encoded["input_ids"])]
    offsets = [tuple(map(int, x)) for x in _flat(encoded["offset_mapping"])]
    if explicit != native or len(offsets) != length:
        return _miss("chat_offset_tokenization_mismatch", length)
    values = []
    for occurrence, match in enumerate(matches):
        value = match.group("value")
        lo, hi = start + match.start("value"), start + match.end("value")
        positions = [i for i, (a, b) in enumerate(offsets) if b > lo and a < hi]
        if len(positions) != 7 or positions != list(range(positions[0], positions[0] + 7)):
            return _miss("value_is_not_seven_native_tokens", length)
        for digit, index in enumerate(positions):
            if offsets[index] != (lo + digit, lo + digit + 1):
                return _miss("value_offsets_do_not_reconstruct", length)
            decoded = tokenizer.decode([native[index]], skip_special_tokens=False).strip()
            if decoded != value[digit]:
                return _miss("value_token_decode_mismatch", length)
        if any(index >= length - 1 for index in positions):
            return _miss("value_reaches_generation_prompt", length)
        values.append({"occurrence": occurrence, "value_text": value, "value_positions": positions,
                       "position_midpoint_normalized": sum(positions) / 7 / max(1, length)})
    return {"found": True, "reason": "ok", "key": key, "prompt_length": length, "values": values}
