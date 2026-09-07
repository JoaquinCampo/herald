"""Tests for exact user-span and donor-budget feasibility checks."""

import hashlib
import sys
from pathlib import Path

import pytest

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_user_span_feasibility as audit  # noqa: E402
from herald_v3.engineering.prompts import EngineeringPrompt  # noqa: E402


class _CharTokenizer:
    """A deterministic tokenizer whose offsets expose boundary mistakes."""

    chat_template = "fixture chat template"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
    ) -> object:
        del return_tensors
        rendered = (
            "<|im_start|>system\n"
            + messages[0]["content"]
            + "<|im_end|>\n"
            + "<|im_start|>user\n"
            + messages[1]["content"]
            + "<|im_end|>\n"
            + ("<|im_start|>assistant\n" if add_generation_prompt else "")
        )
        encoded = _encode(rendered)
        return {"input_ids": [encoded[0]]} if tokenize else rendered

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_tensors: str | None = None,
        return_offsets_mapping: bool = False,
    ) -> object:
        del add_special_tokens, return_tensors
        encoded = _encode(text)
        if return_offsets_mapping:
            return {
                "input_ids": [encoded[0]],
                "offset_mapping": [encoded[1]],
            }
        return {"input_ids": [encoded[0]]}


def _encode(text: str) -> tuple[list[int], list[tuple[int, int]]]:
    return list(range(len(text))), [(i, i + 1) for i in range(len(text))]


def _prompt() -> EngineeringPrompt:
    messages = (
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "keep this"},
    )
    prompt_text = (
        "<|im_start|>system\nrules<|im_end|>\n"
        "<|im_start|>user\nkeep this<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return EngineeringPrompt(
        prompt_id="ifeval_1",
        key=1,
        fold=0,
        fold_hash="fold",
        split_hash="split",
        prompt_text=prompt_text,
        user_prompt="keep this",
        messages=messages,
        instruction_id_list=("fixture:instruction",),
        kwargs=({},),
        prompt_text_sha256=hashlib.sha256(
            prompt_text.encode("utf-8")
        ).hexdigest(),
        user_prompt_sha256=hashlib.sha256(b"keep this").hexdigest(),
    )


def test_exact_span_round_trips_from_rendered_prompt() -> None:
    prompt = _prompt()
    ids, _ = _encode(prompt.prompt_text)

    span = audit._verify_prompt_tokens(_CharTokenizer(), prompt, ids)

    assert span["content_token_count"] == len(prompt.user_prompt)
    assert span["content_token_end"] - span["content_token_start"] == len(
        prompt.user_prompt
    )
    assert span["message_token_start"] < span["content_token_start"]
    assert span["content_token_end"] < span["message_token_end"]


def test_saved_boundary_token_mismatch_fails_closed() -> None:
    prompt = _prompt()
    ids, _ = _encode(prompt.prompt_text)
    ids[4] = -1

    with pytest.raises(
        audit.FeasibilityAuditError, match="templated IDs differ"
    ):
        audit._verify_prompt_tokens(_CharTokenizer(), prompt, ids)


def test_offsets_crossing_user_boundary_fail_closed() -> None:
    with pytest.raises(
        audit.FeasibilityAuditError, match="crosses user span boundary"
    ):
        audit._indices_for_span([(0, 4), (4, 8)], 2, 8, "ifeval_1")


def test_equal_budget_is_bounded_when_all_user_restore_is_impossible() -> (
    None
):
    boundary = {"cache_lengths": [5]}
    span = {"content_token_start": 1, "content_token_end": 4}
    action = {
        "action": {"action_id": "knorm:0.25"},
        "compression": {
            "before_lengths": [5],
            "after_lengths": [2],
            "kept_indices": [[[0, 1]]],
        },
    }

    result = audit._audit_action(
        action,
        "knorm:0.25",
        boundary,
        span,
        "ifeval_1",
        Path("fixture/run.json"),
    )
    head = result["head_records"][0]

    assert head["evicted_user_count"] == 2
    assert head["retained_non_user_count"] == 1
    assert head["max_equal_budget_swaps"] == 1
    assert head["all_user_restoration_feasible"] is False
    assert head["equal_budget_swap_feasible"] is True


def test_invalid_kept_position_is_rejected() -> None:
    boundary = {"cache_lengths": [5]}
    span = {"content_token_start": 1, "content_token_end": 4}
    action = {
        "action": {"action_id": "knorm:0.5"},
        "compression": {
            "before_lengths": [5],
            "after_lengths": [2],
            "kept_indices": [[[0, 5]]],
        },
    }

    with pytest.raises(audit.FeasibilityAuditError, match="kept positions"):
        audit._audit_action(
            action,
            "knorm:0.5",
            boundary,
            span,
            "ifeval_1",
            Path("fixture/run.json"),
        )


def test_real_saved_r5_record_is_checkable() -> None:
    tokenizer_dir = ROOT / "data/retrieval-tokenizer-a09a354"
    raw_file = (
        ROOT / "results/lookahead-outcomes-r5/raw-runs/ifeval_251/run.json"
    )
    if not tokenizer_dir.is_dir() or not raw_file.is_file():
        pytest.skip("saved r5 evidence is unavailable")
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers is unavailable in this test environment")
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
    )
    manifest = audit.load_prompt_manifest(
        ROOT / "data/lookahead-v1/test-prompts.json", limit=76
    )
    prompts = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    result = audit._audit_raw_run(raw_file, tokenizer, prompts)

    assert result["prompt_id"] == "ifeval_251"
    assert result["user_span"]["content_token_count"] > 0
    assert set(result["actions"]) == set(audit.ACTION_IDS)
