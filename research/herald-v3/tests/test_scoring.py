"""Scoring tests for official checker parity and signed differences."""

import pytest

from herald_v3.engineering.scoring import (
    ScoringResourceError,
    check_ifeval_resources,
    score_ifeval_gold,
    score_pair,
)


def test_loose_transforms_can_recover_a_strictly_failed_response() -> None:
    gold = {
        "prompt": "fixture",
        "instruction_id_list": ["startend:quotation"],
        "kwargs": [{}],
    }

    scores = score_ifeval_gold('preamble\n"answer"', gold)

    assert scores.strict == 0.0
    assert scores.loose == 1.0
    assert scores.strict_pass == (False,)
    assert scores.loose_pass == (True,)


def test_signed_pair_keeps_positive_negative_and_zero_values() -> None:
    gold = {
        "prompt": "fixture",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }

    positive = score_pair("plain", "has, comma", gold)
    negative = score_pair("has, comma", "plain", gold)
    zero = score_pair("plain", "plain", gold)

    assert positive.d_loose == 1.0
    assert negative.d_loose == -1.0
    assert zero.d_loose == 0.0
    assert positive.d_strict == 1.0


def test_missing_instruction_resource_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import nltk  # type: ignore[import-untyped]

    def missing(_: str) -> object:
        raise LookupError("fixture missing")

    monkeypatch.setattr(nltk.data, "find", missing)
    gold = {
        "prompt": "fixture",
        "instruction_id_list": ["punctuation:no_comma"],
        "kwargs": [{}],
    }

    with pytest.raises(ScoringResourceError, match="missing NLTK"):
        score_ifeval_gold("plain", gold)


def test_resources_report_versions() -> None:
    resources = check_ifeval_resources()

    assert resources["nltk_resources"] == [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
    ]
