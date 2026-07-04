"""Model-free unit tests for longbench metric functions.

load_longbench reads local ``{subtask}.jsonl`` files (datasets>=5 dropped
script-dataset support); the loader test uses a tmp fixture, so the whole
suite runs offline with no network or downloaded data.
"""

import json
from pathlib import Path

import pytest

from herald.longbench import (
    LONGBENCH_EN_TASKS,
    classification_score,
    count_score,
    longbench_maxlen,
    qa_f1_score,
    retrieval_score,
    score_longbench,
)

# ---------------------------------------------------------------------------
# qa_f1_score
# ---------------------------------------------------------------------------


def test_qa_f1_exact_match() -> None:
    """Exact match after normalization should return 1.0."""
    assert qa_f1_score("Paris", "Paris") == 1.0


def test_qa_f1_exact_match_case_insensitive() -> None:
    """Normalization lowercases, so case differences still give 1.0."""
    assert qa_f1_score("paris", "Paris") == 1.0


def test_qa_f1_exact_match_articles() -> None:
    """Articles are stripped; 'the Paris' matches 'Paris'."""
    assert qa_f1_score("the Paris", "Paris") == 1.0


def test_qa_f1_partial_overlap() -> None:
    """Partial token overlap yields a score strictly between 0 and 1."""
    score = qa_f1_score("New York City", "New York")
    assert 0.0 < score < 1.0


def test_qa_f1_no_overlap() -> None:
    """No token overlap yields 0.0."""
    assert qa_f1_score("London", "Paris") == 0.0


def test_qa_f1_empty_prediction() -> None:
    """Empty prediction yields 0.0 (no common tokens)."""
    assert qa_f1_score("", "Paris") == 0.0


# ---------------------------------------------------------------------------
# classification_score
# ---------------------------------------------------------------------------


def test_classification_exact_single_class() -> None:
    """Single matching class present in prediction and gold -> 1.0."""
    score = classification_score(
        "The answer is ABBR",
        "ABBR",
        all_classes=["ABBR", "HUM", "LOC"],
    )
    assert score == 1.0


def test_classification_wrong_class() -> None:
    """Gold class absent from prediction -> 0.0."""
    score = classification_score(
        "The answer is LOC",
        "ABBR",
        all_classes=["ABBR", "HUM", "LOC"],
    )
    assert score == 0.0


def test_classification_no_match_in_prediction() -> None:
    """No class appears in prediction -> 0.0."""
    score = classification_score(
        "I have no idea",
        "ABBR",
        all_classes=["ABBR", "HUM", "LOC"],
    )
    assert score == 0.0


def test_classification_multiple_classes_in_prediction() -> None:
    """When multiple classes are present, score is penalized (1/n)."""
    score = classification_score(
        "ABBR or HUM",
        "ABBR",
        all_classes=["ABBR", "HUM", "LOC"],
    )
    # Both ABBR and HUM appear; gold is ABBR, so score = 1/2 = 0.5.
    assert score == pytest.approx(0.5)


def test_classification_none_all_classes() -> None:
    """None all_classes treated as empty list -> 0.0."""
    score = classification_score("ABBR", "ABBR", all_classes=None)
    assert score == 0.0


# ---------------------------------------------------------------------------
# count_score
# ---------------------------------------------------------------------------


def test_count_score_exact() -> None:
    """Prediction contains exactly the right number -> 1.0."""
    assert count_score("5", "5") == 1.0


def test_count_score_wrong_number() -> None:
    """All extracted numbers are wrong -> 0.0."""
    assert count_score("3", "5") == 0.0


def test_count_score_mixed_numbers() -> None:
    """Some numbers match, some don't: fraction of correct ones."""
    # prediction has "5" once and "3" once; gold is "5" -> 1/2
    score = count_score("5 paragraphs or maybe 3", "5")
    assert score == pytest.approx(0.5)


def test_count_score_no_numbers() -> None:
    """No numbers extracted -> 0.0."""
    assert count_score("there are no numbers here", "5") == 0.0


# ---------------------------------------------------------------------------
# retrieval_score
# ---------------------------------------------------------------------------


def test_retrieval_score_correct() -> None:
    """Prediction matches the gold paragraph number -> 1.0."""
    score = retrieval_score("Paragraph 7", "Paragraph 7")
    assert score == 1.0


def test_retrieval_score_wrong() -> None:
    """Prediction contains a different paragraph number -> 0.0."""
    score = retrieval_score("Paragraph 3", "Paragraph 7")
    assert score == 0.0


def test_retrieval_score_no_number() -> None:
    """Prediction has no numbers -> 0.0."""
    score = retrieval_score("I don't know", "Paragraph 7")
    assert score == 0.0


# ---------------------------------------------------------------------------
# longbench_maxlen
# ---------------------------------------------------------------------------


def test_longbench_maxlen_known() -> None:
    """gov_report should have maxlen 512 per official config."""
    assert longbench_maxlen("gov_report") == 512


def test_longbench_maxlen_all_tasks() -> None:
    """All English tasks have a registered maxlen that is a positive int."""
    for task in LONGBENCH_EN_TASKS:
        v = longbench_maxlen(task)
        assert isinstance(v, int) and v > 0


def test_longbench_maxlen_unknown_raises() -> None:
    """Unknown subtask raises KeyError."""
    with pytest.raises(KeyError):
        longbench_maxlen("not_a_real_task")


# ---------------------------------------------------------------------------
# score_longbench dispatcher (model-free, hand-crafted gold)
# ---------------------------------------------------------------------------


def test_score_longbench_qa_exact() -> None:
    """QA subtask: exact-match output -> 1.0."""
    gold: dict[str, object] = {
        "subtask": "hotpotqa",
        "answers": ["Paris"],
        "all_classes": [],
    }
    assert score_longbench("Paris", gold) == 1.0


def test_score_longbench_max_over_answers() -> None:
    """Scorer takes max over multiple gold answers."""
    gold: dict[str, object] = {
        "subtask": "hotpotqa",
        "answers": ["London", "Paris"],
        "all_classes": [],
    }
    # "Paris" matches the second answer perfectly.
    assert score_longbench("Paris", gold) == 1.0


def test_score_longbench_classification() -> None:
    """Classification subtask: correct single class -> 1.0."""
    gold: dict[str, object] = {
        "subtask": "trec",
        "answers": ["ABBR"],
        "all_classes": ["ABBR", "HUM", "LOC"],
    }
    # Prediction contains only ABBR -> 1.0.
    assert score_longbench("ABBR", gold) == 1.0


def test_score_longbench_first_line_truncation() -> None:
    """trec/triviaqa/samsum: only first non-empty line is scored."""
    gold: dict[str, object] = {
        "subtask": "trec",
        "answers": ["ABBR"],
        "all_classes": ["ABBR", "HUM", "LOC"],
    }
    # Second line has "HUM" which would also match; only first line counts.
    output = "ABBR\nHUM is also present"
    assert score_longbench(output, gold) == 1.0


def test_score_longbench_count() -> None:
    """passage_count subtask: correct number in prediction -> 1.0."""
    gold: dict[str, object] = {
        "subtask": "passage_count",
        "answers": ["5"],
        "all_classes": [],
    }
    assert score_longbench("5", gold) == 1.0


def test_score_longbench_retrieval() -> None:
    """passage_retrieval_en subtask: correct paragraph -> 1.0."""
    gold: dict[str, object] = {
        "subtask": "passage_retrieval_en",
        "answers": ["Paragraph 12"],
        "all_classes": [],
    }
    assert score_longbench("Paragraph 12", gold) == 1.0


def test_score_longbench_unknown_subtask_raises() -> None:
    """Unknown subtask raises ValueError."""
    gold: dict[str, object] = {
        "subtask": "not_a_task",
        "answers": ["something"],
        "all_classes": [],
    }
    with pytest.raises(ValueError, match="Unknown subtask"):
        score_longbench("output", gold)


# ---------------------------------------------------------------------------
# Loader test (offline, tmp jsonl fixture)
# ---------------------------------------------------------------------------


def test_load_longbench_reads_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loader reads {subtask}.jsonl, takes first n, builds PromptRecords."""
    from herald.longbench import load_longbench

    data_dir = tmp_path / "lb"
    data_dir.mkdir()
    rows = [
        {
            "input": f"q{i}",
            "context": f"ctx{i}",
            "answers": [f"a{i}"],
            "all_classes": [],
        }
        for i in range(3)
    ]
    # Trailing newline on purpose: real jsonl tools append one, and the
    # loader must not choke on the resulting empty final line.
    (data_dir / "narrativeqa.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("HERALD_LONGBENCH_DIR", str(data_dir))

    records = load_longbench(2, subtasks=["narrativeqa"])

    assert len(records) == 2  # first n, not all 3
    for i, r in enumerate(records):
        assert r.task == "longbench"
        assert r.prompt_id == f"longbench-narrativeqa-{i}"
        assert r.messages[0]["role"] == "user"
        assert r.gold == {
            "answers": [f"a{i}"],
            "all_classes": [],
            "subtask": "narrativeqa",
        }


def test_load_longbench_missing_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing subtask jsonl raises FileNotFoundError, not silent empty."""
    from herald.longbench import load_longbench

    monkeypatch.setenv("HERALD_LONGBENCH_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="LongBench data not found"):
        load_longbench(2, subtasks=["qasper"])
