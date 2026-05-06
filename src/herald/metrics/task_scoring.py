"""Deterministic post-hoc task scoring for Phase 1.

GSM8K and HumanEval correctness are scored at sweep time
(`herald.tasks`). IFEval and LongBench/qasper used presence-checks at
sweep time, which left their `runs.parquet.correct` field saturated at
True. This module rescores those two tasks deterministically from the
saved generations and the dataset metadata; it does NOT rerun
generation.

Two scoring entrypoints:

  - ``qasper_score(generated_text, gold_answers, threshold)`` →
    SQuAD-style normalized exact match + token-level F1, taking the
    max over multiple gold answers. Threshold (default 0.5) defines
    binary correctness; keep configurable.

  - ``ifeval_score(generated_text, instruction_id_list, kwargs_list,
    threshold)`` → per-instruction satisfaction over a partial set of
    instruction families that we can verify deterministically from the
    saved metadata. Instruction types we cannot check are explicitly
    reported, never silently treated as satisfied.

For IFEval we first try to import the official Google
``instruction_following_eval`` library; if not present we fall back to
the partial scorer in this module. The fallback always reports
unsupported types via ``ifeval_unsupported_types`` so a caller can
decide whether the run-level binary is well-defined.
"""

import re
import string
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# Qasper / SQuAD-style normalization, EM, and F1
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WS = re.compile(r"\s+")


def qasper_normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip articles + punct,
    collapse whitespace.

    This is the same recipe used by the original SQuAD/Qasper
    F1/EM evaluators. Keeping it self-contained avoids a dependency on
    rouge-score / evaluate.
    """
    if text is None:
        return ""
    s = text.lower()
    s = s.translate(_PUNCT_TABLE)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def _tokens(text: str) -> list[str]:
    return qasper_normalize(text).split()


def qasper_em(prediction: str, ground_truth: str) -> bool:
    """Normalized exact match."""
    return qasper_normalize(prediction) == qasper_normalize(ground_truth)


def qasper_token_f1(prediction: str, ground_truth: str) -> float:
    """SQuAD-style token-level F1.

    F1 = 2 P R / (P + R) where P = common / |pred_tokens| and
    R = common / |gold_tokens|. ``common`` is the multiset
    intersection (same recipe SQuAD uses).
    """
    pred = _tokens(prediction)
    gold = _tokens(ground_truth)
    if not pred or not gold:
        return 0.0
    pred_c = Counter(pred)
    gold_c = Counter(gold)
    common = pred_c & gold_c
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    p = n_common / len(pred)
    r = n_common / len(gold)
    return 2 * p * r / (p + r)


def qasper_score(
    generated_text: str,
    gold_answers: list[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score one prediction against a (possibly multi-) gold list.

    Returns a dict with:
      - qasper_em: float in {0.0, 1.0} (max over golds), or None if
        ``gold_answers`` is empty.
      - qasper_f1: max F1 over golds, or None if no gold.
      - qasper_correct: ``qasper_f1 >= threshold`` (None if no gold).
      - qasper_threshold: the threshold used.
      - qasper_n_golds: number of gold answers compared against.
    """
    if not gold_answers:
        return {
            "qasper_em": None,
            "qasper_f1": None,
            "qasper_correct": None,
            "qasper_threshold": threshold,
            "qasper_n_golds": 0,
        }
    em_max = 0.0
    f1_max = 0.0
    for g in gold_answers:
        em_max = max(em_max, 1.0 if qasper_em(generated_text, g) else 0.0)
        f1_max = max(f1_max, qasper_token_f1(generated_text, g))
    return {
        "qasper_em": em_max,
        "qasper_f1": f1_max,
        "qasper_correct": bool(f1_max >= threshold),
        "qasper_threshold": threshold,
        "qasper_n_golds": len(gold_answers),
    }


# ---------------------------------------------------------------------------
# IFEval: partial deterministic scorer
# ---------------------------------------------------------------------------


# Instruction families we verify deterministically from saved
# metadata. Each maps to a callable
# ``(generated_text, kwargs) -> bool``.
def _check_no_comma(text: str, _kwargs: dict[str, Any]) -> bool:
    return "," not in text


def _check_lowercase(text: str, _kwargs: dict[str, Any]) -> bool:
    return text == text.lower()


def _check_uppercase(text: str, _kwargs: dict[str, Any]) -> bool:
    return text == text.upper()


def _check_capital_word_frequency(text: str, kwargs: dict[str, Any]) -> bool:
    """Number of all-caps words must satisfy a relation against a
    threshold (e.g. >= 3 all-caps words)."""
    rel = kwargs.get("capital_relation") or "at least"
    n_target = kwargs.get("capital_frequency")
    if n_target is None:
        return False
    n = sum(
        1
        for w in re.findall(r"[A-Za-z]+", text)
        if w.isupper() and len(w) > 1
    )
    return _relation(n, int(n_target), rel)


def _relation(actual: int, target: int, relation: str) -> bool:
    """Apply IFEval-style relation strings to integer counts."""
    rel = (relation or "").lower().strip()
    if rel in ("at least", "more than or equal to"):
        return actual >= target
    if rel == "more than":
        return actual > target
    if rel in ("at most", "less than or equal to"):
        return actual <= target
    if rel == "less than":
        return actual < target
    if rel == "exactly":
        return actual == target
    # Default: 'at least' is by far the most common in IFEval.
    return actual >= target


def _word_count(text: str) -> int:
    return len([w for w in re.findall(r"\b[\w'-]+\b", text)])


def _sentence_count(text: str) -> int:
    # Conservative: split on . ! ? followed by space or EOL; non-empty
    # pieces are sentences.
    parts = re.split(r"[.!?]+(?:\s|$)", text)
    return sum(1 for p in parts if p.strip())


def _paragraph_count(text: str) -> int:
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return len(paragraphs)


def _check_number_words(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_words")
    if n_target is None:
        return False
    return _relation(
        _word_count(text), int(n_target), kwargs.get("relation") or ""
    )


def _check_number_sentences(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_sentences")
    if n_target is None:
        return False
    return _relation(
        _sentence_count(text), int(n_target), kwargs.get("relation") or ""
    )


def _check_number_paragraphs(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_paragraphs")
    if n_target is None:
        return False
    return _relation(
        _paragraph_count(text), int(n_target), kwargs.get("relation") or ""
    )


def _check_existence(text: str, kwargs: dict[str, Any]) -> bool:
    keywords = kwargs.get("keywords") or []
    if not keywords:
        return False
    lower = text.lower()
    return all(k.lower() in lower for k in keywords)


def _check_forbidden(text: str, kwargs: dict[str, Any]) -> bool:
    forbidden = kwargs.get("forbidden_words") or []
    if not forbidden:
        return True
    lower = text.lower()
    return not any(k.lower() in lower for k in forbidden)


def _check_keyword_frequency(text: str, kwargs: dict[str, Any]) -> bool:
    kw = kwargs.get("keyword")
    n_target = kwargs.get("frequency")
    if kw is None or n_target is None:
        return False
    count = len(re.findall(re.escape(str(kw)), text, flags=re.IGNORECASE))
    return _relation(
        count, int(n_target), kwargs.get("relation") or "at least"
    )


def _check_letter_frequency(text: str, kwargs: dict[str, Any]) -> bool:
    letter = kwargs.get("letter")
    n_target = kwargs.get("let_frequency")
    if letter is None or n_target is None:
        return False
    count = text.lower().count(str(letter).lower())
    return _relation(
        count, int(n_target), kwargs.get("let_relation") or "at least"
    )


def _check_number_placeholders(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_placeholders")
    if n_target is None:
        return False
    n = len(re.findall(r"\[[^\]\n]+\]", text))
    return _relation(n, int(n_target), kwargs.get("relation") or "at least")


def _check_number_bullets(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_bullets")
    if n_target is None:
        return False
    n = len(re.findall(r"^\s*(?:\*|-|•)\s+", text, flags=re.MULTILINE))
    return _relation(n, int(n_target), kwargs.get("relation") or "at least")


def _check_number_highlighted(text: str, kwargs: dict[str, Any]) -> bool:
    n_target = kwargs.get("num_highlights")
    if n_target is None:
        return False
    # markdown emphasis: *...* or _..._ on a non-empty span
    stars = re.findall(r"\*[^*\n]+\*", text)
    unders = re.findall(r"_[^_\n]+_", text)
    n = len(stars) + len(unders)
    return _relation(n, int(n_target), kwargs.get("relation") or "at least")


def _check_title(text: str, _kwargs: dict[str, Any]) -> bool:
    return bool(re.search(r"<<[^>]+>>", text))


def _check_postscript(text: str, kwargs: dict[str, Any]) -> bool:
    marker = kwargs.get("postscript_marker") or "P.S."
    return str(marker) in text


def _check_quotation(text: str, _kwargs: dict[str, Any]) -> bool:
    s = text.strip()
    if len(s) < 2:
        return False
    return s[0] in {'"', "“"} and s[-1] in {'"', "”"}


def _check_end_phrase(text: str, kwargs: dict[str, Any]) -> bool:
    end = kwargs.get("end_phrase")
    if end is None:
        return False
    return text.strip().rstrip('.!?"” ').endswith(str(end).rstrip())


def _check_json_format(text: str, _kwargs: dict[str, Any]) -> bool:
    import json

    s = text.strip()
    # Strip a single fenced ```json ... ``` block if present.
    fence = re.match(r"```(?:json)?\s*(.*?)\s*```\s*$", s, flags=re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        json.loads(s)
    except Exception:  # noqa: BLE001
        return False
    return True


def _check_multiple_sections(text: str, kwargs: dict[str, Any]) -> bool:
    spliter = kwargs.get("section_spliter") or "Section"
    n_target = kwargs.get("num_sections")
    if n_target is None:
        return False
    n = len(re.findall(re.escape(str(spliter)), text))
    return _relation(n, int(n_target), kwargs.get("relation") or "at least")


def _check_repeat_prompt(text: str, kwargs: dict[str, Any]) -> bool:
    prompt = kwargs.get("prompt_to_repeat")
    if not prompt:
        return False
    return str(prompt) in text


SUPPORTED_INSTRUCTIONS: dict[str, Any] = {
    "punctuation:no_comma": _check_no_comma,
    "change_case:english_lowercase": _check_lowercase,
    "change_case:english_capital": _check_uppercase,
    "change_case:capital_word_frequency": _check_capital_word_frequency,
    "length_constraints:number_words": _check_number_words,
    "length_constraints:number_sentences": _check_number_sentences,
    "length_constraints:number_paragraphs": _check_number_paragraphs,
    "keywords:existence": _check_existence,
    "keywords:forbidden_words": _check_forbidden,
    "keywords:frequency": _check_keyword_frequency,
    "keywords:letter_frequency": _check_letter_frequency,
    "detectable_content:number_placeholders": _check_number_placeholders,
    "detectable_content:postscript": _check_postscript,
    "detectable_format:number_bullet_lists": _check_number_bullets,
    "detectable_format:number_highlighted_sections": (
        _check_number_highlighted
    ),
    "detectable_format:title": _check_title,
    "detectable_format:json_format": _check_json_format,
    "detectable_format:multiple_sections": _check_multiple_sections,
    "startend:end_checker": _check_end_phrase,
    "startend:quotation": _check_quotation,
    "combination:repeat_prompt": _check_repeat_prompt,
}


def ifeval_score(
    generated_text: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any] | None],
    threshold: float = 1.0,
) -> dict[str, Any]:
    """Score one IFEval generation against its constraint list.

    Returns a dict with:
      - ifeval_num_constraints: total instructions in the list.
      - ifeval_num_supported: instructions our partial scorer can verify.
      - ifeval_num_unsupported: instructions we do not verify.
      - ifeval_num_satisfied: number of supported instructions that
        passed.
      - ifeval_score: ``num_satisfied / num_supported`` if any
        supported, else None.
      - ifeval_correct: ``ifeval_score >= threshold`` if every
        instruction is supported, else None (undefined).
      - ifeval_unsupported_types: sorted list of instruction-id strings
        that we could not verify.
      - ifeval_threshold: the threshold used.

    Note: when ``ifeval_correct is None`` it means we are unable to
    prove the run is fully correct (some instruction is unsupported);
    callers MUST treat this as undefined, not as True.
    """
    if len(instruction_id_list) != len(kwargs_list):
        raise ValueError(
            "instruction_id_list and kwargs_list must be the same length; "
            f"got {len(instruction_id_list)} vs {len(kwargs_list)}"
        )
    n_total = len(instruction_id_list)
    if n_total == 0:
        return {
            "ifeval_num_constraints": 0,
            "ifeval_num_supported": 0,
            "ifeval_num_unsupported": 0,
            "ifeval_num_satisfied": 0,
            "ifeval_score": None,
            "ifeval_correct": None,
            "ifeval_unsupported_types": [],
            "ifeval_threshold": threshold,
        }

    n_supported = 0
    n_satisfied = 0
    unsupported: list[str] = []
    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        kw = kw or {}
        # Filter out IFEval's all-None kwargs entries (per-instruction
        # row has every key in a flat schema; only the ones relevant to
        # the instruction are populated).
        kw = {k: v for k, v in kw.items() if v is not None}
        check = SUPPORTED_INSTRUCTIONS.get(inst_id)
        if check is None:
            unsupported.append(inst_id)
            continue
        n_supported += 1
        try:
            if check(generated_text, kw):
                n_satisfied += 1
        except Exception:  # noqa: BLE001
            # Bad kwargs in dataset row → treat as not satisfied; do
            # not crash the whole score build.
            pass

    score = (n_satisfied / n_supported) if n_supported else None
    if unsupported:
        correct: bool | None = None
    elif score is None:
        correct = None
    else:
        correct = bool(score >= threshold)
    return {
        "ifeval_num_constraints": n_total,
        "ifeval_num_supported": n_supported,
        "ifeval_num_unsupported": len(unsupported),
        "ifeval_num_satisfied": n_satisfied,
        "ifeval_score": score,
        "ifeval_correct": correct,
        "ifeval_unsupported_types": sorted(set(unsupported)),
        "ifeval_threshold": threshold,
    }


def has_official_ifeval() -> bool:
    """True iff the official ``instruction_following_eval`` library is
    importable. The build script can prefer it over the partial scorer
    when available; we don't auto-install it."""
    import importlib.util

    return importlib.util.find_spec("instruction_following_eval") is not None
