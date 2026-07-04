"""LongBench (Bai et al. 2023) prompt loader and scorer for HERALD.

Supports the 16 English subtasks of THUDM/LongBench. The three official
config dicts (dataset2prompt, dataset2maxlen, dataset2metric) and all
metric functions are vendored directly from the THUDM/LongBench repository
(https://github.com/THUDM/LongBench, MIT License, LongBench/config/ and
LongBench/metrics.py). Minor adaptations for strict typing and ruff-clean
code are noted inline; scoring logic is unchanged.

Public API
----------
LONGBENCH_EN_TASKS : tuple[str, ...]
    The 16 English subtask names.
load_longbench(n, subtasks=None) -> list[PromptRecord]
    Load the first n examples per subtask from HuggingFace.
score_longbench(output_text, gold) -> float
    Official LongBench metric dispatched by subtask, max over answers.
longbench_maxlen(subtask) -> int
    Official per-subtask max generation length.
"""

import json
import os
import re
import string
from collections import Counter
from collections.abc import Callable
from itertools import islice
from pathlib import Path

from herald.tasks import PromptRecord

# ---------------------------------------------------------------------------
# English subtask list
# ---------------------------------------------------------------------------

LONGBENCH_EN_TASKS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_count",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
)

# ---------------------------------------------------------------------------
# Vendored config dicts (source: THUDM/LongBench, MIT License)
# https://github.com/THUDM/LongBench/blob/main/LongBench/config/
# ---------------------------------------------------------------------------

# dataset2prompt: official few-shot prompt templates, English tasks only.
_DATASET2PROMPT: dict[str, str] = {
    "narrativeqa": (
        "You are given a story, which can be either a novel or a movie"
        " script, and a question. Answer the question asconcisely as you"
        " can, using a single phrase if possible. Do not provide any"
        " explanation.\n\nStory: {context}\n\nNow, answer the question"
        " based on the story asconcisely as you can, using a single phrase"
        " if possible. Do not provide any explanation.\n\nQuestion:"
        " {input}\n\nAnswer:"
    ),
    "qasper": (
        "You are given a scientific article and a question. Answer the"
        " question as concisely as you can, using a single phrase or"
        " sentence if possible. If the question cannot be answered based on"
        ' the information in the article, write "unanswerable". If the'
        ' question is a yes/no question, answer "yes", "no", or'
        ' "unanswerable". Do not provide any explanation.\n\nArticle:'
        " {context}\n\n Answer the question based on the above article as"
        " concisely as you can, using a single phrase or sentence if"
        " possible. If the question cannot be answered based on the"
        ' information in the article, write "unanswerable". If the'
        ' question is a yes/no question, answer "yes", "no", or'
        ' "unanswerable". Do not provide any explanation.\n\nQuestion:'
        " {input}\n\nAnswer:"
    ),
    "multifieldqa_en": (
        "Read the following text and answer briefly.\n\n{context}\n\nNow,"
        " answer the following question based on the above text, only give"
        " me the answer and do not output any other words.\n\nQuestion:"
        " {input}\nAnswer:"
    ),
    "hotpotqa": (
        "Answer the question based on the given passages. Only give me the"
        " answer and do not output any other words.\n\nThe following are"
        " given passages.\n{context}\n\nAnswer the question based on the"
        " given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the"
        " answer and do not output any other words.\n\nThe following are"
        " given passages.\n{context}\n\nAnswer the question based on the"
        " given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "musique": (
        "Answer the question based on the given passages. Only give me the"
        " answer and do not output any other words.\n\nThe following are"
        " given passages.\n{context}\n\nAnswer the question based on the"
        " given passages. Only give me the answer and do not output any"
        " other words.\n\nQuestion: {input}\nAnswer:"
    ),
    "gov_report": (
        "You are given a report by a government agency. Write a one-page"
        " summary of the report.\n\nReport:\n{context}\n\nNow, write a"
        " one-page summary of the report.\n\nSummary:"
    ),
    "qmsum": (
        "You are given a meeting transcript and a query containing a"
        " question or instruction. Answer the query in one or more"
        " sentences.\n\nTranscript:\n{context}\n\nNow, answer the query"
        " based on the above meeting transcript in one or more"
        " sentences.\n\nQuery: {input}\nAnswer:"
    ),
    "multi_news": (
        "You are given several news passages. Write a one-page summary of"
        " all news. \n\nNews:\n{context}\n\nNow, write a one-page summary"
        " of all the news.\n\nSummary:"
    ),
    "trec": (
        "Please determine the type of the question below. Here are some"
        " examples of questions.\n\n{context}\n{input}"
    ),
    "triviaqa": (
        "Answer the question based on the given passage. Only give me the"
        " answer and do not output any other words. The following are some"
        " examples.\n\n{context}\n\n{input}"
    ),
    "samsum": (
        "Summarize the dialogue into a few short sentences. The following"
        " are some examples.\n\n{context}\n\n{input}"
    ),
    "passage_count": (
        "There are some paragraphs below sourced from Wikipedia. Some of"
        " them may be duplicates. Please carefully read these paragraphs"
        " and determine how many unique paragraphs there are after removing"
        " duplicates. In other words, how many non-repeating paragraphs are"
        " there in total?\n\n{context}\n\nPlease enter the final count of"
        " unique paragraphs after removing duplicates. The output format"
        " should only contain the number, such as 1, 2, 3, and so"
        " on.\n\nThe final answer is: "
    ),
    "passage_retrieval_en": (
        "Here are 30 paragraphs from Wikipedia, along with an abstract."
        " Please determine which paragraph the abstract is"
        " from.\n\n{context}\n\nThe following is an"
        " abstract.\n\n{input}\n\nPlease enter the number of the paragraph"
        " that the abstract is from. The answer format must be like"
        ' "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: '
    ),
    "lcc": (
        "Please complete the code given below. \n{context}Next line of"
        " code:\n"
    ),
    "repobench-p": (
        "Please complete the code given below. \n{context}{input}Next line"
        " of code:\n"
    ),
}

# dataset2maxlen: official per-subtask max generation length in tokens.
_DATASET2MAXLEN: dict[str, int] = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "lcc": 64,
    "repobench-p": 64,
}

# Subtasks whose predictions must be first-line truncated before scoring.
# Reproduced from eval.py scorer(): trec, triviaqa, samsum. (lsht is
# Chinese-only and excluded from the English scope.) Load-bearing: omitting
# this truncation corrupts classification and few-shot QA scores.
_FIRST_LINE_SUBTASKS: frozenset[str] = frozenset(
    {"trec", "triviaqa", "samsum"}
)

# ---------------------------------------------------------------------------
# Metric function type alias
# ---------------------------------------------------------------------------

# All metric functions share the same signature.
MetricFn = Callable[
    [str, str, list[str] | None],
    float,
]

# ---------------------------------------------------------------------------
# Vendored metric functions (source: THUDM/LongBench, MIT License)
# https://github.com/THUDM/LongBench/blob/main/LongBench/metrics.py
# Adaptations for this file:
#   - Uniform typed signature: (prediction, ground_truth, all_classes).
#   - bare `except:` -> `except Exception:` (ruff B001/E722).
#   - `return 0` -> `return 0.0` for consistent float return type.
#   - `classification_score` receives all_classes as a named param instead
#     of **kwargs; scoring logic and list-mutation order are unchanged.
# ---------------------------------------------------------------------------


def _normalize_answer(s: str) -> str:
    """Lowercase, remove articles, punctuation, and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _token_f1(
    prediction: list[str],
    ground_truth: list[str],
) -> float:
    """Token-level F1 between two token lists."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction)
    recall = num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def qa_f1_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """Token-level QA F1 after answer normalization."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()
    return _token_f1(pred_tokens, gt_tokens)


def rouge_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """ROUGE-L F1. Returns 0.0 on empty or unparseable inputs."""
    from rouge import Rouge  # noqa: PLC0415

    rouge = Rouge()
    try:
        scores = rouge.get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    rl: dict[str, float] = scores["rouge-l"]
    return rl["f"]


def classification_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """Exact-match classification with multi-match penalty.

    Reproduced verbatim from official metrics.py, including the in-place
    list mutation while iterating. The mutation order is load-bearing for
    score fidelity; do not change it.
    """
    classes: list[str] = all_classes if all_classes is not None else []
    em_match_list: list[str] = []
    for class_name in classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    if ground_truth in em_match_list:
        return 1.0 / len(em_match_list)
    return 0.0


def retrieval_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """Fraction of predicted paragraph numbers matching the gold one."""
    pattern = r"Paragraph (\d+)"
    matches = re.findall(pattern, ground_truth)
    ground_truth_id = matches[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == ground_truth_id)
    return 0.0 if not numbers else right_num / len(numbers)


def count_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """Score for passage_count: fraction of extracted numbers that match."""
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for n in numbers if n == str(ground_truth))
    return 0.0 if not numbers else right_num / len(numbers)


def code_sim_score(
    prediction: str,
    ground_truth: str,
    all_classes: list[str] | None = None,
) -> float:
    """Fuzz ratio between first non-comment code line and ground truth."""
    from fuzzywuzzy import fuzz  # noqa: PLC0415

    all_lines = prediction.lstrip("\n").split("\n")
    pred_line = ""
    for line in all_lines:
        if "`" not in line and "#" not in line and "//" not in line:
            pred_line = line
            break
    ratio: int = fuzz.ratio(pred_line, ground_truth)
    return ratio / 100.0


# dataset2metric: maps subtask name to the metric function to use.
_DATASET2METRIC: dict[str, MetricFn] = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def longbench_maxlen(subtask: str) -> int:
    """Return the official per-subtask max generation length (tokens).

    Args:
        subtask: One of LONGBENCH_EN_TASKS.

    Raises:
        KeyError: If subtask is not in the English task set.
    """
    return _DATASET2MAXLEN[subtask]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _format_prompt(subtask: str, example: dict[str, object]) -> str:
    """Fill the official prompt template for one LongBench example."""
    template = _DATASET2PROMPT[subtask]
    context = str(example.get("context", ""))
    inp = str(example.get("input", ""))
    return template.format(context=context, input=inp)


def _longbench_dir() -> Path:
    """Directory holding the per-subtask ``{subtask}.jsonl`` files.

    ``datasets>=5`` removed loading-script support, so THUDM/LongBench (a
    script-based dataset) can no longer be fetched with ``load_dataset``. We
    read the official ``data.zip`` jsonl files directly instead; their schema
    is identical to what the loading script emitted (input, context, answers,
    all_classes, ...). Override the location with ``HERALD_LONGBENCH_DIR``;
    defaults to ``~/.herald_longbench/data``.
    """
    env = os.environ.get("HERALD_LONGBENCH_DIR")
    return Path(env) if env else Path.home() / ".herald_longbench" / "data"


def load_longbench(
    n: int,
    subtasks: list[str] | None = None,
) -> list[PromptRecord]:
    """Load the first n examples per subtask from LongBench.

    For each subtask, reads the first min(n, len(file)) examples from its
    ``{subtask}.jsonl`` (file order is the dataset order: deterministic, not
    shuffled). Returns a flat list of PromptRecord objects with all subtasks
    concatenated in the order given by `subtasks` (or LONGBENCH_EN_TASKS if
    omitted).

    The jsonl files come from the official LongBench ``data.zip`` extracted
    under ``_longbench_dir()`` (see its docstring for why we bypass
    ``load_dataset``).

    Args:
        n: Number of examples to load per subtask.
        subtasks: Subset of LONGBENCH_EN_TASKS to load. Defaults to all 16.

    Returns:
        Flat list of PromptRecord, one per example.

    Raises:
        FileNotFoundError: If a subtask's jsonl is missing under the data dir.

    Note on cost: LongBench contexts are typically 2-32K tokens, so prefill
    cost dominates inference cost. Budget accordingly when running sweeps.
    """
    selected = subtasks if subtasks is not None else list(LONGBENCH_EN_TASKS)
    for st in selected:
        if st not in LONGBENCH_EN_TASKS:
            raise ValueError(
                f"Unknown subtask {st!r}; must be one of {LONGBENCH_EN_TASKS}"
            )

    base = _longbench_dir()
    # Preflight: fail fast and up front if any requested file is absent,
    # rather than after partially processing earlier subtasks.
    for subtask in selected:
        path = base / f"{subtask}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"LongBench data not found: {path}. Extract data.zip from "
                "zai-org/LongBench here, or set HERALD_LONGBENCH_DIR."
            )

    records: list[PromptRecord] = []
    for subtask in selected:
        path = base / f"{subtask}.jsonl"
        with path.open(encoding="utf-8") as fh:
            # Skip blank lines (a trailing newline yields one) so a tool that
            # appends a final newline never feeds "" to json.loads.
            nonblank = (line for line in fh if line.strip())
            examples: list[dict[str, object]] = [
                json.loads(line) for line in islice(nonblank, n)
            ]
        for i, example in enumerate(examples):
            content = _format_prompt(subtask, example)

            raw_answers = example.get("answers", [])
            answers: list[str] = (
                [str(a) for a in raw_answers]
                if isinstance(raw_answers, list)
                else [str(raw_answers)]
            )

            raw_classes = example.get("all_classes")
            all_classes: list[str] = (
                [str(c) for c in raw_classes]
                if isinstance(raw_classes, list)
                else []
            )

            records.append(
                PromptRecord(
                    task="longbench",
                    prompt_id=f"longbench-{subtask}-{i}",
                    messages=[{"role": "user", "content": content}],
                    gold={
                        "answers": answers,
                        "all_classes": all_classes,
                        "subtask": subtask,
                    },
                )
            )
    return records


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def score_longbench(
    output_text: str,
    gold: dict[str, object],
) -> float:
    """Compute the official LongBench score for one prediction.

    Dispatches to the per-subtask metric function and takes the max over
    all gold answers, matching the official eval.py policy. For trec,
    triviaqa, and samsum the prediction is first-line truncated before
    scoring, also matching eval.py.

    Args:
        output_text: The model's full generated text.
        gold: The gold dict from PromptRecord, containing:
              "subtask" (str), "answers" (list[str]),
              "all_classes" (list[str]).

    Returns:
        Quality score q in [0, 1].

    Raises:
        ValueError: If subtask is not in LONGBENCH_EN_TASKS.
    """
    subtask = str(gold["subtask"])

    raw_answers = gold["answers"]
    answers: list[str] = (
        [str(a) for a in raw_answers]
        if isinstance(raw_answers, list)
        else [str(raw_answers)]
    )

    raw_classes = gold["all_classes"]
    all_classes: list[str] = (
        [str(c) for c in raw_classes] if isinstance(raw_classes, list) else []
    )

    # First-line truncation replicates official eval.py scorer() behavior.
    prediction = output_text
    if subtask in _FIRST_LINE_SUBTASKS:
        prediction = prediction.lstrip("\n").split("\n")[0]

    metric_fn = _DATASET2METRIC.get(subtask)
    if metric_fn is None:
        raise ValueError(
            f"Unknown subtask {subtask!r}; expected one of"
            f" {LONGBENCH_EN_TASKS}"
        )

    best: float = 0.0
    for gt in answers:
        s = metric_fn(prediction, gt, all_classes)
        if s > best:
            best = s
    return best
