"""Task-grounded quality scoring for HERALD.

Public API
----------
score(task, output_text, gold, **kwargs) -> float
    Dispatch to the appropriate scorer by task name.
    Returns quality q in [0, 1].

extract_pred_answer(text) -> str | None
    Extract and normalize a numeric answer from GSM8K model output.
    Exposed for standalone testing.
"""

import contextlib
import os
import re
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# GSM8K helpers
# ---------------------------------------------------------------------------

# Priority 1: #### delimiter (official GSM8K format).
# Take the LAST match; CoT outputs may reference earlier numbers.
_HASH_RE = re.compile(r"####\s*(-?[\d,\.]+)")

# Priority 2: \boxed{} (LaTeX format used by math-trained models).
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")

# Priority 3: "The answer is X" / "The final answer is X" (CoT style).
_ANSWER_IS_RE = re.compile(
    r"[Tt]he\s+(?:final\s+)?answer\s+is[:\s]*\$?(-?[\d,\.]+)"
)

# Priority 4: "Answer: X" (simple-evals / inspect_evals style).
_ANSWER_COLON_RE = re.compile(r"(?i)answer\s*:\s*\$?(-?[\d,\.]+)")

# Priority 5: Last number in text (ultimate fallback).
_LAST_NUMBER_RE = re.compile(r"(-?[\d,]+\.?\d*)")


def _normalize_number(s: str) -> str:
    """Strip commas and trailing periods; convert whole floats to ints.

    GSM8K answers are always positive integers, so float-to-int
    conversion is safe and catches rounding artifacts like "42.0".
    """
    s = s.strip().replace(",", "").rstrip(".")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
    except ValueError:
        pass
    return s


def extract_pred_answer(
    text: str, *, allow_loose: bool = False
) -> str | None:
    """Extract a numeric answer from model output text.

    Tries structured patterns in priority order, always taking the LAST
    match so that CoT reasoning steps do not shadow the final answer.

    Priority:
    1. #### delimiter (official GSM8K format)
    2. \\boxed{} (LaTeX format)
    3. "The answer is X" / "The final answer is X"
    4. "Answer: X"
    5. Last number anywhere (only if allow_loose=True)

    The bare-last-number fallback is OFF by default: HERALD scores
    compression-damaged outputs, which often omit the answer marker, and
    grabbing a stray number from a garbled generation would award false
    credit and pollute the damage labels. With it off, an output without
    a structured answer scores 0, which is the correct damage signal.

    Returns a normalized digit string, or None if none is found.
    """
    # 1. #### delimiter
    matches = _HASH_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])

    # 2. \boxed{}
    matches = _BOXED_RE.findall(text)
    if matches:
        # The content may include non-numeric chars; extract the number.
        num_match = re.search(r"-?\d[\d,]*\.?\d*", matches[-1])
        if num_match:
            return _normalize_number(num_match.group())

    # 3. "The answer is X"
    matches = _ANSWER_IS_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])

    # 4. "Answer: X"
    matches = _ANSWER_COLON_RE.findall(text)
    if matches:
        return _normalize_number(matches[-1])

    # 5. Last number in text (opt-in only)
    if allow_loose:
        matches = _LAST_NUMBER_RE.findall(text)
        if matches:
            return _normalize_number(matches[-1])

    return None


def _score_gsm8k(
    output_text: str,
    gold: dict[str, object],
) -> float:
    """Return 1.0 if extracted answer matches gold["answer"], else 0.0.

    Uses flexible (last-number) extraction. Hand-inspection of hybrid
    runs showed that compression frequently drops the "#### N" delimiter
    while keeping the correct number in prose ("Therefore ... $18"), so
    strict #### matching reports systematic false damage. Last-number is
    the accepted instruct-GSM8K metric (lm-eval flexible-extract). The
    full output text is stored, so a stricter policy is recoverable
    downstream without regeneration.
    """
    gold_answer = str(gold["answer"])
    extracted = extract_pred_answer(output_text, allow_loose=True)
    if extracted is None:
        return 0.0

    gt = _normalize_number(gold_answer)

    # Fast path: string equality.
    if extracted == gt:
        return 1.0

    # Numeric fallback: catches "72" vs "72.00".
    try:
        if abs(float(extracted) - float(gt)) < 1e-6:
            return 1.0
    except ValueError:
        pass

    return 0.0


# ---------------------------------------------------------------------------
# HumanEval helpers
# ---------------------------------------------------------------------------

# Prefer ```python ... ``` fences; fall back to ``` ... ```.
_PYTHON_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_PLAIN_FENCE_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def _extract_code(output_text: str) -> str:
    """Extract Python code from a chat response.

    Preference order:
    1. First ```python ... ``` fenced block.
    2. First ``` ... ``` fenced block (any language or untagged).
    3. The entire output_text.

    Choice rationale: chat models almost always wrap code in fences.
    Taking the first fence avoids accidentally capturing a second
    fragment or test-output block that follows the solution.
    Using the full text as a last resort handles models that output
    bare code without any fencing.
    """
    m = _PYTHON_FENCE_RE.search(output_text)
    if m:
        return m.group(1)

    m = _PLAIN_FENCE_RE.search(output_text)
    if m:
        return m.group(1)

    return output_text


def _build_program(
    code: str,
    gold: dict[str, object],
) -> str:
    """Assemble the runnable program for HumanEval scoring.

    Construction strategy: if the extracted code does not define the
    entry_point function, prepend gold["prompt"] (the function
    signature and docstring) so the function header is present, then
    append the code as the body. This handles completions that omit
    the signature (common with chat-format problems) while still
    working when the full definition is present.

    The assembled program is:
        {code}                 (or {prompt}{code} if needed)
        {test}
        check({entry_point})
    """
    entry_point = str(gold["entry_point"])
    test = str(gold["test"])
    prompt = str(gold["prompt"])

    # Detect whether extracted code already defines the function.
    define_re = re.compile(
        rf"^\s*def\s+{re.escape(entry_point)}\s*\(", re.MULTILINE
    )
    # If code already defines entry_point, use it directly; otherwise
    # prepend the signature/docstring so the function header is present.
    body = code if define_re.search(code) else prompt + code

    return body + "\n" + test + f"\ncheck({entry_point})\n"


def _score_humaneval(
    output_text: str,
    gold: dict[str, object],
    timeout: float = 10.0,
) -> float:
    """Execute model-generated code against the HumanEval check function.

    Returns 1.0 if the program exits with code 0 within timeout, else 0.0.

    Safety notes:
    - ponytail: subprocess+timeout sandbox; no seccomp/container.
      Harden if running untrusted-at-scale.
    - Hard timeout prevents infinite loops from hanging the caller.
    - Code runs in a fresh subprocess (never exec'd in-process) to
      isolate side effects and interpreter state.
    - The temp file is always cleaned up, even on failure.
    - All exceptions are swallowed to 0.0 so a bad completion never
      raises into the caller.
    """
    code = _extract_code(output_text)
    program = _build_program(code, gold)

    tmp_path: str | None = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=".py", prefix="herald_he_")
        os.close(fd)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            fh.write(program)

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        if tmp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_KNOWN_TASKS = {"gsm8k", "humaneval", "ifeval", "longbench"}


def score(
    task: str,
    output_text: str,
    gold: dict[str, object],
    *,
    timeout: float = 10.0,
) -> float:
    """Return task-grounded quality q in [0, 1].

    Parameters
    ----------
    task:
        Task identifier. Currently "gsm8k" or "humaneval".
    output_text:
        The model's full generated text for this prompt.
    gold:
        Ground-truth metadata. Schema depends on task:
        - gsm8k: {"answer": str}
        - humaneval: {"prompt": str, "test": str, "entry_point": str}
    timeout:
        Subprocess timeout in seconds (HumanEval only).

    Raises
    ------
    ValueError
        If task is not a known task name.
    """
    if task == "gsm8k":
        return _score_gsm8k(output_text, gold)
    if task == "humaneval":
        return _score_humaneval(output_text, gold, timeout=timeout)
    if task == "ifeval":
        from herald.ifeval import score_ifeval

        return score_ifeval(output_text, gold)
    if task == "longbench":
        from herald.longbench import score_longbench

        return score_longbench(output_text, gold)
    raise ValueError(
        f"unknown task {task!r}; expected one of {sorted(_KNOWN_TASKS)}"
    )
