"""Task abstractions: benchmark loading and answer scoring.

A Task ties together prompt loading, answer parsing, and
wrong-answer detection for one benchmark. Phase 1 ships four:
GSM8K (math), HumanEval (code), IFEval (instruction following),
LongBenchSingle (long-context single-doc QA).

`is_wrong` for HumanEval / IFEval / LongBench is intentionally cheap
(presence + minimal structural validity). Real pass@1 (sandboxed
exec), per-instruction verifier scores, and ROUGE/F1 are post-hoc
analysis passes that consume the saved generations; they do not gate
damage measurement during the sweep. See gold/research-plan.md
Phase 1 + the Block 1 advisor decision on HumanEval scoring.

`format_prompt` is the task-specific hook for shaping the prompt
string at generation time, when a tokenizer is available. Default
returns the pre-built `question` field unchanged; LongBench overrides
it to deterministically truncate long contexts so prefill + replay
fit within Qwen2.5-7B-Instruct's positional and replay-activation
budgets. See gold/phase-1-longbench-failure-diagnosis.md.
"""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from herald.detectors import detect_answer_failure, parse_gsm8k_answer
from herald.prompts import load_gsm8k


@dataclass(frozen=True)
class DatasetSpec:
    """Pinned Hugging Face dataset coordinates.

    The pin is `revision`; today we use `"main"` because HF datasets
    are append-only revisions of curated benchmarks and we have not
    needed sharper pinning. If a sweep ever needs reproducibility
    tighter than that, switch to a commit SHA here and tag the run
    config accordingly.
    """

    name: str  # HF repo id, e.g. "openai_humaneval"
    config: str | None = None  # subset / config name
    split: str = "test"
    revision: str = "main"
    # Some HF datasets (notably THUDM/LongBench) ship a custom Python
    # loading script that the hub refuses to execute unless this is
    # set. Default off; opt in per dataset only when the upstream
    # repo demands it.
    trust_remote_code: bool = False


class Task(ABC):
    """A benchmark with prompt loading + answer scoring."""

    name: str

    @abstractmethod
    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        """Return ordered list of prompt dicts.

        Each dict must have keys: id, question, ground_truth. May
        optionally carry: system_prompt (str) — overrides the default
        math system prompt in `format_chat`.
        """

    @abstractmethod
    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        """True iff the answer is wrong or not extractable."""

    @abstractmethod
    def parse_answer(self, generated_text: str) -> str | None:
        """Normalized answer string; None if not parseable."""

    def format_prompt(
        self,
        prompt_data: dict[str, Any],
        tokenizer: Any,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, dict[str, Any]]:
        """Return the question string + a metadata dict for sidecar logging.

        Default is a no-op: returns ``prompt_data["question"]`` and an
        empty metadata dict. LongBench overrides to truncate long
        contexts deterministically. ``system_prompt`` and
        ``max_new_tokens`` are forwarded so per-task hooks can budget
        against the chat-templated length.
        """
        return prompt_data["question"], {}


class GSM8KTask(Task):
    """GSM8K math reasoning benchmark."""

    name = "gsm8k"

    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        return load_gsm8k(num_prompts, seed)

    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        return detect_answer_failure(generated_text, ground_truth)

    def parse_answer(self, generated_text: str) -> str | None:
        return parse_gsm8k_answer(generated_text)


HUMANEVAL_SYSTEM_PROMPT = (
    "You are a careful Python programmer. Complete the function below. "
    "Return only the function body, indented to match the signature; "
    "do not repeat the signature or docstring."
)


def _extract_python_code(text: str) -> str:
    """Pull a Python block from a fenced code response, else return text."""
    fence = "```"
    if fence in text:
        first = text.find(fence)
        rest = text[first + len(fence) :]
        nl = rest.find("\n")
        if nl != -1:
            rest = rest[nl + 1 :]
        end = rest.find(fence)
        if end != -1:
            return rest[:end].strip()
    return text.strip()


class HumanEvalTask(Task):
    """OpenAI HumanEval Python code completion.

    `is_wrong` is cheap: True iff the generation is empty, has no
    `def` and no indented body, or fails to parse as Python. Real
    pass@1 (sandboxed exec) is a post-hoc analysis pass.
    """

    name = "humaneval"

    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        from datasets import load_dataset
        from loguru import logger

        logger.info(f"Loading {num_prompts} HumanEval prompts...")
        ds = load_dataset("openai_humaneval", split="test")
        ds = ds.shuffle(seed=seed)
        prompts: list[dict[str, str]] = []
        for i, row in enumerate(ds):
            if i >= num_prompts:
                break
            prompts.append(
                {
                    "id": f"humaneval_{row['task_id'].replace('/', '_')}",
                    "question": row["prompt"],
                    "ground_truth": row["canonical_solution"],
                    "system_prompt": HUMANEVAL_SYSTEM_PROMPT,
                    "test": row.get("test", ""),
                    "entry_point": row.get("entry_point", ""),
                }
            )
        logger.info(f"Loaded {len(prompts)} HumanEval prompts")
        return prompts

    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        code = _extract_python_code(generated_text)
        if not code:
            return True
        try:
            ast.parse(code)
            return False
        except SyntaxError:
            # Model returned only the body; wrap to give it a parseable home.
            wrapped = "def _f():\n" + "\n".join(
                "    " + line for line in code.splitlines()
            )
            try:
                ast.parse(wrapped)
                return False
            except SyntaxError:
                return True

    def parse_answer(self, generated_text: str) -> str | None:
        code = _extract_python_code(generated_text)
        return code or None


IFEVAL_SYSTEM_PROMPT = (
    "Follow the user's instruction exactly. Read all constraints "
    "carefully and satisfy each one in your response."
)


class IFEvalTask(Task):
    """Google IFEval instruction-following benchmark.

    `is_wrong` is a presence check. Per-instruction verifier scores
    (format constraints, length constraints, etc) are a post-hoc
    analysis pass; they require running the official `instructions`
    library against `instruction_id_list`/`kwargs` for each row.
    """

    name = "ifeval"

    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        from datasets import load_dataset
        from loguru import logger

        logger.info(f"Loading {num_prompts} IFEval prompts...")
        ds = load_dataset("google/IFEval", split="train")
        ds = ds.shuffle(seed=seed)
        prompts: list[dict[str, str]] = []
        for i, row in enumerate(ds):
            if i >= num_prompts:
                break
            raw_key = row.get("key", i)
            try:
                key = int(raw_key)
            except (TypeError, ValueError):
                key = i
            prompts.append(
                {
                    "id": f"ifeval_{key}",
                    "question": row["prompt"],
                    "ground_truth": "",
                    "system_prompt": IFEVAL_SYSTEM_PROMPT,
                    "instruction_id_list": ",".join(
                        row.get("instruction_id_list", []) or []
                    ),
                }
            )
        logger.info(f"Loaded {len(prompts)} IFEval prompts")
        return prompts

    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        return not generated_text.strip()

    def parse_answer(self, generated_text: str) -> str | None:
        text = generated_text.strip()
        return text or None


LONGBENCH_SYSTEM_PROMPT = (
    "You are a careful reader. Use ONLY the provided context to "
    "answer the question. Be concise."
)

# LongBench has many subtasks; we use narrativeqa as the headline
# single-document QA. Switching to qasper / multifieldqa_en is a
# one-line change here.
LONGBENCH_SUBTASK = "narrativeqa"

# Total chat-templated prompt-token budget for LongBench prompts.
# Phase 1 Block 2 Option B exposed deterministic CUDA OOM in the
# matched-prefix replay forward (use_cache=False) on six NarrativeQA
# prompts whose chat-templated tokenized length exceeded ~50k tokens
# on Qwen2.5-7B-Instruct (32k positional limit). The binding
# constraint is replay activation memory, not the positional limit
# itself. 16384 tokens leaves ~16k headroom under the positional
# limit and bounds attention activations comfortably below the ~17
# GiB free after model load on the RTX 5090. Generated tokens plus
# this prompt budget plus chat-template overhead must remain under
# the model's 32768 positional limit. See
# gold/phase-1-longbench-failure-diagnosis.md.
LONGBENCH_PROMPT_TOKEN_BUDGET = 16384

# Conservative safety margin (in tokens) absorbed by `format_prompt`
# below to soak up chat-template overhead and any residual
# tokenizer round-trip drift. The truncation loop shrinks the
# context further if the chat-templated length still overshoots.
LONGBENCH_TRUNCATION_SAFETY_TOKENS = 64


class LongBenchSingleTask(Task):
    """LongBench single-document QA (subtask = LONGBENCH_SUBTASK).

    `is_wrong` is a presence check. ROUGE-L / F1 against the gold
    answers is a post-hoc analysis pass.

    `format_prompt` deterministically truncates the (long) context
    so the chat-templated prompt plus generated tokens fit inside a
    fixed budget. The question and instruction text are always
    preserved; only the context is shortened. See
    gold/phase-1-longbench-failure-diagnosis.md.
    """

    name = "longbench_single"

    def load(self, num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
        from datasets import load_dataset
        from loguru import logger

        logger.info(
            f"Loading {num_prompts} LongBench/{LONGBENCH_SUBTASK} prompts..."
        )
        ds = load_dataset(
            "THUDM/LongBench",
            LONGBENCH_SUBTASK,
            split="test",
            trust_remote_code=True,
        )
        ds = ds.shuffle(seed=seed)
        prompts: list[dict[str, str]] = []
        for i, row in enumerate(ds):
            if i >= num_prompts:
                break
            ctx = row["context"]
            inp = row["input"]
            # `question` is the un-truncated, full-context formatted
            # string. `format_prompt` (called with a tokenizer at
            # generation time) returns the truncated form actually
            # fed to the model. `context` and `input` are kept raw
            # so truncation can be re-derived deterministically.
            question = f"Context:\n{ctx}\n\nQuestion: {inp}\n\nAnswer:"
            answers = row.get("answers", []) or []
            gt = answers[0] if answers else ""
            prompts.append(
                {
                    "id": f"longbench_{row['_id']}",
                    "question": question,
                    "context": ctx,
                    "input": inp,
                    "ground_truth": gt,
                    "system_prompt": LONGBENCH_SYSTEM_PROMPT,
                    "all_answers": " | ".join(answers),
                }
            )
        logger.info(f"Loaded {len(prompts)} LongBench prompts")
        return prompts

    def is_wrong(self, generated_text: str, ground_truth: str) -> bool:
        return not generated_text.strip()

    def parse_answer(self, generated_text: str) -> str | None:
        text = generated_text.strip()
        return text or None

    def format_prompt(
        self,
        prompt_data: dict[str, Any],
        tokenizer: Any,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
    ) -> tuple[str, dict[str, Any]]:
        """Truncate the context so the chat-templated prompt fits.

        Determinism: same ``(prompt_data, tokenizer, system_prompt,
        max_new_tokens)`` always produces the same output. The context
        is tokenized, sliced from the head, decoded, and the
        chat-templated length is verified. If the templated length
        still exceeds the budget (chat-template overhead longer than
        the safety margin), the context-token slice is shrunk and the
        check repeats; bounded loop, deterministic.
        """
        ctx = prompt_data.get("context")
        inp = prompt_data.get("input")
        if ctx is None or inp is None:
            # Old-style prompt_data (no raw fields). Fall back to the
            # pre-built question; no truncation possible.
            return prompt_data["question"], {
                "truncated": False,
                "reason": "missing_raw_fields",
            }
        if tokenizer is None:
            return prompt_data["question"], {
                "truncated": False,
                "reason": "no_tokenizer",
            }

        sys_prompt = (
            system_prompt
            if system_prompt is not None
            else prompt_data.get("system_prompt", LONGBENCH_SYSTEM_PROMPT)
        )

        budget = LONGBENCH_PROMPT_TOKEN_BUDGET
        safety = LONGBENCH_TRUNCATION_SAFETY_TOKENS
        original_ctx_tokens = len(
            tokenizer.encode(ctx, add_special_tokens=False)
        )

        # Short-circuit: if the un-truncated chat-templated prompt
        # already fits, return the original `question` verbatim. This
        # avoids a tokenizer encode/decode round-trip on inputs that
        # don't need it (BPE round-trips are not guaranteed to be
        # identity on arbitrary text).
        original_question = prompt_data["question"]
        original_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": original_question},
        ]
        try:
            original_chat = tokenizer.apply_chat_template(
                original_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            original_chat_len = len(
                tokenizer.encode(original_chat, add_special_tokens=False)
            )
        except Exception:  # noqa: BLE001
            original_chat_len = len(
                tokenizer.encode(
                    sys_prompt + original_question,
                    add_special_tokens=False,
                )
            )
        if original_chat_len + max_new_tokens <= budget:
            return original_question, {
                "truncated": False,
                "original_context_tokens": original_ctx_tokens,
                "truncated_context_tokens": original_ctx_tokens,
                "chat_templated_tokens": original_chat_len,
                "budget": budget,
                "max_new_tokens": max_new_tokens,
                "safety_tokens": safety,
            }
        # Initial context-token allowance: the budget minus a generous
        # estimate of fixed overhead (system prompt + question
        # scaffolding + chat-template control tokens). This is a
        # starting point; the verification loop below adjusts down.
        scaffold = f"Context:\n\n\nQuestion: {inp}\n\nAnswer:"
        scaffold_tokens = len(
            tokenizer.encode(scaffold, add_special_tokens=False)
        )
        sys_tokens = len(
            tokenizer.encode(sys_prompt, add_special_tokens=False)
        )
        # Reserve a chat-template overhead estimate (covers role tags
        # plus the generation-prompt suffix). 96 covers Qwen-style
        # ChatML overhead with margin; the loop below corrects if it
        # falls short.
        chat_overhead = 96
        ctx_token_cap = max(
            64,
            budget - safety - scaffold_tokens - sys_tokens - chat_overhead,
        )

        ctx_ids = tokenizer.encode(ctx, add_special_tokens=False)
        truncated = len(ctx_ids) > ctx_token_cap
        truncated_to = min(len(ctx_ids), ctx_token_cap)
        for _ in range(8):
            sliced = ctx_ids[:truncated_to]
            ctx_str = tokenizer.decode(sliced, skip_special_tokens=True)
            question = f"Context:\n{ctx_str}\n\nQuestion: {inp}\n\nAnswer:"
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ]
            try:
                chat_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                chat_ids = tokenizer.encode(
                    chat_text, add_special_tokens=False
                )
            except Exception:  # noqa: BLE001
                # Stub tokenizers without apply_chat_template fall back
                # to a length proxy on the question + system text.
                chat_ids = tokenizer.encode(
                    sys_prompt + question, add_special_tokens=False
                )
            chat_len = len(chat_ids)
            if chat_len + max_new_tokens <= budget:
                meta = {
                    "truncated": truncated and truncated_to < len(ctx_ids),
                    "original_context_tokens": original_ctx_tokens,
                    "truncated_context_tokens": truncated_to,
                    "chat_templated_tokens": chat_len,
                    "budget": budget,
                    "max_new_tokens": max_new_tokens,
                    "safety_tokens": safety,
                }
                return question, meta
            # Shrink context further; how much we overshot.
            overshoot = chat_len + max_new_tokens - budget
            shrink = max(overshoot + safety, 64)
            truncated_to = max(64, truncated_to - shrink)
            truncated = True

        # Loop bounded; emit the last attempt with a flag.
        meta = {
            "truncated": True,
            "original_context_tokens": original_ctx_tokens,
            "truncated_context_tokens": truncated_to,
            "chat_templated_tokens": chat_len,
            "budget": budget,
            "max_new_tokens": max_new_tokens,
            "safety_tokens": safety,
            "loop_exhausted": True,
        }
        return question, meta


DEFAULT_TASK: Task = GSM8KTask()


PHASE1_TASKS: tuple[Task, ...] = (
    GSM8KTask(),
    HumanEvalTask(),
    IFEvalTask(),
    LongBenchSingleTask(),
)


# Single source of truth for Phase 1 dataset coordinates.
# `scripts/prefetch_datasets.py` reads this; the Task loaders above
# call `load_dataset` with the matching arguments. Keep them in sync
# when adding/changing benchmarks.
PHASE1_DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(name="openai/gsm8k", config="main", split="test"),
    DatasetSpec(name="openai_humaneval", split="test"),
    DatasetSpec(name="google/IFEval", split="train"),
    DatasetSpec(
        name="THUDM/LongBench",
        config=LONGBENCH_SUBTASK,
        split="test",
        trust_remote_code=True,
    ),
)
