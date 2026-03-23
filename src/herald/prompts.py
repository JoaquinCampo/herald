"""GSM8K prompt loading and formatting."""

from datasets import load_dataset
from loguru import logger

SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Show your reasoning, then give the final numeric answer after ####."
)

ChatMessages = list[dict[str, str]]


def load_gsm8k(num_prompts: int, seed: int = 42) -> list[dict[str, str]]:
    """Load GSM8K test prompts with deterministic shuffling."""
    logger.info(f"Loading {num_prompts} GSM8K prompts...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=seed)

    prompts = []
    for i, row in enumerate(ds):
        if i >= num_prompts:
            break
        answer_text = row["answer"]
        gt = answer_text.split("####")[-1].strip().replace(",", "")
        prompts.append(
            {
                "id": f"gsm8k_{i}",
                "question": row["question"],
                "ground_truth": gt,
                "full_answer": answer_text,
            }
        )
    logger.info(f"Loaded {len(prompts)} prompts")
    return prompts


def format_chat(question: str) -> ChatMessages:
    """Format a GSM8K question as chat messages for instruct models.

    Returns a list of message dicts ready for tokenizer.apply_chat_template().
    Zero-shot — instruct models are already trained for
    step-by-step reasoning.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
