"""Experiment configuration for the HERALD generation sweep.

A single `Config` instance is the full, reproducible spec of one sweep:
which models, tasks, compressors, and ratios to run, the switch stride,
and how many prompts per task. Defaults describe the full sweep; pass
overrides to construct a slice.
"""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# Base models: two distinct families (different tokenizer + pretraining)
# so the cross-family transfer claim is non-trivial.
MODELS: dict[str, str] = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3": "Qwen/Qwen3-8B",
}

# Five compressors spanning distinct selection principles, so holding
# one out is a real transfer test. All weight-free, none need eager.
COMPRESSORS: tuple[str, ...] = (
    "streaming_llm",
    "snapkv",
    "expected_attention",
    "knorm",
    "random",
)

# Compression ratio = fraction of KV pairs REMOVED (kvpress convention).
RATIOS: tuple[float, ...] = (0.25, 0.5, 0.75, 0.875)


class TaskSpec(BaseModel):
    """One task: where to load it and its generation budget.

    `max_new_tokens` is M, the cap applied identically to the reference
    run and every hybrid, so the only difference between paired runs is
    compression, never the budget.
    """

    name: str
    hf_path: str
    hf_subset: str | None = None
    split: str = "test"
    max_new_tokens: int


TASKS: dict[str, TaskSpec] = {
    "gsm8k": TaskSpec(
        name="gsm8k",
        hf_path="openai/gsm8k",
        hf_subset="main",
        split="test",
        max_new_tokens=512,
    ),
    "humaneval": TaskSpec(
        name="humaneval",
        hf_path="openai/openai_humaneval",
        split="test",
        max_new_tokens=512,
    ),
    # ifeval and longbench use custom loaders (herald.ifeval /
    # herald.longbench); the spec fields are documentation, only
    # max_new_tokens (the budget cap M) is read by the runner. The model
    # stops early at EOS, so the cap mainly bounds rambling outputs.
    "ifeval": TaskSpec(
        name="ifeval",
        hf_path="google/IFEval",
        split="train",
        max_new_tokens=1024,
    ),
    "longbench": TaskSpec(
        name="longbench",
        hf_path="THUDM/LongBench",
        split="test",
        max_new_tokens=512,
    ),
}


class Config(BaseModel):
    """Full spec of one generation sweep."""

    models: list[str] = Field(default_factory=lambda: list(MODELS))
    tasks: list[str] = Field(default_factory=lambda: ["gsm8k", "humaneval"])
    compressors: list[str] = Field(default_factory=lambda: list(COMPRESSORS))
    ratios: list[float] = Field(default_factory=lambda: list(RATIOS))
    # k: switch positions are {0, k, 2k, ...} up to the run length.
    switch_stride: int = 16
    prompts_per_task: int = 200
    # Used only to seed RandomPress evictions for reproducibility;
    # prompt selection is first-N in dataset order and already stable.
    seed: int = 0
    dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    results_dir: Path = Path("results")
    # References batch freely (no press). Hybrids default to batch 1:
    # left-pad batching may corrupt compression (pad tokens enter the
    # press), so it stays 1 until validated equal to batch 1 per press.
    ref_batch_size: int = 16
    hybrid_batch_size: int = 1
    # Attention-reliance tap on reference runs (feature extension
    # phase 1). Off by default: the legacy dataset stays exactly
    # reproducible. Empty tap_layer_indices means quarter-depth.
    tap_attention: bool = False
    tap_layer_indices: tuple[int, ...] = ()

    @field_validator("ratios")
    @classmethod
    def _ratios_in_unit_interval(cls, v: list[float]) -> list[float]:
        for r in v:
            if not 0.0 < r < 1.0:
                raise ValueError(
                    f"ratio {r} must be in the open interval (0, 1)"
                )
        return v

    @field_validator(
        "switch_stride",
        "prompts_per_task",
        "ref_batch_size",
        "hybrid_batch_size",
    )
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("must be >= 1")
        return v
