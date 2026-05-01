"""Experiment configuration and data models."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import torch
from pydantic import BaseModel, BeforeValidator
from pydantic_settings import BaseSettings


def _none_to_nan(v: object) -> float:
    """Coerce None → NaN for JSON roundtrip (NaN→null→None)."""
    if v is None:
        return float("nan")
    return float(v)  # type: ignore[arg-type]


NanFloat = Annotated[float, BeforeValidator(_none_to_nan)]


def compute_prompt_hash(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def make_run_id(
    prompt_id: str, press: str, compression_ratio: float, seed: int
) -> str:
    body = f"{prompt_id}|{press}|{compression_ratio:.6f}|{seed}"
    return hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class GenerationArtifact:
    run_id: str
    input_ids: torch.Tensor
    input_len: int
    generated_token_ids: list[int]
    compressed_scores: list[torch.Tensor] = field(default_factory=list)


class ExperimentConfig(BaseSettings):
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    press_name: str = "streaming_llm"
    compression_ratio: float = 0.875
    max_new_tokens: int = 512
    num_prompts: int = 10
    seed: int = 42
    device: str = "auto"
    output_dir: Path = Path("results")
    prompt_timeout_seconds: float = 300.0
    # Iteration-2 prep: enable to also capture lookback_ratio
    # via model.generate(output_attentions=True). Off by
    # default — adds ~quadratic memory in seq length.
    capture_attention: bool = False

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class TokenSignals(BaseModel):
    entropy: float
    top1_prob: float
    top5_prob: float
    top5_logprobs: list[float] = []
    h_alts: float = 0.0
    avg_logp: float = 0.0
    delta_h: NanFloat = float("nan")
    delta_h_valid: bool = False
    kl_div: NanFloat = float("nan")
    top10_jaccard: NanFloat = float("nan")
    eff_vocab_size: float = 0.0
    tail_mass: float = 0.0
    logit_range: float = 0.0
    # Iteration-2 attention-derived feature. NaN if
    # ExperimentConfig.capture_attention was False.
    lookback_ratio: NanFloat = float("nan")


class RunResult(BaseModel):
    run_id: str
    prompt_id: str
    prompt_text: str
    prompt_hash: str
    model: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    dtype: str = "float16"
    device_class: str = "unknown"
    press: str
    compression_ratio: float
    max_new_tokens: int = 512
    seed: int
    decoding_config: dict[str, Any] = {}
    task: str = "gsm8k"
    baseline_run_id: str
    generated_text: str
    generated_token_ids: list[int] = []
    ground_truth: str
    predicted_answer: str | None
    correct: bool | None
    stop_reason: str
    catastrophes: list[str]
    num_tokens_generated: int
    catastrophe_onsets: dict[str, int] = {}
    signals: list[TokenSignals]
    replay_status: str = "pending"
    replay_error: str | None = None
    created_at: str | None = None
    herald_git_sha: str | None = None
