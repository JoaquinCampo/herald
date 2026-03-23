"""Experiment configuration and data models."""

from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, BeforeValidator
from pydantic_settings import BaseSettings


def _none_to_nan(v: object) -> float:
    """Coerce None → NaN for JSON roundtrip (NaN→null→None)."""
    if v is None:
        return float("nan")
    return float(v)  # type: ignore[arg-type]


NanFloat = Annotated[float, BeforeValidator(_none_to_nan)]


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


class RunResult(BaseModel):
    prompt_id: str
    prompt_text: str
    model: str
    press: str
    compression_ratio: float
    max_new_tokens: int = 512
    seed: int
    generated_text: str
    ground_truth: str
    predicted_answer: str | None
    correct: bool | None
    stop_reason: str
    catastrophes: list[str]
    num_tokens_generated: int
    catastrophe_onsets: dict[str, int] = {}
    signals: list[TokenSignals]
