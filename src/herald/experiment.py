"""Experiment runner: load model, compress KV cache,
generate, extract signals, detect catastrophes."""

import gc
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from herald.config import ExperimentConfig, RunResult
from herald.detectors import (
    detect_all,
    detect_catastrophe_onsets,
)
from herald.prompts import format_chat
from herald.signals import compute_lookback_ratios, extract_signals
from herald.tasks import DEFAULT_TASK, Task


def _checkpoint_path(config: ExperimentConfig) -> Path:
    output_dir = config.output_dir / config.press_name
    model_short = config.model_name.split("/")[-1]
    ratio_str = f"{config.compression_ratio:.3f}"
    return (
        output_dir
        / f"{model_short}_{ratio_str}_{config.num_prompts}p.ckpt.jsonl"
    )


def _load_checkpoint(config: ExperimentConfig) -> list[RunResult]:
    ckpt = _checkpoint_path(config)
    if not ckpt.exists():
        return []
    results = []
    for line in ckpt.read_text().splitlines():
        line = line.strip()
        if line:
            results.append(RunResult.model_validate_json(line))
    return results


def _append_checkpoint(result: RunResult, config: ExperimentConfig) -> None:
    ckpt = _checkpoint_path(config)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    with ckpt.open("a") as f:
        f.write(result.model_dump_json() + "\n")


def _clear_checkpoint(config: ExperimentConfig) -> None:
    ckpt = _checkpoint_path(config)
    if ckpt.exists():
        ckpt.unlink()


def get_press(name: str, compression_ratio: float) -> Any:  # noqa: ANN201
    """Create a kvpress Press object (or None for baseline)."""
    if name == "none":
        return None

    from kvpress import (
        ExpectedAttentionPress,
        KnormPress,
        RandomPress,
        SnapKVPress,
        StreamingLLMPress,
        TOVAPress,
    )

    presses = {
        "streaming_llm": StreamingLLMPress,
        "snapkv": SnapKVPress,
        "knorm": KnormPress,
        "expected_attention": ExpectedAttentionPress,
        "tova": TOVAPress,
        "random": RandomPress,
    }
    if name not in presses:
        raise ValueError(
            f"Unknown press: {name}. Available: {list(presses.keys())}"
        )
    return presses[name](compression_ratio=compression_ratio)


def load_model(
    config: ExperimentConfig,
) -> tuple[AutoModelForCausalLM, AutoTokenizer, str]:
    """Load model and tokenizer onto the target device."""
    device = config.resolve_device()
    logger.info(f"Loading {config.model_name} on {device}...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    )
    model = model.to(device)  # type: ignore[arg-type]
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {param_count:,}")
    return model, tokenizer, device  # type: ignore[return-value]


class _TimeoutCriteria(StoppingCriteria):
    """Stop generation if wall-clock time exceeds a threshold."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout = timeout_seconds
        self.start_time = time.time()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        exceeded = (time.time() - self.start_time) > self.timeout
        return torch.full(
            (input_ids.shape[0],),
            exceeded,
            dtype=torch.bool,
            device=input_ids.device,
        )


def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    press: object | None,
    task: Task = DEFAULT_TASK,
) -> RunResult:
    """Run generation for a single prompt and extract signals."""
    messages = format_chat(prompt_data["question"])
    chat_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)  # type: ignore[operator]
    input_len = inputs["input_ids"].shape[1]

    ctx = press(model) if press is not None else nullcontext()  # type: ignore[operator]
    stopping = StoppingCriteriaList(
        [_TimeoutCriteria(config.prompt_timeout_seconds)]
    )

    with torch.no_grad(), ctx:
        outputs = model.generate(  # type: ignore[attr-defined]
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            output_scores=True,
            output_attentions=config.capture_attention,
            return_dict_in_generate=True,
            stopping_criteria=stopping,
        )

    generated_ids = outputs.sequences[0, input_len:].tolist()
    generated_text = tokenizer.decode(  # type: ignore[attr-defined]
        generated_ids, skip_special_tokens=True
    )

    eos_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
    eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    hit_eos = len(generated_ids) > 0 and generated_ids[-1] in eos_ids
    hit_max = len(generated_ids) >= config.max_new_tokens

    if hit_eos:
        stop_reason = "eos"
    elif hit_max:
        stop_reason = "max_tokens"
    else:
        stop_reason = "timeout"

    # Extract per-token signals, threading StepState between calls
    signals = []
    state = None
    for score in outputs.scores:
        sig, state = extract_signals(score[0], prev=state)
        signals.append(sig)

    if config.capture_attention and outputs.attentions is not None:
        lookbacks = compute_lookback_ratios(
            outputs.attentions, input_len=input_len
        )
        for sig, lb in zip(signals, lookbacks):
            sig.lookback_ratio = lb

    catastrophes = detect_all(
        generated_text,
        generated_ids,
        stop_reason,
        prompt_data["ground_truth"],
        is_wrong_fn=task.is_wrong,
    )
    catastrophe_onsets = detect_catastrophe_onsets(
        generated_ids, stop_reason, catastrophes
    )

    predicted = task.parse_answer(generated_text)
    correct = (
        "wrong_answer" not in catastrophes if predicted is not None else None
    )

    return RunResult(
        prompt_id=prompt_data["id"],
        prompt_text=chat_text,
        model=config.model_name,
        press=config.press_name,
        compression_ratio=config.compression_ratio,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        generated_text=generated_text,
        ground_truth=prompt_data["ground_truth"],
        predicted_answer=predicted,
        correct=correct,
        stop_reason=stop_reason,
        catastrophes=catastrophes,
        num_tokens_generated=len(generated_ids),
        catastrophe_onsets=catastrophe_onsets,
        signals=signals,
    )


def summarize(results: list[RunResult]) -> dict[str, Any]:
    """Compute summary statistics over a batch of results."""
    n = len(results)
    if n == 0:
        return {"total": 0}

    correct = sum(1 for r in results if r.correct)
    has_catastrophe = sum(1 for r in results if r.catastrophes)

    cat_counts: dict[str, int] = {}
    for r in results:
        for c in r.catastrophes:
            cat_counts[c] = cat_counts.get(c, 0) + 1

    return {
        "total": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "catastrophic_failure_rate": round(has_catastrophe / n, 4),
        "catastrophe_counts": cat_counts,
        "avg_tokens": round(
            sum(r.num_tokens_generated for r in results) / n, 1
        ),
    }


def save_results(results: list[RunResult], config: ExperimentConfig) -> Path:
    """Save results + summary to a JSON file."""
    output_dir = config.output_dir / config.press_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_short = config.model_name.split("/")[-1]
    ratio_str = f"{config.compression_ratio:.3f}"
    filename = f"{model_short}_{ratio_str}_{config.num_prompts}p.json"
    path = output_dir / filename

    data = {
        "config": config.model_dump(mode="json"),
        "summary": summarize(results),
        "results": [r.model_dump(mode="json") for r in results],
    }
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Results saved to {path}")
    _clear_checkpoint(config)
    return path


def run_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompts: list[dict[str, Any]],
    config: ExperimentConfig,
    press: object | None,
    task: Task = DEFAULT_TASK,
) -> list[RunResult]:
    """Run all prompts with a pre-loaded model.

    Supports per-prompt checkpointing.
    """
    results = _load_checkpoint(config)
    completed_ids = {r.prompt_id for r in results}
    if completed_ids:
        logger.info(
            f"Resuming from checkpoint: "
            f"{len(completed_ids)}/{len(prompts)} done"
        )

    n_failed = 0
    for i, prompt_data in enumerate(prompts):
        if prompt_data["id"] in completed_ids:
            logger.info(
                f"[{i + 1}/{len(prompts)}] "
                f"{prompt_data['id']} — skipped (ckpt)"
            )
            continue

        logger.info(f"[{i + 1}/{len(prompts)}] {prompt_data['id']}")
        t0 = time.time()

        try:
            result = run_single(
                model, tokenizer, device, prompt_data, config, press, task
            )
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            n_failed += 1
            continue

        elapsed = time.time() - t0
        timed_out = elapsed >= config.prompt_timeout_seconds * 0.95
        status = (
            "TIMEOUT"
            if timed_out
            else ("CORRECT" if result.correct else "WRONG")
        )
        cats = (
            ", ".join(result.catastrophes) if result.catastrophes else "none"
        )
        logger.info(
            f"  {status} | tokens={result.num_tokens_generated} "
            f"| catastrophes=[{cats}] | {elapsed:.1f}s"
        )
        results.append(result)
        _append_checkpoint(result, config)

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if n_failed > 0:
        logger.warning(
            f"{n_failed}/{len(prompts)} prompts failed "
            f"and were excluded from results"
        )

    return results


def print_summary(results: list[RunResult], config: ExperimentConfig) -> None:
    """Log a summary of results for one configuration."""
    s = summarize(results)
    logger.info("=" * 60)
    logger.info(
        f"CONFIG: {config.press_name} @ ratio={config.compression_ratio}"
    )
    if s["total"] > 0:
        logger.info(
            f"SUMMARY: {s['total']} prompts | accuracy={s['accuracy']:.1%}"
            f" | CFR={s['catastrophic_failure_rate']:.1%}"
        )
        if s.get("catastrophe_counts"):
            for cat, count in s["catastrophe_counts"].items():
                logger.info(f"  {cat}: {count}/{s['total']}")
    else:
        logger.warning("No results collected — all prompts failed.")
    logger.info("=" * 60)


def run_experiment(
    config: ExperimentConfig, task: Task = DEFAULT_TASK
) -> list[RunResult]:
    """Run the full experiment: load model, iterate
    prompts, collect results."""
    logger.info(
        f"Experiment: {config.press_name} "
        f"@ compression_ratio={config.compression_ratio}"
    )

    model, tokenizer, device = load_model(config)
    press = get_press(config.press_name, config.compression_ratio)
    prompts = task.load(config.num_prompts, config.seed)
    results = run_prompts(
        model, tokenizer, device, prompts, config, press, task
    )
    print_summary(results, config)
    return results


def result_exists(config: ExperimentConfig) -> bool:
    """Check if a result file already exists for this config."""
    output_dir = config.output_dir / config.press_name
    model_short = config.model_name.split("/")[-1]
    ratio_str = f"{config.compression_ratio:.3f}"
    filename = f"{model_short}_{ratio_str}_{config.num_prompts}p.json"
    return (output_dir / filename).exists()


SWEEP_RATIOS = [0.0, 0.25, 0.5, 0.625, 0.75, 0.875]
SWEEP_METHODS = [
    "streaming_llm",
    "snapkv",
    "knorm",
    "expected_attention",
    "tova",
    "random",
]


def build_sweep_configs(
    num_prompts: int = 50,
    seed: int = 42,
    output_dir: Path = Path("results"),
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt_timeout_seconds: float = 300.0,
) -> list[ExperimentConfig]:
    """Build the full list of sweep configurations."""
    configs = []
    for ratio in SWEEP_RATIOS:
        if ratio == 0.0:
            configs.append(
                ExperimentConfig(
                    model_name=model_name,
                    press_name="none",
                    compression_ratio=0.0,
                    num_prompts=num_prompts,
                    seed=seed,
                    output_dir=output_dir,
                    max_new_tokens=max_new_tokens,
                    prompt_timeout_seconds=prompt_timeout_seconds,
                )
            )
        else:
            for method in SWEEP_METHODS:
                configs.append(
                    ExperimentConfig(
                        model_name=model_name,
                        press_name=method,
                        compression_ratio=ratio,
                        num_prompts=num_prompts,
                        seed=seed,
                        output_dir=output_dir,
                        max_new_tokens=max_new_tokens,
                        prompt_timeout_seconds=prompt_timeout_seconds,
                    )
                )
    return configs


def run_sweep(
    num_prompts: int = 50,
    seed: int = 42,
    output_dir: Path = Path("results"),
    max_new_tokens: int = 512,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    skip_existing: bool = True,
    prompt_timeout_seconds: float = 300.0,
    task: Task = DEFAULT_TASK,
) -> None:
    """Run the full compression sweep, reusing the model across configs."""
    configs = build_sweep_configs(
        num_prompts=num_prompts,
        seed=seed,
        output_dir=output_dir,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )

    pending = [c for c in configs if not (skip_existing and result_exists(c))]
    skipped = len(configs) - len(pending)
    if skipped:
        logger.info(f"Skipping {skipped} existing results")
    if not pending:
        logger.info("All configs already completed.")
        return

    prompts = task.load(num_prompts, seed)
    model, tokenizer, device = load_model(pending[0])

    for i, config in enumerate(pending, 1):
        logger.info(f"\n{'#' * 60}")
        logger.info(
            f"SWEEP [{i}/{len(pending)}] "
            f"{config.press_name} @ {config.compression_ratio}"
        )
        logger.info(f"{'#' * 60}")

        press = get_press(config.press_name, config.compression_ratio)
        results = run_prompts(
            model, tokenizer, device, prompts, config, press, task
        )
        print_summary(results, config)
        save_results(results, config)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
