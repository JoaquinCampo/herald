"""Experiment runner: load model, compress KV cache,
generate, extract signals, detect catastrophes."""

import datetime as _dt
import gc
import json
import math
import subprocess as _sp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from herald.metrics.replay import ReplayMetrics

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from herald.config import (
    ExperimentConfig,
    GenerationArtifact,
    RunResult,
    compute_prompt_hash,
    make_run_id,
)
from herald.detectors import (
    detect_all,
    detect_catastrophe_onsets,
)
from herald.policy import FixedRatioPolicy, Policy
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


# SnapKV asserts query_length > window_size. The kvpress default
# (64) blows up on short GSM8K prompts (chat-templated length 64).
# 32 keeps a meaningful recent-attention window while passing the
# assertion for every prompt observed in the Phase 0 manifest
# (shortest chat-templated q_len = 64). See gold/phase-0-results.md.
SNAPKV_WINDOW_SIZE = 32


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

    if name == "snapkv":
        return SnapKVPress(
            compression_ratio=compression_ratio,
            window_size=SNAPKV_WINDOW_SIZE,
        )

    presses = {
        "streaming_llm": StreamingLLMPress,
        "knorm": KnormPress,
        "expected_attention": ExpectedAttentionPress,
        "tova": TOVAPress,
        "random": RandomPress,
    }
    if name not in presses:
        raise ValueError(
            f"Unknown press: {name}. Available: {['snapkv', *presses.keys()]}"
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


def _append_truncation_sidecar(
    output_root: Path,
    run_id: str,
    prompt_id: str,
    task_name: str,
    truncation_meta: dict[str, Any],
) -> None:
    """Append a single JSONL line with this run's truncation metadata.

    Layout: ``<output_root>/raw/truncation.jsonl``. One line per run.
    The Block 3 watchdog reads this to enforce "stop if LongBench
    truncation metadata is missing for truncated prompts." Empty
    ``truncation_meta`` is still recorded so absence is detectable.
    """
    sidecar = output_root / "raw" / "truncation.jsonl"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "task": task_name,
        **(truncation_meta or {}),
    }
    with sidecar.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _retained_kv_bytes(press: object | None) -> float:
    """Best-effort byte count of the press' retained KV at end of run.

    NaN where the press does not expose it (today: every kvpress press).
    See gold/research-plan.md Phase 1 for the upstream-PR follow-up.
    """
    if press is None:
        return float("nan")
    fn = getattr(press, "retained_kv_bytes", None)
    if callable(fn):
        try:
            return float(fn())
        except Exception:  # noqa: BLE001
            return float("nan")
    return float("nan")


def _generate_compressed(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    policy: Policy,
    task: Task = DEFAULT_TASK,
) -> tuple[GenerationArtifact, dict[str, Any]]:
    """Run compressed generation and capture artifact + extras.

    Replay is the caller's responsibility. `compressed_scores` are kept
    on the model's device until replay consumes them.

    The single-press fast path (FixedRatioPolicy) calls
    `model.generate(...)` inside one `policy.make_context(model)`. The
    headline Phase 1 sweep uses this path exclusively.

    Cost telemetry is captured at generation boundaries: one
    `cuda.synchronize()` at end-of-generate (required for meaningful
    wall-clock), no per-token syncs (would perturb timing).
    """
    if policy.requires_manual_decode:
        raise NotImplementedError(
            "Manual-decode policies (e.g. SwitchAtOffsetPolicy) are "
            "not implemented yet; lands before Block 4."
        )

    question_text, truncation_meta = task.format_prompt(
        prompt_data,
        tokenizer,
        system_prompt=prompt_data.get("system_prompt"),
        max_new_tokens=config.max_new_tokens,
    )
    messages = format_chat(
        question_text,
        system_prompt=prompt_data.get("system_prompt"),
    )
    chat_text = tokenizer.apply_chat_template(  # type: ignore[attr-defined]
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)  # type: ignore[operator]
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    stopping = StoppingCriteriaList(
        [_TimeoutCriteria(config.prompt_timeout_seconds)]
    )

    press, ctx = policy.make_context(model)

    is_cuda = device.startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    t_start = time.perf_counter()

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

    if is_cuda:
        torch.cuda.synchronize()
    t_end = time.perf_counter()
    elapsed = t_end - t_start
    peak_mem_mb = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if is_cuda
        else float("nan")
    )

    generated_ids = outputs.sequences[0, input_len:].tolist()
    generated_text = tokenizer.decode(  # type: ignore[attr-defined]
        generated_ids, skip_special_tokens=True
    )

    eos_id = tokenizer.eos_token_id  # type: ignore[attr-defined]
    eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    hit_eos = len(generated_ids) > 0 and generated_ids[-1] in eos_ids
    hit_max = len(generated_ids) >= config.max_new_tokens
    stop_reason = (
        "eos" if hit_eos else ("max_tokens" if hit_max else "timeout")
    )

    n_gen = len(generated_ids)
    wall_per_token = elapsed / n_gen if n_gen > 0 else float("nan")

    signals = []
    state = None
    compressed_scores: list[torch.Tensor] = []
    for score in outputs.scores:
        sig, state = extract_signals(score[0], prev=state)
        signals.append(sig)
        compressed_scores.append(score[0].detach())

    if config.capture_attention and outputs.attentions is not None:
        lookbacks = compute_lookback_ratios(
            outputs.attentions, input_len=input_len
        )
        for sig, lb in zip(signals, lookbacks):
            sig.lookback_ratio = lb

    run_id = make_run_id(
        prompt_data["id"],
        config.press_name,
        config.compression_ratio,
        config.seed,
    )
    artifact = GenerationArtifact(
        run_id=run_id,
        input_ids=input_ids,
        input_len=input_len,
        generated_token_ids=generated_ids,
        compressed_scores=compressed_scores,
    )
    extras = {
        "chat_text": chat_text,
        "generated_text": generated_text,
        "stop_reason": stop_reason,
        "signals": signals,
        "prompt_data": prompt_data,
        "wall_clock_per_token": wall_per_token,
        "peak_memory_mb": peak_mem_mb,
        "kv_size_at_end": _retained_kv_bytes(press),
        "policy_name": type(policy).__name__,
        "truncation_meta": truncation_meta,
    }
    return artifact, extras


def _finalize_run_result(
    artifact: GenerationArtifact,
    extras: dict[str, Any],
    config: ExperimentConfig,
    device: str,
    baseline_run_id: str | None,
    replay_status: str,
    replay_error: str | None,
    task: Task,
) -> RunResult:
    """Build a RunResult from a generated artifact + extras."""
    prompt_data = extras["prompt_data"]
    generated_text = extras["generated_text"]
    stop_reason = extras["stop_reason"]
    chat_text = extras["chat_text"]

    catastrophes = detect_all(
        generated_text,
        artifact.generated_token_ids,
        stop_reason,
        prompt_data["ground_truth"],
        is_wrong_fn=task.is_wrong,
    )
    catastrophe_onsets = detect_catastrophe_onsets(
        artifact.generated_token_ids, stop_reason, catastrophes
    )
    predicted = task.parse_answer(generated_text)
    correct = (
        "wrong_answer" not in catastrophes if predicted is not None else None
    )

    self_baseline = make_run_id(prompt_data["id"], "none", 0.0, config.seed)
    return RunResult(
        run_id=artifact.run_id,
        prompt_id=prompt_data["id"],
        prompt_text=chat_text,
        prompt_hash=compute_prompt_hash(chat_text),
        model=config.model_name,
        dtype="float16",
        device_class=_device_class(device),
        press=config.press_name,
        compression_ratio=config.compression_ratio,
        max_new_tokens=config.max_new_tokens,
        seed=config.seed,
        decoding_config={"do_sample": "False"},
        task=task.name,
        baseline_run_id=baseline_run_id or self_baseline,
        generated_text=generated_text,
        generated_token_ids=artifact.generated_token_ids,
        ground_truth=prompt_data["ground_truth"],
        predicted_answer=predicted,
        correct=correct,
        stop_reason=stop_reason,
        catastrophes=catastrophes,
        num_tokens_generated=len(artifact.generated_token_ids),
        catastrophe_onsets=catastrophe_onsets,
        signals=extras["signals"],
        replay_status=replay_status,
        replay_error=replay_error,
        created_at=_dt.datetime.utcnow().isoformat() + "Z",
        herald_git_sha=_git_sha(),
        wall_clock_per_token=extras.get("wall_clock_per_token", float("nan")),
        peak_memory_mb=extras.get("peak_memory_mb", float("nan")),
        kv_size_at_end=extras.get("kv_size_at_end", float("nan")),
        policy_name=extras.get("policy_name", "FixedRatioPolicy"),
        replay_wall_clock_seconds=extras.get(
            "replay_wall_clock_seconds", float("nan")
        ),
    )


def _git_sha() -> str:
    try:
        return (
            _sp.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parent.parent.parent,
                stderr=_sp.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _device_class(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device.startswith("mps"):
        return "mps"
    return "cpu"


def _policy_for(config: ExperimentConfig, policy: Policy | None) -> Policy:
    if policy is not None:
        return policy
    return FixedRatioPolicy(
        press_name=config.press_name,
        compression_ratio=config.compression_ratio,
    )


def run_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    policy: Policy | None = None,
    task: Task = DEFAULT_TASK,
) -> RunResult:
    """Run generation for a single prompt and extract signals.

    `policy` defaults to `FixedRatioPolicy(config.press_name,
    config.compression_ratio)` — the Phase 0 / Phase 1 headline path.
    """
    artifact, extras = _generate_compressed(
        model,
        tokenizer,
        device,
        prompt_data,
        config,
        _policy_for(config, policy),
        task,
    )
    return _finalize_run_result(
        artifact=artifact,
        extras=extras,
        config=config,
        device=device,
        baseline_run_id=None,
        replay_status="pending",
        replay_error=None,
        task=task,
    )


def run_single_with_replay(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompt_data: dict[str, Any],
    config: ExperimentConfig,
    baseline_run_id: str | None,
    output_root: Path,
    policy: Policy | None = None,
    task: Task = DEFAULT_TASK,
    top_k: int = 128,
) -> tuple[RunResult, "ReplayMetrics"]:
    """Run compressed generation + inline replay, then write 3 parquets.

    `baseline_run_id=None` means this run is itself a baseline; it
    self-links via baseline_run_id == run_id.

    `policy` defaults to `FixedRatioPolicy(config.press_name,
    config.compression_ratio)` — the Phase 0 / Phase 1 headline path.
    """
    from herald.metrics.io import (
        PerRunPaths,
        signals_to_token_rows,
        write_run_record,
        write_tokens_rows,
    )
    from herald.metrics.replay import ReplayMetrics, replay_run

    artifact, extras = _generate_compressed(
        model,
        tokenizer,
        device,
        prompt_data,
        config,
        _policy_for(config, policy),
        task,
    )

    replay_status = "ok"
    replay_error: str | None = None
    rm: ReplayMetrics
    is_cuda = device.startswith("cuda")
    t_replay_start = time.perf_counter()
    try:
        rm = replay_run(
            model=model,
            tokenizer=tokenizer,
            artifact=artifact,
            output_root=output_root,
            top_k=top_k,
        )
    except Exception as exc:  # noqa: BLE001
        replay_status = "failed"
        replay_error = repr(exc)
        rm = ReplayMetrics(
            run_id=artifact.run_id,
            num_positions=0,
            js_full_max=0.0,
            js_full_mean=0.0,
        )
    if is_cuda:
        torch.cuda.synchronize()
    extras["replay_wall_clock_seconds"] = time.perf_counter() - t_replay_start

    rr = _finalize_run_result(
        artifact=artifact,
        extras=extras,
        config=config,
        device=device,
        baseline_run_id=baseline_run_id,
        replay_status=replay_status,
        replay_error=replay_error,
        task=task,
    )

    paths = PerRunPaths(root=output_root, run_id=rr.run_id)
    token_strs = [
        tokenizer.decode([tid], skip_special_tokens=True)  # type: ignore[attr-defined]
        for tid in artifact.generated_token_ids
    ]
    write_tokens_rows(
        signals_to_token_rows(
            run_id=rr.run_id,
            generated_token_ids=artifact.generated_token_ids,
            token_strs=token_strs,
            signals=extras["signals"],
        ),
        paths.tokens,
    )
    write_run_record(rr.model_dump(mode="json"), paths.run)
    _append_truncation_sidecar(
        output_root=output_root,
        run_id=rr.run_id,
        prompt_id=prompt_data["id"],
        task_name=task.name,
        truncation_meta=extras.get("truncation_meta", {}),
    )

    return rr, rm


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


@dataclass
class CostWatchdog:
    """Block 3 budget guard: abort the cell if observed wall-clock per
    token drifts above the predicted budget.

    Block 2 produces a per-cell predicted seconds-per-token; the
    watchdog is constructed with that prediction. After every prompt
    completes, `observe()` updates a rolling mean of recent runs and
    returns True (abort) when the rolling mean exceeds
    `predicted_s_per_token * tolerance` for `consecutive_breaches`
    consecutive prompts.

    Block 1 ships the class; Block 3 wires it into the sweep loop with
    the actual budget loaded from `gold/phase-1-cost-budget.json`.
    """

    predicted_s_per_token: float
    tolerance: float = 1.25
    consecutive_breaches: int = 3
    # Don't even consider tripping until this many finite observations
    # have accumulated. Avoids overreacting on the first one or two
    # warm-up prompts of a cell.
    min_observations_before_trip: int = 3
    history: list[float] = field(default_factory=list)
    _streak: int = 0

    def observe(self, result: RunResult) -> bool:
        return self.observe_wpt(result.wall_clock_per_token)

    def observe_wpt(self, wpt: float) -> bool:
        """Same as `observe` but takes the raw wpt directly. Lets the
        Block 3 sweep avoid materializing a RunResult from parquet."""
        if not math.isfinite(wpt) or wpt <= 0.0:
            return False
        self.history.append(wpt)
        threshold = self.predicted_s_per_token * self.tolerance
        if wpt > threshold:
            self._streak += 1
        else:
            self._streak = 0
        if len(self.history) < self.min_observations_before_trip:
            return False
        return self._streak >= self.consecutive_breaches


def run_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    prompts: list[dict[str, Any]],
    config: ExperimentConfig,
    policy: Policy | None = None,
    task: Task = DEFAULT_TASK,
    cost_watchdog: "CostWatchdog | None" = None,
) -> list[RunResult]:
    """Run all prompts with a pre-loaded model.

    Supports per-prompt checkpointing. `policy` defaults to
    `FixedRatioPolicy(config.press_name, config.compression_ratio)`.

    `cost_watchdog`, if provided, is consulted after each prompt with
    the new RunResult's cost telemetry. If it returns `abort=True`,
    the loop stops cleanly. See `gold/research-plan.md` Block 3 budget
    discipline.
    """
    pol = _policy_for(config, policy)
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
                model, tokenizer, device, prompt_data, config, pol, task
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

        if cost_watchdog is not None and cost_watchdog.observe(result):
            logger.warning(
                "Cost watchdog triggered abort: observed wall-clock "
                "drift exceeds the budget. See cost_watchdog state."
            )
            break

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
    prompts = task.load(config.num_prompts, config.seed)
    results = run_prompts(
        model, tokenizer, device, prompts, config, None, task
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

        results = run_prompts(
            model, tokenizer, device, prompts, config, None, task
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
