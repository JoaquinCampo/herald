"""Phase 4 decode-time press feasibility gate.

Substrate-feasibility only. No HERALD risk policy, no AIMD, no
LoopGuard. The job of this module is to prove that an existing
kvpress decode-time wrapper actually compresses during decode on
Qwen2.5-7B-Instruct, exposes a stable budget surface, and produces
artifacts that downstream Phase 4 stages can join.

See `gold/phase-4-controller-design.md` and
`gold/phase-4-controller-implementation-plan.md` for context.

The module is structured so that planning, config, schema, and
event aggregation are CPU-testable and do not import kvpress or a
model at import time. Heavy imports (torch, kvpress, transformers,
HF datasets) are deferred to the run path.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --- Constants -------------------------------------------------------

MIN_KVPRESS_VERSION = (0, 5, 3)
SEGMENT_K = 16

PRESS_DECODING_KNORM = "decoding_knorm"
PRESS_DMS_KNORM = "dms_knorm"
SUPPORTED_PRESSES = (PRESS_DECODING_KNORM, PRESS_DMS_KNORM)

DEFAULT_TARGET_SIZES = (256, 512, 1024, 2048)
DEFAULT_COMPRESSION_INTERVAL = 16
DEFAULT_MAX_NEW_TOKENS = 256
DEFAULT_NUM_PROMPTS = 3

# Schema columns. Kept here so tests can assert artifact shape
# without importing pyarrow or polars in CPU-only test paths.
RUNS_COLUMNS = (
    "run_id",
    "prompt_id",
    "task",
    "press",
    "target_size",
    "compression_interval",
    "model",
    "dtype",
    "seed",
    "input_len",
    "num_tokens_generated",
    "stop_reason",
    "wall_clock_seconds",
    "wall_clock_per_token",
    "peak_memory_mb",
    "compression_event_count",
    "hook_fire_count_total",
    "decode_compression_observed",
    "task_score",
    "correct",
    "predicted_answer",
    "ground_truth",
    "generated_text",
    "kvpress_version",
    "created_at",
)

SEGMENTS_COLUMNS = (
    "run_id",
    "segment_idx",
    "segment_start",
    "segment_end",
    "target_size",
    "threshold",
    "mean_entropy",
    "mean_top1_prob",
    "mean_eff_vocab_size",
    "mean_top1_top2_ratio",
    "mean_tail_mass",
    "compression_events_in_segment",
    "wall_clock_seconds_segment",
)

EVENTS_COLUMNS = (
    "run_id",
    "step_idx",
    "layer_idx",
    "event_type",
    "retained_cache_len_before",
    "retained_cache_len_after",
    "target_size",
    "threshold",
    "wall_clock_seconds",
)


# --- Config -----------------------------------------------------------


@dataclass(slots=True)
class FeasibilityConfig:
    """Single feasibility config (one press, one task, many prompts)."""

    press: str
    task: str = "gsm8k"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    num_prompts: int = DEFAULT_NUM_PROMPTS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    target_sizes: tuple[int, ...] = DEFAULT_TARGET_SIZES
    compression_interval: int = DEFAULT_COMPRESSION_INTERVAL
    seed: int = 42
    output_dir: Path = Path("results/phase4/feasibility")
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.press not in SUPPORTED_PRESSES:
            raise ValueError(
                f"Unsupported press {self.press!r}; "
                f"expected one of {SUPPORTED_PRESSES}"
            )
        if self.task != "gsm8k":
            raise ValueError(
                "Phase 4 feasibility v1 supports only task=gsm8k "
                "(HumanEval gated behind GSM8K passing first)."
            )
        if self.num_prompts <= 0:
            raise ValueError("num_prompts must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if self.compression_interval <= 0:
            raise ValueError("compression_interval must be > 0")
        if not self.target_sizes:
            raise ValueError("target_sizes must be non-empty")
        for ts in self.target_sizes:
            if ts <= 0:
                raise ValueError(f"target_size must be > 0; got {ts}")


@dataclass(slots=True)
class PlannedRun:
    press: str
    task: str
    target_size: int
    compression_interval: int
    num_prompts: int
    max_new_tokens: int


def plan_runs(config: FeasibilityConfig) -> list[PlannedRun]:
    """Expand a config into one PlannedRun per (target_size)."""
    return [
        PlannedRun(
            press=config.press,
            task=config.task,
            target_size=ts,
            compression_interval=config.compression_interval,
            num_prompts=config.num_prompts,
            max_new_tokens=config.max_new_tokens,
        )
        for ts in config.target_sizes
    ]


# --- Version gate -----------------------------------------------------


def _parse_version(v: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in v.split("."):
        digits = ""
        for ch in p:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def require_kvpress_version(
    min_version: tuple[int, ...] = MIN_KVPRESS_VERSION,
    _module: Any | None = None,
    _resolver: Any | None = None,
) -> str:
    """Refuse to run unless installed kvpress >= min_version.

    Resolution order: explicit `_module.__version__` (test path),
    then `importlib.metadata.version("kvpress")` (real path —
    kvpress < 0.5.x does not expose `__version__`).
    `_resolver` lets tests inject a fake metadata lookup.
    """
    version: str | None = None
    if _module is not None:
        version = getattr(_module, "__version__", None)
    if version is None:
        if _resolver is None:  # pragma: no cover - exercised on Orion
            import importlib.metadata as _md  # noqa: PLC0415

            def _default(name: str) -> str:
                return _md.version(name)

            resolver = _default
        else:
            resolver = _resolver
        try:
            version = resolver("kvpress")
        except Exception:  # noqa: BLE001
            version = None
    if version is None:
        raise RuntimeError(
            "kvpress version unresolvable via __version__ or "
            "importlib.metadata; cannot verify >= "
            f"{'.'.join(str(x) for x in min_version)}"
        )
    parsed = _parse_version(version)
    if parsed < min_version:
        wanted = ".".join(str(x) for x in min_version)
        raise RuntimeError(
            f"kvpress {version} installed; Phase 4 feasibility "
            f"requires >= {wanted}. Upgrade with "
            "`uv pip install -U kvpress` on Orion before launching."
        )
    return str(version)


# --- Event aggregation ------------------------------------------------


@dataclass(slots=True)
class TokenLog:
    """Per-token cheap signals in a feasibility run."""

    step_idx: int
    entropy: float
    top1_prob: float
    eff_vocab_size: float
    top1_top2_ratio: float
    tail_mass: float
    wall_clock_seconds: float


@dataclass(slots=True)
class CompressionEvent:
    step_idx: int
    layer_idx: int
    event_type: str  # "decode" or "prefill"
    retained_cache_len_before: int
    retained_cache_len_after: int
    target_size: int | None
    threshold: float | None
    wall_clock_seconds: float


def aggregate_segments(
    run_id: str,
    tokens: list[TokenLog],
    events: list[CompressionEvent],
    target_size: int | None,
    threshold: float | None,
    k: int = SEGMENT_K,
) -> list[dict[str, Any]]:
    """Aggregate per-token logs and events into K-sized segments."""
    if k <= 0:
        raise ValueError("k must be > 0")
    if not tokens:
        return []
    rows: list[dict[str, Any]] = []
    n = len(tokens)
    n_segments = math.ceil(n / k)
    for seg_idx in range(n_segments):
        start = seg_idx * k
        end = min(start + k, n)
        chunk = tokens[start:end]
        seg_events = [e for e in events if start <= e.step_idx < end]
        rows.append(
            {
                "run_id": run_id,
                "segment_idx": seg_idx,
                "segment_start": start,
                "segment_end": end,
                "target_size": target_size,
                "threshold": threshold,
                "mean_entropy": _mean(t.entropy for t in chunk),
                "mean_top1_prob": _mean(t.top1_prob for t in chunk),
                "mean_eff_vocab_size": _mean(t.eff_vocab_size for t in chunk),
                "mean_top1_top2_ratio": _mean(
                    t.top1_top2_ratio for t in chunk
                ),
                "mean_tail_mass": _mean(t.tail_mass for t in chunk),
                "compression_events_in_segment": len(seg_events),
                "wall_clock_seconds_segment": sum(
                    t.wall_clock_seconds for t in chunk
                ),
            }
        )
    return rows


def _mean(xs: Any) -> float:
    vals = [float(x) for x in xs]
    return sum(vals) / len(vals) if vals else float("nan")


def summarize_events(
    events: list[CompressionEvent],
) -> dict[str, Any]:
    """Compact summary of decode-time compression activity."""
    decode = [e for e in events if e.event_type == "decode"]
    prefill = [e for e in events if e.event_type == "prefill"]
    layers = sorted({e.layer_idx for e in decode})
    return {
        "total_events": len(events),
        "decode_events": len(decode),
        "prefill_events": len(prefill),
        "decode_layers_touched": layers,
        "decode_compression_observed": len(decode) > 0,
    }


# --- Artifact writers -------------------------------------------------


def write_report(
    output_dir: Path,
    payload: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "decoding_press_report.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def write_parquet(
    output_dir: Path,
    name: str,
    rows: list[dict[str, Any]],
    columns: tuple[str, ...],
) -> Path:
    """Write rows to parquet with stable column order.

    Imports pyarrow lazily so CPU-only tests can avoid the dep.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.parquet"
    if not rows:
        rows = []
    # Defensive: ensure every column exists in every row.
    normalized = [{c: r.get(c) for c in columns} for r in rows]
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pa.Table.from_pylist(
        normalized, schema=_arrow_schema(columns, normalized)
    )
    pq.write_table(table, path)
    return path


def _arrow_schema(
    columns: tuple[str, ...], rows: list[dict[str, Any]]
) -> Any:
    """Best-effort arrow schema; lets pyarrow infer per column."""
    import pyarrow as pa  # noqa: PLC0415

    if not rows:
        # Empty table — declare every column as null type so reads
        # don't crash. Downstream stages tolerate this.
        return pa.schema([(c, pa.null()) for c in columns])
    # Let pyarrow infer types from the data.
    return None


# --- Dry-run printing -------------------------------------------------


def render_dry_run(config: FeasibilityConfig) -> str:
    runs = plan_runs(config)
    lines = [
        "Phase 4 decode-time press feasibility (dry-run)",
        f"  press               : {config.press}",
        f"  task                : {config.task}",
        f"  model               : {config.model_name}",
        f"  num_prompts         : {config.num_prompts}",
        f"  max_new_tokens      : {config.max_new_tokens}",
        f"  compression_interval: {config.compression_interval}",
        f"  target_sizes        : {list(config.target_sizes)}",
        f"  output_dir          : {config.output_dir}",
        f"  planned runs        : {len(runs)} "
        f"(one per target_size, x {config.num_prompts} prompts each)",
        "",
        "Planned runs:",
    ]
    for r in runs:
        lines.append(
            f"  - press={r.press} task={r.task} "
            f"target_size={r.target_size} "
            f"interval={r.compression_interval} "
            f"prompts={r.num_prompts} "
            f"max_new_tokens={r.max_new_tokens}"
        )
    lines.append("")
    lines.append(
        "No model load. Pass without --dry-run to launch. "
        "Refuses to launch if kvpress < "
        f"{'.'.join(str(x) for x in MIN_KVPRESS_VERSION)}."
    )
    return "\n".join(lines)


# --- Press construction (lazy) ----------------------------------------


def build_press(
    press_name: str,
    target_size: int,
    compression_interval: int,
) -> Any:
    """Build the configured kvpress decode-time wrapper.

    Imports kvpress lazily. The exact parameter names follow the
    Phase 4 design doc; if upstream renames land, fix here.
    """
    from kvpress import (  # noqa: PLC0415
        DecodingPress,
        DMSPress,
        KnormPress,
    )

    if press_name == PRESS_DECODING_KNORM:
        return DecodingPress(
            base_press=KnormPress(),
            target_size=target_size,
            compression_interval=compression_interval,
        )
    if press_name == PRESS_DMS_KNORM:
        # DMS is threshold-based; we treat target_size as threshold
        # encoding for the v1 gate. The launch packet documents this.
        return DMSPress(
            base_press=KnormPress(),
            decoding=True,
        )
    raise ValueError(f"Unsupported press {press_name!r}")


# --- Per-event observer wrapper ---------------------------------------


@dataclass(slots=True)
class Recorder:
    """Holds the live event log and fire counter for one run.

    Returned by `instrument_press`. The caller passes the same press
    object to `with press(model): ...` and inspects the recorder
    afterward.
    """

    events: list[CompressionEvent]
    fire_count: int = 0


def instrument_press(
    press: Any,
    target_size: int | None,
    threshold: float | None,
) -> Recorder:
    """Monkey-patch `press.compress` to log every fire.

    kvpress's attention forward hook calls `press.compress(...)` via
    bound-method lookup (`self.compress` at fire time), so patching
    the attribute on the press instance is sufficient: subsequent
    hook fires resolve to the patched callable.

    Implementation detail: the patched method is *not* a bound
    method. kvpress always calls `press.compress(module, ...)` with
    the layer module as the first positional after the press. The
    original `compress` is captured in closure so we can forward.
    """
    original = press.compress
    recorder = Recorder(events=[])

    def patched(
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        recorder.fire_count += 1
        before = int(keys.shape[-2]) if hasattr(keys, "shape") else -1
        t0 = time.perf_counter()
        out = original(
            module, hidden_states, keys, values, attentions, kwargs
        )
        dt = time.perf_counter() - t0
        try:
            new_keys = out[0] if isinstance(out, tuple) else keys
            after = (
                int(new_keys.shape[-2]) if hasattr(new_keys, "shape") else -1
            )
        except Exception:  # noqa: BLE001
            after = before
        layer_idx = int(getattr(module, "layer_idx", -1))
        # Read target_size live from the press so the controller's
        # mid-run mutations get logged accurately. Falls back to the
        # constructor arg for presses (DMS) that don't expose it.
        live_ts = getattr(press, "target_size", target_size)
        recorder.events.append(
            CompressionEvent(
                step_idx=-1,
                layer_idx=layer_idx,
                event_type="unknown",
                retained_cache_len_before=before,
                retained_cache_len_after=after,
                target_size=int(live_ts) if live_ts is not None else None,
                threshold=threshold,
                wall_clock_seconds=dt,
            )
        )
        return out

    press.compress = patched
    return recorder


def classify_decode_events(events: list[CompressionEvent]) -> None:
    """Mark first per-layer event as prefill, rest as decode (in place).

    Heuristic: kvpress fires once per layer at end-of-prefill, then
    every `compression_interval` steps during decode. The first fire
    per layer is therefore prefill, subsequent fires are decode.
    Refine after the first Orion run if the actual ordering differs.
    """
    seen: set[int] = set()
    for ev in events:
        if ev.layer_idx in seen:
            ev.event_type = "decode"
        else:
            seen.add(ev.layer_idx)
            ev.event_type = "prefill"


# --- Run loop (Orion only) --------------------------------------------


def run_feasibility(  # noqa: PLR0915
    config: FeasibilityConfig,
) -> dict[str, Any]:
    """Execute the feasibility gate. Imports torch / model lazily.

    Writes runs.parquet, segments.parquet, events.parquet, and
    decoding_press_report.json into config.output_dir.
    """
    kvpress_version = require_kvpress_version()

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from herald.prompts import format_chat, load_gsm8k  # noqa: PLC0415
    from herald.signals import extract_signals  # noqa: PLC0415

    device = _resolve_device(config.device)
    if device == "cpu":
        raise RuntimeError(
            "Phase 4 feasibility refuses to run on CPU. "
            "Launch on Orion (CUDA)."
        )

    prompts = load_gsm8k(config.num_prompts, seed=config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name, torch_dtype=torch.float16
    ).to(device)  # type: ignore[arg-type]
    model.eval()

    runs_rows: list[dict[str, Any]] = []
    segments_rows: list[dict[str, Any]] = []
    events_rows: list[dict[str, Any]] = []

    for target_size in config.target_sizes:
        for prompt in prompts:
            run_id = (
                f"phase4-feas-{config.press}-ts{target_size}-{prompt['id']}"
            )
            press = build_press(
                config.press, target_size, config.compression_interval
            )
            recorder = instrument_press(
                press, target_size=target_size, threshold=None
            )

            messages = format_chat(prompt["question"])
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            torch.cuda.reset_peak_memory_stats()
            t_start = time.perf_counter()
            tokens: list[TokenLog] = []
            with torch.no_grad(), press(model):
                # Switch to decode phase after prefill: kvpress fires
                # one round of compress() during the first forward.
                # We approximate by stepping manually.
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            generated_ids = outputs.sequences[0, input_len:].tolist()
            generated_text = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            )
            n_gen = len(generated_ids)

            classify_decode_events(recorder.events)
            events = recorder.events

            state = None
            for i, score in enumerate(outputs.scores):
                sig, state = extract_signals(score[0], prev=state)
                top1 = sig.top1_prob
                top2 = (
                    math.exp(sig.top5_logprobs[1])
                    if len(sig.top5_logprobs) > 1
                    else float("nan")
                )
                ratio = float("nan")
                if top2 and not math.isnan(top2):
                    ratio = top1 / top2
                tokens.append(
                    TokenLog(
                        step_idx=i,
                        entropy=sig.entropy,
                        top1_prob=top1,
                        eff_vocab_size=sig.eff_vocab_size,
                        top1_top2_ratio=ratio,
                        tail_mass=sig.tail_mass,
                        wall_clock_seconds=float("nan"),
                    )
                )

            event_summary = summarize_events(events)
            stop_reason = _stop_reason(
                generated_ids, tokenizer, config.max_new_tokens
            )
            try:
                from herald.detectors import (
                    parse_gsm8k_answer,  # noqa: PLC0415
                )

                predicted = parse_gsm8k_answer(str(generated_text))
            except Exception:  # noqa: BLE001
                predicted = None
            gt = str(prompt["ground_truth"]).strip()
            correct: bool | None
            if predicted is None:
                correct = None
            else:
                correct = str(predicted).strip() == gt

            runs_rows.append(
                {
                    "run_id": run_id,
                    "prompt_id": prompt["id"],
                    "task": config.task,
                    "press": config.press,
                    "target_size": target_size,
                    "compression_interval": config.compression_interval,
                    "model": config.model_name,
                    "dtype": "float16",
                    "seed": config.seed,
                    "input_len": input_len,
                    "num_tokens_generated": n_gen,
                    "stop_reason": stop_reason,
                    "wall_clock_seconds": elapsed,
                    "wall_clock_per_token": (
                        elapsed / n_gen if n_gen else float("nan")
                    ),
                    "peak_memory_mb": peak_mem_mb,
                    "compression_event_count": len(events),
                    "hook_fire_count_total": recorder.fire_count,
                    "decode_compression_observed": event_summary[
                        "decode_compression_observed"
                    ],
                    "task_score": (
                        float(correct) if correct is not None else None
                    ),
                    "correct": correct,
                    "predicted_answer": predicted,
                    "ground_truth": prompt["ground_truth"],
                    "generated_text": generated_text,
                    "kvpress_version": kvpress_version,
                    "created_at": _utcnow(),
                }
            )

            segments_rows.extend(
                aggregate_segments(
                    run_id=run_id,
                    tokens=tokens,
                    events=events,
                    target_size=target_size,
                    threshold=None,
                    k=SEGMENT_K,
                )
            )
            for ev in events:
                events_rows.append(
                    {
                        "run_id": run_id,
                        "step_idx": ev.step_idx,
                        "layer_idx": ev.layer_idx,
                        "event_type": ev.event_type,
                        "retained_cache_len_before": (
                            ev.retained_cache_len_before
                        ),
                        "retained_cache_len_after": (
                            ev.retained_cache_len_after
                        ),
                        "target_size": ev.target_size,
                        "threshold": ev.threshold,
                        "wall_clock_seconds": ev.wall_clock_seconds,
                    }
                )

    write_parquet(config.output_dir, "runs", runs_rows, RUNS_COLUMNS)
    write_parquet(
        config.output_dir, "segments", segments_rows, SEGMENTS_COLUMNS
    )
    write_parquet(config.output_dir, "events", events_rows, EVENTS_COLUMNS)

    decode_seen = any(r["decode_compression_observed"] for r in runs_rows)
    report = {
        "press": config.press,
        "task": config.task,
        "model": config.model_name,
        "num_prompts": config.num_prompts,
        "target_sizes": list(config.target_sizes),
        "compression_interval": config.compression_interval,
        "max_new_tokens": config.max_new_tokens,
        "kvpress_version": kvpress_version,
        "n_runs": len(runs_rows),
        "n_events": len(events_rows),
        "decode_compression_observed": decode_seen,
        "pass_gate": decode_seen,
        "created_at": _utcnow(),
    }
    write_report(config.output_dir, report)
    return report


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _stop_reason(
    generated_ids: list[int], tokenizer: Any, max_new_tokens: int
) -> str:
    eos_id = tokenizer.eos_token_id
    eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    if generated_ids and generated_ids[-1] in eos_ids:
        return "eos"
    if len(generated_ids) >= max_new_tokens:
        return "max_tokens"
    return "other"


def _utcnow() -> str:
    import datetime as _dt  # noqa: PLC0415

    return _dt.datetime.utcnow().isoformat() + "Z"
