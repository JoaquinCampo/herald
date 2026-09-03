"""Run the faithful resumable IFEval intervention regeneration.

Example::

    uv run python scripts/run_intervention_sweep.py --model llama \
        --prompts 200 --stride 16 --results-dir results/ifeval-intervention
"""

import argparse
import json
from pathlib import Path

import torch

from herald.config import MODELS, RATIOS, Config
from herald.generate import load_model
from herald.intervention_sweep import (
    APPROVED_COMPRESSORS,
    GpuContentionError,
    GpuMonitoringError,
    InterventionSweepConfig,
    guard_gpu_contention,
    initialize_intervention_config,
    run_intervention_sweep,
)
from herald.sweep_provenance import initialize_sweep_config
from herald.tasks import load_prompts


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "expected a non-empty comma-separated list"
        )
    return values


def _ratios(value: str) -> tuple[float, ...]:
    try:
        return tuple(float(item) for item in _csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ratios must be decimal numbers"
        ) from error


def _progress(event: dict[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def _gpu_stop_event(
    error: GpuContentionError | GpuMonitoringError,
) -> dict[str, object]:
    if isinstance(error, GpuContentionError):
        return {
            "event": "gpu_contention",
            "own_pid": error.event.own_pid,
            "allowed_pids": list(error.event.allowed_pids),
            "unknown_pids": list(error.event.unknown_pids),
        }
    return {"event": "gpu_monitor_error", "error": str(error)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Faithful IFEval intervention sweep"
    )
    parser.add_argument("--model", choices=tuple(MODELS), default="llama")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompts", type=int, default=200)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument(
        "--compressors", type=_csv, default=APPROVED_COMPRESSORS
    )
    parser.add_argument("--ratios", type=_ratios, default=RATIOS)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/intervention")
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allowed-gpu-pid",
        action="append",
        type=int,
        default=[],
        help=(
            "repeatable compute PID allowed to coexist (e.g. Orion keepalive)"
        ),
    )
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"requested CUDA device {args.device!r} is unavailable")
    unknown = set(args.compressors) - set(APPROVED_COMPRESSORS)
    if unknown:
        parser.error(f"unknown compressor(s): {sorted(unknown)}")
    if any(ratio not in RATIOS for ratio in args.ratios):
        parser.error(f"ratios must be drawn from approved scope {RATIOS}")
    standard_config = Config(
        models=[args.model],
        tasks=["ifeval"],
        compressors=list(args.compressors),
        ratios=list(args.ratios),
        switch_stride=args.stride,
        prompts_per_task=args.prompts,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn,
        results_dir=args.results_dir,
        ref_batch_size=1,
        hybrid_batch_size=1,
    )
    initialize_sweep_config(args.results_dir, standard_config)
    config = InterventionSweepConfig(
        model_key=args.model,
        task="ifeval",
        prompt_count=args.prompts,
        stride=args.stride,
        compressors=args.compressors,
        ratios=args.ratios,
        max_new_tokens=args.max_new_tokens,
        results_dir=args.results_dir,
        model_id=args.model_id,
        dtype=args.dtype,
        attn_implementation=args.attn,
        device=args.device,
        seed=args.seed,
    )
    initialize_intervention_config(args.results_dir, config)
    if args.device.startswith("cuda"):
        try:
            guard_gpu_contention(args.allowed_gpu_pid)
        except (GpuContentionError, GpuMonitoringError) as error:
            _progress(_gpu_stop_event(error))
            return 75
    _progress({"event": "loading_prompts", "count": args.prompts})
    records = load_prompts("ifeval", args.prompts)
    _progress(
        {
            "event": "loading_model",
            "model_id": args.model_id or MODELS[args.model],
        }
    )
    lm = load_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn,
        model_id=args.model_id,
    )
    try:
        counts = run_intervention_sweep(
            lm,
            records,
            config,
            allowed_gpu_pids=args.allowed_gpu_pid,
            progress=_progress,
        )
    except (GpuContentionError, GpuMonitoringError) as error:
        _progress(_gpu_stop_event(error))
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return 75
    _progress({"event": "sweep_complete", **counts})
    # Keep the model loaded only for the duration of the command.  Explicitly
    # release allocator blocks after a successful run as well.
    del lm
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
