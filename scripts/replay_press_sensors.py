"""Replay locked, label-free M2 compressor-action sensors.

The command opens only references whose IDs occur in the frozen v2 development
allowlist.  It performs no continuation, evaluator, or quality scoring.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--v2-lock", type=Path, required=True)
    parser.add_argument("--sensor-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default="llama")
    parser.add_argument("--task", default="ifeval")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--allowed-gpu-pid", type=int, action="append", default=[]
    )
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids")
    return parser.parse_args()


def main() -> None:
    from herald.intervention_sweep import guard_gpu_contention
    from herald.sensor_replay import (
        ReplayConfig,
        locked_prompt_ids,
        replay_locked,
        validate_sensor_lock,
    )

    args = parse_args()
    validate_sensor_lock(args.sensor_lock, args.evidence, args.v2_lock)
    if args.device.startswith("cuda"):
        guard_gpu_contention(args.allowed_gpu_pid)
    # Validate scope before constructing/loading a model.  This also makes a
    # quarantined one-prompt pilot fail before opening any reference JSON.
    allowed, _, _ = locked_prompt_ids(args.evidence, args.v2_lock)
    if args.prompt_ids is not None and any(
        item not in allowed for item in args.prompt_ids
    ):
        bad = next(item for item in args.prompt_ids if item not in allowed)
        raise SystemExit(f"refusing unexpected or quarantined prompt: {bad}")
    config = ReplayConfig(
        results_root=args.results_root,
        evidence_path=args.evidence,
        lock_path=args.v2_lock,
        sensor_lock_path=args.sensor_lock,
        output_root=args.output_root,
        model_key=args.model,
        task=args.task,
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        allowed_gpu_pids=tuple(args.allowed_gpu_pid),
    )
    count = replay_locked(config, prompt_ids=args.prompt_ids)
    print(f"wrote {count} sensor records")


if __name__ == "__main__":
    main()
