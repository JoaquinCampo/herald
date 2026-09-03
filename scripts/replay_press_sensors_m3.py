"""Replay locked, label-free M3 layer-band compressor sensors."""

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
    parser.add_argument("--m3-lock", type=Path, required=True)
    parser.add_argument("--m2-result-freeze", type=Path, required=True)
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
    from herald.sensor_replay_m3 import (
        ReplayConfig,
        replay_locked_m3,
        validate_m3_lock,
    )

    args = parse_args()
    validate_m3_lock(
        args.m3_lock,
        args.evidence,
        args.v2_lock,
        args.sensor_lock,
        args.m2_result_freeze,
    )
    config = ReplayConfig(
        results_root=args.results_root,
        evidence_path=args.evidence,
        lock_path=args.v2_lock,
        sensor_lock_path=args.sensor_lock,
        m3_lock_path=args.m3_lock,
        m2_result_freeze_path=args.m2_result_freeze,
        output_root=args.output_root,
        model_key=args.model,
        task=args.task,
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn_implementation,
        allowed_gpu_pids=tuple(args.allowed_gpu_pid),
    )
    count = replay_locked_m3(config, prompt_ids=args.prompt_ids)
    print(f"wrote {count} M3 layer-band records")


if __name__ == "__main__":
    main()
