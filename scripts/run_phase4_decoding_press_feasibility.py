"""Phase 4 decode-time press feasibility CLI.

Substrate-feasibility only. No HERALD risk policy yet.

Examples:

    # CPU-safe dry-run (no model load, no kvpress import).
    uv run python scripts/run_phase4_decoding_press_feasibility.py \\
        --press decoding_knorm --dry-run

    # Real run on Orion (CUDA, kvpress >= 0.5.3).
    uv run python scripts/run_phase4_decoding_press_feasibility.py \\
        --press decoding_knorm \\
        --task gsm8k \\
        --num-prompts 3 \\
        --max-new-tokens 256 \\
        --target-sizes 256 512 1024 \\
        --compression-interval 16 \\
        --output-dir results/phase4/feasibility
"""

import argparse
import sys
from pathlib import Path

from herald.phase4_feasibility import (
    DEFAULT_COMPRESSION_INTERVAL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_PROMPTS,
    DEFAULT_TARGET_SIZES,
    PRESS_DECODING_KNORM,
    SUPPORTED_PRESSES,
    FeasibilityConfig,
    render_dry_run,
    run_feasibility,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--press",
        choices=SUPPORTED_PRESSES,
        default=PRESS_DECODING_KNORM,
    )
    ap.add_argument("--task", default="gsm8k", choices=["gsm8k"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    ap.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS
    )
    ap.add_argument(
        "--target-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TARGET_SIZES),
    )
    ap.add_argument(
        "--compression-interval",
        type=int,
        default=DEFAULT_COMPRESSION_INTERVAL,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4/feasibility"),
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and exit. No model load, no kvpress.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = FeasibilityConfig(
        press=args.press,
        task=args.task,
        model_name=args.model,
        num_prompts=args.num_prompts,
        max_new_tokens=args.max_new_tokens,
        target_sizes=tuple(args.target_sizes),
        compression_interval=args.compression_interval,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    if args.dry_run:
        print(render_dry_run(config))
        return 0
    report = run_feasibility(config)
    print("Phase 4 feasibility complete:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    return 0 if report.get("pass_gate") else 1


if __name__ == "__main__":
    sys.exit(main())
