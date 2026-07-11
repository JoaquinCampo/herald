# pyright: reportMissingImports=false

"""CLI to run a HERALD generation sweep (or slice).

Build a Config from flags, persist it next to the results for
reproducibility, then run the resumable sweep. Re-running with the same
flags resumes; already-stored cells are skipped.

Example (slice):
    uv run python scripts/run_sweep.py \\
        --models llama --tasks gsm8k --prompts 30 \\
        --results-dir results/slice --device cuda
"""

import argparse
import json
from pathlib import Path

from herald.config import COMPRESSORS, RATIOS, Config
from herald.expected_attention_stats import StatisticsArtifact
from herald.runner import run_sweep
from herald.sweep_provenance import initialize_sweep_config


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _ratios(value: str) -> list[float]:
    """Parse a comma-separated ratio list for argparse."""
    try:
        return [float(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ratios must be comma-separated finite decimal numbers"
        ) from error


def _tap_layers(value: str) -> tuple[int, ...]:
    """Parse a comma-separated attention-layer list for argparse."""
    try:
        return tuple(int(item) for item in _csv(value))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "tap layers must be comma-separated integers"
        ) from error


def main() -> None:
    p = argparse.ArgumentParser(description="HERALD generation sweep")
    p.add_argument("--models", type=_csv, default=["llama", "qwen3"])
    p.add_argument("--tasks", type=_csv, default=["gsm8k", "humaneval"])
    p.add_argument("--compressors", type=_csv, default=list(COMPRESSORS))
    p.add_argument(
        "--ratios",
        type=_ratios,
        default=list(RATIOS),
    )
    p.add_argument("--prompts", type=int, default=200)
    p.add_argument("--switch-stride", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--ref-batch", type=int, default=16)
    p.add_argument("--hybrid-batch", type=int, default=1)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--tap-attention", action="store_true")
    p.add_argument(
        "--expected-attention-stats",
        type=Path,
        default=None,
        help="frozen local ExpectedAttentionStatsPress artifact directory",
    )
    p.add_argument(
        "--tap-layers",
        type=_tap_layers,
        default=(),
    )
    args = p.parse_args()
    statistics_digest = (
        StatisticsArtifact.load(args.expected_attention_stats).digest
        if args.expected_attention_stats is not None
        else None
    )

    config = Config(
        models=args.models,
        tasks=args.tasks,
        compressors=args.compressors,
        ratios=args.ratios,
        prompts_per_task=args.prompts,
        switch_stride=args.switch_stride,
        seed=args.seed,
        dtype=args.dtype,
        attn_implementation=args.attn,
        ref_batch_size=args.ref_batch,
        hybrid_batch_size=args.hybrid_batch,
        results_dir=args.results_dir,
        tap_attention=args.tap_attention,
        tap_layer_indices=args.tap_layers,
        expected_attention_stats_path=args.expected_attention_stats,
        expected_attention_stats_sha256=statistics_digest,
    )
    initialize_sweep_config(args.results_dir, config)
    print(
        json.dumps(
            {"event": "sweep_config", **config.model_dump(mode="json")}
        )
    )
    run_sweep(config, device=args.device)


if __name__ == "__main__":
    main()
