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
from herald.runner import run_sweep


def _csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="HERALD generation sweep")
    p.add_argument("--models", type=_csv, default=["llama", "qwen3"])
    p.add_argument("--tasks", type=_csv, default=["gsm8k", "humaneval"])
    p.add_argument(
        "--compressors", type=_csv, default=list(COMPRESSORS)
    )
    p.add_argument(
        "--ratios",
        type=lambda v: [float(x) for x in _csv(v)],
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
        "--tap-layers",
        type=lambda v: tuple(int(x) for x in _csv(v)),
        default=(),
    )
    args = p.parse_args()

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
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    (args.results_dir / "config.json").write_text(
        config.model_dump_json(indent=2)
    )
    print(json.dumps({"event": "sweep_config", **json.loads(
        config.model_dump_json()
    )}))
    run_sweep(config, device=args.device)


if __name__ == "__main__":
    main()
