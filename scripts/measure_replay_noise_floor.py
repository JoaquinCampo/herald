"""Measure the actual fp16 noise floor for replay sanity checks.

Runs 5 prompts with press=none, replays, and reports the
99th-percentile per-token JS along the trajectory plus the t=0
max-abs logit diff vs generation scores. The numbers it prints
are what gets pinned into Phase 0 gate 3 / gate 4 thresholds.

Requires Tasks 5/7/8 (replay_forward + run_single_with_replay)
and runs on Orion only.
"""

from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    from herald.config import ExperimentConfig
    from herald.experiment import (
        load_model,
        run_single_with_replay,
    )
    from herald.metrics.io import PerRunPaths
    from herald.tasks import DEFAULT_TASK

    cfg = ExperimentConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        press_name="none",
        compression_ratio=0.0,
        num_prompts=5,
        seed=42,
        output_dir=Path("results/noise_floor"),
        max_new_tokens=64,
        prompt_timeout_seconds=120.0,
    )
    model, tok, device = load_model(cfg)
    prompts = DEFAULT_TASK.load(num_prompts=5, seed=42)

    js_p99: list[float] = []
    for p in prompts:
        rr, _ = run_single_with_replay(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_data=p,
            config=cfg,
            baseline_run_id=None,
            output_root=cfg.output_dir,
        )
        rep = pl.read_parquet(
            PerRunPaths(root=cfg.output_dir, run_id=rr.run_id).replay
        )
        js_p99.append(float(np.percentile(rep["js_full"].to_numpy(), 99)))
    print("99th-pct per-token JS across 5 baseline runs:")
    for j in js_p99:
        print(f"  {j:.3e}")
    print(f"max:    {max(js_p99):.3e}")
    print(f"median: {np.median(js_p99):.3e}")


if __name__ == "__main__":
    main()
