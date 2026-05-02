"""Full Phase 0 sweep runner: 20 prompts x 2 presses x 3 ratios + 20 baselines.

Usage on orion (env-prefixed):
    HERALD_MODEL_PATH=/tmp/qwen7b \\
    HF_HOME=/tmp/hf_cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1 \\
    .venv/bin/python -u scripts/run_phase0_full.py
"""

import os
import sys
import time
from pathlib import Path


def main() -> None:
    print(f"[full] starting at {time.strftime('%H:%M:%S')}", flush=True)
    from herald.phase0 import run_phase0_sweep

    print("[full] imports done", flush=True)
    model_path = os.environ.get(
        "HERALD_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"
    )
    out_root = Path(
        os.environ.get("HERALD_OUTPUT_ROOT", "results/phase0")
    )
    n = int(os.environ.get("HERALD_NUM_PROMPTS", "20"))
    max_new = int(os.environ.get("HERALD_MAX_NEW_TOKENS", "512"))
    print(f"[full] model={model_path} n={n} mnt={max_new}", flush=True)
    t0 = time.time()
    run_phase0_sweep(
        model_name=model_path,
        manifest_path=Path("gold/phase0-random-manifest.json"),
        presses=("streaming_llm", "snapkv"),
        ratios=(0.5, 0.875, 0.9375),
        max_new_tokens=max_new,
        output_root=out_root,
        prompt_timeout_seconds=300.0,
        num_prompts=n,
    )
    print(
        f"[full] done in {time.time() - t0:.1f}s "
        f"at {time.strftime('%H:%M:%S')}",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
