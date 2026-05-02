"""Quick Phase 0 smoke runner (unbuffered, prints progress).

Usage on orion:
    HERALD_MODEL_PATH=/tmp/qwen7b \\
    HF_HOME=/tmp/hf_cache HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \\
    TRANSFORMERS_OFFLINE=1 \\
    .venv/bin/python -u scripts/run_phase0_smoke.py
"""

import os
import sys
from pathlib import Path


def main() -> None:
    print("[smoke] starting", flush=True)
    from herald.phase0 import run_phase0_sweep

    print("[smoke] imports done", flush=True)
    model_path = os.environ.get(
        "HERALD_MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct"
    )
    n = int(os.environ.get("HERALD_NUM_PROMPTS", "2"))
    max_new = int(os.environ.get("HERALD_MAX_NEW_TOKENS", "64"))
    out_root = Path(
        os.environ.get("HERALD_OUTPUT_ROOT", "results/phase0_smoke")
    )
    print(f"[smoke] model={model_path} n={n} mnt={max_new}", flush=True)
    run_phase0_sweep(
        model_name=model_path,
        manifest_path=Path("gold/phase0-random-manifest.json"),
        presses=("streaming_llm",),
        ratios=(0.5,),
        max_new_tokens=max_new,
        output_root=out_root,
        prompt_timeout_seconds=180.0,
        num_prompts=n,
    )
    print("[smoke] done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
