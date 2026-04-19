"""Launch sweep on orion. Usage: python scripts/run_sweep.py [--num-prompts 500]"""

import sys
sys.path.insert(0, "src")

from herald.experiment import run_sweep

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    run_sweep(num_prompts=args.num_prompts, model_name=args.model)
