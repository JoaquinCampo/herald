"""Full-model engineering smoke; no quality or prediction claim."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from herald_v3.engineering import engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = "Count from 1 to 100, writing every number in order."
    source = Path(engine.__file__)
    manifest = {
        "kind": "synthetic_engineering_smoke_not_quality_acceptance",
        "prompt": prompt,
        "model_snapshot": args.model,
        "max_new_tokens": 40,
        "engine_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "kvpress")
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("Loading pinned model offline", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    eos = model.generation_config.eos_token_id
    eos_ids = frozenset(eos if isinstance(eos, list) else [eos])
    print("Model loaded; running paired engine gates", flush=True)
    result = engine.run_acceptance(
        model, input_ids, max_new_tokens=40, eos_ids=eos_ids
    )
    payload = result.to_dict()
    (args.output / "result.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": result.passed,
                "gates": [gate.to_dict() for gate in result.gates],
            }
        ),
        flush=True,
    )
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
