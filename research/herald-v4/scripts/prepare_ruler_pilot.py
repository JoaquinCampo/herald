#!/usr/bin/env python3
"""Generate and freeze the 12-row RULER v4 development pilot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ruler"
RAW_DIR = ROOT / "data" / "ruler-pilot-v1" / "raw"
DEFAULT_OUTPUT = ROOT / "data" / "ruler-pilot-v1" / "manifest.json"
DEFAULT_TOKENIZER = (
    Path("/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3")
    / "data"
    / "retrieval-tokenizer-a09a354"
)
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
INITIAL_SEED = 2026090601
TASKS = (("niah_single_2", 8, 0), ("cwe", 4, 1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    relative_paths = (
        "scripts/data/prepare.py",
        "scripts/synthetic.yaml",
        "scripts/data/synthetic/niah.py",
        "scripts/data/synthetic/common_words_extraction.py",
        "scripts/data/tokenizer.py",
        "scripts/data/template.py",
        "scripts/data/manifest_utils.py",
        "scripts/data/synthetic/json/download_paulgraham_essay.py",
        "scripts/eval/evaluate.py",
        "scripts/eval/synthetic/constants.py",
    )
    return {
        path: sha256_file(VENDOR / path)
        for path in relative_paths
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_niah_corpus() -> None:
    source = ROOT / "data" / "ruler-pilot-v1" / "PaulGrahamEssays.json"
    target = VENDOR / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    if not source.is_file():
        raise FileNotFoundError(
            f"Missing bounded public corpus {source}. Run the documented corpus "
            "acquisition step before generating niah_single_2."
        )
    if target.is_symlink() or target.exists():
        if target.is_symlink() and target.resolve() == source.resolve():
            return
        if target.is_file():
            if sha256_file(target) == sha256_file(source):
                return
            raise RuntimeError(f"RULER corpus path exists with a different hash: {target}")
        raise RuntimeError(f"RULER corpus path is not a regular file: {target}")
    target.symlink_to(source)


def task_config(task: str) -> dict[str, Any]:
    with (VENDOR / "scripts" / "synthetic.yaml").open(encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    if task not in configs:
        raise KeyError(f"Task {task!r} is absent from the pinned synthetic.yaml")
    return configs[task]


def run_official_generator(
    task: str,
    num_samples: int,
    seed: int,
    tokenizer_path: Path,
) -> Path:
    config = task_config(task)
    if config["task"] not in {"niah", "common_words_extraction"}:
        raise RuntimeError(f"Unexpected pinned task implementation for {task}: {config}")
    raw_path = RAW_DIR / task / "validation.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if not raw_path.is_file() or len(load_jsonl(raw_path)) != num_samples:
        command = [
            sys.executable,
            str(VENDOR / "scripts" / "data" / "prepare.py"),
            "--save_dir",
            str(RAW_DIR),
            "--benchmark",
            "synthetic",
            "--task",
            task,
            "--tokenizer_path",
            str(tokenizer_path),
            "--tokenizer_type",
            "hf",
            "--max_seq_length",
            "4096",
            "--model_template_type",
            "base",
            "--num_samples",
            str(num_samples),
            "--random_seed",
            str(seed),
        ]
        environment = os.environ.copy()
        environment["PATH"] = f"{Path(sys.executable).parent}:{environment.get('PATH', '')}"
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(VENDOR / "scripts" / "data"), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        subprocess.run(command, check=True, env=environment)
    rows = load_jsonl(raw_path)
    if len(rows) != num_samples:
        raise RuntimeError(f"Official generator produced {len(rows)} rows at {raw_path}")
    for row in rows:
        if row.get("length", 0) > 4096:
            raise RuntimeError(f"Official sizing exceeded 4096 for {task}: {row['length']}")
    return raw_path


def build_manifest(output_path: Path, tokenizer_path: Path, initial_seed: int) -> None:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "The isolated v4 environment needs transformers to count Qwen tokens."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_hashes = {
        str(path.relative_to(tokenizer_path)): sha256_file(path)
        for path in sorted(tokenizer_path.rglob("*"))
        if path.is_file()
    }
    rows: list[dict[str, Any]] = []
    generator_hashes = source_hashes()
    corpus_path = ROOT / "data" / "ruler-pilot-v1" / "PaulGrahamEssays.json"
    corpus_urls = ROOT / "data" / "ruler-pilot-v1" / "PaulGrahamEssays_URLs.txt"
    for task, count, offset in TASKS:
        seed = initial_seed + offset
        raw_path = run_official_generator(task, count, seed, tokenizer_path)
        for index, source_row in enumerate(load_jsonl(raw_path)):
            prompt = source_row["input"]
            answers = source_row["outputs"]
            raw_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            rendered_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "id": f"ruler-pilot-v1-{task}-{index:03d}",
                    "prompt": prompt,
                    "answers": answers,
                    "task": task,
                    "seed": seed,
                    "max_new_tokens": {"niah_single_2": 128, "cwe": 120}[task],
                    "provenance": {
                        "generator": "NVIDIA/RULER scripts/data/prepare.py",
                        "ruler_commit": RULER_COMMIT,
                        "generator_source_sha256": generator_hashes,
                        "corpus_sha256": sha256_file(corpus_path),
                        "corpus_urls_sha256": sha256_file(corpus_urls),
                        "config": "vendor/ruler/scripts/synthetic.yaml",
                        "tokenizer_path": str(tokenizer_path),
                        "tokenizer_files_sha256": tokenizer_hashes,
                        "official_raw_record": {
                            "index": source_row.get("index"),
                            "length": source_row.get("length"),
                            "answer_prefix": source_row.get("answer_prefix"),
                            "token_position_answer": source_row.get("token_position_answer"),
                        },
                        "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)",
                    },
                    "token_counts": {
                        "ruler_prompt_tokens": len(raw_tokens),
                        "qwen_chat_prompt_tokens": len(rendered_tokens),
                        "official_prompt_plus_generation_cap": source_row.get("length"),
                        "fixed_max_seq_length": 4096,
                    },
                }
            )
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 pilot rows, found {len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--initial-seed", type=int, default=INITIAL_SEED)
    args = parser.parse_args()
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"Qwen tokenizer directory not found: {args.tokenizer_path}")
    ensure_niah_corpus()
    build_manifest(args.output, args.tokenizer_path, args.initial_seed)
    print(f"Wrote {args.output} with 12 rows")


if __name__ == "__main__":
    main()
