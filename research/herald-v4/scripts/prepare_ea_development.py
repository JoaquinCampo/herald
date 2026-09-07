#!/usr/bin/env python3
"""Generate and freeze a fresh RULER EA development namespace."""

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
NAMESPACE = "ruler-ea-dev-v1"
RAW_DIR = ROOT / "data" / NAMESPACE / "raw"
DEFAULT_OUTPUT = ROOT / "data" / NAMESPACE / "manifest.json"
DEFAULT_METADATA = ROOT / "data" / NAMESPACE / "metadata.json"
DEFAULT_TOKENIZER = (
    Path("/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3")
    / "data"
    / "retrieval-tokenizer-a09a354"
)
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TASKS = (
    ("niah_single_2", 12, 2026090603),
    ("cwe", 8, 2026090604),
)
OLD_MANIFEST = ROOT / "data" / "ruler-pilot-v1" / "manifest.json"
CORPUS = ROOT / "data" / "ruler-pilot-v1" / "PaulGrahamEssays.json"
CORPUS_URLS = ROOT / "data" / "ruler-pilot-v1" / "PaulGrahamEssays_URLs.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    return {path: sha256_file(VENDOR / path) for path in relative_paths}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def ensure_inputs(tokenizer_path: Path) -> None:
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Qwen tokenizer directory not found: {tokenizer_path}")
    if not CORPUS.is_file() or not CORPUS_URLS.is_file():
        raise FileNotFoundError("bounded v1 corpus inputs are missing")
    target = VENDOR / "scripts" / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    if not target.is_symlink() or target.resolve() != CORPUS.resolve():
        raise RuntimeError(
            "RULER corpus link is not the existing v1 corpus; refusing to mutate vendor inputs"
        )


def task_config(task: str) -> dict[str, Any]:
    with (VENDOR / "scripts" / "synthetic.yaml").open(encoding="utf-8") as handle:
        configs = yaml.safe_load(handle)
    if task not in configs:
        raise KeyError(f"task {task!r} is absent from pinned synthetic.yaml")
    return configs[task]


def run_official_generator(task: str, count: int, seed: int, tokenizer_path: Path) -> Path:
    config = task_config(task)
    if config["task"] not in {"niah", "common_words_extraction"}:
        raise RuntimeError(f"unexpected task implementation for {task}: {config}")
    raw_path = RAW_DIR / task / "validation.jsonl"
    if raw_path.exists():
        rows = load_jsonl(raw_path)
        if len(rows) != count:
            raise RuntimeError(f"existing raw file has {len(rows)} rows, expected {count}")
        return raw_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
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
        str(count),
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
    if len(rows) != count:
        raise RuntimeError(f"official generator produced {len(rows)} rows at {raw_path}")
    return raw_path


def assign_folds(raw_rows: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], int]:
    assignments: dict[tuple[str, int], int] = {}
    for task, rows in raw_rows.items():
        def sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            index, row = item
            if task == "niah_single_2":
                return (int(row.get("token_position_answer", 0)), index)
            return (int(row.get("length", 0)), index)

        ordered = sorted(enumerate(rows), key=sort_key)
        for rank, (index, _) in enumerate(ordered):
            assignments[(task, index)] = rank % 4
    return assignments


def prompt_key(row: dict[str, Any]) -> str:
    return sha256_json({"task": row["task"], "prompt": row["prompt"], "answers": row["answers"]})


def build_manifest(output_path: Path, metadata_path: Path, tokenizer_path: Path) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    tokenizer_hashes = {
        str(path.relative_to(tokenizer_path)): sha256_file(path)
        for path in sorted(tokenizer_path.rglob("*"))
        if path.is_file()
    }
    raw_rows: dict[str, list[dict[str, Any]]] = {}
    raw_hashes: dict[str, str] = {}
    for task, count, seed in TASKS:
        path = run_official_generator(task, count, seed, tokenizer_path)
        rows = load_jsonl(path)
        if len(rows) != count or any(int(row.get("length", 0)) > 4096 for row in rows):
            raise RuntimeError(f"invalid official output for {task}")
        raw_rows[task] = rows
        raw_hashes[str(path.relative_to(ROOT))] = sha256_file(path)

    old_rows = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    old_keys = {prompt_key({**row, "task": row["task"], "prompt": row["prompt"], "answers": row["answers"]}) for row in old_rows}
    folds = assign_folds(raw_rows)
    generator_hashes = source_hashes()
    rows: list[dict[str, Any]] = []
    for task, count, seed in TASKS:
        for index, source_row in enumerate(raw_rows[task]):
            prompt = source_row["input"]
            answers = source_row["outputs"]
            row = {"task": task, "prompt": prompt, "answers": answers}
            key = prompt_key(row)
            if key in old_keys:
                raise RuntimeError(f"new prompt overlaps old pilot: {task}/{index}")
            raw_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            rendered_tokens = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "id": f"{NAMESPACE}-{task}-{index:03d}",
                    "group_id": f"{NAMESPACE}-{task}-{index:03d}",
                    "development_fold": folds[(task, index)],
                    "prompt": prompt,
                    "answers": answers,
                    "task": task,
                    "seed": seed,
                    "max_new_tokens": {"niah_single_2": 128, "cwe": 120}[task],
                    "provenance": {
                        "generator": "NVIDIA/RULER scripts/data/prepare.py",
                        "ruler_commit": RULER_COMMIT,
                        "generator_source_sha256": generator_hashes,
                        "corpus_sha256": sha256_file(CORPUS),
                        "corpus_urls_sha256": sha256_file(CORPUS_URLS),
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
                        "prompt_key_sha256": key,
                    },
                    "token_counts": {
                        "ruler_prompt_tokens": len(raw_tokens),
                        "qwen_chat_prompt_tokens": len(rendered_tokens),
                        "official_prompt_plus_generation_cap": source_row.get("length"),
                        "fixed_max_seq_length": 4096,
                    },
                }
            )
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 rows, found {len(rows)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fold_counts = {
        str(fold): {task: sum(row["development_fold"] == fold and row["task"] == task for row in rows) for task, _, _ in TASKS}
        for fold in range(4)
    }
    metadata = {
        "namespace": NAMESPACE,
        "schema_version": "ea_development.v1",
        "counts": {task: count for task, count, _ in TASKS},
        "seeds": {task: seed for task, _, seed in TASKS},
        "fixed_max_seq_length": 4096,
        "max_new_tokens": {"niah_single_2": 128, "cwe": 120},
        "fold_rule": "sort NIAH by token_position_answer and CWE by official length, then round-robin four folds",
        "fold_counts": fold_counts,
        "old_manifest_sha256": sha256_file(OLD_MANIFEST),
        "old_prompt_overlap_count": 0,
        "raw_jsonl_sha256": raw_hashes,
        "manifest_sha256": sha256_file(output_path),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path} and {metadata_path}")
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER)
    args = parser.parse_args()
    ensure_inputs(args.tokenizer_path)
    build_manifest(args.output, args.metadata, args.tokenizer_path)


if __name__ == "__main__":
    main()
