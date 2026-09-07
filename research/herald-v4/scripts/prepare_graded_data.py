#!/usr/bin/env python3
"""Freeze the 16 prompt graded niah_multivalue pilot."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from prepare_functional_data import hashes as row_hashes
from prepare_value_head_data import (
    CORPUS, CORPUS_URLS, ROOT, TOKENIZER, VENDOR, TEMPLATE, ensure_inputs, load_jsonl,
    sha256_file, sha256_text, source_hashes, split_prompt, tokenizer_hashes,
)


OUTPUT = ROOT / "data/graded-v1"
RAW = OUTPUT / "raw/validation.jsonl"
PROTECTED = (
    ROOT / "data/ruler-pilot-v1/manifest.json",
    ROOT / "data/ruler-ea-dev-v1/manifest.json",
    ROOT / "data/value-head-v1/discovery.jsonl",
    ROOT / "data/value-head-v1/evaluation.jsonl",
    ROOT / "data/functional-v1/discovery.jsonl",
    ROOT / "data/functional-v1/evaluation.jsonl",
)
SEED = 2026090644
COUNT = 16


def run_raw(tokenizer_path: Path) -> None:
    if RAW.exists():
        if len(load_jsonl(RAW)) != COUNT:
            raise RuntimeError(f"existing raw output has the wrong count: {RAW}")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(VENDOR / "scripts/data/synthetic/niah.py"),
        "--save_dir", str(RAW.parent.parent), "--save_name", "raw", "--subset", "validation",
        "--tokenizer_path", str(tokenizer_path), "--tokenizer_type", "hf", "--max_seq_length", "4096",
        "--tokens_to_generate", "128", "--num_samples", str(COUNT), "--random_seed", str(SEED),
        "--type_haystack", "essay", "--type_needle_k", "words", "--type_needle_v", "numbers",
        "--num_needle_k", "1", "--num_needle_v", "4", "--num_needle_q", "1", "--template", TEMPLATE]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(VENDOR / "scripts/data"), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(command, check=True, env=environment)
    if len(load_jsonl(RAW)) != COUNT:
        raise RuntimeError(f"official generator produced the wrong count: {RAW}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))


def build_rows(raw: list[dict[str, Any]], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw):
        answers = item.get("outputs")
        if (not isinstance(answers, list) or len(answers) != 4 or len(set(answers)) != 4 or
                any(not isinstance(value, str) or len(value) != 7 or not value.isdigit() for value in answers)):
            raise RuntimeError(f"invalid four-value answer set at raw row {index}")
        prompt = item["input"]
        context, _ = split_prompt(prompt)
        context_hash = sha256_text(context)
        rows.append({"id": f"graded-v1-{index:03d}", "group_id": f"graded-v1-context-{context_hash}",
                     "prompt": prompt, "answers": answers, "task": "niah_multivalue", "max_new_tokens": 128,
                     "seed": SEED, "provenance": {**provenance, "source_index": item.get("index"),
                     "official_length": item.get("length"), "raw_context_sha256": context_hash}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    global RAW
    RAW = args.output_dir / "raw/validation.jsonl"
    ensure_inputs(args.tokenizer_path)
    source = source_hashes()
    provenance = {"generator": "NVIDIA/RULER scripts/data/synthetic/niah.py", "ruler_commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
                  "generator_source_sha256": source, "config": "vendor/ruler/scripts/synthetic.yaml:niah_multivalue",
                  "corpus_sha256": sha256_file(CORPUS), "corpus_urls_sha256": sha256_file(CORPUS_URLS),
                  "tokenizer_path": str(args.tokenizer_path), "tokenizer_files_sha256": tokenizer_hashes(args.tokenizer_path),
                  "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)",
                  "wrapper_source_sha256": sha256_file(Path(__file__))}
    run_raw(args.tokenizer_path)
    rows = build_rows(load_jsonl(RAW), provenance)
    prompts, contexts = row_hashes(rows)
    old = [row for path in PROTECTED[:2] for row in read_rows(path)]
    value = [row for path in PROTECTED[2:4] for row in read_rows(path)]
    functional = [row for path in PROTECTED[4:] for row in read_rows(path)]
    old_prompts, old_contexts = row_hashes(old)
    value_prompts, value_contexts = row_hashes(value)
    functional_prompts, functional_contexts = row_hashes(functional)
    overlaps = {"old_exact_prompt": len(prompts & old_prompts), "value_head_exact_prompt": len(prompts & value_prompts),
                "functional_exact_prompt": len(prompts & functional_prompts), "old_raw_context": len(contexts & old_contexts),
                "value_head_raw_context": len(contexts & value_contexts), "functional_raw_context": len(contexts & functional_contexts),
                "within_prompt": COUNT - len(prompts), "within_raw_context": COUNT - len(contexts)}
    if any(overlaps.values()):
        raise RuntimeError(f"blocked graded design overlaps: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.output_dir / "manifest.jsonl"
    manifest = args.output_dir / "manifest.json"
    jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"namespace": "graded-v1", "schema_version": "graded.v1", "task": "niah_multivalue", "count": COUNT,
                "seed": SEED, "fixed_max_seq_length": 4096, "max_new_tokens": 128,
                "generator_config": {"type_haystack": "essay", "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1},
                "protected_manifest_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED},
                "overlap_counts": overlaps, "raw_sha256": sha256_file(RAW), "manifest_sha256": {"jsonl": sha256_file(jsonl), "json": sha256_file(manifest)},
                "generator_source_sha256": source, "wrapper_source_sha256": provenance["wrapper_source_sha256"]}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
