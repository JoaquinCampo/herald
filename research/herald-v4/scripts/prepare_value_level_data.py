#!/usr/bin/env python3
"""Generate the frozen value-level niah_multivalue population."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from prepare_value_head_data import (
    CORPUS, CORPUS_URLS, ROOT, TOKENIZER, VENDOR, TEMPLATE, ensure_inputs,
    load_jsonl, sha256_file, sha256_text, source_hashes, split_prompt,
    tokenizer_hashes,
)


OUTPUT = ROOT / "data/value-level-v1"
SPLITS = (("discovery", 128, 2026090645), ("evaluation", 64, 2026090646))
PROTECTED = (
    ROOT / "data/ruler-pilot-v1/manifest.json", ROOT / "data/ruler-ea-dev-v1/manifest.json",
    ROOT / "data/value-head-v1/discovery.jsonl", ROOT / "data/value-head-v1/evaluation.jsonl",
    ROOT / "data/functional-v1/discovery.jsonl", ROOT / "data/functional-v1/evaluation.jsonl",
    ROOT / "data/graded-v1/manifest.jsonl",
)


def raw_path(output: Path, split: str) -> Path:
    return output / "raw" / split / "validation.jsonl"


def run_raw(output: Path, split: str, count: int, seed: int, tokenizer_path: Path) -> Path:
    path = raw_path(output, split)
    if path.exists():
        if len(load_jsonl(path)) != count:
            raise RuntimeError(f"existing raw output has wrong count: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(VENDOR / "scripts/data/synthetic/niah.py"),
        "--save_dir", str(path.parent.parent), "--save_name", split, "--subset", "validation",
        "--tokenizer_path", str(tokenizer_path), "--tokenizer_type", "hf", "--max_seq_length", "4096",
        "--tokens_to_generate", "128", "--num_samples", str(count), "--random_seed", str(seed),
        "--type_haystack", "essay", "--type_needle_k", "words", "--type_needle_v", "numbers",
        "--num_needle_k", "1", "--num_needle_v", "4", "--num_needle_q", "1", "--template", TEMPLATE]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(VENDOR / "scripts/data"), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(command, check=True, env=environment)
    if len(load_jsonl(path)) != count:
        raise RuntimeError(f"official generator produced wrong count: {path}")
    return path


def read_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))


def row_hashes(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    prompts, contexts = set(), set()
    for row in rows:
        prompts.add(sha256_text(row["prompt"]))
        context, _ = split_prompt(row["prompt"])
        contexts.add(sha256_text(context))
    return prompts, contexts


def build_rows(raw: list[dict[str, Any]], split: str, seed: int, provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw):
        answers = item.get("outputs")
        if (not isinstance(answers, list) or len(answers) != 4 or len(set(answers)) != 4 or
                any(not isinstance(value, str) or len(value) != 7 or not value.isdigit() for value in answers)):
            raise RuntimeError(f"invalid four distinct seven digit answers at raw row {index}")
        prompt = item["input"]
        context, _ = split_prompt(prompt)
        context_hash = sha256_text(context)
        rows.append({"id": f"value-level-v1-{split}-{index:03d}",
                     "group_id": f"value-level-v1-context-{context_hash}", "prompt": prompt,
                     "answers": answers, "task": "niah_multivalue", "max_new_tokens": 128,
                     "seed": seed, "provenance": {**provenance, "split": split,
                     "source_index": item.get("index"), "official_length": item.get("length"),
                     "raw_context_sha256": context_hash}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_inputs(args.tokenizer_path)
    source = source_hashes()
    provenance = {"generator": "NVIDIA/RULER scripts/data/synthetic/niah.py",
                  "ruler_commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a", "generator_source_sha256": source,
                  "config": "vendor/ruler/scripts/synthetic.yaml:niah_multivalue", "corpus_sha256": sha256_file(CORPUS),
                  "corpus_urls_sha256": sha256_file(CORPUS_URLS), "tokenizer_path": str(args.tokenizer_path),
                  "tokenizer_files_sha256": tokenizer_hashes(args.tokenizer_path),
                  "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)",
                  "wrapper_source_sha256": sha256_file(Path(__file__))}
    protected = [row for path in PROTECTED for row in read_rows(path)]
    protected_prompts, protected_contexts = row_hashes(protected)
    generated: dict[str, list[dict[str, Any]]] = {}
    for split, count, seed in SPLITS:
        generated[split] = build_rows(load_jsonl(run_raw(args.output_dir, split, count, seed, args.tokenizer_path)), split, seed, provenance)
    all_rows = [row for rows in generated.values() for row in rows]
    prompts, contexts = row_hashes(all_rows)
    overlaps = {"protected_exact_prompt": len(prompts & protected_prompts), "protected_raw_context": len(contexts & protected_contexts),
                "within_or_cross_prompt": len(all_rows) - len(prompts), "within_or_cross_raw_context": len(all_rows) - len(contexts)}
    if any(overlaps.values()):
        raise RuntimeError(f"blocked value-level design overlaps: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, dict[str, str]] = {}
    for split, rows in generated.items():
        jsonl, manifest = args.output_dir / f"{split}.jsonl", args.output_dir / f"{split}.json"
        jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_hashes[split] = {"jsonl": sha256_file(jsonl), "json": sha256_file(manifest)}
    metadata = {"namespace": "value-level-v1", "schema_version": "value_level.v1", "task": "niah_multivalue",
                "counts": {split: len(rows) for split, rows in generated.items()}, "seeds": {split: seed for split, _, seed in SPLITS},
                "fixed_max_seq_length": 4096, "max_new_tokens": 128,
                "generator_config": {"type_haystack": "essay", "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1},
                "protected_manifest_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in PROTECTED},
                "overlap_counts": overlaps, "raw_jsonl_sha256": {split: sha256_file(raw_path(args.output_dir, split)) for split, _, _ in SPLITS},
                "manifest_sha256": manifest_hashes, "generator_source_sha256": source,
                "wrapper_source_sha256": provenance["wrapper_source_sha256"]}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
