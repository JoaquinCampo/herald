#!/usr/bin/env python3
"""Freeze discovery and locked evaluation prompts for the value-head study."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor" / "ruler"
OUTPUT = ROOT / "data" / "value-head-v1"
TOKENIZER = Path("/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/data/retrieval-tokenizer-a09a354")
CORPUS = ROOT / "data/ruler-pilot-v1/PaulGrahamEssays.json"
CORPUS_URLS = ROOT / "data/ruler-pilot-v1/PaulGrahamEssays_URLs.txt"
OLD_MANIFESTS = (ROOT / "data/ruler-pilot-v1/manifest.json", ROOT / "data/ruler-ea-dev-v1/manifest.json")
SPLITS = (("discovery", 48, 2026090635), ("evaluation", 48, 2026090636))
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. "
    "Make sure to memorize it. I will quiz you about the {type_needle_v} afterwards.\n"
    "{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the "
    "provided text? The special magic {type_needle_v} for {query} mentioned in "
    "the provided text are"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def source_hashes() -> dict[str, str]:
    paths = ("scripts/data/prepare.py", "scripts/synthetic.yaml", "scripts/data/synthetic/niah.py",
             "scripts/data/tokenizer.py", "scripts/data/template.py", "scripts/data/manifest_utils.py",
             "scripts/data/synthetic/constants.py")
    return {path: sha256_file(VENDOR / path) for path in paths}


def tokenizer_hashes(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def ensure_inputs(tokenizer_path: Path) -> None:
    if not tokenizer_path.is_dir() or not CORPUS.is_file() or not CORPUS_URLS.is_file():
        raise FileNotFoundError("bounded corpus or cached tokenizer is missing")
    corpus_link = VENDOR / "scripts/data/synthetic/json/PaulGrahamEssays.json"
    if not corpus_link.is_symlink() or corpus_link.resolve() != CORPUS.resolve():
        raise RuntimeError("refusing to alter or replace the pinned RULER corpus link")
    for old in OLD_MANIFESTS:
        if not old.is_file():
            raise FileNotFoundError(f"old v4 manifest is missing: {old}")


def generate_raw(split: str, count: int, seed: int, tokenizer_path: Path, raw_path: Path) -> None:
    if raw_path.exists():
        rows = load_jsonl(raw_path)
        if len(rows) != count:
            raise RuntimeError(f"existing {raw_path} has {len(rows)} rows, expected {count}")
        return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(VENDOR / "scripts/data/synthetic/niah.py"),
        "--save_dir", str(raw_path.parent.parent), "--save_name", split, "--subset", "validation",
        "--tokenizer_path", str(tokenizer_path), "--tokenizer_type", "hf", "--max_seq_length", "4096",
        "--tokens_to_generate", "128", "--num_samples", str(count), "--random_seed", str(seed),
        "--type_haystack", "essay", "--type_needle_k", "words", "--type_needle_v", "numbers",
        "--num_needle_k", "4", "--num_needle_v", "1", "--num_needle_q", "1", "--template", TEMPLATE]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(VENDOR / "scripts/data"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(command, check=True, env=environment)
    rows = load_jsonl(raw_path)
    if len(rows) != count:
        raise RuntimeError(f"official generator produced {len(rows)} rows for {split}")


def split_prompt(prompt: str) -> tuple[str, str]:
    lines = list(re.finditer(r"[^\n]+", prompt))
    if len(lines) < 2:
        raise ValueError("prompt lacks intro, raw context, and final question")
    final = lines[-1]
    separator = prompt.find("\n")
    context_end = final.start() - (prompt[final.start() - 1] == "\n")
    if separator < 0 or not prompt[separator + 1 : context_end]:
        raise ValueError("prompt raw context separator is missing")
    return prompt[separator + 1 : context_end], final.group(0).strip()


def old_prompt_hashes() -> set[str]:
    return {
        sha256_text(row["prompt"])
        for path in OLD_MANIFESTS
        for row in json.loads(path.read_text(encoding="utf-8"))
    }


def build_rows(split: str, seed: int, raw_rows: list[dict[str, Any]], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, raw in enumerate(raw_rows):
        prompt = raw["input"]
        context, _ = split_prompt(prompt)
        context_hash = sha256_text(context)
        rows.append({"id": f"value-head-v1-{split}-{index:03d}",
                     "group_id": f"value-head-v1-context-{context_hash}", "prompt": prompt,
                     "answers": raw["outputs"], "task": "niah_multikey_1", "max_new_tokens": 128,
                     "seed": seed, "provenance": {**provenance, "split": split,
                     "source_index": raw.get("index"), "official_length": raw.get("length"),
                     "raw_context_sha256": context_hash}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_inputs(args.tokenizer_path)
    source = source_hashes()
    provenance = {"generator": "NVIDIA/RULER scripts/data/synthetic/niah.py", "ruler_commit": RULER_COMMIT,
                  "generator_source_sha256": source, "config": "vendor/ruler/scripts/synthetic.yaml:niah_multikey_1",
                  "corpus_sha256": sha256_file(CORPUS), "corpus_urls_sha256": sha256_file(CORPUS_URLS),
                  "tokenizer_path": str(args.tokenizer_path), "tokenizer_files_sha256": tokenizer_hashes(args.tokenizer_path),
                  "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)"}
    old_hashes = old_prompt_hashes()
    all_rows: dict[str, list[dict[str, Any]]] = {}
    for split, count, seed in SPLITS:
        raw_path = args.output_dir / "raw" / split / "validation.jsonl"
        generate_raw(split, count, seed, args.tokenizer_path, raw_path)
        raw_rows = load_jsonl(raw_path)
        if len(raw_rows) != count or any(int(row.get("length", 0)) > 4096 for row in raw_rows):
            raise RuntimeError(f"invalid official output for {split}")
        all_rows[split] = build_rows(split, seed, raw_rows, provenance)
    prompt_hashes = {sha256_text(row["prompt"]) for rows in all_rows.values() for row in rows}
    if len(prompt_hashes) != 96:
        raise RuntimeError("duplicate prompts within or across generated splits")
    old_overlap = prompt_hashes & old_hashes
    if old_overlap:
        raise RuntimeError(f"new prompts overlap old v4 manifests: {len(old_overlap)}")
    contexts = {split: {row["provenance"]["raw_context_sha256"] for row in rows} for split, rows in all_rows.items()}
    collisions = contexts["discovery"] & contexts["evaluation"]
    if collisions:
        raise RuntimeError(f"cross-split parent context collision: {len(collisions)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {}
    for split, rows in all_rows.items():
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        manifest_hashes[split] = sha256_file(path)
    metadata = {
        "namespace": "value-head-v1",
        "schema_version": "value_head_discovery.v1",
        "task": "niah_multikey_1",
        "counts": {split: len(rows) for split, rows in all_rows.items()},
        "seeds": {split: seed for split, _, seed in SPLITS},
        "fixed_max_seq_length": 4096,
        "max_new_tokens": 128,
        "generator_config": {"type_haystack": "essay", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1},
        "old_manifest_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in OLD_MANIFESTS},
        "old_exact_prompt_overlap_count": 0,
        "cross_split_context_overlap_count": 0,
        "raw_jsonl_sha256": {split: sha256_file(args.output_dir / "raw" / split / "validation.jsonl") for split, _, _ in SPLITS},
        "manifest_sha256": manifest_hashes,
        "adapter_source_sha256": sha256_file(ROOT / "scripts/value_span_adapter.py"),
        "generator_source_sha256": sha256_file(ROOT / "scripts/prepare_value_head_data.py"),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
