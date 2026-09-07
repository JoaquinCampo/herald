#!/usr/bin/env python3
"""Generate the functional matched NIAH population and both manifest forms."""

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_value_head_data import (
    CORPUS, CORPUS_URLS, OLD_MANIFESTS, ROOT, TOKENIZER, ensure_inputs,
    generate_raw, load_jsonl, sha256_file, sha256_text, source_hashes,
    split_prompt, tokenizer_hashes,
)


OUTPUT = ROOT / "data/functional-v1"
VALUE_MANIFESTS = (ROOT / "data/value-head-v1/discovery.jsonl", ROOT / "data/value-head-v1/evaluation.jsonl")
SPLITS = (("discovery", 48, 2026090641), ("evaluation", 48, 2026090642))


def rows_for(split: str, seed: int, raw: list[dict[str, Any]], provenance: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, item in enumerate(raw):
        prompt = item["input"]
        context, _ = split_prompt(prompt)
        context_hash = sha256_text(context)
        rows.append({"id": f"functional-v1-{split}-{index:03d}",
                     "group_id": f"functional-v1-context-{context_hash}", "prompt": prompt,
                     "answers": item["outputs"], "task": "niah_multikey_1", "max_new_tokens": 128,
                     "seed": seed, "provenance": {**provenance, "split": split,
                     "source_index": item.get("index"), "official_length": item.get("length"),
                     "raw_context_sha256": context_hash}})
    return rows


def hashes(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    prompts, contexts = set(), set()
    for row in rows:
        prompts.add(sha256_text(row["prompt"]))
        context, _ = split_prompt(row["prompt"])
        contexts.add(row.get("provenance", {}).get("raw_context_sha256", sha256_text(context)))
    return prompts, contexts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    ensure_inputs(args.tokenizer_path)
    source = source_hashes()
    provenance = {"generator": "NVIDIA/RULER scripts/data/synthetic/niah.py",
                  "ruler_commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
                  "generator_source_sha256": source, "config": "vendor/ruler/scripts/synthetic.yaml:niah_multikey_1",
                  "corpus_sha256": sha256_file(CORPUS), "corpus_urls_sha256": sha256_file(CORPUS_URLS),
                  "tokenizer_path": str(args.tokenizer_path), "tokenizer_files_sha256": tokenizer_hashes(args.tokenizer_path),
                  "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)",
                  "wrapper_source_sha256": sha256_file(Path(__file__))}
    protected_old = [row for path in OLD_MANIFESTS for row in json.loads(path.read_text(encoding="utf-8"))]
    protected_value = [row for path in VALUE_MANIFESTS for row in load_jsonl(path)]
    old_prompts, old_contexts = hashes(protected_old)
    value_prompts, value_contexts = hashes(protected_value)
    generated: dict[str, list[dict[str, Any]]] = {}
    for split, count, seed in SPLITS:
        raw_path = args.output_dir / "raw" / split / "validation.jsonl"
        generate_raw(split, count, seed, args.tokenizer_path, raw_path)
        raw = load_jsonl(raw_path)
        if len(raw) != count or any(int(item.get("length", 0)) > 4096 for item in raw):
            raise RuntimeError(f"invalid official output for {split}")
        generated[split] = rows_for(split, seed, raw, provenance)
    all_rows = [row for rows in generated.values() for row in rows]
    prompts, contexts = hashes(all_rows)
    overlaps = {"old_exact_prompt": len(prompts & old_prompts), "value_head_exact_prompt": len(prompts & value_prompts),
                "old_raw_context": len(contexts & old_contexts), "value_head_raw_context": len(contexts & value_contexts),
                "within_or_cross_prompt": len(all_rows) - len(prompts),
                "within_or_cross_raw_context": len(all_rows) - len(contexts)}
    if any(overlaps.values()):
        raise RuntimeError(f"blocked functional design overlaps: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_hashes: dict[str, dict[str, str]] = {}
    for split, rows in generated.items():
        jsonl = args.output_dir / f"{split}.jsonl"
        json_manifest = args.output_dir / f"{split}.json"
        jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
        json_manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_hashes[split] = {"jsonl": sha256_file(jsonl), "json": sha256_file(json_manifest)}
    metadata = {"namespace": "functional-v1", "schema_version": "functional.v1", "task": "niah_multikey_1",
                "counts": {split: len(rows) for split, rows in generated.items()},
                "seeds": {split: seed for split, _, seed in SPLITS}, "fixed_max_seq_length": 4096, "max_new_tokens": 128,
                "generator_config": {"type_haystack": "essay", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1},
                "old_manifest_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in OLD_MANIFESTS},
                "value_head_manifest_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in VALUE_MANIFESTS},
                "overlap_counts": overlaps, "raw_jsonl_sha256": {split: sha256_file(args.output_dir / "raw" / split / "validation.jsonl") for split, _, _ in SPLITS},
                "generator_source_sha256": source, "wrapper_source_sha256": provenance["wrapper_source_sha256"],
                "manifest_sha256": manifest_hashes}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
