#!/usr/bin/env python3
"""Verify and freeze the 16 prompt variable-tracking pilot."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import yaml

from prepare_value_head_data import ROOT, VENDOR, load_jsonl, sha256_file, sha256_text


OUTPUT = ROOT / "data/vt-pilot-v1"
RAW = OUTPUT / "raw/vt/validation.jsonl"
SEED = 2026090750
COUNT = 16
MAX_LENGTH = 4096
MAX_NEW_TOKENS = 30
PROTECTED = (
    ROOT / "data/ruler-pilot-v1/manifest.json", ROOT / "data/ruler-ea-dev-v1/manifest.json",
    ROOT / "data/value-head-v1/discovery.jsonl", ROOT / "data/value-head-v1/evaluation.jsonl",
    ROOT / "data/functional-v1/discovery.jsonl", ROOT / "data/functional-v1/evaluation.jsonl",
    ROOT / "data/graded-v1/manifest.jsonl", ROOT / "data/value-level-v1/discovery.jsonl",
    ROOT / "data/value-level-v1/evaluation.jsonl",
)
GENERATOR = VENDOR / "scripts/data/synthetic/variable_tracking.py"
DATA_CONSTANTS = VENDOR / "scripts/data/synthetic/constants.py"
EVAL_CONSTANTS = VENDOR / "scripts/eval/synthetic/constants.py"
TASK_CONFIG = VENDOR / "scripts/synthetic.yaml"
TOKENIZER_DIR = ROOT.parent / "herald-v3/data/retrieval-tokenizer-a09a354"


def sha256(path: Path) -> str:
    return sha256_file(path)


def module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def official_template_and_verify() -> str:
    config = yaml.safe_load(TASK_CONFIG.read_text(encoding="utf-8"))
    expected = {"task": "variable_tracking", "args": {"type_haystack": "noise", "num_chains": 1, "num_hops": 4}}
    if config.get("vt") != expected:
        raise RuntimeError(f"vendored vt config mismatch: {config.get('vt')}")
    task = module(DATA_CONSTANTS, "vt_data_constants").TASKS["variable_tracking"]
    template = task["template"] + task["answer_prefix"]
    if task["tokens_to_generate"] != MAX_NEW_TOKENS or not {"{context}", "{query}", "{num_v}"} <= set(re.findall(r"\{[^}]+\}", template)):
        raise RuntimeError("vendored variable_tracking template or horizon mismatch")
    source = GENERATOR.read_text(encoding="utf-8")
    for required in ("args.type_haystack == 'noise'", "num_chains", "num_hops", "num_hops+1", "tokens_to_generate"):
        if required not in source:
            raise RuntimeError(f"vendored variable_tracking generator lacks {required}")
    scorer = module(EVAL_CONSTANTS, "vt_eval_constants")
    if scorer.TASKS["variable_tracking"]["metric_fn"] is not scorer.string_match_all:
        raise RuntimeError("variable_tracking is not mapped to string_match_all")
    if scorer.string_match_all(["A B C D E"], [["A", "B", "C", "D", "E"]]) != 100.0:
        raise RuntimeError("official string_match_all full score mismatch")
    if scorer.string_match_all(["A C"], [["A", "B", "C", "D", "E"]]) != 40.0:
        raise RuntimeError("official string_match_all partial score mismatch")
    return template


def run_official(template: str, tokenizer_path: Path) -> None:
    if RAW.exists():
        if len(load_jsonl(RAW)) != COUNT:
            raise RuntimeError(f"existing raw output has wrong count: {RAW}")
        return
    RAW.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(GENERATOR), "--save_dir", str(RAW.parent.parent), "--save_name", "vt",
        "--subset", "validation", "--tokenizer_path", str(tokenizer_path), "--tokenizer_type", "hf",
        "--max_seq_length", str(MAX_LENGTH), "--tokens_to_generate", str(MAX_NEW_TOKENS), "--num_samples", str(COUNT),
        "--random_seed", str(SEED), "--type_haystack", "noise", "--num_chains", "1", "--num_hops", "4",
        "--template", template]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(VENDOR / "scripts/data"), environment.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    subprocess.run(command, check=True, env=environment)
    if len(load_jsonl(RAW)) != COUNT:
        raise RuntimeError(f"official generator produced wrong count: {RAW}")


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    return load_jsonl(path) if path.suffix == ".jsonl" else json.loads(text)


def context_hash(prompt: str) -> str:
    lines = list(re.finditer(r"[^\n]+", prompt))
    if len(lines) < 2:
        raise ValueError("prompt lacks final question")
    first_newline = prompt.find("\n")
    context = prompt[first_newline + 1 : lines[-1].start()].strip()
    if not context:
        raise ValueError("prompt has empty context")
    return sha256_text(context)


def row_hashes(rows: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    return ({sha256_text(row["prompt"]) for row in rows}, {context_hash(row["prompt"]) for row in rows})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"cached tokenizer missing: {args.tokenizer_path}")
    global RAW
    RAW = args.output_dir / "raw/vt/validation.jsonl"
    template = official_template_and_verify()
    run_official(template, args.tokenizer_path)
    raw = load_jsonl(RAW)
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in (GENERATOR, DATA_CONSTANTS, EVAL_CONSTANTS, TASK_CONFIG)}
    provenance = {"generator": "NVIDIA/RULER scripts/data/synthetic/variable_tracking.py", "ruler_commit": "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a",
                  "generator_source_sha256": source_hashes, "config": "vendor/ruler/scripts/synthetic.yaml:vt",
                  "tokenizer_path": str(args.tokenizer_path), "tokenizer_sha256": {str(p.relative_to(args.tokenizer_path)): sha256(p) for p in args.tokenizer_path.rglob("*") if p.is_file()},
                  "runner_transform": "Qwen apply_chat_template(user, add_generation_prompt=True)",
                  "wrapper_source_sha256": sha256(Path(__file__))}
    rows = []
    for index, item in enumerate(raw):
        answers = item.get("outputs")
        if not isinstance(answers, list) or len(answers) != 5 or len(set(answers)) != 5:
            raise RuntimeError(f"official row {index} does not contain five distinct answers")
        prompt = item["input"]
        context = context_hash(prompt)
        rows.append({"id": f"vt-pilot-v1-{index:03d}", "group_id": f"vt-pilot-v1-context-{context}",
                     "prompt": prompt, "answers": answers, "task": "variable_tracking", "max_new_tokens": MAX_NEW_TOKENS,
                     "seed": SEED, "provenance": {**provenance, "source_index": item.get("index"),
                     "official_length": item.get("length"), "raw_context_sha256": context}})
    protected = [row for path in PROTECTED for row in read_rows(path)]
    prompts, contexts = row_hashes(rows)
    old_prompts, old_contexts = row_hashes(protected)
    overlaps = {"protected_exact_prompt": len(prompts & old_prompts), "protected_raw_context": len(contexts & old_contexts),
                "within_prompt": COUNT - len(prompts), "within_raw_context": COUNT - len(contexts)}
    if any(overlaps.values()):
        raise RuntimeError(f"blocked vt pilot overlaps: {overlaps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl, manifest = args.output_dir / "manifest.jsonl", args.output_dir / "manifest.json"
    jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {"namespace": "vt-pilot-v1", "schema_version": "vt_pilot.v1", "task": "variable_tracking", "count": COUNT,
                "seed": SEED, "fixed_max_seq_length": MAX_LENGTH, "max_new_tokens": MAX_NEW_TOKENS, "num_chains": 1, "num_hops": 4,
                "official_answer_count": 5, "protected_manifest_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in PROTECTED},
                "overlap_counts": overlaps, "raw_sha256": sha256(RAW), "manifest_sha256": {"jsonl": sha256(jsonl), "json": sha256(manifest)},
                "generator_source_sha256": source_hashes, "wrapper_source_sha256": provenance["wrapper_source_sha256"]}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
