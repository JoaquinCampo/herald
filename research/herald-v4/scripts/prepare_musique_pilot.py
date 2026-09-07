#!/usr/bin/env python3
"""Freeze the fixed 16-row MuSiQue answerable two-hop pilot."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "musique-pilot-v1"
RAW_SOURCE = OUTPUT / "raw" / "musique_ans_dev.jsonl"
OFFICIAL_SOURCE = OUTPUT / "raw" / "musique_ans_dev_official_format.jsonl"
SELECTED_RAW = OUTPUT / "selected_raw.jsonl"
SELECTED_GOLD = OUTPUT / "selected_official.jsonl"
MANIFEST_JSONL = OUTPUT / "manifest.jsonl"
MANIFEST_JSON = OUTPUT / "manifest.json"
METADATA = OUTPUT / "metadata.json"
CONVERTER = (
    OUTPUT / "provenance" / "official_converter" / "raw_data_to_official_format.py"
)
SCORER = OUTPUT / "provenance" / "official_scorer" / "evaluate_v1.0.py"
TOKENIZER_DEFAULT = ROOT.parent / "herald-v3" / "data" / "retrieval-tokenizer-a09a354"

COUNT = 16
HORIZON = 64
CONTEXT_LIMIT = 32768
SEED = 0
TASK = "musique_ans_2hop"
SOURCE_URL = (
    "https://drive.google.com/file/d/1TRXU68wveSehVbQrRRtWsUsFkKUF43QS/view?usp=sharing"
)
SOURCE_DIRECT_URL = "https://drive.usercontent.google.com/download?id=1TRXU68wveSehVbQrRRtWsUsFkKUF43QS&export=download&confirm=t"
SOURCE_REPOSITORY = "https://github.com/stonybrooknlp/musique"
SOURCE_LICENSE_URL = (
    "https://raw.githubusercontent.com/stonybrooknlp/musique/main/LICENSE"
)
SOURCE_INFO_URL = "https://raw.githubusercontent.com/stonybrooknlp/musique/main/.all_data_information.json"
CONVERTER_URL = "https://raw.githubusercontent.com/stonybrooknlp/musique/main/raw_data_to_official_format.py"
SCORER_URL = (
    "https://raw.githubusercontent.com/stonybrooknlp/musique/main/evaluate_v1.0.py"
)
TEMPLATE_VERSION = "musique_qa_answer_only_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def context_text(row: dict[str, Any]) -> str:
    return "\n".join(
        f"[{paragraph['idx']}] {paragraph['title']}\n{paragraph['paragraph_text']}"
        for paragraph in row["paragraphs"]
    )


def group_key(row: dict[str, Any]) -> str:
    components = row["question_decomposition"]
    payload = {
        "question": normalized_text(row["question"]),
        "component_ids": sorted(str(item["id"]) for item in components),
        "component_answers": sorted(
            normalized_text(str(item["answer"])) for item in components
        ),
        "supporting_paragraphs": sorted(
            sha256_text(
                normalized_text(f"{paragraph['title']}\n{paragraph['paragraph_text']}")
            )
            for paragraph in row["paragraphs"]
            if paragraph["is_supporting"]
        ),
    }
    return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def format_prompt(row: dict[str, Any]) -> str:
    paragraphs = []
    for paragraph in row["paragraphs"]:
        paragraphs.append(
            f"[Paragraph {paragraph['idx']}] {paragraph['title']}\n"
            f"{paragraph['paragraph_text']}"
        )
    return (
        "Read the passages and answer the question. Return only the short answer, "
        "without an explanation.\n\n"
        + "\n\n".join(paragraphs)
        + f"\n\nQuestion: {row['question']}\nAnswer:"
    )


def label_injection_audit(row: dict[str, Any], prompt: str) -> dict[str, Any]:
    forbidden_field_tokens = (
        '"answer":',
        '"answer_aliases":',
        '"answerable":',
        '"question_decomposition":',
        '"is_supporting":',
        '"paragraph_support_idx":',
    )
    if any(token in prompt for token in forbidden_field_tokens):
        raise RuntimeError(
            f"gold field serialization detected in prompt for {row['id']}"
        )
    answers = [row["answer"], *row["answer_aliases"]]
    occurrences = {
        answer: prompt.casefold().count(answer.casefold()) for answer in answers
    }
    return {
        "gold_field_serialization_absent": True,
        "natural_answer_occurrences": occurrences,
    }


def validate_row(row: dict[str, Any]) -> None:
    required = {
        "id",
        "paragraphs",
        "question",
        "question_decomposition",
        "answer",
        "answer_aliases",
        "answerable",
    }
    if set(row) < required:
        raise ValueError(
            f"official row {row.get('id')} lacks {sorted(required - set(row))}"
        )
    if not row["answerable"] or not str(row["id"]).startswith("2hop__"):
        raise ValueError(f"row {row['id']} is not an answerable two-hop item")
    paragraphs = row["paragraphs"]
    if len(paragraphs) != 20:
        raise ValueError(
            f"row {row['id']} has {len(paragraphs)} paragraphs, expected 20"
        )
    indices = [paragraph.get("idx") for paragraph in paragraphs]
    if indices != list(range(20)):
        raise ValueError(f"row {row['id']} has noncanonical paragraph indices")
    if sum(bool(paragraph.get("is_supporting")) for paragraph in paragraphs) < 2:
        raise ValueError(f"row {row['id']} has fewer than two supporting paragraphs")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"row {row['id']} has an empty question")
    if not isinstance(row["answer"], str) or not row["answer"].strip():
        raise ValueError(f"row {row['id']} has an empty answer")
    if not isinstance(row["answer_aliases"], list) or not all(
        isinstance(alias, str) and alias.strip() for alias in row["answer_aliases"]
    ):
        raise ValueError(f"row {row['id']} has invalid answer aliases")
    decomposed = row["question_decomposition"]
    if len(decomposed) != 2:
        raise ValueError(f"row {row['id']} has {len(decomposed)} decomposition steps")
    for item in decomposed:
        index = item.get("paragraph_support_idx")
        if index is None or index not in indices:
            raise ValueError(f"row {row['id']} has unresolved decomposition support")


def protected_rows() -> list[dict[str, Any]]:
    paths = (
        ROOT / "data" / "ruler-pilot-v1" / "manifest.json",
        ROOT / "data" / "ruler-ea-dev-v1" / "manifest.json",
        ROOT / "data" / "graded-v1" / "manifest.jsonl",
        ROOT / "data" / "value-head-v1" / "discovery.jsonl",
        ROOT / "data" / "value-head-v1" / "evaluation.jsonl",
        ROOT / "data" / "functional-v1" / "discovery.jsonl",
        ROOT / "data" / "functional-v1" / "evaluation.jsonl",
        ROOT / "data" / "value-level-v1" / "discovery.jsonl",
        ROOT / "data" / "value-level-v1" / "evaluation.jsonl",
        ROOT / "data" / "vt-pilot-v1" / "manifest.json",
    )
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            rows.extend(load_jsonl(path))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(value if isinstance(value, list) else value.get("prompts", []))
    return [row for row in rows if isinstance(row, dict)]


def protected_overlap(
    prompt: str, row: dict[str, Any], protected: list[dict[str, Any]]
) -> dict[str, int]:
    prompt_hash = sha256_text(prompt)
    context_hash = sha256_text(context_text(row))
    question_hash = sha256_text(normalized_text(row["question"]))
    protected_prompt_hashes = {
        sha256_text(str(item["prompt"]))
        for item in protected
        if isinstance(item.get("prompt"), str)
    }
    protected_context_hashes = {
        sha256_text(str(item.get("prompt")))
        for item in protected
        if isinstance(item.get("prompt"), str)
    }
    protected_question_hashes = {
        sha256_text(normalized_text(str(item["question"])))
        for item in protected
        if isinstance(item.get("question"), str)
    }
    return {
        "exact_prompt": int(prompt_hash in protected_prompt_hashes),
        "prompt_context": int(context_hash in protected_context_hashes),
        "normalized_question": int(question_hash in protected_question_hashes),
    }


def scorer_checks() -> dict[str, Any]:
    import sys

    sys.path.insert(0, str(SCORER.parent))
    from metrics.answer import AnswerMetric, compute_f1, normalize_answer

    metric = AnswerMetric()
    metric("NC", ["North Carolina", "NC"])
    exact, alias_f1 = metric.get_metric()
    checks = {
        "official_normalizer_articles": normalize_answer("The Royal Academy!")
        == "royal academy",
        "official_alias_max_f1": exact == 1.0 and alias_f1 == 1.0,
        "official_partial_f1": abs(compute_f1("green", "blue green") - (2.0 / 3.0))
        < 1e-12,
    }
    if not all(checks.values()):
        raise RuntimeError(f"official scorer checks failed: {checks}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_DEFAULT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    raw_source = output / "raw" / RAW_SOURCE.name
    official_source = output / "raw" / OFFICIAL_SOURCE.name
    if not raw_source.is_file() or not official_source.is_file():
        raise FileNotFoundError(
            "official raw and converted dev files must be acquired before preparation"
        )
    if not args.tokenizer_path.is_dir():
        raise FileNotFoundError(f"pinned tokenizer is missing: {args.tokenizer_path}")
    converter = output / CONVERTER.relative_to(OUTPUT)
    scorer = output / SCORER.relative_to(OUTPUT)
    if not converter.is_file() or not scorer.is_file():
        raise FileNotFoundError(
            "official converter and scorer provenance files are required"
        )

    raw_rows = load_jsonl(raw_source)
    official_rows = load_jsonl(official_source)
    if len(raw_rows) != 2417 or len(official_rows) != 2417:
        raise ValueError(
            f"unexpected official dev row count raw={len(raw_rows)} official={len(official_rows)}"
        )
    for index, (raw, official) in enumerate(zip(raw_rows, official_rows)):
        raw_id = str(raw.get("id", ""))
        official_id = str(official.get("id", ""))
        if not raw_id.startswith(("double__", "triple_", "quadruple_")):
            raise ValueError(f"raw row {index} has an unexpected source ID")
        if not official_id.startswith(("2hop__", "3hop", "4hop")):
            raise ValueError(f"official row {index} has an unexpected converted ID")
    candidates = [
        row
        for row in official_rows
        if row.get("answerable") is True and str(row.get("id", "")).startswith("2hop__")
    ]
    selected = sorted(candidates, key=lambda row: str(row["id"]))[:COUNT]
    if len(selected) != COUNT:
        raise ValueError(f"only {len(selected)} answerable two-hop rows available")
    raw_by_converted_id = {
        str(row["id"]).replace("double__", "2hop__", 1): row for row in raw_rows
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, local_files_only=True, use_fast=True
    )
    protected = protected_rows()
    output_rows = []
    selected_raw_rows = []
    group_ids = set()
    overlap_counts = {"exact_prompt": 0, "prompt_context": 0, "normalized_question": 0}
    max_answer_tokens = 0
    answer_token_lengths = []
    for index, row in enumerate(selected):
        validate_row(row)
        prompt = format_prompt(row)
        if format_prompt(row) != prompt:
            raise AssertionError("prompt formatter is not deterministic")
        label_audit = label_injection_audit(row, prompt)
        prompt_chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_tokens = len(prompt_ids)
        answers = [row["answer"], *row["answer_aliases"]]
        lengths = [
            len(tokenizer(answer, add_special_tokens=False)["input_ids"])
            for answer in answers
        ]
        max_tokens = max(lengths)
        max_answer_tokens = max(max_answer_tokens, max_tokens)
        answer_token_lengths.append(
            {
                "id": row["id"],
                "answer": lengths[0],
                "aliases": lengths[1:],
                "max": max_tokens,
            }
        )
        if HORIZON < max_tokens + 16:
            raise RuntimeError(
                f"horizon {HORIZON} does not cover gold answer budget for {row['id']}"
            )
        if prompt_tokens + HORIZON > CONTEXT_LIMIT:
            raise RuntimeError(
                f"prompt plus horizon exceeds context limit for {row['id']}"
            )
        group = group_key(row)
        if group in group_ids:
            raise RuntimeError(f"duplicate MuSiQue group in selected rows: {row['id']}")
        group_ids.add(group)
        overlaps = protected_overlap(prompt, row, protected)
        for key, value in overlaps.items():
            overlap_counts[key] += value
        if any(overlaps.values()):
            raise RuntimeError(f"protected data overlap for {row['id']}: {overlaps}")
        raw = raw_by_converted_id.get(str(row["id"]))
        if raw is None:
            raise RuntimeError(f"missing raw counterpart for {row['id']}")
        selected_raw_rows.append(raw)
        output_rows.append(
            {
                "id": row["id"],
                "group_id": f"{TASK}-{group}",
                "prompt": prompt,
                "answers": answers,
                "task": TASK,
                "max_new_tokens": HORIZON,
                "seed": SEED,
                "provenance": {
                    "official_index": official_rows.index(row),
                    "raw_id": raw["id"],
                    "source_url": SOURCE_URL,
                    "source_direct_url": SOURCE_DIRECT_URL,
                    "prompt_template": TEMPLATE_VERSION,
                    "prompt_sha256": sha256_text(prompt),
                    "rendered_chat_prompt_sha256": sha256_text(prompt_chat),
                    "qwen_chat_prompt_tokens": prompt_tokens,
                    "answer_token_lengths": lengths,
                    "gold_labels_not_in_prompt_template": label_audit,
                },
            }
        )
    scorer_check_results = scorer_checks()
    if overlap_counts != {
        "exact_prompt": 0,
        "prompt_context": 0,
        "normalized_question": 0,
    }:
        raise RuntimeError(f"protected overlap counts are nonzero: {overlap_counts}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "selected_raw.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in selected_raw_rows
        ),
        encoding="utf-8",
    )
    (output / "selected_official.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows),
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps(output_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metadata = {
        "schema_version": "musique_pilot.v1",
        "task": TASK,
        "count": COUNT,
        "selection": "official MuSiQue-Ans dev, answerable two-hop rows, official ID ascending, first 16",
        "source_repository": SOURCE_REPOSITORY,
        "source_url": SOURCE_URL,
        "source_direct_url": SOURCE_DIRECT_URL,
        "source_license_url": SOURCE_LICENSE_URL,
        "source_info_url": SOURCE_INFO_URL,
        "raw_source_sha256": sha256_file(raw_source),
        "official_source_sha256": sha256_file(official_source),
        "raw_source_rows": len(raw_rows),
        "official_source_rows": len(official_rows),
        "selected_ids": [row["id"] for row in selected],
        "selected_raw_ids": [row["id"] for row in selected_raw_rows],
        "selected_official_sha256": sha256_file(output / "selected_official.jsonl"),
        "selected_raw_sha256": sha256_file(output / "selected_raw.jsonl"),
        "manifest_sha256": sha256_file(output / "manifest.jsonl"),
        "manifest_json_sha256": sha256_file(output / "manifest.json"),
        "converter_url": CONVERTER_URL,
        "converter_sha256": sha256_file(converter),
        "scorer_url": SCORER_URL,
        "scorer_sha256": sha256_file(scorer),
        "tokenizer_path": str(args.tokenizer_path),
        "tokenizer_files_sha256": {
            str(path.relative_to(args.tokenizer_path)): sha256_file(path)
            for path in sorted(args.tokenizer_path.rglob("*"))
            if path.is_file()
        },
        "prompt_template": TEMPLATE_VERSION,
        "horizon": HORIZON,
        "context_limit": CONTEXT_LIMIT,
        "seed": SEED,
        "decoding": "greedy with native EOS stopping",
        "answer_token_lengths": answer_token_lengths,
        "max_gold_answer_tokens": max_answer_tokens,
        "horizon_margin": HORIZON - max_answer_tokens,
        "overlap_counts": overlap_counts,
        "group_count": len(group_ids),
        "scorer_checks": scorer_check_results,
        "labels_preserved_in_selected_official": True,
        "labels_excluded_from_prompt": True,
        "complete_context_preserved": True,
        "no_test_rows_inspected": True,
    }
    METADATA_PATH = output / "metadata.json"
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
