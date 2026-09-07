#!/usr/bin/env python3
"""Measure a frozen TF-IDF question locator and B0 span eviction feature.

This script intentionally accepts only the restricted lexical-span input view.
It does not open outcomes, answers, manifests, model files, or annotations.
"""

import argparse
import hashlib
import json
import platform
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer


ALLOWED_ROW_KEYS = {"id", "prompt", "prompt_length", "native_masks"}
SENTENCE_PATTERN = re.compile(r"[^.!?\n]+[.!?]*")
NONEMPTY_LINE_PATTERN = re.compile(r"[^\n]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> tuple[str, list[dict[str, str]]]:
    files: list[dict[str, str]] = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        files.append({"path": relative, "sha256": sha256_file(file_path)})
    digest = hashlib.sha256()
    for item in files:
        digest.update(item["path"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(item["sha256"].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), files


def rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value
    return value * 1024


def as_one_dimensional_ids(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise ValueError("tokenizer returned an unexpected ID container")
    while len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not all(isinstance(item, (int, np.integer)) for item in value):
        raise ValueError("tokenizer returned non-integer token IDs")
    return [int(item) for item in value]


def final_question(prompt: str) -> tuple[str, int, int, str]:
    matches = list(NONEMPTY_LINE_PATTERN.finditer(prompt))
    if not matches:
        raise ValueError("prompt has no newline-delimited question")
    final = matches[-1]
    leading = len(final.group(0)) - len(final.group(0).lstrip())
    trailing = len(final.group(0).rstrip())
    start = final.start() + leading
    end = final.start() + trailing
    question = prompt[start:end]
    if not question:
        raise ValueError("final nonempty line has no question text")
    return question, start, end, prompt[: final.start()]


def split_context(context: str) -> list[dict[str, object]]:
    sentences: list[dict[str, object]] = []
    for index, match in enumerate(SENTENCE_PATTERN.finditer(context)):
        raw_start, raw_end = match.span()
        text = match.group(0)
        left_trimmed = text.lstrip()
        right_trimmed = left_trimmed.rstrip()
        start = raw_start + len(text) - len(left_trimmed)
        end = start + len(right_trimmed)
        if start == end:
            continue
        sentences.append(
            {
                "sentence_index": index,
                "raw_start": start,
                "raw_end": end,
                "sentence": context[start:end],
            }
        )
    if not sentences:
        raise ValueError("context has no nonempty sentences")
    return sentences


def span_positions(
    tokenizer: object,
    rendered: str,
    prompt: str,
    prompt_length: int,
    span_start: int,
    span_end: int,
) -> tuple[list[int], list[int], bool]:
    prompt_offset = rendered.find(prompt)
    if prompt_offset < 0 or rendered.count(prompt) != 1:
        raise ValueError("raw prompt is not unique in rendered chat template")
    rendered_start = prompt_offset + span_start
    rendered_end = prompt_offset + span_end
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    explicit_ids = as_one_dimensional_ids(encoded["input_ids"])
    chat_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    chat_ids = as_one_dimensional_ids(chat_ids)
    parity = explicit_ids == chat_ids and len(explicit_ids) == prompt_length
    if not parity:
        raise ValueError(
            "native chat-template tokenization differs from rendered offsets "
            f"(explicit={len(explicit_ids)}, chat={len(chat_ids)}, "
            f"declared={prompt_length})"
        )
    offsets = encoded["offset_mapping"]
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    if isinstance(offsets, tuple):
        offsets = list(offsets)
    while len(offsets) == 1 and isinstance(offsets[0], list):
        offsets = offsets[0]
    positions = [
        index
        for index, pair in enumerate(offsets)
        if int(pair[1]) > rendered_start and int(pair[0]) < rendered_end
    ]
    if not positions:
        raise ValueError("located span has no tokenizer offsets")
    prefix_limit = prompt_length - 1
    if any(index < 0 or index >= prefix_limit for index in positions):
        raise ValueError("located span reaches the pending B0 token")
    token_ids = [explicit_ids[index] for index in positions]
    return positions, token_ids, parity


def locate(
    tokenizer: object,
    prompt: str,
    prompt_length: int,
    question: str,
    context: str,
) -> dict[str, object]:
    sentences = split_context(context)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 1),
        norm="l2",
    )
    sentence_matrix = vectorizer.fit_transform(
        [str(item["sentence"]) for item in sentences]
    )
    question_vector = vectorizer.transform([question])
    similarities = (sentence_matrix @ question_vector.T).toarray().ravel()
    if len(similarities) != len(sentences):
        raise ValueError("TF-IDF similarity count does not match sentence count")
    top_similarity = float(np.max(similarities))
    tied_indices = [
        index
        for index, similarity in enumerate(similarities)
        if float(similarity) == top_similarity
    ]
    if not tied_indices:
        raise ValueError("TF-IDF produced no top sentence")
    tied_matches: list[dict[str, object]] = []
    for index in tied_indices:
        item = sentences[index]
        positions, token_ids, parity = span_positions(
            tokenizer,
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            ),
            prompt,
            prompt_length,
            int(item["raw_start"]),
            int(item["raw_end"]),
        )
        tied_matches.append(
            {
                "sentence_index": int(item["sentence_index"]),
                "raw_start": int(item["raw_start"]),
                "raw_end": int(item["raw_end"]),
                "sentence": str(item["sentence"]),
                "similarity": float(similarities[index]),
                "token_positions": positions,
                "token_ids": token_ids,
                "chat_tokenization_ids_equal": parity,
            }
        )
    chosen = tied_matches[0]
    return {
        "raw_start": int(chosen["raw_start"]),
        "raw_end": int(chosen["raw_end"]),
        "sentence": str(chosen["sentence"]),
        "sentence_index": int(chosen["sentence_index"]),
        "similarity": float(chosen["similarity"]),
        "token_positions": list(chosen["token_positions"]),
        "token_ids": list(chosen["token_ids"]),
        "chat_tokenization_ids_equal": bool(
            chosen["chat_tokenization_ids_equal"]
        ),
        "tied_top_matches": tied_matches,
        "sentence_count": len(sentences),
        "vocabulary_size": len(vectorizer.vocabulary_),
    }


def span_feature(
    located: dict[str, object], native_masks: object, prompt_length: int
) -> dict[str, object]:
    positions = [int(value) for value in located["token_positions"]]
    if not positions:
        raise ValueError("cannot measure an empty located span")
    if len(set(positions)) != len(positions):
        raise ValueError("located span contains duplicate token positions")
    if not isinstance(native_masks, list) or not native_masks:
        raise ValueError("native_masks must be a nonempty layer list")
    if any(position >= prompt_length - 1 for position in positions):
        raise ValueError("located span is outside the B0 prefix")
    per_head: list[list[float]] = []
    for layer_index, layer in enumerate(native_masks):
        if not isinstance(layer, list) or not layer:
            raise ValueError(f"native mask layer {layer_index} is empty")
        layer_values: list[float] = []
        for head_index, kept in enumerate(layer):
            if not isinstance(kept, list):
                raise ValueError(
                    f"native mask layer {layer_index} head {head_index} is not a list"
                )
            kept_positions = {int(value) for value in kept}
            missing = sum(position not in kept_positions for position in positions)
            layer_values.append(float(missing / len(positions)))
        per_head.append(layer_values)
    flattened = [value for layer in per_head for value in layer]
    z = float(np.mean(flattened))
    normalized_position = float(
        np.mean(positions) / max(1, prompt_length - 2)
    )
    return {
        "token_positions": positions,
        "z": z,
        "per_head_fractions": per_head,
        "normalized_position": normalized_position,
    }


def restricted_rows(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("restricted input view must be a JSON list")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f"input row {index} is not an object")
        unexpected = set(row) - ALLOWED_ROW_KEYS
        missing = ALLOWED_ROW_KEYS - set(row)
        if unexpected or missing:
            raise ValueError(
                f"input row {index} schema mismatch, unexpected={sorted(unexpected)}, "
                f"missing={sorted(missing)}"
            )
        rows.append(row)
    return rows


def measure(args: argparse.Namespace) -> dict[str, object]:
    input_path = Path(args.input).resolve()
    tokenizer_path = Path(args.tokenizer).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_wall_start = time.perf_counter_ns()
    run_cpu_start = time.process_time_ns()
    memory_before = rss_bytes()
    input_sha = sha256_file(input_path)
    tokenizer_sha, tokenizer_files = sha256_directory(tokenizer_path)
    script_path = Path(__file__).resolve()
    script_sha = sha256_file(script_path)
    rows = restricted_rows(input_path)
    ordered_rows = sorted(rows, key=lambda row: str(row["id"]))
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer does not provide a chat template")
    output_rows: list[dict[str, object]] = []
    for index, row in enumerate(ordered_rows):
        row_wall_start = time.perf_counter_ns()
        row_cpu_start = time.process_time_ns()
        result: dict[str, object] = {
            "id": str(row["id"]),
            "status": "failed",
        }
        try:
            prompt = row["prompt"]
            prompt_length = int(row["prompt_length"])
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            question, question_start, question_end, context = final_question(prompt)
            next_row = ordered_rows[(index + 1) % len(ordered_rows)]
            next_question, next_start, next_end, _ = final_question(
                str(next_row["prompt"])
            )
            matched_start = time.perf_counter_ns()
            matched = locate(
                tokenizer, prompt, prompt_length, question, context
            )
            matched_seconds = (time.perf_counter_ns() - matched_start) / 1e9
            shifted_start = time.perf_counter_ns()
            shifted = locate(
                tokenizer, prompt, prompt_length, next_question, context
            )
            shifted_seconds = (time.perf_counter_ns() - shifted_start) / 1e9
            feature_start = time.perf_counter_ns()
            matched_feature = span_feature(
                matched, row["native_masks"], prompt_length
            )
            shifted_feature = span_feature(
                shifted, row["native_masks"], prompt_length
            )
            feature_seconds = (time.perf_counter_ns() - feature_start) / 1e9
            matched.update(matched_feature)
            shifted.update(shifted_feature)
            result.update(
                {
                    "prompt_length": prompt_length,
                    "question": {
                        "raw_start": question_start,
                        "raw_end": question_end,
                        "text": question,
                    },
                    "shifted_question": {
                        "source_id": str(next_row["id"]),
                        "raw_start": next_start,
                        "raw_end": next_end,
                        "text": next_question,
                    },
                    "matched": matched,
                    "shifted": shifted,
                    "timings": {
                        "matched_seconds": matched_seconds,
                        "shifted_seconds": shifted_seconds,
                        "feature_seconds": feature_seconds,
                    },
                    "status": "completed",
                }
            )
        except Exception as exc:  # retain every row failure and continue
            result["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        result["timings"] = {
            **dict(result.get("timings", {})),
            "row_wall_seconds": (time.perf_counter_ns() - row_wall_start) / 1e9,
            "row_cpu_seconds": (time.process_time_ns() - row_cpu_start) / 1e9,
        }
        output_rows.append(result)
        output_path.write_text(
            json.dumps(
                {
                    "schema": "lexical-span-v1",
                    "status": "in_progress",
                    "rows": output_rows,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    payload = {
        "schema": "lexical-span-v1",
        "status": "completed",
        "method": {
            "sentence_regex": SENTENCE_PATTERN.pattern,
            "tfidf": {
                "lowercase": True,
                "ngram_range": [1, 1],
                "stop_words": "english",
                "norm": "l2",
            },
            "z": "mean selected-position missing fraction across every native layer/KV head",
            "question_control": "next question in lexicographic ID order, cyclic",
        },
        "provenance": {
            "input_path": str(input_path),
            "input_sha256": input_sha,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_sha256": tokenizer_sha,
            "tokenizer_files": tokenizer_files,
            "tokenizer_class": type(tokenizer).__name__,
            "tokenizer_name_or_path": str(tokenizer.name_or_path),
            "script_path": str(script_path),
            "script_sha256": script_sha,
            "python": sys.executable,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "run": {
            "row_count": len(output_rows),
            "completed_count": sum(
                row["status"] == "completed" for row in output_rows
            ),
            "failed_count": sum(row["status"] == "failed" for row in output_rows),
            "wall_seconds": (time.perf_counter_ns() - run_wall_start) / 1e9,
            "cpu_seconds": (time.process_time_ns() - run_cpu_start) / 1e9,
            "memory_rss_before_bytes": memory_before,
            "memory_rss_peak_bytes": rss_bytes(),
        },
        "rows": output_rows,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/lexical-span-input/view.json",
    )
    parser.add_argument(
        "--tokenizer",
        default="/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3/data/retrieval-tokenizer-a09a354",
    )
    parser.add_argument(
        "--output",
        default="results/lexical-span-v1/measurement.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = measure(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["run"]["row_count"],
                "completed": result["run"]["completed_count"],
                "failed": result["run"]["failed_count"],
                "script_sha256": result["provenance"]["script_sha256"],
                "input_sha256": result["provenance"]["input_sha256"],
                "output": str(Path(parse_args().output).resolve()),
            },
            sort_keys=True,
        )
    )
