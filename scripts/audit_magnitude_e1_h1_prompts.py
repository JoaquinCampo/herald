"""Audit H1 development prompt provenance and cross-fold template overlap."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasets import load_dataset  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy.sparse import spmatrix  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)

DEVELOPMENT_PROMPTS = 154
QUARANTINED_PROMPTS = 46
FOLD_COUNTS = (31, 31, 31, 31, 30)
SCHEMA_VERSION = "herald.magnitude_e1_h1_prompt_audit.v1"
WORD_PATTERN = re.compile(r"(?u)\b\w\w+\b")
QUOTED_PATTERN = re.compile(r"(['\"`]).*?\1")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
SPACE_PATTERN = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hash_prompt_ids(values: list[str] | tuple[str, ...]) -> str:
    prompts = sorted(str(value) for value in values)
    return hashlib.sha256(("\n".join(prompts) + "\n").encode()).hexdigest()


def hash_prompt_texts(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.sort_values("prompt_id").itertuples(index=False):
        digest.update(str(row.prompt_id).encode())
        digest.update(b"\0")
        digest.update(str(row.prompt_text).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_prompt(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    return SPACE_PATTERN.sub(" ", text).strip()


def prompt_skeleton(value: str) -> str:
    text = normalize_prompt(value)
    text = QUOTED_PATTERN.sub("<quoted>", text)
    return NUMBER_PATTERN.sub("<number>", text)


def duplicate_groups(
    frame: pd.DataFrame, transform: Callable[[str], str]
) -> list[list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for row in frame.itertuples(index=False):
        groups[transform(str(row.prompt_text))].append(str(row.prompt_id))
    return sorted(
        (sorted(group) for group in groups.values() if len(group) > 1),
        key=lambda group: (-len(group), group),
    )


def load_development_prompts(
    development: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dataset = load_dataset("google/IFEval", split="train")
    keys = [int(value.as_py()) for value in dataset.data.column("key")]
    if len(keys) != len(set(keys)):
        raise ValueError("IFEval source keys are not unique")
    index_by_key = {key: index for index, key in enumerate(keys)}
    development_keys = [
        int(prompt_id.removeprefix("ifeval-")) for prompt_id in development
    ]
    missing = sorted(set(development_keys) - set(index_by_key))
    if missing:
        raise ValueError(f"development keys absent from IFEval: {missing}")
    indices = [index_by_key[key] for key in development_keys]
    subset = dataset.select(indices).select_columns(["key", "prompt"])
    records = [
        {
            "prompt_id": f"ifeval-{int(example['key'])}",
            "prompt_text": str(example["prompt"]),
        }
        for example in subset
    ]
    frame = pd.DataFrame(records)
    if len(frame) != DEVELOPMENT_PROMPTS or set(frame["prompt_id"]) != set(
        development
    ):
        raise ValueError(
            "IFEval prompt subset differs from development roster"
        )
    cache_hashes = []
    for item in dataset.cache_files:
        filename = Path(str(item["filename"]))
        cache_hashes.append(
            {
                "name": filename.name,
                "sha256": sha256_file(filename),
            }
        )
    source = {
        "dataset": "google/IFEval",
        "split": "train",
        "source_rows": len(dataset),
        "source_columns": list(dataset.column_names),
        "source_fingerprint": str(getattr(dataset, "_fingerprint", "")),
        "cache_files": cache_hashes,
        "columns_accessed_before_subset": ["key"],
        "columns_materialized_after_development_subset": ["key", "prompt"],
        "gold_columns_materialized": [],
        "protected_prompt_text_rows_materialized": 0,
    }
    return frame, source


def load_fold_roster(
    path: Path, development: tuple[str, ...]
) -> pd.DataFrame:
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=["prompt_id", "fold"],
    )
    frame = table.to_pandas().drop_duplicates()
    if (
        len(frame) != DEVELOPMENT_PROMPTS
        or frame["prompt_id"].duplicated().any()
        or set(frame["prompt_id"]) != set(development)
    ):
        raise ValueError("OOF fold roster differs from development split")
    return frame


def sparse_row_maxima(
    validation: spmatrix, training: spmatrix
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    similarity = (validation @ training.T).toarray()
    return (
        np.max(similarity, axis=1).astype(np.float64),
        np.argmax(similarity, axis=1).astype(np.int64),
    )


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(quantile): float(np.quantile(array, quantile))
        for quantile in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def overlap_audit(frame: pd.DataFrame) -> dict[str, Any]:
    configurations = {
        "character_3_5": {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 2,
            "lowercase": False,
            "sublinear_tf": True,
        },
        "word_1_2": {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 1,
            "lowercase": True,
            "sublinear_tf": True,
        },
    }
    similarities: dict[str, list[float]] = {
        name: [] for name in configurations
    }
    nearest: dict[str, list[dict[str, Any]]] = {
        name: [] for name in configurations
    }
    token_coverages: list[float] = []
    fold_summary = {}
    for fold in sorted(int(value) for value in frame["fold"].unique()):
        training = frame[frame["fold"] != fold].sort_values("prompt_id")
        validation = frame[frame["fold"] == fold].sort_values("prompt_id")
        training_text = [str(value) for value in training["prompt_text"]]
        validation_text = [str(value) for value in validation["prompt_text"]]
        fold_summary[str(fold)] = {
            "training_prompts": len(training),
            "validation_prompts": len(validation),
        }
        training_tokens = {
            token
            for text in training_text
            for token in WORD_PATTERN.findall(text.casefold())
        }
        for text in validation_text:
            tokens = WORD_PATTERN.findall(text.casefold())
            coverage = (
                sum(token in training_tokens for token in tokens)
                / len(tokens)
                if tokens
                else 1.0
            )
            token_coverages.append(float(coverage))
        for name, parameters in configurations.items():
            vectorizer = TfidfVectorizer(norm="l2", **parameters)
            training_matrix = vectorizer.fit_transform(training_text)
            validation_matrix = vectorizer.transform(validation_text)
            maxima, indices = sparse_row_maxima(
                validation_matrix, training_matrix
            )
            similarities[name].extend(float(value) for value in maxima)
            for row_index, (value, train_index) in enumerate(
                zip(maxima, indices, strict=True)
            ):
                nearest[name].append(
                    {
                        "fold": fold,
                        "validation_prompt_id": str(
                            validation.iloc[row_index]["prompt_id"]
                        ),
                        "training_prompt_id": str(
                            training.iloc[int(train_index)]["prompt_id"]
                        ),
                        "similarity": float(value),
                    }
                )
    return {
        "folds": fold_summary,
        "nearest_train_similarity": {
            name: {
                "quantiles": quantiles(values),
                "at_or_above_0.8": sum(value >= 0.8 for value in values),
                "at_or_above_0.9": sum(value >= 0.9 for value in values),
                "at_or_above_0.95": sum(value >= 0.95 for value in values),
                "top_pairs": sorted(
                    nearest[name],
                    key=lambda row: (
                        -float(row["similarity"]),
                        str(row["validation_prompt_id"]),
                    ),
                )[:10],
            }
            for name, values in similarities.items()
        },
        "validation_word_token_train_coverage": {
            "quantiles": quantiles(token_coverages),
            "below_0.8": sum(value < 0.8 for value in token_coverages),
        },
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H1 prompt audit: {args.output}"
        )
    evidence = load_json(args.evidence)
    protocol = load_json(args.protocol_lock)
    development = tuple(
        str(value) for value in evidence["split"]["train_prompt_ids"]
    )
    quarantine = tuple(
        str(value) for value in evidence["split"]["test_prompt_ids"]
    )
    if (
        len(development) != DEVELOPMENT_PROMPTS
        or len(quarantine) != QUARANTINED_PROMPTS
        or set(development) & set(quarantine)
    ):
        raise ValueError("frozen development/quarantine roster is invalid")
    if sha256_file(args.evidence) != protocol["source_evidence_sha256"]:
        raise ValueError("evidence hash differs from protocol lock")
    if (
        hash_prompt_ids(development)
        != protocol["prompt_partitions"]["development_prompt_ids_sha256"]
        or hash_prompt_ids(quarantine)
        != protocol["prompt_partitions"]["quarantined_prompt_ids_sha256"]
    ):
        raise ValueError("prompt roster hashes differ from protocol lock")

    prompts, source = load_development_prompts(development)
    folds = load_fold_roster(args.source_oof, development)
    frame = prompts.merge(folds, on="prompt_id", validate="one_to_one")
    fold_counts = tuple(
        int(value) for value in frame["fold"].value_counts().sort_index()
    )
    fold_hashes = [
        hash_prompt_ids(
            [
                str(value)
                for value in frame.loc[frame["fold"] == fold, "prompt_id"]
            ]
        )
        for fold in range(len(FOLD_COUNTS))
    ]
    if (
        sorted(fold_counts) != sorted(FOLD_COUNTS)
        or fold_hashes
        != protocol["prompt_partitions"]["fold_prompt_ids_sha256"]
    ):
        raise ValueError("OOF fold hashes differ from protocol lock")

    raw_duplicates = duplicate_groups(frame, lambda value: value)
    normalized_duplicates = duplicate_groups(frame, normalize_prompt)
    skeleton_duplicates = duplicate_groups(frame, prompt_skeleton)
    prompt_lengths = frame["prompt_text"].str.len().to_numpy(dtype=np.int64)
    prompt_text_sha256 = hash_prompt_texts(frame)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "label_and_gold_blind_development_prompt_audit",
        "provenance": {
            "evidence_sha256": sha256_file(args.evidence),
            "protocol_lock_sha256": sha256_file(args.protocol_lock),
            "source_oof_sha256": sha256_file(args.source_oof),
            "development_prompt_ids_sha256": hash_prompt_ids(development),
            "quarantined_prompt_ids_sha256": hash_prompt_ids(quarantine),
            "development_prompt_text_sha256": prompt_text_sha256,
            "source": source,
        },
        "isolation": {
            "development_prompts": len(development),
            "quarantined_prompts": len(quarantine),
            "development_quarantine_overlap": 0,
            "confirmation_status": "sealed_not_run",
            "outcome_or_label_columns_loaded": [],
            "gold_instruction_columns_loaded": [],
            "protected_prompt_text_rows_materialized": 0,
            "fold_counts": list(fold_counts),
            "fold_prompt_ids_sha256": fold_hashes,
        },
        "prompt_geometry": {
            "raw_exact_duplicate_groups": raw_duplicates,
            "normalized_exact_duplicate_groups": normalized_duplicates,
            "skeleton_duplicate_groups": skeleton_duplicates,
            "characters": {
                "minimum": int(np.min(prompt_lengths)),
                "median": float(np.median(prompt_lengths)),
                "maximum": int(np.max(prompt_lengths)),
            },
        },
        "cross_fold_overlap": overlap_audit(frame),
        "interpretation": {
            "allowed": (
                "Label-blind assessment of whether frozen prompt folds "
                "contain lexical or template duplication that could inflate "
                "an H1 text probe."
            ),
            "forbidden": [
                "candidate selection",
                "outcome inference",
                "future-prompt generalization",
                "inspection of quarantined, confirmation, or reserve "
                "prompt text",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "h1_prompt_audit_complete",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "prompt_text_sha256": prompt_text_sha256,
                "raw_duplicate_groups": len(raw_duplicates),
                "normalized_duplicate_groups": len(normalized_duplicates),
                "skeleton_duplicate_groups": len(skeleton_duplicates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
