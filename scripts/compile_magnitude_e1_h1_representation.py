"""Freeze the label-blind fold-local representation for E1 H1."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset  # type: ignore[import-untyped]
from magnitude_e1_h1_features import (
    EDF_TOLERANCE,
    SLOPE_EDF,
    Representation,
    build_representation,
)
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]

PROMPTS = 154
QUARANTINED_PROMPTS = 46
FOLD_COUNTS = (31, 31, 31, 31, 30)
ARCHIVE_SCHEMA = "herald.magnitude_e1_h1_representation.v1"
MANIFEST_SCHEMA = "herald.magnitude_e1_h1_representation_manifest.v1"
LOCK_SCHEMA = "herald.magnitude_e1_h1_protocol_lock.v1"
STRATEGY_SCHEMA = "herald.magnitude_e1_h1_strategy.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--feasibility-audit", type=Path, required=True)
    parser.add_argument("--understanding", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--feature-script", type=Path, required=True)
    parser.add_argument("--score-script", type=Path, required=True)
    parser.add_argument("--prompt-table", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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


def hash_array(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def hash_strings(values: tuple[str, ...]) -> str:
    return hashlib.sha256(("\0".join(values) + "\0").encode()).hexdigest()


def hash_sparse(matrix: sparse.csr_matrix) -> str:
    canonical = matrix.copy()
    canonical.sum_duplicates()
    canonical.sort_indices()
    digest = hashlib.sha256()
    digest.update(str(canonical.shape).encode())
    for array in (canonical.data, canonical.indices, canonical.indptr):
        digest.update(array.dtype.str.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def representation_hashes(value: Representation) -> dict[str, str]:
    return {
        "text_training": hash_sparse(value.text_training),
        "text_validation": hash_sparse(value.text_validation),
        "surface_training": hash_array(value.surface_training),
        "surface_validation": hash_array(value.surface_validation),
        "nearest_training_similarity": hash_array(
            value.nearest_training_similarity
        ),
        "word_features": hash_strings(value.word_features),
        "char_features": hash_strings(value.char_features),
        "word_idf": hash_array(value.word_idf),
        "char_idf": hash_array(value.char_idf),
        "surface_mean": hash_array(value.surface_mean),
        "surface_scale": hash_array(value.surface_scale),
    }


def validate_lock(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = load_json(args.protocol_lock)
    strategy = load_json(args.strategy)
    feasibility = load_json(args.feasibility_audit)
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "locked_before_label_blind_representation"
    ):
        raise ValueError("H1 protocol lock schema/status is invalid")
    if (
        strategy.get("schema_version") != STRATEGY_SCHEMA
        or strategy.get("status")
        != "selected_before_h1_representation_freeze"
        or strategy.get("selected_hypothesis")
        != "S3_equal_word_character_union"
    ):
        raise ValueError("H1 strategy is not the frozen S3 selection")
    expected_anchors = {
        "source_oof_sha256": sha256_file(args.source_oof),
        "evidence_sha256": sha256_file(args.evidence),
        "prompt_audit_sha256": sha256_file(args.prompt_audit),
        "target_audit_sha256": sha256_file(args.target_audit),
        "feasibility_audit_sha256": sha256_file(args.feasibility_audit),
        "understanding_sha256": sha256_file(args.understanding),
        "strategy_sha256": sha256_file(args.strategy),
    }
    if lock.get("anchors") != expected_anchors:
        raise ValueError("H1 protocol anchor hashes differ")
    expected_stage_a = {
        "allowed_oof_columns": ["prompt_id", "fold"],
        "allowed_ifeval_columns": ["key", "prompt"],
        "slope_edf": SLOPE_EDF,
        "edf_tolerance": EDF_TOLERANCE,
        "low_similarity_count": 77,
        "outcome_access": False,
        "gold_instruction_access": False,
        "protected_prompt_text_rows_materialized": 0,
    }
    if lock.get("stage_a") != expected_stage_a:
        raise ValueError("H1 Stage-A configuration differs from lock")
    expected_implementation = {
        "scripts/magnitude_e1_h1_features.py": sha256_file(
            args.feature_script
        ),
        "scripts/compile_magnitude_e1_h1_representation.py": sha256_file(
            Path(__file__)
        ),
        "scripts/score_magnitude_e1_h1.py": sha256_file(args.score_script),
    }
    if lock.get("implementation_sha256") != expected_implementation:
        raise ValueError("H1 implementation differs from protocol lock")
    if lock.get(
        "confirmation_status"
    ) != "sealed_not_run" or not feasibility.get("pass"):
        raise ValueError("H1 feasibility or protected-data status is invalid")
    return lock, feasibility


def load_development_prompts(
    development: tuple[str, ...],
) -> pd.DataFrame:
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
    subset = dataset.select(
        [index_by_key[key] for key in development_keys]
    ).select_columns(["key", "prompt"])
    frame = pd.DataFrame(
        [
            {
                "prompt_id": f"ifeval-{int(example['key'])}",
                "prompt_text": str(example["prompt"]),
            }
            for example in subset
        ]
    )
    if len(frame) != PROMPTS or set(frame["prompt_id"]) != set(development):
        raise ValueError("development prompt extraction is incomplete")
    return frame


def load_fold_roster(
    path: Path, development: tuple[str, ...]
) -> pd.DataFrame:
    columns = ["prompt_id", "fold"]
    table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    if table.column_names != columns:
        raise ValueError("Stage A loaded unexpected OOF columns")
    roster = table.to_pandas().drop_duplicates()
    if (
        len(roster) != PROMPTS
        or roster["prompt_id"].duplicated().any()
        or set(roster["prompt_id"]) != set(development)
    ):
        raise ValueError("OOF fold roster differs from development prompts")
    return roster


def add_csr(
    payload: dict[str, NDArray[Any]], prefix: str, matrix: sparse.csr_matrix
) -> None:
    canonical = matrix.copy()
    canonical.sum_duplicates()
    canonical.sort_indices()
    payload[f"{prefix}_data"] = canonical.data
    payload[f"{prefix}_indices"] = canonical.indices
    payload[f"{prefix}_indptr"] = canonical.indptr
    payload[f"{prefix}_shape"] = np.asarray(canonical.shape, dtype=np.int64)


def add_fold_payload(
    payload: dict[str, NDArray[Any]],
    fold: int,
    training_ids: list[str],
    validation_ids: list[str],
    representation: Representation,
) -> None:
    prefix = f"fold_{fold}"
    payload[f"{prefix}_training_prompt_ids"] = np.asarray(
        training_ids, dtype=np.str_
    )
    payload[f"{prefix}_validation_prompt_ids"] = np.asarray(
        validation_ids, dtype=np.str_
    )
    add_csr(payload, f"{prefix}_text_training", representation.text_training)
    add_csr(
        payload, f"{prefix}_text_validation", representation.text_validation
    )
    payload[f"{prefix}_surface_training"] = representation.surface_training
    payload[f"{prefix}_surface_validation"] = (
        representation.surface_validation
    )
    payload[f"{prefix}_nearest_training_similarity"] = (
        representation.nearest_training_similarity
    )
    payload[f"{prefix}_text_alpha"] = np.asarray(
        [representation.text_alpha], dtype=np.float64
    )
    payload[f"{prefix}_surface_alpha"] = np.asarray(
        [representation.surface_alpha], dtype=np.float64
    )
    payload[f"{prefix}_text_rank"] = np.asarray(
        [representation.text_rank], dtype=np.int64
    )
    payload[f"{prefix}_surface_rank"] = np.asarray(
        [representation.surface_rank], dtype=np.int64
    )
    payload[f"{prefix}_text_edf"] = np.asarray(
        [representation.text_edf], dtype=np.float64
    )
    payload[f"{prefix}_surface_edf"] = np.asarray(
        [representation.surface_edf], dtype=np.float64
    )
    payload[f"{prefix}_word_features"] = np.asarray(
        representation.word_features, dtype=np.str_
    )
    payload[f"{prefix}_char_features"] = np.asarray(
        representation.char_features, dtype=np.str_
    )
    payload[f"{prefix}_word_idf"] = representation.word_idf
    payload[f"{prefix}_char_idf"] = representation.char_idf
    payload[f"{prefix}_surface_mean"] = representation.surface_mean
    payload[f"{prefix}_surface_scale"] = representation.surface_scale


def build_archive(
    frame: pd.DataFrame,
    feasibility: dict[str, Any],
) -> tuple[dict[str, NDArray[Any]], list[dict[str, Any]], tuple[str, ...]]:
    payload: dict[str, NDArray[Any]] = {
        "schema_version": np.asarray([ARCHIVE_SCHEMA], dtype=np.str_)
    }
    expected_folds = {
        int(record["fold"]): record for record in feasibility["folds"]
    }
    fold_records: list[dict[str, Any]] = []
    similarities: list[tuple[str, float]] = []
    for fold in sorted(int(value) for value in frame["fold"].unique()):
        training = frame[frame["fold"] != fold].sort_values("prompt_id")
        validation = frame[frame["fold"] == fold].sort_values("prompt_id")
        training_ids = [str(value) for value in training["prompt_id"]]
        validation_ids = [str(value) for value in validation["prompt_id"]]
        representation = build_representation(
            [str(value) for value in training["prompt_text"]],
            [str(value) for value in validation["prompt_text"]],
        )
        hashes = representation_hashes(representation)
        expected = expected_folds[fold]
        if (
            hashes != expected["representation_hashes"]
            or representation.text_alpha != expected["text_alpha"]
            or representation.surface_alpha != expected["surface_alpha"]
            or representation.text_rank != expected["text_rank"]
            or representation.surface_rank != expected["surface_rank"]
        ):
            raise ValueError(f"fold {fold} differs from feasibility freeze")
        add_fold_payload(
            payload,
            fold,
            training_ids,
            validation_ids,
            representation,
        )
        similarities.extend(
            zip(
                validation_ids,
                (
                    float(value)
                    for value in representation.nearest_training_similarity
                ),
                strict=True,
            )
        )
        fold_records.append(
            {
                "fold": fold,
                "training_prompt_ids_sha256": hash_prompt_ids(training_ids),
                "validation_prompt_ids_sha256": hash_prompt_ids(
                    validation_ids
                ),
                "training_prompts": len(training_ids),
                "validation_prompts": len(validation_ids),
                "text_shape": list(representation.text_training.shape),
                "surface_shape": list(representation.surface_training.shape),
                "text_alpha": representation.text_alpha,
                "surface_alpha": representation.surface_alpha,
                "text_rank": representation.text_rank,
                "surface_rank": representation.surface_rank,
                "text_edf": representation.text_edf,
                "surface_edf": representation.surface_edf,
                "content_hashes": hashes,
            }
        )
    ranked = sorted(similarities, key=lambda item: (item[1], item[0]))
    low_half = tuple(prompt_id for prompt_id, _value in ranked[:77])
    expected_low_hash = feasibility["low_similarity_control"][
        "prompt_ids_sha256"
    ]
    if hash_prompt_ids(list(low_half)) != expected_low_hash:
        raise ValueError(
            "low-similarity prompt rank differs from feasibility"
        )
    payload["low_similarity_prompt_ids"] = np.asarray(low_half, dtype=np.str_)
    return payload, fold_records, low_half


def ensure_new(*paths: Path) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite H1 Stage-A artifacts: {collisions}"
        )


def main() -> None:
    args = parse_args()
    ensure_new(args.prompt_table, args.archive, args.manifest)
    lock, feasibility = validate_lock(args)
    evidence = load_json(args.evidence)
    prompt_audit = load_json(args.prompt_audit)
    development = tuple(
        str(value) for value in evidence["split"]["train_prompt_ids"]
    )
    quarantine = tuple(
        str(value) for value in evidence["split"]["test_prompt_ids"]
    )
    if (
        len(development) != PROMPTS
        or len(quarantine) != QUARANTINED_PROMPTS
        or set(development) & set(quarantine)
    ):
        raise ValueError("development/quarantine split is invalid")
    prompts = load_development_prompts(development)
    prompt_text_sha = hash_prompt_texts(prompts)
    if (
        prompt_text_sha
        != prompt_audit["provenance"]["development_prompt_text_sha256"]
    ):
        raise ValueError("development prompt text differs from prompt audit")
    roster = load_fold_roster(args.source_oof, development)
    frame = (
        prompts.merge(roster, on="prompt_id", validate="one_to_one")
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    fold_counts = tuple(
        int(value) for value in frame["fold"].value_counts().sort_index()
    )
    if sorted(fold_counts) != sorted(FOLD_COUNTS):
        raise ValueError("fold counts differ from frozen protocol")

    payload, fold_records, low_half = build_archive(frame, feasibility)
    args.prompt_table.parent.mkdir(parents=True, exist_ok=True)
    prompt_table = pa.Table.from_pandas(frame, preserve_index=False)
    metadata = dict(prompt_table.schema.metadata or {})
    metadata.update(
        {
            b"herald.schema_version": b"herald.magnitude_e1_h1_prompts.v1",
            b"herald.labels_loaded": b"false",
            b"herald.gold_loaded": b"false",
            b"herald.prompt_text_sha256": prompt_text_sha.encode(),
        }
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        prompt_table.replace_schema_metadata(metadata), args.prompt_table
    )
    np.savez_compressed(args.archive, **payload)  # type: ignore[arg-type]
    prompt_table_sha = sha256_file(args.prompt_table)
    archive_sha = sha256_file(args.archive)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "representation_frozen_before_outcome_fit",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "source_oof_sha256": sha256_file(args.source_oof),
        "evidence_sha256": sha256_file(args.evidence),
        "prompt_audit_sha256": sha256_file(args.prompt_audit),
        "target_audit_sha256": sha256_file(args.target_audit),
        "feasibility_audit_sha256": sha256_file(args.feasibility_audit),
        "understanding_sha256": sha256_file(args.understanding),
        "strategy_sha256": sha256_file(args.strategy),
        "prompt_table_sha256": prompt_table_sha,
        "representation_archive_sha256": archive_sha,
        "development_prompt_text_sha256": prompt_text_sha,
        "development_prompt_ids_sha256": hash_prompt_ids(development),
        "low_similarity_prompt_ids_sha256": hash_prompt_ids(list(low_half)),
        "development_prompts": len(frame),
        "low_similarity_prompts": len(low_half),
        "fold_counts": list(fold_counts),
        "folds": fold_records,
        "archive_keys": sorted(payload),
        "allowed_oof_columns": ["prompt_id", "fold"],
        "allowed_ifeval_columns": ["key", "prompt"],
        "outcome_or_label_columns_loaded": [],
        "gold_instruction_columns_loaded": [],
        "protected_prompt_text_rows_materialized": 0,
        "implementation_sha256": lock["implementation_sha256"],
        "confirmation_status": "sealed_not_run",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "h1_label_blind_representation_frozen",
                "prompt_table": str(args.prompt_table),
                "prompt_table_sha256": prompt_table_sha,
                "archive": str(args.archive),
                "archive_sha256": archive_sha,
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "low_similarity_prompt_ids_sha256": manifest[
                    "low_similarity_prompt_ids_sha256"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
