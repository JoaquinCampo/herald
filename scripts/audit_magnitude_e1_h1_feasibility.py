"""Run the final label-blind mechanical checks before locking E1 H1."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn.linear_model as linear_model  # type: ignore[import-untyped]
from datasets import load_dataset  # type: ignore[import-untyped]
from magnitude_e1_h1_features import (
    EDF_TOLERANCE,
    SLOPE_EDF,
    Representation,
    build_representation,
)
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
RATIOS = (0.25, 0.5, 0.75, 0.875)
PROMPTS = 154
QUARANTINED_PROMPTS = 46
FOLD_COUNTS = (31, 31, 31, 31, 30)
SCHEMA_VERSION = "herald.magnitude_e1_h1_feasibility_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--prompt-audit", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--feature-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        raise ValueError("prompt source differs from development roster")
    return frame


def load_key_geometry(
    path: Path, development: tuple[str, ...]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = ["prompt_id", "compressor", "ratio", "s", "fold"]
    table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    if table.column_names != columns:
        raise ValueError("key-only audit loaded unexpected source columns")
    frame = table.to_pandas()
    if (
        set(frame["prompt_id"]) != set(development)
        or set(frame["compressor"]) != set(COMPRESSORS)
        or {float(value) for value in frame["ratio"]} != set(RATIOS)
    ):
        raise ValueError("key-only geometry differs from H1 scope")
    violations = []
    for (compressor, prompt_id), group in frame.groupby(
        ["compressor", "prompt_id"], sort=True
    ):
        state_sets = {
            float(ratio): tuple(
                sorted(int(value) for value in ratio_group["s"])
            )
            for ratio, ratio_group in group.groupby("ratio", sort=True)
        }
        if (
            set(state_sets) != set(RATIOS)
            or len(set(state_sets.values())) != 1
        ):
            violations.append(f"{compressor}/{prompt_id}")
    roster = frame[["prompt_id", "fold"]].drop_duplicates()
    if len(roster) != PROMPTS or roster["prompt_id"].duplicated().any():
        raise ValueError("fold roster is not one-to-one with prompts")
    return roster, {
        "source_columns": columns,
        "prompt_compressor_curves": PROMPTS * len(COMPRESSORS),
        "state_multiset_mismatches_across_ratios": violations,
        "equal_ratio_macro_equals_row_mean": not violations,
    }


def synthetic_target(prompt_ids: list[str]) -> NDArray[np.float64]:
    return np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(
                    f"herald-h1-solver\0{value}".encode()
                ).digest()[:8],
                "big",
            )
            / 2**64
            for value in prompt_ids
        ],
        dtype=np.float64,
    )


def fit_synthetic(
    representation: Representation,
    training_ids: list[str],
) -> dict[str, Any]:
    target = synthetic_target(training_ids)
    records = {}
    for name, training, validation, alpha in (
        (
            "text",
            representation.text_training,
            representation.text_validation,
            representation.text_alpha,
        ),
        (
            "surface",
            representation.surface_training,
            representation.surface_validation,
            representation.surface_alpha,
        ),
    ):
        predictions = []
        iterations = []
        for _ in range(2):
            model = linear_model.Ridge(
                alpha=alpha,
                fit_intercept=True,
                solver="lsqr",
                tol=1e-10,
                max_iter=10000,
            )
            model.fit(training, target)
            prediction = np.asarray(
                model.predict(validation), dtype=np.float64
            )
            predictions.append(prediction)
            n_iter = np.asarray(model.n_iter_).reshape(-1)
            iterations.append(int(n_iter[0]) if len(n_iter) else 0)
        maximum_difference = float(
            np.max(np.abs(predictions[0] - predictions[1]))
        )
        records[name] = {
            "deterministic_prediction_max_difference": maximum_difference,
            "prediction_sha256": hash_array(predictions[0]),
            "iterations": iterations,
            "converged_within_cap": all(
                value < 10000 for value in iterations
            ),
        }
    return records


def fold_feasibility(
    frame: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[tuple[str, float]]]:
    records: list[dict[str, Any]] = []
    similarities: list[tuple[str, float]] = []
    for fold in sorted(int(value) for value in frame["fold"].unique()):
        training = frame[frame["fold"] != fold].sort_values("prompt_id")
        validation = frame[frame["fold"] == fold].sort_values("prompt_id")
        training_ids = [str(value) for value in training["prompt_id"]]
        validation_ids = [str(value) for value in validation["prompt_id"]]
        training_texts = [str(value) for value in training["prompt_text"]]
        validation_texts = [str(value) for value in validation["prompt_text"]]
        first = build_representation(training_texts, validation_texts)
        second = build_representation(training_texts, validation_texts)
        first_hashes = representation_hashes(first)
        second_hashes = representation_hashes(second)
        similarities.extend(
            zip(
                validation_ids,
                (float(value) for value in first.nearest_training_similarity),
                strict=True,
            )
        )
        text_norms = np.sqrt(
            np.asarray(
                first.text_training.multiply(first.text_training).sum(axis=1)
            )
            .reshape(-1)
            .astype(np.float64)
        )
        records.append(
            {
                "fold": fold,
                "training_prompts": len(training),
                "validation_prompts": len(validation),
                "training_prompt_ids_sha256": hash_prompt_ids(training_ids),
                "validation_prompt_ids_sha256": hash_prompt_ids(
                    validation_ids
                ),
                "word_features": len(first.word_features),
                "char_features": len(first.char_features),
                "text_shape": list(first.text_training.shape),
                "surface_shape": list(first.surface_training.shape),
                "text_rank": first.text_rank,
                "surface_rank": first.surface_rank,
                "text_alpha": first.text_alpha,
                "surface_alpha": first.surface_alpha,
                "text_achieved_slope_edf": first.text_edf,
                "surface_achieved_slope_edf": first.surface_edf,
                "text_training_row_norm_max_error_from_one": float(
                    np.max(np.abs(text_norms - 1.0))
                ),
                "representation_hashes": first_hashes,
                "second_build_hashes_match": first_hashes == second_hashes,
                "second_build_alphas_match": (
                    first.text_alpha == second.text_alpha
                    and first.surface_alpha == second.surface_alpha
                ),
                "synthetic_solver": fit_synthetic(first, training_ids),
            }
        )
    return records, similarities


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H1 feasibility: {args.output}"
        )
    evidence = load_json(args.evidence)
    protocol = load_json(args.protocol_lock)
    prompt_audit = load_json(args.prompt_audit)
    target_audit = load_json(args.target_audit)
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
    if sha256_file(args.evidence) != protocol["source_evidence_sha256"]:
        raise ValueError("evidence hash differs from protocol")
    if (
        sha256_file(args.source_oof)
        != target_audit["provenance"]["source_oof_sha256"]
    ):
        raise ValueError("source OOF differs from target audit")

    prompts = load_development_prompts(development)
    prompt_text_sha = hash_prompt_texts(prompts)
    if (
        prompt_text_sha
        != prompt_audit["provenance"]["development_prompt_text_sha256"]
    ):
        raise ValueError("development prompt text differs from prompt audit")
    roster, target_equivalence = load_key_geometry(
        args.source_oof, development
    )
    frame = prompts.merge(roster, on="prompt_id", validate="one_to_one")
    fold_counts = tuple(
        int(value) for value in frame["fold"].value_counts().sort_index()
    )
    if sorted(fold_counts) != sorted(FOLD_COUNTS):
        raise ValueError("fold counts differ from frozen protocol")

    folds, similarities = fold_feasibility(frame)
    sorted_similarity = sorted(
        similarities, key=lambda item: (item[1], item[0])
    )
    low_half = tuple(
        prompt_id for prompt_id, _value in sorted_similarity[:77]
    )
    low_half_sha256 = hash_prompt_ids(list(low_half))
    checks = {
        "target_equivalence": target_equivalence[
            "equal_ratio_macro_equals_row_mean"
        ],
        "all_text_ranks_above_edf": all(
            int(fold["text_rank"]) > SLOPE_EDF for fold in folds
        ),
        "all_surface_ranks_above_edf": all(
            int(fold["surface_rank"]) > SLOPE_EDF for fold in folds
        ),
        "all_text_edf_exact": all(
            abs(float(fold["text_achieved_slope_edf"]) - SLOPE_EDF)
            <= EDF_TOLERANCE
            for fold in folds
        ),
        "all_surface_edf_exact": all(
            abs(float(fold["surface_achieved_slope_edf"]) - SLOPE_EDF)
            <= EDF_TOLERANCE
            for fold in folds
        ),
        "all_representation_builds_deterministic": all(
            bool(fold["second_build_hashes_match"])
            and bool(fold["second_build_alphas_match"])
            for fold in folds
        ),
        "all_synthetic_solver_predictions_deterministic": all(
            float(values["deterministic_prediction_max_difference"]) == 0.0
            and bool(values["converged_within_cap"])
            for fold in folds
            for values in dict(fold["synthetic_solver"]).values()
        ),
        "all_text_union_training_rows_unit_norm": all(
            float(fold["text_training_row_norm_max_error_from_one"]) <= 1e-12
            for fold in folds
        ),
        "low_similarity_half_has_77_unique_prompts": len(low_half) == 77
        and len(set(low_half)) == 77,
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "label_blind_h1_lock_feasibility_complete",
        "pass": all(checks.values()),
        "checks": checks,
        "provenance": {
            "source_oof_sha256": sha256_file(args.source_oof),
            "evidence_sha256": sha256_file(args.evidence),
            "protocol_lock_sha256": sha256_file(args.protocol_lock),
            "prompt_audit_sha256": sha256_file(args.prompt_audit),
            "target_audit_sha256": sha256_file(args.target_audit),
            "feature_implementation_sha256": sha256_file(args.feature_script),
            "development_prompt_ids_sha256": hash_prompt_ids(development),
            "development_prompt_text_sha256": prompt_text_sha,
            "confirmation_status": "sealed_not_run",
            "outcome_or_label_columns_loaded": [],
            "gold_instruction_columns_loaded": [],
            "protected_prompt_text_rows_materialized": 0,
        },
        "target_equivalence": target_equivalence,
        "folds": folds,
        "low_similarity_control": {
            "definition": (
                "first 77 by ascending fold-local equal-union "
                "nearest-training cosine, prompt_id tie break"
            ),
            "prompt_ids_sha256": low_half_sha256,
            "minimum_similarity": sorted_similarity[0][1],
            "maximum_included_similarity": sorted_similarity[76][1],
            "minimum_excluded_similarity": sorted_similarity[77][1],
            "maximum_similarity": sorted_similarity[-1][1],
            "prompt_count": len(low_half),
        },
        "interpretation": {
            "allowed": (
                "Mechanical label-blind proof that the exact H1 "
                "representation, EDF rule, solver, target weighting, and "
                "transfer rank are lockable."
            ),
            "forbidden": [
                "outcome inference",
                "candidate comparison",
                "hyperparameter tuning",
                "future-prompt or confirmatory claims",
            ],
        },
    }
    if not audit["pass"]:
        raise ValueError(f"H1 feasibility check failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "h1_feasibility_complete",
                "pass": audit["pass"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "low_similarity_prompt_ids_sha256": low_half_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
