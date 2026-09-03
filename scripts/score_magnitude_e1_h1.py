"""Fit and score the frozen E1 H1 prompt-semantics representation."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn.linear_model as linear_model  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
RATIOS = (0.25, 0.5, 0.75, 0.875)
SOURCE_COLUMNS = ("prompt_id", "compressor", "ratio", "s", "dq", "fold")
PROMPTS = 154
TOTAL_ROWS = 45180
FOLDS = 5
LOW_PROMPTS = 77
RESAMPLES = 10000
SEED = 314159
CONFIDENCE = 0.95
MINIMUM_OVERALL_SKILL = 0.01
RIDGE_TOLERANCE = 1e-10
RIDGE_MAX_ITERATIONS = 10000
REPORT_SCHEMA = "herald.magnitude_e1_h1.v1"
OOF_SCHEMA = "herald.magnitude_e1_h1_oof.v1"
ARCHIVE_SCHEMA = "herald.magnitude_e1_h1_representation.v1"
MANIFEST_SCHEMA = "herald.magnitude_e1_h1_representation_manifest.v1"
PROTOCOL_LOCK_SCHEMA = "herald.magnitude_e1_h1_protocol_lock.v1"
SCORING_LOCK_SCHEMA = "herald.magnitude_e1_h1_scoring_lock.v1"
CLAIM_KINDS = (
    "overall_vs_mean",
    "overall_vs_surface",
    "low_vs_mean",
    "low_vs_surface",
)


@dataclass(frozen=True, slots=True)
class FoldRepresentation:
    training_prompt_ids: tuple[str, ...]
    validation_prompt_ids: tuple[str, ...]
    text_training: sparse.csr_matrix
    text_validation: sparse.csr_matrix
    surface_training: NDArray[np.float64]
    surface_validation: NDArray[np.float64]
    nearest_training_similarity: NDArray[np.float64]
    text_alpha: float
    surface_alpha: float
    text_rank: int
    surface_rank: int
    text_edf: float
    surface_edf: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--prompt-table", type=Path, required=True)
    parser.add_argument("--representation-archive", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--scoring-lock", type=Path, required=True)
    parser.add_argument("--output-oof", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
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


def validate_locks(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(args.protocol_lock)
    scoring = load_json(args.scoring_lock)
    manifest = load_json(args.representation_manifest)
    if (
        protocol.get("schema_version") != PROTOCOL_LOCK_SCHEMA
        or protocol.get("status")
        != "locked_before_label_blind_representation"
    ):
        raise ValueError("H1 protocol lock schema/status is invalid")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status")
        != "representation_frozen_before_outcome_fit"
    ):
        raise ValueError(
            "H1 representation manifest schema/status is invalid"
        )
    if (
        scoring.get("schema_version") != SCORING_LOCK_SCHEMA
        or scoring.get("status")
        != "locked_after_representation_before_outcome_fit"
    ):
        raise ValueError("H1 scoring lock schema/status is invalid")
    expected_hashes = {
        "source_oof_sha256": sha256_file(args.source_oof),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "prompt_table_sha256": sha256_file(args.prompt_table),
        "representation_archive_sha256": sha256_file(
            args.representation_archive
        ),
        "representation_manifest_sha256": sha256_file(
            args.representation_manifest
        ),
    }
    for key, expected in expected_hashes.items():
        if scoring.get(key) != expected:
            raise ValueError(f"H1 scoring lock {key} mismatch")
    if (
        manifest.get("protocol_lock_sha256")
        != expected_hashes["protocol_lock_sha256"]
        or manifest.get("source_oof_sha256")
        != expected_hashes["source_oof_sha256"]
        or manifest.get("prompt_table_sha256")
        != expected_hashes["prompt_table_sha256"]
        or manifest.get("representation_archive_sha256")
        != expected_hashes["representation_archive_sha256"]
    ):
        raise ValueError("H1 representation provenance differs from lock")
    if (
        manifest.get("outcome_or_label_columns_loaded") != []
        or manifest.get("gold_instruction_columns_loaded") != []
        or manifest.get("protected_prompt_text_rows_materialized") != 0
    ):
        raise ValueError("H1 representation manifest is not label blind")
    expected_implementation = {
        "scripts/magnitude_e1_h1_features.py": protocol[
            "implementation_sha256"
        ]["scripts/magnitude_e1_h1_features.py"],
        "scripts/compile_magnitude_e1_h1_representation.py": protocol[
            "implementation_sha256"
        ]["scripts/compile_magnitude_e1_h1_representation.py"],
        "scripts/score_magnitude_e1_h1.py": sha256_file(Path(__file__)),
    }
    if (
        protocol.get("implementation_sha256") != expected_implementation
        or manifest.get("implementation_sha256") != expected_implementation
        or scoring.get("implementation_sha256") != expected_implementation
    ):
        raise ValueError("H1 implementation differs across frozen artifacts")
    expected_scoring = {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "cluster": "complete_prompt",
        "same_draws_across_all_claims": True,
        "claims": 12,
        "bounds": "one_sided_simultaneous_max_t_lower_and_upper",
        "minimum_overall_skill": MINIMUM_OVERALL_SKILL,
        "low_similarity_prompt_count": LOW_PROMPTS,
    }
    if scoring.get("scoring") != expected_scoring:
        raise ValueError("H1 scoring configuration differs from lock")
    if (
        scoring.get("confirmation_status") != "sealed_not_run"
        or manifest.get("confirmation_status") != "sealed_not_run"
    ):
        raise ValueError("H1 protected-data status is invalid")
    return scoring, manifest


def csr_from_archive(
    archive: Mapping[str, NDArray[Any]], prefix: str
) -> sparse.csr_matrix:
    shape_values = np.asarray(archive[f"{prefix}_shape"], dtype=np.int64)
    if shape_values.shape != (2,):
        raise ValueError(f"invalid sparse shape for {prefix}")
    shape = (int(shape_values[0]), int(shape_values[1]))
    matrix = sparse.csr_matrix(
        (
            np.asarray(archive[f"{prefix}_data"], dtype=np.float64),
            np.asarray(archive[f"{prefix}_indices"], dtype=np.int32),
            np.asarray(archive[f"{prefix}_indptr"], dtype=np.int32),
        ),
        shape=shape,
    )
    if not np.isfinite(matrix.data).all():
        raise ValueError(f"nonfinite sparse representation for {prefix}")
    return matrix


def archive_scalar(
    archive: Mapping[str, NDArray[Any]], key: str, dtype: type[Any]
) -> float | int:
    values = np.asarray(archive[key]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"archive scalar {key} has unexpected shape")
    return cast("float | int", dtype(values[0]))


def load_representation(
    path: Path, manifest: dict[str, Any]
) -> tuple[dict[int, FoldRepresentation], tuple[str, ...]]:
    with np.load(path, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    if sorted(archive) != manifest["archive_keys"]:
        raise ValueError(
            "H1 representation archive keys differ from manifest"
        )
    schema = tuple(str(value) for value in archive["schema_version"])
    if schema != (ARCHIVE_SCHEMA,):
        raise ValueError("H1 representation archive schema is invalid")
    low_ids = tuple(
        str(value) for value in archive["low_similarity_prompt_ids"]
    )
    if (
        len(low_ids) != LOW_PROMPTS
        or len(set(low_ids)) != LOW_PROMPTS
        or hash_prompt_ids(list(low_ids))
        != manifest["low_similarity_prompt_ids_sha256"]
    ):
        raise ValueError("frozen H1 low-similarity roster is invalid")
    representations = {}
    for fold in range(FOLDS):
        prefix = f"fold_{fold}"
        training_ids = tuple(
            str(value) for value in archive[f"{prefix}_training_prompt_ids"]
        )
        validation_ids = tuple(
            str(value) for value in archive[f"{prefix}_validation_prompt_ids"]
        )
        value = FoldRepresentation(
            training_prompt_ids=training_ids,
            validation_prompt_ids=validation_ids,
            text_training=csr_from_archive(
                archive, f"{prefix}_text_training"
            ),
            text_validation=csr_from_archive(
                archive, f"{prefix}_text_validation"
            ),
            surface_training=np.asarray(
                archive[f"{prefix}_surface_training"], dtype=np.float64
            ),
            surface_validation=np.asarray(
                archive[f"{prefix}_surface_validation"], dtype=np.float64
            ),
            nearest_training_similarity=np.asarray(
                archive[f"{prefix}_nearest_training_similarity"],
                dtype=np.float64,
            ),
            text_alpha=float(
                archive_scalar(archive, f"{prefix}_text_alpha", float)
            ),
            surface_alpha=float(
                archive_scalar(archive, f"{prefix}_surface_alpha", float)
            ),
            text_rank=int(
                archive_scalar(archive, f"{prefix}_text_rank", int)
            ),
            surface_rank=int(
                archive_scalar(archive, f"{prefix}_surface_rank", int)
            ),
            text_edf=float(
                archive_scalar(archive, f"{prefix}_text_edf", float)
            ),
            surface_edf=float(
                archive_scalar(archive, f"{prefix}_surface_edf", float)
            ),
        )
        if (
            value.text_training.shape[0] != len(training_ids)
            or value.text_validation.shape[0] != len(validation_ids)
            or value.surface_training.shape != (len(training_ids), 16)
            or value.surface_validation.shape != (len(validation_ids), 16)
            or value.nearest_training_similarity.shape
            != (len(validation_ids),)
            or value.text_training.shape[1] != value.text_validation.shape[1]
            or not np.isfinite(value.surface_training).all()
            or not np.isfinite(value.surface_validation).all()
            or not np.isfinite(value.nearest_training_similarity).all()
            or not np.isfinite(value.text_alpha)
            or not np.isfinite(value.surface_alpha)
            or value.text_alpha <= 0.0
            or value.surface_alpha <= 0.0
            or value.text_rank <= 10
            or value.surface_rank <= 10
            or abs(value.text_edf - 10.0) > 1e-10
            or abs(value.surface_edf - 10.0) > 1e-10
        ):
            raise ValueError(f"fold {fold} representation is invalid")
        representations[fold] = value
    return representations, low_ids


def read_prompt_roster(path: Path) -> pd.DataFrame:
    columns = ["prompt_id", "fold"]
    table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    if table.column_names != columns:
        raise ValueError("H1 scorer loaded unexpected prompt-table columns")
    frame = table.to_pandas()
    if (
        len(frame) != PROMPTS
        or frame["prompt_id"].duplicated().any()
        or frame["fold"].nunique() != FOLDS
    ):
        raise ValueError("H1 prompt roster is invalid")
    return frame


def read_source(path: Path) -> pd.DataFrame:
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("H1 scorer loaded unexpected source columns")
    frame = table.to_pandas()
    if (
        len(frame) != TOTAL_ROWS
        or frame[["prompt_id", "compressor", "ratio", "s"]].duplicated().any()
        or set(frame["compressor"]) != set(COMPRESSORS)
        or {float(value) for value in frame["ratio"]} != set(RATIOS)
        or frame["prompt_id"].nunique() != PROMPTS
        or not np.isfinite(frame["dq"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("H1 scoring source differs from contract")
    return frame


def build_targets(source: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    source_roster = (
        source[["prompt_id", "fold"]]
        .drop_duplicates()
        .sort_values("prompt_id")
        .reset_index(drop=True)
    )
    expected_roster = roster.sort_values("prompt_id").reset_index(drop=True)
    if (
        len(source_roster) != PROMPTS
        or source_roster["prompt_id"].duplicated().any()
        or not source_roster.equals(expected_roster)
    ):
        raise ValueError("source fold roster differs from prompt table")
    state_mismatches = 0
    for (_prompt_id, _compressor), group in source.groupby(
        ["prompt_id", "compressor"], sort=True
    ):
        state_sets = {
            float(ratio_value): tuple(
                sorted(int(value) for value in ratio_group["s"])
            )
            for ratio_value, ratio_group in group.groupby("ratio", sort=True)
        }
        if (
            set(state_sets) != set(RATIOS)
            or len(set(state_sets.values())) != 1
        ):
            state_mismatches += 1
    if state_mismatches:
        raise ValueError(
            f"{state_mismatches} prompt-compressor state sets differ by ratio"
        )
    ratio = (
        source.groupby(
            ["prompt_id", "compressor", "ratio"], observed=True, sort=True
        )
        .agg(ratio_dq=("dq", "mean"), states=("s", "count"))
        .reset_index()
    )
    ratio_counts = ratio.groupby(["prompt_id", "compressor"])[
        "ratio"
    ].nunique()
    if int((ratio_counts != len(RATIOS)).sum()):
        raise ValueError("H1 target has incomplete ratio curves")
    target = (
        ratio.groupby(["prompt_id", "compressor"], sort=True)["ratio_dq"]
        .mean()
        .rename("target")
        .reset_index()
    )
    row_mean = (
        source.groupby(["prompt_id", "compressor"], sort=True)["dq"]
        .mean()
        .rename("row_mean")
        .reset_index()
    )
    target = target.merge(
        row_mean,
        on=["prompt_id", "compressor"],
        validate="one_to_one",
    )
    if not np.allclose(
        target["target"], target["row_mean"], rtol=0.0, atol=1e-15
    ):
        raise ValueError("explicit equal-ratio target differs from row mean")
    target = target.drop(columns="row_mean").merge(
        roster, on="prompt_id", validate="many_to_one"
    )
    if (
        len(target) != PROMPTS * len(COMPRESSORS)
        or target[["prompt_id", "compressor"]].duplicated().any()
    ):
        raise ValueError("H1 prompt-macro target table is incomplete")
    return target


def model_iterations(model: Any) -> int:
    values = np.asarray(model.n_iter_).reshape(-1)
    return int(values[0]) if len(values) else 0


def fit_oof(
    targets: pd.DataFrame,
    representations: dict[int, FoldRepresentation],
    low_ids: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    low_set = set(low_ids)
    rows = []
    solver: dict[str, Any] = {
        compressor: {str(fold): {} for fold in range(FOLDS)}
        for compressor in COMPRESSORS
    }
    for fold, representation in representations.items():
        expected_training = set(
            targets.loc[targets["fold"] != fold, "prompt_id"]
        )
        expected_validation = set(
            targets.loc[targets["fold"] == fold, "prompt_id"]
        )
        if (
            set(representation.training_prompt_ids) != expected_training
            or set(representation.validation_prompt_ids)
            != expected_validation
        ):
            raise ValueError(f"fold {fold} archive roster mismatch")
        for compressor in COMPRESSORS:
            group = targets[targets["compressor"] == compressor].set_index(
                "prompt_id"
            )
            training_target = group.loc[
                list(representation.training_prompt_ids), "target"
            ].to_numpy(dtype=np.float64)
            validation_target = group.loc[
                list(representation.validation_prompt_ids), "target"
            ].to_numpy(dtype=np.float64)
            predictions = {}
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
                model = linear_model.Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                    solver="lsqr",
                    tol=RIDGE_TOLERANCE,
                    max_iter=RIDGE_MAX_ITERATIONS,
                )
                model.fit(training, training_target)
                prediction = np.asarray(
                    model.predict(validation), dtype=np.float64
                )
                iterations = model_iterations(model)
                if (
                    prediction.shape != validation_target.shape
                    or not np.isfinite(prediction).all()
                    or iterations >= RIDGE_MAX_ITERATIONS
                ):
                    raise ValueError(
                        f"{compressor}/{fold}/{name} fit is invalid"
                    )
                predictions[name] = prediction
                solver[compressor][str(fold)][name] = {
                    "iterations": iterations,
                    "alpha": alpha,
                }
            mean_prediction = float(np.mean(training_target))
            for index, prompt_id in enumerate(
                representation.validation_prompt_ids
            ):
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "compressor": compressor,
                        "fold": fold,
                        "target": float(validation_target[index]),
                        "text_prediction": float(predictions["text"][index]),
                        "surface_prediction": float(
                            predictions["surface"][index]
                        ),
                        "mean_prediction": mean_prediction,
                        "nearest_training_similarity": float(
                            representation.nearest_training_similarity[index]
                        ),
                        "low_similarity": prompt_id in low_set,
                    }
                )
    frame = (
        pd.DataFrame(rows)
        .sort_values(["compressor", "prompt_id"])
        .reset_index(drop=True)
    )
    if (
        len(frame) != PROMPTS * len(COMPRESSORS)
        or frame[["prompt_id", "compressor"]].duplicated().any()
        or frame.groupby("compressor")["prompt_id"].nunique().to_dict()
        != dict.fromkeys(COMPRESSORS, PROMPTS)
        or frame.groupby("compressor")["low_similarity"].sum().to_dict()
        != dict.fromkeys(COMPRESSORS, LOW_PROMPTS)
    ):
        raise ValueError("H1 OOF prediction table is incomplete")
    return frame, solver


def claim_name(compressor: str, kind: str) -> str:
    return f"{compressor}::{kind}"


def skill(
    numerator: NDArray[np.float64], denominator: NDArray[np.float64]
) -> float:
    denominator_mean = float(np.mean(denominator))
    if denominator_mean <= 0.0:
        raise ValueError("H1 comparator loss is not positive")
    return 1.0 - float(np.mean(numerator)) / denominator_mean


def bootstrap_skill(
    numerator: NDArray[np.float64],
    denominator: NDArray[np.float64],
    draws: NDArray[np.int64],
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    selected_numerator = numerator[draws]
    selected_denominator = denominator[draws]
    if mask is None:
        numerator_mean = np.mean(selected_numerator, axis=1)
        denominator_mean = np.mean(selected_denominator, axis=1)
    else:
        selected = mask[draws]
        counts = np.sum(selected, axis=1)
        if np.any(counts == 0):
            raise ValueError("a global bootstrap draw omitted the low half")
        numerator_mean = (
            np.sum(selected_numerator * selected, axis=1) / counts
        )
        denominator_mean = (
            np.sum(selected_denominator * selected, axis=1) / counts
        )
    if np.any(denominator_mean <= 0.0):
        raise ValueError("a bootstrap comparator loss is not positive")
    return np.asarray(
        1.0 - numerator_mean / denominator_mean, dtype=np.float64
    )


def max_t_intervals(
    estimates: Mapping[str, float],
    bootstrap: Mapping[str, NDArray[np.float64]],
    direction: str,
) -> dict[str, Any]:
    claims = tuple(sorted(estimates))
    if len(claims) != len(COMPRESSORS) * len(CLAIM_KINDS):
        raise ValueError("H1 max-T family does not contain 12 claims")
    matrix = np.column_stack([bootstrap[claim] for claim in claims])
    point = np.asarray(
        [estimates[claim] for claim in claims], dtype=np.float64
    )
    standard_errors = np.std(matrix, axis=0, ddof=1)
    centered = matrix - point
    root = centered if direction == "lower" else -centered
    standardized = np.divide(
        root,
        standard_errors,
        out=np.zeros_like(root),
        where=standard_errors > 0.0,
    )
    critical = max(
        0.0,
        float(np.quantile(np.max(standardized, axis=1), CONFIDENCE)),
    )
    intervals = {}
    for column, claim in enumerate(claims):
        bound = critical * standard_errors[column]
        interval = {
            "estimate": float(point[column]),
            "standard_error": float(standard_errors[column]),
        }
        interval[direction] = float(
            point[column] - bound
            if direction == "lower"
            else point[column] + bound
        )
        intervals[claim] = interval
    return {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "direction": direction,
        "sidedness": "one-sided",
        "multiplicity": "simultaneous max-T over 12 claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_sets = {
        compressor: tuple(sorted(str(value) for value in group["prompt_id"]))
        for compressor, group in frame.groupby("compressor", sort=True)
    }
    if len(set(prompt_sets.values())) != 1:
        raise ValueError("H1 compressors do not share the same prompt order")
    prompts = next(iter(prompt_sets.values()))
    draws = np.random.default_rng(SEED).integers(
        0, PROMPTS, size=(RESAMPLES, PROMPTS), dtype=np.int64
    )
    estimates = {}
    bootstrap = {}
    metrics = {}
    diagnostics = {}
    for compressor in COMPRESSORS:
        group = (
            frame[frame["compressor"] == compressor]
            .set_index("prompt_id")
            .loc[list(prompts)]
        )
        target = group["target"].to_numpy(dtype=np.float64)
        losses = {
            name: (
                target
                - group[f"{name}_prediction"].to_numpy(dtype=np.float64)
            )
            ** 2
            for name in ("text", "surface", "mean")
        }
        low = group["low_similarity"].to_numpy(dtype=np.bool_)
        if int(np.sum(low)) != LOW_PROMPTS:
            raise ValueError("H1 low-similarity mask is incomplete")
        claims = {
            "overall_vs_mean": (losses["text"], losses["mean"], None),
            "overall_vs_surface": (
                losses["text"],
                losses["surface"],
                None,
            ),
            "low_vs_mean": (losses["text"], losses["mean"], low),
            "low_vs_surface": (
                losses["text"],
                losses["surface"],
                low,
            ),
        }
        compressor_skills = {}
        for kind, (numerator, denominator, mask) in claims.items():
            point = skill(
                numerator if mask is None else numerator[mask],
                denominator if mask is None else denominator[mask],
            )
            name = claim_name(compressor, kind)
            estimates[name] = point
            bootstrap[name] = bootstrap_skill(
                numerator, denominator, draws, mask
            )
            compressor_skills[kind] = point
        high = ~low
        metrics[compressor] = {
            "loss": {
                name: float(np.mean(values))
                for name, values in losses.items()
            },
            "skills": compressor_skills,
        }
        fold_diagnostics = {}
        for fold in range(FOLDS):
            fold_mask = group["fold"].to_numpy(dtype=np.int64) == fold
            fold_diagnostics[str(fold)] = {
                "prompts": int(np.sum(fold_mask)),
                "text_vs_mean_skill": skill(
                    losses["text"][fold_mask], losses["mean"][fold_mask]
                ),
                "text_vs_surface_skill": skill(
                    losses["text"][fold_mask],
                    losses["surface"][fold_mask],
                ),
            }
        diagnostics[compressor] = {
            "surface_vs_mean_skill": skill(losses["surface"], losses["mean"]),
            "low_surface_vs_mean_skill": skill(
                losses["surface"][low], losses["mean"][low]
            ),
            "high_text_vs_mean_skill": skill(
                losses["text"][high], losses["mean"][high]
            ),
            "high_text_vs_surface_skill": skill(
                losses["text"][high], losses["surface"][high]
            ),
            "folds": fold_diagnostics,
            "prediction_summary": {
                name: {
                    "mean": float(np.mean(group[f"{name}_prediction"])),
                    "std": float(np.std(group[f"{name}_prediction"], ddof=1)),
                }
                for name in ("text", "surface", "mean")
            },
        }
    lower = max_t_intervals(estimates, bootstrap, "lower")
    upper = max_t_intervals(estimates, bootstrap, "upper")
    qualifying = []
    ruled_out = {}
    for compressor in COMPRESSORS:
        thresholds = {
            "overall_vs_mean": MINIMUM_OVERALL_SKILL,
            "overall_vs_surface": MINIMUM_OVERALL_SKILL,
            "low_vs_mean": 0.0,
            "low_vs_surface": 0.0,
        }
        qualifies = all(
            lower["intervals"][claim_name(compressor, kind)]["lower"]
            > threshold
            for kind, threshold in thresholds.items()
        )
        ruled_out[compressor] = any(
            upper["intervals"][claim_name(compressor, kind)]["upper"]
            <= threshold
            for kind, threshold in thresholds.items()
        )
        if qualifies:
            qualifying.append(compressor)
    if qualifying:
        status = "positive"
        action = "freeze_qualifiers_then_design_separate_incremental_C0_test"
    elif all(ruled_out.values()):
        status = "negative"
        action = "retire_cheap_H1_branch_and_return_to_distinct_hypothesis"
    else:
        status = "ambiguous"
        action = "no_go_for_H1_and_return_to_distinct_hypothesis"
    decision = {
        "status": status,
        "go": status == "positive",
        "qualifying_compressors": qualifying,
        "compressor_ruled_out": ruled_out,
        "action": action,
    }
    return {
        "metrics": metrics,
        "diagnostics": diagnostics,
        "inference": {
            "minimum_overall_skill": MINIMUM_OVERALL_SKILL,
            "lower": lower,
            "upper": upper,
        },
        "decision": decision,
    }, {name: values.tolist() for name, values in bootstrap.items()}


def ensure_new(*paths: Path) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite H1 results: {collisions}"
        )


def main() -> None:
    args = parse_args()
    ensure_new(args.output_oof, args.output_report)
    scoring_lock, manifest = validate_locks(args)
    representations, low_ids = load_representation(
        args.representation_archive, manifest
    )
    roster = read_prompt_roster(args.prompt_table)
    source = read_source(args.source_oof)
    if set(roster["prompt_id"]) != set(source["prompt_id"]):
        raise ValueError("H1 prompt-table/source roster mismatch")
    targets = build_targets(source, roster)
    oof, solver = fit_oof(targets, representations, low_ids)
    results, bootstrap_values = evaluate(oof)

    single_state = (
        source.groupby(["prompt_id", "compressor"])["s"].nunique() == 1
    )
    single_state_prompts = int(
        source.groupby("prompt_id")["s"].nunique().eq(1).sum()
    )
    checks = {
        "representation_manifest_label_blind": manifest[
            "outcome_or_label_columns_loaded"
        ]
        == [],
        "representation_manifest_gold_blind": manifest[
            "gold_instruction_columns_loaded"
        ]
        == [],
        "protected_prompt_text_unmaterialized": manifest[
            "protected_prompt_text_rows_materialized"
        ]
        == 0,
        "source_and_prompt_rosters_match": set(roster["prompt_id"])
        == set(source["prompt_id"]),
        "all_prompt_compressor_targets_present": len(targets)
        == PROMPTS * len(COMPRESSORS),
        "all_oof_predictions_present": len(oof) == PROMPTS * len(COMPRESSORS),
        "low_similarity_roster_exact": hash_prompt_ids(list(low_ids))
        == manifest["low_similarity_prompt_ids_sha256"],
        "single_state_prompts_retained": single_state_prompts == 3
        and int(single_state.sum()) == 3 * len(COMPRESSORS),
        "all_solver_fits_converged": all(
            int(values["iterations"]) < RIDGE_MAX_ITERATIONS
            for compressor in solver.values()
            for fold in compressor.values()
            for values in fold.values()
        ),
        "all_outputs_finite": bool(
            np.isfinite(
                oof[
                    [
                        "target",
                        "text_prediction",
                        "surface_prediction",
                        "mean_prediction",
                        "nearest_training_similarity",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
        "confirmation_remained_sealed": scoring_lock["confirmation_status"]
        == "sealed_not_run",
    }
    falsification_pass = all(checks.values())
    if not falsification_pass:
        results["decision"] = {
            "status": "invalid",
            "go": False,
            "qualifying_compressors": [],
            "compressor_ruled_out": {},
            "action": "return_to_understand_and_repair_exact_protocol",
        }

    args.output_oof.parent.mkdir(parents=True, exist_ok=True)
    oof_table = pa.Table.from_pandas(oof, preserve_index=False)
    metadata = dict(oof_table.schema.metadata or {})
    metadata.update(
        {
            b"herald.schema_version": OOF_SCHEMA.encode(),
            b"herald.source_oof_sha256": sha256_file(
                args.source_oof
            ).encode(),
            b"herald.scoring_lock_sha256": sha256_file(
                args.scoring_lock
            ).encode(),
        }
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        oof_table.replace_schema_metadata(metadata), args.output_oof
    )
    oof_sha = sha256_file(args.output_oof)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "evaluated_adaptive_development_semantics_probe",
        "provenance": {
            "source_oof_sha256": sha256_file(args.source_oof),
            "prompt_table_sha256": sha256_file(args.prompt_table),
            "representation_archive_sha256": sha256_file(
                args.representation_archive
            ),
            "representation_manifest_sha256": sha256_file(
                args.representation_manifest
            ),
            "protocol_lock_sha256": sha256_file(args.protocol_lock),
            "scoring_lock_sha256": sha256_file(args.scoring_lock),
            "implementation_sha256": sha256_file(Path(__file__)),
            "oof_predictions_sha256": oof_sha,
            "development_prompts": PROMPTS,
            "prediction_rows": len(oof),
            "confirmation_status": "sealed_not_run",
        },
        **results,
        "solver": solver,
        "falsification": {
            "pass": falsification_pass,
            "checks": checks,
        },
        "bootstrap_audit": {
            "claim_draw_minimum": {
                claim: float(np.min(values))
                for claim, values in bootstrap_values.items()
            },
            "claim_draw_maximum": {
                claim: float(np.max(values))
                for claim, values in bootstrap_values.items()
            },
        },
        "interpretation": {
            "allowed": (
                "Adaptive direct prompt-macro lexical-transfer evidence on "
                "the frozen development folds only."
            ),
            "forbidden": [
                "incremental improvement over C0 or M3",
                "future-prompt generalization",
                "confirmatory evidence",
                "opening protected data",
                "semantic understanding claim",
            ],
        },
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "h1_scored",
                "decision": results["decision"],
                "skills": {
                    compressor: results["metrics"][compressor]["skills"]
                    for compressor in COMPRESSORS
                },
                "falsification_pass": falsification_pass,
                "oof_sha256": oof_sha,
                "report": str(args.output_report),
                "report_sha256": sha256_file(args.output_report),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
