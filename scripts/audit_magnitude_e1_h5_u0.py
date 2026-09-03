"""Audit H5-U0 causal-representation inputs without loading outcomes."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from herald.attention_features import default_layer_indices, tap_feature_names
from herald.features import FEATURE_NAMES, derive_features
from herald.magnitude_v2 import hash_prompt_ids, sha256_file

OOF_COLUMNS = ("prompt_id", "fold")
EXPECTED_PROMPTS = 154
EXPECTED_FOLD_COUNTS = (30, 31, 31, 31, 31)
EXPECTED_RAW_WIDTH = 20
EXPECTED_TAP_WIDTH = 16
EXPECTED_RAW_DTYPE = np.dtype(np.float16)
KL_NAME = "kl_prev"
EWMA_NAMES = ("kl_prev_ewma_8", "kl_prev_ewma_32")
EXPECTED_HASHES = {
    "source_oof_sha256": (
        "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
    ),
    "sweep_config_sha256": (
        "8c228094df9f36cd582752070d578fbfe60f49cd53553b3b507076d17c874408"
    ),
    "reference_manifest_sha256": (
        "5f3d65d72ff506cfdd8105dd4a26cb065c2234c4ede3263bacf0547b1dc5cf88"
    ),
    "features_source_sha256": (
        "abc152eac453fb9e3de92ab1bfed1d628b56d77673f3542e349a1d417e28fc1d"
    ),
    "attention_source_sha256": (
        "5233b9bc13afcee585c3ba70df27240da9ad5bd4d4746141243404ac0e157c39"
    ),
    "generate_source_sha256": (
        "0c5f43e87ed4c508b12f977a2ca117441acce244700c592218f742daa833de1f"
    ),
    "runner_source_sha256": (
        "5d084543c72ab82fd099e4277a87ec9e5e1bb61fe071093454620ef578f83bc0"
    ),
}
SCHEMA_VERSION = "herald.magnitude_e1_h5_u0_understanding_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--features-source", type=Path, required=True)
    parser.add_argument("--attention-source", type=Path, required=True)
    parser.add_argument("--generate-source", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "source_oof_sha256": args.source_oof,
        "sweep_config_sha256": args.sweep_config,
        "reference_manifest_sha256": args.reference_manifest,
        "features_source_sha256": args.features_source,
        "attention_source_sha256": args.attention_source,
        "generate_source_sha256": args.generate_source,
        "runner_source_sha256": args.runner_source,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        differences = {
            name: {"expected": EXPECTED_HASHES[name], "actual": value}
            for name, value in actual.items()
            if value != EXPECTED_HASHES[name]
        }
        raise ValueError(f"H5-U0 input hashes changed: {differences}")
    return actual


def development_roster(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    if not set(OOF_COLUMNS) <= set(schema.names):
        raise ValueError("M3 OOF lacks the H5-U0 roster columns")
    table = pq.read_table(path, columns=list(OOF_COLUMNS))  # type: ignore[no-untyped-call]
    if tuple(table.column_names) != OOF_COLUMNS:
        raise ValueError("H5-U0 roster reader loaded an unexpected column")
    frame = table.to_pandas().drop_duplicates()
    prompt_folds = frame.groupby("prompt_id")["fold"].nunique()
    if int((prompt_folds != 1).sum()):
        raise ValueError("H5-U0 prompt fold is not constant")
    prompt_fold = frame.drop_duplicates("prompt_id")
    if len(prompt_fold) != EXPECTED_PROMPTS:
        raise ValueError("H5-U0 development prompt count changed")
    folds = {
        str(int(fold)): int(count)
        for fold, count in prompt_fold["fold"]
        .value_counts()
        .sort_index()
        .items()
    }
    if tuple(sorted(folds.values())) != EXPECTED_FOLD_COUNTS:
        raise ValueError(f"H5-U0 fold counts changed: {folds}")
    prompts = tuple(sorted(str(value) for value in prompt_fold["prompt_id"]))
    return prompts, folds


def declared_reference_arrays(
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    entries = manifest.get("artifact_files")
    if not isinstance(entries, list):
        raise ValueError("reference manifest artifact list is missing")
    declared: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("reference manifest artifact entry is invalid")
        relative = str(entry.get("path", ""))
        if not relative.endswith(".npy"):
            continue
        if relative in declared:
            raise ValueError(
                f"duplicate reference array declaration: {relative}"
            )
        declared[relative] = entry
    return declared


def quantiles(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        f"p{int(probability * 100):02d}": float(
            np.quantile(array, probability)
        )
        for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def raw_trace_audit(
    prompts: tuple[str, ...],
    reference_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    declared = declared_reference_arrays(manifest)
    kl_index = FEATURE_NAMES.index(KL_NAME)
    trace_lengths: list[int] = []
    source_rows = 0
    first_kl_nan_errors = 0
    internal_kl_nan_rows = 0
    non_kl_nonfinite_rows = 0
    width_errors = 0
    hash_errors = 0
    legacy_ewma_finite = {name: 0 for name in EWMA_NAMES}
    for prompt_id in prompts:
        relative = f"{prompt_id}.npy"
        path = reference_dir / relative
        entry = declared.get(relative)
        if entry is None or not path.is_file():
            raise ValueError(
                f"development reference trace is missing: {relative}"
            )
        if sha256_file(path) != entry.get("sha256"):
            hash_errors += 1
        raw = np.load(path, allow_pickle=False)
        if raw.ndim != 2 or raw.shape[1] != EXPECTED_RAW_WIDTH:
            width_errors += 1
            continue
        if raw.dtype != EXPECTED_RAW_DTYPE:
            raise ValueError(
                f"reference trace dtype {raw.dtype} changed for {prompt_id}"
            )
        trace_lengths.append(int(raw.shape[0]))
        source_rows += int(raw.shape[0])
        kl = raw[:, kl_index]
        first_kl_nan_errors += int(len(kl) == 0 or not np.isnan(kl[0]))
        internal_kl_nan_rows += int(np.isnan(kl[1:]).sum())
        non_kl = np.delete(raw, kl_index, axis=1)
        non_kl_nonfinite_rows += int((~np.isfinite(non_kl)).any(axis=1).sum())
        derived, names = derive_features(raw)
        for name in EWMA_NAMES:
            legacy_ewma_finite[name] += int(
                np.isfinite(derived[:, names.index(name)]).sum()
            )
    checks = {
        "development_arrays_exact": len(trace_lengths) == EXPECTED_PROMPTS,
        "array_hash_errors_zero": hash_errors == 0,
        "array_width_errors_zero": width_errors == 0,
        "first_kl_nan_errors_zero": first_kl_nan_errors == 0,
        "non_kl_nonfinite_rows_zero": non_kl_nonfinite_rows == 0,
        "legacy_ewmas_entirely_null": all(
            count == 0 for count in legacy_ewma_finite.values()
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "development_arrays": len(trace_lengths),
        "source_rows": source_rows,
        "raw_dtype": str(EXPECTED_RAW_DTYPE),
        "trace_length_quantiles": quantiles(trace_lengths),
        "first_kl_nan_errors": first_kl_nan_errors,
        "internal_kl_nan_rows": internal_kl_nan_rows,
        "non_kl_nonfinite_rows": non_kl_nonfinite_rows,
        "array_width_errors": width_errors,
        "array_hash_errors": hash_errors,
        "legacy_ewma_finite_rows": legacy_ewma_finite,
        "raw_reference_json_files_read": 0,
        "deduplication_unit": (
            "one raw reference array per development prompt"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H5-U0 understanding audit: {args.output}"
        )
    hashes = verify_hashes(args)
    config = load_json(args.sweep_config)
    manifest = load_json(args.reference_manifest)
    if config.get("tap_attention") is not False:
        raise ValueError("source sweep unexpectedly contains attention taps")
    if config.get("tap_layer_indices") != []:
        raise ValueError("source sweep unexpectedly selected tap layers")
    if config.get("ref_batch_size") != 1:
        raise ValueError("source reference batch size changed")
    prompts, fold_counts = development_roster(args.source_oof)
    raw_audit = raw_trace_audit(prompts, args.reference_dir, manifest)
    if not raw_audit["pass"]:
        raise ValueError(f"H5-U0 raw trace audit failed: {raw_audit}")

    tap_names = tap_feature_names()
    tap_layers = default_layer_indices(32)
    checks = {
        "input_hashes_exact": hashes == EXPECTED_HASHES,
        "development_prompt_count_exact": len(prompts) == EXPECTED_PROMPTS,
        "base_feature_width_exact": len(FEATURE_NAMES) == EXPECTED_RAW_WIDTH,
        "tap_feature_width_exact": len(tap_names) == EXPECTED_TAP_WIDTH,
        "tap_layers_exact": tap_layers == [8, 16, 24],
        "source_sweep_taps_absent": config.get("tap_attention") is False,
        "raw_trace_audit": raw_audit["pass"],
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_h5_u0_understanding_before_strategy",
        "pass": all(checks.values()),
        "checks": checks,
        "provenance": {
            **hashes,
            "source_artifact_aggregate_sha256": manifest.get(
                "artifact_aggregate_sha256"
            ),
            "development_prompt_ids_sha256": hash_prompt_ids(prompts),
            "development_fold_counts": fold_counts,
            "confirmation_status": "sealed_not_run",
        },
        "source_access": {
            "oof_columns_loaded": list(OOF_COLUMNS),
            "raw_array_columns": list(FEATURE_NAMES),
            "outcome_or_quality_columns_loaded": [],
            "reference_json_files_read": 0,
            "protected_rows_materialized": False,
        },
        "raw_trace_audit": raw_audit,
        "frozen_attention_family": {
            "feature_names": tap_names,
            "layer_indices_for_32_layers": tap_layers,
            "pooling": "mean_and_population_variance_across_layers",
            "source_sweep_attention_taps_present": False,
            "same_roster_attention_nonredundancy_measurable": False,
        },
        "legacy_ewma_defect": {
            "columns": list(EWMA_NAMES),
            "spans": [8, 32],
            "root_cause": (
                "The first kl_prev value is a design NaN that seeds "
                "both batch and online accumulators, so every later "
                "value remains NaN."
            ),
            "implementation_locations": [
                "herald.features._ewma",
                "herald.features.IncrementalDerived._update_ewma",
            ],
        },
        "interpretation_boundary": {
            "allowed": (
                "Causal timing, implementation correctness, trace coverage, "
                "and label-blind representation redundancy only."
            ),
            "forbidden": [
                "dq, q_control, q_hybrid, q_ref, or C0 residuals",
                "output text or grader material",
                "predictive relevance claims",
                "GPU replay",
                "confirmation or reserve access",
            ],
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    if not output["pass"]:
        raise ValueError(f"H5-U0 understanding checks failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
