"""Audit label-blind trajectory geometry before choosing an H4 experiment."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from herald.magnitude_v2 import hash_prompt_ids, sha256_file

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
SOURCE_COLUMNS = (
    *KEY_COLUMNS,
    "fold",
    "selected_candidate",
    "m2_prediction",
)
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
EXPECTED_ROWS = 45_180
EXPECTED_ROWS_PER_COMPRESSOR = 15_060
EXPECTED_ROWS_PER_COMPRESSOR_RATIO = 3_765
EXPECTED_PROMPTS = 154
EXPECTED_TRAJECTORIES = 1_848
EXPECTED_STRIDE = 16
EXPECTED_M3_RESULT_FREEZE_SHA256 = (
    "68658017124be0cf32c0c2344a43c43fe04c42fbcaf7776a78e9272bb2aaa57d"
)
EXPECTED_M3_REPORT_SHA256 = (
    "8cd5258cea3f8cf820bccf6d02af983ae6ee1c077324743bda7430e8dcdeb7ae"
)
EXPECTED_M3_OOF_SHA256 = (
    "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
)
EXPECTED_M3_PROTOCOL_SHA256 = (
    "04921e166324bc5b730c2e20c363769fe9eae2846b2909c41ee4e6f4e9ba7144"
)
EXPECTED_E1_AUDIT_SHA256 = (
    "4cd5e2bc211db380423bfe417270dacca99c1ebf6b767ce189dfbb569633a870"
)
SCHEMA_VERSION = "herald.magnitude_e1_h4_understanding_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--result-freeze", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--e1-audit", type=Path, required=True)
    parser.add_argument("--predictor-contract", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def require_hash(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash {actual} != {expected}")


def validate_provenance(args: argparse.Namespace) -> dict[str, Any]:
    require_hash(
        args.result_freeze,
        EXPECTED_M3_RESULT_FREEZE_SHA256,
        "M3 result freeze",
    )
    require_hash(args.source_oof, EXPECTED_M3_OOF_SHA256, "M3 OOF")
    require_hash(
        args.protocol_lock,
        EXPECTED_M3_PROTOCOL_SHA256,
        "M3 protocol lock",
    )
    require_hash(args.e1_audit, EXPECTED_E1_AUDIT_SHA256, "E1 audit")

    result_freeze = load_json(args.result_freeze)
    protocol_lock = load_json(args.protocol_lock)
    e1_audit = load_json(args.e1_audit)
    if result_freeze.get("status") != "frozen_final_no_confirmation":
        raise ValueError("M3 result freeze status changed")
    if result_freeze.get("report") != {
        "schema_version": "herald.magnitude_v3.v1",
        "sha256": EXPECTED_M3_REPORT_SHA256,
        "oof_sha256": EXPECTED_M3_OOF_SHA256,
    }:
        raise ValueError("M3 report binding changed")
    if result_freeze.get("confirmation") != {
        "status": "sealed_not_run",
        "confirmation_slice": "[200:320]",
        "future_reserve_slice": "[320:541]",
    }:
        raise ValueError("M3 protected-split status changed")
    selections = result_freeze.get("selected_candidates_by_outer_fold")
    if selections != {compressor: ["C0"] * 5 for compressor in COMPRESSORS}:
        raise ValueError("M3 outer-fold selections are not all frozen C0")

    confirmation = protocol_lock.get("confirmation")
    if not isinstance(confirmation, dict) or confirmation.get("status") != (
        "sealed_until_a_compressor_passes_every_m3_development_gate"
    ):
        raise ValueError("M3 protocol confirmation status changed")
    safety = e1_audit.get("safety")
    if not isinstance(safety, dict):
        raise ValueError("E1 safety audit is missing")
    expected_safety = {
        "confirmation_or_reserve_rows_materialized": False,
        "development_prompts": EXPECTED_PROMPTS,
        "overlap": 0,
        "quarantined_prompts": 46,
        "scanner_filter_applied_before_python_conversion": True,
    }
    for key, expected in expected_safety.items():
        if safety.get(key) != expected:
            raise ValueError(f"E1 safety field {key} changed")
    return {
        "m3_result_freeze_sha256": EXPECTED_M3_RESULT_FREEZE_SHA256,
        "m3_report_sha256": EXPECTED_M3_REPORT_SHA256,
        "m3_oof_sha256": EXPECTED_M3_OOF_SHA256,
        "m3_protocol_lock_sha256": EXPECTED_M3_PROTOCOL_SHA256,
        "e1_integrity_audit_sha256": EXPECTED_E1_AUDIT_SHA256,
        "confirmation_status": "sealed_not_run",
        "quarantine_overlap": 0,
    }


def load_source(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    missing = set(SOURCE_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"M3 OOF lacks H4 source columns: {sorted(missing)}")
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("H4 source reader loaded an unexpected column")
    frame = table.to_pandas()
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"H4 source row count {len(frame)} != {EXPECTED_ROWS}"
        )
    if frame[list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("H4 source contains duplicate keys")
    if frame["prompt_id"].nunique() != EXPECTED_PROMPTS:
        raise ValueError("H4 source prompt count changed")
    if set(frame["compressor"]) != set(COMPRESSORS):
        raise ValueError("H4 source compressor roster changed")
    if set(float(value) for value in frame["ratio"].unique()) != set(RATIOS):
        raise ValueError("H4 source ratio roster changed")
    if set(frame["selected_candidate"]) != {"C0"}:
        raise ValueError("H4 source contains a non-C0 outer selection")
    if not np.isfinite(
        frame["m2_prediction"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("H4 source C0 predictions are nonfinite")

    compressor_counts = frame["compressor"].value_counts().to_dict()
    if compressor_counts != {
        compressor: EXPECTED_ROWS_PER_COMPRESSOR for compressor in COMPRESSORS
    }:
        raise ValueError(
            f"H4 compressor row counts changed: {compressor_counts}"
        )
    ratio_counts = frame.groupby(["compressor", "ratio"], sort=True).size()
    if set(int(value) for value in ratio_counts) != {
        EXPECTED_ROWS_PER_COMPRESSOR_RATIO
    }:
        raise ValueError("H4 compressor-ratio row counts changed")
    return frame


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        f"p{int(probability * 100):02d}": float(
            np.quantile(values, probability)
        )
        for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    }


def trajectory_audit(frame: pd.DataFrame) -> dict[str, Any]:
    ordered = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    group_columns = ["compressor", "prompt_id", "ratio"]
    groups = ordered.groupby(group_columns, sort=True)
    if groups.ngroups != EXPECTED_TRAJECTORIES:
        raise ValueError(
            f"H4 trajectory count {groups.ngroups} != {EXPECTED_TRAJECTORIES}"
        )

    lengths: list[int] = []
    prompt_rosters: dict[str, tuple[int, ...]] = {}
    adjacent_by_compressor: dict[str, list[float]] = {
        compressor: [] for compressor in COMPRESSORS
    }
    for (compressor, prompt_id, _ratio), group in groups:
        positions = group["s"].to_numpy(dtype=np.int64)
        if positions[0] != 0:
            raise ValueError(
                f"trajectory {compressor}/{prompt_id} does not start at 0"
            )
        if len(positions) > 1 and not np.array_equal(
            np.diff(positions),
            np.full(len(positions) - 1, EXPECTED_STRIDE, dtype=np.int64),
        ):
            raise ValueError(
                f"trajectory {compressor}/{prompt_id} has a boundary gap"
            )
        roster = tuple(int(position) for position in positions)
        prior_roster = prompt_rosters.setdefault(str(prompt_id), roster)
        if roster != prior_roster:
            raise ValueError(
                f"trajectory roster differs within prompt {prompt_id}"
            )
        lengths.append(len(group))
        predictions = group["m2_prediction"].to_numpy(dtype=np.float64)
        if len(predictions) > 1:
            adjacent_by_compressor[str(compressor)].extend(
                np.diff(predictions).tolist()
            )

    prompt_fold = ordered[["prompt_id", "fold"]].drop_duplicates()
    if len(prompt_fold) != EXPECTED_PROMPTS:
        raise ValueError("prompt folds are not constant within prompt")
    fold_counts = {
        str(int(fold)): int(count)
        for fold, count in prompt_fold["fold"]
        .value_counts()
        .sort_index()
        .items()
    }
    if list(fold_counts.values()) != [31, 31, 31, 31, 30]:
        raise ValueError(f"prompt fold counts changed: {fold_counts}")

    length_array = np.asarray(lengths, dtype=np.int64)
    roughness: dict[str, Any] = {}
    for compressor, values in adjacent_by_compressor.items():
        delta = np.asarray(values, dtype=np.float64)
        roughness[compressor] = {
            "adjacent_cells": int(len(delta)),
            "exact_same_fraction": float(np.mean(delta == 0.0)),
            "mean_absolute_difference": float(np.mean(np.abs(delta))),
            "root_mean_square_difference": float(np.sqrt(np.mean(delta**2))),
            "maximum_absolute_difference": float(np.max(np.abs(delta))),
            "absolute_difference_quantiles": quantiles(np.abs(delta)),
        }
    return {
        "rows": len(ordered),
        "unique_keys": len(ordered),
        "prompts": ordered["prompt_id"].nunique(),
        "prompt_ids_sha256": hash_prompt_ids(
            sorted(str(value) for value in ordered["prompt_id"].unique())
        ),
        "prompt_fold_counts": fold_counts,
        "compressors": list(COMPRESSORS),
        "ratios": list(RATIOS),
        "trajectories": groups.ngroups,
        "trajectories_per_prompt": EXPECTED_TRAJECTORIES // EXPECTED_PROMPTS,
        "adjacent_cells": int(
            sum(len(values) for values in adjacent_by_compressor.values())
        ),
        "stride": EXPECTED_STRIDE,
        "minimum_s": int(ordered["s"].min()),
        "maximum_s": int(ordered["s"].max()),
        "trajectory_length_quantiles": quantiles(length_array),
        "single_boundary_trajectories": int(np.sum(length_array == 1)),
        "all_prompt_trajectory_rosters_exact": len(prompt_rosters)
        == EXPECTED_PROMPTS,
        "row_pool_longest_to_shortest_prompt_weight_ratio": float(
            length_array.max() / length_array.min()
        ),
        "required_loss_weighting": (
            "mean_s_then_equal_ratio_then_equal_prompt"
        ),
        "c0_adjacent_prediction_roughness": roughness,
    }


def runtime_audit(contract_path: Path, manifest_path: Path) -> dict[str, Any]:
    contract = contract_path.read_text()
    required_fragments = (
        "the predictor emits",
        "at the decision boundary",
        "constant amount of causal aggregation per token",
    )
    missing = [
        fragment
        for fragment in required_fragments
        if fragment not in contract
    ]
    if missing:
        raise ValueError(
            f"predictor contract lost H4 runtime evidence: {missing}"
        )
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != "herald.magnitude_bundle.v1":
        raise ValueError("C0 native model manifest schema changed")
    if manifest.get("compressors") != list(COMPRESSORS):
        raise ValueError("C0 native model compressor roster changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("C0 native model manifest artifacts are missing")
    paths = {
        str(artifact.get("path"))
        for artifact in artifacts
        if isinstance(artifact, dict)
    }
    expected_fits = {f"{compressor}/fit.joblib" for compressor in COMPRESSORS}
    if not expected_fits <= paths:
        raise ValueError("C0 native fit artifacts are incomplete")
    return {
        "status": "admissible_in_principle_not_live_promoted",
        "evidence": [
            (
                "The current protocol defines one pre-switch prediction for "
                "each known compressor, ratio, and sampled decision boundary."
            ),
            "Native per-compressor C0 fit artifacts exist.",
            (
                "Retaining one previous score for each of four ratios "
                "requires four scalar values per compressor and no future "
                "state."
            ),
        ],
        "predictor_contract_sha256": sha256_file(contract_path),
        "model_manifest_sha256": sha256_file(manifest_path),
        "state_requirement": (
            "four_prior_same_ratio_scalar_scores_per_compressor"
        ),
        "caveat": (
            "M3 was not promoted, so this establishes causal deployability "
            "of the state dependency rather than a currently running "
            "controller API."
        ),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H4 audit: {args.output}"
        )
    provenance = validate_provenance(args)
    frame = load_source(args.source_oof)
    geometry = trajectory_audit(frame)
    runtime = runtime_audit(args.predictor_contract, args.model_manifest)
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_h4_understanding_before_strategy",
        "pass": True,
        "provenance": provenance,
        "source_access": {
            "columns_loaded": list(SOURCE_COLUMNS),
            "outcome_or_label_columns_loaded": [],
            "protected_rows_materialized": False,
        },
        "runtime_precondition": runtime,
        "trajectory_geometry": geometry,
        "interpretation_boundary": {
            "allowed": (
                "Label-blind H4 feasibility and causal-trajectory planning "
                "on frozen development C0 predictions."
            ),
            "forbidden": [
                "new outcome scoring before a separate protocol lock",
                "future or centered smoothing",
                "H1 or H2 variants",
                "M3 reinterpretation",
                "confirmation or reserve access",
            ],
        },
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
