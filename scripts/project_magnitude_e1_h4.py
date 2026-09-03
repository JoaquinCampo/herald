"""Build the locked label-blind H4 finite causal-average transform."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from herald.magnitude_v2 import hash_prompt_ids, sha256_file

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
SOURCE_COLUMNS = (*KEY_COLUMNS, "m2_prediction")
OUTPUT_COLUMNS = (*KEY_COLUMNS, "c0_prediction", "causal_avg_prediction")
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
EXPECTED_ROWS = 45_180
EXPECTED_ROWS_PER_COMPRESSOR = 15_060
EXPECTED_PROMPTS = 154
EXPECTED_TRAJECTORIES = 1_848
EXPECTED_STRIDE = 16
CURRENT_WEIGHT = 0.5
PREVIOUS_WEIGHT = 0.5
MINIMUM_WORTHWHILE_SKILL = 0.01
FROZEN_C0_LOSS = {
    "expected_attention": 0.06047634774320178,
    "knorm": 0.14725539737292004,
    "streaming_llm": 0.1337407724970253,
}
EXPECTED_SOURCE_SHA256 = (
    "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
)
EXPECTED_H2_REPORT_SHA256 = (
    "e72f5e71781e6e3dc824aad21d1f2400cce9fffb6a0444380b06c44a8e0affe1"
)
SCHEMA_VERSION = "herald.magnitude_e1_h4_transform.v1"
MANIFEST_SCHEMA_VERSION = "herald.magnitude_e1_h4_transform_manifest.v1"
LOCK_SCHEMA_VERSION = "herald.magnitude_e1_h4_protocol_lock.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--h2-report", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--score-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def causal_average(values: np.ndarray) -> np.ndarray:
    """Apply the finite two-point average without recursive feedback."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ValueError(
            "causal-average input must be a finite nonempty vector"
        )
    transformed = source.copy()
    transformed[1:] = (
        CURRENT_WEIGHT * source[1:] + PREVIOUS_WEIGHT * source[:-1]
    )
    return transformed


def self_test() -> None:
    cases = (
        (np.asarray([2.0]), np.asarray([2.0])),
        (np.asarray([1.0, 3.0, 5.0]), np.asarray([1.0, 2.0, 4.0])),
        (np.asarray([0.0, 2.0, 4.0]), np.asarray([0.0, 1.0, 3.0])),
        (np.asarray([-1.0, 1.0, -3.0]), np.asarray([-1.0, 0.0, -1.0])),
    )
    for source, expected in cases:
        transformed = causal_average(source)
        if not np.array_equal(transformed, expected):
            raise AssertionError(
                f"causal-average self-test failed: {source} -> {transformed}"
            )
        if transformed[0] != source[0]:
            raise AssertionError("causal average changed the first boundary")


def expected_transform_contract() -> dict[str, Any]:
    return {
        "allowed_source_columns": list(SOURCE_COLUMNS),
        "group_key": ["compressor", "prompt_id", "ratio"],
        "order_key": "s",
        "required_stride": EXPECTED_STRIDE,
        "algorithm": "finite_two_point_causal_average_float64",
        "current_weight": CURRENT_WEIGHT,
        "previous_weight": PREVIOUS_WEIGHT,
        "recursive": False,
        "first_boundary": "identity",
        "single_boundary_trajectory": "identity",
        "labels_loaded": False,
        "loss_weighting": "mean_s_then_equal_ratio_then_equal_prompt",
        "frozen_c0_loss": FROZEN_C0_LOSS,
        "skill_upper_bound": "2*sqrt(D/L)-D/L",
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
    }


def validate_h2_report(path: Path) -> None:
    if sha256_file(path) != EXPECTED_H2_REPORT_SHA256:
        raise ValueError("published H2 report hash changed")
    report = load_json(path)
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("published H2 metrics are missing")
    observed = {
        compressor: float(metrics[compressor]["c0_loss"])
        for compressor in COMPRESSORS
    }
    if observed != FROZEN_C0_LOSS:
        raise ValueError(f"published C0 losses changed: {observed}")
    if report.get("decision", {}).get("status") != "failure":
        raise ValueError("H2 result is not the frozen failure")


def validate_lock(
    lock_path: Path,
    strategy_path: Path,
    source_path: Path,
    h2_report_path: Path,
    score_script: Path,
) -> dict[str, Any]:
    lock = load_json(lock_path)
    strategy = load_json(strategy_path)
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("status") != "locked_before_label_blind_h4_transform"
    ):
        raise ValueError("H4 protocol lock schema/status is invalid")
    if strategy.get("selected_hypothesis") != "H4":
        raise ValueError("strategy did not select H4")
    expected_hashes = {
        "strategy_sha256": sha256_file(strategy_path),
        "source_oof_sha256": sha256_file(source_path),
        "h2_report_sha256": sha256_file(h2_report_path),
    }
    for name, expected in expected_hashes.items():
        if lock.get(name) != expected:
            raise ValueError(f"H4 lock {name} changed")
    if lock.get("source_oof_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("H4 source is not immutable M3 OOF")
    if lock.get("transform") != expected_transform_contract():
        raise ValueError("H4 transform contract differs from lock")
    expected_implementation = {
        "scripts/project_magnitude_e1_h4.py": sha256_file(Path(__file__)),
        "scripts/score_magnitude_e1_h4.py": sha256_file(score_script),
    }
    if lock.get("implementation_sha256") != expected_implementation:
        raise ValueError("H4 implementation differs from protocol lock")
    if lock.get("confirmation_status") != "sealed_not_run":
        raise ValueError("confirmation is not sealed")
    return lock


def input_frame(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    missing = set(SOURCE_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(f"source OOF lacks H4 columns: {sorted(missing)}")
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("H4 reader loaded an unexpected source column")
    frame = table.to_pandas()
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"source OOF row count {len(frame)} != {EXPECTED_ROWS}"
        )
    if frame[list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("source OOF contains duplicate H4 keys")
    if frame["prompt_id"].nunique() != EXPECTED_PROMPTS:
        raise ValueError("source OOF prompt count differs from H4 contract")
    if set(frame["compressor"]) != set(COMPRESSORS):
        raise ValueError("source OOF compressor set differs from H4 contract")
    if set(float(value) for value in frame["ratio"].unique()) != set(RATIOS):
        raise ValueError("source OOF ratio set differs from H4 contract")
    if not np.isfinite(
        frame["m2_prediction"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("source C0 predictions are nonfinite")
    counts = frame["compressor"].value_counts().to_dict()
    if counts != {
        compressor: EXPECTED_ROWS_PER_COMPRESSOR for compressor in COMPRESSORS
    }:
        raise ValueError(f"source compressor row counts differ: {counts}")
    return frame


def transform(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = frame.groupby(
        ["compressor", "prompt_id", "ratio"],
        sort=True,
    )
    if groups.ngroups != EXPECTED_TRAJECTORIES:
        raise ValueError("H4 source trajectory count differs from contract")
    first_boundary_errors = 0
    recurrence_errors = 0
    gap_errors = 0
    changed_rows = {compressor: 0 for compressor in COMPRESSORS}
    single_boundary_trajectories = 0
    for (compressor, prompt_id, ratio), group in groups:
        ordered = group.sort_values("s")
        positions = ordered["s"].to_numpy(dtype=np.int64)
        if positions[0] != 0 or (
            len(positions) > 1
            and not np.array_equal(
                np.diff(positions),
                np.full(len(positions) - 1, EXPECTED_STRIDE, dtype=np.int64),
            )
        ):
            gap_errors += 1
            continue
        source = ordered["m2_prediction"].to_numpy(dtype=np.float64)
        transformed = causal_average(source)
        if len(source) == 1:
            single_boundary_trajectories += 1
        if transformed[0] != source[0]:
            first_boundary_errors += 1
        if len(source) > 1 and not np.array_equal(
            transformed[1:],
            CURRENT_WEIGHT * source[1:] + PREVIOUS_WEIGHT * source[:-1],
        ):
            recurrence_errors += 1
        changed_rows[str(compressor)] += int(np.sum(transformed != source))
        for source_row, value in zip(
            ordered.itertuples(index=False),
            transformed,
            strict=True,
        ):
            rows.append(
                {
                    "prompt_id": str(prompt_id),
                    "compressor": str(compressor),
                    "ratio": float(ratio),
                    "s": int(source_row.s),
                    "c0_prediction": float(source_row.m2_prediction),
                    "causal_avg_prediction": float(value),
                }
            )
    invariants = {
        "trajectories": groups.ngroups,
        "single_boundary_trajectories": single_boundary_trajectories,
        "changed_rows": changed_rows,
        "first_boundary_errors": first_boundary_errors,
        "recurrence_errors": recurrence_errors,
        "gap_errors": gap_errors,
    }
    if any(
        invariants[name]
        for name in (
            "first_boundary_errors",
            "recurrence_errors",
            "gap_errors",
        )
    ):
        raise ValueError(f"H4 transform invariant failure: {invariants}")
    result = (
        pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    if (
        len(result) != len(frame)
        or result[list(KEY_COLUMNS)].duplicated().any()
        or tuple(result.columns) != OUTPUT_COLUMNS
    ):
        raise ValueError("H4 transformed keyed table is incomplete")
    return result, invariants


def feasibility(transformed: pd.DataFrame) -> dict[str, Any]:
    work = transformed.assign(
        _energy=(
            transformed["causal_avg_prediction"].to_numpy(dtype=np.float64)
            - transformed["c0_prediction"].to_numpy(dtype=np.float64)
        )
        ** 2
    )
    by_compressor: dict[str, Any] = {}
    for compressor in COMPRESSORS:
        group = work[work["compressor"] == compressor]
        cells = group.groupby(
            ["prompt_id", "ratio"],
            observed=True,
            sort=True,
        )["_energy"].mean()
        table = cells.unstack("ratio").reindex(columns=list(RATIOS))
        values = table.to_numpy(dtype=np.float64)
        if (
            values.shape != (EXPECTED_PROMPTS, len(RATIOS))
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"H4 energy table is incomplete for {compressor}"
            )
        energy = float(np.mean(values))
        loss = FROZEN_C0_LOSS[compressor]
        relative_energy = energy / loss
        bound = 2.0 * np.sqrt(relative_energy) - relative_energy
        by_compressor[compressor] = {
            "perturbation_energy": energy,
            "frozen_c0_loss": loss,
            "relative_energy": relative_energy,
            "skill_upper_bound": float(bound),
            "upper_bound_exceeds_0_01": bool(
                bound > MINIMUM_WORTHWHILE_SKILL
            ),
            "per_ratio_perturbation_energy": {
                str(ratio): float(np.mean(values[:, index]))
                for index, ratio in enumerate(RATIOS)
            },
        }
    score_stage_b = any(
        value["upper_bound_exceeds_0_01"] for value in by_compressor.values()
    )
    return {
        "by_compressor": by_compressor,
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
        "score_stage_b": score_stage_b,
        "action": (
            "freeze_transform_then_create_separate_scoring_lock"
            if score_stage_b
            else "retire_H4_without_loading_row_outcomes"
        ),
    }


def ensure_new(*paths: Path) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite H4 artifacts: {collisions}"
        )


def main() -> None:
    args = parse_args()
    ensure_new(args.output, args.manifest)
    self_test()
    validate_h2_report(args.h2_report)
    validate_lock(
        args.protocol_lock,
        args.strategy,
        args.source_oof,
        args.h2_report,
        args.score_script,
    )
    frame = input_frame(args.source_oof)
    transformed, invariants = transform(frame)
    gate = feasibility(transformed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(transformed, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"herald.schema_version": SCHEMA_VERSION.encode(),
            b"herald.source_oof_sha256": EXPECTED_SOURCE_SHA256.encode(),
            b"herald.protocol_lock_sha256": sha256_file(
                args.protocol_lock
            ).encode(),
            b"herald.labels_loaded": b"false",
        }
    )
    pq.write_table(  # type: ignore[no-untyped-call]
        table.replace_schema_metadata(metadata),
        args.output,
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "label_blind_h4_transform_frozen_before_scoring",
        "source_oof_sha256": EXPECTED_SOURCE_SHA256,
        "h2_report_sha256": EXPECTED_H2_REPORT_SHA256,
        "strategy_sha256": sha256_file(args.strategy),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "transform_sha256": sha256_file(args.output),
        "implementation_sha256": sha256_file(Path(__file__)),
        "allowed_source_columns": list(SOURCE_COLUMNS),
        "outcome_or_label_columns_loaded": [],
        "confirmation_status": "sealed_not_run",
        "development_rows": len(transformed),
        "development_prompts": frame["prompt_id"].nunique(),
        "development_prompt_ids_sha256": hash_prompt_ids(
            sorted(str(value) for value in frame["prompt_id"].unique())
        ),
        "transform_contract": expected_transform_contract(),
        "invariants": invariants,
        "feasibility": gate,
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
