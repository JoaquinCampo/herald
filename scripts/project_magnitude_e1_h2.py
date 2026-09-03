"""Create a label-blind monotone projection of frozen M3 C0 OOF curves."""

from __future__ import annotations

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
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
ROWS_PER_COMPRESSOR = 15060
TOTAL_ROWS = 45180
PROMPTS = 154
SCHEMA_VERSION = "herald.magnitude_e1_h2_projection.v1"
MANIFEST_SCHEMA_VERSION = "herald.magnitude_e1_h2_projection_manifest.v1"
LOCK_SCHEMA_VERSION = "herald.magnitude_e1_h2_protocol_lock.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
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


def pava(values: np.ndarray) -> np.ndarray:
    """Return the unweighted nondecreasing L2 projection in float64."""
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ValueError("PAVA input must be a finite nonempty vector")
    means: list[float] = []
    weights: list[int] = []
    for value in source:
        means.append(float(value))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (
                means[-2] * weights[-2] + means[-1] * weights[-1]
            ) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    return np.repeat(np.asarray(means, dtype=np.float64), weights)


def self_test() -> None:
    cases = (
        (np.asarray([0.0, 1.0, 2.0, 3.0]), np.asarray([0.0, 1.0, 2.0, 3.0])),
        (np.asarray([3.0, 2.0, 1.0, 0.0]), np.asarray([1.5, 1.5, 1.5, 1.5])),
        (np.asarray([0.0, 2.0, 1.0, 3.0]), np.asarray([0.0, 1.5, 1.5, 3.0])),
        (np.asarray([2.0, 0.0, 1.0, 4.0]), np.asarray([1.0, 1.0, 1.0, 4.0])),
    )
    for source, expected in cases:
        projected = pava(source)
        if not np.array_equal(projected, expected):
            raise AssertionError(
                f"PAVA self-test failed: {source} -> {projected}"
            )
        if not np.array_equal(pava(projected), projected):
            raise AssertionError("PAVA is not idempotent")
        if not np.isclose(
            source.mean(), projected.mean(), rtol=0.0, atol=1e-15
        ):
            raise AssertionError("PAVA did not preserve the vector mean")


def validate_lock(
    lock_path: Path,
    strategy_path: Path,
    source_path: Path,
    score_script: Path,
) -> dict[str, Any]:
    lock = load_json(lock_path)
    strategy = load_json(strategy_path)
    if (
        lock.get("schema_version") != LOCK_SCHEMA_VERSION
        or lock.get("status") != "locked_before_label_blind_projection"
    ):
        raise ValueError("H2 protocol lock schema/status is invalid")
    if strategy.get("selected_hypothesis") != "H2":
        raise ValueError("strategy did not select H2")
    if lock.get("strategy_sha256") != sha256_file(strategy_path):
        raise ValueError("strategy hash differs from H2 lock")
    if lock.get("source_oof_sha256") != sha256_file(source_path):
        raise ValueError("source OOF hash differs from H2 lock")
    expected_projection = {
        "allowed_source_columns": list(SOURCE_COLUMNS),
        "group_key": ["compressor", "prompt_id", "s"],
        "ratio_order": list(RATIOS),
        "algorithm": "unweighted_float64_l2_pava_nondecreasing",
        "labels_loaded": False,
    }
    if lock.get("projection") != expected_projection:
        raise ValueError("projection contract differs from H2 lock")
    implementation = lock.get("implementation_sha256")
    expected_implementation = {
        "scripts/project_magnitude_e1_h2.py": sha256_file(Path(__file__)),
        "scripts/score_magnitude_e1_h2.py": sha256_file(score_script),
    }
    if implementation != expected_implementation:
        raise ValueError("H2 implementation differs from protocol lock")
    if lock.get("confirmation_status") != "sealed_not_run":
        raise ValueError("confirmation is not sealed")
    return lock


def input_frame(path: Path) -> pd.DataFrame:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    missing = set(SOURCE_COLUMNS) - set(schema.names)
    if missing:
        raise ValueError(
            f"source OOF lacks projection columns: {sorted(missing)}"
        )
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("source reader loaded an unexpected column")
    frame = table.to_pandas()
    if len(frame) != TOTAL_ROWS:
        raise ValueError(f"source OOF row count {len(frame)} != {TOTAL_ROWS}")
    if frame[list(KEY_COLUMNS)].duplicated().any():
        raise ValueError("source OOF contains duplicate projection keys")
    if set(frame["compressor"]) != set(COMPRESSORS):
        raise ValueError("source OOF compressor set differs from contract")
    if set(float(value) for value in frame["ratio"].unique()) != set(RATIOS):
        raise ValueError("source OOF ratio set differs from contract")
    if frame["prompt_id"].nunique() != PROMPTS:
        raise ValueError("source OOF prompt count differs from contract")
    if not np.isfinite(
        frame["m2_prediction"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("source C0 predictions are nonfinite")
    counts = frame["compressor"].value_counts().to_dict()
    if counts != {
        compressor: ROWS_PER_COMPRESSOR for compressor in COMPRESSORS
    }:
        raise ValueError(f"source compressor row counts differ: {counts}")
    return frame


def pooled_block_means_are_exact(
    source: np.ndarray,
    projected: np.ndarray,
    *,
    tolerance: float = 1e-14,
) -> bool:
    start = 0
    while start < len(projected):
        stop = start + 1
        while stop < len(projected) and projected[stop] == projected[start]:
            stop += 1
        if not np.isclose(
            projected[start],
            source[start:stop].mean(),
            rtol=0.0,
            atol=tolerance,
        ):
            return False
        start = stop
    return True


def project(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    changed_by_compressor = {compressor: 0 for compressor in COMPRESSORS}
    input_violations = {compressor: 0 for compressor in COMPRESSORS}
    group_count = 0
    mean_errors = 0
    monotonic_errors = 0
    idempotence_errors = 0
    unchanged_errors = 0
    pooled_mean_errors = 0
    for (compressor, prompt_id, s), group in frame.groupby(
        ["compressor", "prompt_id", "s"],
        sort=True,
    ):
        ordered = group.sort_values("ratio")
        ratios = tuple(float(value) for value in ordered["ratio"])
        if ratios != RATIOS or len(ordered) != len(RATIOS):
            raise ValueError(
                f"incomplete ratio curve for {compressor}/{prompt_id}/{s}"
            )
        source = ordered["m2_prediction"].to_numpy(dtype=np.float64)
        was_monotone = bool(np.all(np.diff(source) >= -1e-15))
        projected = pava(source)
        group_count += 1
        if not was_monotone:
            input_violations[str(compressor)] += 1
        if not np.array_equal(source, projected):
            changed_by_compressor[str(compressor)] += 1
        if not np.all(np.diff(projected) >= -1e-15):
            monotonic_errors += 1
        if not np.array_equal(pava(projected), projected):
            idempotence_errors += 1
        if was_monotone and not np.array_equal(source, projected):
            unchanged_errors += 1
        if not np.isclose(
            source.mean(), projected.mean(), rtol=0.0, atol=1e-14
        ):
            mean_errors += 1
        if not pooled_block_means_are_exact(source, projected):
            pooled_mean_errors += 1
        for source_row, value in zip(
            ordered.itertuples(index=False), projected, strict=True
        ):
            rows.append(
                {
                    "prompt_id": str(source_row.prompt_id),
                    "compressor": str(source_row.compressor),
                    "ratio": float(source_row.ratio),
                    "s": int(source_row.s),
                    "c0_prediction": float(source_row.m2_prediction),
                    "pava_prediction": float(value),
                }
            )
    invariants = {
        "groups": group_count,
        "input_nonmonotone_groups": input_violations,
        "changed_groups": changed_by_compressor,
        "projected_monotonicity_errors": monotonic_errors,
        "idempotence_errors": idempotence_errors,
        "already_monotone_changed_errors": unchanged_errors,
        "vector_mean_errors": mean_errors,
        "pooled_block_mean_errors": pooled_mean_errors,
    }
    if any(
        invariants[name]
        for name in (
            "projected_monotonicity_errors",
            "idempotence_errors",
            "already_monotone_changed_errors",
            "vector_mean_errors",
            "pooled_block_mean_errors",
        )
    ):
        raise ValueError(f"PAVA invariant failure: {invariants}")
    result = (
        pd.DataFrame(rows)
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    if (
        len(result) != len(frame)
        or result[list(KEY_COLUMNS)].duplicated().any()
    ):
        raise ValueError("projected keyed table is incomplete or duplicated")
    return result, invariants


def ensure_new(*paths: Path) -> None:
    collisions = [str(path) for path in paths if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite H2 artifacts: {collisions}"
        )


def main() -> None:
    args = parse_args()
    ensure_new(args.output, args.manifest)
    self_test()
    lock = validate_lock(
        args.protocol_lock,
        args.strategy,
        args.source_oof,
        args.score_script,
    )
    frame = input_frame(args.source_oof)
    projected, invariants = project(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(projected, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update(
        {
            b"herald.schema_version": SCHEMA_VERSION.encode(),
            b"herald.source_oof_sha256": sha256_file(
                args.source_oof
            ).encode(),
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
    output_sha = sha256_file(args.output)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "label_blind_projection_frozen_before_scoring",
        "source_oof_sha256": sha256_file(args.source_oof),
        "strategy_sha256": sha256_file(args.strategy),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "projection_sha256": output_sha,
        "implementation_sha256": sha256_file(Path(__file__)),
        "allowed_source_columns": list(SOURCE_COLUMNS),
        "outcome_or_label_columns_loaded": [],
        "development_rows": len(projected),
        "development_prompts": int(projected["prompt_id"].nunique()),
        "development_prompt_ids_sha256": hash_prompt_ids(
            tuple(str(value) for value in projected["prompt_id"].unique())
        ),
        "compressor_counts": {
            str(key): int(value)
            for key, value in projected["compressor"].value_counts().items()
        },
        "ratios": list(RATIOS),
        "algorithm": lock["projection"]["algorithm"],
        "invariants": invariants,
        "confirmation_status": "sealed_not_run",
    }
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "label_blind_projection_frozen",
                "manifest": str(args.manifest),
                "manifest_sha256": sha256_file(args.manifest),
                "projection": str(args.output),
                "projection_sha256": output_sha,
                "invariants": invariants,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
