"""Prove the development predictor roster is causal and oracle-free."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from materialize_current_state_damage_dev import (
    HALF_LIVES,
    HISTORY_BASES,
    HORIZONS,
    IDENTITY_COLUMNS,
    INSTANTANEOUS_SENSORS,
    TARGET_COLUMNS,
    WINDOWS,
)

SCHEMA_VERSION = "herald.current_state_damage_feature_audit.v1"
METADATA_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "fold",
)
ACTION_COLUMNS = ("press", "compression_ratio")
POSITION_COLUMNS = ("log_token_clock",)
BAND_COLUMNS = (
    "band_rate_0_5",
    "band_rate_5_10",
    "band_rate_10_25",
    "band_rate_25_50",
)


def history_columns() -> tuple[str, ...]:
    names: list[str] = []
    for base in HISTORY_BASES:
        for window in WINDOWS:
            names.extend(
                (
                    f"{base}_causal_mean_{window}",
                    f"{base}_causal_std_{window}",
                )
            )
        for half_life in HALF_LIVES:
            names.append(f"{base}_causal_ewma_hl{half_life}")
    return tuple(names)


HISTORY_COLUMNS = history_columns()
PREDICTOR_COLUMNS = (
    *ACTION_COLUMNS,
    *POSITION_COLUMNS,
    *INSTANTANEOUS_SENSORS,
    *HISTORY_COLUMNS,
)
LABEL_COLUMNS = (*TARGET_COLUMNS, *BAND_COLUMNS)
EXPECTED_COLUMNS = (
    *IDENTITY_COLUMNS,
    *INSTANTANEOUS_SENSORS,
    *TARGET_COLUMNS,
    "fold",
    "log_token_clock",
    *HISTORY_COLUMNS,
    *BAND_COLUMNS,
)
FORBIDDEN_EXACT = {
    "js_full",
    "kl_unc_comp_full",
    "sum_js",
    "sum_kl",
    "nll_ratio",
    "nll_ratio_flipped",
    "first_divergence_point",
    "gross_harm_final",
    "gross_help_final",
    "quality_delta",
    "compressed_quality_score",
    "baseline_quality_score",
    "rouge_l_drop",
    "char_edit_ratio",
    "length_diff_ratio",
    "embedding_cosine_drop",
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
    "relative_progress",
    "output_length_so_far",
}
FORBIDDEN_PREFIXES = ("future_",)
RELEASED_AGGREGATE_MARKERS = ("_mean_", "_std_", "_ewma_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def assert_close(actual: pd.Series, expected: pd.Series, name: str) -> float:
    actual_values = actual.to_numpy(dtype=np.float64)
    expected_values = expected.to_numpy(dtype=np.float64)
    if not np.allclose(
        actual_values,
        expected_values,
        rtol=2e-6,
        atol=2e-6,
        equal_nan=True,
    ):
        raise ValueError(f"causal recomputation mismatch for {name}")
    errors = np.abs(actual_values - expected_values)
    finite = errors[np.isfinite(errors)]
    return float(finite.max(initial=0.0))


def audit_sample(source: Path, materialized: Path) -> dict[str, Any]:
    materialized_dataset = ds.dataset(  # type: ignore[no-untyped-call]
        materialized, format="parquet"
    )
    run_table = materialized_dataset.to_table(columns=["run_id"])
    run_id = min(run_table["run_id"].to_pylist())
    run_filter = ds.field("run_id") == run_id  # type: ignore[attr-defined,no-untyped-call]
    output = materialized_dataset.to_table(filter=run_filter).to_pandas()
    source_dataset = ds.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
    raw = source_dataset.to_table(
        columns=["run_id", "token_pos", *INSTANTANEOUS_SENSORS],
        filter=run_filter,
    ).to_pandas()
    output.sort_values("token_pos", inplace=True, ignore_index=True)
    raw.sort_values("token_pos", inplace=True, ignore_index=True)
    if not np.array_equal(output["token_pos"], raw["token_pos"]):
        raise ValueError(f"sample token identity mismatch in {source.name}")
    maximum_error: dict[str, float] = {}
    expected_clock = pd.Series(
        np.log1p(raw["token_pos"].to_numpy(dtype=np.float64) + 1.0)
    )
    maximum_error["log_token_clock"] = assert_close(
        output["log_token_clock"], expected_clock, "log_token_clock"
    )
    for sensor in INSTANTANEOUS_SENSORS:
        maximum_error[sensor] = assert_close(
            output[sensor], raw[sensor], sensor
        )
    for base in HISTORY_BASES:
        series = raw[base]
        for window in WINDOWS:
            mean_name = f"{base}_causal_mean_{window}"
            std_name = f"{base}_causal_std_{window}"
            maximum_error[mean_name] = assert_close(
                output[mean_name],
                series.rolling(window, min_periods=1).mean(),
                mean_name,
            )
            maximum_error[std_name] = assert_close(
                output[std_name],
                series.rolling(window, min_periods=1).std(ddof=1),
                std_name,
            )
        for half_life in HALF_LIVES:
            name = f"{base}_causal_ewma_hl{half_life}"
            expected = series.ewm(
                halflife=half_life,
                adjust=False,
                ignore_na=True,
            ).mean()
            maximum_error[name] = assert_close(output[name], expected, name)
    return {
        "run_id": run_id,
        "rows": len(output),
        "maximum_absolute_error": maximum_error,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = load_json(args.protocol_lock)
    manifest_path = args.materialized_root / "manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("protocol_lock_sha256") != sha256_file(
        args.protocol_lock
    ):
        raise ValueError("materialization protocol mismatch")
    if manifest.get("confirmation_prompts_projected") != 0:
        raise ValueError("confirmation quarantine violated")
    if manifest.get("released_aggregate_columns_loaded") != []:
        raise ValueError("released aggregate projection detected")
    if manifest.get("oracle_columns_loaded") != []:
        raise ValueError("oracle projection detected")
    if tuple(lock["dataset"]["horizons"]) != HORIZONS:
        raise ValueError("protocol horizons changed")

    predictor_set = set(PREDICTOR_COLUMNS)
    forbidden = predictor_set & FORBIDDEN_EXACT
    forbidden.update(
        column
        for column in predictor_set
        if any(column.startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    )
    if forbidden:
        raise ValueError(f"forbidden predictor columns: {sorted(forbidden)}")
    if predictor_set & set(LABEL_COLUMNS):
        raise ValueError("label included as predictor")
    if "task" in predictor_set or "token_pos" in predictor_set:
        raise ValueError("metadata included as predictor")
    released_aggregates = {
        column
        for column in PREDICTOR_COLUMNS
        if any(marker in column for marker in RELEASED_AGGREGATE_MARKERS)
        and "_causal_" not in column
    }
    if released_aggregates:
        raise ValueError("released aggregate included as predictor")

    samples: dict[str, Any] = {}
    for path in sorted(args.materialized_root.glob("*.parquet")):
        columns = tuple(
            pq.ParquetFile(  # type: ignore[no-untyped-call]
                path
            ).schema_arrow.names
        )
        if columns != EXPECTED_COLUMNS:
            raise ValueError(f"unexpected materialized schema in {path.name}")
        source = args.dataset_root / "tokens" / path.name
        samples[path.stem] = audit_sample(source, path)
        if sha256_file(path) != manifest["files"][path.stem]["sha256"]:
            raise ValueError(f"materialized hash mismatch in {path.name}")
    if set(samples) != set(lock["dataset"]["presses"]):
        raise ValueError("materialized press roster changed")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "causal_predictor_roster_verified",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "materialization_manifest_sha256": sha256_file(manifest_path),
        "predictor_columns": list(PREDICTOR_COLUMNS),
        "metadata_columns": list(METADATA_COLUMNS),
        "label_columns": list(LABEL_COLUMNS),
        "forbidden_predictor_intersection": [],
        "released_aggregate_predictor_intersection": [],
        "task_is_predictor": False,
        "confirmation_prompts_projected": 0,
        "independent_causal_recomputation": samples,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
