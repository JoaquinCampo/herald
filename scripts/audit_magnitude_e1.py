"""Dev-only integrity and target-geometry audit for post-M3 exploration."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from herald.magnitude import CAUSAL_FEATURE_COLUMNS, KNOWN_COMPRESSORS
from herald.magnitude_v2 import hash_prompt_ids, make_folds, sha256_file

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
RATIOS = (0.25, 0.5, 0.75, 0.875)
STRICT_COLUMNS = (
    "q_ref_strict",
    "q_control_strict",
    "q_hybrid_strict",
    "dq_strict",
    "damaged_strict",
    "major_damage_strict",
)
CORE_COLUMNS = (
    "model",
    "task",
    "prompt_id",
    "compressor",
    "ratio",
    "s",
    "relative_s",
    "ref_len",
    "q_ref",
    "q_control",
    "q_hybrid",
    "dq",
    "damaged",
    "major_damage",
    "feature_timing",
    *STRICT_COLUMNS,
    "intervention_semantics",
    "generation_provenance",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--v2-lock", type=Path, required=True)
    parser.add_argument("--m3-lock", type=Path, required=True)
    parser.add_argument("--m3-freeze", type=Path, required=True)
    parser.add_argument("--m3-report", type=Path, required=True)
    parser.add_argument("--m3-result-freeze", type=Path, required=True)
    parser.add_argument("--m3-oof", type=Path, required=True)
    parser.add_argument("--m2-oof", type=Path, required=True)
    parser.add_argument("--global-sidecar", type=Path, required=True)
    parser.add_argument("--band-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {}
    return {
        "min": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
    }


def sign_label(values: pd.Series, tolerance: float = 1e-12) -> pd.Series:
    array = values.to_numpy(dtype=np.float64)
    labels = np.full(len(array), "zero", dtype=object)
    labels[array > tolerance] = "positive"
    labels[array < -tolerance] = "negative"
    return pd.Series(labels, index=values.index, dtype="object")


def eta_squared(frame: pd.DataFrame, columns: list[str]) -> float:
    target = frame["dq"].to_numpy(dtype=np.float64)
    total = float(np.sum((target - np.mean(target)) ** 2))
    if total == 0.0:
        return 0.0
    means = frame.groupby(columns, observed=True)["dq"].transform("mean")
    between = float(
        np.sum((means.to_numpy(dtype=np.float64) - np.mean(target)) ** 2)
    )
    return between / total


def in_sample_group_mse(frame: pd.DataFrame, columns: list[str]) -> float:
    prediction = frame.groupby(columns, observed=True)["dq"].transform("mean")
    error = frame["dq"].to_numpy(dtype=np.float64) - prediction.to_numpy(
        dtype=np.float64
    )
    return float(np.mean(error**2))


def load_dev_frame(
    base_path: Path,
    development: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    schema = pq.read_schema(base_path)  # type: ignore[no-untyped-call]
    selected_features = sorted(CAUSAL_FEATURE_COLUMNS)
    columns = list(CORE_COLUMNS) + selected_features
    missing = sorted(set(columns) - set(schema.names))
    if missing:
        raise ValueError(f"base parquet lacks audit columns: {missing}")
    task_field = ds.field("task")  # type: ignore[attr-defined, no-untyped-call]
    prompt_field = ds.field(  # type: ignore[attr-defined, no-untyped-call]
        "prompt_id"
    )
    compressor_field = ds.field(  # type: ignore[attr-defined, no-untyped-call]
        "compressor"
    )
    predicate = (task_field == "ifeval") & prompt_field.isin(
        list(development)
    )
    predicate = predicate & compressor_field.isin(list(KNOWN_COMPRESSORS))
    table = ds.dataset(  # type: ignore[no-untyped-call]
        base_path, format="parquet"
    ).to_table(
        columns=columns,
        filter=predicate,
    )
    frame = table.to_pandas()
    metadata = {
        key.decode(errors="replace"): value.decode(errors="replace")
        for key, value in (schema.metadata or {}).items()
    }
    parquet_metadata = pq.ParquetFile(  # type: ignore[no-untyped-call]
        base_path
    ).metadata
    return frame, {
        "physical_rows": int(parquet_metadata.num_rows),
        "row_groups": int(parquet_metadata.num_row_groups),
        "columns": int(parquet_metadata.num_columns),
        "schema_metadata": metadata,
    }


def audit_sidecar(
    path: Path,
    development: set[str],
    quarantine: set[str],
) -> tuple[dict[str, Any], dict[tuple[str, str, float, int], str]]:
    keys: list[tuple[str, str, float, int]] = []
    prefixes: dict[tuple[str, str, float, int], str] = {}
    feature_names: set[tuple[str, ...]] = set()
    null_values = 0
    nonfinite_values = 0
    protected_rows = 0
    unknown_rows = 0
    protocol_versions: Counter[str] = Counter()
    state_semantics: Counter[str] = Counter()
    compressor_counts: Counter[str] = Counter()
    ratio_counts: Counter[str] = Counter()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            row = json.loads(line)
            prompt_id = str(row["prompt_id"])
            if prompt_id in quarantine:
                protected_rows += 1
            if prompt_id not in development:
                unknown_rows += 1
            key = (
                prompt_id,
                str(row["compressor"]),
                float(row["ratio"]),
                int(row["s"]),
            )
            keys.append(key)
            prefixes[key] = str(row["prefix_hash"])
            names = tuple(str(name) for name in row["feature_names"])
            feature_names.add(names)
            sensors = row.get("sensors")
            if not isinstance(sensors, dict) or set(sensors) != set(names):
                raise ValueError(
                    f"invalid sensor schema at {path}:{line_number}"
                )
            for value in sensors.values():
                if value is None:
                    null_values += 1
                elif not finite_number(value):
                    nonfinite_values += 1
            protocol_versions[str(row.get("protocol_version"))] += 1
            state_semantics[
                str(row.get("state_semantics", "not_recorded"))
            ] += 1
            compressor_counts[str(row["compressor"])] += 1
            ratio_counts[str(float(row["ratio"]))] += 1
    duplicate_count = len(keys) - len(set(keys))
    if len(prefixes) != len(set(keys)):
        raise ValueError(f"duplicate sidecar keys in {path}")
    return (
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(keys),
            "unique_keys": len(set(keys)),
            "duplicate_keys": duplicate_count,
            "protected_rows": protected_rows,
            "unknown_prompt_rows": unknown_rows,
            "feature_schema_count": len(feature_names),
            "feature_count": len(next(iter(feature_names)))
            if feature_names
            else 0,
            "null_sensor_values": null_values,
            "nonfinite_sensor_values": nonfinite_values,
            "protocol_versions": dict(sorted(protocol_versions.items())),
            "state_semantics": dict(sorted(state_semantics.items())),
            "compressor_counts": dict(sorted(compressor_counts.items())),
            "ratio_counts": dict(sorted(ratio_counts.items())),
            "keys": set(keys),
        },
        prefixes,
    )


def audit_oof(
    path: Path,
    m2_path: Path,
    base_keys: set[tuple[str, str, float, int]],
    development: tuple[str, ...],
    folds: tuple[Any, ...],
) -> dict[str, Any]:
    frame = pd.read_parquet(path)
    m2 = pd.read_parquet(m2_path)
    keys = [
        tuple(row)
        for row in frame[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    m2_keys = [
        tuple(row)
        for row in m2[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    diagnostics: list[dict[str, Any]] = []
    for (compressor, fold), group in frame.groupby(
        ["compressor", "fold"], sort=True
    ):
        unique: dict[str, Any] = {}
        for column in (
            "candidate_scores",
            "candidate_standard_errors",
            "candidate_lambdas",
            "c4_alpha",
            "c4_alpha_scores",
            "selected_candidate",
            "fold_train_prompt_ids_sha256",
            "fold_test_prompt_ids_sha256",
        ):
            values = {stable_json(value) for value in group[column]}
            if len(values) != 1:
                raise ValueError(
                    f"{column} is not fold-constant for {compressor}/{fold}"
                )
            unique[column] = group.iloc[0][column]
        scores = {
            str(key): float(value)
            for key, value in unique["candidate_scores"].items()
        }
        errors = {
            str(key): float(value)
            for key, value in unique["candidate_standard_errors"].items()
        }
        lambdas = {
            str(key): float(value)
            for key, value in unique["candidate_lambdas"].items()
        }
        best = min(
            scores, key=lambda candidate: (scores[candidate], candidate)
        )
        threshold = scores[best] + errors[best]
        diagnostics.append(
            {
                "compressor": str(compressor),
                "fold": int(fold),
                "test_prompts": int(group["prompt_id"].nunique()),
                "selected": str(unique["selected_candidate"]),
                "numerical_best": best,
                "scores": scores,
                "standard_errors": errors,
                "one_se_threshold": float(threshold),
                "c0_gap_from_best": float(scores["C0"] - scores[best]),
                "c0_within_one_se": bool(scores["C0"] <= threshold + 1e-15),
                "lambdas": lambdas,
                "c4_alpha": float(unique["c4_alpha"]),
                "c4_alpha_scores": {
                    str(key): float(value)
                    for key, value in unique["c4_alpha_scores"].items()
                },
                "train_hash": str(unique["fold_train_prompt_ids_sha256"]),
                "test_hash": str(unique["fold_test_prompt_ids_sha256"]),
            }
        )
    expected_fold_by_prompt = {
        prompt_id: fold.index
        for fold in folds
        for prompt_id in fold.prompt_ids
    }
    actual_fold_by_prompt = frame.groupby("prompt_id")["fold"].nunique()
    fold_assignment_errors = int((actual_fold_by_prompt != 1).sum())
    for prompt_id, group in frame.groupby("prompt_id"):
        if (
            int(group["fold"].iloc[0])
            != expected_fold_by_prompt[str(prompt_id)]
        ):
            fold_assignment_errors += 1
    merged = frame.merge(
        m2[[*KEY_COLUMNS, "m2_prediction"]],
        on=list(KEY_COLUMNS),
        how="outer",
        suffixes=("_m3", "_frozen"),
        indicator=True,
        validate="one_to_one",
    )
    parity = np.abs(
        merged["m2_prediction_m3"].to_numpy(dtype=np.float64)
        - merged["m2_prediction_frozen"].to_numpy(dtype=np.float64)
    )
    selected_parity = np.abs(
        frame["m3_prediction"].to_numpy(dtype=np.float64)
        - frame["m2_prediction"].to_numpy(dtype=np.float64)
    )
    numeric_columns = [
        "dq",
        "m3_prediction",
        "m2_prediction",
        "positive_risk",
        "positive_risk_baseline",
        "locked_baseline_mean",
        "locked_baseline_median",
    ]
    nonfinite = {
        column: int(
            (~np.isfinite(frame[column].to_numpy(dtype=np.float64))).sum()
        )
        for column in numeric_columns
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "unique_keys": len(set(keys)),
        "duplicate_keys": len(keys) - len(set(keys)),
        "base_missing_keys": len(base_keys - set(keys)),
        "base_extra_keys": len(set(keys) - base_keys),
        "m2_unique_keys": len(set(m2_keys)),
        "m2_merge_counts": {
            str(key): int(value)
            for key, value in merged["_merge"].value_counts().items()
        },
        "m2_prediction_max_abs_difference": float(np.nanmax(parity)),
        "selected_m3_m2_max_abs_difference": float(np.max(selected_parity)),
        "fold_assignment_errors": fold_assignment_errors,
        "fold_prompt_counts": {
            str(key): int(value)
            for key, value in frame.groupby("fold")["prompt_id"]
            .nunique()
            .items()
        },
        "prompt_ids_sha256": hash_prompt_ids(
            tuple(frame["prompt_id"].unique())
        ),
        "nonfinite_predictions": nonfinite,
        "selection_diagnostics": diagnostics,
    }


def label_summary(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for compressor, group in frame.groupby("compressor", sort=True):
        signs = sign_label(group["dq"])
        strict_signs = sign_label(group["dq_strict"])
        support = group["dq"].value_counts().sort_values(ascending=False)
        strict_support = (
            group["dq_strict"].value_counts().sort_values(ascending=False)
        )
        transitions = pd.crosstab(signs, strict_signs)
        by_ratio: dict[str, Any] = {}
        for ratio, ratio_group in group.groupby("ratio", sort=True):
            ratio_sign = sign_label(ratio_group["dq"])
            by_ratio[str(float(ratio))] = {
                "rows": len(ratio_group),
                "mean": float(ratio_group["dq"].mean()),
                "std": float(ratio_group["dq"].std(ddof=0)),
                "zero": int((ratio_sign == "zero").sum()),
                "positive": int((ratio_sign == "positive").sum()),
                "negative": int((ratio_sign == "negative").sum()),
                "major": int((ratio_group["dq"] >= 0.5 - 1e-12).sum()),
            }
        prompt_events = (
            group.assign(_sign=signs)
            .groupby("prompt_id", sort=True)
            .agg(
                rows=("dq", "size"),
                nonzero=(
                    "dq",
                    lambda values: int((np.abs(values) > 1e-12).sum()),
                ),
                positive=("dq", lambda values: int((values > 1e-12).sum())),
                negative=("dq", lambda values: int((values < -1e-12).sum())),
                major=(
                    "dq",
                    lambda values: int((values >= 0.5 - 1e-12).sum()),
                ),
                energy=(
                    "dq",
                    lambda values: float(np.sum(np.asarray(values) ** 2)),
                ),
            )
        )
        energy = prompt_events["energy"].sort_values(ascending=False)
        total_energy = float(energy.sum())
        concentration = {
            f"top_{count}_prompt_share": (
                float(energy.iloc[:count].sum() / total_energy)
                if total_energy
                else 0.0
            )
            for count in (1, 5, 10, 25)
        }
        output[str(compressor)] = {
            "rows": len(group),
            "mean": float(group["dq"].mean()),
            "std": float(group["dq"].std(ddof=0)),
            "zero": int((signs == "zero").sum()),
            "positive": int((signs == "positive").sum()),
            "negative": int((signs == "negative").sum()),
            "major": int((group["dq"] >= 0.5 - 1e-12).sum()),
            "unique_dq_values": int(group["dq"].nunique()),
            "most_common_dq": [
                {"value": float(value), "count": int(count)}
                for value, count in support.iloc[:20].items()
            ],
            "unique_strict_dq_values": int(group["dq_strict"].nunique()),
            "most_common_strict_dq": [
                {"value": float(value), "count": int(count)}
                for value, count in strict_support.iloc[:20].items()
            ],
            "loose_strict_exact_agreement": float(
                np.mean(
                    np.isclose(
                        group["dq"].to_numpy(dtype=np.float64),
                        group["dq_strict"].to_numpy(dtype=np.float64),
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            ),
            "loose_to_strict_sign": {
                str(row): {
                    str(column): int(transitions.loc[row, column])
                    for column in transitions.columns
                }
                for row in transitions.index
            },
            "by_ratio": by_ratio,
            "prompt_event_support": {
                "prompts": len(prompt_events),
                "with_any_nonzero": int((prompt_events["nonzero"] > 0).sum()),
                "with_any_positive": int(
                    (prompt_events["positive"] > 0).sum()
                ),
                "with_any_negative": int(
                    (prompt_events["negative"] > 0).sum()
                ),
                "with_any_major": int((prompt_events["major"] > 0).sum()),
                "rows_per_prompt": quantiles(prompt_events["rows"]),
                "nonzero_rows_per_prompt": quantiles(
                    prompt_events["nonzero"]
                ),
                "energy_concentration": concentration,
            },
        }
    return output


def control_audit(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for compressor, group in frame.groupby("compressor", sort=True):
        shift = group["q_control"].to_numpy(dtype=np.float64) - group[
            "q_ref"
        ].to_numpy(dtype=np.float64)
        result: dict[str, Any] = {
            "rows": len(group),
            "nonzero_q_control_minus_q_ref": int(
                (np.abs(shift) > 1e-12).sum()
            ),
            "max_abs_q_control_minus_q_ref": float(np.max(np.abs(shift))),
        }
        if compressor == "expected_attention":
            cells = group[
                ["prompt_id", "s", "q_ref", "q_control"]
            ].drop_duplicates()
            cell_shift = cells["q_control"].to_numpy(
                dtype=np.float64
            ) - cells["q_ref"].to_numpy(dtype=np.float64)
            live_delta = group["q_ref"] - group["q_hybrid"]
            locked_sign = sign_label(group["dq"])
            live_sign = sign_label(live_delta)
            result.update(
                {
                    "unique_prompt_s_controls": len(cells),
                    "nonzero_sham_shift_cells": int(
                        (np.abs(cell_shift) > 1e-12).sum()
                    ),
                    "mean_abs_sham_shift_cells": float(
                        np.mean(np.abs(cell_shift))
                    ),
                    "sign_changes_vs_live_reference_delta": int(
                        (locked_sign != live_sign).sum()
                    ),
                    "magnitude_changes_vs_live_reference_delta": int(
                        (
                            ~np.isclose(
                                group["dq"], live_delta, rtol=0.0, atol=1e-12
                            )
                        ).sum()
                    ),
                }
            )
        output[str(compressor)] = result
    return output


def trajectory_summary(frame: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for compressor, group in frame.groupby("compressor", sort=True):
        pivot = group.pivot(
            index=["prompt_id", "s"], columns="ratio", values="dq"
        )
        complete = pivot.dropna().loc[:, list(RATIOS)]
        differences = np.diff(complete.to_numpy(dtype=np.float64), axis=1)
        monotone = np.all(differences >= -1e-12, axis=1)
        flat = np.all(np.abs(differences) <= 1e-12, axis=1)
        adjacent_differences: list[float] = []
        adjacent_sign_changes = 0
        adjacent_opposite_sign = 0
        adjacent_pairs = 0
        for _, trajectory in group.groupby(
            ["prompt_id", "ratio"], sort=False
        ):
            ordered = trajectory.sort_values("s")
            values = ordered["dq"].to_numpy(dtype=np.float64)
            if len(values) < 2:
                continue
            delta = np.diff(values)
            adjacent_differences.extend(np.abs(delta).tolist())
            left = np.sign(values[:-1])
            right = np.sign(values[1:])
            adjacent_pairs += len(delta)
            adjacent_sign_changes += int(np.sum(left != right))
            adjacent_opposite_sign += int(np.sum(left * right < 0))
        trajectory_sizes = group.groupby(
            ["prompt_id", "ratio"], sort=False
        ).size()
        global_mse = float(np.mean((group["dq"] - group["dq"].mean()) ** 2))
        output[str(compressor)] = {
            "prompt_s_cells": len(pivot),
            "complete_four_ratio_cells": len(complete),
            "monotone_nondecreasing_cells": int(monotone.sum()),
            "monotone_fraction": float(monotone.mean()),
            "flat_across_ratio_cells": int(flat.sum()),
            "any_ratio_decrease_cells": int((~monotone).sum()),
            "adjacent_s_pairs": adjacent_pairs,
            "adjacent_s_exact_same_fraction": float(
                np.mean(np.asarray(adjacent_differences) <= 1e-12)
            ),
            "adjacent_s_sign_change_fraction": (
                adjacent_sign_changes / adjacent_pairs
                if adjacent_pairs
                else 0.0
            ),
            "adjacent_s_opposite_sign_fraction": (
                adjacent_opposite_sign / adjacent_pairs
                if adjacent_pairs
                else 0.0
            ),
            "adjacent_s_abs_jump": quantiles(adjacent_differences),
            "adjacent_s_jump_ge_0_5_fraction": float(
                np.mean(np.asarray(adjacent_differences) >= 0.5 - 1e-12)
            ),
            "boundaries_per_prompt_ratio": quantiles(trajectory_sizes),
            "s_support": {
                "unique": int(group["s"].nunique()),
                "range": [int(group["s"].min()), int(group["s"].max())],
                "counts": {
                    str(int(key)): int(value)
                    for key, value in group["s"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
            },
            "descriptive_variance": {
                "total_mse_about_global_mean": global_mse,
                "eta_squared_prompt": eta_squared(group, ["prompt_id"]),
                "eta_squared_ratio": eta_squared(group, ["ratio"]),
                "eta_squared_s": eta_squared(group, ["s"]),
                "in_sample_mse_ratio_mean": in_sample_group_mse(
                    group, ["ratio"]
                ),
                "in_sample_mse_ratio_s_mean": in_sample_group_mse(
                    group, ["ratio", "s"]
                ),
                "in_sample_mse_prompt_mean": in_sample_group_mse(
                    group, ["prompt_id"]
                ),
                "in_sample_mse_prompt_ratio_mean": in_sample_group_mse(
                    group, ["prompt_id", "ratio"]
                ),
                "warning": (
                    "Descriptive in-sample decompositions; "
                    "prompt-conditioned values are not deployable "
                    "performance estimates."
                ),
            },
        }
    return output


def representative_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = frame.sort_values(["compressor", "prompt_id", "ratio", "s"])
    for compressor, group in ordered.groupby("compressor", sort=True):
        categories: Mapping[str, pd.Series] = {
            "zero": np.isclose(group["dq"], 0.0, rtol=0.0, atol=1e-12),
            "positive": group["dq"] > 1e-12,
            "negative": group["dq"] < -1e-12,
            "major": group["dq"] >= 0.5 - 1e-12,
            "earliest": group["s"] == group["s"].min(),
            "latest": group["s"] == group["s"].max(),
        }
        for category, mask in categories.items():
            subset = group.loc[mask]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    "compressor": str(compressor),
                    "category": category,
                    "prompt_id": str(row["prompt_id"]),
                    "ratio": float(row["ratio"]),
                    "s": int(row["s"]),
                    "dq": float(row["dq"]),
                    "dq_strict": float(row["dq_strict"]),
                    "q_ref": float(row["q_ref"]),
                    "q_control": float(row["q_control"]),
                    "q_hybrid": float(row["q_hybrid"]),
                    "intervention_semantics": str(
                        row["intervention_semantics"]
                    ),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    evidence = load_json(args.evidence)
    v2_lock = load_json(args.v2_lock)
    m3_lock = load_json(args.m3_lock)
    m3_freeze = load_json(args.m3_freeze)
    report = load_json(args.m3_report)
    result_freeze = load_json(args.m3_result_freeze)
    development = tuple(
        str(value) for value in evidence["split"]["train_prompt_ids"]
    )
    quarantine = tuple(
        str(value) for value in evidence["split"]["test_prompt_ids"]
    )
    if len(development) != len(set(development)) or len(quarantine) != len(
        set(quarantine)
    ):
        raise ValueError("prompt rosters contain duplicates")
    if set(development) & set(quarantine):
        raise ValueError("development and quarantine overlap")
    if len(development) != 154 or len(quarantine) != 46:
        raise ValueError("unexpected prompt roster size")
    if (
        hash_prompt_ids(development)
        != m3_lock["prompt_partitions"]["development_prompt_ids_sha256"]
    ):
        raise ValueError("development prompt digest mismatch")
    folds = make_folds(development, v2_lock)

    frame, parquet_metadata = load_dev_frame(args.base, development)
    if set(frame["prompt_id"]) != set(development):
        raise ValueError(
            "filtered parquet does not contain the exact development roster"
        )
    if set(frame["prompt_id"]) & set(quarantine):
        raise ValueError("protected prompt entered Python memory")
    keys = [
        tuple(row)
        for row in frame[list(KEY_COLUMNS)].itertuples(index=False, name=None)
    ]
    base_keys = set(keys)

    parity = np.abs(
        frame["dq"].to_numpy(dtype=np.float64)
        - (
            frame["q_control"].to_numpy(dtype=np.float64)
            - frame["q_hybrid"].to_numpy(dtype=np.float64)
        )
    )
    strict_parity = np.abs(
        frame["dq_strict"].to_numpy(dtype=np.float64)
        - (
            frame["q_control_strict"].to_numpy(dtype=np.float64)
            - frame["q_hybrid_strict"].to_numpy(dtype=np.float64)
        )
    )
    damaged_errors = int(
        np.sum(
            frame["damaged"].to_numpy(dtype=np.int64)
            != (frame["dq"].to_numpy(dtype=np.float64) > 1e-12)
        )
    )
    major_errors = int(
        np.sum(
            frame["major_damage"].to_numpy(dtype=np.int64)
            != (frame["dq"].to_numpy(dtype=np.float64) >= 0.5 - 1e-12)
        )
    )
    numeric_audit_columns = [
        "ratio",
        "s",
        "relative_s",
        "ref_len",
        "q_ref",
        "q_control",
        "q_hybrid",
        "dq",
        *STRICT_COLUMNS,
        *sorted(CAUSAL_FEATURE_COLUMNS),
    ]
    missingness: dict[str, Any] = {}
    for column in numeric_audit_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        nulls = int(frame[column].isna().sum())
        nonfinite = int((~np.isfinite(values)).sum())
        if nulls or nonfinite:
            missingness[column] = {"null": nulls, "nonfinite": nonfinite}

    global_audit, global_prefixes = audit_sidecar(
        args.global_sidecar, set(development), set(quarantine)
    )
    band_audit, band_prefixes = audit_sidecar(
        args.band_sidecar, set(development), set(quarantine)
    )
    global_keys = global_audit.pop("keys")
    band_keys = band_audit.pop("keys")
    prefix_mismatches = sum(
        global_prefixes[key] != band_prefixes[key]
        for key in global_keys & band_keys
    )
    oof_audit = audit_oof(
        args.m3_oof, args.m2_oof, base_keys, development, folds
    )

    compressor_counts = {
        str(key): int(value)
        for key, value in frame["compressor"].value_counts().items()
    }
    ratio_counts = {
        f"{compressor}|{float(ratio)}": int(len(group))
        for (compressor, ratio), group in frame.groupby(
            ["compressor", "ratio"], sort=True
        )
    }
    grids = {
        compressor: {
            (str(row.prompt_id), float(row.ratio), int(row.s))
            for row in group[["prompt_id", "ratio", "s"]].itertuples(
                index=False
            )
        }
        for compressor, group in frame.groupby("compressor", sort=True)
    }
    grid_differences = {
        f"{left}|{right}": len(grids[left] ^ grids[right])
        for left in sorted(grids)
        for right in sorted(grids)
        if left < right
    }
    fold_support: dict[str, Any] = {}
    fold_by_prompt = {
        prompt: fold.index for fold in folds for prompt in fold.prompt_ids
    }
    frame_with_fold = frame.assign(
        _fold=frame["prompt_id"].map(fold_by_prompt)
    )
    for (compressor, fold), group in frame_with_fold.groupby(
        ["compressor", "_fold"], sort=True
    ):
        prompt = group.groupby("prompt_id")["dq"]
        fold_support[f"{compressor}|{int(fold)}"] = {
            "prompts": int(group["prompt_id"].nunique()),
            "rows": len(group),
            "prompts_any_positive": int(
                prompt.apply(
                    lambda values: bool((values > 1e-12).any())
                ).sum()
            ),
            "prompts_any_negative": int(
                prompt.apply(
                    lambda values: bool((values < -1e-12).any())
                ).sum()
            ),
            "prompts_any_major": int(
                prompt.apply(
                    lambda values: bool((values >= 0.5 - 1e-12).any())
                ).sum()
            ),
        }

    actual_hashes = {
        "base": sha256_file(args.base),
        "evidence": sha256_file(args.evidence),
        "v2_lock": sha256_file(args.v2_lock),
        "m3_lock": sha256_file(args.m3_lock),
        "m3_freeze": sha256_file(args.m3_freeze),
        "m3_report": sha256_file(args.m3_report),
        "m3_result_freeze": sha256_file(args.m3_result_freeze),
        "m3_oof": sha256_file(args.m3_oof),
        "m2_oof": sha256_file(args.m2_oof),
    }
    expected_hashes = {
        "base": m3_freeze["input_sha256"]["base_parquet"],
        "evidence": m3_freeze["input_sha256"]["evidence"],
        "m3_lock": m3_freeze["input_sha256"]["m3_protocol_lock"],
        "m3_freeze": result_freeze["m3_prefit_freeze_sha256"],
        "m3_report": result_freeze["report"]["sha256"],
        "m3_oof": result_freeze["oof"]["sha256"],
        "m2_oof": m3_freeze["input_sha256"]["m2_oof"],
    }
    hash_mismatches = {
        name: {"expected": expected, "actual": actual_hashes[name]}
        for name, expected in expected_hashes.items()
        if actual_hashes[name] != expected
    }
    output = {
        "schema_version": "herald.magnitude_e1_understand_audit.v1",
        "status": "descriptive_development_only",
        "safety": {
            "development_prompts": len(development),
            "development_prompt_ids_sha256": hash_prompt_ids(development),
            "quarantined_prompts": len(quarantine),
            "quarantined_prompt_ids_sha256": hash_prompt_ids(quarantine),
            "overlap": 0,
            "scanner_filter_applied_before_python_conversion": True,
            "confirmation_or_reserve_rows_materialized": False,
        },
        "hashes": {
            "actual": actual_hashes,
            "expected": expected_hashes,
            "mismatches": hash_mismatches,
        },
        "base_integrity": {
            "parquet_metadata": parquet_metadata,
            "development_rows": len(frame),
            "unique_keys": len(base_keys),
            "duplicate_keys": len(keys) - len(base_keys),
            "compressor_counts": compressor_counts,
            "ratio_counts": ratio_counts,
            "model_values": sorted(
                str(value) for value in frame["model"].unique()
            ),
            "task_values": sorted(
                str(value) for value in frame["task"].unique()
            ),
            "prompt_count": int(frame["prompt_id"].nunique()),
            "dq_parity_max_abs_error": float(np.max(parity)),
            "strict_dq_parity_max_abs_error": float(np.max(strict_parity)),
            "damaged_flag_errors": damaged_errors,
            "major_damage_flag_errors": major_errors,
            "q_range": [
                float(frame[["q_ref", "q_control", "q_hybrid"]].min().min()),
                float(frame[["q_ref", "q_control", "q_hybrid"]].max().max()),
            ],
            "dq_range": [float(frame["dq"].min()), float(frame["dq"].max())],
            "nonfinite_or_null_columns": missingness,
            "cross_compressor_grid_symmetric_differences": grid_differences,
            "fold_support": fold_support,
        },
        "sensor_integrity": {
            "global": global_audit,
            "bands": band_audit,
            "global_missing_base_keys": len(base_keys - global_keys),
            "global_extra_keys": len(global_keys - base_keys),
            "band_missing_base_keys": len(base_keys - band_keys),
            "band_extra_keys": len(band_keys - base_keys),
            "global_band_prefix_hash_mismatches": prefix_mismatches,
        },
        "oof_integrity": oof_audit,
        "label_geometry": label_summary(frame),
        "control_semantics": control_audit(frame),
        "trajectory_geometry": trajectory_summary(frame),
        "representative_rows": representative_rows(frame),
        "interpretation_limits": [
            "All analyses are descriptive on the repeatedly reused "
            "154-prompt development roster.",
            "Rows and boundaries are repeated measurements; only prompts "
            "are independent split units.",
            "In-sample prompt or trajectory decompositions are not "
            "deployable performance estimates.",
            "Strict labels, ref_len, relative_s, and control/reference "
            "outcomes were inspected only to audit the development estimand "
            "and are forbidden candidate inputs.",
            "No confirmation or reserve outcomes were materialized.",
        ],
        "frozen_report_claim": {
            "schema_version": report["schema_version"],
            "any_compressor_pass": report["development_gate"][
                "any_compressor_pass"
            ],
            "passing_compressors": report["development_gate"][
                "passing_compressors"
            ],
        },
    }
    if hash_mismatches:
        raise ValueError(f"frozen hash mismatches: {hash_mismatches}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "development_rows": len(frame),
                "duplicate_keys": len(keys) - len(base_keys),
                "hash_mismatches": len(hash_mismatches),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
