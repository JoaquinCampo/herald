"""Independently audit and adversarially challenge the locked H4 result."""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from herald.magnitude_v2 import hash_prompt_ids, sha256_file

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
SOURCE_COLUMNS = (*KEY_COLUMNS, "m2_prediction", "dq", "fold")
TRANSFORM_COLUMNS = (
    *KEY_COLUMNS,
    "c0_prediction",
    "causal_avg_prediction",
)
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
EXPECTED_ROWS = 45_180
EXPECTED_PROMPTS = 154
EXPECTED_STRIDE = 16
RESAMPLES = 10_000
SEED = 314_159
CONFIDENCE = 0.95
MINIMUM_WORTHWHILE_SKILL = 0.01
PLATEAU_TOLERANCE = 1e-12
LARGE_JUMP_THRESHOLD = 0.5
EXPECTED_HASHES = {
    "source_oof_sha256": (
        "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
    ),
    "h2_report_sha256": (
        "e72f5e71781e6e3dc824aad21d1f2400cce9fffb6a0444380b06c44a8e0affe1"
    ),
    "strategy_sha256": (
        "f8ef719388ffd3d9681b4f5e90eb727138e20420dc6870a83ac9bed5b9b16164"
    ),
    "protocol_lock_sha256": (
        "09257b4fa4bc728cbaff5515aa7b3e117b48ad98b4adb68a216e3453392d319e"
    ),
    "transform_sha256": (
        "775c45f9b0fb879a15148051c516ecf0575b0b98789cdd871e7e169cc377db93"
    ),
    "manifest_sha256": (
        "2ceb04a38e34ca629e35e08fdb35d28b20df892b03240ba307a89c891ed2d443"
    ),
    "stage_a_log_sha256": (
        "bc98895d6c3d84c295bf22dd3f2cb12c93094346499b69efe1a6b5e40d078c40"
    ),
    "scoring_lock_sha256": (
        "93b8aaf734c811ee7f6e882af3511daf9613e3d760d22725035841af7ed6e35b"
    ),
    "report_sha256": (
        "ab185944071cd4a0dfef8077494b74b0cf4dae93e8e14ad2db26ce1d7cdf2158"
    ),
    "execution_log_sha256": (
        "8a5ec8cca652aba81b203bb2e3a94c4e7746ca738dff79b9ce65eb005c2d44a3"
    ),
    "transform_implementation_sha256": (
        "acc78a0384cdf4c0ccbd976eeee5d6591728ddf2a9fa2892cdb6b4d06c256834"
    ),
    "score_implementation_sha256": (
        "04c2f79e5e263c04b0d508ee7f98bafc81198ca3562f7b65b5007aa144d6aa2a"
    ),
}
SCHEMA_VERSION = "herald.magnitude_e1_h4_independent_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--h2-report", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-a-log", type=Path, required=True)
    parser.add_argument("--scoring-lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execution-log", type=Path, required=True)
    parser.add_argument("--transform-script", type=Path, required=True)
    parser.add_argument("--score-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def actual_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        "source_oof_sha256": sha256_file(args.source_oof),
        "h2_report_sha256": sha256_file(args.h2_report),
        "strategy_sha256": sha256_file(args.strategy),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "transform_sha256": sha256_file(args.transform),
        "manifest_sha256": sha256_file(args.manifest),
        "stage_a_log_sha256": sha256_file(args.stage_a_log),
        "scoring_lock_sha256": sha256_file(args.scoring_lock),
        "report_sha256": sha256_file(args.report),
        "execution_log_sha256": sha256_file(args.execution_log),
        "transform_implementation_sha256": sha256_file(args.transform_script),
        "score_implementation_sha256": sha256_file(args.score_script),
    }


def read_columns(
    path: Path, columns: tuple[str, ...], label: str
) -> pd.DataFrame:
    schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
    missing = set(columns) - set(schema.names)
    if missing:
        raise ValueError(f"{label} lacks columns: {sorted(missing)}")
    table = pq.read_table(path, columns=list(columns))  # type: ignore[no-untyped-call]
    if tuple(table.column_names) != columns:
        raise ValueError(f"{label} reader loaded an unexpected column")
    return table.to_pandas()


def load_joined(source_path: Path, transform_path: Path) -> pd.DataFrame:
    source = read_columns(source_path, SOURCE_COLUMNS, "source OOF")
    transformed = read_columns(
        transform_path,
        TRANSFORM_COLUMNS,
        "H4 transform",
    )
    for label, frame in (("source", source), ("transform", transformed)):
        if len(frame) != EXPECTED_ROWS:
            raise ValueError(f"{label} row count differs from H4 contract")
        if frame[list(KEY_COLUMNS)].duplicated().any():
            raise ValueError(f"{label} contains duplicate H4 keys")
    joined = source.merge(
        transformed,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != EXPECTED_ROWS or set(joined["_merge"]) != {"both"}:
        raise ValueError("independent H4 key join is incomplete")
    joined = joined.drop(columns="_merge")
    numeric = joined[
        ["m2_prediction", "dq", "c0_prediction", "causal_avg_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("independent H4 inputs contain nonfinite values")
    return joined


def artifact_checks(
    args: argparse.Namespace, hashes: Mapping[str, str]
) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    protocol = load_json(args.protocol_lock)
    scoring = load_json(args.scoring_lock)
    report = load_json(args.report)
    stage_a_log = load_json(args.stage_a_log)
    execution_log = load_json(args.execution_log)
    metadata = pq.read_metadata(args.transform).metadata or {}  # type: ignore[no-untyped-call]
    decoded_metadata = {
        key.decode(): value.decode() for key, value in metadata.items()
    }
    checks = {
        "all_hashes_exact": dict(hashes) == EXPECTED_HASHES,
        "protocol_locked_before_transform": (
            protocol.get("status") == "locked_before_label_blind_h4_transform"
        ),
        "manifest_frozen_before_scoring": (
            manifest.get("status")
            == "label_blind_h4_transform_frozen_before_scoring"
        ),
        "manifest_label_blind": (
            manifest.get("outcome_or_label_columns_loaded") == []
        ),
        "manifest_authorized_scoring": (
            manifest.get("feasibility", {}).get("score_stage_b") is True
        ),
        "stage_a_logged_before_scoring": (
            stage_a_log.get("status")
            == "label_blind_feasibility_logged_before_scoring_lock"
        ),
        "scoring_locked_before_outcome_join": (
            scoring.get("status")
            == "locked_after_label_blind_feasibility_before_outcome_join"
        ),
        "scoring_prelock_outcomes_empty": (
            scoring.get("label_free_gate", {}).get(
                "outcome_or_label_columns_loaded_before_lock"
            )
            == []
        ),
        "report_status_exact": (
            report.get("status") == "locked_h4_scoring_complete"
        ),
        "execution_log_precedes_reflection": (
            execution_log.get("status")
            == "locked_result_logged_before_independent_reflection"
        ),
        "confirmation_sealed_everywhere": all(
            value == "sealed_not_run"
            for value in (
                protocol.get("confirmation_status"),
                manifest.get("confirmation_status"),
                scoring.get("confirmation_status"),
                report.get("provenance", {}).get("confirmation_status"),
                stage_a_log.get("source_access", {}).get(
                    "confirmation_status"
                ),
                execution_log.get("execution_checks", {}).get(
                    "confirmation_status"
                ),
            )
        ),
        "parquet_metadata_label_blind": (
            decoded_metadata.get("herald.labels_loaded") == "false"
        ),
        "parquet_metadata_source_exact": (
            decoded_metadata.get("herald.source_oof_sha256")
            == EXPECTED_HASHES["source_oof_sha256"]
        ),
        "parquet_metadata_protocol_exact": (
            decoded_metadata.get("herald.protocol_lock_sha256")
            == EXPECTED_HASHES["protocol_lock_sha256"]
        ),
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "parquet_metadata": decoded_metadata,
    }


def independent_transform(
    frame: pd.DataFrame,
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    reproduced = np.empty(len(frame), dtype=np.float64)
    maximum_difference = 0.0
    first_boundary_errors = 0
    gap_errors = 0
    groups = frame.groupby(["compressor", "prompt_id", "ratio"], sort=True)
    for (_compressor, _prompt_id, _ratio), group in groups:
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
        expected = source.copy()
        expected[1:] = (source[1:] + source[:-1]) / 2.0
        observed = ordered["causal_avg_prediction"].to_numpy(dtype=np.float64)
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(expected - observed))),
        )
        first_boundary_errors += int(observed[0] != source[0])
        reproduced[ordered.index.to_numpy(dtype=np.int64)] = expected
    c0_parity = float(
        np.max(
            np.abs(
                frame["m2_prediction"].to_numpy(dtype=np.float64)
                - frame["c0_prediction"].to_numpy(dtype=np.float64)
            )
        )
    )
    checks = {
        "trajectory_count_exact": groups.ngroups == 1_848,
        "gap_errors_zero": gap_errors == 0,
        "first_boundary_errors_zero": first_boundary_errors == 0,
        "c0_parity_exact": c0_parity == 0.0,
        "transform_reproduced_exact": maximum_difference == 0.0,
    }
    return reproduced, {
        "checks": checks,
        "pass": all(checks.values()),
        "trajectories": groups.ngroups,
        "gap_errors": gap_errors,
        "first_boundary_errors": first_boundary_errors,
        "c0_parity_maximum_absolute_difference": c0_parity,
        "transform_maximum_absolute_difference": maximum_difference,
    }


def prompt_ratio_values(
    frame: pd.DataFrame,
    values: NDArray[np.float64],
    prompts: tuple[str, ...],
) -> tuple[NDArray[np.float64], dict[float, NDArray[np.float64]]]:
    work = frame.assign(_value=values)
    cells = work.groupby(
        ["prompt_id", "ratio"],
        observed=True,
        sort=True,
    )["_value"].mean()
    table = cells.unstack("ratio").reindex(
        index=list(prompts),
        columns=list(RATIOS),
    )
    matrix = table.to_numpy(dtype=np.float64)
    if (
        matrix.shape != (EXPECTED_PROMPTS, len(RATIOS))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("independent H4 prompt-ratio table is incomplete")
    by_ratio = {
        ratio: matrix[:, index].copy() for index, ratio in enumerate(RATIOS)
    }
    return np.mean(matrix, axis=1), by_ratio


def independent_feasibility(
    frame: pd.DataFrame,
    manifest: Mapping[str, Any],
    h2_report: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reproduced: dict[str, Any] = {}
    maximum_difference = 0.0
    manifest_values = manifest["feasibility"]["by_compressor"]
    for compressor in COMPRESSORS:
        group = frame[frame["compressor"] == compressor]
        energy_rows = (
            group["causal_avg_prediction"].to_numpy(dtype=np.float64)
            - group["c0_prediction"].to_numpy(dtype=np.float64)
        ) ** 2
        prompts = tuple(
            sorted(str(value) for value in group["prompt_id"].unique())
        )
        energy_prompt, energy_ratio = prompt_ratio_values(
            group,
            energy_rows,
            prompts,
        )
        energy = float(np.mean(energy_prompt))
        loss = float(h2_report["metrics"][compressor]["c0_loss"])
        relative = energy / loss
        bound = float(2.0 * np.sqrt(relative) - relative)
        observed = manifest_values[compressor]
        comparisons = (
            abs(energy - float(observed["perturbation_energy"])),
            abs(relative - float(observed["relative_energy"])),
            abs(bound - float(observed["skill_upper_bound"])),
            *(
                abs(
                    float(np.mean(energy_ratio[ratio]))
                    - float(
                        observed["per_ratio_perturbation_energy"][str(ratio)]
                    )
                )
                for ratio in RATIOS
            ),
        )
        maximum_difference = max(maximum_difference, *comparisons)
        reproduced[compressor] = {
            "perturbation_energy": energy,
            "frozen_c0_loss": loss,
            "relative_energy": relative,
            "skill_upper_bound": bound,
            "upper_bound_exceeds_0_01": bool(
                bound > MINIMUM_WORTHWHILE_SKILL
            ),
            "per_ratio_perturbation_energy": {
                str(ratio): float(np.mean(energy_ratio[ratio]))
                for ratio in RATIOS
            },
        }
    score_stage_b = any(
        value["upper_bound_exceeds_0_01"] for value in reproduced.values()
    )
    reproduced_gate = {
        "by_compressor": reproduced,
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
        "score_stage_b": score_stage_b,
        "action": (
            "freeze_transform_then_create_separate_scoring_lock"
            if score_stage_b
            else "retire_H4_without_loading_row_outcomes"
        ),
    }
    checks = {
        "feasibility_maximum_absolute_difference": maximum_difference,
        "score_stage_b_exact": (
            score_stage_b == manifest["feasibility"]["score_stage_b"]
        ),
        "action_exact": (
            reproduced_gate["action"] == manifest["feasibility"]["action"]
        ),
    }
    return reproduced_gate, {
        **checks,
        "pass": maximum_difference <= 1e-15
        and checks["score_stage_b_exact"]
        and checks["action_exact"],
    }


def max_t_intervals(
    estimates: Mapping[str, float],
    bootstrap_values: Mapping[str, NDArray[np.float64]],
    *,
    direction: str,
) -> dict[str, Any]:
    claims = tuple(sorted(estimates))
    matrix = np.column_stack([bootstrap_values[claim] for claim in claims])
    point = np.asarray(
        [estimates[claim] for claim in claims], dtype=np.float64
    )
    standard_errors = np.std(matrix, axis=0, ddof=1)
    centered = matrix - point
    standardized = np.divide(
        centered if direction == "lower" else -centered,
        standard_errors,
        out=np.zeros_like(centered),
        where=standard_errors > 0,
    )
    critical = max(
        0.0,
        float(np.quantile(np.max(standardized, axis=1), CONFIDENCE)),
    )
    intervals: dict[str, dict[str, float]] = {}
    for column, claim in enumerate(claims):
        width = critical * standard_errors[column]
        interval = {
            "estimate": float(point[column]),
            "standard_error": float(standard_errors[column]),
        }
        if direction == "lower":
            interval["lower"] = float(point[column] - width)
        else:
            interval["upper"] = float(point[column] + width)
        intervals[claim] = interval
    return {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "direction": direction,
        "sidedness": "one-sided",
        "multiplicity": "simultaneous max-T over 3 claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def independent_jump_decomposition(
    frame: pd.DataFrame,
    total_improvement: float,
) -> dict[str, Any]:
    categories = {
        name: {
            "cells": 0,
            "macro_weight": 0.0,
            "c0_loss_contribution": 0.0,
            "causal_avg_loss_contribution": 0.0,
            "absolute_loss_improvement_contribution": 0.0,
        }
        for name in ("plateau", "small_jump", "large_jump")
    }
    first_boundary_cells = 0
    for (_prompt_id, _ratio), group in frame.groupby(
        ["prompt_id", "ratio"],
        sort=True,
    ):
        ordered = group.sort_values("s")
        count = len(ordered)
        weight = 1.0 / (EXPECTED_PROMPTS * len(RATIOS) * count)
        target = ordered["dq"].to_numpy(dtype=np.float64)
        c0 = ordered["c0_prediction"].to_numpy(dtype=np.float64)
        causal = ordered["causal_avg_prediction"].to_numpy(dtype=np.float64)
        first_boundary_cells += 1
        for index in range(1, count):
            jump = abs(target[index] - target[index - 1])
            if jump <= PLATEAU_TOLERANCE:
                category = "plateau"
            elif jump < LARGE_JUMP_THRESHOLD:
                category = "small_jump"
            else:
                category = "large_jump"
            c0_loss = (target[index] - c0[index]) ** 2
            causal_loss = (target[index] - causal[index]) ** 2
            cell = categories[category]
            cell["cells"] = int(cell["cells"]) + 1
            cell["macro_weight"] = float(cell["macro_weight"]) + weight
            cell["c0_loss_contribution"] = (
                float(cell["c0_loss_contribution"]) + weight * c0_loss
            )
            cell["causal_avg_loss_contribution"] = (
                float(cell["causal_avg_loss_contribution"])
                + weight * causal_loss
            )
            cell["absolute_loss_improvement_contribution"] = float(
                cell["absolute_loss_improvement_contribution"]
            ) + weight * (c0_loss - causal_loss)
    contribution_sum = float(
        sum(
            float(value["absolute_loss_improvement_contribution"])
            for value in categories.values()
        )
    )
    error = abs(contribution_sum - total_improvement)
    return {
        "scope": "t>1_only",
        "decision_use": False,
        "plateau_tolerance": PLATEAU_TOLERANCE,
        "large_jump_threshold": LARGE_JUMP_THRESHOLD,
        "first_boundary_cells_excluded": first_boundary_cells,
        "categories": categories,
        "contribution_sum": contribution_sum,
        "total_macro_absolute_loss_improvement": total_improvement,
        "contribution_sum_absolute_error": error,
        "pass": error <= 1e-12,
    }


def independent_evaluation(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    prompts = tuple(
        sorted(str(value) for value in frame["prompt_id"].unique())
    )
    rng = np.random.default_rng(SEED)
    draws = rng.integers(
        0, EXPECTED_PROMPTS, size=(RESAMPLES, EXPECTED_PROMPTS)
    )
    metrics: dict[str, Any] = {}
    estimates: dict[str, float] = {}
    bootstrap: dict[str, NDArray[np.float64]] = {}
    decomposition: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    for compressor in COMPRESSORS:
        group = frame[frame["compressor"] == compressor]
        c0_error = (
            group["dq"].to_numpy(dtype=np.float64)
            - group["c0_prediction"].to_numpy(dtype=np.float64)
        ) ** 2
        causal_error = (
            group["dq"].to_numpy(dtype=np.float64)
            - group["causal_avg_prediction"].to_numpy(dtype=np.float64)
        ) ** 2
        c0, c0_ratio = prompt_ratio_values(group, c0_error, prompts)
        causal, causal_ratio = prompt_ratio_values(
            group, causal_error, prompts
        )
        c0_loss = float(np.mean(c0))
        causal_loss = float(np.mean(causal))
        skill = 1.0 - causal_loss / c0_loss
        estimates[compressor] = skill
        c0_draw = np.mean(c0[draws], axis=1)
        causal_draw = np.mean(causal[draws], axis=1)
        bootstrap[compressor] = 1.0 - causal_draw / c0_draw
        per_ratio: dict[str, Any] = {}
        bootstrap_ratio: dict[float, NDArray[np.float64]] = {}
        for ratio in RATIOS:
            raw_loss = float(np.mean(c0_ratio[ratio]))
            transformed_loss = float(np.mean(causal_ratio[ratio]))
            per_ratio[str(ratio)] = {
                "c0_loss": raw_loss,
                "causal_avg_loss": transformed_loss,
                "skill": 1.0 - transformed_loss / raw_loss,
            }
            bootstrap_ratio[ratio] = 1.0 - (
                np.mean(causal_ratio[ratio][draws], axis=1)
                / np.mean(c0_ratio[ratio][draws], axis=1)
            )
        metrics[compressor] = {
            "c0_loss": c0_loss,
            "causal_avg_loss": causal_loss,
            "absolute_loss_improvement": c0_loss - causal_loss,
            "skill": skill,
            "per_ratio": per_ratio,
        }
        decomposition[compressor] = independent_jump_decomposition(
            group,
            c0_loss - causal_loss,
        )

        loo_skills: list[float] = []
        loo_point_gates = 0
        for omitted in range(EXPECTED_PROMPTS):
            keep = np.arange(EXPECTED_PROMPTS) != omitted
            loo_skill = 1.0 - float(np.mean(causal[keep])) / float(
                np.mean(c0[keep])
            )
            loo_skills.append(loo_skill)
            ratio_positive = all(
                1.0
                - float(np.mean(causal_ratio[ratio][keep]))
                / float(np.mean(c0_ratio[ratio][keep]))
                > 0.0
                for ratio in RATIOS
            )
            loo_point_gates += int(
                loo_skill > MINIMUM_WORTHWHILE_SKILL and ratio_positive
            )
        bootstrap_point_gate = (
            bootstrap[compressor] > MINIMUM_WORTHWHILE_SKILL
        )
        for ratio in RATIOS:
            bootstrap_point_gate &= bootstrap_ratio[ratio] > 0.0
        prompt_improvement = c0 - causal
        absolute = np.abs(prompt_improvement)
        top_five_fraction = float(
            np.sort(absolute)[-5:].sum() / absolute.sum()
        )
        prompt_lengths = group.groupby("prompt_id", sort=True)[
            "s"
        ].count().reindex(list(prompts)).to_numpy(dtype=np.float64) / len(
            RATIOS
        )
        length_correlation = float(
            np.corrcoef(prompt_lengths, prompt_improvement)[0, 1]
        )
        fold_skills: dict[str, float] = {}
        prompt_fold = (
            group[["prompt_id", "fold"]]
            .drop_duplicates()
            .set_index("prompt_id")["fold"]
        )
        for fold in sorted(int(value) for value in prompt_fold.unique()):
            mask = np.asarray(
                [int(prompt_fold.loc[prompt]) == fold for prompt in prompts]
            )
            fold_skills[str(fold)] = 1.0 - float(
                np.mean(causal[mask])
            ) / float(np.mean(c0[mask]))
        sensitivity[compressor] = {
            "bootstrap_draws_with_all_point_gates": int(
                np.sum(bootstrap_point_gate)
            ),
            "bootstrap_fraction_with_all_point_gates": float(
                np.mean(bootstrap_point_gate)
            ),
            "leave_one_prompt_out_skill": {
                "minimum": float(np.min(loo_skills)),
                "maximum": float(np.max(loo_skills)),
            },
            "single_prompt_removals_with_all_point_gates": loo_point_gates,
            "causal_average_beats_c0_prompt_fraction": float(
                np.mean(prompt_improvement > 0.0)
            ),
            "top_five_absolute_prompt_contribution_fraction": (
                top_five_fraction
            ),
            "prompt_length_vs_loss_improvement_correlation": (
                length_correlation
            ),
            "fold_skills": fold_skills,
        }

    lower = max_t_intervals(estimates, bootstrap, direction="lower")
    upper = max_t_intervals(estimates, bootstrap, direction="upper")
    inference = {
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
        "lower": lower,
        "upper": upper,
    }
    qualifying = []
    for compressor in COMPRESSORS:
        if lower["intervals"][compressor][
            "lower"
        ] > MINIMUM_WORTHWHILE_SKILL and all(
            metrics[compressor]["per_ratio"][str(ratio)]["skill"] > 0.0
            for ratio in RATIOS
        ):
            qualifying.append(compressor)
    all_upper_futile = all(
        upper["intervals"][compressor]["upper"] <= MINIMUM_WORTHWHILE_SKILL
        for compressor in COMPRESSORS
    )
    if qualifying:
        status = "survives"
        action = "preregister_separate_development_only_h4_trajectory_model"
    elif all_upper_futile:
        status = "robust_failure"
        action = "retire_H4_without_variants"
    else:
        status = "ambiguous_no_go"
        action = "retire_H4_without_variants"
    decision = {
        "status": status,
        "go": bool(qualifying),
        "action": action,
        "qualifying_compressors": qualifying,
        "all_simultaneous_uppers_at_or_below_0_01": all_upper_futile,
        "compressor_ruled_out": {
            compressor: bool(
                upper["intervals"][compressor]["upper"]
                <= MINIMUM_WORTHWHILE_SKILL
            )
            for compressor in COMPRESSORS
        },
    }
    evaluation = {
        "metrics": metrics,
        "inference": inference,
        "jump_decomposition": decomposition,
    }
    return evaluation, decision, sensitivity


def compare_json(
    observed: object,
    reproduced: object,
    *,
    path: str = "root",
) -> tuple[list[str], float]:
    if isinstance(observed, bool) or isinstance(reproduced, bool):
        return ([] if observed == reproduced else [path], 0.0)
    if isinstance(observed, int | float) and isinstance(
        reproduced, int | float
    ):
        difference = abs(float(observed) - float(reproduced))
        return ([] if difference <= 1e-12 else [path], difference)
    if isinstance(observed, dict) and isinstance(reproduced, dict):
        if set(observed) != set(reproduced):
            return [f"{path}.keys"], 0.0
        errors: list[str] = []
        maximum = 0.0
        for key in sorted(observed):
            nested, difference = compare_json(
                observed[key],
                reproduced[key],
                path=f"{path}.{key}",
            )
            errors.extend(nested)
            maximum = max(maximum, difference)
        return errors, maximum
    if isinstance(observed, list) and isinstance(reproduced, list):
        if len(observed) != len(reproduced):
            return [f"{path}.length"], 0.0
        errors = []
        maximum = 0.0
        for index, (left, right) in enumerate(
            zip(observed, reproduced, strict=True)
        ):
            nested, difference = compare_json(
                left,
                right,
                path=f"{path}[{index}]",
            )
            errors.extend(nested)
            maximum = max(maximum, difference)
        return errors, maximum
    return ([] if observed == reproduced else [path], 0.0)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H4 audit: {args.output}"
        )
    hashes = actual_hashes(args)
    artifacts = artifact_checks(args, hashes)
    frame = load_joined(args.source_oof, args.transform)
    reproduced_prediction, transform_audit = independent_transform(frame)
    stored_prediction = frame["causal_avg_prediction"].to_numpy(
        dtype=np.float64
    )
    prediction_difference = float(
        np.max(np.abs(reproduced_prediction - stored_prediction))
    )

    manifest = load_json(args.manifest)
    h2_report = load_json(args.h2_report)
    reproduced_gate, feasibility_audit = independent_feasibility(
        frame,
        manifest,
        h2_report,
    )
    report = load_json(args.report)
    evaluation, decision, sensitivity = independent_evaluation(frame)
    metric_errors, metric_difference = compare_json(
        report["metrics"],
        evaluation["metrics"],
        path="metrics",
    )
    inference_errors, inference_difference = compare_json(
        report["inference"],
        evaluation["inference"],
        path="inference",
    )
    decision_errors, _ = compare_json(
        report["decision"],
        decision,
        path="decision",
    )
    jump_errors, jump_difference = compare_json(
        report["jump_decomposition"],
        evaluation["jump_decomposition"],
        path="jump_decomposition",
    )
    gate_errors, gate_difference = compare_json(
        manifest["feasibility"],
        reproduced_gate,
        path="feasibility",
    )
    scoring_audit = {
        "metric_errors": metric_errors,
        "metric_maximum_absolute_difference": metric_difference,
        "inference_errors": inference_errors,
        "inference_maximum_absolute_difference": inference_difference,
        "decision_errors": decision_errors,
        "jump_decomposition_errors": jump_errors,
        "jump_decomposition_maximum_absolute_difference": jump_difference,
        "feasibility_errors": gate_errors,
        "feasibility_maximum_absolute_difference": gate_difference,
    }
    scoring_pass = not any(
        (
            metric_errors,
            inference_errors,
            decision_errors,
            jump_errors,
            gate_errors,
        )
    )
    prompt_hash = hash_prompt_ids(
        sorted(str(value) for value in frame["prompt_id"].unique())
    )
    checks = {
        "artifacts": artifacts["pass"],
        "transform": transform_audit["pass"] and prediction_difference == 0.0,
        "feasibility": feasibility_audit["pass"] and not gate_errors,
        "scoring": scoring_pass,
        "prompt_roster": prompt_hash
        == "da39c75b8bbe6c4acd43015979c21bddaf9c5625e44722970ca59eee9006b371",
        "confirmation": report["provenance"]["confirmation_status"]
        == "sealed_not_run",
    }
    passed = all(checks.values())
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_h4_result_audit_complete",
        "pass": passed,
        "conclusion": (
            "valid_ambiguous_no_go_retire_H4_without_variants"
            if passed and decision["status"] == "ambiguous_no_go"
            else "invalid_or_unexpected_h4_result"
        ),
        "checks": checks,
        "hashes": hashes,
        "artifact_audit": artifacts,
        "transform_audit": {
            **transform_audit,
            "stored_prediction_maximum_absolute_difference": (
                prediction_difference
            ),
        },
        "feasibility_audit": feasibility_audit,
        "scoring_reproduction": scoring_audit,
        "adversarial_sensitivity": sensitivity,
        "scientific_interpretation": {
            "supported": [
                (
                    "The fixed two-point causal average reduces "
                    "development macro MSE at the point estimate for all "
                    "three compressors."
                ),
                (
                    "Knorm and StreamingLLM have positive point skill at "
                    "every frozen ratio."
                ),
                (
                    "The additive diagnostics attribute most absolute "
                    "gain to target plateaus while large-jump "
                    "contributions are also positive in aggregate."
                ),
            ],
            "not_supported": [
                "H4 qualification",
                "a simultaneous lower bound above 0.01",
                "future-prompt generalization",
                "confirmatory evidence",
                (
                    "any alternate smoothing weight, lag, window, subgroup, "
                    "EWMA, or changepoint model"
                ),
            ],
            "next_action": (
                "Freeze the ambiguous no-go, run no H4 variant, and "
                "return to a genuinely distinct hypothesis."
            ),
        },
        "confirmation_status": "sealed_not_run",
        "implementation_sha256": sha256_file(Path(__file__)),
    }
    if not passed:
        raise ValueError(f"independent H4 audit failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
