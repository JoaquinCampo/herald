"""Score the frozen H4 causal-average transform after its feasibility gate."""

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
CURRENT_WEIGHT = 0.5
PREVIOUS_WEIGHT = 0.5
PLATEAU_TOLERANCE = 1e-12
LARGE_JUMP_THRESHOLD = 0.5
EXPECTED_SOURCE_SHA256 = (
    "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
)
EXPECTED_H2_REPORT_SHA256 = (
    "e72f5e71781e6e3dc824aad21d1f2400cce9fffb6a0444380b06c44a8e0affe1"
)
REPORT_SCHEMA_VERSION = "herald.magnitude_e1_h4_report.v1"
PROTOCOL_LOCK_SCHEMA_VERSION = "herald.magnitude_e1_h4_protocol_lock.v1"
SCORING_LOCK_SCHEMA_VERSION = "herald.magnitude_e1_h4_scoring_lock.v1"
MANIFEST_SCHEMA_VERSION = "herald.magnitude_e1_h4_transform_manifest.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--h2-report", type=Path, required=True)
    parser.add_argument("--transform", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--scoring-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def expected_scoring_contract() -> dict[str, Any]:
    return {
        "target": "signed_dq",
        "prediction_methods": ["c0_prediction", "causal_avg_prediction"],
        "loss_weighting": "mean_s_then_equal_ratio_then_equal_prompt",
        "skill": "1-causal_avg_loss/c0_loss",
        "per_ratio_skills": list(RATIOS),
        "bootstrap_unit": "complete_prompt",
        "bootstrap_resamples": RESAMPLES,
        "bootstrap_seed": SEED,
        "bootstrap_same_draws_across_compressors": True,
        "confidence": CONFIDENCE,
        "sidedness": "one_sided",
        "multiplicity": (
            "simultaneous_max_t_over_three_compressor_macro_skills"
        ),
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
        "survival": (
            "any simultaneous lower strictly above 0.01 and all four "
            "per-ratio point skills strictly positive for that compressor"
        ),
        "robust_failure": (
            "all simultaneous uppers less than or equal to 0.01"
        ),
        "otherwise": "ambiguous_no_go",
        "jump_diagnostics_decision_use": False,
    }


def validate_locks(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_json(args.manifest)
    protocol = load_json(args.protocol_lock)
    scoring = load_json(args.scoring_lock)
    strategy = load_json(args.strategy)
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status")
        != "label_blind_h4_transform_frozen_before_scoring"
    ):
        raise ValueError("H4 transform manifest schema/status is invalid")
    if manifest.get("source_oof_sha256") != EXPECTED_SOURCE_SHA256:
        raise ValueError("H4 manifest source hash changed")
    if manifest.get("h2_report_sha256") != EXPECTED_H2_REPORT_SHA256:
        raise ValueError("H4 manifest H2 report hash changed")
    if manifest.get("outcome_or_label_columns_loaded") != []:
        raise ValueError("H4 transform manifest is not label-blind")
    feasibility = manifest.get("feasibility")
    if (
        not isinstance(feasibility, dict)
        or feasibility.get("score_stage_b") is not True
    ):
        raise ValueError("H4 label-free gate did not authorize scoring")
    if (
        protocol.get("schema_version") != PROTOCOL_LOCK_SCHEMA_VERSION
        or protocol.get("status") != "locked_before_label_blind_h4_transform"
    ):
        raise ValueError("H4 protocol lock schema/status is invalid")
    if (
        scoring.get("schema_version") != SCORING_LOCK_SCHEMA_VERSION
        or scoring.get("status")
        != "locked_after_label_blind_feasibility_before_outcome_join"
    ):
        raise ValueError("H4 scoring lock schema/status is invalid")
    expected_hashes = {
        "source_oof_sha256": sha256_file(args.source_oof),
        "h2_report_sha256": sha256_file(args.h2_report),
        "transform_sha256": sha256_file(args.transform),
        "manifest_sha256": sha256_file(args.manifest),
        "strategy_sha256": sha256_file(args.strategy),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "score_implementation_sha256": sha256_file(Path(__file__)),
    }
    for name, expected in expected_hashes.items():
        if scoring.get(name) != expected:
            raise ValueError(f"H4 scoring lock {name} changed")
    if scoring.get("scoring") != expected_scoring_contract():
        raise ValueError("H4 scoring contract differs from lock")
    if strategy.get("selected_hypothesis") != "H4":
        raise ValueError("H4 strategy selection changed")
    if scoring.get("confirmation_status") != "sealed_not_run":
        raise ValueError("confirmation is not sealed")
    return manifest, scoring


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


def joined_frame(source_path: Path, transform_path: Path) -> pd.DataFrame:
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
        raise ValueError("H4 source/transform key join is incomplete")
    joined = joined.drop(columns="_merge")
    parity = np.abs(
        joined["m2_prediction"].to_numpy(dtype=np.float64)
        - joined["c0_prediction"].to_numpy(dtype=np.float64)
    )
    if float(np.max(parity)) != 0.0:
        raise ValueError("H4 transform C0 parity changed")
    numeric = joined[
        ["m2_prediction", "dq", "c0_prediction", "causal_avg_prediction"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("H4 scoring inputs contain nonfinite values")
    if joined["prompt_id"].nunique() != EXPECTED_PROMPTS:
        raise ValueError("H4 scoring prompt count changed")
    if set(joined["compressor"]) != set(COMPRESSORS):
        raise ValueError("H4 scoring compressor roster changed")
    if set(float(value) for value in joined["ratio"].unique()) != set(RATIOS):
        raise ValueError("H4 scoring ratio roster changed")
    return joined


def prompt_ratio_losses(
    frame: pd.DataFrame,
    prediction: str,
    prompts: tuple[str, ...],
) -> tuple[NDArray[np.float64], dict[float, NDArray[np.float64]]]:
    work = frame.assign(
        _loss=(
            frame["dq"].to_numpy(dtype=np.float64)
            - frame[prediction].to_numpy(dtype=np.float64)
        )
        ** 2
    )
    trajectory = work.groupby(
        ["prompt_id", "ratio"],
        observed=True,
        sort=True,
    )["_loss"].mean()
    table = trajectory.unstack("ratio").reindex(
        index=list(prompts),
        columns=list(RATIOS),
    )
    values = table.to_numpy(dtype=np.float64)
    if (
        values.shape != (EXPECTED_PROMPTS, len(RATIOS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("H4 prompt-ratio loss table is incomplete")
    by_ratio = {
        ratio: values[:, index].copy() for index, ratio in enumerate(RATIOS)
    }
    return np.mean(values, axis=1), by_ratio


def max_t_intervals(
    estimates: Mapping[str, float],
    bootstrap_values: Mapping[str, NDArray[np.float64]],
    *,
    direction: str,
) -> dict[str, Any]:
    if direction not in {"lower", "upper"} or set(estimates) != set(
        bootstrap_values
    ):
        raise ValueError("invalid H4 max-T inputs")
    claims = tuple(sorted(estimates))
    matrix = np.column_stack([bootstrap_values[claim] for claim in claims])
    if (
        matrix.shape != (RESAMPLES, len(claims))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("H4 max-T bootstrap matrix differs from contract")
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


def recurrence_audit(frame: pd.DataFrame) -> dict[str, Any]:
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
        source = ordered["c0_prediction"].to_numpy(dtype=np.float64)
        observed = ordered["causal_avg_prediction"].to_numpy(dtype=np.float64)
        expected = source.copy()
        expected[1:] = (
            CURRENT_WEIGHT * source[1:] + PREVIOUS_WEIGHT * source[:-1]
        )
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(observed - expected))),
        )
        first_boundary_errors += int(observed[0] != source[0])
    return {
        "trajectories": groups.ngroups,
        "maximum_recurrence_absolute_difference": maximum_difference,
        "first_boundary_errors": first_boundary_errors,
        "gap_errors": gap_errors,
        "pass": maximum_difference == 0.0
        and first_boundary_errors == 0
        and gap_errors == 0,
    }


def jump_decomposition(
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
    groups = frame.groupby(["prompt_id", "ratio"], sort=True)
    for (_prompt_id, _ratio), group in groups:
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
            float(cell["absolute_loss_improvement_contribution"])
            for cell in categories.values()
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


def evaluate(frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt_sets = {
        compressor: tuple(
            sorted(str(value) for value in group["prompt_id"].unique())
        )
        for compressor, group in frame.groupby("compressor", sort=True)
    }
    if (
        set(prompt_sets) != set(COMPRESSORS)
        or len(set(prompt_sets.values())) != 1
    ):
        raise ValueError("compressors do not share one H4 prompt roster")
    prompts = next(iter(prompt_sets.values()))
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError("H4 prompt roster differs from contract")
    prompt_folds = frame.groupby("prompt_id")["fold"].nunique()
    fold_counts = frame.groupby("fold")["prompt_id"].nunique().to_dict()
    if int((prompt_folds != 1).sum()) or sorted(
        int(value) for value in fold_counts.values()
    ) != [30, 31, 31, 31, 31]:
        raise ValueError("frozen H4 prompt-fold assignment is invalid")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(
        0, EXPECTED_PROMPTS, size=(RESAMPLES, EXPECTED_PROMPTS)
    )
    metrics: dict[str, Any] = {}
    estimates: dict[str, float] = {}
    bootstrap: dict[str, NDArray[np.float64]] = {}
    decomposition: dict[str, Any] = {}
    for compressor in COMPRESSORS:
        group = frame[frame["compressor"] == compressor]
        c0, c0_ratio = prompt_ratio_losses(group, "c0_prediction", prompts)
        causal, causal_ratio = prompt_ratio_losses(
            group,
            "causal_avg_prediction",
            prompts,
        )
        c0_loss = float(np.mean(c0))
        causal_loss = float(np.mean(causal))
        skill = 1.0 - causal_loss / c0_loss
        estimates[compressor] = skill
        c0_draw = np.mean(c0[draws], axis=1)
        causal_draw = np.mean(causal[draws], axis=1)
        bootstrap[compressor] = 1.0 - causal_draw / c0_draw
        per_ratio: dict[str, Any] = {}
        for ratio in RATIOS:
            raw_loss = float(np.mean(c0_ratio[ratio]))
            smoothed_loss = float(np.mean(causal_ratio[ratio]))
            per_ratio[str(ratio)] = {
                "c0_loss": raw_loss,
                "causal_avg_loss": smoothed_loss,
                "skill": 1.0 - smoothed_loss / raw_loss,
            }
        metrics[compressor] = {
            "c0_loss": c0_loss,
            "causal_avg_loss": causal_loss,
            "absolute_loss_improvement": c0_loss - causal_loss,
            "skill": skill,
            "per_ratio": per_ratio,
        }
        decomposition[compressor] = jump_decomposition(
            group,
            c0_loss - causal_loss,
        )

    lower_inference = max_t_intervals(
        estimates,
        bootstrap,
        direction="lower",
    )
    upper_inference = max_t_intervals(
        estimates,
        bootstrap,
        direction="upper",
    )
    inference = {
        "minimum_worthwhile_skill": MINIMUM_WORTHWHILE_SKILL,
        "lower": lower_inference,
        "upper": upper_inference,
    }
    lower = lower_inference["intervals"]
    upper = upper_inference["intervals"]
    qualifying = []
    for compressor in COMPRESSORS:
        all_ratio_positive = all(
            metrics[compressor]["per_ratio"][str(ratio)]["skill"] > 0.0
            for ratio in RATIOS
        )
        if (
            lower[compressor]["lower"] > MINIMUM_WORTHWHILE_SKILL
            and all_ratio_positive
        ):
            qualifying.append(compressor)
    all_upper_futile = all(
        upper[compressor]["upper"] <= MINIMUM_WORTHWHILE_SKILL
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
                upper[compressor]["upper"] <= MINIMUM_WORTHWHILE_SKILL
            )
            for compressor in COMPRESSORS
        },
    }
    return {
        "metrics": metrics,
        "inference": inference,
        "jump_decomposition": decomposition,
    }, decision


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H4 report: {args.output}"
        )
    manifest, scoring_lock = validate_locks(args)
    frame = joined_frame(args.source_oof, args.transform)
    recurrence = recurrence_audit(frame)
    if not recurrence["pass"]:
        raise ValueError(f"H4 recurrence audit failed: {recurrence}")
    evaluation, decision = evaluate(frame)
    decomposition_pass = all(
        value["pass"] for value in evaluation["jump_decomposition"].values()
    )
    falsification_checks = {
        "source_and_transform_keys_exact": True,
        "all_predictions_and_targets_finite": True,
        "c0_source_parity_exact": True,
        "prompt_roster_exact": True,
        "prompt_fold_assignment_exact": True,
        "causal_recurrence_exact": recurrence["pass"],
        "jump_decomposition_additive": decomposition_pass,
        "transform_manifest_label_blind": (
            manifest.get("outcome_or_label_columns_loaded") == []
        ),
        "label_free_gate_authorized_scoring": (
            manifest.get("feasibility", {}).get("score_stage_b") is True
        ),
        "confirmation_remained_sealed": (
            scoring_lock.get("confirmation_status") == "sealed_not_run"
        ),
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "locked_h4_scoring_complete",
        "decision": decision,
        **evaluation,
        "falsification": {
            "pass": all(falsification_checks.values()),
            "checks": falsification_checks,
            "recurrence": recurrence,
        },
        "interpretation": {
            "allowed": (
                "Adaptive causal-trajectory diagnostic on frozen development "
                "OOF predictions only."
            ),
            "forbidden": [
                "M3 reinterpretation",
                "future-prompt generalization",
                "confirmatory evidence",
                "opening confirmation or reserve",
                "deployable-model promotion",
                "filter or subgroup tuning",
            ],
        },
        "provenance": {
            "source_oof_sha256": sha256_file(args.source_oof),
            "source_prompt_ids_sha256": hash_prompt_ids(
                sorted(str(value) for value in frame["prompt_id"].unique())
            ),
            "h2_report_sha256": sha256_file(args.h2_report),
            "transform_sha256": sha256_file(args.transform),
            "manifest_sha256": sha256_file(args.manifest),
            "strategy_sha256": sha256_file(args.strategy),
            "protocol_lock_sha256": sha256_file(args.protocol_lock),
            "scoring_lock_sha256": sha256_file(args.scoring_lock),
            "implementation_sha256": sha256_file(Path(__file__)),
            "confirmation_status": "sealed_not_run",
        },
    }
    if not report["falsification"]["pass"]:
        raise ValueError("H4 falsification checks failed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
