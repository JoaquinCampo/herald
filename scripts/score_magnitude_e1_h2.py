"""Score a frozen label-blind H2 projection on development OOF labels."""

from __future__ import annotations

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
PROJECTION_COLUMNS = (*KEY_COLUMNS, "c0_prediction", "pava_prediction")
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
TOTAL_ROWS = 45180
PROMPTS = 154
RESAMPLES = 10000
SEED = 314159
CONFIDENCE = 0.95
MINIMUM_SKILL = 0.01
SCHEMA_VERSION = "herald.magnitude_e1_h2.v1"
PROJECTION_MANIFEST_SCHEMA = "herald.magnitude_e1_h2_projection_manifest.v1"
PROTOCOL_LOCK_SCHEMA = "herald.magnitude_e1_h2_protocol_lock.v1"
SCORING_LOCK_SCHEMA = "herald.magnitude_e1_h2_scoring_lock.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--score-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def pava(values: np.ndarray) -> np.ndarray:
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


def validate_locks(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(args.protocol_lock)
    score_lock = load_json(args.score_lock)
    manifest = load_json(args.projection_manifest)
    if (
        protocol.get("schema_version") != PROTOCOL_LOCK_SCHEMA
        or protocol.get("status") != "locked_before_label_blind_projection"
    ):
        raise ValueError("H2 protocol lock schema/status is invalid")
    if (
        manifest.get("schema_version") != PROJECTION_MANIFEST_SCHEMA
        or manifest.get("status")
        != "label_blind_projection_frozen_before_scoring"
    ):
        raise ValueError("projection manifest schema/status is invalid")
    if (
        score_lock.get("schema_version") != SCORING_LOCK_SCHEMA
        or score_lock.get("status")
        != "locked_after_label_blind_projection_before_scoring"
    ):
        raise ValueError("H2 scoring lock schema/status is invalid")
    expected_hashes = {
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "source_oof_sha256": sha256_file(args.source_oof),
        "projection_sha256": sha256_file(args.projection),
        "projection_manifest_sha256": sha256_file(args.projection_manifest),
    }
    for key, expected in expected_hashes.items():
        if score_lock.get(key) != expected:
            raise ValueError(f"H2 scoring lock {key} mismatch")
    if (
        manifest.get("projection_sha256")
        != expected_hashes["projection_sha256"]
    ):
        raise ValueError("projection hash differs from manifest")
    if (
        manifest.get("protocol_lock_sha256")
        != expected_hashes["protocol_lock_sha256"]
    ):
        raise ValueError("protocol hash differs from projection manifest")
    if manifest.get("outcome_or_label_columns_loaded") != []:
        raise ValueError("label-blind manifest records outcome access")
    if manifest.get("allowed_source_columns") != [
        "prompt_id",
        "compressor",
        "ratio",
        "s",
        "m2_prediction",
    ]:
        raise ValueError(
            "projection manifest source columns are not label-blind"
        )
    expected_implementation = {
        "scripts/project_magnitude_e1_h2.py": str(
            protocol["implementation_sha256"][
                "scripts/project_magnitude_e1_h2.py"
            ]
        ),
        "scripts/score_magnitude_e1_h2.py": sha256_file(Path(__file__)),
    }
    if score_lock.get("implementation_sha256") != expected_implementation:
        raise ValueError("scoring implementation differs from lock")
    expected_scoring = {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "minimum_worthwhile_skill": MINIMUM_SKILL,
        "cluster": "complete_prompt",
        "same_draws_across_compressors": True,
        "loss": "equal_prompt_equal_ratio_mean_boundary_mse",
        "bounds": "one_sided_simultaneous_max_t_lower_and_upper",
    }
    if score_lock.get("scoring") != expected_scoring:
        raise ValueError("scoring configuration differs from lock")
    if score_lock.get("confirmation_status") != "sealed_not_run":
        raise ValueError("confirmation is not sealed")
    return score_lock, manifest


def read_projection(path: Path) -> pd.DataFrame:
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(PROJECTION_COLUMNS),
    )
    if tuple(table.column_names) != PROJECTION_COLUMNS:
        raise ValueError("projection reader loaded unexpected columns")
    frame = table.to_pandas()
    if (
        len(frame) != TOTAL_ROWS
        or frame[list(KEY_COLUMNS)].duplicated().any()
    ):
        raise ValueError("projection keyed table is incomplete or duplicated")
    numeric = frame[["c0_prediction", "pava_prediction"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("projection contains nonfinite predictions")
    return frame


def read_source(path: Path) -> pd.DataFrame:
    table = pq.read_table(  # type: ignore[no-untyped-call]
        path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("source scorer loaded unexpected columns")
    frame = table.to_pandas()
    if (
        len(frame) != TOTAL_ROWS
        or frame[list(KEY_COLUMNS)].duplicated().any()
    ):
        raise ValueError("source keyed table is incomplete or duplicated")
    numeric = frame[["m2_prediction", "dq"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("source scoring columns are nonfinite")
    return frame


def join_inputs(
    source: pd.DataFrame, projection: pd.DataFrame
) -> pd.DataFrame:
    joined = source.merge(
        projection,
        on=list(KEY_COLUMNS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    counts = joined["_merge"].value_counts().to_dict()
    if (
        counts.get("both", 0) != TOTAL_ROWS
        or counts.get("left_only", 0)
        or counts.get("right_only", 0)
    ):
        raise ValueError(f"projection/source key mismatch: {counts}")
    parity = np.abs(
        joined["m2_prediction"].to_numpy(dtype=np.float64)
        - joined["c0_prediction"].to_numpy(dtype=np.float64)
    )
    if float(np.max(parity)) > 0.0:
        raise ValueError(
            "projection C0 column differs from frozen M2 prediction"
        )
    return joined.drop(columns="_merge")


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
        ["prompt_id", "ratio"], observed=True, sort=True
    )["_loss"].mean()
    table = trajectory.unstack("ratio").reindex(
        index=list(prompts), columns=list(RATIOS)
    )
    values = table.to_numpy(dtype=np.float64)
    if (
        values.shape != (PROMPTS, len(RATIOS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("prompt-ratio loss table is incomplete")
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
        raise ValueError("invalid max-T inputs")
    claims = tuple(sorted(estimates))
    matrix = np.column_stack([bootstrap_values[claim] for claim in claims])
    if (
        matrix.shape != (RESAMPLES, len(claims))
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("max-T bootstrap matrix differs from contract")
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
        bound = critical * standard_errors[column]
        interval = {
            "estimate": float(point[column]),
            "standard_error": float(standard_errors[column]),
        }
        if direction == "lower":
            interval["lower"] = float(point[column] - bound)
        else:
            interval["upper"] = float(point[column] + bound)
        intervals[claim] = interval
    return {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "direction": direction,
        "sidedness": "one-sided",
        "multiplicity": f"simultaneous max-T over {len(claims)} claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def projection_falsification(frame: pd.DataFrame) -> dict[str, Any]:
    theorem_violations = 0
    independent_projection_mismatches = 0
    incomplete_curves = 0
    changed = {compressor: 0 for compressor in COMPRESSORS}
    observed_monotone = {compressor: 0 for compressor in COMPRESSORS}
    decomposition = {
        compressor: {
            "observed_monotone": {
                "groups": 0,
                "c0_sse": 0.0,
                "pava_sse": 0.0,
            },
            "observed_nonmonotone": {
                "groups": 0,
                "c0_sse": 0.0,
                "pava_sse": 0.0,
            },
        }
        for compressor in COMPRESSORS
    }
    for (compressor, _prompt_id, _s), group in frame.groupby(
        ["compressor", "prompt_id", "s"], sort=True
    ):
        ordered = group.sort_values("ratio")
        if tuple(float(value) for value in ordered["ratio"]) != RATIOS:
            incomplete_curves += 1
            continue
        raw = ordered["c0_prediction"].to_numpy(dtype=np.float64)
        projected = ordered["pava_prediction"].to_numpy(dtype=np.float64)
        target = ordered["dq"].to_numpy(dtype=np.float64)
        if not np.allclose(pava(raw), projected, rtol=0.0, atol=1e-15):
            independent_projection_mismatches += 1
        if not np.array_equal(raw, projected):
            changed[str(compressor)] += 1
        monotone_target = bool(np.all(np.diff(target) >= -1e-12))
        category = (
            "observed_monotone" if monotone_target else "observed_nonmonotone"
        )
        if monotone_target:
            observed_monotone[str(compressor)] += 1
        raw_sse = float(np.sum((target - raw) ** 2))
        projected_sse = float(np.sum((target - projected) ** 2))
        cell = decomposition[str(compressor)][category]
        cell["groups"] = int(cell["groups"]) + 1
        cell["c0_sse"] = float(cell["c0_sse"]) + raw_sse
        cell["pava_sse"] = float(cell["pava_sse"]) + projected_sse
        if monotone_target and projected_sse > raw_sse + 1e-12:
            theorem_violations += 1
    projection_reproduced = independent_projection_mismatches == 0
    checks = {
        "complete_ratio_curves": incomplete_curves == 0,
        "independent_projection_reproduction": projection_reproduced,
        "monotone_target_projection_theorem": theorem_violations == 0,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "incomplete_curves": incomplete_curves,
        "independent_projection_mismatches": (
            independent_projection_mismatches
        ),
        "monotone_target_theorem_violations": theorem_violations,
        "changed_groups": changed,
        "observed_monotone_groups": observed_monotone,
        "loss_decomposition_unweighted_cell_sse": decomposition,
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
        raise ValueError("compressors do not share one prompt roster")
    prompts = next(iter(prompt_sets.values()))
    if len(prompts) != PROMPTS:
        raise ValueError("prompt roster differs from contract")
    folds = frame.groupby("prompt_id")["fold"].nunique()
    fold_counts = frame.groupby("fold")["prompt_id"].nunique().to_dict()
    if int((folds != 1).sum()) or sorted(
        int(value) for value in fold_counts.values()
    ) != [
        30,
        31,
        31,
        31,
        31,
    ]:
        raise ValueError("frozen fold assignment is invalid")

    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, PROMPTS, size=(RESAMPLES, PROMPTS))
    metrics: dict[str, Any] = {}
    estimates: dict[str, float] = {}
    bootstrap: dict[str, NDArray[np.float64]] = {}
    for compressor in COMPRESSORS:
        group = frame[frame["compressor"] == compressor]
        c0, c0_ratio = prompt_ratio_losses(group, "c0_prediction", prompts)
        projected, projected_ratio = prompt_ratio_losses(
            group, "pava_prediction", prompts
        )
        c0_loss = float(np.mean(c0))
        projected_loss = float(np.mean(projected))
        skill = 1.0 - projected_loss / c0_loss
        estimates[compressor] = skill
        c0_draw = np.mean(c0[draws], axis=1)
        projected_draw = np.mean(projected[draws], axis=1)
        bootstrap[compressor] = 1.0 - projected_draw / c0_draw
        per_ratio = {}
        for ratio in RATIOS:
            raw_loss = float(np.mean(c0_ratio[ratio]))
            pava_loss = float(np.mean(projected_ratio[ratio]))
            per_ratio[str(ratio)] = {
                "c0_loss": raw_loss,
                "pava_loss": pava_loss,
                "skill": 1.0 - pava_loss / raw_loss,
            }
        metrics[compressor] = {
            "c0_loss": c0_loss,
            "pava_loss": projected_loss,
            "skill": skill,
            "per_ratio": per_ratio,
        }
    lower = max_t_intervals(estimates, bootstrap, direction="lower")
    upper = max_t_intervals(estimates, bootstrap, direction="upper")
    successful = [
        compressor
        for compressor in COMPRESSORS
        if lower["intervals"][compressor]["lower"] > MINIMUM_SKILL
        and all(
            values["skill"] > 0.0
            for values in metrics[compressor]["per_ratio"].values()
        )
    ]
    if successful:
        decision = "success"
    elif all(
        upper["intervals"][compressor]["upper"] <= MINIMUM_SKILL
        for compressor in COMPRESSORS
    ):
        decision = "failure"
    else:
        decision = "ambiguous"
    inference = {
        "lower": lower,
        "upper": upper,
        "minimum_worthwhile_skill": MINIMUM_SKILL,
    }
    decision_record = {
        "status": decision,
        "successful_compressors": successful,
        "go": decision == "success",
        "action": {
            "success": (
                "return_to_strategy_for_one_predictive_joint_or_"
                "soft_curve_model"
            ),
            "failure": (
                "retire_hard_monotone_projection_and_return_to_H1_strategy"
            ),
            "ambiguous": ("no_go_for_H2_return_to_understand_without_tuning"),
        }[decision],
    }
    return {"metrics": metrics, "inference": inference}, decision_record


def ensure_new(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite H2 report: {path}")


def main() -> None:
    args = parse_args()
    ensure_new(args.output)
    score_lock, manifest = validate_locks(args)
    projection = read_projection(args.projection)
    source = read_source(args.source_oof)
    frame = join_inputs(source, projection)
    results, decision = evaluate(frame)
    falsification = projection_falsification(frame)
    manifest_invariants = manifest["invariants"]
    invariant_failures = {
        key: value
        for key, value in manifest_invariants.items()
        if key.endswith("_errors") and int(value) != 0
    }
    provenance_checks = {
        "source_sha_matches_lock": score_lock["source_oof_sha256"]
        == sha256_file(args.source_oof),
        "projection_sha_matches_lock": score_lock["projection_sha256"]
        == sha256_file(args.projection),
        "manifest_sha_matches_lock": score_lock["projection_manifest_sha256"]
        == sha256_file(args.projection_manifest),
        "projection_label_blind": manifest["outcome_or_label_columns_loaded"]
        == [],
        "projection_manifest_invariants": not invariant_failures,
        "prompt_roster_sha_matches": manifest["development_prompt_ids_sha256"]
        == hash_prompt_ids(
            tuple(str(value) for value in frame["prompt_id"].unique())
        ),
    }
    all_checks_pass = (
        all(provenance_checks.values()) and falsification["pass"]
    )
    if not all_checks_pass:
        decision = {
            "status": "invalid",
            "successful_compressors": [],
            "go": False,
            "action": "return_to_understand_and_repair_invalid_diagnostic",
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "evaluated_adaptive_development_diagnostic",
        "provenance": {
            "source_oof_sha256": sha256_file(args.source_oof),
            "projection_sha256": sha256_file(args.projection),
            "projection_manifest_sha256": sha256_file(
                args.projection_manifest
            ),
            "protocol_lock_sha256": sha256_file(args.protocol_lock),
            "scoring_lock_sha256": sha256_file(args.score_lock),
            "implementation_sha256": sha256_file(Path(__file__)),
            "development_rows": len(frame),
            "development_prompts": int(frame["prompt_id"].nunique()),
            "confirmation_status": "sealed_not_run",
        },
        **results,
        "decision": decision,
        "falsification": {
            "pass": all_checks_pass,
            "provenance_checks": provenance_checks,
            "projection_checks": falsification,
            "manifest_invariant_failures": invariant_failures,
        },
        "interpretation": {
            "allowed": (
                "Adaptive structural diagnostic on the frozen development "
                "OOF table only."
            ),
            "forbidden": [
                "M3 reinterpretation",
                "future-prompt generalization",
                "confirmatory evidence",
                "opening confirmation or reserve",
                "deployable-model promotion",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "h2_scored",
                "report": str(args.output),
                "report_sha256": sha256_file(args.output),
                "decision": decision,
                "skills": {
                    compressor: results["metrics"][compressor]["skill"]
                    for compressor in COMPRESSORS
                },
                "falsification_pass": all_checks_pass,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
