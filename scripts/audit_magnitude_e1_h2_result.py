"""Independently audit and adversarially stress the frozen E1 H2 result."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn.isotonic as isotonic  # type: ignore[import-untyped]
from numpy.typing import NDArray

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
PROJECTION_COLUMNS = (*KEY_COLUMNS, "c0_prediction", "pava_prediction")
SOURCE_COLUMNS = (*KEY_COLUMNS, "m2_prediction", "dq", "fold")
RATIOS = (0.25, 0.5, 0.75, 0.875)
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
TOTAL_ROWS = 45180
PROMPTS = 154
RESAMPLES = 10000
SEED = 314159
CONFIDENCE = 0.95
MINIMUM_SKILL = 0.01
SCHEMA_VERSION = "herald.magnitude_e1_h2_independent_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--score-lock", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--project-script", type=Path, required=True)
    parser.add_argument("--score-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def prompt_roster_sha256(values: pd.Series) -> str:
    prompts = sorted(str(value) for value in values.unique())
    return hashlib.sha256(("\n".join(prompts) + "\n").encode()).hexdigest()


def read_inputs(
    source_path: Path, projection_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    projection_table = pq.read_table(  # type: ignore[no-untyped-call]
        projection_path,
        columns=list(PROJECTION_COLUMNS),
    )
    source_table = pq.read_table(  # type: ignore[no-untyped-call]
        source_path,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(projection_table.column_names) != PROJECTION_COLUMNS:
        raise ValueError(
            "projection columns differ from independent audit contract"
        )
    if tuple(source_table.column_names) != SOURCE_COLUMNS:
        raise ValueError(
            "source columns differ from independent audit contract"
        )
    return source_table.to_pandas(), projection_table.to_pandas()


def join_inputs(
    source: pd.DataFrame, projection: pd.DataFrame
) -> pd.DataFrame:
    for name, frame in (("source", source), ("projection", projection)):
        if (
            len(frame) != TOTAL_ROWS
            or frame[list(KEY_COLUMNS)].duplicated().any()
        ):
            raise ValueError(
                f"{name} keyed table is incomplete or duplicated"
            )
    joined = source.merge(
        projection,
        on=list(KEY_COLUMNS),
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    if len(joined) != TOTAL_ROWS or not (joined["_merge"] == "both").all():
        raise ValueError("source and projection keys do not match exactly")
    joined = joined.drop(columns="_merge")
    numeric = joined[
        ["m2_prediction", "c0_prediction", "pava_prediction", "dq"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("independent audit inputs contain nonfinite values")
    return joined


def verify_projection_with_sklearn(frame: pd.DataFrame) -> dict[str, Any]:
    estimator = isotonic.IsotonicRegression(
        increasing=True, out_of_bounds="raise"
    )
    max_reference_error = 0.0
    monotonicity_errors = 0
    unchanged_errors = 0
    mean_errors = 0
    changed_groups = dict.fromkeys(COMPRESSORS, 0)
    shift_rows: list[float] = []
    for (compressor, _prompt_id, _s), group in frame.groupby(
        ["compressor", "prompt_id", "s"], sort=True
    ):
        ordered = group.sort_values("ratio")
        if tuple(float(value) for value in ordered["ratio"]) != RATIOS:
            raise ValueError(
                "independent audit found an incomplete ratio curve"
            )
        raw = ordered["c0_prediction"].to_numpy(dtype=np.float64)
        projected = ordered["pava_prediction"].to_numpy(dtype=np.float64)
        reference = estimator.fit_transform(np.asarray(RATIOS), raw)
        max_reference_error = max(
            max_reference_error,
            float(np.max(np.abs(projected - reference))),
        )
        if np.any(np.diff(projected) < -1e-15):
            monotonicity_errors += 1
        was_monotone = bool(np.all(np.diff(raw) >= -1e-15))
        changed = not np.array_equal(raw, projected)
        if changed:
            changed_groups[str(compressor)] += 1
        if was_monotone and changed:
            unchanged_errors += 1
        if not np.isclose(raw.mean(), projected.mean(), rtol=0.0, atol=1e-14):
            mean_errors += 1
        shift_rows.extend(float(value) for value in np.abs(projected - raw))
    shifts = np.asarray(shift_rows, dtype=np.float64)
    nonzero = shifts[shifts > 0.0]
    return {
        "max_sklearn_reference_error": max_reference_error,
        "projected_monotonicity_errors": monotonicity_errors,
        "already_monotone_changed_errors": unchanged_errors,
        "vector_mean_errors": mean_errors,
        "changed_groups": changed_groups,
        "changed_rows": len(nonzero),
        "maximum_absolute_prediction_shift": float(np.max(shifts)),
        "mean_absolute_shift_on_changed_rows": (
            float(np.mean(nonzero)) if len(nonzero) else 0.0
        ),
    }


def prompt_ratio_losses(
    frame: pd.DataFrame,
    prediction: str,
    prompts: tuple[str, ...],
) -> tuple[NDArray[np.float64], dict[float, NDArray[np.float64]]]:
    error = (
        frame["dq"].to_numpy(dtype=np.float64)
        - frame[prediction].to_numpy(dtype=np.float64)
    ) ** 2
    work = frame.assign(_independent_squared_error=error)
    table = (
        work.groupby(["prompt_id", "ratio"], observed=True, sort=True)[
            "_independent_squared_error"
        ]
        .mean()
        .unstack("ratio")
        .reindex(index=list(prompts), columns=list(RATIOS))
    )
    values = table.to_numpy(dtype=np.float64)
    if (
        values.shape != (PROMPTS, len(RATIOS))
        or not np.isfinite(values).all()
    ):
        raise ValueError("independent prompt-ratio loss table is incomplete")
    by_ratio = {
        ratio: values[:, index].copy() for index, ratio in enumerate(RATIOS)
    }
    return np.mean(values, axis=1), by_ratio


def max_t_intervals(
    estimates: Mapping[str, float],
    bootstrap: Mapping[str, NDArray[np.float64]],
    direction: str,
) -> dict[str, Any]:
    claims = tuple(sorted(estimates))
    point = np.asarray(
        [estimates[claim] for claim in claims], dtype=np.float64
    )
    matrix = np.column_stack([bootstrap[claim] for claim in claims])
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
    intervals: dict[str, dict[str, float]] = {}
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
        "multiplicity": f"simultaneous max-T over {len(claims)} claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def evaluate_independently(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, NDArray[np.float64]]]:
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
        raise ValueError(
            "independent audit found inconsistent prompt rosters"
        )
    prompts = next(iter(prompt_sets.values()))
    if len(prompts) != PROMPTS:
        raise ValueError(
            "independent audit prompt count differs from contract"
        )

    draws = np.random.default_rng(SEED).integers(
        0, PROMPTS, size=(RESAMPLES, PROMPTS)
    )
    metrics: dict[str, Any] = {}
    estimates: dict[str, float] = {}
    bootstrap: dict[str, NDArray[np.float64]] = {}
    loss_vectors: dict[str, NDArray[np.float64]] = {}
    for compressor in COMPRESSORS:
        group = frame[frame["compressor"] == compressor]
        c0, c0_ratio = prompt_ratio_losses(group, "c0_prediction", prompts)
        pava, pava_ratio = prompt_ratio_losses(
            group, "pava_prediction", prompts
        )
        c0_loss = float(np.mean(c0))
        pava_loss = float(np.mean(pava))
        skill = 1.0 - pava_loss / c0_loss
        estimates[compressor] = skill
        bootstrap[compressor] = 1.0 - (
            np.mean(pava[draws], axis=1) / np.mean(c0[draws], axis=1)
        )
        loss_vectors[f"{compressor}:c0"] = c0
        loss_vectors[f"{compressor}:pava"] = pava
        metrics[compressor] = {
            "c0_loss": c0_loss,
            "pava_loss": pava_loss,
            "skill": skill,
            "per_ratio": {
                str(ratio): {
                    "c0_loss": float(np.mean(c0_ratio[ratio])),
                    "pava_loss": float(np.mean(pava_ratio[ratio])),
                    "skill": 1.0
                    - float(np.mean(pava_ratio[ratio]))
                    / float(np.mean(c0_ratio[ratio])),
                }
                for ratio in RATIOS
            },
        }
    inference = {
        "lower": max_t_intervals(estimates, bootstrap, "lower"),
        "minimum_worthwhile_skill": MINIMUM_SKILL,
        "upper": max_t_intervals(estimates, bootstrap, "upper"),
    }
    return metrics, inference, {**bootstrap, **loss_vectors}


def compare_nested(
    expected: Any,
    observed: Any,
    *,
    path: str = "root",
) -> tuple[list[str], float]:
    errors: list[str] = []
    max_difference = 0.0
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            errors.append(f"{path}: key mismatch")
            return errors, max_difference
        for key in expected:
            child_errors, child_difference = compare_nested(
                expected[key], observed[key], path=f"{path}.{key}"
            )
            errors.extend(child_errors)
            max_difference = max(max_difference, child_difference)
        return errors, max_difference
    if isinstance(expected, list) and isinstance(observed, list):
        if len(expected) != len(observed):
            errors.append(f"{path}: list length mismatch")
            return errors, max_difference
        for index, (left, right) in enumerate(
            zip(expected, observed, strict=True)
        ):
            child_errors, child_difference = compare_nested(
                left, right, path=f"{path}[{index}]"
            )
            errors.extend(child_errors)
            max_difference = max(max_difference, child_difference)
        return errors, max_difference
    if (
        isinstance(expected, int | float)
        and not isinstance(expected, bool)
        and isinstance(observed, int | float)
        and not isinstance(observed, bool)
    ):
        difference = abs(float(expected) - float(observed))
        scale = max(1.0, abs(float(expected)), abs(float(observed)))
        if difference > 1e-12 * scale:
            errors.append(
                f"{path}: numeric mismatch {expected!r} != {observed!r}"
            )
        return errors, difference
    if expected != observed:
        errors.append(f"{path}: value mismatch {expected!r} != {observed!r}")
    return errors, max_difference


def adversarial_sensitivity(
    frame: pd.DataFrame,
    samples: Mapping[str, NDArray[np.float64]],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_fold = (
        frame.groupby("prompt_id", sort=True)["fold"]
        .first()
        .reindex(sorted(str(value) for value in frame["prompt_id"].unique()))
    )
    result: dict[str, Any] = {}
    for compressor in COMPRESSORS:
        c0 = samples[f"{compressor}:c0"]
        pava = samples[f"{compressor}:pava"]
        bootstrap = samples[compressor]
        c0_total = float(np.sum(c0))
        pava_total = float(np.sum(pava))
        leave_one_out = 1.0 - (pava_total - pava) / (c0_total - c0)
        prompt_delta = c0 - pava
        fold_skill = {}
        for fold in sorted(int(value) for value in prompt_fold.unique()):
            mask = prompt_fold.to_numpy(dtype=np.int64) == fold
            fold_skill[str(fold)] = float(
                1.0 - np.mean(pava[mask]) / np.mean(c0[mask])
            )
        upper = float(
            report["inference"]["upper"]["intervals"][compressor]["upper"]
        )
        result[compressor] = {
            "reported_simultaneous_upper": upper,
            "margin_from_worthwhile_threshold": MINIMUM_SKILL - upper,
            "bootstrap_minimum": float(np.min(bootstrap)),
            "bootstrap_maximum": float(np.max(bootstrap)),
            "bootstrap_draws_at_or_above_threshold": int(
                np.sum(bootstrap >= MINIMUM_SKILL)
            ),
            "leave_one_prompt_out_minimum": float(np.min(leave_one_out)),
            "leave_one_prompt_out_maximum": float(np.max(leave_one_out)),
            "prompt_loss_improvement_fraction": float(
                np.mean(prompt_delta > 0.0)
            ),
            "prompt_loss_worsening_fraction": float(
                np.mean(prompt_delta < 0.0)
            ),
            "prompt_loss_delta_quantiles": {
                str(quantile): float(np.quantile(prompt_delta, quantile))
                for quantile in (0.0, 0.25, 0.5, 0.75, 1.0)
            },
            "fold_skills": fold_skill,
            "all_four_per_ratio_skills_positive": all(
                float(values["skill"]) > 0.0
                for values in report["metrics"][compressor][
                    "per_ratio"
                ].values()
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H2 audit: {args.output}"
        )
    manifest = load_json(args.projection_manifest)
    protocol = load_json(args.protocol_lock)
    score_lock = load_json(args.score_lock)
    report = load_json(args.report)

    hashes = {
        "source_oof_sha256": sha256_file(args.source_oof),
        "projection_sha256": sha256_file(args.projection),
        "projection_manifest_sha256": sha256_file(args.projection_manifest),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "scoring_lock_sha256": sha256_file(args.score_lock),
        "report_sha256": sha256_file(args.report),
        "project_script_sha256": sha256_file(args.project_script),
        "score_script_sha256": sha256_file(args.score_script),
    }
    artifact_checks = {
        "source_matches_protocol": hashes["source_oof_sha256"]
        == protocol["source_oof_sha256"],
        "source_matches_score_lock": hashes["source_oof_sha256"]
        == score_lock["source_oof_sha256"],
        "source_matches_report": hashes["source_oof_sha256"]
        == report["provenance"]["source_oof_sha256"],
        "projection_matches_manifest": hashes["projection_sha256"]
        == manifest["projection_sha256"],
        "projection_matches_score_lock": hashes["projection_sha256"]
        == score_lock["projection_sha256"],
        "projection_matches_report": hashes["projection_sha256"]
        == report["provenance"]["projection_sha256"],
        "manifest_matches_score_lock": hashes["projection_manifest_sha256"]
        == score_lock["projection_manifest_sha256"],
        "manifest_matches_report": hashes["projection_manifest_sha256"]
        == report["provenance"]["projection_manifest_sha256"],
        "protocol_matches_score_lock": hashes["protocol_lock_sha256"]
        == score_lock["protocol_lock_sha256"],
        "protocol_matches_report": hashes["protocol_lock_sha256"]
        == report["provenance"]["protocol_lock_sha256"],
        "score_lock_matches_report": hashes["scoring_lock_sha256"]
        == report["provenance"]["scoring_lock_sha256"],
        "project_script_matches_lock": hashes["project_script_sha256"]
        == score_lock["implementation_sha256"][
            "scripts/project_magnitude_e1_h2.py"
        ],
        "score_script_matches_lock": hashes["score_script_sha256"]
        == score_lock["implementation_sha256"][
            "scripts/score_magnitude_e1_h2.py"
        ],
        "score_script_matches_report": hashes["score_script_sha256"]
        == report["provenance"]["implementation_sha256"],
        "projection_manifest_is_label_blind": manifest[
            "outcome_or_label_columns_loaded"
        ]
        == [],
        "confirmation_remained_sealed": report["provenance"][
            "confirmation_status"
        ]
        == "sealed_not_run",
    }

    source, projection = read_inputs(args.source_oof, args.projection)
    frame = join_inputs(source, projection)
    roster_hash = prompt_roster_sha256(frame["prompt_id"])
    data_checks = {
        "rows_exact": len(frame) == TOTAL_ROWS,
        "prompts_exact": frame["prompt_id"].nunique() == PROMPTS,
        "compressors_exact": set(frame["compressor"]) == set(COMPRESSORS),
        "ratios_exact": {float(value) for value in frame["ratio"]}
        == set(RATIOS),
        "c0_source_parity_exact": bool(
            np.array_equal(
                frame["m2_prediction"].to_numpy(dtype=np.float64),
                frame["c0_prediction"].to_numpy(dtype=np.float64),
            )
        ),
        "prompt_roster_matches_manifest": roster_hash
        == manifest["development_prompt_ids_sha256"],
        "fold_assignment_one_per_prompt": bool(
            (frame.groupby("prompt_id")["fold"].nunique() == 1).all()
        ),
    }

    projection_audit = verify_projection_with_sklearn(frame)
    projection_checks = {
        "sklearn_reference_matches": projection_audit[
            "max_sklearn_reference_error"
        ]
        <= 1e-12,
        "all_outputs_monotone": projection_audit[
            "projected_monotonicity_errors"
        ]
        == 0,
        "monotone_inputs_unchanged": projection_audit[
            "already_monotone_changed_errors"
        ]
        == 0,
        "all_vector_means_preserved": projection_audit["vector_mean_errors"]
        == 0,
        "changed_group_counts_match_manifest": projection_audit[
            "changed_groups"
        ]
        == manifest["invariants"]["changed_groups"],
    }

    metrics, inference, samples = evaluate_independently(frame)
    metric_errors, metric_max_difference = compare_nested(
        report["metrics"], metrics, path="metrics"
    )
    inference_errors, inference_max_difference = compare_nested(
        report["inference"], inference, path="inference"
    )
    reproduction_checks = {
        "reported_metrics_reproduced": not metric_errors,
        "reported_inference_reproduced": not inference_errors,
        "reported_failure_decision_matches_lock": report["decision"]["status"]
        == "failure"
        and all(
            float(inference["upper"]["intervals"][compressor]["upper"])
            <= MINIMUM_SKILL
            for compressor in COMPRESSORS
        ),
    }

    sensitivity = adversarial_sensitivity(frame, samples, report)
    sensitivity_checks = {
        "no_bootstrap_draw_reaches_threshold": all(
            values["bootstrap_draws_at_or_above_threshold"] == 0
            for values in sensitivity.values()
        ),
        "no_leave_one_prompt_out_result_reaches_threshold": all(
            values["leave_one_prompt_out_maximum"] < MINIMUM_SKILL
            for values in sensitivity.values()
        ),
        "all_simultaneous_upper_bounds_below_threshold": all(
            values["reported_simultaneous_upper"] <= MINIMUM_SKILL
            for values in sensitivity.values()
        ),
    }

    checks = {
        "artifacts": artifact_checks,
        "data": data_checks,
        "projection": projection_checks,
        "reproduction": reproduction_checks,
        "adversarial_sensitivity": sensitivity_checks,
    }
    passed = all(
        bool(value)
        for section in checks.values()
        for value in section.values()
    )
    conclusion = (
        "robust_failure_retire_hard_monotone_projection"
        if passed
        else "invalid_or_nonrobust_return_to_understand"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_result_audit_complete",
        "pass": passed,
        "conclusion": conclusion,
        "hashes": hashes,
        "checks": checks,
        "projection_audit": projection_audit,
        "reproduction": {
            "metric_errors": metric_errors,
            "metric_max_absolute_difference": metric_max_difference,
            "inference_errors": inference_errors,
            "inference_max_absolute_difference": inference_max_difference,
        },
        "adversarial_sensitivity": sensitivity,
        "scientific_interpretation": {
            "supported": (
                "The fixed hard monotone correction is materially futile "
                "on this frozen development OOF table."
            ),
            "not_supported": [
                "ratio structure is useless to predictive models",
                "future-prompt generalization",
                "confirmatory evidence",
                "M3 reinterpretation",
            ],
            "next_action": (
                "Retire H2 without variants and return to the preregistered "
                "H1 prompt-semantics strategy."
            ),
        },
        "confirmation_status": "sealed_not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "h2_independent_audit_complete",
                "pass": passed,
                "conclusion": conclusion,
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
