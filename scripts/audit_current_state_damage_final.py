"""Independently audit the immutable confirmation horizon evidence."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BAND_COLUMNS = (
    "band_rate_0_5",
    "band_rate_5_10",
    "band_rate_10_25",
    "band_rate_25_50",
)
HORIZONS = (5, 10, 25, 50)
DURATIONS = np.asarray((5.0, 5.0, 15.0, 25.0), dtype=np.float64)
SCHEMA_VERSION = "herald.current_state_damage_final_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--prefit-lock", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-open-root", type=Path, required=True)
    parser.add_argument(
        "--confirmation-result-root", type=Path, required=True
    )
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


def prompt_aggregate(
    frame: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
) -> pd.DataFrame:
    selected = pd.DataFrame(
        {
            "prompt_id": frame.loc[eligible, "prompt_id"].to_numpy(),
            "task": frame.loc[eligible, "task"].to_numpy(),
            "weighted": values[eligible] * weights[eligible],
            "weight": weights[eligible],
        }
    )
    grouped = selected.groupby("prompt_id", sort=True)
    return pd.DataFrame(
        {
            "value": grouped["weighted"].sum() / grouped["weight"].sum(),
            "task": grouped["task"].first(),
        }
    ).reset_index()


def independent_upper_bound(
    matrix: np.ndarray, tasks: np.ndarray, resamples: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    point = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    maxima = np.empty(resamples, dtype=np.float64)
    for bootstrap_index in range(resamples):
        indices = np.concatenate(
            [
                rng.choice(group, size=len(group), replace=True)
                for group in groups
            ]
        )
        sample = matrix[indices]
        sample_mean = sample.mean(axis=0)
        sample_error = sample.std(axis=0, ddof=1) / np.sqrt(len(sample))
        statistic = np.divide(
            sample_mean - point,
            sample_error,
            out=np.zeros_like(point),
            where=sample_error > 0,
        )
        maxima[bootstrap_index] = statistic.max()
    critical = float(np.quantile(maxima, 0.95))
    upper = point + critical * standard_error
    return {
        "point": point.tolist(),
        "upper": upper.tolist(),
        "critical": critical,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    protocol = load_json(args.protocol_lock)
    prefit = load_json(args.prefit_lock)
    opened = load_json(args.confirmation_open_root / "open_report.json")
    result = load_json(args.confirmation_result_root / "report.json")
    development_prompts = {
        row["prompt_id"] for row in load_json(args.development_manifest)
    }
    confirmation_prompts = {
        row["prompt_id"] for row in load_json(args.confirmation_manifest)
    }
    if development_prompts & confirmation_prompts:
        raise ValueError("development/confirmation prompt overlap")
    if len(development_prompts) != 508 or len(confirmation_prompts) != 256:
        raise ValueError("prompt manifest cardinality changed")
    if prefit["confirmation"]["prompt_manifest_sha256"] != sha256_file(
        args.confirmation_manifest
    ):
        raise ValueError("prefit confirmation manifest mismatch")
    if opened.get("prefit_lock_sha256") != sha256_file(args.prefit_lock):
        raise ValueError("opening prefit hash mismatch")
    if result.get("prefit_lock_sha256") != sha256_file(args.prefit_lock):
        raise ValueError("result prefit hash mismatch")
    if result.get("open_report_sha256") != sha256_file(
        args.confirmation_open_root / "open_report.json"
    ):
        raise ValueError("result opening hash mismatch")
    if result.get("confirmation_attempts") != 1:
        raise ValueError("confirmation attempt count is not one")
    if opened.get("secondary_targets_loaded") != []:
        raise ValueError("secondary target quarantine violated")
    if opened.get("released_aggregate_columns_loaded") != []:
        raise ValueError("released aggregate leakage detected")
    if opened.get("forbidden_outcome_columns_loaded") != []:
        raise ValueError("forbidden outcome leakage detected")

    prediction_path = args.confirmation_result_root / "predictions.parquet"
    if result.get("prediction_sha256") != sha256_file(prediction_path):
        raise ValueError("confirmation prediction hash mismatch")
    frame = (
        ds.dataset(  # type: ignore[no-untyped-call]
            prediction_path, format="parquet"
        )
        .to_table()
        .to_pandas()
    )
    if set(frame["prompt_id"].unique()) != confirmation_prompts:
        raise ValueError("prediction prompt identity mismatch")
    if frame.duplicated(["run_id", "token_pos"]).any():
        raise ValueError("duplicate prediction token identity")

    truth_rates = frame[list(BAND_COLUMNS)].to_numpy(dtype=np.float64)
    weights = [
        frame[f"weight_{label}"].to_numpy(dtype=np.float64)
        for label in BAND_COLUMNS
    ]
    scales = np.asarray(
        prefit["training"]["calibration_scales"], dtype=np.float32
    )
    model_rates = (
        frame[
            [f"pred_causal_xgb_{label}" for label in BAND_COLUMNS]
        ].to_numpy(dtype=np.float32)
        * scales
    ).astype(np.float64)
    comparator_rates = frame[
        [f"pred_action_clock_{label}" for label in BAND_COLUMNS]
    ].to_numpy(dtype=np.float64)
    expected = np.isfinite(truth_rates)
    for name, rates in (
        ("candidate", model_rates),
        ("comparator", comparator_rates),
    ):
        if not np.array_equal(np.isfinite(rates), expected):
            raise ValueError(f"{name} censoring mask mismatch")
        if np.nanmin(rates) < 0:
            raise ValueError(f"{name} has negative band prediction")
    truth = np.cumsum(truth_rates * DURATIONS, axis=1)
    candidate = np.cumsum(model_rates * DURATIONS, axis=1)
    comparator = np.cumsum(comparator_rates * DURATIONS, axis=1)
    for cumulative_values in (candidate, comparator):
        if np.nanmin(np.diff(cumulative_values, axis=1)) < -1e-7:
            raise ValueError("nonmonotone cumulative prediction")

    losses: dict[str, list[float]] = {
        "causal_xgb_calibrated": [],
        "action_clock": [],
    }
    prompt_columns: list[np.ndarray] = []
    reference: pd.DataFrame | None = None
    subgroup_points: dict[str, dict[str, list[float]]] = {}
    row_difference = np.full_like(truth, np.nan)
    for index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth[:, index])
        candidate_loss = np.square(
            (truth[:, index] - candidate[:, index]) / horizon
        )
        comparator_loss = np.square(
            (truth[:, index] - comparator[:, index]) / horizon
        )
        row_difference[eligible, index] = (
            candidate_loss[eligible] - comparator_loss[eligible]
        )
        losses["causal_xgb_calibrated"].append(
            float(
                np.average(
                    candidate_loss[eligible], weights=weights[index][eligible]
                )
            )
        )
        losses["action_clock"].append(
            float(
                np.average(
                    comparator_loss[eligible],
                    weights=weights[index][eligible],
                )
            )
        )
        prompt = prompt_aggregate(
            frame, row_difference[:, index], weights[index], eligible
        )
        if reference is None:
            reference = prompt[["prompt_id", "task"]]
        aligned = prompt.set_index("prompt_id").loc[reference["prompt_id"]]
        prompt_columns.append(aligned["value"].to_numpy(dtype=np.float64))
    if reference is None:
        raise ValueError("no confirmation prompt aggregates")

    for dimension in ("task", "press", "compression_ratio"):
        dimension_points: dict[str, list[float]] = {}
        for group_key, indices in frame.groupby(
            dimension, observed=True
        ).indices.items():
            rows = np.asarray(indices, dtype=np.int64)
            group_values: list[float] = []
            for index in range(len(HORIZONS)):
                eligible_rows = rows[np.isfinite(row_difference[rows, index])]
                group_values.append(
                    float(
                        np.average(
                            row_difference[eligible_rows, index],
                            weights=weights[index][eligible_rows],
                        )
                    )
                )
            dimension_points[str(group_key)] = group_values
        subgroup_points[dimension] = dimension_points
    if not all(
        value < 0
        for dimension in subgroup_points.values()
        for group in dimension.values()
        for value in group
    ):
        raise ValueError("independent subgroup skill audit failed")

    matrix = np.column_stack(prompt_columns)
    independent_bound = independent_upper_bound(
        matrix,
        reference["task"].to_numpy(),
        int(protocol["inference"]["resamples"]),
        int(protocol["inference"]["seed"]),
    )
    if not all(value < 0 for value in independent_bound["upper"]):
        raise ValueError("independent simultaneous superiority audit failed")
    for model, model_losses in losses.items():
        reported = result["horizon_normalized_mse"][model]
        for horizon, value in zip(HORIZONS, model_losses, strict=True):
            if not np.isclose(
                value, reported[str(horizon)], rtol=1e-10, atol=1e-12
            ):
                raise ValueError(
                    f"reported loss mismatch for {model} H={horizon}"
                )
    if result.get("pass") is not True or not all(result["gates"].values()):
        raise ValueError("reported confirmation gate did not pass")

    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_confirmation_evidence_verified",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "prefit_lock_sha256": sha256_file(args.prefit_lock),
        "open_report_sha256": sha256_file(
            args.confirmation_open_root / "open_report.json"
        ),
        "confirmation_report_sha256": sha256_file(
            args.confirmation_result_root / "report.json"
        ),
        "prediction_sha256": sha256_file(prediction_path),
        "prompt_manifests_disjoint": True,
        "confirmation_attempts": 1,
        "prediction_masks_exact": True,
        "predictions_nonnegative_monotone": True,
        "recomputed_horizon_normalized_mse": {
            model: dict(zip(map(str, HORIZONS), model_losses, strict=True))
            for model, model_losses in losses.items()
        },
        "independent_simultaneous_superiority": independent_bound,
        "independent_subgroup_point_differences": subgroup_points,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
