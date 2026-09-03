"""Fit the frozen development pipeline once and score confirmation once."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from evaluate_current_state_damage_dev import (
    prompt_values,
    simultaneous_bound,
    stratified_draws,
    subgroup_differences,
)
from train_current_state_damage_tabular import (
    BAND_COLUMNS,
    DURATIONS,
    HORIZONS,
    load_data,
    loss_weights,
    weighted_mean,
)

SCHEMA_VERSION = "herald.current_state_damage_confirmation.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--confirmation-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--prefit-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def fit_and_predict(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    features: list[str],
    label: str,
    weights: np.ndarray,
    rounds: int,
    parameters: dict[str, Any],
    model_path: Path,
) -> np.ndarray:
    train = development[label].notna().to_numpy()
    test = confirmation[label].notna().to_numpy()
    train_matrix = xgb.QuantileDMatrix(
        development.loc[train, features],
        label=development.loc[train, label],
        weight=weights[train],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )
    model = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    test_matrix = xgb.QuantileDMatrix(
        confirmation.loc[test, features],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
        ref=train_matrix,
    )
    output = np.full(len(confirmation), np.nan, dtype=np.float32)
    output[test] = np.maximum(model.predict(test_matrix), 0.0).astype(
        np.float32
    )
    model.save_model(model_path)
    del model, test_matrix, train_matrix
    gc.collect()
    return output


def baseline_predictions(
    development: pd.DataFrame,
    confirmation: pd.DataFrame,
    label: str,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eligible = development[label].notna().to_numpy()
    test = confirmation[label].notna().to_numpy()
    values = development[label].to_numpy(dtype=np.float64)
    global_output = np.full(len(confirmation), np.nan, dtype=np.float32)
    global_output[test] = weighted_mean(values[eligible], weights[eligible])
    training = development.loc[eligible, ["run_id", "action_code"]].copy()
    training["value"] = values[eligible]
    run_sizes = training.groupby("run_id", sort=False)["run_id"].transform(
        "size"
    )
    training["cell_weight"] = 1.0 / run_sizes.to_numpy(dtype=np.float64)
    training["weighted_value"] = training["value"] * training["cell_weight"]
    grouped = training.groupby("action_code", observed=True, sort=False)
    means = grouped["weighted_value"].sum() / grouped["cell_weight"].sum()
    action_output = np.full(len(confirmation), np.nan, dtype=np.float32)
    action_output[test] = (
        confirmation.loc[test, "action_code"]
        .map(means)
        .to_numpy(dtype=np.float32)
    )
    return global_output, action_output


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to rerun confirmation {args.output_root}"
        )
    protocol = load_json(args.protocol_lock)
    prefit = load_json(args.prefit_lock)
    open_report = load_json(args.confirmation_root / "open_report.json")
    if open_report.get("pass") is not True:
        raise ValueError("confirmation opening audit failed")
    if open_report.get("prefit_lock_sha256") != sha256_file(args.prefit_lock):
        raise ValueError("confirmation opening prefit mismatch")
    presses = list(protocol["dataset"]["presses"])
    frozen_features = list(prefit["training"]["feature_order"])
    action_features = list(prefit["hard_comparator"]["feature_order"])
    derived = {"press_category", "ratio_category", "action_category"}
    sensors = [
        feature
        for feature in frozen_features
        if feature not in derived | {"compression_ratio", "log_token_clock"}
    ]
    columns = [
        "run_id",
        "prompt_id",
        "task",
        "press",
        "compression_ratio",
        "fold",
        "token_pos",
        "log_token_clock",
        *sensors,
        *BAND_COLUMNS,
    ]
    development = load_data(args.development_root, columns, presses)
    confirmation = load_data(
        args.confirmation_root / "tokens", columns, presses
    )
    if set(confirmation["fold"].unique()) != {-1}:
        raise ValueError("confirmation materialization identity changed")
    for feature in sensors:
        development[feature] = development[feature].astype(np.float32)
        confirmation[feature] = confirmation[feature].astype(np.float32)
    weights_development = [
        loss_weights(development, label) for label in BAND_COLUMNS
    ]
    weights_confirmation = [
        loss_weights(confirmation, label) for label in BAND_COLUMNS
    ]
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    parameters = dict(prefit["training"]["parameters"])
    parameters["device"] = "cuda"
    predictions = {
        name: np.full(
            (len(confirmation), len(BAND_COLUMNS)), np.nan, dtype=np.float32
        )
        for name in (
            "global_mean",
            "action_only",
            "action_clock",
            "causal_xgb",
        )
    }
    for index, label in enumerate(BAND_COLUMNS):
        global_prediction, action_prediction = baseline_predictions(
            development,
            confirmation,
            label,
            weights_development[index],
        )
        predictions["global_mean"][:, index] = global_prediction
        predictions["action_only"][:, index] = action_prediction
        for candidate, features, round_map in (
            (
                "action_clock",
                action_features,
                prefit["hard_comparator"]["num_boost_round"],
            ),
            (
                "causal_xgb",
                frozen_features,
                prefit["training"]["num_boost_round"],
            ),
        ):
            predictions[candidate][:, index] = fit_and_predict(
                development,
                confirmation,
                features,
                label,
                weights_development[index],
                int(round_map[label]),
                parameters,
                model_root / f"{candidate}.{label}.json",
            )
            print(f"fit and predicted {candidate} {label}", flush=True)
    expected = confirmation[list(BAND_COLUMNS)].notna().to_numpy()
    for name, values in predictions.items():
        if not np.array_equal(np.isfinite(values), expected):
            raise ValueError(f"incomplete confirmation prediction for {name}")
    scales = np.asarray(
        prefit["training"]["calibration_scales"], dtype=np.float64
    )
    calibrated = predictions["causal_xgb"] * scales.astype(np.float32)
    truth_rates = confirmation[list(BAND_COLUMNS)].to_numpy(dtype=np.float64)
    truth = np.cumsum(truth_rates * DURATIONS, axis=1)
    cumulative_predictions = {
        name: np.cumsum(values.astype(np.float64) * DURATIONS, axis=1)
        for name, values in {
            **predictions,
            "causal_xgb_calibrated": calibrated,
        }.items()
    }
    losses: dict[str, list[float]] = {
        name: [] for name in cumulative_predictions
    }
    row_losses: dict[str, np.ndarray] = {
        name: np.full_like(truth, np.nan) for name in cumulative_predictions
    }
    for index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth[:, index])
        for name, cumulative_values in cumulative_predictions.items():
            squared = np.square(
                (truth[eligible, index] - cumulative_values[eligible, index])
                / horizon
            )
            row_losses[name][eligible, index] = squared
            losses[name].append(
                float(
                    np.average(
                        squared,
                        weights=weights_confirmation[index][eligible],
                    )
                )
            )
    candidate_name = "causal_xgb_calibrated"
    comparator_name = "action_clock"
    difference = row_losses[candidate_name] - row_losses[comparator_name]
    reference: pd.DataFrame | None = None
    prompt_columns: list[np.ndarray] = []
    for index in range(len(HORIZONS)):
        eligible = np.isfinite(truth[:, index])
        prompt = prompt_values(
            confirmation,
            difference[:, index],
            weights_confirmation[index],
            eligible,
        )
        if reference is None:
            reference = prompt[["prompt_id", "task"]]
        aligned = prompt.set_index("prompt_id").loc[reference["prompt_id"]]
        prompt_columns.append(aligned["value"].to_numpy(dtype=np.float64))
    if reference is None:
        raise ValueError("no confirmation prompt results")
    prompt_matrix = np.column_stack(prompt_columns)
    prompt_tasks = reference["task"].to_numpy()
    draws = stratified_draws(
        prompt_tasks,
        int(protocol["inference"]["resamples"]),
        int(protocol["inference"]["seed"]),
    )
    superiority = simultaneous_bound(
        prompt_matrix, prompt_tasks, draws, "upper"
    )
    subgroups = subgroup_differences(
        confirmation,
        difference,
        weights_confirmation,
    )
    residual_columns: list[np.ndarray] = []
    slope_columns: list[np.ndarray] = []
    for index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth[:, index])
        cumulative_prediction = cumulative_predictions[candidate_name][
            :, index
        ]
        residual = (truth[:, index] - cumulative_prediction) / horizon
        residual_prompt = (
            prompt_values(
                confirmation,
                residual,
                weights_confirmation[index],
                eligible,
            )
            .set_index("prompt_id")
            .loc[reference["prompt_id"]]
        )
        residual_columns.append(
            residual_prompt["value"].to_numpy(dtype=np.float64)
        )
        numerator = (
            prompt_values(
                confirmation,
                truth[:, index] * cumulative_prediction,
                weights_confirmation[index],
                eligible,
            )
            .set_index("prompt_id")
            .loc[reference["prompt_id"]]["value"]
            .to_numpy(dtype=np.float64)
        )
        denominator = (
            prompt_values(
                confirmation,
                np.square(cumulative_prediction),
                weights_confirmation[index],
                eligible,
            )
            .set_index("prompt_id")
            .loc[reference["prompt_id"]]["value"]
            .to_numpy(dtype=np.float64)
        )
        slope_columns.append(numerator / denominator)
    residual_bound = simultaneous_bound(
        np.column_stack(residual_columns), prompt_tasks, draws, "two-sided"
    )
    slope_bound = simultaneous_bound(
        np.column_stack(slope_columns), prompt_tasks, draws, "lower"
    )
    superiority_pass = all(value < 0 for value in superiority["upper"])
    subgroup_pass = all(
        value < 0
        for dimension in subgroups.values()
        for group in dimension.values()
        for value in group
    )
    calibration_pass = all(
        lower <= 0 <= upper
        for lower, upper in zip(
            residual_bound["lower"], residual_bound["upper"], strict=True
        )
    ) and all(value > 0 for value in slope_bound["lower"])
    prediction_output = confirmation[
        [
            "run_id",
            "prompt_id",
            "task",
            "press",
            "compression_ratio",
            "token_pos",
            *BAND_COLUMNS,
        ]
    ].copy()
    for index, label in enumerate(BAND_COLUMNS):
        prediction_output[f"weight_{label}"] = weights_confirmation[index]
        for name, values in predictions.items():
            prediction_output[f"pred_{name}_{label}"] = values[:, index]
    prediction_path = args.output_root / "predictions.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pandas(prediction_output, preserve_index=False),
        prediction_path,
        compression="zstd",
        use_dictionary=("run_id", "prompt_id", "task", "press"),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "confirmation_scored_once_no_tuning",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "prefit_lock_sha256": sha256_file(args.prefit_lock),
        "open_report_sha256": sha256_file(
            args.confirmation_root / "open_report.json"
        ),
        "horizon_normalized_mse": {
            name: dict(zip(map(str, HORIZONS), values, strict=True))
            for name, values in losses.items()
        },
        "simultaneous_superiority": superiority,
        "subgroup_point_differences": subgroups,
        "calibration": {
            "mean_normalized_residual": residual_bound,
            "slope": slope_bound,
        },
        "gates": {
            "superiority": superiority_pass,
            "all_task_press_ratio_points_negative": subgroup_pass,
            "calibration": calibration_pass,
        },
        "pass": superiority_pass and subgroup_pass and calibration_pass,
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_attempts": 1,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
