"""Train selected XGBoost pipeline in leave-one-task-out stress tests."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from evaluate_current_state_damage_dev import (
    prompt_values,
    simultaneous_bound,
    stratified_draws,
)
from train_current_state_damage_tabular import (
    BAND_COLUMNS,
    DURATIONS,
    HORIZONS,
    cumulative_loss,
    load_data,
    loss_weights,
    weighted_mean,
    xgb_parameters,
)

SCHEMA_VERSION = "herald.current_state_damage_loto.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
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


def fit_task_model(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    weights: np.ndarray,
    heldout_task: str,
    parameters: dict[str, Any],
    lock: dict[str, Any],
    model_path: Path,
) -> tuple[np.ndarray, int]:
    task = frame["task"].to_numpy()
    fold = frame["fold"].to_numpy(dtype=np.int8)
    eligible = frame[label].notna().to_numpy()
    validation_fold = 0
    inner_train = (
        eligible & (task != heldout_task) & (fold != validation_fold)
    )
    inner_valid = (
        eligible & (task != heldout_task) & (fold == validation_fold)
    )
    refit_train = eligible & (task != heldout_task)
    test = eligible & (task == heldout_task)
    maximum_rounds = int(
        lock["models"]["first"]["parameters"]["num_boost_round_max"]
    )
    early_stopping = int(
        lock["models"]["first"]["parameters"]["early_stopping_rounds"]
    )
    train_matrix = xgb.QuantileDMatrix(
        frame.loc[inner_train, features],
        label=frame.loc[inner_train, label],
        weight=weights[inner_train],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )
    valid_matrix = xgb.QuantileDMatrix(
        frame.loc[inner_valid, features],
        label=frame.loc[inner_valid, label],
        weight=weights[inner_valid],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
        ref=train_matrix,
    )
    selector = xgb.train(
        parameters,
        train_matrix,
        num_boost_round=maximum_rounds,
        evals=[(valid_matrix, "validation")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    rounds = int(selector.best_iteration) + 1
    del selector, valid_matrix, train_matrix
    gc.collect()
    refit_matrix = xgb.QuantileDMatrix(
        frame.loc[refit_train, features],
        label=frame.loc[refit_train, label],
        weight=weights[refit_train],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )
    model = xgb.train(
        parameters, refit_matrix, num_boost_round=rounds, verbose_eval=False
    )
    test_matrix = xgb.QuantileDMatrix(
        frame.loc[test, features],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
        ref=refit_matrix,
    )
    prediction = np.maximum(model.predict(test_matrix), 0.0).astype(
        np.float32
    )
    model.save_model(model_path)
    del model, test_matrix, refit_matrix
    gc.collect()
    return prediction, rounds


def fit_task_baselines(
    frame: pd.DataFrame,
    label: str,
    weights: np.ndarray,
    heldout_task: str,
    global_output: np.ndarray,
    action_output: np.ndarray,
) -> None:
    eligible = frame[label].notna().to_numpy()
    task = frame["task"].to_numpy()
    train = eligible & (task != heldout_task)
    test = eligible & (task == heldout_task)
    values = frame[label].to_numpy(dtype=np.float64)
    global_output[test] = weighted_mean(values[train], weights[train])
    training = frame.loc[train, ["run_id", "action_code"]].copy()
    training["value"] = values[train]
    run_sizes = training.groupby("run_id", sort=False)["run_id"].transform(
        "size"
    )
    training["cell_weight"] = 1.0 / run_sizes.to_numpy(dtype=np.float64)
    training["weighted_value"] = training["value"] * training["cell_weight"]
    grouped = training.groupby("action_code", observed=True, sort=False)
    means = grouped["weighted_value"].sum() / grouped["cell_weight"].sum()
    action_output[test] = (
        frame.loc[test, "action_code"].map(means).to_numpy(dtype=np.float32)
    )


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    lock = load_json(args.protocol_lock)
    audit = load_json(args.feature_audit)
    if audit.get("pass") is not True:
        raise ValueError("feature audit failed")
    presses = list(lock["dataset"]["presses"])
    sensors = [
        column
        for column in audit["predictor_columns"]
        if column not in {"press", "compression_ratio", "log_token_clock"}
    ]
    load_columns = [
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
    frame = load_data(args.data_root, load_columns, presses)
    for column in sensors:
        frame[column] = frame[column].astype(np.float32)
    action_features = [
        "press_category",
        "ratio_category",
        "compression_ratio",
        "action_category",
        "log_token_clock",
    ]
    causal_features = [*action_features, *sensors]
    weights = [loss_weights(frame, label) for label in BAND_COLUMNS]
    predictions = {
        name: np.full(
            (len(frame), len(BAND_COLUMNS)), np.nan, dtype=np.float32
        )
        for name in (
            "global_mean",
            "action_only",
            "action_clock",
            "causal_xgb",
        )
    }
    tasks = list(lock["dataset"]["tasks"])
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    parameters = xgb_parameters(lock, "cuda")
    rounds: dict[str, int] = {}
    for task in tasks:
        task_rows = frame["task"].to_numpy() == task
        for band_index, label in enumerate(BAND_COLUMNS):
            fit_task_baselines(
                frame,
                label,
                weights[band_index],
                task,
                predictions["global_mean"][:, band_index],
                predictions["action_only"][:, band_index],
            )
            eligible_test = task_rows & frame[label].notna().to_numpy()
            for candidate, features in (
                ("action_clock", action_features),
                ("causal_xgb", causal_features),
            ):
                predicted, selected_rounds = fit_task_model(
                    frame,
                    features,
                    label,
                    weights[band_index],
                    task,
                    parameters,
                    lock,
                    model_root / f"{candidate}.{label}.holdout-{task}.json",
                )
                predictions[candidate][eligible_test, band_index] = predicted
                rounds[f"{candidate}.{label}.holdout-{task}"] = (
                    selected_rounds
                )
                print(
                    f"completed {candidate} {label} holdout={task} "
                    f"rounds={selected_rounds}",
                    flush=True,
                )
    expected = frame[list(BAND_COLUMNS)].notna().to_numpy()
    for name, prediction in predictions.items():
        if not np.array_equal(np.isfinite(prediction), expected):
            raise ValueError(f"incomplete LOTO predictions for {name}")
    losses = {
        name: cumulative_loss(frame, weights, prediction)
        for name, prediction in predictions.items()
    }
    hard_models = [
        min(
            ("global_mean", "action_only", "action_clock"),
            key=lambda name: losses[name][index],
        )
        for index in range(len(HORIZONS))
    ]
    truth = np.cumsum(
        frame[list(BAND_COLUMNS)].to_numpy(dtype=np.float64) * DURATIONS,
        axis=1,
    )
    cumulative_predictions = {
        name: np.cumsum(prediction.astype(np.float64) * DURATIONS, axis=1)
        for name, prediction in predictions.items()
    }
    differences: list[np.ndarray] = []
    prompt_ids: list[str] | None = None
    prompt_tasks: np.ndarray | None = None
    task_points: dict[str, list[float]] = {task: [] for task in tasks}
    for index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth[:, index])
        candidate_loss = np.square(
            (truth[:, index] - cumulative_predictions["causal_xgb"][:, index])
            / horizon
        )
        comparator = hard_models[index]
        comparator_loss = np.square(
            (truth[:, index] - cumulative_predictions[comparator][:, index])
            / horizon
        )
        difference = candidate_loss - comparator_loss
        prompt = prompt_values(frame, difference, weights[index], eligible)
        if prompt_ids is None:
            prompt_ids = prompt["prompt_id"].tolist()
            prompt_tasks = prompt["task"].to_numpy()
        aligned = prompt.set_index("prompt_id").loc[prompt_ids]
        differences.append(aligned["value"].to_numpy(dtype=np.float64))
        for task in tasks:
            selected = aligned["task"].to_numpy() == task
            task_points[task].append(
                float(aligned.loc[selected, "value"].mean())
            )
    if prompt_ids is None or prompt_tasks is None:
        raise ValueError("no prompt-level LOTO results")
    matrix = np.column_stack(differences)
    draws = stratified_draws(
        prompt_tasks,
        int(lock["inference"]["resamples"]),
        int(lock["inference"]["seed"]),
    )
    bound = simultaneous_bound(matrix, prompt_tasks, draws, "upper")
    pass_gate = all(value < 0 for value in bound["upper"]) and all(
        value < 0 for values in task_points.values() for value in values
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_loto_complete_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "feature_audit_sha256": sha256_file(args.feature_audit),
        "validation_fold_within_training_tasks": 0,
        "selected_rounds": rounds,
        "hard_comparator_by_horizon": dict(
            zip(map(str, HORIZONS), hard_models, strict=True)
        ),
        "pooled_simultaneous_upper": bound,
        "heldout_task_point_differences": task_points,
        "pass": pass_gate,
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
