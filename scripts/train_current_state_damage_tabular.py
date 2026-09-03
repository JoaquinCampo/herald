"""Cross-fit frozen action-clock and causal XGBoost damage models."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import xgboost as xgb

SCHEMA_VERSION = "herald.current_state_damage_tabular_oof.v1"
HORIZONS = (5, 10, 25, 50)
BAND_COLUMNS = (
    "band_rate_0_5",
    "band_rate_5_10",
    "band_rate_10_25",
    "band_rate_25_50",
)
DURATIONS = np.asarray((5.0, 5.0, 15.0, 25.0), dtype=np.float64)
CANDIDATES = ("action_clock", "causal_xgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_data(
    root: Path, columns: list[str], presses: list[str]
) -> pd.DataFrame:
    parts = [
        ds.dataset(root / f"{press}.parquet", format="parquet")  # type: ignore[no-untyped-call]
        .to_table(columns=columns)
        .to_pandas()
        for press in presses
    ]
    frame = pd.concat(parts, ignore_index=True)
    del parts
    frame["press_category"] = pd.Categorical(
        frame["press"], categories=presses
    )
    ratios = sorted(
        float(value) for value in frame["compression_ratio"].unique()
    )
    ratio_labels = [str(value) for value in ratios]
    frame["ratio_category"] = pd.Categorical(
        frame["compression_ratio"].astype(str), categories=ratio_labels
    )
    press_codes = frame["press_category"].cat.codes.to_numpy(dtype=np.int16)
    ratio_codes = frame["ratio_category"].cat.codes.to_numpy(dtype=np.int16)
    action_codes = press_codes * len(ratios) + ratio_codes
    frame["action_category"] = pd.Categorical(
        action_codes, categories=range(len(presses) * len(ratios))
    )
    frame["action_code"] = action_codes
    return frame


def loss_weights(frame: pd.DataFrame, label: str) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=np.float32)
    eligible = frame[label].notna()
    selected = frame.loc[eligible, ["run_id", "prompt_id"]]
    tokens_per_run = selected.groupby("run_id", sort=False)[
        "run_id"
    ].transform("size")
    actions_per_prompt = (
        selected.drop_duplicates("run_id")
        .groupby("prompt_id", sort=False)["run_id"]
        .count()
    )
    raw = 1.0 / (
        tokens_per_run.to_numpy(dtype=np.float64)
        * selected["prompt_id"]
        .map(actions_per_prompt)
        .to_numpy(dtype=np.float64)
    )
    raw *= len(raw) / raw.sum()
    weights[eligible.to_numpy()] = raw.astype(np.float32)
    return weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights, dtype=np.float64) / weights.sum())


def fit_baselines(
    frame: pd.DataFrame,
    label: str,
    weights: np.ndarray,
    outer_fold: int,
    output_global: np.ndarray,
    output_action: np.ndarray,
) -> None:
    eligible = frame[label].notna().to_numpy()
    train = eligible & (frame["fold"].to_numpy() != outer_fold)
    test = eligible & (frame["fold"].to_numpy() == outer_fold)
    values = frame[label].to_numpy(dtype=np.float64)
    output_global[test] = weighted_mean(values[train], weights[train])

    training = frame.loc[train, ["run_id", "action_code"]].copy()
    training["value"] = values[train]
    run_sizes = training.groupby("run_id", sort=False)["run_id"].transform(
        "size"
    )
    training["cell_weight"] = 1.0 / run_sizes.to_numpy(dtype=np.float64)
    training["weighted_value"] = training["value"] * training["cell_weight"]
    grouped = training.groupby("action_code", observed=True, sort=False)
    means = grouped["weighted_value"].sum() / grouped["cell_weight"].sum()
    predicted = frame.loc[test, "action_code"].map(means)
    if predicted.isna().any():
        raise ValueError("outer fold has an unseen action cell")
    output_action[test] = predicted.to_numpy(dtype=np.float32)


def xgb_parameters(lock: dict[str, Any], device: str) -> dict[str, Any]:
    frozen = lock["models"]["first"]["parameters"]
    return {
        "objective": frozen["objective"],
        "tree_method": frozen["tree_method"],
        "device": device,
        "eta": frozen["eta"],
        "max_depth": frozen["max_depth"],
        "min_child_weight": frozen["min_child_weight"],
        "subsample": frozen["subsample"],
        "colsample_bytree": frozen["colsample_bytree"],
        "reg_alpha": frozen["reg_alpha"],
        "reg_lambda": frozen["reg_lambda"],
        "max_bin": frozen["max_bin"],
        "seed": frozen["seed"],
        "eval_metric": "rmse",
        "nthread": max(1, (os.cpu_count() or 2) - 1),
    }


def train_outer_model(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    weights: np.ndarray,
    outer_fold: int,
    parameters: dict[str, Any],
    lock: dict[str, Any],
    model_path: Path,
) -> tuple[np.ndarray, int]:
    fold = frame["fold"].to_numpy(dtype=np.int8)
    eligible = frame[label].notna().to_numpy()
    inner_fold = (outer_fold + 1) % 5
    inner_train = eligible & (fold != outer_fold) & (fold != inner_fold)
    inner_valid = eligible & (fold == inner_fold)
    outer_train = eligible & (fold != outer_fold)
    outer_test = eligible & (fold == outer_fold)
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
        evals=[(valid_matrix, "inner_validation")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    rounds = int(selector.best_iteration) + 1
    del selector, valid_matrix, train_matrix
    gc.collect()

    refit_matrix = xgb.QuantileDMatrix(
        frame.loc[outer_train, features],
        label=frame.loc[outer_train, label],
        weight=weights[outer_train],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )
    model = xgb.train(
        parameters,
        refit_matrix,
        num_boost_round=rounds,
        verbose_eval=False,
    )
    test_matrix = xgb.QuantileDMatrix(
        frame.loc[outer_test, features],
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


def cumulative_loss(
    frame: pd.DataFrame,
    weights: list[np.ndarray],
    prediction: np.ndarray,
) -> list[float]:
    losses: list[float] = []
    truth_rates = frame[list(BAND_COLUMNS)].to_numpy(dtype=np.float64)
    for index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth_rates[:, index])
        truth = truth_rates[eligible, : index + 1] @ DURATIONS[: index + 1]
        predicted = prediction[eligible, : index + 1] @ DURATIONS[: index + 1]
        squared = np.square((truth - predicted) / horizon)
        losses.append(weighted_mean(squared, weights[index][eligible]))
    return losses


def calibration_scales(
    frame: pd.DataFrame, weights: list[np.ndarray], prediction: np.ndarray
) -> np.ndarray:
    scales = np.ones(len(BAND_COLUMNS), dtype=np.float64)
    for index, label in enumerate(BAND_COLUMNS):
        eligible = frame[label].notna().to_numpy()
        y = frame.loc[eligible, label].to_numpy(dtype=np.float64)
        p = prediction[eligible, index].astype(np.float64)
        w = weights[index][eligible].astype(np.float64)
        denominator = np.sum(w * p * p)
        scales[index] = max(0.0, float(np.sum(w * y * p) / denominator))
    return scales


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    lock = load_json(args.protocol_lock)
    feature_audit = load_json(args.feature_audit)
    if feature_audit.get("pass") is not True:
        raise ValueError("feature audit failed")
    if feature_audit.get("materialization_manifest_sha256") != sha256_file(
        args.data_root / "manifest.json"
    ):
        raise ValueError("feature audit materialization mismatch")
    presses = list(lock["dataset"]["presses"])
    sensor_features = [
        column
        for column in feature_audit["predictor_columns"]
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
        *sensor_features,
        *BAND_COLUMNS,
    ]
    frame = load_data(args.data_root, load_columns, presses)
    action_features = [
        "press_category",
        "ratio_category",
        "compression_ratio",
        "action_category",
        "log_token_clock",
    ]
    causal_features = [*action_features, *sensor_features]
    for column in sensor_features:
        frame[column] = frame[column].astype(np.float32)
    weights = [loss_weights(frame, label) for label in BAND_COLUMNS]
    n_rows = len(frame)
    predictions = {
        name: np.full((n_rows, len(BAND_COLUMNS)), np.nan, dtype=np.float32)
        for name in ("global_mean", "action_only", *CANDIDATES)
    }
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    parameters = xgb_parameters(lock, args.device)
    rounds: dict[str, dict[str, int]] = {name: {} for name in CANDIDATES}

    for band_index, label in enumerate(BAND_COLUMNS):
        for outer_fold in range(5):
            fit_baselines(
                frame,
                label,
                weights[band_index],
                outer_fold,
                predictions["global_mean"][:, band_index],
                predictions["action_only"][:, band_index],
            )
            outer = frame["fold"].to_numpy() == outer_fold
            eligible_outer = outer & frame[label].notna().to_numpy()
            for candidate, features in (
                ("action_clock", action_features),
                ("causal_xgb", causal_features),
            ):
                model_path = (
                    model_root / f"{candidate}.{label}.fold{outer_fold}.json"
                )
                predicted, selected_rounds = train_outer_model(
                    frame,
                    features,
                    label,
                    weights[band_index],
                    outer_fold,
                    parameters,
                    lock,
                    model_path,
                )
                predictions[candidate][eligible_outer, band_index] = predicted
                rounds[candidate][f"{label}.fold{outer_fold}"] = (
                    selected_rounds
                )
                print(
                    f"completed {candidate} {label} fold={outer_fold} "
                    f"rounds={selected_rounds}",
                    flush=True,
                )

    for name, values in predictions.items():
        expected = frame[list(BAND_COLUMNS)].notna().to_numpy()
        if not np.array_equal(np.isfinite(values), expected):
            raise ValueError(f"incomplete OOF predictions for {name}")

    scales = calibration_scales(frame, weights, predictions["causal_xgb"])
    calibrated = predictions["causal_xgb"] * scales.astype(np.float32)
    losses = {
        name: cumulative_loss(frame, weights, values)
        for name, values in {
            **predictions,
            "causal_xgb_calibrated": calibrated,
        }.items()
    }
    output = frame[
        [
            "run_id",
            "prompt_id",
            "task",
            "press",
            "compression_ratio",
            "fold",
            "token_pos",
            *BAND_COLUMNS,
        ]
    ].copy()
    for index, label in enumerate(BAND_COLUMNS):
        output[f"weight_{label}"] = weights[index]
        for name, values in predictions.items():
            output[f"pred_{name}_{label}"] = values[:, index]
    prediction_path = args.output_root / "oof_predictions.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pandas(output, preserve_index=False),
        prediction_path,
        compression="zstd",
        use_dictionary=("run_id", "prompt_id", "task", "press"),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_oof_complete_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "feature_audit_sha256": sha256_file(args.feature_audit),
        "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "device": args.device,
        "parameters": parameters,
        "action_feature_order": action_features,
        "causal_feature_order": causal_features,
        "selected_rounds": rounds,
        "calibration_scales": scales.tolist(),
        "horizon_normalized_mse": {
            name: dict(zip(map(str, HORIZONS), values, strict=True))
            for name, values in losses.items()
        },
        "macro_normalized_mse": {
            name: float(np.mean(values)) for name, values in losses.items()
        },
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
