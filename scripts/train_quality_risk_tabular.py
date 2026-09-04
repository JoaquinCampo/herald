"""Cross-fit action-clock and causal XGBoost quality-risk classifier."""

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
from sklearn.metrics import roc_auc_score  # type: ignore[import-untyped]

SCHEMA_VERSION = "herald.quality_risk_tabular_oof.v1"
LABEL = "damage"
CANDIDATES = ("action_clock", "causal_xgb")
BASELINES = ("global_mean", "action_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--exploratory-task-feature",
        action="store_true",
        help="NON-V1 EXPLORATORY: add task identity as a predictor "
        "(forbidden by the v1 protocol; v2-feasibility pilot only).",
    )
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


def loss_weights(frame: pd.DataFrame) -> np.ndarray:
    """Prompt-equal, then action-equal, then token-equal row weights."""
    tokens_per_run = frame.groupby("run_id", sort=False)["run_id"].transform(
        "size"
    )
    actions_per_prompt = (
        frame.drop_duplicates("run_id")
        .groupby("prompt_id", sort=False)["run_id"]
        .count()
    )
    raw = 1.0 / (
        tokens_per_run.to_numpy(dtype=np.float64)
        * frame["prompt_id"]
        .map(actions_per_prompt)
        .to_numpy(dtype=np.float64)
    )
    raw *= len(raw) / raw.sum()
    return raw.astype(np.float32)  # type: ignore[no-any-return]


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights, dtype=np.float64) / weights.sum())


def fit_baselines(
    frame: pd.DataFrame,
    weights: np.ndarray,
    outer_fold: int,
    output_global: np.ndarray,
    output_action: np.ndarray,
) -> None:
    train = frame["fold"].to_numpy() != outer_fold
    test = frame["fold"].to_numpy() == outer_fold
    values = frame[LABEL].to_numpy(dtype=np.float64)
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


def xgb_parameters(device: str) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": device,
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "seed": 2718,
        "eval_metric": "logloss",
        "nthread": max(1, (os.cpu_count() or 2) - 1),
    }


def train_outer_model(
    frame: pd.DataFrame,
    features: list[str],
    weights: np.ndarray,
    outer_fold: int,
    parameters: dict[str, Any],
    model_path: Path,
    maximum_rounds: int = 1000,
    early_stopping: int = 50,
) -> tuple[np.ndarray, int]:
    fold = frame["fold"].to_numpy(dtype=np.int8)
    inner_fold = (outer_fold + 1) % 5
    inner_train = (fold != outer_fold) & (fold != inner_fold)
    inner_valid = fold == inner_fold
    outer_train = fold != outer_fold
    outer_test = fold == outer_fold
    train_matrix = xgb.QuantileDMatrix(
        frame.loc[inner_train, features],
        label=frame.loc[inner_train, LABEL],
        weight=weights[inner_train],
        enable_categorical=True,
        max_bin=parameters["max_bin"],
    )
    valid_matrix = xgb.QuantileDMatrix(
        frame.loc[inner_valid, features],
        label=frame.loc[inner_valid, LABEL],
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
        label=frame.loc[outer_train, LABEL],
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
    prediction = model.predict(test_matrix).astype(np.float32)
    model.save_model(model_path)
    del model, test_matrix, refit_matrix
    gc.collect()
    return prediction, rounds


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
    manifest = load_json(args.data_root / "manifest.json")
    sample_columns = manifest["files"][presses[0]]["columns"]
    sensor_features = [
        column
        for column in sample_columns
        if column
        not in {
            "run_id",
            "token_pos",
            "prompt_id",
            "task",
            "press",
            "compression_ratio",
            "fold",
            "run_length",
            "log_token_clock",
            "damage",
            "catastrophe",
            "imputed_compressed_zero",
            "lift_reference_zero",
        }
    ]
    load_columns = [
        "run_id",
        "prompt_id",
        "task",
        "press",
        "compression_ratio",
        "fold",
        "token_pos",
        "run_length",
        "log_token_clock",
        "damage",
        "catastrophe",
        *sensor_features,
    ]
    frame = load_data(args.data_root, load_columns, presses)
    action_features = [
        "press_category",
        "ratio_category",
        "compression_ratio",
        "action_category",
        "token_pos",
        "log_token_clock",
    ]
    causal_features = [*action_features, *sensor_features]
    if args.exploratory_task_feature:
        tasks = sorted(str(value) for value in frame["task"].unique())
        frame["task_category"] = pd.Categorical(
            frame["task"], categories=tasks
        )
        causal_features = [*causal_features, "task_category"]
    for column in sensor_features:
        frame[column] = frame[column].astype(np.float32)
    weights = loss_weights(frame)
    n_rows = len(frame)
    predictions = {
        name: np.full(n_rows, np.nan, dtype=np.float32)
        for name in (*BASELINES, *CANDIDATES)
    }
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    parameters = xgb_parameters(args.device)
    rounds: dict[str, dict[str, int]] = {name: {} for name in CANDIDATES}
    for outer_fold in range(5):
        fit_baselines(
            frame,
            weights,
            outer_fold,
            predictions["global_mean"],
            predictions["action_only"],
        )
        outer = frame["fold"].to_numpy() == outer_fold
        for candidate, features in (
            ("action_clock", action_features),
            ("causal_xgb", causal_features),
        ):
            model_path = model_root / f"{candidate}.fold{outer_fold}.json"
            predicted, selected_rounds = train_outer_model(
                frame,
                features,
                weights,
                outer_fold,
                parameters,
                model_path,
            )
            predictions[candidate][outer] = predicted
            rounds[candidate][f"fold{outer_fold}"] = selected_rounds
            print(
                f"completed {candidate} fold={outer_fold} "
                f"rounds={selected_rounds}",
                flush=True,
            )
    for name, values in predictions.items():
        if not np.isfinite(values).all():
            raise ValueError(f"incomplete OOF predictions for {name}")
    truth = frame[LABEL].to_numpy(dtype=np.float64)
    clipped = {
        name: np.clip(values.astype(np.float64), 1e-12, 1 - 1e-12)
        for name, values in predictions.items()
    }
    clipped_losses = {
        name: float(
            np.average(
                -(truth * np.log(values) + (1 - truth) * np.log(1 - values)),
                weights=weights.astype(np.float64),
            )
        )
        for name, values in clipped.items()
    }
    aurocs = {
        name: float(roc_auc_score(truth, values, sample_weight=weights))
        for name, values in predictions.items()
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
            "run_length",
            "damage",
            "catastrophe",
        ]
    ].copy()
    output["weight"] = weights
    for name, values in predictions.items():
        output[f"pred_{name}"] = values
    prediction_path = args.output_root / "oof_predictions.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pandas(output, preserve_index=False),
        prediction_path,
        compression="zstd",
        use_dictionary=("run_id", "prompt_id", "task", "press"),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "exploratory_task_conditioned_not_a_v1_claim"
            if args.exploratory_task_feature
            else "development_oof_complete_confirmation_unread"
        ),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "feature_audit_sha256": sha256_file(args.feature_audit),
        "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "device": args.device,
        "parameters": parameters,
        "action_feature_order": action_features,
        "causal_feature_order": causal_features,
        "selected_rounds": rounds,
        "weighted_log_loss": clipped_losses,
        "weighted_auroc": aurocs,
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
