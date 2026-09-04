"""Cross-fit the matched action+task+clock comparator (analysis only).

Same nested early-stopping scheme and hyperparameters as the tabular
trainer. Output feeds the pilot-margin comparison, not any v1 claim.
"""

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
import pyarrow.parquet as pq
import xgboost as xgb

SCHEMA_VERSION = "herald.quality_risk_matched_comparator_oof.v1"
LABEL = "damage"


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
    from train_quality_risk_tabular import load_data, loss_weights

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
    ]
    frame = load_data(args.data_root, load_columns, presses)
    tasks = sorted(str(value) for value in frame["task"].unique())
    frame["task_category"] = pd.Categorical(frame["task"], categories=tasks)
    features = [
        "press_category",
        "ratio_category",
        "compression_ratio",
        "action_category",
        "task_category",
        "token_pos",
        "log_token_clock",
    ]
    weights = loss_weights(frame)
    parameters = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": args.device,
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
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    rounds: dict[str, int] = {}
    for outer_fold in range(5):
        predicted, selected = train_outer_model(
            frame,
            features,
            weights,
            outer_fold,
            parameters,
            model_root / f"action_task_clock.fold{outer_fold}.json",
        )
        predictions[frame["fold"].to_numpy() == outer_fold] = predicted
        rounds[f"fold{outer_fold}"] = selected
        print(
            f"completed action_task_clock fold={outer_fold} "
            f"rounds={selected}",
            flush=True,
        )
    if not np.isfinite(predictions).all():
        raise ValueError("incomplete OOF predictions")
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
        ]
    ].copy()
    output["pred_action_task_clock"] = predictions
    prediction_path = args.output_root / "oof_predictions.parquet"
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pandas(output, preserve_index=False),
        prediction_path,
        compression="zstd",
        use_dictionary=("run_id", "prompt_id", "task", "press"),
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "analysis_only_not_a_claim",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "feature_audit_sha256": sha256_file(args.feature_audit),
        "data_manifest_sha256": sha256_file(args.data_root / "manifest.json"),
        "parameters": parameters,
        "feature_order": features,
        "selected_rounds": rounds,
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
