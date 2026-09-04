"""Resume TCN OOF prediction from saved refit models (no retraining)."""

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from train_quality_risk_tabular import (
    LABEL,
    load_data,
    loss_weights,
)
from train_quality_risk_tcn import predict_outer, run_slices

SCHEMA_VERSION = "herald.quality_risk_tcn_oof.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
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


def parse_selected_epochs(log_path: Path) -> dict[str, int]:
    """Best selection epochs per outer fold from the training log."""
    epochs: dict[str, int] = {}
    pattern = re.compile(r"completed TCN outer fold=(\d+) epochs=(\d+)")
    for line in log_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            epochs[match.group(1)] = int(match.group(2))
    if sorted(epochs) != ["0", "1", "2", "3", "4"]:
        raise ValueError("incomplete selection epochs in training log")
    return epochs


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    # Single-threaded torch: avoids a oneDNN/OpenMP race that segfaults
    # fresh processes on this host; predict-only, no math change.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    lock = load_json(args.protocol_lock)
    audit = load_json(args.feature_audit)
    if audit.get("pass") is not True:
        raise ValueError("feature audit failed")
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
    continuous_features = [
        "compression_ratio",
        "log_token_clock",
        *sensor_features,
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
        *sensor_features,
        LABEL,
    ]
    selected_epochs = parse_selected_epochs(args.train_log)
    ratios: list[str] | None = None
    parts: list[pd.DataFrame] = []
    for press in presses:
        press_frame = load_data(args.data_root, load_columns, [press])
        press_frame.sort_values(
            ["run_id", "token_pos"], inplace=True, ignore_index=True
        )
        press_frame["press_category"] = pd.Categorical(
            press_frame["press"], categories=presses
        )
        press_ratios = sorted(
            float(value)
            for value in press_frame["compression_ratio"].unique()
        )
        if ratios is None:
            ratios = [str(value) for value in press_ratios]
        elif ratios != [str(value) for value in press_ratios]:
            raise ValueError(f"ratio roster differs in {press}")
        press_frame["ratio_category"] = pd.Categorical(
            press_frame["compression_ratio"].astype(str),
            categories=ratios,
        )
        press_codes = press_frame["press_category"].cat.codes.to_numpy(
            dtype=np.int16
        )
        ratio_codes = press_frame["ratio_category"].cat.codes.to_numpy(
            dtype=np.int16
        )
        press_frame["action_code"] = press_codes * len(ratios) + ratio_codes
        press_slices = run_slices(press_frame)
        press_folds = np.asarray(
            [press_frame.iloc[start]["fold"] for start, _ in press_slices],
            dtype=np.int8,
        )
        press_arrays = (
            press_frame[continuous_features].to_numpy(dtype=np.float32),
            press_frame["press_category"].cat.codes.to_numpy(dtype=np.int16),
            press_frame["ratio_category"].cat.codes.to_numpy(dtype=np.int16),
            press_frame["action_code"].to_numpy(dtype=np.int16),
            press_frame[LABEL].to_numpy(dtype=np.float32),
            loss_weights(press_frame).astype(np.float32),
        )
        press_oof = np.full(len(press_frame), np.nan, dtype=np.float32)
        for outer_fold in range(5):
            refit_path = args.model_root / f"refit.fold{outer_fold}.pt"
            normalization = np.load(
                args.model_root / f"refit.fold{outer_fold}.normalization.npz"
            )
            fold_prediction = predict_outer(
                press_arrays,
                press_slices,
                press_folds,
                outer_fold,
                normalization["mean"],
                normalization["scale"],
                refit_path,
            )
            fold_rows = press_frame["fold"].to_numpy() == outer_fold
            press_oof[fold_rows] = fold_prediction[fold_rows]
            print(
                f"predicted TCN {press} outer fold={outer_fold}", flush=True
            )
        if not np.isfinite(press_oof).all():
            raise ValueError(f"incomplete TCN OOF predictions for {press}")
        press_frame["pred_tcn"] = press_oof
        parts.append(
            press_frame[
                [
                    "run_id",
                    "prompt_id",
                    "task",
                    "press",
                    "compression_ratio",
                    "fold",
                    "token_pos",
                    "run_length",
                    LABEL,
                    "pred_tcn",
                ]
            ]
        )
        del press_frame, press_arrays, press_oof
    gc.collect()
    args.output_root.mkdir(parents=True)
    output = pd.concat(parts, ignore_index=True)
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
        "continuous_feature_order": continuous_features,
        "categorical_encoding": "one_hot_press_6_ratio_7_action_42",
        "device": "cpu",
        "selected_epochs": selected_epochs,
        "resumed_from_saved_refits": True,
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
