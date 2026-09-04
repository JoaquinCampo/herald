"""Cross-fit the causal TCN quality-risk challenger on CPU."""

import argparse
import gc
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from train_quality_risk_tabular import (
    LABEL,
    load_data,
    loss_weights,
)

SCHEMA_VERSION = "herald.quality_risk_tcn_oof.v1"
DILATIONS = (1, 2, 4, 8, 16)
BATCH_RUNS = 64


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.convolution(F.pad(inputs, (self.left_padding, 0)))
        return inputs + self.dropout(F.gelu(hidden))  # type: ignore[no-any-return]


class RiskTCN(nn.Module):
    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, 64, kernel_size=1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(64, dilation, 0.1) for dilation in DILATIONS
        )
        self.head = nn.Conv1d(64, 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.input_projection(inputs.transpose(1, 2))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden).transpose(1, 2).squeeze(-1)  # type: ignore[no-any-return]


def run_slices(frame: pd.DataFrame) -> list[tuple[int, int]]:
    run_ids = frame["run_id"].to_numpy()
    changes = np.flatnonzero(run_ids[1:] != run_ids[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(frame)]))
    return [
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def fit_normalization(
    continuous: np.ndarray, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    selected = continuous[rows]
    mean = np.nanmean(selected, axis=0, dtype=np.float64)
    scale = np.nanstd(selected, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return (  # type: ignore[return-value]
        mean.astype(np.float32),
        scale.astype(np.float32),
    )


def make_batches(
    slices: list[tuple[int, int]],
    selected_runs: list[int],
    shuffle: bool,
    seed: int,
) -> list[list[int]]:
    ordered = sorted(
        selected_runs, key=lambda index: slices[index][1] - slices[index][0]
    )
    buckets = [
        ordered[start : start + BATCH_RUNS]
        for start in range(0, len(ordered), BATCH_RUNS)
    ]
    if shuffle:
        generator = random.Random(seed)
        generator.shuffle(buckets)
        for bucket in buckets:
            generator.shuffle(bucket)
    return buckets


def batch_tensors(
    batch: list[int],
    slices: list[tuple[int, int]],
    continuous: np.ndarray,
    press_codes: np.ndarray,
    ratio_codes: np.ndarray,
    action_codes: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor, list[tuple[int, int]]]:
    lengths = [slices[index][1] - slices[index][0] for index in batch]
    maximum = max(lengths)
    input_size = 6 + 7 + 42 + continuous.shape[1]
    inputs = np.zeros((len(batch), maximum, input_size), dtype=np.float32)
    output_targets = np.zeros((len(batch), maximum), dtype=np.float32)
    output_weights = np.zeros((len(batch), maximum), dtype=np.float32)
    mask = np.zeros((len(batch), maximum), dtype=bool)
    locations: list[tuple[int, int]] = []
    for batch_index, run_index in enumerate(batch):
        start, end = slices[run_index]
        length = end - start
        normalized = np.nan_to_num(
            (continuous[start:end] - mean) / scale,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        inputs[batch_index, :length, 55:] = normalized
        inputs[batch_index, np.arange(length), press_codes[start:end]] = 1.0
        inputs[
            batch_index,
            np.arange(length),
            6 + ratio_codes[start:end],
        ] = 1.0
        inputs[
            batch_index,
            np.arange(length),
            13 + action_codes[start:end],
        ] = 1.0
        output_targets[batch_index, :length] = targets[start:end]
        output_weights[batch_index, :length] = weights[start:end]
        mask[batch_index, :length] = True
        locations.append((start, end))
    return (
        torch.from_numpy(inputs),
        torch.from_numpy(output_targets),
        torch.from_numpy(output_weights),
        torch.from_numpy(mask),
        locations,
    )


def train_model(
    arrays: tuple[np.ndarray, ...],
    slices: list[tuple[int, int]],
    run_folds: np.ndarray,
    outer_fold: int,
    validation_fold: int | None,
    epochs: int,
    patience: int,
    parameters: dict[str, Any],
    model_path: Path,
    normalization_path: Path,
) -> tuple[int, np.ndarray, np.ndarray]:
    seed = int(parameters["seed"])
    seed_everything(seed)
    device = torch.device("cpu")
    (
        continuous,
        press_codes,
        ratio_codes,
        action_codes,
        targets,
        weights,
    ) = arrays
    training_runs = [
        index
        for index, fold in enumerate(run_folds)
        if fold != outer_fold
        and (validation_fold is None or fold != validation_fold)
    ]
    validation_runs = (
        []
        if validation_fold is None
        else [
            index
            for index, fold in enumerate(run_folds)
            if fold == validation_fold
        ]
    )
    training_rows = np.concatenate(
        [np.arange(*slices[index], dtype=np.int64) for index in training_runs]
    )
    mean, scale = fit_normalization(continuous, training_rows)
    model = RiskTCN(6 + 7 + 42 + continuous.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    best_loss = math.inf
    best_epoch = epochs
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        batches = make_batches(slices, training_runs, True, seed + epoch)
        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            inputs, target, weight, mask, _ = batch_tensors(
                batch,
                slices,
                continuous,
                press_codes,
                ratio_codes,
                action_codes,
                targets,
                weights,
                mean,
                scale,
            )
            logits = model(inputs)
            per_row = criterion(logits, target)
            loss = (weight * per_row * mask).sum() / weight[mask].sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(parameters["gradient_clip_norm"])
            )
            optimizer.step()
        if validation_fold is None:
            continue
        validation_loss = evaluate(
            model,
            make_batches(slices, validation_runs, False, seed),
            slices,
            arrays,
            mean,
            scale,
        )
        print(
            f"outer={outer_fold} epoch={epoch} "
            f"validation={validation_loss:.9g}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-10:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), model_path)
        else:
            stale += 1
            if stale >= patience:
                break
    if validation_fold is None:
        torch.save(model.state_dict(), model_path)
    np.savez(normalization_path, mean=mean, scale=scale)
    del model, optimizer
    gc.collect()
    return best_epoch, mean, scale


def evaluate(
    model: RiskTCN,
    batches: list[list[int]],
    slices: list[tuple[int, int]],
    arrays: tuple[np.ndarray, ...],
    mean: np.ndarray,
    scale: np.ndarray,
) -> float:
    model.eval()
    numerator = 0.0
    denominator = 0.0
    continuous, press_codes, ratio_codes, action_codes, targets, weights = (
        arrays
    )
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    with torch.no_grad():
        for batch in batches:
            inputs, truth, batch_weights, mask, _ = batch_tensors(
                batch,
                slices,
                continuous,
                press_codes,
                ratio_codes,
                action_codes,
                targets,
                weights,
                mean,
                scale,
            )
            logits = model(inputs)
            per_row = criterion(logits, truth).numpy()
            mask_np = mask.numpy()
            batch_w = batch_weights.numpy()
            numerator += float(np.sum(batch_w[mask_np] * per_row[mask_np]))
            denominator += float(np.sum(batch_w[mask_np]))
    return float(numerator / denominator)


def predict_outer(
    arrays: tuple[np.ndarray, ...],
    slices: list[tuple[int, int]],
    run_folds: np.ndarray,
    outer_fold: int,
    mean: np.ndarray,
    scale: np.ndarray,
    model_path: Path,
) -> np.ndarray:
    (
        continuous,
        press_codes,
        ratio_codes,
        action_codes,
        targets,
        weights,
    ) = arrays
    model = RiskTCN(6 + 7 + 42 + arrays[0].shape[1])
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    selected = [
        index for index, fold in enumerate(run_folds) if fold == outer_fold
    ]
    result = np.full(len(arrays[0]), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in make_batches(slices, selected, False, 0):
            inputs, _, _, _, locations = batch_tensors(
                batch,
                slices,
                continuous,
                press_codes,
                ratio_codes,
                action_codes,
                targets,
                weights,
                mean,
                scale,
            )
            prediction = (
                torch.sigmoid(model(inputs)).cpu().numpy().astype(np.float32)
            )
            for batch_index, (start, end) in enumerate(locations):
                result[start:end] = prediction[batch_index, : end - start]
    return result


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
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
    frame = load_data(args.data_root, load_columns, presses)
    frame.sort_values(
        ["run_id", "token_pos"], inplace=True, ignore_index=True
    )
    slices = run_slices(frame)
    run_folds = np.asarray(
        [frame.iloc[start]["fold"] for start, _ in slices], dtype=np.int8
    )
    continuous = frame[continuous_features].to_numpy(dtype=np.float32)
    press_codes = frame["press_category"].cat.codes.to_numpy(dtype=np.int16)
    ratio_codes = frame["ratio_category"].cat.codes.to_numpy(dtype=np.int16)
    action_codes = frame["action_code"].to_numpy(dtype=np.int16)
    targets = frame[LABEL].to_numpy(dtype=np.float32)
    weights = loss_weights(frame).astype(np.float32)
    arrays = (
        continuous,
        press_codes,
        ratio_codes,
        action_codes,
        targets,
        weights,
    )
    parameters = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 50,
        "early_stopping_patience": 5,
        "gradient_clip_norm": 1.0,
        "seed": 2718,
    }
    args.output_root.mkdir(parents=True)
    model_root = args.output_root / "models"
    model_root.mkdir()
    oof = np.full(len(frame), np.nan, dtype=np.float32)
    selected_epochs: dict[str, int] = {}
    for outer_fold in range(5):
        selection_path = model_root / f"selection.fold{outer_fold}.pt"
        best_epoch, _, _ = train_model(
            arrays,
            slices,
            run_folds,
            outer_fold,
            (outer_fold + 1) % 5,
            int(parameters["max_epochs"]),
            int(parameters["early_stopping_patience"]),
            parameters,
            selection_path,
            model_root / f"selection.fold{outer_fold}.normalization.npz",
        )
        refit_path = model_root / f"refit.fold{outer_fold}.pt"
        _, mean, scale = train_model(
            arrays,
            slices,
            run_folds,
            outer_fold,
            None,
            best_epoch,
            0,
            parameters,
            refit_path,
            model_root / f"refit.fold{outer_fold}.normalization.npz",
        )
        fold_prediction = predict_outer(
            arrays,
            slices,
            run_folds,
            outer_fold,
            mean,
            scale,
            refit_path,
        )
        fold_rows = frame["fold"].to_numpy() == outer_fold
        oof[fold_rows] = fold_prediction[fold_rows]
        selected_epochs[str(outer_fold)] = best_epoch
        print(
            f"completed TCN outer fold={outer_fold} epochs={best_epoch}",
            flush=True,
        )
    if not np.isfinite(oof).all():
        raise ValueError("incomplete TCN OOF predictions")
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
            LABEL,
        ]
    ].copy()
    output["weight"] = weights
    output["pred_tcn"] = oof
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
        "prediction_sha256": sha256_file(prediction_path),
        "confirmation_prompts_projected": 0,
    }
    (args.output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
