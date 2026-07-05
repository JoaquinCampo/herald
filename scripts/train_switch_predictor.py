"""Train canonical switch-level HERALD predictor experiments."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from herald.switch_baselines import (
    dataset_fingerprint,
    duplicate_audit,
    evaluate_switch_predictions,
    leave_one_compressor_splits,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
PROMPT_GROUP_FIELDS = ("model", "task", "prompt_id")
DUPLICATE_KEY_FIELDS = (
    "model",
    "task",
    "prompt_id",
    "compressor",
    "ratio",
    "s",
)
ROLLING_MARKERS = (
    "_accel",
    "_delta",
    "_ewma_",
    "_riqr_",
    "_rmax_",
    "_rmean_",
    "_rmedian_",
    "_rmin_",
    "_rstd_",
    "_slope_",
)


@dataclass(frozen=True)
class TrainConfig:
    """Huber training configuration."""

    name: str
    dataset: Path
    out_dir: Path
    compressors: list[str]
    seed: int
    feature_set: str
    model_type: str
    epochs: int
    learning_rate: float
    weight_decay: float
    huber_beta: float
    hidden_dim: int
    dropout: float
    bootstrap_resamples: int
    device: str
    clip_predictions: bool


@dataclass(frozen=True)
class Preprocessor:
    """Train-fitted median imputation and standardization."""

    columns: list[str]
    medians: np.ndarray[Any, np.dtype[np.float32]]
    means: np.ndarray[Any, np.dtype[np.float32]]
    scales: np.ndarray[Any, np.dtype[np.float32]]
    dropped_all_missing: list[str]


@dataclass(frozen=True)
class FittedSplit:
    """Outputs for one held-out-compressor fit."""

    heldout_compressor: str
    preprocessor: Preprocessor
    losses: list[float]
    n_train: int
    n_test: int
    n_features: int
    y_train_min: float
    y_train_max: float


def main() -> None:
    """Run a tracked switch-predictor experiment."""
    config = _parse_args()
    device = _resolve_device(config.device)
    rows = _load_rows(config.dataset, config.compressors)
    predictions, fit_summaries = _fit_leave_one_compressor(
        rows, config, device
    )
    predicted_rows = _attach_predictions(rows, predictions)
    summary = evaluate_switch_predictions(
        predicted_rows,
        prediction_key="predicted_dq",
        model_name=config.name,
        compressors=config.compressors,
        seed=config.seed,
        bootstrap_resamples=config.bootstrap_resamples,
        command=sys.argv,
    )
    artifact = _build_artifact(config, rows, summary, fit_summaries, device)
    _write_outputs(config, predicted_rows, summary, artifact)


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("results/predictor/switch_dataset.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/experiments"),
    )
    parser.add_argument("--name", default="linear_huber_all_mps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compressor",
        action="append",
        dest="compressors",
        default=None,
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "no_position", "static", "rolling"],
        default="all",
    )
    parser.add_argument(
        "--model-type", choices=["linear", "mlp"], default="linear"
    )
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--huber-beta", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Disable clipping to the train target range for each split.",
    )
    args = parser.parse_args()
    compressors = (
        args.compressors if args.compressors else PRIMARY_COMPRESSORS
    )
    return TrainConfig(
        name=args.name,
        dataset=args.dataset,
        out_dir=args.out_dir,
        compressors=list(compressors),
        seed=args.seed,
        feature_set=args.feature_set,
        model_type=args.model_type,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_beta=args.huber_beta,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        bootstrap_resamples=args.bootstrap_resamples,
        device=args.device,
        clip_predictions=not args.no_clip_predictions,
    )


def _resolve_device(name: str) -> Any:
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit(
            "MPS requested but torch.backends.mps is unavailable"
        )
    return cast(Any, torch).device(name)


def _load_rows(path: Path, compressors: list[str]) -> list[dict[str, Any]]:
    parquet = import_module("pyarrow.parquet")
    table = cast(Any, parquet.read_table)(path)
    rows = cast(list[dict[str, Any]], cast(Any, table.to_pylist)())
    selected = set(compressors)
    usable: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("compressor")) not in selected:
            continue
        dq = _as_float(row.get("dq"))
        if dq is None:
            continue
        usable.append(row)
    if not usable:
        raise SystemExit("no usable rows after compressor and dq filtering")
    return usable


def _fit_leave_one_compressor(
    rows: list[dict[str, Any]], config: TrainConfig, device: Any
) -> tuple[dict[tuple[object, ...], float], list[FittedSplit]]:
    torch.manual_seed(config.seed)
    predictions: dict[tuple[object, ...], float] = {}
    summaries: list[FittedSplit] = []
    splits = leave_one_compressor_splits(
        rows,
        compressors=config.compressors,
        seed=config.seed,
    )
    for split in splits:
        columns = _feature_columns(split.train, config.feature_set)
        preprocessor = _fit_preprocessor(split.train, columns)
        train_x = _transform(split.train, preprocessor)
        test_x = _transform(split.test, preprocessor)
        train_y = np.asarray(
            [_required_float(row["dq"]) for row in split.train],
            dtype=np.float32,
        ).reshape(-1, 1)
        model, losses = _train_huber_model(
            train_x,
            train_y,
            config,
            device,
        )
        pred = _predict(model, test_x, device)
        y_min = _array_min(train_y)
        y_max = _array_max(train_y)
        if config.clip_predictions:
            pred = np.clip(pred, y_min, y_max)
        for row, value in zip(split.test, pred.tolist(), strict=True):
            prediction = _required_float(value)
            predictions[_duplicate_key(row)] = prediction
        summaries.append(
            FittedSplit(
                heldout_compressor=split.heldout_compressor,
                preprocessor=preprocessor,
                losses=losses,
                n_train=len(split.train),
                n_test=len(split.test),
                n_features=train_x.shape[1],
                y_train_min=y_min,
                y_train_max=y_max,
            )
        )
    return predictions, summaries


def _feature_columns(
    rows: list[dict[str, Any]], feature_set: str
) -> list[str]:
    columns = sorted(
        key for row in rows for key in row if str(key).startswith("feat__")
    )
    unique = list(dict.fromkeys(columns))
    if feature_set == "all":
        selected = unique
    elif feature_set == "no_position":
        selected = [col for col in unique if col != "feat__position"]
    elif feature_set == "static":
        selected = [
            col
            for col in unique
            if col == "feat__position" or not _is_rolling_feature(col)
        ]
    elif feature_set == "rolling":
        selected = [col for col in unique if _is_rolling_feature(col)]
    else:
        raise ValueError(f"unknown feature set: {feature_set}")
    return ["ratio", *selected]


def _is_rolling_feature(column: str) -> bool:
    return any(marker in column for marker in ROLLING_MARKERS)


def _fit_preprocessor(
    rows: list[dict[str, Any]], columns: list[str]
) -> Preprocessor:
    raw = _raw_matrix(rows, columns)
    keep: list[int] = []
    dropped: list[str] = []
    for idx, column in enumerate(columns):
        if np.isfinite(raw[:, idx]).any():
            keep.append(idx)
        else:
            dropped.append(column)
    kept_columns = [columns[idx] for idx in keep]
    kept = raw[:, keep]
    medians = np.nanmedian(kept, axis=0).astype(np.float32)
    imputed = np.where(np.isfinite(kept), kept, medians)
    means = imputed.mean(axis=0).astype(np.float32)
    scales = imputed.std(axis=0).astype(np.float32)
    scales = np.where(scales > 0.0, scales, 1.0).astype(np.float32)
    return Preprocessor(
        columns=kept_columns,
        medians=medians,
        means=means,
        scales=scales,
        dropped_all_missing=dropped,
    )


def _transform(
    rows: list[dict[str, Any]], preprocessor: Preprocessor
) -> np.ndarray[Any, np.dtype[np.float32]]:
    raw = _raw_matrix(rows, preprocessor.columns)
    imputed = np.where(np.isfinite(raw), raw, preprocessor.medians)
    transformed = (
        (imputed - preprocessor.means) / preprocessor.scales
    ).astype(np.float32)
    return cast(np.ndarray[Any, np.dtype[np.float32]], transformed)


def _raw_matrix(
    rows: list[dict[str, Any]], columns: list[str]
) -> np.ndarray[Any, np.dtype[np.float32]]:
    data = np.empty((len(rows), len(columns)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        for col_idx, column in enumerate(columns):
            value = _as_float(row.get(column))
            data[row_idx, col_idx] = math.nan if value is None else value
    return data


def _train_huber_model(
    train_x: np.ndarray[Any, np.dtype[np.float32]],
    train_y: np.ndarray[Any, np.dtype[np.float32]],
    config: TrainConfig,
    device: Any,
) -> tuple[torch.nn.Module, list[float]]:
    x = cast(Any, torch).from_numpy(train_x).to(device)
    y = cast(Any, torch).from_numpy(train_y).to(device)
    model = _build_model(config, train_x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = torch.nn.HuberLoss(delta=config.huber_beta)
    losses: list[float] = []
    for epoch in range(config.epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == config.epochs - 1:
            losses.append(_tensor_scalar(loss))
    return model, losses


def _build_model(config: TrainConfig, n_features: int) -> torch.nn.Module:
    if config.model_type == "linear":
        return torch.nn.Linear(n_features, 1)
    if config.model_type == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(n_features, config.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(config.dropout),
            torch.nn.Linear(config.hidden_dim, 1),
        )
    raise ValueError(f"unknown model type: {config.model_type}")


def _predict(
    model: torch.nn.Module,
    data: np.ndarray[Any, np.dtype[np.float32]],
    device: Any,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    with torch.no_grad():
        x = cast(Any, torch).from_numpy(data).to(device)
        pred = model(x).detach().cpu().numpy().reshape(-1)
    typed = pred.astype(np.float32)
    return cast(np.ndarray[Any, np.dtype[np.float32]], typed)


def _attach_predictions(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[object, ...], float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        copied["predicted_dq"] = predictions.get(_duplicate_key(row), 0.0)
        copied["scored_by_predictor"] = _duplicate_key(row) in predictions
        out.append(copied)
    return out


def _build_artifact(
    config: TrainConfig,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    fits: list[FittedSplit],
    device: Any,
) -> dict[str, Any]:
    split_metrics = []
    for split in summary["splits"]:
        metric = split["prediction"]
        split_metrics.append(
            {
                "heldout_compressor": split["heldout_compressor"],
                "mae": metric["mae"],
                "relative_mae_improvement_vs_locked": metric[
                    "relative_mae_improvement_vs_locked"
                ],
                "top_decile_lift_dq_gt_0": metric["controller_metrics"][
                    "dq_gt_0"
                ]["top_decile_lift"],
                "top_decile_lift_dq_ge_0_5": metric["controller_metrics"][
                    "dq_ge_0_5"
                ]["top_decile_lift"],
                "auprc_dq_gt_0": metric["ranking_metrics"]["dq_gt_0"][
                    "auprc"
                ],
                "recall_at_10_fpr_dq_gt_0": metric["ranking_metrics"][
                    "dq_gt_0"
                ]["recall_at_10_fpr"],
                "bootstrap_ci": metric["bootstrap_ci"],
            }
        )
    rel_values = [
        item["relative_mae_improvement_vs_locked"] for item in split_metrics
    ]
    return {
        "config": _config_json(config),
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "dataset_fingerprint": dataset_fingerprint(rows),
        "n_rows": len(rows),
        "duplicate_audit": duplicate_audit(rows),
        "split_metrics": split_metrics,
        "mean_relative_mae_improvement": sum(rel_values) / len(rel_values),
        "all_heldouts_improved": all(value > 0.0 for value in rel_values),
        "ci_excludes_zero_count": sum(
            1
            for item in split_metrics
            if item["bootstrap_ci"]["low"] is not None
            and item["bootstrap_ci"]["low"] > 0.0
        ),
        "fit_summaries": [_fit_json(fit) for fit in fits],
    }


def _write_outputs(
    config: TrainConfig,
    predicted_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    out_dir = config.out_dir / config.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2))
    (out_dir / "report.md").write_text(_markdown_report(config, artifact))
    _write_parquet(out_dir / "predictions.parquet", predicted_rows)
    _append_experiment_log(
        config.out_dir / "experiment_log.md", config, artifact
    )
    print(f"experiment={config.name}")
    print(f"device={artifact['device']}")
    print(f"rows={artifact['n_rows']}")
    print(
        "mean_relative_mae_improvement="
        f"{artifact['mean_relative_mae_improvement']:.6f}"
    )
    print(f"summary -> {out_dir / 'summary.json'}")
    print(f"report -> {out_dir / 'report.md'}")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pyarrow = import_module("pyarrow")
    table = cast(Any, pyarrow.Table).from_pylist(rows)
    pq = import_module("pyarrow.parquet")
    cast(Any, pq.write_table)(table, path)


def _append_experiment_log(
    path: Path, config: TrainConfig, artifact: dict[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n" + _markdown_report(config, artifact)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _markdown_report(config: TrainConfig, artifact: dict[str, Any]) -> str:
    lines = [
        f"## EXECUTE, {config.name}",
        "",
        f"- Device: `{artifact['device']}`.",
        f"- Dataset fingerprint: `{artifact['dataset_fingerprint']}`.",
        f"- Rows: {artifact['n_rows']}.",
        f"- Feature set: `{config.feature_set}`.",
        f"- Model type: `{config.model_type}`.",
        f"- Epochs: {config.epochs}.",
        f"- Mean relative MAE improvement: "
        f"{artifact['mean_relative_mae_improvement']:.4f}.",
        f"- Improved all held-outs: {artifact['all_heldouts_improved']}.",
        f"- CIs excluding zero: {artifact['ci_excludes_zero_count']}.",
        "",
        "| Held-out | Rel MAE | MAE | Lift dq > 0 | "
        "Lift dq >= 0.5 | CI low | CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in artifact["split_metrics"]:
        ci = item["bootstrap_ci"]
        lines.append(
            f"| {item['heldout_compressor']} | "
            f"{item['relative_mae_improvement_vs_locked']:.4f} | "
            f"{item['mae']:.4f} | "
            f"{item['top_decile_lift_dq_gt_0']:.4f} | "
            f"{item['top_decile_lift_dq_ge_0_5']:.4f} | "
            f"{_fmt(ci['low'])} | {_fmt(ci['high'])} |"
        )
    lines.extend(
        [
            "",
            "### REFLECT",
            "",
            "This is a canonical leave-one-compressor-out evaluation with "
            "prompt-group-disjoint splits and train-only preprocessing. "
            "Trust still depends on ablations and leakage checks before "
            "claiming any tier.",
            "",
        ]
    )
    return "\n".join(lines)


def _config_json(config: TrainConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "dataset": str(config.dataset),
        "out_dir": str(config.out_dir),
        "compressors": config.compressors,
        "seed": config.seed,
        "feature_set": config.feature_set,
        "model_type": config.model_type,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "huber_beta": config.huber_beta,
        "hidden_dim": config.hidden_dim,
        "dropout": config.dropout,
        "bootstrap_resamples": config.bootstrap_resamples,
        "device": config.device,
        "clip_predictions": config.clip_predictions,
    }


def _fit_json(fit: FittedSplit) -> dict[str, Any]:
    return {
        "heldout_compressor": fit.heldout_compressor,
        "n_train": fit.n_train,
        "n_test": fit.n_test,
        "n_features": fit.n_features,
        "losses": fit.losses,
        "y_train_min": fit.y_train_min,
        "y_train_max": fit.y_train_max,
        "dropped_all_missing": fit.preprocessor.dropped_all_missing,
        "columns": fit.preprocessor.columns,
    }


def _duplicate_key(row: dict[str, Any]) -> tuple[object, ...]:
    return tuple(row[field] for field in DUPLICATE_KEY_FIELDS)


def _tensor_scalar(value: Any) -> float:
    item = cast(Any, value.detach().cpu()).item()
    return _required_float(item)


def _array_min(values: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    items = values.reshape(-1).tolist()
    if not items:
        raise ValueError("cannot compute minimum of empty values")
    return min(items)


def _array_max(values: np.ndarray[Any, np.dtype[np.float32]]) -> float:
    items = values.reshape(-1).tolist()
    if not items:
        raise ValueError("cannot compute maximum of empty values")
    return max(items)


def _required_float(value: object) -> float:
    result = _as_float(value)
    if result is None:
        raise ValueError(f"expected finite float, got {value!r}")
    return result


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value + 0.0
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return ""


if __name__ == "__main__":
    main()
