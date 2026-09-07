"""Frozen residual augmentation with the first-pending-token JS feature."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "results/task-aware-mse-audit/predictions.json"
BOUNDARY_DIRS = (
    ROOT / "results/boundary-js-v1-first",
    ROOT / "results/boundary-js-v1-rest",
)
ACTIONS = (0.05, 0.10, 0.20)
FOLDS = (0, 1, 2, 3)
TASKS = ("cwe", "niah_single_2")


def _action_id(action):
    return f"knorm:{float(action):g}"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_boundary_records():
    records = {}
    for directory in BOUNDARY_DIRS:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if run.get("status") != "completed" or run.get("failures"):
            raise ValueError(f"incomplete boundary run: {directory}")
        for path in sorted(directory.glob("*.json")):
            if path.name == "run.json":
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "completed":
                raise ValueError(f"incomplete boundary record: {path}")
            prompt_id = record.get("manifest_row", {}).get("id")
            if not prompt_id or prompt_id in records:
                raise ValueError(f"invalid or duplicate boundary record: {path}")
            records[prompt_id] = record
    return records


def _load_rows(records):
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    source_rows = source.get("task_action_mean")
    if not isinstance(source_rows, list) or len(source_rows) != 60:
        raise ValueError("task_action_mean must contain 60 rows")
    prompt_ids = {row["prompt_id"] for row in source_rows}
    if len(prompt_ids) != 20 or prompt_ids != set(records):
        raise ValueError("boundary records and source prompts do not match")
    expected = {(row["prompt_id"], float(row["action"])) for row in source_rows}
    if len(expected) != 60 or {action for _, action in expected} != set(ACTIONS):
        raise ValueError("source action roster is not the locked roster")

    rows = []
    for source_row in source_rows:
        prompt_id = source_row["prompt_id"]
        action = float(source_row["action"])
        boundary_action = records[prompt_id]["actions"].get(_action_id(action))
        probe = boundary_action.get("probe") if boundary_action else None
        if not isinstance(probe, dict):
            raise ValueError(f"missing probe: {prompt_id}, {action}")
        js = float(probe["js_divergence"])
        if not math.isfinite(js) or js < 0.0:
            raise ValueError(f"invalid JS: {prompt_id}, {action}")
        rows.append(
            {
                "prompt_id": prompt_id,
                "task": source_row["task"],
                "fold": int(source_row["fold"]),
                "action": action,
                "signed_loss": float(source_row["signed_loss"]),
                "js_divergence": js,
            }
        )
    return rows


def _cells(train):
    cells = {}
    for task in TASKS:
        for action in ACTIONS:
            members = [
                row
                for row in train
                if row["task"] == task and row["action"] == action
            ]
            if not members:
                raise ValueError(f"missing training cell: {task}, {action}")
            js = np.asarray([row["js_divergence"] for row in members])
            cells[task, action] = {
                "loss_mean": float(
                    np.mean([row["signed_loss"] for row in members])
                ),
                "js_mean": float(np.mean(js)),
                "js_variable": bool(np.var(js) > 0.0),
            }
    return cells


def _design(row, cells):
    cell = cells[row["task"], row["action"]]
    result = np.zeros(2)
    if cell["js_variable"]:
        result[TASKS.index(row["task"])] = (
            row["js_divergence"] - cell["js_mean"]
        )
    return result


def _metrics(rows, prediction_key):
    targets = [row["signed_loss"] for row in rows]
    predictions = [row[prediction_key] for row in rows]
    return {
        "mse": float(mean_squared_error(targets, predictions)),
        "mae": float(mean_absolute_error(targets, predictions)),
        "bias": float(np.mean(np.asarray(predictions) - targets)),
        "prompts": len({row["prompt_id"] for row in rows}),
        "actions": len(rows),
    }


def _report(rows):
    return {
        name: {
            "pooled": _metrics(rows, name),
            "folds": {
                str(fold): _metrics(
                    [row for row in rows if row["fold"] == fold], name
                )
                for fold in FOLDS
            },
            "tasks": {
                task: _metrics(
                    [row for row in rows if row["task"] == task], name
                )
                for task in TASKS
            },
        }
        for name in ("baseline", "boundary_js_prediction")
    }


def main():
    records = _load_boundary_records()
    rows = _load_rows(records)
    predicted = []
    coefficients = {}
    training_cell_js_variance = {}
    for fold in FOLDS:
        train = [row for row in rows if row["fold"] != fold]
        test = [row for row in rows if row["fold"] == fold]
        cells = _cells(train)
        training_cell_js_variance[str(fold)] = {
            f"{task}|{action:g}": cell["js_variable"]
            for (task, action), cell in cells.items()
        }
        x_train = np.asarray([_design(row, cells) for row in train])
        y_train = np.asarray(
            [
                row["signed_loss"]
                - cells[row["task"], row["action"]]["loss_mean"]
                for row in train
            ]
        )
        scaler = StandardScaler().fit(x_train)
        model = Ridge(alpha=1.0, fit_intercept=False).fit(
            scaler.transform(x_train), y_train
        )
        coefficients[str(fold)] = model.coef_.tolist()
        corrections = model.predict(
            scaler.transform(np.asarray([_design(row, cells) for row in test]))
        )
        for row, correction in zip(test, corrections, strict=True):
            cell = cells[row["task"], row["action"]]
            predicted.append(
                {
                    **row,
                    "baseline": cell["loss_mean"],
                    "js_cell_mean": cell["js_mean"],
                    "js_centered": row["js_divergence"] - cell["js_mean"],
                    "js_cell_variable": cell["js_variable"],
                    "boundary_js_prediction": cell["loss_mean"]
                    + float(correction),
                }
            )
    predicted.sort(key=lambda row: (row["prompt_id"], row["action"]))
    if len(predicted) != 60:
        raise AssertionError("expected exactly 60 OOF predictions")
    report = _report(predicted)
    gain = 1.0 - (
        report["boundary_js_prediction"]["pooled"]["mse"]
        / report["baseline"]["pooled"]["mse"]
    )
    fold_wins = sum(
        report["boundary_js_prediction"]["folds"][str(fold)]["mse"]
        < report["baseline"]["folds"][str(fold)]["mse"]
        for fold in FOLDS
    )
    summary = {
        "primary_metric": "mse",
        "metrics": report,
        "gain_fraction": float(gain),
        "gain_percent": float(100.0 * gain),
        "fold_wins": fold_wins,
        "gates": {
            "mse_gain_at_least_10_percent": gain >= 0.10,
            "at_least_three_fold_wins": fold_wins >= 3,
        },
        "proceed": bool(gain >= 0.10 and fold_wins >= 3),
        "coefficients": coefficients,
        "training_cell_js_variance": training_cell_js_variance,
        "source_sha256": _sha256(Path(__file__).resolve()),
        "source_predictions_sha256": _sha256(SOURCE_PATH),
        "scope": (
            "Existing 20 exposed RULER prompts, fixed preassigned folds, "
            "raw first-pending-token js_divergence only, no confirmation."
        ),
    }
    output = ROOT / "results/boundary-js-model"
    output.mkdir(exist_ok=True)
    (output / "predictions.json").write_text(
        json.dumps(predicted, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
