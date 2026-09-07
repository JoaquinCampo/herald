#!/usr/bin/env python3
"""Fit the frozen LOO retrieval-probe models from persisted measurements."""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIRST = ROOT / "results/retrieval-probe-v1-first"
DEFAULT_REST = ROOT / "results/retrieval-probe-v1-rest"
DEFAULT_SOURCE = ROOT / "results/task-aware-mse-audit/predictions.json"
DEFAULT_OUTPUT = ROOT / "results/retrieval-probe-model"
TASK = "niah_single_2"
ACTION = 0.10
EXPECTED_COUNT = 12
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 2_026_090_627


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def load_records(directories):
    records = {}
    run_metadata = {}
    for directory in directories:
        directory = Path(directory).expanduser().resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        run_path = directory / "run.json"
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("status") != "completed" or run.get("failures"):
            raise ValueError(f"incomplete retrieval-probe run: {directory}")
        run_metadata[str(run_path.relative_to(ROOT))] = run
        for path in sorted(directory.glob("*.json")):
            if path.name == "run.json":
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("status") != "completed":
                raise ValueError(f"incomplete retrieval-probe record: {path}")
            prompt_id = record.get("manifest_row", {}).get("id")
            if not isinstance(prompt_id, str) or prompt_id in records:
                raise ValueError(
                    f"invalid or duplicate retrieval-probe record: {path}"
                )
            features = record.get("features")
            if not isinstance(features, dict):
                raise ValueError(f"missing features: {path}")
            for name in ("unforced", "probe"):
                feature = features.get(name)
                if not isinstance(feature, dict):
                    raise ValueError(f"missing {name} feature: {path}")
                _finite(feature.get("z"), f"{prompt_id} {name}.z")
            records[prompt_id] = {"record": record, "path": path}
    if len(records) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} records, found {len(records)}"
        )
    return records, run_metadata


def load_rows(source_path, records):
    source_path = Path(source_path).expanduser().resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_rows = source.get("task_action_mean")
    if not isinstance(source_rows, list) or len(source_rows) != 60:
        raise ValueError("source task_action_mean must contain 60 rows")
    selected = [
        row
        for row in source_rows
        if row.get("task") == TASK
        and math.isclose(float(row.get("action")), ACTION)
    ]
    if len(selected) != EXPECTED_COUNT:
        raise ValueError(
            f"expected {EXPECTED_COUNT} filtered source rows, "
            f"found {len(selected)}"
        )
    rows = []
    seen = set()
    for source_row in selected:
        prompt_id = source_row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id in seen:
            raise ValueError(
                f"invalid or duplicate source prompt: {prompt_id}"
            )
        seen.add(prompt_id)
        if prompt_id not in records:
            raise ValueError(
                f"missing measurement for source prompt: {prompt_id}"
            )
        fold = source_row.get("fold")
        signed_loss = source_row.get("signed_loss")
        if not isinstance(fold, int) or fold < 0:
            raise ValueError(f"invalid source fold: {prompt_id}")
        signed_loss = _finite(signed_loss, f"{prompt_id} signed_loss")
        record = records[prompt_id]["record"]
        features = record["features"]
        rows.append(
            {
                "prompt_id": prompt_id,
                "task": source_row["task"],
                "fold": fold,
                "action": float(source_row["action"]),
                "signed_loss": signed_loss,
                "unforced_z": _finite(
                    features["unforced"]["z"], f"{prompt_id} unforced.z"
                ),
                "probe_z": _finite(
                    features["probe"]["z"], f"{prompt_id} probe.z"
                ),
            }
        )
    if seen != set(records):
        raise ValueError(
            "filtered source and retrieval-probe records do not match"
        )
    labels = [row["signed_loss"] for row in rows]
    if (
        sum(value > 0.0 for value in labels) != 8
        or sum(value == 0.0 for value in labels) != 4
    ):
        raise ValueError(
            "filtered labels must contain exactly 8 failures and "
            "4 correct cases"
        )
    return rows, {
        "path": str(source_path),
        "sha256": sha256(source_path),
        "key": "task_action_mean",
        "rows_before_filter": len(source_rows),
        "filter": {"task": TASK, "action": ACTION},
        "rows_after_filter": len(rows),
        "fields_used": ["prompt_id", "task", "fold", "action", "signed_loss"],
        "prediction_fields_ignored": ["prediction"],
        "observed_class_counts": {
            "failure_signed_loss_gt_zero": 8,
            "correct_signed_loss_eq_zero": 4,
        },
    }


def fit_loo(rows, feature_name):
    y = np.asarray([row["signed_loss"] for row in rows], dtype=float)
    x = np.asarray([[row[feature_name]] for row in rows], dtype=float)
    predictions = []
    fold_models = []
    for test_index in range(len(rows)):
        train_indices = np.asarray(
            [index for index in range(len(rows)) if index != test_index],
            dtype=int,
        )
        scaler = StandardScaler().fit(x[train_indices])
        model = Ridge(alpha=1.0, fit_intercept=True).fit(
            scaler.transform(x[train_indices]), y[train_indices]
        )
        prediction = float(
            model.predict(scaler.transform(x[[test_index]]))[0]
        )
        predictions.append(prediction)
        fold_models.append(
            {
                "held_out_prompt_id": rows[test_index]["prompt_id"],
                "train_count": int(len(train_indices)),
                "scaler_mean": float(scaler.mean_[0]),
                "scaler_scale": float(scaler.scale_[0]),
                "coefficient": float(model.coef_[0]),
                "intercept": float(model.intercept_),
            }
        )
    return np.asarray(predictions), fold_models


def baseline_loo(rows):
    y = np.asarray([row["signed_loss"] for row in rows], dtype=float)
    return np.asarray(
        [float(np.mean(np.delete(y, index))) for index in range(len(rows))]
    )


def metrics(y, prediction):
    error = prediction - y
    return {
        "mse": float(mean_squared_error(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "bias_prediction_minus_target": float(np.mean(error)),
        "n": int(len(y)),
    }


def paired_wins(y, baseline, prediction):
    baseline_error = np.abs(baseline - y)
    model_error = np.abs(prediction - y)
    return {
        "model_wins": int(np.sum(model_error < baseline_error)),
        "baseline_wins": int(np.sum(model_error > baseline_error)),
        "ties": int(np.sum(model_error == baseline_error)),
        "n": int(len(y)),
    }


def auc(values, failure):
    ranks = rankdata(values, method="average")
    positives = np.asarray(failure, dtype=bool)
    positive_count = int(np.sum(positives))
    negative_count = int(len(values) - positive_count)
    return float(
        (np.sum(ranks[positives]) - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def exact_auc_permutation(values, failure):
    values = np.asarray(values, dtype=float)
    failure = np.asarray(failure, dtype=bool)
    positive_count = int(np.sum(failure))
    negative_count = int(len(values) - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("AUC requires both observed classes")

    def concordance_twice(positive_indices):
        positive = set(positive_indices)
        total = 0
        for positive_index in positive_indices:
            for negative_index in range(len(values)):
                if negative_index in positive:
                    continue
                if values[positive_index] > values[negative_index]:
                    total += 2
                elif values[positive_index] == values[negative_index]:
                    total += 1
        return total

    observed_indices = np.flatnonzero(failure).tolist()
    observed_twice = concordance_twice(observed_indices)
    extreme = 0
    total_assignments = 0
    for assignment in itertools.combinations(
        range(len(values)), positive_count
    ):
        total_assignments += 1
        if concordance_twice(assignment) >= observed_twice:
            extreme += 1
    return {
        "observed_auc": float(
            observed_twice / (2 * positive_count * negative_count)
        ),
        "positive_direction": "larger_z_predicts_failure",
        "orientation_choice": False,
        "positive_class": "signed_loss_gt_zero",
        "negative_class": "signed_loss_eq_zero",
        "assignments": total_assignments,
        "extreme_assignments_auc_at_least_observed": extreme,
        "one_sided_p_value": float(extreme / total_assignments),
    }


def bootstrap_gain(y, baseline, prediction):
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(y), size=(BOOTSTRAP_REPLICATES, len(y)))
    baseline_mse = np.mean((baseline[indices] - y[indices]) ** 2, axis=1)
    model_mse = np.mean((prediction[indices] - y[indices]) ** 2, axis=1)
    gains = (baseline_mse - model_mse) / baseline_mse
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "resampling_unit": "prompt",
        "uses_fixed_oof_error_pairs": True,
        "retraining_per_replicate": False,
        "mse_gain_fraction": float(
            (baseline_mse.mean() - model_mse.mean()) / baseline_mse.mean()
        ),
        "percentile_2.5": float(np.percentile(gains, 2.5)),
        "percentile_50": float(np.percentile(gains, 50.0)),
        "percentile_97.5": float(np.percentile(gains, 97.5)),
    }


def analyze(rows):
    y = np.asarray([row["signed_loss"] for row in rows], dtype=float)
    baseline = baseline_loo(rows)
    failure = y > 0.0
    feature_results = {}
    predictions = {"baseline": baseline}
    for feature_name in ("unforced_z", "probe_z"):
        prediction, fold_models = fit_loo(rows, feature_name)
        predictions[feature_name] = prediction
        baseline_metrics = metrics(y, baseline)
        model_metrics = metrics(y, prediction)
        gain = (
            baseline_metrics["mse"] - model_metrics["mse"]
        ) / baseline_metrics["mse"]
        wins = paired_wins(y, baseline, prediction)
        feature_results[feature_name] = {
            "feature": feature_name,
            "model": (
                "StandardScaler fitted on 11 training prompts, then "
                "Ridge(alpha=1, fit_intercept=True)"
            ),
            "loo_train_count": 11,
            "metrics": model_metrics,
            "mse_gain_fraction_vs_B0_mean": float(gain),
            "mse_gain_percent_vs_B0_mean": float(100.0 * gain),
            "paired_prompt_wins_vs_B0_mean": wins,
            "raw_z_auc": {
                "auc": auc(
                    np.asarray([row[feature_name] for row in rows]), failure
                ),
                "exact_one_sided_permutation": exact_auc_permutation(
                    np.asarray([row[feature_name] for row in rows]), failure
                ),
            },
            "bootstrap_95_percent_mse_gain": bootstrap_gain(
                y, baseline, prediction
            ),
            "fold_models": fold_models,
            "gate": {
                "mse_gain_at_least_10_percent": bool(gain >= 0.10),
                "at_least_8_of_12_paired_prompt_wins": bool(
                    wins["model_wins"] >= 8
                ),
                "raw_z_auc_at_least_0.80_fixed_positive_direction": bool(
                    auc(
                        np.asarray([row[feature_name] for row in rows]),
                        failure,
                    )
                    >= 0.80
                ),
            },
        }
    baseline_result = {
        "model": "B0 LOO11 action-task mean",
        "metrics": metrics(y, baseline),
    }
    return predictions, baseline_result, feature_results


def write_outputs(rows, source_info, run_metadata, records, output):
    predictions, baseline_result, feature_results = analyze(rows)
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_rows = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                **row,
                "baseline_prediction": float(predictions["baseline"][index]),
                "unforced_z_prediction": float(
                    predictions["unforced_z"][index]
                ),
                "probe_z_prediction": float(predictions["probe_z"][index]),
                "baseline_absolute_error": float(
                    abs(predictions["baseline"][index] - row["signed_loss"])
                ),
                "unforced_z_absolute_error": float(
                    abs(predictions["unforced_z"][index] - row["signed_loss"])
                ),
                "probe_z_absolute_error": float(
                    abs(predictions["probe_z"][index] - row["signed_loss"])
                ),
            }
        )
    prediction_rows.sort(key=lambda row: row["prompt_id"])
    summary = {
        "schema_version": "retrieval_probe_model.v1",
        "status": "completed",
        "design": {
            "task": TASK,
            "action": ACTION,
            "rows": EXPECTED_COUNT,
            "features_fit_separately": ["unforced_z", "probe_z"],
            "baseline": "B0 LOO11 task-action mean",
            "model": "Ridge(alpha=1, fit_intercept=True)",
            "standardization": (
                "StandardScaler fitted within each 11-row training set"
            ),
            "combined_features": False,
            "clipping_or_extra_transforms": False,
            "penalty_or_orientation_search": False,
        },
        "source": source_info,
        "measurement_runs": {
            key: {
                "status": value["status"],
                "ids": value.get("ids", []),
                "manifest_sha256": value.get("manifest_sha256"),
                "engine_sha256": value.get("engine_sha256"),
                "diagnostic_source_sha256": value.get(
                    "diagnostic_source_sha256"
                ),
                "model_revision": value.get("model", {})
                .get("model_signature", {})
                .get("checkpoint_revision"),
                "device": value.get("device"),
                "dtype": value.get("dtype"),
            }
            for key, value in run_metadata.items()
        },
        "measurement_record_sha256": {
            prompt_id: sha256(value["path"])
            for prompt_id, value in records.items()
        },
        "baseline": baseline_result,
        "features": feature_results,
        "cue_increment": {
            "probe_mse_less_than_unforced_mse": bool(
                feature_results["probe_z"]["metrics"]["mse"]
                < feature_results["unforced_z"]["metrics"]["mse"]
            ),
            "probe_mse_minus_unforced_mse": float(
                feature_results["probe_z"]["metrics"]["mse"]
                - feature_results["unforced_z"]["metrics"]["mse"]
            ),
        },
        "gates": {
            feature_name: result["gate"]
            for feature_name, result in feature_results.items()
        },
        "predictions_file": "predictions.json",
        "source_script_sha256": sha256(Path(__file__).resolve()),
        "scope": (
            "Fixed exposed 12-prompt NIAH action-0.10 diagnostic. OOF "
            "bootstrap intervals quantify fixed prediction error pairs, "
            "not model retraining uncertainty."
        ),
    }
    (output / "predictions.json").write_text(
        json.dumps(prediction_rows, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--rest", type=Path, default=DEFAULT_REST)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    records, run_metadata = load_records((args.first, args.rest))
    rows, source_info = load_rows(args.source, records)
    summary = write_outputs(
        rows, source_info, run_metadata, records, args.output
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
