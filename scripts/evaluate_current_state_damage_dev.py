"""Evaluate development OOF skill without opening confirmation prompts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from train_current_state_damage_tabular import (
    BAND_COLUMNS,
    DURATIONS,
    HORIZONS,
)

SCHEMA_VERSION = "herald.current_state_damage_development_evaluation.v1"
BASELINES = ("global_mean", "action_only", "action_clock")
CANDIDATES = ("causal_xgb", "tcn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--tabular-root", type=Path, required=True)
    parser.add_argument("--tcn-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_predictions(
    tabular_root: Path, tcn_root: Path
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    tabular_columns = [
        "run_id",
        "token_pos",
        "prompt_id",
        "task",
        "press",
        "compression_ratio",
        *BAND_COLUMNS,
        *[f"weight_{label}" for label in BAND_COLUMNS],
        *[
            f"pred_{model}_{label}"
            for model in (*BASELINES, "causal_xgb")
            for label in BAND_COLUMNS
        ],
    ]
    tabular = (
        ds.dataset(  # type: ignore[no-untyped-call]
            tabular_root / "oof_predictions.parquet", format="parquet"
        )
        .to_table(columns=tabular_columns)
        .to_pandas()
    )
    tcn_columns = [
        "run_id",
        "token_pos",
        *[f"pred_tcn_{label}" for label in BAND_COLUMNS],
    ]
    tcn = (
        ds.dataset(  # type: ignore[no-untyped-call]
            tcn_root / "oof_predictions.parquet", format="parquet"
        )
        .to_table(columns=tcn_columns)
        .to_pandas()
    )
    frame = tabular.merge(
        tcn,
        on=["run_id", "token_pos"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(tabular):
        raise ValueError("TCN/tabular OOF identity mismatch")
    scales = {
        "causal_xgb": np.asarray(
            load_json(tabular_root / "report.json")["calibration_scales"],
            dtype=np.float64,
        ),
        "tcn": np.asarray(
            load_json(tcn_root / "report.json")["calibration_scales"],
            dtype=np.float64,
        ),
    }
    return frame, scales


def rate_predictions(
    frame: pd.DataFrame, model: str, scales: dict[str, np.ndarray]
) -> np.ndarray:
    prefix = "pred_tcn" if model == "tcn" else f"pred_{model}"
    values = frame[[f"{prefix}_{label}" for label in BAND_COLUMNS]].to_numpy(
        dtype=np.float64
    )
    if model in scales:
        values *= scales[model]
    return values  # type: ignore[no-any-return]


def cumulative(values: np.ndarray) -> np.ndarray:
    return np.cumsum(values * DURATIONS, axis=1)


def prompt_values(
    frame: pd.DataFrame,
    values: np.ndarray,
    weights: np.ndarray,
    eligible: np.ndarray,
) -> pd.DataFrame:
    selected = pd.DataFrame(
        {
            "prompt_id": frame.loc[eligible, "prompt_id"].to_numpy(),
            "task": frame.loc[eligible, "task"].to_numpy(),
            "weighted": values[eligible] * weights[eligible],
            "weight": weights[eligible],
        }
    )
    grouped = selected.groupby("prompt_id", sort=True)
    result = pd.DataFrame(
        {
            "value": grouped["weighted"].sum() / grouped["weight"].sum(),
            "task": grouped["task"].first(),
        }
    ).reset_index()
    return result


def stratified_draws(tasks: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    draws = np.empty((count, len(tasks)), dtype=np.int32)
    for bootstrap_index in range(count):
        offset = 0
        for group in groups:
            sample = rng.choice(group, size=len(group), replace=True)
            draws[bootstrap_index, offset : offset + len(group)] = sample
            offset += len(group)
    return draws


def simultaneous_bound(
    matrix: np.ndarray,
    tasks: np.ndarray,
    draws: np.ndarray,
    side: Literal["upper", "lower", "two-sided"],
) -> dict[str, Any]:
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite prompt matrix")
    point = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    bootstrap_mean = matrix[draws].mean(axis=1)
    bootstrap_error = matrix[draws].std(axis=1, ddof=1) / np.sqrt(len(matrix))
    studentized = np.divide(
        bootstrap_mean - point,
        bootstrap_error,
        out=np.zeros_like(bootstrap_mean),
        where=bootstrap_error > 0,
    )
    if side == "upper":
        critical = float(np.quantile(studentized.max(axis=1), 0.95))
        bound = point + critical * standard_error
        return {
            "point": point.tolist(),
            "upper": bound.tolist(),
            "critical": critical,
        }
    if side == "lower":
        critical = float(np.quantile((-studentized).max(axis=1), 0.95))
        bound = point - critical * standard_error
        return {
            "point": point.tolist(),
            "lower": bound.tolist(),
            "critical": critical,
        }
    critical = float(np.quantile(np.abs(studentized).max(axis=1), 0.95))
    lower = point - critical * standard_error
    upper = point + critical * standard_error
    return {
        "point": point.tolist(),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "critical": critical,
    }


def subgroup_differences(
    frame: pd.DataFrame,
    difference: np.ndarray,
    weights: list[np.ndarray],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dimension in ("task", "press", "compression_ratio"):
        groups: dict[str, list[float]] = {}
        for value, indices in frame.groupby(
            dimension, observed=True
        ).indices.items():
            horizon_values: list[float] = []
            index_array = np.asarray(indices, dtype=np.int64)
            for horizon_index in range(len(HORIZONS)):
                valid = np.isfinite(difference[index_array, horizon_index])
                rows = index_array[valid]
                horizon_values.append(
                    float(
                        np.average(
                            difference[rows, horizon_index],
                            weights=weights[horizon_index][rows],
                        )
                    )
                )
            groups[str(value)] = horizon_values
        result[dimension] = groups
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = load_json(args.protocol_lock)
    frame, scales = load_predictions(args.tabular_root, args.tcn_root)
    truth_rates = frame[list(BAND_COLUMNS)].to_numpy(dtype=np.float64)
    truth = cumulative(truth_rates)
    weights = [
        frame[f"weight_{label}"].to_numpy(dtype=np.float64)
        for label in BAND_COLUMNS
    ]
    prediction = {
        model: cumulative(rate_predictions(frame, model, scales))
        for model in (*BASELINES, *CANDIDATES)
    }
    losses: dict[str, list[float]] = {model: [] for model in prediction}
    row_losses: dict[str, np.ndarray] = {
        model: np.full_like(truth, np.nan) for model in prediction
    }
    for horizon_index, horizon in enumerate(HORIZONS):
        eligible = np.isfinite(truth[:, horizon_index])
        for model in prediction:
            squared = np.square(
                (
                    truth[eligible, horizon_index]
                    - prediction[model][eligible, horizon_index]
                )
                / horizon
            )
            row_losses[model][eligible, horizon_index] = squared
            losses[model].append(
                float(
                    np.average(
                        squared, weights=weights[horizon_index][eligible]
                    )
                )
            )
    hard_models = [
        min(BASELINES, key=lambda model: losses[model][index])
        for index in range(len(HORIZONS))
    ]
    hard_loss = np.column_stack(
        [
            row_losses[model][:, index]
            for index, model in enumerate(hard_models)
        ]
    )

    reference = prompt_values(
        frame,
        np.zeros(len(frame)),
        weights[0],
        np.isfinite(truth[:, 0]),
    )
    prompt_ids = reference["prompt_id"].tolist()
    tasks = reference["task"].to_numpy()
    draws = stratified_draws(
        tasks,
        int(lock["inference"]["resamples"]),
        int(lock["inference"]["seed"]),
    )
    superiority: dict[str, Any] = {}
    calibration: dict[str, Any] = {}
    subgroups: dict[str, Any] = {}
    prompt_candidate_losses: dict[str, np.ndarray] = {}
    for candidate in CANDIDATES:
        against: dict[str, Any] = {}
        candidate_prompt_columns: list[np.ndarray] = []
        for comparator_name, comparator_loss in (
            ("action_clock", row_losses["action_clock"]),
            ("hard_comparator", hard_loss),
        ):
            columns: list[np.ndarray] = []
            for index in range(len(HORIZONS)):
                eligible = np.isfinite(truth[:, index])
                values = (
                    row_losses[candidate][:, index]
                    - comparator_loss[:, index]
                )
                prompt = prompt_values(
                    frame, values, weights[index], eligible
                )
                prompt = prompt.set_index("prompt_id").loc[prompt_ids]
                columns.append(prompt["value"].to_numpy(dtype=np.float64))
            matrix = np.column_stack(columns)
            against[comparator_name] = simultaneous_bound(
                matrix, tasks, draws, "upper"
            )
            if comparator_name == "hard_comparator":
                subgroups[candidate] = subgroup_differences(
                    frame,
                    row_losses[candidate] - hard_loss,
                    weights,
                )
        superiority[candidate] = against
        for index in range(len(HORIZONS)):
            eligible = np.isfinite(truth[:, index])
            prompt = (
                prompt_values(
                    frame,
                    row_losses[candidate][:, index],
                    weights[index],
                    eligible,
                )
                .set_index("prompt_id")
                .loc[prompt_ids]
            )
            candidate_prompt_columns.append(
                prompt["value"].to_numpy(dtype=np.float64)
            )
        prompt_candidate_losses[candidate] = np.column_stack(
            candidate_prompt_columns
        )

        residual_columns: list[np.ndarray] = []
        slope_columns: list[np.ndarray] = []
        for index, horizon in enumerate(HORIZONS):
            eligible = np.isfinite(truth[:, index])
            residual = (
                truth[:, index] - prediction[candidate][:, index]
            ) / horizon
            residual_prompt = (
                prompt_values(frame, residual, weights[index], eligible)
                .set_index("prompt_id")
                .loc[prompt_ids]
            )
            residual_columns.append(
                residual_prompt["value"].to_numpy(dtype=np.float64)
            )
            slope_numerator = (
                truth[:, index] * prediction[candidate][:, index]
            )
            slope_denominator = np.square(prediction[candidate][:, index])
            numerator_prompt = (
                prompt_values(
                    frame, slope_numerator, weights[index], eligible
                )
                .set_index("prompt_id")
                .loc[prompt_ids]["value"]
                .to_numpy(dtype=np.float64)
            )
            denominator_prompt = (
                prompt_values(
                    frame, slope_denominator, weights[index], eligible
                )
                .set_index("prompt_id")
                .loc[prompt_ids]["value"]
                .to_numpy(dtype=np.float64)
            )
            slope_columns.append(numerator_prompt / denominator_prompt)
        residual_matrix = np.column_stack(residual_columns)
        slope_matrix = np.column_stack(slope_columns)
        calibration[candidate] = {
            "mean_normalized_residual": simultaneous_bound(
                residual_matrix, tasks, draws, "two-sided"
            ),
            "slope": simultaneous_bound(slope_matrix, tasks, draws, "lower"),
        }

    comparison = (
        prompt_candidate_losses["tcn"] - prompt_candidate_losses["causal_xgb"]
    )
    macro_comparison = comparison.mean(axis=1, keepdims=True)
    selection_bound = simultaneous_bound(
        macro_comparison, tasks, draws, "upper"
    )
    xgb_gate = all(
        value < 0
        for comparator in superiority["causal_xgb"].values()
        for value in comparator["upper"]
    )
    tcn_gate = all(
        value < 0
        for comparator in superiority["tcn"].values()
        for value in comparator["upper"]
    )
    subgroup_gate = {
        candidate: all(
            horizon_value < 0
            for dimension in subgroups[candidate].values()
            for group in dimension.values()
            for horizon_value in group
        )
        for candidate in CANDIDATES
    }
    calibration_gate = {
        candidate: all(
            lower <= 0 <= upper
            for lower, upper in zip(
                calibration[candidate]["mean_normalized_residual"]["lower"],
                calibration[candidate]["mean_normalized_residual"]["upper"],
                strict=True,
            )
        )
        and all(
            value > 0 for value in calibration[candidate]["slope"]["lower"]
        )
        for candidate in CANDIDATES
    }
    preliminary_qualified = {
        "causal_xgb": xgb_gate
        and subgroup_gate["causal_xgb"]
        and calibration_gate["causal_xgb"],
        "tcn": tcn_gate and subgroup_gate["tcn"] and calibration_gate["tcn"],
    }
    if preliminary_qualified["causal_xgb"] and preliminary_qualified["tcn"]:
        tcn_strictly_better = selection_bound["upper"][0] < 0 and all(
            value <= 0 for value in comparison.mean(axis=0)
        )
        nominee = "tcn" if tcn_strictly_better else "causal_xgb"
    elif preliminary_qualified["causal_xgb"]:
        nominee = "causal_xgb"
    elif preliminary_qualified["tcn"]:
        nominee = "tcn"
    else:
        nominee = None
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_preliminary_loto_pending_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "tabular_report_sha256": sha256_file(
            args.tabular_root / "report.json"
        ),
        "tcn_report_sha256": sha256_file(args.tcn_root / "report.json"),
        "horizon_normalized_mse": {
            model: dict(zip(map(str, HORIZONS), values, strict=True))
            for model, values in losses.items()
        },
        "hard_comparator_by_horizon": dict(
            zip(map(str, HORIZONS), hard_models, strict=True)
        ),
        "superiority": superiority,
        "subgroup_point_differences": subgroups,
        "calibration": calibration,
        "tcn_minus_xgb_macro": selection_bound,
        "preliminary_qualified": preliminary_qualified,
        "preliminary_nominee": nominee,
        "leave_one_task_out_complete": False,
        "development_gate_complete": False,
        "confirmation_prompts_projected": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
