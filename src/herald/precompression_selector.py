"""Prompt-disjoint training for irreversible pre-compression selection."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from herald.grace_window import GateBundle

XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": 1,
}
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class SelectorSplit:
    train_prompt_ids: list[str]
    calibration_prompt_ids: list[str]
    target_prompt_ids: list[str]


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    mean_opportunity: float
    n_commits: int
    damage_rate: float
    major_damage_rate: float


def deterministic_selector_split(
    prompt_ids: Sequence[str], target_prompt_ids: Sequence[str]
) -> SelectorSplit:
    """Freeze sorted development train/calibration identities."""
    all_ids = set(prompt_ids)
    target = set(target_prompt_ids)
    missing = target - all_ids
    if missing:
        missing_preview = sorted(missing)[:5]
        raise ValueError(
            f"target prompts missing from selector data: {missing_preview}"
        )
    development = sorted(all_ids - target)
    calibration = development[::5]
    calibration_set = set(calibration)
    train = [
        prompt_id
        for prompt_id in development
        if prompt_id not in calibration_set
    ]
    if not train or not calibration or not target:
        raise ValueError(
            "selector split requires train, calibration, and target prompts"
        )
    return SelectorSplit(train, calibration, sorted(target))


def feature_columns(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return the frozen feature-only selector columns."""
    if not rows:
        raise ValueError("selector rows are empty")
    columns = sorted(key for key in rows[0] if key.startswith("feat__"))
    if "ratio" not in columns:
        columns.append("ratio")
    if not columns:
        raise ValueError("selector data has no pre-compression features")
    return columns


def featurize(
    rows: Sequence[dict[str, Any]], columns: Sequence[str]
) -> np.ndarray:
    matrix = np.full((len(rows), len(columns)), np.nan, dtype=np.float32)
    for row_index, row in enumerate(rows):
        for column_index, column in enumerate(columns):
            value = row.get(column)
            if value is not None:
                matrix[row_index, column_index] = float(value)
    return matrix


def safe_labels(rows: Sequence[dict[str, Any]]) -> np.ndarray:
    """Safe means within margin and outside major damage."""
    return np.asarray(
        [
            float(row["dq"]) <= 0.01 and not bool(row["major_damage"])
            for row in rows
        ],
        dtype=np.float32,
    )


def select_safe_threshold(
    rows: Sequence[dict[str, Any]], scores: Sequence[float]
) -> ThresholdResult:
    """Maximize opportunity among thresholds with zero calibration damage."""
    if len(rows) != len(scores) or not rows:
        raise ValueError("selector calibration rows and scores must align")
    candidates = sorted({float(score) for score in scores}, reverse=True)
    candidates.append(float("inf"))
    best = ThresholdResult(float("inf"), 0.0, 0, 0.0, 0.0)
    for threshold in candidates:
        selected: list[dict[str, Any]] = []
        grouped: dict[str, list[tuple[dict[str, Any], float]]] = {}
        for row, score in zip(rows, scores, strict=True):
            grouped.setdefault(str(row["prompt_id"]), []).append(
                (row, float(score))
            )
        for group in grouped.values():
            eligible = [item for item in group if item[1] >= threshold]
            if eligible:
                selected.append(
                    min(eligible, key=lambda item: int(item[0]["s"]))[0]
                )
        if not selected:
            result = ThresholdResult(threshold, 0.0, 0, 0.0, 0.0)
        else:
            damage_rate = float(
                np.mean([float(row["dq"]) > 0.01 for row in selected])
            )
            major_rate = float(
                np.mean([bool(row["major_damage"]) for row in selected])
            )
            opportunity = float(
                np.sum(
                    [
                        max(
                            0.0, 1.0 - float(row["s"]) / float(row["ref_len"])
                        )
                        for row in selected
                    ]
                )
                / len(grouped)
            )
            result = ThresholdResult(
                threshold, opportunity, len(selected), damage_rate, major_rate
            )
        if (
            result.damage_rate == 0.0
            and result.major_damage_rate == 0.0
            and (result.mean_opportunity, result.threshold)
            > (best.mean_opportunity, best.threshold)
        ):
            best = result
    return best


def fit_selector_bundle(
    train_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    *,
    compressor: str,
    meta: dict[str, Any],
    num_boost_round: int = 300,
) -> GateBundle:
    """Fit the frozen ensemble and calibrate only on disjoint prompts."""
    import xgboost as xgb

    train_ids = {str(row["prompt_id"]) for row in train_rows}
    calibration_ids = {str(row["prompt_id"]) for row in calibration_rows}
    if train_ids & calibration_ids:
        raise ValueError("selector train and calibration prompts overlap")
    columns = feature_columns(train_rows)
    train_matrix = featurize(train_rows, columns)
    labels = safe_labels(train_rows)
    calibration_matrix = featurize(calibration_rows, columns)
    boosters = []
    score_sum = np.zeros(len(calibration_rows), dtype=np.float64)
    for seed in SEEDS:
        booster = xgb.train(
            {**XGB_PARAMS, "seed": seed},
            xgb.DMatrix(train_matrix, label=labels),
            num_boost_round=num_boost_round,
        )
        boosters.append(booster)
        score_sum += booster.predict(xgb.DMatrix(calibration_matrix))
    calibration_scores = score_sum / len(SEEDS)
    threshold = select_safe_threshold(
        calibration_rows, calibration_scores.tolist()
    )
    bundle_meta = {
        **meta,
        "variant": "irreversible_feature_only_precompression",
        "n_train_rows": len(train_rows),
        "n_train_prompts": len(train_ids),
        "n_calibration_rows": len(calibration_rows),
        "n_calibration_prompts": len(calibration_ids),
        "calibration": threshold.__dict__,
        "xgb_params": XGB_PARAMS,
        "num_boost_round": num_boost_round,
        "seeds": list(SEEDS),
    }
    return GateBundle(
        compressor=compressor,
        g_tau=threshold.threshold,
        feature_cols=columns,
        boosters=boosters,
        meta=bundle_meta,
    )
