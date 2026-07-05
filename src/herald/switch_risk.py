"""Per-row worst-case risk model for the switch controller.

A gradient-boosted regressor predicts the expected worst-compressor
damage (max dq over the fit compressors at the same prompt, ratio,
and position) from online-known inputs only: task, ratio, position,
and the causal ``feat__*`` logit statistics of the reference stream.
The prediction is in dq units, the same currency as the locked cost
budget, so the safety cut is directly comparable to epsilon. The
held-out compressor never contributes labels.

The score exposed to the locked tau rule has three regions:

- risk <= cut: the raw predicted risk (graded), so tau can admit
  model-approved early switches;
- risk > cut at position 0: one flat wall value. Admitting the wall
  switches every remaining group at its first position, which busts
  the train budget, so tau can never pass it: savings saturate at the
  cut instead of creeping until the train budget is spent;
- risk > cut later: values graded above the wall, unreachable for the
  policy but finely ordered for catastrophe recall.

The cut is a conservatism knob selected by internal cross-validation
that holds one train primary compressor out of the fit pool (donors
always stay in fit) and requires the epsilon budget on it.
"""

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

PRE_WALL = 2.0
RISKY_BASE = 2.1
EARLINESS_SCALE = 0.0005
POSITION_SCALE = 2048.0
POINT_FIELDS: tuple[str, ...] = ("model", "task", "prompt_id", "ratio")
BASE_INPUT_FIELDS: tuple[str, ...] = ("ratio",)
TASKS: tuple[str, ...] = ("gsm8k", "humaneval", "ifeval")
DEFAULT_CUTS: tuple[float, ...] = (0.0025, 0.005, 0.01, 0.02, 0.04)
MODEL_CONFIGS: dict[str, dict[str, float | int]] = {
    "deep": {
        "max_depth": 6,
        "eta": 0.05,
        "min_child_weight": 10,
        "num_boost_round": 300,
    },
    "shallow": {
        "max_depth": 3,
        "eta": 0.03,
        "min_child_weight": 50,
        "num_boost_round": 800,
    },
}


def feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    """All causal feature columns present on the rows."""
    names: set[str] = set()
    for row in rows:
        names.update(key for key in row if key.startswith("feat__"))
    return sorted(names)


def build_worst_points(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe rows to grid points labelled with the worst dq.

    Features are computed from the reference stream, so they are
    identical across compressors at the same point; the label is the
    max dq over the compressors present.
    """
    points: dict[tuple[object, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in POINT_FIELDS) + (
            float(row["feat__position"]),
        )
        dq = float(row["dq"])
        existing = points.get(key)
        if existing is None:
            point = dict(row)
            point["worst_dq"] = dq
            points[key] = point
        elif dq > existing["worst_dq"]:
            existing["worst_dq"] = dq
    return list(points.values())


def build_consensus_points(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Dedupe rows to grid points labelled with the mean dq.

    The consensus label (mean damage over the compressors present)
    has a far higher transfer ceiling than the worst-case label:
    thresholding the true mean-of-others dq clears the 0.10 rung on
    every held-out compressor, while the max-of-others label is
    nearly binary on knorm. Averaging also reduces label noise.
    """
    points: dict[tuple[object, ...], dict[str, Any]] = {}
    values: dict[tuple[object, ...], list[float]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in POINT_FIELDS) + (
            float(row["feat__position"]),
        )
        values.setdefault(key, []).append(float(row["dq"]))
        if key not in points:
            points[key] = dict(row)
    out: list[dict[str, Any]] = []
    for key, point in points.items():
        point["consensus_dq"] = float(np.mean(values[key]))
        out.append(point)
    return out


def fit_consensus_model(
    points: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    kind: str = "classifier",
    config: str = "deep",
    seed: int = 0,
) -> Any:
    """Fit a consensus-damage head on deduped grid points.

    ``classifier`` predicts P(consensus dq > 0) and drives the safe
    region and the cut; ``magnitude`` regresses consensus dq and
    grades the risky region for catastrophe recall. Boosting rounds
    are chosen by early stopping on a prompt-disjoint validation
    slice, so capacity is tuned for unseen prompts.
    """
    import xgboost as xgb

    settings = MODEL_CONFIGS[config]
    x = featurize(points, features)
    dq = np.asarray(
        [float(point["consensus_dq"]) for point in points],
        dtype=np.float32,
    )
    if kind == "classifier":
        y = (dq > 0.0).astype(np.float32)
        objective = "binary:logistic"
    elif kind == "magnitude":
        y = dq
        objective = "reg:squarederror"
    else:
        raise ValueError(f"unknown consensus head kind: {kind!r}")
    holdout = np.asarray(
        [_validation_slice(point) for point in points], dtype=bool
    )
    params = {
        "objective": objective,
        "max_depth": int(settings["max_depth"]),
        "eta": float(settings["eta"]),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(settings["min_child_weight"]),
        "seed": seed,
        "nthread": -1,
    }
    return xgb.train(
        params,
        xgb.DMatrix(x[~holdout], label=y[~holdout]),
        num_boost_round=1500,
        evals=[(xgb.DMatrix(x[holdout], label=y[holdout]), "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )


def featurize(
    rows: Sequence[dict[str, Any]], features: Sequence[str]
) -> np.ndarray:
    """Model input matrix: task one-hot, ratio, causal features."""
    n = len(rows)
    width = len(TASKS) + len(BASE_INPUT_FIELDS) + len(features)
    out = np.full((n, width), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        task = str(row.get("task"))
        for j, name in enumerate(TASKS):
            out[i, j] = 1.0 if task == name else 0.0
        col = len(TASKS)
        for field in BASE_INPUT_FIELDS:
            value = row.get(field)
            out[i, col] = np.nan if value is None else float(value)
            col += 1
        for feature in features:
            value = row.get(feature)
            out[i, col] = np.nan if value is None else float(value)
            col += 1
    return out


def fit_risk_model(
    points: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    config: str = "deep",
    seed: int = 0,
) -> Any:
    """Fit the expected worst-case damage regressor (dq units)."""
    import xgboost as xgb

    settings = MODEL_CONFIGS[config]
    x = featurize(points, features)
    y = np.asarray(
        [float(point["worst_dq"]) for point in points],
        dtype=np.float32,
    )
    params = {
        "objective": "reg:squarederror",
        "max_depth": int(settings["max_depth"]),
        "eta": float(settings["eta"]),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(settings["min_child_weight"]),
        "seed": seed,
        "nthread": -1,
    }
    return xgb.train(
        params,
        xgb.DMatrix(x, label=y),
        num_boost_round=int(settings["num_boost_round"]),
    )


def predict_risk(
    model: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
) -> np.ndarray:
    """Predicted damage (or risk probability) per row."""
    import xgboost as xgb

    matrix = xgb.DMatrix(featurize(rows, features))
    best = getattr(model, "best_iteration", None)
    if best is not None:
        prediction = model.predict(matrix, iteration_range=(0, int(best) + 1))
    else:
        prediction = model.predict(matrix)
    return np.asarray(prediction, dtype=np.float64)


CUMULATIVE_BASES: tuple[str, ...] = (
    "feat__entropy",
    "feat__kl_prev",
    "feat__max_prob",
    "feat__margin_prob",
    "feat__h_alts",
)


def attach_cumulative_features(
    rows: Sequence[dict[str, Any]],
) -> None:
    """Add running mean/max/std of key stats since generation start.

    The stock causal features stop at 32-token windows; these running
    aggregates summarise the whole trajectory up to the current
    position (a prompt-difficulty fingerprint) using only the group's
    own earlier sampled points, so they stay causal and online-known.
    Values are identical across compressors at the same point.
    """
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in POINT_FIELDS)
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        _attach_group_cumulatives(members)


def _attach_group_cumulatives(
    members: list[dict[str, Any]],
) -> None:
    """Running aggregates over one decision group's positions."""
    by_position: dict[float, list[dict[str, Any]]] = {}
    for row in members:
        by_position.setdefault(float(row["feat__position"]), []).append(row)
    history: dict[str, list[float]] = {base: [] for base in CUMULATIVE_BASES}
    for position in sorted(by_position):
        rows_at = by_position[position]
        for base in CUMULATIVE_BASES:
            value = rows_at[0].get(base)
            if value is not None and math.isfinite(float(value)):
                history[base].append(float(value))
            stats = _running_stats(history[base])
            for row in rows_at:
                name = base[len("feat__") :]
                row[f"feat__cum_{name}_mean"] = stats[0]
                row[f"feat__cum_{name}_max"] = stats[1]
                row[f"feat__cum_{name}_std"] = stats[2]


def _running_stats(
    values: list[float],
) -> tuple[float, float, float]:
    """Mean, max, and std of the trajectory so far."""
    if not values:
        return (np.nan, np.nan, np.nan)
    spread = float(np.std(values)) if len(values) > 1 else 0.0
    return (float(np.mean(values)), float(np.max(values)), spread)


def compressor_severity(
    rows: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Mean dq per compressor on the fit rows (train-only scalar)."""
    totals: dict[str, list[float]] = {}
    for row in rows:
        totals.setdefault(str(row["compressor"]), []).append(float(row["dq"]))
    return {comp: float(np.mean(values)) for comp, values in totals.items()}


def fit_severity_model(
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    severity: dict[str, float],
    *,
    config: str = "deep",
    seed: int = 0,
    label_key: str = "dq",
) -> Any:
    """Fit damage as a function of (inputs, compressor severity).

    Every fit row is one (point, compressor) sample; the compressor
    enters ONLY through its severity scalar, so the model learns how
    damage scales with compressor harshness per row and can be
    queried at a hypothetical severity for an unseen compressor.
    Boosting rounds are chosen by early stopping on a prompt-disjoint
    validation slice, so capacity is tuned for unseen prompts.
    """
    import xgboost as xgb

    settings = MODEL_CONFIGS[config]
    x = featurize(rows, features)
    sev = np.asarray(
        [severity[str(row["compressor"])] for row in rows],
        dtype=np.float32,
    )
    x = np.column_stack([x, sev])
    y = np.asarray([float(row[label_key]) for row in rows], dtype=np.float32)
    holdout = np.asarray([_validation_slice(row) for row in rows], dtype=bool)
    train = xgb.DMatrix(x[~holdout], label=y[~holdout])
    val = xgb.DMatrix(x[holdout], label=y[holdout])
    params = {
        "objective": "reg:squarederror",
        "max_depth": int(settings["max_depth"]),
        "eta": float(settings["eta"]),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": int(settings["min_child_weight"]),
        "seed": seed,
        "nthread": -1,
    }
    return xgb.train(
        params,
        train,
        num_boost_round=2000,
        evals=[(val, "val")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )


def _validation_slice(row: dict[str, Any]) -> bool:
    """Deterministic ~15 percent prompt slice for early stopping."""
    import hashlib

    key = "\0".join(
        [
            "sev-val",
            str(row.get("model")),
            str(row.get("task")),
            str(row.get("prompt_id")),
        ]
    )
    return hashlib.sha256(key.encode()).digest()[0] < 38


def predict_severity_risk(
    model: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    severity_value: float,
) -> np.ndarray:
    """Predicted damage at a hypothetical compressor severity."""
    import xgboost as xgb

    x = featurize(rows, features)
    sev = np.full((x.shape[0], 1), severity_value, dtype=np.float32)
    matrix = xgb.DMatrix(np.column_stack([x, sev]))
    best = getattr(model, "best_iteration", None)
    if best is not None:
        prediction = model.predict(matrix, iteration_range=(0, int(best) + 1))
    else:
        prediction = model.predict(matrix)
    return np.asarray(prediction, dtype=np.float64)


def smooth_risk_causal(
    rows: Sequence[dict[str, Any]], risk: np.ndarray
) -> np.ndarray:
    """Max of own and previous sampled position's risk per group.

    Switching requires the local neighbourhood (current and previous
    sample) to look safe, which halves single-point false positives
    at the price of switching at most one sample later. Only past
    positions are used, so the transform stays causal.
    """
    order: dict[tuple[object, ...], list[tuple[float, int]]] = {}
    for idx, row in enumerate(rows):
        key = tuple(row.get(field) for field in POINT_FIELDS)
        order.setdefault(key, []).append((float(row["feat__position"]), idx))
    out = np.array(risk, dtype=np.float64, copy=True)
    for members in order.values():
        members.sort()
        for j in range(1, len(members)):
            prev_idx = members[j - 1][1]
            idx = members[j][1]
            out[idx] = max(risk[idx], risk[prev_idx])
    return out


def severity_level_value(severity: dict[str, float], level: str) -> float:
    """Resolve a severity knob level to a value, train-only."""
    values = sorted(severity.values())
    if level == "max":
        return values[-1]
    if level == "second":
        return values[-2] if len(values) > 1 else values[-1]
    if level == "mean":
        return float(np.mean(values))
    raise ValueError(f"unknown severity level: {level!r}")


def shape_scores(
    risk: np.ndarray,
    positions: np.ndarray,
    *,
    cut: float,
    magnitude: np.ndarray | None = None,
) -> np.ndarray:
    """Map predicted risk to the three-region controller score.

    The risky region is graded by predicted damage magnitude when
    available (better catastrophe ranking than the probability, which
    saturates over the risky mass), by the probability otherwise.
    """
    risk = np.asarray(risk, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    grade_by = (
        risk if magnitude is None else np.asarray(magnitude, dtype=np.float64)
    )
    earliness = np.clip(1.0 - positions / POSITION_SCALE, 0.0, 1.0)
    graded = (
        RISKY_BASE + np.clip(grade_by, 0.0, 1.0) + EARLINESS_SCALE * earliness
    )
    out = np.where(risk <= cut, risk, graded)
    wall = (risk > cut) & (positions <= 0.0)
    out[wall] = PRE_WALL
    return out


def score_rows(
    model: Any,
    rows: Sequence[dict[str, Any]],
    features: Sequence[str],
    *,
    cut: float,
    magnitude_model: Any | None = None,
) -> list[float]:
    """End-to-end controller scores for evaluation rows."""
    risk = predict_risk(model, rows, features)
    magnitude = (
        None
        if magnitude_model is None
        else predict_risk(magnitude_model, rows, features)
    )
    positions = np.asarray(
        [float(row["feat__position"]) for row in rows],
        dtype=np.float64,
    )
    return [
        float(value)
        for value in shape_scores(
            risk, positions, cut=cut, magnitude=magnitude
        )
    ]
