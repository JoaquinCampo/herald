"""Locked development-only direct and hierarchical magnitude models.

The v2 protocol is intentionally independent from :mod:`herald.magnitude`.
The only rows handed to Python by the parquet loader are rows whose prompt IDs
are in the frozen development partition.
"""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn import linear_model  # type: ignore[import-untyped]

from herald.magnitude import (
    CAUSAL_FEATURE_COLUMNS,
    KNOWN_COMPRESSORS,
    POSITION_BUCKET,
    REQUIRED_COLUMNS,
    _baseline_prediction,
    _feature_matrix,
    _fit_baseline,
    _metrics,
    _ratio_metrics,
    trajectory_weights,
    validate_provenance,
    validate_rows,
)
from herald.sweep_provenance import TABLE_CONTENT_SHA256_METADATA_KEY

SCHEMA_VERSION = "herald.magnitude_v2.v1"
PROTOCOL_SCHEMA_VERSION = "herald.magnitude_v2_protocol_lock.v1"
DEFAULT_EVIDENCE = Path(
    "results/recovered/ifeval-intervention-v1/magnitude_evidence_v2.json"
)
DEFAULT_LOCK = Path(
    "results/recovered/ifeval-intervention-v1/magnitude_v2_protocol_lock.json"
)


@dataclass(frozen=True, slots=True)
class Fold:
    index: int
    prompt_ids: tuple[str, ...]
    hash: str


@dataclass(frozen=True, slots=True)
class PlattCalibration:
    slope: float
    intercept: float
    prevalence: float

    def predict(self, raw: np.ndarray) -> np.ndarray:
        z = np.clip(
            self.slope * np.asarray(raw, dtype=np.float64) + self.intercept,
            -40.0,
            40.0,
        )
        return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True, slots=True)
class HurdleFit:
    occurrence: tuple[Any, ...]
    sign: tuple[Any, ...]
    positive: tuple[Any, ...]
    negative: tuple[Any, ...]
    occurrence_calibration: PlattCalibration
    sign_calibration: PlattCalibration
    positive_fallback: float
    negative_fallback: float
    features: tuple[str, ...]
    train_prompts: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FoldPrediction:
    fold: int
    compressor: str
    rows: tuple[dict[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def hash_prompt_ids(
    prompt_ids: list[str] | tuple[str, ...] | set[str],
) -> str:
    """Hash canonical sorted prompt IDs (the protocol uses a trailing LF)."""
    values = sorted(str(value) for value in prompt_ids)
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def fold_order(
    prompt_ids: list[str] | tuple[str, ...] | set[str],
) -> list[str]:
    return sorted(
        {str(prompt_id) for prompt_id in prompt_ids},
        key=lambda prompt_id: hashlib.sha256(
            f"herald-magnitude-v2-dev\0{prompt_id}".encode()
        ).hexdigest(),
    )


def make_folds(
    prompt_ids: list[str] | tuple[str, ...] | set[str],
    lock: dict[str, Any],
) -> tuple[Fold, ...]:
    """Build and verify the frozen contiguous prompt folds."""
    partition = lock["prompt_partitions"]
    raw_ids = [str(value) for value in prompt_ids]
    if len(raw_ids) != len(set(raw_ids)):
        raise ValueError("development prompt IDs contain duplicates")
    ordered = fold_order(raw_ids)
    expected_count = int(partition["development_count"])
    counts = [int(value) for value in partition["fold_counts"]]
    if len(ordered) != expected_count or sum(counts) != expected_count:
        raise ValueError(
            "development prompt count does not match protocol lock"
        )
    if hash_prompt_ids(ordered) != partition["development_prompt_ids_sha256"]:
        raise ValueError(
            "development prompt hash does not match protocol lock"
        )
    expected_hashes = list(partition["fold_prompt_ids_sha256"])
    if len(counts) != len(expected_hashes):
        raise ValueError("fold count/hash cardinality mismatch")
    folds: list[Fold] = []
    start = 0
    for index, count in enumerate(counts):
        ids = tuple(sorted(ordered[start : start + count]))
        start += count
        digest = hash_prompt_ids(ids)
        if digest != expected_hashes[index]:
            raise ValueError(
                f"fold {index} prompt hash does not match protocol lock"
            )
        folds.append(Fold(index, ids, digest))
    if (
        start != len(ordered)
        or len({p for fold in folds for p in fold.prompt_ids})
        != expected_count
    ):
        raise ValueError("outer folds are not an exact prompt partition")
    return tuple(folds)


def _json_sha256(path: Path) -> str:
    return sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read JSON artifact {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return value


def validate_lock_and_evidence(
    evidence_path: Path, lock_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = _load_json(evidence_path)
    lock = _load_json(lock_path)
    if (
        lock.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or lock.get("status") != "locked_before_v2_model_results"
    ):
        raise ValueError("protocol lock schema/status is invalid")
    if lock.get("source_evidence_sha256") != _json_sha256(evidence_path):
        raise ValueError("source evidence hash does not match protocol lock")
    split = evidence.get("split")
    partition = lock.get("prompt_partitions")
    if not isinstance(split, dict) or not isinstance(partition, dict):
        raise ValueError("evidence split or protocol partition is missing")
    development = tuple(
        str(value) for value in split.get("train_prompt_ids", ())
    )
    quarantine = tuple(
        str(value) for value in split.get("test_prompt_ids", ())
    )
    if set(development) & set(quarantine):
        raise ValueError("development and quarantined prompt IDs overlap")
    if len(set(development)) != int(partition["development_count"]):
        raise ValueError("development prompt count is not locked")
    if len(set(quarantine)) != int(partition["quarantined_count"]):
        raise ValueError("quarantined prompt count is not locked")
    if (
        hash_prompt_ids(development)
        != partition["development_prompt_ids_sha256"]
    ):
        raise ValueError("development prompt IDs do not match lock")
    if (
        hash_prompt_ids(quarantine)
        != partition["quarantined_prompt_ids_sha256"]
    ):
        raise ValueError("quarantined prompt IDs do not match lock")
    if (
        partition.get("source")
        != "magnitude_evidence_v2.json split.train_prompt_ids/test_prompt_ids"
    ):
        raise ValueError("protocol partition source is not authoritative")
    if not isinstance(lock.get("baseline_lock"), dict):
        raise ValueError("baseline lock is missing")
    config = lock.get("development_config")
    if not isinstance(config, dict):
        raise ValueError("development config is missing")
    if (
        tuple(config.get("model_seeds", ())) != (0, 1, 2)
        or int(config.get("outer_folds", 0)) != 5
        or int(config.get("inner_folds", 0)) != 4
    ):
        raise ValueError("model/fold configuration is not locked")
    if (
        int(config.get("num_boost_round", 0)) != 300
        or int(config.get("bootstrap_resamples", 0)) != 1000
        or config.get("probability_calibration")
        != "prompt-cross-fitted Platt"
    ):
        raise ValueError("boosting/calibration configuration is not locked")
    if list(partition.get("fold_counts", ())) != [31, 31, 31, 31, 30]:
        raise ValueError("outer fold counts are not locked")
    expected_baselines = {
        "expected_attention": {
            "mean": "ratio_position_mean",
            "median": "global_median",
        },
        "knorm": {
            "mean": "ratio_position_mean",
            "median": "ratio_position_median",
        },
        "streaming_llm": {
            "mean": "ratio_position_mean",
            "median": "global_median",
        },
    }
    if lock.get("baseline_lock") != expected_baselines:
        raise ValueError("baseline names differ from locked protocol")
    expected_xgb = {
        "colsample_bytree": 0.8,
        "eta": 0.05,
        "max_depth": 4,
        "min_child_weight": 1,
        "subsample": 0.8,
    }
    if config.get("xgboost") != expected_xgb:
        raise ValueError("XGBoost configuration differs from locked protocol")
    return evidence, lock


def load_development_rows(
    parquet_path: Path,
    *,
    evidence_path: Path = DEFAULT_EVIDENCE,
    lock_path: Path = DEFAULT_LOCK,
    compressors: tuple[str, ...] = KNOWN_COMPRESSORS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read only validated development rows from a frozen Parquet dataset.

    The scanner predicate is constructed before ``to_pylist``. Quarantined
    label columns therefore never enter Python memory.
    """
    evidence, lock = validate_lock_and_evidence(evidence_path, lock_path)
    expected_file_hash = str(evidence.get("input", {}).get("sha256", ""))
    actual_file_hash = sha256_file(parquet_path)
    if expected_file_hash and actual_file_hash != expected_file_hash:
        raise ValueError(
            "authoritative parquet SHA256 does not match evidence"
        )
    try:
        import pyarrow.dataset as ds
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required for the v2 dataset loader"
        ) from error
    schema = pq.read_schema(parquet_path)  # type: ignore[no-untyped-call]
    metadata = schema.metadata or {}
    validate_provenance(metadata)
    for key, expected in (
        (
            b"herald.sweep_config_sha256",
            evidence.get("sweep_config", {}).get("sha256"),
        ),
        (
            b"herald.source_manifest_sha256",
            evidence.get("source_manifest", {}).get("sha256"),
        ),
    ):
        if expected is not None and metadata.get(key, b"").decode() != str(
            expected
        ):
            raise ValueError(
                f"parquet provenance {key.decode()} does not match evidence"
            )
    if TABLE_CONTENT_SHA256_METADATA_KEY not in metadata:
        raise ValueError("parquet lacks table-content provenance")
    development = [
        str(value) for value in evidence["split"]["train_prompt_ids"]
    ]
    expected_features = tuple(
        str(name)
        for name in evidence.get("features", ())
        if name == "ratio" or name == "s" or str(name).startswith("feat__")
    )
    if len(expected_features) < 3:
        raise ValueError(
            "evidence does not contain the locked causal feature schema"
        )
    selected_columns = sorted(set(REQUIRED_COLUMNS).union(expected_features))
    missing_columns = set(selected_columns) - set(schema.names)
    if missing_columns:
        raise ValueError(
            f"parquet lacks locked columns: {sorted(missing_columns)}"
        )
    dataset = ds.dataset(  # type: ignore[no-untyped-call]
        parquet_path,
        format="parquet",
    )
    predicate = ds.field("task") == "ifeval"  # type: ignore[attr-defined, no-untyped-call]
    predicate = predicate & ds.field("prompt_id").isin(  # type: ignore[attr-defined, no-untyped-call]
        development
    )
    if compressors:
        predicate = predicate & ds.field("compressor").isin(  # type: ignore[attr-defined, no-untyped-call]
            list(compressors)
        )
    # This is the first conversion from Arrow values to Python.
    rows = dataset.to_table(
        columns=selected_columns,
        filter=predicate,
    ).to_pylist()
    checked = validate_rows(
        rows,
        compressors=compressors,
        metadata=metadata,
        require_provenance=True,
    )
    present = {str(row["prompt_id"]) for row in checked}
    if not present <= set(development) or present & set(
        evidence["split"]["test_prompt_ids"]
    ):
        raise ValueError("quarantined prompt entered development rows")
    forbidden = {
        name
        for row in checked
        for name in row
        if str(name).startswith("probe__")
        or str(name).startswith("post_switch__")
        or str(name).startswith("hybrid__")
    }
    if forbidden:
        raise ValueError(
            f"forbidden post-switch fields present: {sorted(forbidden)}"
        )
    provenance = {
        "parquet_sha256": actual_file_hash,
        "source_evidence_sha256": sha256_file(evidence_path),
        "protocol_lock_sha256": sha256_file(lock_path),
        "development_prompt_ids_sha256": hash_prompt_ids(development),
        "quarantined_prompt_ids_sha256": hash_prompt_ids(
            evidence["split"]["test_prompt_ids"]
        ),
        "expected_development_row_counts": {
            compressor: int(evidence["compressors"][compressor]["n_train"])
            + int(evidence["compressors"][compressor]["n_validation"])
            for compressor in compressors
        },
        "metadata": {
            key.decode(errors="replace"): value.decode(errors="replace")
            for key, value in metadata.items()
        },
        "features": list(expected_features),
    }
    return checked, provenance


def _fit_platt(
    raw: np.ndarray,
    target: np.ndarray,
    prevalence: float | None = None,
    weights: np.ndarray | None = None,
) -> PlattCalibration:
    raw = np.asarray(raw, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(raw) != len(target):
        raise ValueError("Platt margins and targets differ in length")
    if weights is None:
        weights = np.ones(len(target), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(weights) != len(target):
        raise ValueError("Platt weights and targets differ in length")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Platt weights must be finite and nonnegative")
    if len(target) and not np.any(weights > 0):
        raise ValueError("Platt weights must have positive mass")
    if prevalence is None:
        prevalence = (
            float(np.average(target, weights=weights)) if len(target) else 0.0
        )
    odds = float(np.clip(prevalence, 1e-6, 1.0 - 1e-6))
    intercept = math.log(odds / (1.0 - odds))
    if len(raw) == 0 or len(np.unique(target)) < 2:
        return PlattCalibration(0.0, intercept, prevalence)
    if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(target)):
        raise ValueError("Platt margins and targets must be finite")
    model = linear_model.LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
    )
    model.fit(
        raw.reshape(-1, 1),
        target,
        sample_weight=weights,
    )
    return PlattCalibration(
        float(model.coef_[0, 0]),
        float(model.intercept_[0]),
        prevalence,
    )


def prompt_inner_folds(
    prompt_ids: list[str] | tuple[str, ...], folds: int = 4
) -> tuple[tuple[str, ...], ...]:
    if folds < 2:
        raise ValueError("inner calibration requires at least two folds")
    ordered = fold_order(prompt_ids)
    if len(ordered) < folds:
        folds = len(ordered)
    result = tuple(tuple(ordered[index::folds]) for index in range(folds))
    if any(not fold for fold in result):
        raise ValueError("inner prompt fold is empty")
    return result


def _xgb_train(
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    target: np.ndarray,
    *,
    objective: str,
    seed: int,
    rounds: int,
    weights: np.ndarray,
) -> Any:
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError("xgboost is required for magnitude v2") from error
    matrix = xgb.DMatrix(
        _feature_matrix(rows, features),
        label=target,
        weight=weights,
    )
    params = {
        "objective": objective,
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "seed": int(seed),
        "nthread": -1,
    }
    return xgb.train(
        params,
        matrix,
        num_boost_round=int(rounds),
        verbose_eval=False,
    )


def _xgb_raw(
    model: Any,
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    output_margin: bool = False,
) -> np.ndarray:
    import xgboost as xgb

    return np.asarray(
        model.predict(
            xgb.DMatrix(_feature_matrix(rows, features)),
            output_margin=output_margin,
        ),
        dtype=np.float64,
    )


def cross_fitted_platt(
    rows: list[dict[str, Any]],
    target: np.ndarray,
    features: tuple[str, ...],
    *,
    objective: str = "binary:logistic",
    folds: int = 4,
    rounds: int = 300,
    seed: int = 0,
    seeds: tuple[int, ...] | None = None,
) -> PlattCalibration:
    """Fit calibration on prompt-disjoint inner ensemble predictions."""
    target = np.asarray(target, dtype=np.float64)
    if len(rows) != len(target):
        raise ValueError("calibration rows and targets differ in length")
    weights = trajectory_weights(rows)
    prevalence = (
        float(np.average(target, weights=weights)) if len(target) else 0.0
    )
    if len(target) == 0 or len(np.unique(target)) < 2:
        return _fit_platt(np.empty(0), np.empty(0), prevalence)
    model_seeds = seeds or (seed,)
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    inner = prompt_inner_folds(prompts, folds)
    out = np.full(len(rows), np.nan, dtype=np.float64)
    for validation_prompts in inner:
        validation = set(validation_prompts)
        train_mask = np.asarray(
            [str(row["prompt_id"]) not in validation for row in rows],
            dtype=bool,
        )
        validation_mask = ~train_mask
        train_rows = [
            row for row, keep in zip(rows, train_mask, strict=True) if keep
        ]
        val_rows = [
            row
            for row, keep in zip(rows, validation_mask, strict=True)
            if keep
        ]
        if not train_rows or not val_rows:
            continue
        margins = []
        for model_seed in model_seeds:
            model = _xgb_train(
                train_rows,
                features,
                target[train_mask],
                objective=objective,
                seed=model_seed,
                rounds=rounds,
                weights=trajectory_weights(train_rows),
            )
            margins.append(
                _xgb_raw(
                    model,
                    val_rows,
                    features,
                    output_margin=True,
                )
            )
        out[validation_mask] = np.mean(np.stack(margins), axis=0)
    mask = np.isfinite(out)
    return _fit_platt(
        out[mask],
        target[mask],
        float(np.average(target[mask], weights=weights[mask]))
        if np.any(mask)
        else prevalence,
        weights[mask],
    )


def _fit_hurdle(
    train_rows: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    seed: int,
    rounds: int,
    inner_folds: int,
    seeds: tuple[int, ...] | None = None,
) -> HurdleFit:
    model_seeds = seeds or (seed,)
    y = np.asarray([float(row["dq"]) for row in train_rows], dtype=np.float64)
    weights = trajectory_weights(train_rows)
    occurrence = (y != 0).astype(np.float64)
    nonzero = y != 0
    positive = y > 0
    negative = y < 0
    calibration_occurrence = cross_fitted_platt(
        train_rows,
        occurrence,
        features,
        folds=inner_folds,
        rounds=rounds,
        seeds=model_seeds,
    )
    sign_rows = [
        row for row, keep in zip(train_rows, nonzero, strict=True) if keep
    ]
    sign_target = (y[nonzero] > 0).astype(np.float64)
    calibration_sign = cross_fitted_platt(
        sign_rows,
        sign_target,
        features,
        folds=inner_folds,
        rounds=rounds,
        seeds=model_seeds,
    )

    def fit_ensemble(
        selected_rows: list[dict[str, Any]],
        selected_target: np.ndarray,
        objective: str,
        selected_weights: np.ndarray,
        seed_offset: int = 0,
    ) -> tuple[Any, ...]:
        if not selected_rows or len(np.unique(selected_target)) < (
            2 if objective == "binary:logistic" else 1
        ):
            return ()
        return tuple(
            _xgb_train(
                selected_rows,
                features,
                selected_target,
                objective=objective,
                seed=model_seed + seed_offset,
                rounds=rounds,
                weights=selected_weights,
            )
            for model_seed in model_seeds
        )

    sign_models = fit_ensemble(
        sign_rows,
        sign_target,
        "binary:logistic",
        trajectory_weights(sign_rows),
    )
    occurrence_models = fit_ensemble(
        train_rows,
        occurrence,
        "binary:logistic",
        weights,
    )
    pos_rows = [
        row for row, keep in zip(train_rows, positive, strict=True) if keep
    ]
    neg_rows = [
        row for row, keep in zip(train_rows, negative, strict=True) if keep
    ]
    pos_models = fit_ensemble(
        pos_rows,
        y[positive],
        "reg:squarederror",
        trajectory_weights(pos_rows),
    )
    neg_models = fit_ensemble(
        neg_rows,
        -y[negative],
        "reg:squarederror",
        trajectory_weights(neg_rows),
    )
    positive_fallback = (
        float(
            np.average(
                y[positive],
                weights=trajectory_weights(pos_rows),
            )
        )
        if pos_rows
        else 0.0
    )
    negative_fallback = (
        float(
            np.average(
                -y[negative],
                weights=trajectory_weights(neg_rows),
            )
        )
        if neg_rows
        else 0.0
    )
    return HurdleFit(
        occurrence_models,
        sign_models,
        pos_models,
        neg_models,
        calibration_occurrence,
        calibration_sign,
        positive_fallback,
        negative_fallback,
        features,
        tuple(sorted({str(row["prompt_id"]) for row in train_rows})),
    )


def signed_mixture_prediction(
    p_nz: np.ndarray | float,
    p_sign: np.ndarray | float,
    m_plus: np.ndarray | float,
    m_minus: np.ndarray | float,
) -> np.ndarray:
    return np.asarray(
        np.asarray(p_nz)
        * (
            np.asarray(p_sign) * np.asarray(m_plus)
            - (1.0 - np.asarray(p_sign)) * np.asarray(m_minus)
        ),
        dtype=np.float64,
    )


def _ensemble_raw(
    models: tuple[Any, ...],
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
    *,
    output_margin: bool = False,
) -> np.ndarray:
    if not models:
        return np.empty(0, dtype=np.float64)
    return np.asarray(
        np.mean(
            np.stack(
                [
                    _xgb_raw(
                        model,
                        rows,
                        features,
                        output_margin=output_margin,
                    )
                    for model in models
                ]
            ),
            axis=0,
        ),
        dtype=np.float64,
    )


def _hurdle_predict(
    fit: HurdleFit, rows: list[dict[str, Any]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if fit.occurrence:
        p_nz = fit.occurrence_calibration.predict(
            _ensemble_raw(
                fit.occurrence,
                rows,
                fit.features,
                output_margin=True,
            )
        )
    else:
        p_nz = np.full(len(rows), fit.occurrence_calibration.prevalence)
    if fit.sign:
        p_sign = fit.sign_calibration.predict(
            _ensemble_raw(
                fit.sign,
                rows,
                fit.features,
                output_margin=True,
            )
        )
    else:
        p_sign = np.full(len(rows), fit.sign_calibration.prevalence)
    p_plus = np.full(len(rows), fit.positive_fallback)
    p_minus = np.full(len(rows), fit.negative_fallback)
    if fit.positive:
        p_plus = np.maximum(
            0.0,
            _ensemble_raw(fit.positive, rows, fit.features),
        )
    if fit.negative:
        p_minus = np.maximum(
            0.0,
            _ensemble_raw(fit.negative, rows, fit.features),
        )
    return (
        signed_mixture_prediction(p_nz, p_sign, p_plus, p_minus),
        p_nz,
        p_sign,
        p_plus,
        p_minus,
    )


def _fit_m0(
    train_rows: list[dict[str, Any]],
    features: tuple[str, ...],
    seeds: tuple[int, ...],
    rounds: int,
) -> tuple[Any, ...]:
    y = np.asarray([float(row["dq"]) for row in train_rows], dtype=np.float64)
    weights = trajectory_weights(train_rows)
    return tuple(
        _xgb_train(
            train_rows,
            features,
            y,
            objective="reg:squarederror",
            seed=seed,
            rounds=rounds,
            weights=weights,
        )
        for seed in seeds
    )


def _m0_predict(
    models: tuple[Any, ...],
    rows: list[dict[str, Any]],
    features: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        np.mean(
            np.stack([_xgb_raw(model, rows, features) for model in models]),
            axis=0,
        ),
        dtype=np.float64,
    )


def _baseline_prevalence(
    rows: list[dict[str, Any]], target: np.ndarray, keys: tuple[str, ...]
) -> dict[tuple[Any, ...], float]:
    weights = trajectory_weights(rows)
    groups: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[
            tuple(
                int(row["s"]) // POSITION_BUCKET
                if key == "position_bucket"
                else row[key]
                for key in keys
            )
        ].append(index)
    return {
        key: float(np.average(target[idx], weights=weights[idx]))
        for key, idx in (
            (key, np.asarray(value, dtype=np.int64))
            for key, value in groups.items()
        )
    }


def _prevalence_prediction(
    rows: list[dict[str, Any]],
    values: dict[tuple[Any, ...], float],
    global_value: float,
    keys: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [
            values.get(
                tuple(
                    int(row["s"]) // POSITION_BUCKET
                    if key == "position_bucket"
                    else row[key]
                    for key in keys
                ),
                global_value,
            )
            for row in rows
        ],
        dtype=np.float64,
    )


def paired_prompt_bootstrap(
    rows: list[dict[str, Any]],
    m0: np.ndarray,
    m1: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 1729,
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    grouped = {
        prompt: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if str(row["prompt_id"]) == prompt
            ],
            dtype=np.int64,
        )
        for prompt in prompts
    }
    target = np.asarray([float(row["dq"]) for row in rows], dtype=np.float64)
    weights = trajectory_weights(rows)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    for draw in rng.integers(
        0,
        len(prompts),
        size=(resamples, len(prompts)),
    ):
        indices = np.concatenate(
            [grouped[prompts[int(index)]] for index in draw]
        )
        sampled_rows = [rows[int(index)] for index in indices]
        sampled_weights = weights[indices]
        mse0 = _ratio_metrics(
            target[indices],
            m0[indices],
            sampled_weights,
            sampled_rows,
        )["overall"]["mse"]
        mse1 = _ratio_metrics(
            target[indices],
            m1[indices],
            sampled_weights,
            sampled_rows,
        )["overall"]["mse"]
        deltas.append(float(mse1) - float(mse0))
    values = np.asarray(deltas)
    return {
        "resamples": resamples,
        "seed": seed,
        "metric": "ratio_macro_mse_m1_minus_m0",
        "estimate": float(np.mean(values)),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
    }


def bootstrap_locked_mean_skill(
    rows: list[dict[str, Any]],
    m1: np.ndarray,
    baseline: np.ndarray,
    *,
    resamples: int = 1000,
    seed: int = 1729,
) -> dict[str, Any]:
    """Prompt bootstrap of ratio-macro MSE skill against locked mean."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    grouped = {
        prompt: np.asarray(
            [
                i
                for i, row in enumerate(rows)
                if str(row["prompt_id"]) == prompt
            ],
            dtype=np.int64,
        )
        for prompt in prompts
    }
    target = np.asarray([float(row["dq"]) for row in rows], dtype=np.float64)
    weights = trajectory_weights(rows)
    point_mse = float(
        _ratio_metrics(target, m1, weights, rows)["overall"]["mse"]
    )
    point_base = float(
        _ratio_metrics(target, baseline, weights, rows)["overall"]["mse"]
    )
    rng = np.random.default_rng(seed)
    skills: list[float] = []
    for draw in rng.integers(0, len(prompts), size=(resamples, len(prompts))):
        indices = np.concatenate([grouped[prompts[int(i)]] for i in draw])
        sampled_rows = [rows[int(i)] for i in indices]
        sampled_target = target[indices]
        sampled_weights = weights[indices]
        model_mse = float(
            _ratio_metrics(
                sampled_target,
                m1[indices],
                sampled_weights,
                sampled_rows,
            )["overall"]["mse"]
        )
        base_mse = float(
            _ratio_metrics(
                sampled_target,
                baseline[indices],
                sampled_weights,
                sampled_rows,
            )["overall"]["mse"]
        )
        skills.append((base_mse - model_mse) / base_mse if base_mse else 0.0)
    values = np.asarray(skills, dtype=np.float64)
    return {
        "resamples": resamples,
        "seed": seed,
        "metric": "ratio_macro_mse_skill_vs_locked_mean",
        "estimate": (point_base - point_mse) / point_base
        if point_base
        else 0.0,
        "lower": float(np.quantile(values, 0.05)),
        "upper": float(np.quantile(values, 0.95)),
    }


def _model_metrics(
    rows: list[dict[str, Any]], target: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    weights = trajectory_weights(rows)
    result = {
        "ratio_macro": _ratio_metrics(target, prediction, weights, rows)[
            "overall"
        ],
        "positive_rows": _metrics(
            target[target > 0], prediction[target > 0], weights[target > 0]
        ),
        "nonzero_rows": _metrics(
            target[target != 0], prediction[target != 0], weights[target != 0]
        ),
        "negative_rows": _metrics(
            target[target < 0], prediction[target < 0], weights[target < 0]
        ),
        "major_rows": _metrics(
            target[target >= 0.5],
            prediction[target >= 0.5],
            weights[target >= 0.5],
        ),
        "prediction_mean": float(np.average(prediction, weights=weights)),
        "target_mean": float(np.average(target, weights=weights)),
    }
    return result


def fit_development(
    rows: list[dict[str, Any]],
    *,
    lock: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit nested prompt OOF models and return exact cell predictions."""
    if not rows:
        raise ValueError("development rows are empty")
    if provenance is None:
        raise ValueError("validated loader provenance is required")
    compressors = tuple(sorted({str(row["compressor"]) for row in rows}))
    if compressors != tuple(sorted(KNOWN_COMPRESSORS)):
        raise ValueError(
            "development rows do not contain every locked compressor"
        )
    folds = make_folds(
        sorted({str(row["prompt_id"]) for row in rows}),
        lock,
    )
    locked_features = tuple(
        str(name)
        for name in provenance.get("features", ())
        if str(name) not in {"ratio", "s"}
    )
    if set(locked_features) != set(CAUSAL_FEATURE_COLUMNS):
        raise ValueError("feature schema differs from locked causal features")
    features = tuple(sorted(locked_features))
    grids = {
        compressor: {
            (str(row["prompt_id"]), float(row["ratio"]), int(row["s"]))
            for row in rows
            if str(row["compressor"]) == compressor
        }
        for compressor in compressors
    }
    expected_counts = provenance.get("expected_development_row_counts")
    if not isinstance(expected_counts, dict) or set(expected_counts) != set(
        compressors
    ):
        raise ValueError("validated per-compressor row counts are required")
    for compressor in compressors:
        expected = int(expected_counts[compressor])
        actual = len(grids[compressor])
        if actual != expected:
            raise ValueError(
                f"{compressor} development row count {actual} != {expected}"
            )
    first_grid = grids[compressors[0]]
    if any(grid != first_grid for grid in grids.values()):
        raise ValueError("compressor grids differ in development rows")
    features = tuple(name for name in features if name not in {"ratio", "s"})
    if not features:
        raise ValueError("no locked causal features")
    config = lock["development_config"]
    seeds = tuple(int(value) for value in config["model_seeds"])
    rounds = int(config["num_boost_round"])
    inner = int(config["inner_folds"])
    oof: list[dict[str, Any]] = []
    for compressor in compressors:
        comp_rows = [
            row for row in rows if str(row["compressor"]) == compressor
        ]
        for fold in folds:
            test_prompts = set(fold.prompt_ids)
            train = [
                row
                for row in comp_rows
                if str(row["prompt_id"]) not in test_prompts
            ]
            test = [
                row
                for row in comp_rows
                if str(row["prompt_id"]) in test_prompts
            ]
            if not train or not test:
                raise ValueError(
                    f"fold {fold.index} has empty train/test partition"
                )
            m0_fit = _fit_m0(train, features, seeds, rounds)
            m1_fit = _fit_hurdle(
                train,
                features,
                seed=seeds[0],
                seeds=seeds,
                rounds=rounds,
                inner_folds=inner,
            )
            pred0 = _m0_predict(m0_fit, test, features)
            pred1, p_nz, p_sign, m_plus, m_minus = _hurdle_predict(
                m1_fit,
                test,
            )
            baseline_names = lock["baseline_lock"][compressor]
            fit_baselines: dict[str, Any] = {}
            baseline_rows = [
                dict(row, position_bucket=int(row["s"]) // POSITION_BUCKET)
                for row in train
            ]
            requested_names = {
                str(value) for value in baseline_names.values()
            }
            for name in tuple(requested_names):
                if "ratio_position" in name:
                    requested_names.add("ratio_" + name.rsplit("_", 1)[-1])
            for name in sorted(requested_names):
                statistic = "median" if "median" in name else "mean"
                baseline_keys = (
                    ("ratio", "position_bucket")
                    if "ratio_position" in name
                    else (("ratio",) if name.startswith("ratio_") else ())
                )
                fallback = fit_baselines.get("ratio_" + statistic)
                fit_baselines[name] = _fit_baseline(
                    baseline_rows,
                    name,
                    statistic,
                    baseline_keys,
                    fallback=fallback,
                )
            train_target = np.asarray(
                [float(row["dq"]) for row in train], dtype=np.float64
            )
            train_occurrence = (train_target != 0).astype(float)
            train_positive = (train_target > 0).astype(float)
            ratio_occurrence = _baseline_prevalence(
                baseline_rows,
                train_occurrence,
                ("ratio", "position_bucket"),
            )
            ratio_occurrence_global = float(
                np.average(
                    train_occurrence, weights=trajectory_weights(train)
                )
            )
            ratio_positive = _baseline_prevalence(
                baseline_rows,
                train_positive,
                ("ratio", "position_bucket"),
            )
            ratio_positive_global = float(
                np.average(train_positive, weights=trajectory_weights(train))
            )
            test_occurrence_baseline = _prevalence_prediction(
                test,
                ratio_occurrence,
                ratio_occurrence_global,
                ("ratio", "position_bucket"),
            )
            test_positive_baseline = _prevalence_prediction(
                test,
                ratio_positive,
                ratio_positive_global,
                ("ratio", "position_bucket"),
            )
            for row, a, b, pn, ps, mp, mm, bo, bp in zip(
                test,
                pred0,
                pred1,
                p_nz,
                p_sign,
                m_plus,
                m_minus,
                test_occurrence_baseline,
                test_positive_baseline,
                strict=True,
            ):
                selected_mean = str(baseline_names["mean"])
                selected_median = str(baseline_names["median"])
                baseline_mean = _baseline_prediction(
                    fit_baselines[selected_mean],
                    dict(
                        row, position_bucket=int(row["s"]) // POSITION_BUCKET
                    ),
                )
                baseline_median = _baseline_prediction(
                    fit_baselines[selected_median],
                    dict(
                        row, position_bucket=int(row["s"]) // POSITION_BUCKET
                    ),
                )
                oof.append(
                    {
                        "model": row["model"],
                        "task": row["task"],
                        "prompt_id": str(row["prompt_id"]),
                        "compressor": compressor,
                        "ratio": float(row["ratio"]),
                        "s": int(row["s"]),
                        "dq": float(row["dq"]),
                        "fold": fold.index,
                        "m0_prediction": float(a),
                        "m1_prediction": float(b),
                        "m1_p_nonzero": float(pn),
                        "m1_p_positive": float(pn * ps),
                        "m1_m_plus": float(mp),
                        "m1_m_minus": float(mm),
                        "locked_baseline_mean": float(baseline_mean),
                        "locked_baseline_median": float(baseline_median),
                        "fold_occurrence_prevalence": float(bo),
                        "fold_positive_prevalence": float(bp),
                        "fold_train_prompt_ids_sha256": hash_prompt_ids(
                            sorted({str(r["prompt_id"]) for r in train})
                        ),
                        "fold_test_prompt_ids_sha256": fold.hash,
                    }
                )
    cell_keys = [
        (r["compressor"], r["prompt_id"], r["ratio"], r["s"]) for r in oof
    ]
    if len(cell_keys) != len(set(cell_keys)) or len(cell_keys) != len(rows):
        raise ValueError("OOF predictions are not exact-once cell coverage")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "input": provenance or {},
        "protocol": {"lock": lock, "features": list(features)},
        "folds": [
            {
                "index": fold.index,
                "count": len(fold.prompt_ids),
                "prompt_ids_sha256": fold.hash,
                "prompt_ids": list(fold.prompt_ids),
            }
            for fold in folds
        ],
        "compressors": {},
    }
    for compressor in compressors:
        comp = [row for row in oof if row["compressor"] == compressor]
        target = np.asarray([row["dq"] for row in comp], dtype=np.float64)
        m0_prediction = np.asarray(
            [row["m0_prediction"] for row in comp], dtype=np.float64
        )
        m1_prediction = np.asarray(
            [row["m1_prediction"] for row in comp], dtype=np.float64
        )
        bmean = np.asarray(
            [row["locked_baseline_mean"] for row in comp],
            dtype=np.float64,
        )
        bmedian = np.asarray(
            [row["locked_baseline_median"] for row in comp],
            dtype=np.float64,
        )
        weights = trajectory_weights(comp)
        occurrence = (target != 0).astype(float)
        positive = (target > 0).astype(float)
        m0_metrics = _model_metrics(comp, target, m0_prediction)
        m1_metrics = _model_metrics(comp, target, m1_prediction)
        m0_metrics["per_ratio"] = _ratio_metrics(
            target,
            m0_prediction,
            weights,
            comp,
        )["per_ratio"]
        m1_metrics["per_ratio"] = _ratio_metrics(
            target,
            m1_prediction,
            weights,
            comp,
        )["per_ratio"]
        fold_occurrence = np.asarray(
            [row["fold_occurrence_prevalence"] for row in comp],
            dtype=np.float64,
        )
        fold_positive = np.asarray(
            [row["fold_positive_prevalence"] for row in comp],
            dtype=np.float64,
        )
        occurrence_prediction = np.asarray(
            [row["m1_p_nonzero"] for row in comp],
            dtype=np.float64,
        )
        positive_prediction = np.asarray(
            [row["m1_p_positive"] for row in comp],
            dtype=np.float64,
        )
        occurrence_brier = float(
            np.average(
                (occurrence_prediction - occurrence) ** 2, weights=weights
            )
        )
        occurrence_brier_baseline = float(
            np.average((fold_occurrence - occurrence) ** 2, weights=weights)
        )
        locked_mean_metrics = _model_metrics(comp, target, bmean)
        locked_median_metrics = _model_metrics(comp, target, bmedian)
        locked_mean_metrics["per_ratio"] = _ratio_metrics(
            target, bmean, weights, comp
        )["per_ratio"]
        locked_median_metrics["per_ratio"] = _ratio_metrics(
            target, bmedian, weights, comp
        )["per_ratio"]

        def skill(
            model_value: float | None,
            baseline_value: float | None,
        ) -> float:
            if (
                model_value is None
                or baseline_value is None
                or baseline_value == 0.0
            ):
                return 0.0
            return (baseline_value - model_value) / baseline_value

        m1_vs_mean = {
            "mse": skill(
                m1_metrics["ratio_macro"]["mse"],
                locked_mean_metrics["ratio_macro"]["mse"],
            ),
            "positive_mse": skill(
                m1_metrics["positive_rows"]["mse"],
                locked_mean_metrics["positive_rows"]["mse"],
            ),
            "major_mse": skill(
                m1_metrics["major_rows"]["mse"],
                locked_mean_metrics["major_rows"]["mse"],
            ),
        }
        m1_vs_median = {
            "mae": skill(
                m1_metrics["ratio_macro"]["mae"],
                locked_median_metrics["ratio_macro"]["mae"],
            )
        }
        positive_brier = float(
            np.average((positive_prediction - positive) ** 2, weights=weights)
        )
        positive_brier_baseline = float(
            np.average((fold_positive - positive) ** 2, weights=weights)
        )
        baseline_bootstrap = bootstrap_locked_mean_skill(
            comp,
            m1_prediction,
            bmean,
            resamples=int(config["bootstrap_resamples"]),
            seed=1729,
        )
        report["compressors"][compressor] = {
            "baseline_names": lock["baseline_lock"][compressor],
            "m0": m0_metrics,
            "locked_mean": locked_mean_metrics,
            "locked_median": locked_median_metrics,
            "m1_vs_locked_mean": m1_vs_mean,
            "m1_vs_locked_median": m1_vs_median,
            "bootstrap_locked_mean": baseline_bootstrap,
            "m1": m1_metrics,
            "occurrence_brier": occurrence_brier,
            "occurrence_brier_baseline": occurrence_brier_baseline,
            "occurrence_brier_skill": (
                (occurrence_brier_baseline - occurrence_brier)
                / occurrence_brier_baseline
                if occurrence_brier_baseline
                else 0.0
            ),
            "positive_risk_brier": positive_brier,
            "positive_risk_brier_baseline": positive_brier_baseline,
            "positive_risk_brier_skill": (
                (positive_brier_baseline - positive_brier)
                / positive_brier_baseline
                if positive_brier_baseline
                else 0.0
            ),
            "occurrence_target_rate": float(
                np.average(occurrence, weights=weights)
            ),
            "positive_target_rate": float(
                np.average(positive, weights=weights)
            ),
            "paired_bootstrap": paired_prompt_bootstrap(
                comp,
                m0_prediction,
                m1_prediction,
                resamples=int(config["bootstrap_resamples"]),
                seed=1729,
            ),
        }
    criteria: dict[str, bool] = {}
    per_compressor: dict[str, dict[str, Any]] = {}
    for compressor, value in report["compressors"].items():
        m1_metrics = value["m1"]
        mean_metrics = value["locked_mean"]
        local_criteria = {
            "every_ratio_mse_skill_gt_zero": all(
                float(m1_metrics["per_ratio"][ratio]["mse"])
                < float(mean_metrics["per_ratio"][ratio]["mse"])
                for ratio in m1_metrics["per_ratio"]
            ),
            "mae_skill_gt_zero": (value["m1_vs_locked_median"]["mae"] > 0.0),
            "mse_bootstrap_lower_gt_zero": (
                float(value["bootstrap_locked_mean"]["lower"]) > 0.0
            ),
            "positive_mse_not_worse": (
                value["m1_vs_locked_mean"]["positive_mse"] >= 0.0
            ),
            "major_mse_not_worse": (
                value["m1_vs_locked_mean"]["major_mse"] >= 0.0
            ),
            "positive_risk_brier_skill_gt_zero": (
                float(value["positive_risk_brier_skill"]) > 0.0
            ),
        }
        criteria.update(
            {
                f"{compressor}.{name}": passed
                for name, passed in local_criteria.items()
            }
        )
        per_compressor[compressor] = {
            "criteria": local_criteria,
            "all_criteria_pass": all(local_criteria.values()),
        }
    passing = [
        compressor
        for compressor, result in per_compressor.items()
        if result["all_criteria_pass"]
    ]
    report["development_gate"] = {
        "criteria": criteria,
        "per_compressor": per_compressor,
        "passing_compressors": passing,
        "any_compressor_pass": bool(passing),
        "all_comparison_criteria_pass": all(criteria.values()),
        "locked_criteria": lock.get("development_gate", {}),
        "status": "evaluated",
    }
    return report, oof


def write_oof_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to write v2 OOF predictions"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)  # type: ignore[no-untyped-call]


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
