"""Strict, per-compressor IFEval switch-damage magnitude evidence.

This module keeps the training protocol deliberately small and fixed. It
loads only validated IFEval switch rows and creates one prompt-disjoint split
shared by all compressors. It fits independent baselines and regressors.
Controller thresholds and cross-compressor labels are excluded.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from herald.features import IncrementalDerived
from herald.sweep_provenance import (
    TABLE_CONTENT_SHA256_METADATA_KEY,
    sha256_file,
    table_content_sha256,
    validate_intervention_manifest,
)
from herald.switch_baselines import split_prompt_ids

KNOWN_COMPRESSORS: tuple[str, ...] = (
    "streaming_llm",
    "knorm",
    "expected_attention",
)
INTERVENTION_SEMANTICS = {
    "streaming_llm": "herald.cache_native_pending_v1",
    "knorm": "herald.cache_native_pending_v1",
    "expected_attention": "herald.matched_reprefill_v1",
}

CAUSAL_FEATURE_COLUMNS = frozenset(
    f"feat__{name}"
    for name in IncrementalDerived().names()
    if name != "position"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def rows_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash validated rows in a stable order for report provenance."""
    encoded = [
        json.dumps(
            dict(row), sort_keys=True, separators=(",", ":"), default=str
        )
        for row in rows
    ]
    digest = hashlib.sha256()
    for value in sorted(encoded):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


SCHEMA_VERSION = "herald.magnitude.v1"
DEFAULT_TEST_FRACTION = 0.25
DEFAULT_BOOTSTRAP_RESAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 1729
DEFAULT_MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
POSITION_BUCKET = 16
REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "model",
        "task",
        "prompt_id",
        "compressor",
        "intervention_semantics",
        "feature_timing",
        "ratio",
        "s",
        "q_ref",
        "q_control",
        "q_hybrid",
        "dq",
        "damaged",
        "major_damage",
    }
)


@dataclass(frozen=True, slots=True)
class PromptSplit:
    """One common outer split, represented by prompt IDs."""

    train_prompt_ids: tuple[str, ...]
    test_prompt_ids: tuple[str, ...]
    seed: int
    test_fraction: float
    hash: str


@dataclass(frozen=True, slots=True)
class Baseline:
    """Train-only grouped baseline."""

    name: str
    statistic: str
    keys: tuple[str, ...]
    global_value: float
    values: dict[tuple[object, ...], float]
    fallback: Baseline | None = None


@dataclass(frozen=True, slots=True)
class WeightedHuber:
    """Huber model with trajectory-weighted imputation and scaling."""

    fill: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    model: Any

    def predict(self, values: np.ndarray) -> np.ndarray:
        filled = np.where(np.isfinite(values), values, self.fill)
        standardized = (filled - self.mean) / self.scale
        return np.asarray(self.model.predict(standardized), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class CompressorFit:
    """Models and baselines fitted on one compressor's outer-train rows."""

    compressor: str
    features: tuple[str, ...]
    baselines: dict[str, Baseline]
    huber: Any
    xgb_models: tuple[Any, ...]
    model_config: dict[str, Any]
    train_prompt_ids: tuple[str, ...]
    validation_prompt_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Evidence:
    """Complete report and optional fitted objects."""

    report: dict[str, Any]
    fits: dict[str, CompressorFit]


def _finite(value: object, name: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _metadata_value(
    metadata: Mapping[bytes, bytes], key: bytes
) -> str | None:
    raw = metadata.get(key)
    if raw is None:
        return None
    try:
        return raw.decode().strip().lower()
    except UnicodeDecodeError as error:
        raise ValueError(
            f"invalid parquet metadata encoding for {key!r}"
        ) from error


def validate_provenance(metadata: Mapping[bytes, bytes]) -> None:
    """Require the exact validated faithful-sweep provenance marker."""
    marker = _metadata_value(metadata, b"herald.validation_status")
    if marker != "validated":
        raise ValueError(
            "switch parquet lacks the validated intervention/scorer marker"
        )
    for key in (
        b"herald.intervention_validation",
        b"herald.parity_status",
        b"herald.dataset_status",
    ):
        value = _metadata_value(metadata, key)
        if value is not None and value not in {
            "validated",
            "passed",
            "pass",
            "ok",
        }:
            raise ValueError(f"switch parquet has conflicting status {key!r}")
    for key, label in (
        (b"herald.sweep_config_sha256", "sweep config"),
        (b"herald.source_manifest_sha256", "source manifest"),
        (TABLE_CONTENT_SHA256_METADATA_KEY, "table content"),
    ):
        digest_value = metadata.get(key)
        if digest_value is None:
            raise ValueError(f"switch parquet lacks {label} provenance")
        try:
            digest = digest_value.decode().strip().lower()
        except UnicodeDecodeError as error:
            raise ValueError(
                f"invalid {label} provenance encoding"
            ) from error
        if len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"switch parquet has invalid {label} provenance")


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    metadata: Mapping[bytes, bytes] | None = None,
    require_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Validate and select IFEval rows; reject label or causal-data errors."""
    if require_provenance:
        if metadata is None:
            raise ValueError("validated parquet provenance is required")
        validate_provenance(metadata)
    selected = (
        tuple(compressors) if compressors is not None else KNOWN_COMPRESSORS
    )
    unknown = set(selected) - set(KNOWN_COMPRESSORS)
    if unknown:
        raise ValueError(f"unknown compressors: {sorted(unknown)}")
    if not selected:
        raise ValueError("at least one compressor is required")
    output: list[dict[str, Any]] = []
    seen: set[tuple[object, ...]] = set()
    reference_scores: dict[tuple[str, str], float] = {}
    for raw in rows:
        if str(raw.get("task")) != "ifeval":
            continue
        compressor = str(raw.get("compressor"))
        if compressor not in selected:
            continue
        missing = REQUIRED_COLUMNS - set(raw)
        if missing:
            raise ValueError(
                f"row is missing required columns: {sorted(missing)}"
            )
        row = dict(raw)
        ratio = _finite(row["ratio"], "ratio")
        position = _finite(row["s"], "s")
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"ratio must be in (0, 1), got {ratio}")
        if position < 0.0 or position != math.floor(position):
            raise ValueError(
                f"s must be a non-negative integer, got {position}"
            )
        if int(position) % POSITION_BUCKET != 0:
            raise ValueError(
                f"s must follow the {POSITION_BUCKET}-token stride"
            )
        semantics = str(row["intervention_semantics"])
        if semantics != INTERVENTION_SEMANTICS[compressor]:
            raise ValueError(
                f"intervention semantics mismatch for {compressor!r}"
            )
        if str(row["feature_timing"]) != "herald.pre_switch_pending_logit_v1":
            raise ValueError("feature timing contract mismatch")
        q_ref = _finite(row["q_ref"], "q_ref")
        q_control = _finite(row["q_control"], "q_control")
        q_hybrid = _finite(row["q_hybrid"], "q_hybrid")
        dq = _finite(row["dq"], "dq")
        for name, quality in (
            ("q_ref", q_ref),
            ("q_control", q_control),
            ("q_hybrid", q_hybrid),
        ):
            if not 0.0 <= quality <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {quality}")
        if not -1.0 <= dq <= 1.0:
            raise ValueError(f"dq must be in [-1, 1], got {dq}")
        damaged = _finite(row["damaged"], "damaged")
        major_damage = _finite(row["major_damage"], "major_damage")
        if damaged not in {0.0, 1.0} or damaged != float(dq > 0.0):
            raise ValueError(
                f"damaged flag mismatch for {row.get('prompt_id')!r}"
            )
        if major_damage not in {0.0, 1.0} or major_damage != float(dq >= 0.5):
            raise ValueError(
                f"major_damage flag mismatch for {row.get('prompt_id')!r}"
            )
        # Parquet decimal round-trips can differ by one binary ulp; only a
        # tiny serialization tolerance is permitted, never a data correction.
        if not math.isclose(
            dq,
            q_control - q_hybrid,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"dq mismatch for {row.get('prompt_id')!r}, "
                f"ratio {ratio}, s {position}"
            )
        key = (
            str(row["model"]),
            "ifeval",
            str(row["prompt_id"]),
            compressor,
            ratio,
            int(position),
        )
        if key in seen:
            raise ValueError(f"duplicate switch row {key!r}")
        seen.add(key)
        reference_key = (str(row["model"]), str(row["prompt_id"]))
        prior_reference = reference_scores.setdefault(reference_key, q_ref)
        if not math.isclose(
            prior_reference,
            q_ref,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"inconsistent q_ref for prompt {reference_key!r}"
            )
        row.update(
            {
                "task": "ifeval",
                "compressor": compressor,
                "ratio": ratio,
                "s": int(position),
                "q_ref": q_ref,
                "q_control": q_control,
                "q_hybrid": q_hybrid,
                "dq": dq,
                "damaged": int(damaged),
                "intervention_semantics": semantics,
                "major_damage": int(major_damage),
            }
        )
        for name, value in row.items():
            if name.startswith("feat__") and value is not None:
                numeric = float(value)
                if not math.isfinite(numeric) and not math.isnan(numeric):
                    raise ValueError(f"feature {name!r} is non-finite")
        output.append(row)
    if not output:
        raise ValueError("no validated IFEval rows for selected compressors")
    present = {str(row["compressor"]) for row in output}
    missing_compressors = set(selected) - present
    if missing_compressors:
        raise ValueError(
            "selected compressors have no IFEval rows: "
            f"{sorted(missing_compressors)}"
        )
    grids = {
        compressor: {
            (str(row["prompt_id"]), float(row["ratio"]), int(row["s"]))
            for row in output
            if str(row["compressor"]) == compressor
        }
        for compressor in selected
    }
    first_compressor = selected[0]
    expected_grid = grids[first_compressor]
    for compressor in selected[1:]:
        if grids[compressor] != expected_grid:
            raise ValueError(
                "compressor switch grids differ: "
                f"{first_compressor!r} vs {compressor!r}"
            )
    return output


def load_validated_parquet(
    path: Path,
    *,
    compressors: Sequence[str] | None = None,
    sweep_config: Path | None = None,
    source_manifest: Path | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Read parquet and enforce schema, provenance, and label parity."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path)  # type: ignore[no-untyped-call]
        rows = table.to_pylist()
        metadata = table.schema.metadata or {}
    except (ImportError, OSError, ValueError) as error:
        raise ValueError(f"could not read switch parquet {path}") from error
    bound_content = metadata.get(TABLE_CONTENT_SHA256_METADATA_KEY)
    if (
        bound_content is None
        or bound_content.decode() != table_content_sha256(table)
    ):
        raise ValueError("parquet table content digest does not match")
    if not isinstance(rows, list):
        raise ValueError("parquet reader did not return row mappings")
    selected = validate_rows(
        rows,
        compressors=compressors,
        metadata=metadata,
        require_provenance=True,
    )
    if (sweep_config is None) != (source_manifest is None):
        raise ValueError(
            "sweep config and source manifest must be supplied together"
        )
    if sweep_config is not None and source_manifest is not None:
        manifest = validate_intervention_manifest(
            source_manifest,
            sweep_config,
        )
        expected_prompts = int(manifest["completion_counts"]["references"])
        actual_prompts = len({str(row["prompt_id"]) for row in selected})
        if actual_prompts != expected_prompts:
            raise ValueError(
                f"parquet prompt coverage {actual_prompts} != "
                f"manifest coverage {expected_prompts}"
            )
        manifest_config = manifest.get("config")
        if not isinstance(manifest_config, dict):
            raise ValueError("intervention manifest config is invalid")
        stride = int(manifest_config["stride"])
        if any(int(row["s"]) % stride for row in selected):
            raise ValueError(
                "parquet switch positions violate manifest stride"
            )
        if metadata[b"herald.sweep_config_sha256"].decode() != sha256_file(
            sweep_config
        ):
            raise ValueError("parquet sweep config identity does not match")
        if metadata[b"herald.source_manifest_sha256"].decode() != sha256_file(
            source_manifest
        ):
            raise ValueError(
                "parquet source manifest identity does not match"
            )
    return selected, _file_sha256(path)


def causal_feature_names(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return the exact causal base-logit feature schema present in rows."""
    names = set().union(*(set(row) for row in rows)) if rows else set()
    return tuple(sorted(names & CAUSAL_FEATURE_COLUMNS))


def trajectory_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Assign each (prompt, ratio) trajectory total mass exactly one."""
    counts: defaultdict[tuple[str, float], int] = defaultdict(int)
    for row in rows:
        counts[(str(row["prompt_id"]), float(row["ratio"]))] += 1
    return np.asarray(
        [
            1.0 / counts[(str(row["prompt_id"]), float(row["ratio"]))]
            for row in rows
        ],
        dtype=np.float64,
    )


def make_outer_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 0,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> PromptSplit:
    """Freeze a prompt-disjoint split shared by every selected compressor."""
    prompt_models = {(str(row["model"]), str(row["task"])) for row in rows}
    if len(prompt_models) != 1:
        raise ValueError("split requires one model and one task")
    model, task = next(iter(prompt_models))
    prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    train, test = split_prompt_ids(
        prompt_ids,
        model=model,
        task=task,
        seed=seed,
        test_group_fraction=test_fraction,
    )
    if not train or not test:
        raise ValueError(
            "outer split must contain both train and test prompts"
        )
    digest = hashlib.sha256(
        json.dumps(
            {"train": train, "test": test},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return PromptSplit(
        tuple(train),
        tuple(test),
        seed,
        test_fraction,
        digest,
    )


def _feature_matrix(
    rows: Sequence[Mapping[str, Any]],
    features: Sequence[str],
) -> np.ndarray:
    out = np.full((len(rows), 2 + len(features)), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        out[i, 0] = float(row["ratio"])
        out[i, 1] = float(row["s"])
        for j, feature in enumerate(features, start=2):
            value = row.get(feature)
            if value is not None:
                number = float(value)
                if math.isfinite(number):
                    out[i, j] = number
    return out


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    order = np.argsort(values, kind="mergesort")
    value = values[order]
    weight = weights[order]
    threshold = quantile * float(weight.sum())
    index = int(np.searchsorted(np.cumsum(weight), threshold, side="left"))
    return float(value[min(index, len(value) - 1)])


def _fit_baseline(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    statistic: str,
    keys: tuple[str, ...],
    fallback: Baseline | None = None,
) -> Baseline:
    y = np.asarray([float(row["dq"]) for row in rows], dtype=np.float64)
    weights = trajectory_weights(rows)
    global_value = (
        float(np.average(y, weights=weights))
        if statistic == "mean"
        else _weighted_quantile(y, weights, 0.5)
    )
    grouped: defaultdict[tuple[object, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[tuple(row[key] for key in keys)].append(index)
    values: dict[tuple[object, ...], float] = {}
    for key, indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        values[key] = (
            float(np.average(y[idx], weights=weights[idx]))
            if statistic == "mean"
            else _weighted_quantile(y[idx], weights[idx], 0.5)
        )
    return Baseline(name, statistic, keys, global_value, values, fallback)


def _baseline_prediction(
    baseline: Baseline,
    row: Mapping[str, Any],
) -> float:
    if baseline.keys:
        key = tuple(
            int(row["s"]) // POSITION_BUCKET
            if name == "position_bucket"
            else row[name]
            for name in baseline.keys
        )
        if key in baseline.values:
            return baseline.values[key]
    if baseline.fallback is not None:
        return _baseline_prediction(baseline.fallback, row)
    return baseline.global_value


def fit_compressor(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    compressor: str,
    features: Sequence[str],
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
) -> CompressorFit:
    """Fit fixed baselines, Huber, and a three-seed XGBoost ensemble."""
    if not train_rows or not validation_rows:
        raise ValueError(
            "compressor fit requires non-empty train and validation rows"
        )
    baselines: dict[str, Baseline] = {}
    for statistic in ("mean", "median"):
        baselines[f"global_{statistic}"] = _fit_baseline(
            train_rows,
            f"global_{statistic}",
            statistic,
            (),
        )
        ratio_baseline = _fit_baseline(
            train_rows,
            f"ratio_{statistic}",
            statistic,
            ("ratio",),
        )
        baselines[ratio_baseline.name] = ratio_baseline
        positioned = [
            dict(row, position_bucket=int(row["s"]) // POSITION_BUCKET)
            for row in train_rows
        ]
        baselines[f"ratio_position_{statistic}"] = _fit_baseline(
            positioned,
            f"ratio_position_{statistic}",
            statistic,
            ("ratio", "position_bucket"),
            fallback=ratio_baseline,
        )

    try:
        from sklearn.linear_model import (  # type: ignore[import-untyped]
            HuberRegressor,
        )
    except ImportError as error:
        raise RuntimeError(
            "scikit-learn is required for the Huber baseline"
        ) from error
    x_train = _feature_matrix(train_rows, features)
    y_train = np.asarray(
        [float(row["dq"]) for row in train_rows], dtype=np.float64
    )
    train_weights = trajectory_weights(train_rows)
    fill = np.zeros(x_train.shape[1], dtype=np.float64)
    for column in range(x_train.shape[1]):
        finite = np.isfinite(x_train[:, column])
        if np.any(finite):
            fill[column] = _weighted_quantile(
                x_train[finite, column],
                train_weights[finite],
                0.5,
            )
    filled_train = np.where(np.isfinite(x_train), x_train, fill)
    weighted_mean = np.average(
        filled_train,
        axis=0,
        weights=train_weights,
    )
    weighted_variance = np.average(
        (filled_train - weighted_mean) ** 2,
        axis=0,
        weights=train_weights,
    )
    weighted_scale = np.sqrt(weighted_variance)
    weighted_scale[weighted_scale == 0.0] = 1.0
    huber_model = HuberRegressor(max_iter=5000)
    huber_model.fit(
        (filled_train - weighted_mean) / weighted_scale,
        y_train,
        sample_weight=train_weights,
    )
    huber = WeightedHuber(
        fill,
        weighted_mean,
        weighted_scale,
        huber_model,
    )

    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "xgboost is required for the fixed magnitude model"
        ) from error
    validation_x = _feature_matrix(validation_rows, features)
    validation_y = np.asarray(
        [float(row["dq"]) for row in validation_rows], dtype=np.float64
    )
    config: dict[str, Any] = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "num_boost_round": 300,
        "early_stopping_rounds": 30,
        "seeds": list(model_seeds),
    }
    train_matrix = xgb.DMatrix(
        x_train,
        label=y_train,
        weight=train_weights,
    )
    val_matrix = xgb.DMatrix(
        validation_x,
        label=validation_y,
        weight=trajectory_weights(validation_rows),
    )
    models: list[Any] = []
    for seed in model_seeds:
        params = {
            key: value
            for key, value in config.items()
            if key
            not in {"num_boost_round", "early_stopping_rounds", "seeds"}
        }
        params["seed"] = int(seed)
        params["nthread"] = -1
        models.append(
            xgb.train(
                params,
                train_matrix,
                num_boost_round=int(config["num_boost_round"]),
                evals=[(val_matrix, "validation")],
                early_stopping_rounds=int(config["early_stopping_rounds"]),
                verbose_eval=False,
            )
        )
    return CompressorFit(
        compressor,
        tuple(features),
        baselines,
        huber,
        tuple(models),
        config,
        tuple(sorted({str(row["prompt_id"]) for row in train_rows})),
        tuple(sorted({str(row["prompt_id"]) for row in validation_rows})),
    )


def _predict_model(
    fit: CompressorFit, rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray]:
    x = _feature_matrix(rows, fit.features)
    huber = np.asarray(fit.huber.predict(x), dtype=np.float64)
    try:
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "xgboost is required for model prediction"
        ) from error
    matrix = xgb.DMatrix(x)
    values = []
    for model in fit.xgb_models:
        best = getattr(model, "best_iteration", None)
        predicted = (
            model.predict(matrix, iteration_range=(0, int(best) + 1))
            if best is not None
            else model.predict(matrix)
        )
        values.append(np.asarray(predicted, dtype=np.float64))
    return np.asarray(huber), np.mean(np.stack(values), axis=0)


def _ratio_metrics(
    y: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ratios = sorted({float(row["ratio"]) for row in rows})
    by_ratio: dict[str, dict[str, float | int | None]] = {}
    for ratio in ratios:
        mask = np.asarray(
            [float(row["ratio"]) == ratio for row in rows], dtype=bool
        )
        by_ratio[str(ratio)] = _metrics(y[mask], pred[mask], weights[mask])
    overall: dict[str, float | int | None] = {
        key: float(
            np.mean(
                [_metric_value(value, key) for value in by_ratio.values()]
            )
        )
        for key in ("mse", "rmse", "mae")
    }
    pooled = _metrics(y, pred, weights)
    for key in (
        "n",
        "mean_target",
        "mean_prediction",
        "prediction_std",
        "target_std",
        "rank_correlation",
    ):
        overall[key] = pooled[key]
    return {"overall": overall, "per_ratio": by_ratio}


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while (
            end < len(values) and values[order[end]] == values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _weighted_rank_correlation(
    y: np.ndarray,
    pred: np.ndarray,
    weights: np.ndarray,
) -> float | None:
    y_rank = _average_ranks(y)
    pred_rank = _average_ranks(pred)
    normalized = weights / weights.sum()
    y_centered = y_rank - np.sum(normalized * y_rank)
    pred_centered = pred_rank - np.sum(normalized * pred_rank)
    denominator = math.sqrt(
        float(np.sum(normalized * y_centered**2))
        * float(np.sum(normalized * pred_centered**2))
    )
    if denominator == 0.0:
        return None
    return float(
        np.sum(normalized * y_centered * pred_centered) / denominator
    )


def _metrics(
    y: np.ndarray, pred: np.ndarray, weights: np.ndarray
) -> dict[str, float | int | None]:
    if not len(y):
        return {
            "n": 0,
            "mse": None,
            "rmse": None,
            "mae": None,
            "mean_target": None,
            "mean_prediction": None,
            "prediction_std": None,
            "target_std": None,
            "rank_correlation": None,
        }
    normalized = weights / weights.sum()
    error = pred - y
    mean_target = float(np.sum(normalized * y))
    mean_prediction = float(np.sum(normalized * pred))
    return {
        "n": int(len(y)),
        "mse": float(np.sum(normalized * error**2)),
        "rmse": float(np.sqrt(np.sum(normalized * error**2))),
        "mae": float(np.sum(normalized * np.abs(error))),
        "mean_target": mean_target,
        "mean_prediction": mean_prediction,
        "prediction_std": float(
            np.sqrt(np.sum(normalized * (pred - mean_prediction) ** 2))
        ),
        "target_std": float(
            np.sqrt(np.sum(normalized * (y - mean_target) ** 2))
        ),
        "rank_correlation": _weighted_rank_correlation(y, pred, weights),
    }


def _metric_value(
    metrics: Mapping[str, float | int | None], key: str
) -> float:
    """Return one required metric from a non-empty evaluation."""
    value = metrics.get(key)
    if value is None:
        raise ValueError(f"required metric {key!r} is missing")
    return float(value)


def _calibration(
    y: np.ndarray, pred: np.ndarray, weights: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    if not len(y):
        return []
    order = np.argsort(pred, kind="mergesort")
    chunks = np.array_split(order, min(bins, len(y)))
    result = []
    for index, chunk in enumerate(chunks):
        w = weights[chunk]
        w = w / w.sum()
        result.append(
            {
                "bin": index,
                "n": int(len(chunk)),
                "mean_pred": float(np.sum(w * pred[chunk])),
                "mean_dq": float(np.sum(w * y[chunk])),
            }
        )
    return result


def bootstrap_prompt_indices(
    rows: Sequence[Mapping[str, Any]], *, seed: int, resamples: int
) -> list[np.ndarray]:
    """Resample prompt clusters while retaining duplicate cluster draws."""
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    prompts = sorted({str(row["prompt_id"]) for row in rows})
    indices = {
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
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [
                indices[prompts[int(i)]]
                for i in rng.integers(0, len(prompts), size=len(prompts))
            ]
        )
        for _ in range(resamples)
    ]


def _bootstrap(
    rows: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    pred: np.ndarray,
    mean_pred: np.ndarray,
    median_pred: np.ndarray,
    *,
    seed: int,
    resamples: int,
) -> dict[str, Any]:
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    weights = trajectory_weights(rows)
    point_mse = _ratio_metrics(y, pred, weights, rows)["overall"]["mse"]
    point_base_mse = _ratio_metrics(y, mean_pred, weights, rows)["overall"][
        "mse"
    ]
    point_mae = _ratio_metrics(y, pred, weights, rows)["overall"]["mae"]
    point_base_mae = _ratio_metrics(y, median_pred, weights, rows)["overall"][
        "mae"
    ]
    mse_skills: list[float] = []
    mae_skills: list[float] = []
    for chosen in bootstrap_prompt_indices(
        rows, seed=seed, resamples=resamples
    ):
        sampled_rows = [rows[int(i)] for i in chosen]
        # Keep each duplicated sampled cluster's original trajectory mass.
        sampled_weights = weights[chosen]
        model = _ratio_metrics(
            y[chosen], pred[chosen], sampled_weights, sampled_rows
        )["overall"]["mse"]
        base = _ratio_metrics(
            y[chosen], mean_pred[chosen], sampled_weights, sampled_rows
        )["overall"]["mse"]
        model_mae = _ratio_metrics(
            y[chosen], pred[chosen], sampled_weights, sampled_rows
        )["overall"]["mae"]
        base_mae = _ratio_metrics(
            y[chosen], median_pred[chosen], sampled_weights, sampled_rows
        )["overall"]["mae"]
        mse_skills.append(float((base - model) / base) if base else 0.0)
        mae_skills.append(
            float((base_mae - model_mae) / base_mae) if base_mae else 0.0
        )
    return {
        "resamples": resamples,
        "seed": seed,
        "mse_skill": {
            "estimate": float((point_base_mse - point_mse) / point_base_mse)
            if point_base_mse
            else 0.0,
            "lower": float(np.quantile(mse_skills, 0.025)),
            "upper": float(np.quantile(mse_skills, 0.975)),
        },
        "mae_skill": {
            "estimate": float((point_base_mae - point_mae) / point_base_mae)
            if point_base_mae
            else 0.0,
            "lower": float(np.quantile(mae_skills, 0.025)),
            "upper": float(np.quantile(mae_skills, 0.975)),
        },
    }


def _weighted_rate(mask: np.ndarray, weights: np.ndarray) -> float:
    """Return a trajectory-weighted prevalence."""
    if not np.any(mask):
        return 0.0
    return float(weights[mask].sum() / weights.sum())


def _select_baselines(
    fit: CompressorFit, validation_rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str]:
    """Select grouped baselines on the train-only validation fold."""
    y = np.asarray(
        [float(row["dq"]) for row in validation_rows], dtype=np.float64
    )
    weights = trajectory_weights(validation_rows)
    mean_names = ("global_mean", "ratio_mean", "ratio_position_mean")
    median_names = ("global_median", "ratio_median", "ratio_position_median")
    scores: dict[str, tuple[float, float]] = {}
    for name in (*mean_names, *median_names):
        prediction = np.asarray(
            [
                _baseline_prediction(fit.baselines[name], row)
                for row in validation_rows
            ],
            dtype=np.float64,
        )
        metrics = _ratio_metrics(
            y,
            prediction,
            weights,
            validation_rows,
        )["overall"]
        scores[name] = (
            _metric_value(metrics, "mse"),
            _metric_value(metrics, "mae"),
        )
    return (
        min(mean_names, key=lambda name: scores[name][0]),
        min(median_names, key=lambda name: scores[name][1]),
    )


def _evaluate(
    fit: CompressorFit,
    rows: Sequence[Mapping[str, Any]],
    *,
    strongest_mean_name: str,
    strongest_median_name: str,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    y = np.asarray([float(row["dq"]) for row in rows], dtype=np.float64)
    weights = trajectory_weights(rows)
    huber_pred, xgb_pred = _predict_model(fit, rows)
    baseline_names = (
        "global_mean",
        "ratio_mean",
        "ratio_position_mean",
        "global_median",
        "ratio_median",
        "ratio_position_median",
    )
    baseline_predictions = {
        name: np.asarray(
            [_baseline_prediction(fit.baselines[name], row) for row in rows],
            dtype=np.float64,
        )
        for name in baseline_names
    }
    baseline_metrics: dict[str, Any] = {}
    for name in baseline_names:
        metrics = _ratio_metrics(y, baseline_predictions[name], weights, rows)
        metrics["positive_rows"] = _metrics(
            y[y > 0], baseline_predictions[name][y > 0], weights[y > 0]
        )
        metrics["nonzero_rows"] = _metrics(
            y[y != 0], baseline_predictions[name][y != 0], weights[y != 0]
        )
        metrics["major_damage_rows"] = _metrics(
            y[y >= 0.5],
            baseline_predictions[name][y >= 0.5],
            weights[y >= 0.5],
        )
        baseline_metrics[name] = metrics
    strongest_mean_pred = baseline_predictions[strongest_mean_name]
    strongest_median_pred = baseline_predictions[strongest_median_name]
    base_mse = float(baseline_metrics[strongest_mean_name]["overall"]["mse"])
    base_mae = float(
        baseline_metrics[strongest_median_name]["overall"]["mae"]
    )
    base_positive = baseline_metrics[strongest_mean_name]["positive_rows"]
    base_major_damage = baseline_metrics[strongest_mean_name][
        "major_damage_rows"
    ]
    model_metrics: dict[str, Any] = {}
    for name, prediction in (("huber", huber_pred), ("xgboost", xgb_pred)):
        metrics = _ratio_metrics(y, prediction, weights, rows)
        metrics["mse_skill_vs_strongest_mean"] = (
            (base_mse - float(metrics["overall"]["mse"])) / base_mse
            if base_mse
            else 0.0
        )
        metrics["mae_skill_vs_strongest_median"] = (
            (base_mae - float(metrics["overall"]["mae"])) / base_mae
            if base_mae
            else 0.0
        )
        metrics["positive_rows"] = _metrics(
            y[y > 0], prediction[y > 0], weights[y > 0]
        )
        metrics["nonzero_rows"] = _metrics(
            y[y != 0], prediction[y != 0], weights[y != 0]
        )
        metrics["major_damage_rows"] = _metrics(
            y[y >= 0.5],
            prediction[y >= 0.5],
            weights[y >= 0.5],
        )
        metrics["calibration_bins"] = _calibration(y, prediction, weights)
        metrics["positive_mse_vs_strongest_mean"] = (
            (
                float(base_positive["mse"])
                - float(metrics["positive_rows"]["mse"])
            )
            / float(base_positive["mse"])
            if base_positive["mse"]
            else 0.0
        )
        metrics["major_damage_mse_vs_strongest_mean"] = (
            (
                float(base_major_damage["mse"])
                - float(metrics["major_damage_rows"]["mse"])
            )
            / float(base_major_damage["mse"])
            if base_major_damage["mse"]
            else 0.0
        )
        model_metrics[name] = metrics
    prevalence = {
        "positive": _weighted_rate(y > 0, weights),
        "zero": _weighted_rate(y == 0, weights),
        "negative": _weighted_rate(y < 0, weights),
        "nonzero": _weighted_rate(y != 0, weights),
        "major_damage": _weighted_rate(y >= 0.5, weights),
    }
    xgb_bootstrap = _bootstrap(
        rows,
        y,
        xgb_pred,
        strongest_mean_pred,
        strongest_median_pred,
        seed=seed,
        resamples=bootstrap_resamples,
    )
    ratio_table: dict[str, Any] = {}
    for ratio in sorted({float(row["ratio"]) for row in rows}):
        mask = np.asarray(
            [float(row["ratio"]) == ratio for row in rows], dtype=bool
        )
        ratio_table[str(ratio)] = {
            "n": int(mask.sum()),
            "xgboost": _metrics(y[mask], xgb_pred[mask], weights[mask]),
            "strongest_mean": _metrics(
                y[mask], strongest_mean_pred[mask], weights[mask]
            ),
        }
    prompt_metrics = []
    for prompt in sorted({str(row["prompt_id"]) for row in rows}):
        mask = np.asarray(
            [str(row["prompt_id"]) == prompt for row in rows], dtype=bool
        )
        prompt_metrics.append(
            _metrics(y[mask], xgb_pred[mask], weights[mask])
        )
    macro = {
        key: float(
            np.mean([_metric_value(item, key) for item in prompt_metrics])
        )
        for key in ("mse", "rmse", "mae")
    }
    return {
        "baselines": baseline_metrics,
        "strongest_mean": strongest_mean_name,
        "strongest_median": strongest_median_name,
        "models": model_metrics,
        "prevalence": prevalence,
        "bootstrap_intervals": xgb_bootstrap,
        "per_ratio": ratio_table,
        "prompt_macro": macro,
    }


def fit_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    model_seeds: Sequence[int] = DEFAULT_MODEL_SEEDS,
) -> Evidence:
    """Fit independent compressors under the frozen magnitude protocol."""
    checked = validate_rows(rows, compressors=compressors)
    split = make_outer_split(checked, seed=seed, test_fraction=test_fraction)
    features = causal_feature_names(checked)
    if not features:
        raise ValueError(
            "no causal feat__* features remain after forbidden-field "
            "filtering"
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "data_hash": rows_fingerprint(checked),
        "split": {
            "seed": seed,
            "test_fraction": test_fraction,
            "train_prompt_ids": list(split.train_prompt_ids),
            "test_prompt_ids": list(split.test_prompt_ids),
            "hash": split.hash,
        },
        "features": ["ratio", "s", *features],
        "model_config": {
            "xgboost": {
                "objective": "reg:squarederror",
                "max_depth": 4,
                "eta": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "num_boost_round": 300,
                "early_stopping_rounds": 30,
                "seeds": list(model_seeds),
            },
            "position_bucket": POSITION_BUCKET,
        },
        "compressors": {},
    }
    fits: dict[str, CompressorFit] = {}
    all_prompts = set(split.train_prompt_ids)
    _, validation_ids = split_prompt_ids(
        sorted(all_prompts),
        model=str(checked[0]["model"]),
        task="ifeval",
        seed=seed + 1,
        test_group_fraction=0.2,
    )
    validation_set = set(validation_ids)
    if not validation_set or validation_set == all_prompts:
        raise ValueError("training-only validation fold is empty")
    for compressor in sorted({str(row["compressor"]) for row in checked}):
        comp_train = [
            row
            for row in checked
            if str(row["compressor"]) == compressor
            and str(row["prompt_id"]) in set(split.train_prompt_ids)
        ]
        validation = [
            row
            for row in comp_train
            if str(row["prompt_id"]) in validation_set
        ]
        fit_train = [
            row
            for row in comp_train
            if str(row["prompt_id"]) not in validation_set
        ]
        test = [
            row
            for row in checked
            if str(row["compressor"]) == compressor
            and str(row["prompt_id"]) in set(split.test_prompt_ids)
        ]
        fit = fit_compressor(
            fit_train,
            validation,
            compressor=compressor,
            features=features,
            model_seeds=model_seeds,
        )
        strongest_mean, strongest_median = _select_baselines(fit, validation)
        fits[compressor] = fit
        report["compressors"][compressor] = {
            "n_train": len(fit_train),
            "n_validation": len(validation),
            "n_test": len(test),
            "train_prompt_ids": list(fit.train_prompt_ids),
            "validation_prompt_ids": list(fit.validation_prompt_ids),
            "test_prompt_ids": list(split.test_prompt_ids),
            "baseline_selection": {
                "validation_strongest_mean": strongest_mean,
                "validation_strongest_median": strongest_median,
            },
            "evaluation": _evaluate(
                fit,
                test,
                strongest_mean_name=strongest_mean,
                strongest_median_name=strongest_median,
                seed=bootstrap_seed,
                bootstrap_resamples=bootstrap_resamples,
            ),
        }
    report["decision"] = _decision(report["compressors"])
    return Evidence(report, fits)


def _decision(compressors: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "go", "compressors": {}}
    for name, content in compressors.items():
        evaluation = content["evaluation"]
        interval = evaluation["bootstrap_intervals"]["mse_skill"]
        model = evaluation["models"]["xgboost"]
        per_ratio = evaluation["per_ratio"]
        reasons = []
        if float(interval["lower"]) <= 0.0:
            reasons.append(
                "MSE-skill bootstrap interval is not strictly above zero"
            )
        baseline_positive = float(
            evaluation["baselines"][evaluation["strongest_mean"]][
                "positive_rows"
            ]["mse"]
            or 0.0
        )
        if float(model["positive_rows"]["mse"] or 0.0) > baseline_positive:
            reasons.append("positive-row MSE worsens")
        baseline_major = float(
            evaluation["baselines"][evaluation["strongest_mean"]][
                "major_damage_rows"
            ]["mse"]
            or 0.0
        )
        if (
            float(evaluation["prevalence"]["major_damage"]) > 0.0
            and float(model["major_damage_rows"]["mse"] or 0.0)
            > baseline_major
        ):
            reasons.append("major-damage-row MSE worsens")
        target_std = float(model["overall"]["target_std"] or 0.0)
        prediction_std = float(model["overall"]["prediction_std"] or 0.0)
        if target_std > 1e-6 and prediction_std <= max(
            1e-4,
            0.05 * target_std,
        ):
            reasons.append("weighted calibration materially collapses")
        if any(
            float(value["xgboost"]["mse"] or 0.0)
            >= float(value["strongest_mean"]["mse"] or 0.0)
            for value in per_ratio.values()
        ):
            reasons.append("at least one ratio has no MSE improvement")
        result["compressors"][name] = {
            "status": "go" if not reasons else "stop",
            "reasons": reasons,
        }
        if reasons:
            result["status"] = "stop"
    return result


def write_model_bundle(evidence: Evidence, path: Path) -> Path:
    """Persist native XGBoost models and trusted local Python fit state."""
    try:
        import joblib  # type: ignore[import-untyped]
    except ImportError as error:
        raise RuntimeError("joblib is required for model export") from error
    path.mkdir(parents=True, exist_ok=False)
    report_path = path / "report.json"
    write_report(evidence.report, report_path)
    for compressor, fit in sorted(evidence.fits.items()):
        compressor_dir = path / compressor
        compressor_dir.mkdir()
        state = {
            "compressor": fit.compressor,
            "features": fit.features,
            "baselines": fit.baselines,
            "huber": fit.huber,
            "model_config": fit.model_config,
            "train_prompt_ids": fit.train_prompt_ids,
            "validation_prompt_ids": fit.validation_prompt_ids,
        }
        joblib.dump(state, compressor_dir / "fit.joblib")
        for index, model in enumerate(fit.xgb_models):
            model.save_model(str(compressor_dir / f"xgboost-{index}.ubj"))
    artifacts = [
        {
            "path": str(artifact.relative_to(path)),
            "sha256": _file_sha256(artifact),
            "size": artifact.stat().st_size,
        }
        for artifact in sorted(path.rglob("*"))
        if artifact.is_file()
    ]
    manifest = {
        "schema_version": "herald.magnitude_bundle.v1",
        "compressors": sorted(evidence.fits),
        "artifacts": artifacts,
    }
    manifest_path = path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest_path


def load_model_bundle(path: Path, compressor: str) -> CompressorFit:
    """Load a model bundle produced locally by :func:`write_model_bundle`."""
    try:
        import joblib
        import xgboost as xgb
    except ImportError as error:
        raise RuntimeError(
            "joblib and xgboost are required for model loading"
        ) from error
    state_path = path / compressor / "fit.joblib"
    state = joblib.load(state_path)
    if not isinstance(state, dict):
        raise ValueError(f"invalid fit state: {state_path}")
    model_paths = sorted((path / compressor).glob("xgboost-*.ubj"))
    expected_models = len(state["model_config"]["seeds"])
    if len(model_paths) != expected_models:
        raise ValueError(
            f"expected {expected_models} XGBoost models, "
            f"found {len(model_paths)}"
        )
    models = []
    for model_path in model_paths:
        model = xgb.Booster()
        model.load_model(str(model_path))
        models.append(model)
    return CompressorFit(
        compressor=str(state["compressor"]),
        features=tuple(state["features"]),
        baselines=dict(state["baselines"]),
        huber=state["huber"],
        xgb_models=tuple(models),
        model_config=dict(state["model_config"]),
        train_prompt_ids=tuple(state["train_prompt_ids"]),
        validation_prompt_ids=tuple(state["validation_prompt_ids"]),
    )


def write_report(report: Mapping[str, Any], path: Path) -> None:
    """Write stable, human-readable JSON evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
