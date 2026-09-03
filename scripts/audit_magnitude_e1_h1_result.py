"""Independently reproduce and adversarially audit the frozen E1 H1 result."""

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn.feature_extraction.text as text_features  # type: ignore[import-untyped]
import sklearn.linear_model as linear_model  # type: ignore[import-untyped]
import sklearn.preprocessing as preprocessing  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
RATIOS = (0.25, 0.5, 0.75, 0.875)
PROMPTS = 154
LOW_PROMPTS = 77
FOLDS = 5
RESAMPLES = 10000
SEED = 314159
CONFIDENCE = 0.95
MINIMUM_OVERALL_SKILL = 0.01
REPRESENTATION_REPRODUCTION_TOLERANCE = 1e-12
PREDICTION_REPRODUCTION_TOLERANCE = 1e-9
WORD_PATTERN_TEXT = r"(?u)\b\w\w+\b"
DECIMAL_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
LIST_MARKER_PATTERN = re.compile(r"(?m)^\s*(?:[-*+]|(?:\d+|[A-Za-z])[.)])\s+")
WORD_PATTERN = re.compile(WORD_PATTERN_TEXT)
SCHEMA_VERSION = "herald.magnitude_e1_h1_independent_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--prompt-table", type=Path, required=True)
    parser.add_argument("--representation-archive", type=Path, required=True)
    parser.add_argument("--representation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--scoring-lock", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--feature-script", type=Path, required=True)
    parser.add_argument("--compile-script", type=Path, required=True)
    parser.add_argument("--score-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def hash_prompt_ids(values: list[str] | tuple[str, ...]) -> str:
    prompts = sorted(str(value) for value in values)
    return hashlib.sha256(("\n".join(prompts) + "\n").encode()).hexdigest()


def word_vectorizer() -> Any:
    return text_features.TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        token_pattern=WORD_PATTERN_TEXT,
        lowercase=True,
        min_df=1,
        max_df=1.0,
        max_features=None,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        stop_words=None,
        strip_accents=None,
        dtype=np.float64,
    )


def character_vectorizer() -> Any:
    return text_features.TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=False,
        min_df=2,
        max_df=1.0,
        max_features=None,
        sublinear_tf=True,
        use_idf=True,
        smooth_idf=True,
        norm="l2",
        strip_accents=None,
        dtype=np.float64,
    )


def safe_fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def surface_vector(text: str) -> NDArray[np.float64]:
    words = WORD_PATTERN.findall(text)
    word_lengths = np.asarray([len(word) for word in words], dtype=np.float64)
    lines = text.split("\n")
    letters = [character for character in text if character.isalpha()]
    code_points = len(text)
    values = (
        math.log1p(code_points),
        math.log1p(len(words)),
        math.log1p(len({word.casefold() for word in words})),
        float(np.mean(word_lengths)) if len(word_lengths) else 0.0,
        float(np.std(word_lengths, ddof=0)) if len(word_lengths) else 0.0,
        math.log1p(int(np.max(word_lengths)) if len(word_lengths) else 0),
        math.log1p(len(lines)),
        math.log1p(max((len(line) for line in lines), default=0)),
        safe_fraction(
            sum(character.isupper() for character in letters), len(letters)
        ),
        safe_fraction(
            sum(character.isdigit() for character in text), code_points
        ),
        safe_fraction(
            sum(character.isspace() for character in text), code_points
        ),
        safe_fraction(
            sum(
                unicodedata.category(character).startswith("P")
                for character in text
            ),
            code_points,
        ),
        math.log1p(len(DECIMAL_PATTERN.findall(text))),
        math.log1p(sum(character in "'\"`" for character in text)),
        math.log1p(len(LIST_MARKER_PATTERN.findall(text))),
        math.log1p(sum(character in ".?!" for character in text)),
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("independent surface vector is invalid")
    return result


def surface_matrix(texts: list[str]) -> NDArray[np.float64]:
    return np.vstack([surface_vector(text) for text in texts])


def independent_alpha(
    matrix: sparse.spmatrix | NDArray[np.float64],
) -> tuple[float, int, float]:
    product = matrix @ matrix.T
    gram = (
        cast("Any", product).toarray()
        if sparse.issparse(product)
        else np.asarray(product, dtype=np.float64)
    )
    row_mean = np.mean(gram, axis=1, keepdims=True)
    centered = gram - row_mean - row_mean.T + float(np.mean(gram))
    eigenvalues = np.linalg.eigvalsh((centered + centered.T) * 0.5)
    maximum = float(np.max(eigenvalues))
    positive = eigenvalues[eigenvalues > maximum * 1e-12]
    if len(positive) <= 10:
        raise ValueError(
            "independent representation rank is not above EDF 10"
        )

    def edf(alpha: float) -> float:
        return float(np.sum(positive / (positive + alpha)))

    lower = 0.0
    upper = maximum
    while edf(upper) > 10.0:
        upper *= 2.0
    for _ in range(200):
        midpoint = (lower + upper) * 0.5
        if edf(midpoint) > 10.0:
            lower = midpoint
        else:
            upper = midpoint
    alpha = (lower + upper) * 0.5
    return alpha, len(positive), edf(alpha)


def archive_csr(
    archive: Mapping[str, NDArray[Any]], prefix: str
) -> sparse.csr_matrix:
    shape_values = np.asarray(archive[f"{prefix}_shape"], dtype=np.int64)
    return sparse.csr_matrix(
        (
            np.asarray(archive[f"{prefix}_data"], dtype=np.float64),
            np.asarray(archive[f"{prefix}_indices"], dtype=np.int32),
            np.asarray(archive[f"{prefix}_indptr"], dtype=np.int32),
        ),
        shape=(int(shape_values[0]), int(shape_values[1])),
    )


def max_sparse_difference(
    left: sparse.csr_matrix, right: sparse.csr_matrix
) -> float:
    if left.shape != right.shape:
        return math.inf
    difference = left - right
    return float(np.max(np.abs(difference.data))) if difference.nnz else 0.0


def archive_scalar(archive: Mapping[str, NDArray[Any]], key: str) -> float:
    values = np.asarray(archive[key]).reshape(-1)
    if len(values) != 1:
        raise ValueError(f"archive scalar {key} has unexpected shape")
    return float(values[0])


def load_prompt_table(path: Path) -> pd.DataFrame:
    columns = ["prompt_id", "prompt_text", "fold"]
    table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    if table.column_names != columns:
        raise ValueError("independent audit prompt columns differ")
    frame = table.to_pandas()
    if len(frame) != PROMPTS or frame["prompt_id"].duplicated().any():
        raise ValueError("independent prompt table is incomplete")
    return frame


def load_source(path: Path) -> pd.DataFrame:
    columns = ["prompt_id", "compressor", "ratio", "s", "dq", "fold"]
    table = pq.read_table(path, columns=columns)  # type: ignore[no-untyped-call]
    if table.column_names != columns:
        raise ValueError("independent audit source columns differ")
    frame = table.to_pandas()
    numeric = frame[["ratio", "s", "dq", "fold"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("independent audit source contains nonfinite values")
    return frame


def build_targets(
    source: pd.DataFrame, prompts: pd.DataFrame
) -> pd.DataFrame:
    ratio = (
        source.groupby(
            ["prompt_id", "compressor", "ratio"], observed=True, sort=True
        )["dq"]
        .mean()
        .rename("target_ratio")
        .reset_index()
    )
    target = (
        ratio.groupby(["prompt_id", "compressor"], sort=True)["target_ratio"]
        .mean()
        .rename("target")
        .reset_index()
        .merge(
            prompts[["prompt_id", "fold"]],
            on="prompt_id",
            validate="many_to_one",
        )
    )
    if len(target) != PROMPTS * len(COMPRESSORS):
        raise ValueError("independent target table is incomplete")
    return target


def rebuild_and_refit(
    prompts: pd.DataFrame,
    targets: pd.DataFrame,
    archive: Mapping[str, NDArray[Any]],
    frozen_oof: pd.DataFrame,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    representation_checks: list[dict[str, Any]] = []
    prediction_max_difference = {
        compressor: {"text": 0.0, "surface": 0.0, "mean": 0.0}
        for compressor in COMPRESSORS
    }
    similarities: list[tuple[str, float]] = []
    for fold in range(FOLDS):
        prefix = f"fold_{fold}"
        training = prompts[prompts["fold"] != fold].sort_values("prompt_id")
        validation = prompts[prompts["fold"] == fold].sort_values("prompt_id")
        training_ids = tuple(str(value) for value in training["prompt_id"])
        validation_ids = tuple(
            str(value) for value in validation["prompt_id"]
        )
        if training_ids != tuple(
            str(value) for value in archive[f"{prefix}_training_prompt_ids"]
        ) or validation_ids != tuple(
            str(value) for value in archive[f"{prefix}_validation_prompt_ids"]
        ):
            raise ValueError(f"fold {fold} prompt order differs from archive")
        training_text = [str(value) for value in training["prompt_text"]]
        validation_text = [str(value) for value in validation["prompt_text"]]
        word = word_vectorizer()
        character = character_vectorizer()
        word_training = word.fit_transform(training_text).tocsr()
        word_validation = word.transform(validation_text).tocsr()
        char_training = character.fit_transform(training_text).tocsr()
        char_validation = character.transform(validation_text).tocsr()
        scale = 1.0 / math.sqrt(2.0)
        text_training = sparse.hstack(
            [word_training * scale, char_training * scale], format="csr"
        )
        text_validation = sparse.hstack(
            [word_validation * scale, char_validation * scale], format="csr"
        )
        raw_surface_training = surface_matrix(training_text)
        raw_surface_validation = surface_matrix(validation_text)
        scaler = preprocessing.StandardScaler()
        surface_training = np.asarray(
            scaler.fit_transform(raw_surface_training), dtype=np.float64
        )
        surface_validation = np.asarray(
            scaler.transform(raw_surface_validation), dtype=np.float64
        )
        nearest = np.max(
            (
                (word_validation @ word_training.T).toarray()
                + (char_validation @ char_training.T).toarray()
            )
            * 0.5,
            axis=1,
        )
        text_alpha, text_rank, text_edf = independent_alpha(text_training)
        surface_alpha, surface_rank, surface_edf = independent_alpha(
            surface_training
        )
        representation_checks.append(
            {
                "fold": fold,
                "text_training_max_difference": max_sparse_difference(
                    text_training,
                    archive_csr(archive, f"{prefix}_text_training"),
                ),
                "text_validation_max_difference": max_sparse_difference(
                    text_validation,
                    archive_csr(archive, f"{prefix}_text_validation"),
                ),
                "surface_training_max_difference": float(
                    np.max(
                        np.abs(
                            surface_training
                            - np.asarray(
                                archive[f"{prefix}_surface_training"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "surface_validation_max_difference": float(
                    np.max(
                        np.abs(
                            surface_validation
                            - np.asarray(
                                archive[f"{prefix}_surface_validation"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "nearest_similarity_max_difference": float(
                    np.max(
                        np.abs(
                            nearest
                            - np.asarray(
                                archive[
                                    f"{prefix}_nearest_training_similarity"
                                ],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "word_features_exact": np.array_equal(
                    word.get_feature_names_out(),
                    archive[f"{prefix}_word_features"],
                ),
                "char_features_exact": np.array_equal(
                    character.get_feature_names_out(),
                    archive[f"{prefix}_char_features"],
                ),
                "word_idf_max_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(word.idf_, dtype=np.float64)
                            - np.asarray(
                                archive[f"{prefix}_word_idf"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "char_idf_max_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(character.idf_, dtype=np.float64)
                            - np.asarray(
                                archive[f"{prefix}_char_idf"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "surface_mean_max_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(scaler.mean_, dtype=np.float64)
                            - np.asarray(
                                archive[f"{prefix}_surface_mean"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "surface_scale_max_difference": float(
                    np.max(
                        np.abs(
                            np.asarray(scaler.scale_, dtype=np.float64)
                            - np.asarray(
                                archive[f"{prefix}_surface_scale"],
                                dtype=np.float64,
                            )
                        )
                    )
                ),
                "text_alpha_difference": abs(
                    text_alpha
                    - archive_scalar(archive, f"{prefix}_text_alpha")
                ),
                "surface_alpha_difference": abs(
                    surface_alpha
                    - archive_scalar(archive, f"{prefix}_surface_alpha")
                ),
                "text_rank": text_rank,
                "surface_rank": surface_rank,
                "text_edf": text_edf,
                "surface_edf": surface_edf,
            }
        )
        similarities.extend(zip(validation_ids, nearest, strict=True))
        for compressor in COMPRESSORS:
            group = targets[targets["compressor"] == compressor].set_index(
                "prompt_id"
            )
            training_target = group.loc[
                list(training_ids), "target"
            ].to_numpy(dtype=np.float64)
            expected = (
                frozen_oof[
                    (frozen_oof["compressor"] == compressor)
                    & (frozen_oof["fold"] == fold)
                ]
                .set_index("prompt_id")
                .loc[list(validation_ids)]
            )
            for name, train_matrix, validation_matrix, alpha in (
                ("text", text_training, text_validation, text_alpha),
                (
                    "surface",
                    surface_training,
                    surface_validation,
                    surface_alpha,
                ),
            ):
                model = linear_model.Ridge(
                    alpha=alpha,
                    fit_intercept=True,
                    solver="lsqr",
                    tol=1e-10,
                    max_iter=10000,
                )
                model.fit(train_matrix, training_target)
                prediction = np.asarray(
                    model.predict(validation_matrix), dtype=np.float64
                )
                difference = float(
                    np.max(
                        np.abs(
                            prediction
                            - expected[f"{name}_prediction"].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )
                )
                prediction_max_difference[compressor][name] = max(
                    prediction_max_difference[compressor][name], difference
                )
            mean_difference = float(
                np.max(
                    np.abs(
                        float(np.mean(training_target))
                        - expected["mean_prediction"].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
            )
            prediction_max_difference[compressor]["mean"] = max(
                prediction_max_difference[compressor]["mean"], mean_difference
            )
    ranked = sorted(similarities, key=lambda item: (float(item[1]), item[0]))
    low_ids = tuple(prompt_id for prompt_id, _value in ranked[:LOW_PROMPTS])
    checks = {
        "folds": representation_checks,
        "prediction_max_difference": prediction_max_difference,
        "maximum_representation_difference": max(
            max(
                float(value)
                for key, value in fold.items()
                if key.endswith("_difference")
            )
            for fold in representation_checks
        ),
        "all_feature_names_exact": all(
            bool(fold["word_features_exact"])
            and bool(fold["char_features_exact"])
            for fold in representation_checks
        ),
    }
    return checks, low_ids


def nested_compare(
    expected: Any, observed: Any, path: str = "root"
) -> tuple[list[str], float]:
    errors = []
    maximum = 0.0
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        if set(expected) != set(observed):
            return [f"{path}: key mismatch"], maximum
        for key in expected:
            child, difference = nested_compare(
                expected[key], observed[key], f"{path}.{key}"
            )
            errors.extend(child)
            maximum = max(maximum, difference)
        return errors, maximum
    if (
        isinstance(expected, int | float)
        and not isinstance(expected, bool)
        and isinstance(observed, int | float)
        and not isinstance(observed, bool)
    ):
        difference = abs(float(expected) - float(observed))
        scale = max(1.0, abs(float(expected)), abs(float(observed)))
        if difference > 1e-12 * scale:
            errors.append(f"{path}: {expected!r} != {observed!r}")
        return errors, difference
    if expected != observed:
        errors.append(f"{path}: {expected!r} != {observed!r}")
    return errors, maximum


def point_skill(numerator: np.ndarray, denominator: np.ndarray) -> float:
    return 1.0 - float(np.mean(numerator)) / float(np.mean(denominator))


def bootstrap_skill(
    numerator: np.ndarray,
    denominator: np.ndarray,
    draws: NDArray[np.int64],
    mask: NDArray[np.bool_] | None,
) -> NDArray[np.float64]:
    if mask is None:
        return np.asarray(
            1.0
            - np.mean(numerator[draws], axis=1)
            / np.mean(denominator[draws], axis=1),
            dtype=np.float64,
        )
    selected = mask[draws]
    counts = np.sum(selected, axis=1)
    return np.asarray(
        1.0
        - (np.sum(numerator[draws] * selected, axis=1) / counts)
        / (np.sum(denominator[draws] * selected, axis=1) / counts),
        dtype=np.float64,
    )


def max_t(
    estimates: Mapping[str, float],
    bootstrap: Mapping[str, NDArray[np.float64]],
    direction: str,
) -> dict[str, Any]:
    claims = tuple(sorted(estimates))
    matrix = np.column_stack([bootstrap[claim] for claim in claims])
    point = np.asarray(
        [estimates[claim] for claim in claims], dtype=np.float64
    )
    standard_errors = np.std(matrix, axis=0, ddof=1)
    centered = matrix - point
    root = centered if direction == "lower" else -centered
    standardized = np.divide(
        root,
        standard_errors,
        out=np.zeros_like(root),
        where=standard_errors > 0.0,
    )
    critical = max(
        0.0,
        float(np.quantile(np.max(standardized, axis=1), CONFIDENCE)),
    )
    intervals = {}
    for index, claim in enumerate(claims):
        bound = critical * standard_errors[index]
        interval = {
            "estimate": float(point[index]),
            "standard_error": float(standard_errors[index]),
        }
        interval[direction] = float(
            point[index] - bound
            if direction == "lower"
            else point[index] + bound
        )
        intervals[claim] = interval
    return {
        "resamples": RESAMPLES,
        "seed": SEED,
        "confidence": CONFIDENCE,
        "direction": direction,
        "sidedness": "one-sided",
        "multiplicity": "simultaneous max-T over 12 claims",
        "max_t_critical": critical,
        "intervals": intervals,
    }


def independent_score(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, NDArray[np.float64]], dict[str, Any]]:
    prompts = tuple(
        sorted(str(value) for value in frame["prompt_id"].unique())
    )
    draws = np.random.default_rng(SEED).integers(
        0, PROMPTS, size=(RESAMPLES, PROMPTS), dtype=np.int64
    )
    estimates = {}
    bootstrap = {}
    metrics = {}
    loss_store = {}
    for compressor in COMPRESSORS:
        group = (
            frame[frame["compressor"] == compressor]
            .set_index("prompt_id")
            .loc[list(prompts)]
        )
        target = group["target"].to_numpy(dtype=np.float64)
        losses = {
            name: (
                target
                - group[f"{name}_prediction"].to_numpy(dtype=np.float64)
            )
            ** 2
            for name in ("text", "surface", "mean")
        }
        low = group["low_similarity"].to_numpy(dtype=np.bool_)
        definitions = {
            "overall_vs_mean": ("mean", None),
            "overall_vs_surface": ("surface", None),
            "low_vs_mean": ("mean", low),
            "low_vs_surface": ("surface", low),
        }
        skills = {}
        for kind, (comparator, mask) in definitions.items():
            name = f"{compressor}::{kind}"
            numerator = losses["text"]
            denominator = losses[comparator]
            estimates[name] = point_skill(
                numerator if mask is None else numerator[mask],
                denominator if mask is None else denominator[mask],
            )
            bootstrap[name] = bootstrap_skill(
                numerator, denominator, draws, mask
            )
            skills[kind] = estimates[name]
        metrics[compressor] = {
            "loss": {
                name: float(np.mean(values))
                for name, values in losses.items()
            },
            "skills": skills,
        }
        loss_store[compressor] = {
            "losses": losses,
            "low": low,
            "fold": group["fold"].to_numpy(dtype=np.int64),
            "similarity": group["nearest_training_similarity"].to_numpy(
                dtype=np.float64
            ),
        }
    inference: dict[str, Any] = {
        "minimum_overall_skill": MINIMUM_OVERALL_SKILL,
        "lower": max_t(estimates, bootstrap, "lower"),
        "upper": max_t(estimates, bootstrap, "upper"),
    }
    qualifying: list[str] = []
    ruled_out: dict[str, bool] = {}
    thresholds = {
        "overall_vs_mean": MINIMUM_OVERALL_SKILL,
        "overall_vs_surface": MINIMUM_OVERALL_SKILL,
        "low_vs_mean": 0.0,
        "low_vs_surface": 0.0,
    }
    for compressor in COMPRESSORS:
        qualifies = all(
            inference["lower"]["intervals"][f"{compressor}::{kind}"]["lower"]
            > threshold
            for kind, threshold in thresholds.items()
        )
        ruled_out[compressor] = any(
            inference["upper"]["intervals"][f"{compressor}::{kind}"]["upper"]
            <= threshold
            for kind, threshold in thresholds.items()
        )
        if qualifies:
            qualifying.append(compressor)
    status = (
        "positive"
        if qualifying
        else "negative"
        if all(ruled_out.values())
        else "ambiguous"
    )
    decision = {
        "status": status,
        "go": status == "positive",
        "qualifying_compressors": qualifying,
        "compressor_ruled_out": ruled_out,
        "action": {
            "positive": (
                "freeze_qualifiers_then_design_separate_incremental_C0_test"
            ),
            "negative": (
                "retire_cheap_H1_branch_and_return_to_distinct_hypothesis"
            ),
            "ambiguous": ("no_go_for_H1_and_return_to_distinct_hypothesis"),
        }[status],
    }
    return (
        {
            "metrics": metrics,
            "inference": inference,
            "decision": decision,
        },
        bootstrap,
        loss_store,
    )


def adversarial_sensitivity(
    loss_store: Mapping[str, Any],
    bootstrap: Mapping[str, NDArray[np.float64]],
) -> dict[str, Any]:
    result = {}
    thresholds = {
        "overall_vs_mean": MINIMUM_OVERALL_SKILL,
        "overall_vs_surface": MINIMUM_OVERALL_SKILL,
        "low_vs_mean": 0.0,
        "low_vs_surface": 0.0,
    }
    for compressor in COMPRESSORS:
        data = loss_store[compressor]
        losses = data["losses"]
        low = np.asarray(data["low"], dtype=np.bool_)
        similarity = np.asarray(data["similarity"], dtype=np.float64)
        delta = np.asarray(
            losses["surface"] - losses["text"], dtype=np.float64
        )
        if np.std(similarity) and np.std(delta):
            similarity_delta_correlation = float(
                np.corrcoef(similarity, delta)[0, 1]
            )
        else:
            similarity_delta_correlation = 0.0
        absolute = np.sort(np.abs(delta))[::-1]
        total_absolute = float(np.sum(absolute))
        leave_one_out: dict[str, list[float]] = {
            kind: [] for kind in thresholds
        }
        point_qualifying_removals = 0
        for removed in range(PROMPTS):
            keep = np.arange(PROMPTS) != removed
            low_keep = low & keep
            values = {
                "overall_vs_mean": point_skill(
                    losses["text"][keep], losses["mean"][keep]
                ),
                "overall_vs_surface": point_skill(
                    losses["text"][keep], losses["surface"][keep]
                ),
                "low_vs_mean": point_skill(
                    losses["text"][low_keep], losses["mean"][low_keep]
                ),
                "low_vs_surface": point_skill(
                    losses["text"][low_keep], losses["surface"][low_keep]
                ),
            }
            for kind, value in values.items():
                leave_one_out[kind].append(value)
            point_qualifying_removals += int(
                all(
                    values[kind] > threshold
                    for kind, threshold in thresholds.items()
                )
            )
        bootstrap_joint_gate = np.ones(RESAMPLES, dtype=np.bool_)
        for kind, threshold in thresholds.items():
            bootstrap_joint_gate &= (
                bootstrap[f"{compressor}::{kind}"] > threshold
            )
        result[compressor] = {
            "text_beats_surface_prompt_fraction": float(np.mean(delta > 0.0)),
            "similarity_vs_text_surface_loss_gain_correlation": (
                similarity_delta_correlation
            ),
            "top_five_absolute_prompt_contribution_fraction": (
                float(np.sum(absolute[:5]) / total_absolute)
                if total_absolute
                else 0.0
            ),
            "leave_one_prompt_out": {
                kind: {
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
                for kind, values in leave_one_out.items()
            },
            "single_prompt_removals_with_all_point_gates": (
                point_qualifying_removals
            ),
            "bootstrap_draws_with_all_point_gates": int(
                np.sum(bootstrap_joint_gate)
            ),
            "bootstrap_fraction_with_all_point_gates": float(
                np.mean(bootstrap_joint_gate)
            ),
        }
    return result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H1 audit: {args.output}"
        )
    manifest = load_json(args.representation_manifest)
    protocol = load_json(args.protocol_lock)
    scoring_lock = load_json(args.scoring_lock)
    report = load_json(args.report)
    hashes = {
        "source_oof_sha256": sha256_file(args.source_oof),
        "prompt_table_sha256": sha256_file(args.prompt_table),
        "representation_archive_sha256": sha256_file(
            args.representation_archive
        ),
        "representation_manifest_sha256": sha256_file(
            args.representation_manifest
        ),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "scoring_lock_sha256": sha256_file(args.scoring_lock),
        "oof_predictions_sha256": sha256_file(args.oof),
        "report_sha256": sha256_file(args.report),
        "feature_script_sha256": sha256_file(args.feature_script),
        "compile_script_sha256": sha256_file(args.compile_script),
        "score_script_sha256": sha256_file(args.score_script),
    }
    implementation = {
        "scripts/magnitude_e1_h1_features.py": hashes[
            "feature_script_sha256"
        ],
        "scripts/compile_magnitude_e1_h1_representation.py": hashes[
            "compile_script_sha256"
        ],
        "scripts/score_magnitude_e1_h1.py": hashes["score_script_sha256"],
    }
    artifact_checks = {
        "source_matches_protocol": hashes["source_oof_sha256"]
        == protocol["anchors"]["source_oof_sha256"],
        "source_matches_scoring_lock": hashes["source_oof_sha256"]
        == scoring_lock["source_oof_sha256"],
        "source_matches_report": hashes["source_oof_sha256"]
        == report["provenance"]["source_oof_sha256"],
        "prompt_table_matches_manifest": hashes["prompt_table_sha256"]
        == manifest["prompt_table_sha256"],
        "prompt_table_matches_scoring_lock": hashes["prompt_table_sha256"]
        == scoring_lock["prompt_table_sha256"],
        "archive_matches_manifest": hashes["representation_archive_sha256"]
        == manifest["representation_archive_sha256"],
        "archive_matches_scoring_lock": hashes[
            "representation_archive_sha256"
        ]
        == scoring_lock["representation_archive_sha256"],
        "manifest_matches_scoring_lock": hashes[
            "representation_manifest_sha256"
        ]
        == scoring_lock["representation_manifest_sha256"],
        "protocol_matches_manifest": hashes["protocol_lock_sha256"]
        == manifest["protocol_lock_sha256"],
        "protocol_matches_scoring_lock": hashes["protocol_lock_sha256"]
        == scoring_lock["protocol_lock_sha256"],
        "scoring_lock_matches_report": hashes["scoring_lock_sha256"]
        == report["provenance"]["scoring_lock_sha256"],
        "oof_matches_report": hashes["oof_predictions_sha256"]
        == report["provenance"]["oof_predictions_sha256"],
        "report_implementation_matches": hashes["score_script_sha256"]
        == report["provenance"]["implementation_sha256"],
        "implementation_matches_protocol": implementation
        == protocol["implementation_sha256"],
        "implementation_matches_manifest": implementation
        == manifest["implementation_sha256"],
        "implementation_matches_scoring_lock": implementation
        == scoring_lock["implementation_sha256"],
        "representation_label_blind": manifest[
            "outcome_or_label_columns_loaded"
        ]
        == [],
        "representation_gold_blind": manifest[
            "gold_instruction_columns_loaded"
        ]
        == [],
        "protected_prompt_text_unmaterialized": manifest[
            "protected_prompt_text_rows_materialized"
        ]
        == 0,
        "confirmation_remained_sealed": report["provenance"][
            "confirmation_status"
        ]
        == "sealed_not_run",
    }

    prompts = load_prompt_table(args.prompt_table)
    source = load_source(args.source_oof)
    targets = build_targets(source, prompts)
    frozen_oof = pq.read_table(args.oof).to_pandas()  # type: ignore[no-untyped-call]
    with np.load(args.representation_archive, allow_pickle=False) as loaded:
        archive = {key: loaded[key] for key in loaded.files}
    representation_audit, low_ids = rebuild_and_refit(
        prompts, targets, archive, frozen_oof
    )
    low_hash = hash_prompt_ids(list(low_ids))
    reproduction_checks = {
        "maximum_representation_difference_within_tolerance": (
            representation_audit["maximum_representation_difference"]
            <= REPRESENTATION_REPRODUCTION_TOLERANCE
        ),
        "all_feature_names_exact": representation_audit[
            "all_feature_names_exact"
        ],
        "all_model_predictions_reproduced": all(
            float(value) <= PREDICTION_REPRODUCTION_TOLERANCE
            for compressor in representation_audit[
                "prediction_max_difference"
            ].values()
            for value in compressor.values()
        ),
        "low_similarity_roster_reproduced": low_hash
        == manifest["low_similarity_prompt_ids_sha256"],
    }

    independent, bootstrap, loss_store = independent_score(frozen_oof)
    metric_errors, metric_difference = nested_compare(
        report["metrics"], independent["metrics"], "metrics"
    )
    inference_errors, inference_difference = nested_compare(
        report["inference"], independent["inference"], "inference"
    )
    decision_errors, _decision_difference = nested_compare(
        report["decision"], independent["decision"], "decision"
    )
    scoring_checks = {
        "metrics_reproduced": not metric_errors,
        "inference_reproduced": not inference_errors,
        "decision_reproduced": not decision_errors,
        "locked_result_is_ambiguous_no_go": independent["decision"]["status"]
        == "ambiguous"
        and not independent["decision"]["go"],
    }
    sensitivity = adversarial_sensitivity(loss_store, bootstrap)
    checks = {
        "artifacts": artifact_checks,
        "representation_and_models": reproduction_checks,
        "scoring": scoring_checks,
    }
    passed = all(
        bool(value)
        for section in checks.values()
        for value in section.values()
    )
    conclusion = (
        "valid_ambiguous_no_go_retire_current_H1_probe"
        if passed
        else "invalid_return_to_understand_and_repair"
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "independent_h1_result_audit_complete",
        "pass": passed,
        "conclusion": conclusion,
        "hashes": hashes,
        "checks": checks,
        "representation_audit": representation_audit,
        "numerical_reproduction_tolerances": {
            "representation_absolute": (
                REPRESENTATION_REPRODUCTION_TOLERANCE
            ),
            "prediction_absolute": PREDICTION_REPRODUCTION_TOLERANCE,
        },
        "scoring_reproduction": {
            "metric_errors": metric_errors,
            "metric_max_absolute_difference": metric_difference,
            "inference_errors": inference_errors,
            "inference_max_absolute_difference": inference_difference,
            "decision_errors": decision_errors,
        },
        "adversarial_sensitivity": sensitivity,
        "scientific_interpretation": {
            "supported": [
                (
                    "The union text probe predicts prompt-macro dq beyond "
                    "fold means on frozen development folds."
                ),
                (
                    "Generic surface form explains nearly all of that gain "
                    "for ExpectedAttention and Knorm."
                ),
                (
                    "StreamingLLM has favorable text-over-surface point "
                    "estimates that remain inconclusive under the locked "
                    "simultaneous gates."
                ),
            ],
            "not_supported": [
                "H1 qualification",
                "incremental improvement over C0 or M3",
                "semantic understanding",
                "future-prompt generalization",
                "confirmatory evidence",
            ],
            "next_action": (
                "Freeze the no-go, run no H1 variant, and return to a "
                "genuinely distinct hypothesis."
            ),
        },
        "confirmation_status": "sealed_not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "event": "h1_independent_audit_complete",
                "pass": passed,
                "conclusion": conclusion,
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
