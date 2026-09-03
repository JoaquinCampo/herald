"""Label-blind feature mechanics for the E1 H1 prompt-semantics probe."""

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import sklearn.preprocessing as preprocessing  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import sparse  # type: ignore[import-untyped]
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer,
)

WORD_PATTERN_TEXT = r"(?u)\b\w\w+\b"
DECIMAL_PATTERN_TEXT = r"\b\d+(?:\.\d+)?\b"
LIST_MARKER_PATTERN_TEXT = r"(?m)^\s*(?:[-*+]|(?:\d+|[A-Za-z])[.)])\s+"
WORD_PATTERN = re.compile(WORD_PATTERN_TEXT)
DECIMAL_PATTERN = re.compile(DECIMAL_PATTERN_TEXT)
LIST_MARKER_PATTERN = re.compile(LIST_MARKER_PATTERN_TEXT)
SLOPE_EDF = 10.0
EDF_TOLERANCE = 1e-10
RANK_RELATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class Representation:
    text_training: sparse.csr_matrix
    text_validation: sparse.csr_matrix
    surface_training: NDArray[np.float64]
    surface_validation: NDArray[np.float64]
    nearest_training_similarity: NDArray[np.float64]
    text_alpha: float
    surface_alpha: float
    text_rank: int
    surface_rank: int
    text_edf: float
    surface_edf: float
    word_features: tuple[str, ...]
    char_features: tuple[str, ...]
    word_idf: NDArray[np.float64]
    char_idf: NDArray[np.float64]
    surface_mean: NDArray[np.float64]
    surface_scale: NDArray[np.float64]


def word_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
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


def character_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
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
    quote_count = sum(character in "'\"`" for character in text)
    punctuation_count = sum(
        unicodedata.category(character).startswith("P") for character in text
    )
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
        safe_fraction(punctuation_count, code_points),
        math.log1p(len(DECIMAL_PATTERN.findall(text))),
        math.log1p(quote_count),
        math.log1p(len(LIST_MARKER_PATTERN.findall(text))),
        math.log1p(sum(character in ".?!" for character in text)),
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.isfinite(result).all():
        raise ValueError("surface feature vector is invalid")
    return result


def surface_matrix(texts: list[str]) -> NDArray[np.float64]:
    matrix = np.vstack([surface_vector(text) for text in texts])
    if matrix.shape != (len(texts), 16):
        raise ValueError("surface feature matrix has an unexpected shape")
    return matrix


def centered_gram_eigenvalues(
    matrix: sparse.spmatrix | NDArray[np.float64],
) -> NDArray[np.float64]:
    product = matrix @ matrix.T
    gram = (
        cast("Any", product).toarray()
        if sparse.issparse(product)
        else np.asarray(product, dtype=np.float64)
    )
    row_mean = np.mean(gram, axis=1, keepdims=True)
    centered = gram - row_mean - row_mean.T + float(np.mean(gram))
    centered = (centered + centered.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(centered)
    maximum = float(np.max(eigenvalues))
    if not np.isfinite(eigenvalues).all() or maximum <= 0.0:
        raise ValueError("centered representation Gram matrix is degenerate")
    tolerance = maximum * RANK_RELATIVE_TOLERANCE
    return eigenvalues[eigenvalues > tolerance]


def effective_degrees_of_freedom(
    eigenvalues: NDArray[np.float64], alpha: float
) -> float:
    return float(np.sum(eigenvalues / (eigenvalues + alpha)))


def alpha_for_slope_edf(
    matrix: sparse.spmatrix | NDArray[np.float64],
    target: float = SLOPE_EDF,
) -> tuple[float, int, float]:
    eigenvalues = centered_gram_eigenvalues(matrix)
    rank = len(eigenvalues)
    if rank <= target:
        raise ValueError(
            f"centered representation rank {rank} <= EDF {target}"
        )
    lower = 0.0
    upper = float(np.max(eigenvalues))
    while effective_degrees_of_freedom(eigenvalues, upper) > target:
        upper *= 2.0
    for _ in range(200):
        midpoint = (lower + upper) * 0.5
        if effective_degrees_of_freedom(eigenvalues, midpoint) > target:
            lower = midpoint
        else:
            upper = midpoint
    alpha = (lower + upper) * 0.5
    achieved = effective_degrees_of_freedom(eigenvalues, alpha)
    if (
        not math.isfinite(alpha)
        or alpha <= 0.0
        or abs(achieved - target) > EDF_TOLERANCE
    ):
        raise ValueError("failed to solve the label-blind EDF constraint")
    return alpha, rank, achieved


def build_representation(
    training_texts: list[str], validation_texts: list[str]
) -> Representation:
    word = word_vectorizer()
    character = character_vectorizer()
    word_training = word.fit_transform(training_texts).tocsr()
    word_validation = word.transform(validation_texts).tocsr()
    char_training = character.fit_transform(training_texts).tocsr()
    char_validation = character.transform(validation_texts).tocsr()
    matrices = (
        word_training,
        word_validation,
        char_training,
        char_validation,
    )
    if any(
        np.any(np.asarray(matrix.getnnz(axis=1), dtype=np.int64) == 0)
        for matrix in matrices
    ):
        raise ValueError("TF-IDF produced an empty prompt row")
    block_scale = 1.0 / math.sqrt(2.0)
    text_training = sparse.hstack(
        [word_training * block_scale, char_training * block_scale],
        format="csr",
        dtype=np.float64,
    )
    text_validation = sparse.hstack(
        [word_validation * block_scale, char_validation * block_scale],
        format="csr",
        dtype=np.float64,
    )
    word_similarity = (word_validation @ word_training.T).toarray()
    char_similarity = (char_validation @ char_training.T).toarray()
    nearest = np.max((word_similarity + char_similarity) * 0.5, axis=1)

    raw_surface_training = surface_matrix(training_texts)
    raw_surface_validation = surface_matrix(validation_texts)
    scaler = preprocessing.StandardScaler()
    scaled_surface_training = np.asarray(
        scaler.fit_transform(raw_surface_training), dtype=np.float64
    )
    scaled_surface_validation = np.asarray(
        scaler.transform(raw_surface_validation), dtype=np.float64
    )
    if not (
        np.isfinite(text_training.data).all()
        and np.isfinite(text_validation.data).all()
        and np.isfinite(scaled_surface_training).all()
        and np.isfinite(scaled_surface_validation).all()
        and np.isfinite(nearest).all()
    ):
        raise ValueError("H1 representation contains nonfinite values")
    text_alpha, text_rank, text_edf = alpha_for_slope_edf(text_training)
    surface_alpha, surface_rank, surface_edf = alpha_for_slope_edf(
        scaled_surface_training
    )
    return Representation(
        text_training=text_training,
        text_validation=text_validation,
        surface_training=scaled_surface_training,
        surface_validation=scaled_surface_validation,
        nearest_training_similarity=np.asarray(nearest, dtype=np.float64),
        text_alpha=text_alpha,
        surface_alpha=surface_alpha,
        text_rank=text_rank,
        surface_rank=surface_rank,
        text_edf=text_edf,
        surface_edf=surface_edf,
        word_features=tuple(
            str(value) for value in word.get_feature_names_out()
        ),
        char_features=tuple(
            str(value) for value in character.get_feature_names_out()
        ),
        word_idf=np.asarray(word.idf_, dtype=np.float64),
        char_idf=np.asarray(character.idf_, dtype=np.float64),
        surface_mean=np.asarray(scaler.mean_, dtype=np.float64),
        surface_scale=np.asarray(scaler.scale_, dtype=np.float64),
    )
