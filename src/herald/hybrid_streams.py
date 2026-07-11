"""Extract aligned grace-window streams from completed sweep artifacts."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from herald.features import FEATURE_NAMES
from herald.storage import hybrid_feature_path, safe_id


@dataclass(frozen=True)
class HybridStreams:
    """Fixed-width hybrid blocks aligned one-for-one with switch rows."""

    blocks: np.ndarray
    lengths: np.ndarray
    trailing: np.ndarray
    keys: np.ndarray
    source_parquet_sha256: str


def row_key(row: dict[str, Any]) -> str:
    """Return the canonical identity used to verify stream alignment."""
    return json.dumps(
        [
            str(row["model"]),
            str(row["task"]),
            str(row["prompt_id"]),
            str(row["compressor"]),
            _finite_float(row["ratio"], source="ratio"),
            _integer(row["s"], source="switch position"),
        ],
        separators=(",", ":"),
    )


def extract_hybrid_streams(
    rows: Sequence[dict[str, Any]],
    results_dir: Path,
    *,
    source_parquet_sha256: str,
    max_block_tokens: int = 16,
    trailing_tokens: int = 8,
) -> HybridStreams:
    """Materialize the training blocks used by live alarm bundle export."""
    if max_block_tokens < 1:
        raise ValueError("max_block_tokens must be positive")
    if trailing_tokens < 1:
        raise ValueError("trailing_tokens must be positive")
    if len(source_parquet_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in source_parquet_sha256
    ):
        raise ValueError(
            "source_parquet_sha256 must be 64 hexadecimal characters"
        )
    n_rows = len(rows)
    n_features = len(FEATURE_NAMES)
    blocks = np.full(
        (n_rows, max_block_tokens, n_features),
        np.nan,
        dtype=np.float16,
    )
    lengths = np.zeros(n_rows, dtype=np.int32)
    trailing = np.full((n_rows, n_features), np.nan, dtype=np.float32)
    keys: list[str] = []
    reference_cache: dict[tuple[str, str, str], np.ndarray] = {}

    for index, row in enumerate(rows):
        model = str(row["model"])
        task = str(row["task"])
        prompt_id = str(row["prompt_id"])
        compressor = str(row["compressor"])
        ratio = _finite_float(row["ratio"], source="ratio")
        switch_s = _integer(row["s"], source="switch position")
        if switch_s < 0:
            raise ValueError("switch position must be non-negative")
        reference_key = (model, task, prompt_id)
        reference = reference_cache.get(reference_key)
        if reference is None:
            reference = _load_feature_matrix(
                results_dir
                / model
                / task
                / "references"
                / f"{safe_id(prompt_id)}.npy",
                source="reference",
            )
            reference_cache[reference_key] = reference
        if switch_s > len(reference):
            raise ValueError(
                f"switch position {switch_s} exceeds reference length "
                f"for {prompt_id!r}"
            )
        hybrid = _load_feature_matrix(
            hybrid_feature_path(
                results_dir,
                model,
                task,
                compressor,
                ratio,
                prompt_id,
                switch_s,
            ),
            source="hybrid",
        )
        block_length = min(len(hybrid), max_block_tokens)
        blocks[index, :block_length] = hybrid[:block_length]
        lengths[index] = block_length
        if switch_s:
            trailing[index] = reference[
                max(0, switch_s - trailing_tokens) : switch_s
            ].mean(axis=0)
        keys.append(row_key(row))
    return HybridStreams(
        blocks=blocks,
        lengths=lengths,
        trailing=trailing,
        keys=np.asarray(keys),
        source_parquet_sha256=source_parquet_sha256,
    )


def save_hybrid_streams(path: Path, streams: HybridStreams) -> None:
    """Write a portable numeric NPZ with explicit row-identity keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        blocks=streams.blocks,
        lengths=streams.lengths,
        trailing=streams.trailing,
        keys=streams.keys,
        source_parquet_sha256=np.asarray(streams.source_parquet_sha256),
    )


def validate_stream_alignment(
    rows: Sequence[dict[str, Any]], keys: np.ndarray
) -> None:
    """Reject streams that do not match the source parquet row-for-row."""
    expected = [row_key(row) for row in rows]
    observed = [str(key) for key in keys.tolist()]
    if observed != expected:
        raise ValueError(
            "hybrid stream keys do not align with the selected parquet rows"
        )


def _load_feature_matrix(
    path: Path, *, source: str
) -> npt.NDArray[np.float32]:
    try:
        matrix = cast(
            npt.NDArray[np.float32],
            np.load(path, allow_pickle=False).astype(np.float32),
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            f"could not load {source} features at {path}"
        ) from error
    if matrix.ndim != 2 or matrix.shape[1] < len(FEATURE_NAMES):
        raise ValueError(
            f"{source} features at {path} do not contain raw logit features"
        )
    return matrix[:, : len(FEATURE_NAMES)]


def _finite_float(value: object, *, source: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected numeric {source}, got {value!r}")
    try:
        out = float(value)
    except ValueError as error:
        raise ValueError(
            f"expected numeric {source}, got {value!r}"
        ) from error
    if not np.isfinite(out):
        raise ValueError(f"expected finite {source}, got {value!r}")
    return out


def _integer(value: object, *, source: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected integer {source}, got {value!r}")
    try:
        return int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"expected integer {source}, got {value!r}"
        ) from error
