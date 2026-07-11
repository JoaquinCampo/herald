# pyright: reportMissingImports=false

"""Extraction tests for frozen grace-window training streams."""

from pathlib import Path

import numpy as np
import pytest

from herald.features import FEATURE_NAMES
from herald.hybrid_streams import (
    extract_hybrid_streams,
    row_key,
    save_hybrid_streams,
    validate_stream_alignment,
)
from herald.storage import append_hybrid, save_reference


def _features(steps: int, offset: float = 0.0) -> np.ndarray:
    values = np.arange(steps * len(FEATURE_NAMES), dtype=np.float32).reshape(
        steps, len(FEATURE_NAMES)
    )
    return values + offset


def _row() -> dict[str, object]:
    return {
        "model": "llama",
        "task": "ifeval",
        "prompt_id": "p0",
        "compressor": "expected_attention_stats",
        "ratio": 0.5,
        "s": 2,
    }


def test_extracts_aligned_quantized_blocks_and_trailing_window(
    tmp_path: Path,
) -> None:
    reference = _features(5)
    hybrid = _features(4, offset=100.0)
    save_reference(
        tmp_path,
        "llama",
        "ifeval",
        prompt_id="p0",
        prompt_input_ids=[1, 2],
        gen_ids=[3, 4, 5, 6, 7],
        text="reference",
        q=1.0,
        features=reference,
    )
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "expected_attention_stats",
        0.5,
        prompt_id="p0",
        s=2,
        new_ids=[5, 6, 7],
        text="hybrid",
        q=1.0,
        dq=0.0,
        features=hybrid,
    )

    row = _row()
    streams = extract_hybrid_streams(
        [row],
        tmp_path,
        source_parquet_sha256="a" * 64,
        max_block_tokens=3,
    )

    assert streams.lengths.tolist() == [3]
    np.testing.assert_array_equal(
        streams.blocks[0], hybrid[:3].astype(np.float16)
    )
    np.testing.assert_allclose(
        streams.trailing[0], reference[:2].mean(axis=0)
    )
    assert streams.keys.tolist() == [row_key(row)]
    validate_stream_alignment([row], streams.keys)

    out = tmp_path / "streams.npz"
    save_hybrid_streams(out, streams)
    with np.load(out, allow_pickle=False) as data:
        assert data["keys"].tolist() == [row_key(row)]
        assert data["source_parquet_sha256"].item() == "a" * 64


def test_rejects_misaligned_stream_keys() -> None:
    row = _row()
    with pytest.raises(ValueError, match="do not align"):
        validate_stream_alignment([row], np.asarray(["wrong-key"]))
