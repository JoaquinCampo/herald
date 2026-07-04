from pathlib import Path

import numpy as np

from herald.features import FEATURE_NAMES
from herald.hybrid_token_dataset import build_hybrid_token_dataset
from herald.storage import append_hybrid, save_reference


def _features(steps: int) -> np.ndarray:
    arr = np.zeros((steps, len(FEATURE_NAMES)), dtype=np.float32)
    for i in range(len(FEATURE_NAMES)):
        arr[:, i] = np.arange(steps, dtype=np.float32) + i
    arr[0, FEATURE_NAMES.index("kl_prev")] = np.nan
    return arr


def test_build_hybrid_token_dataset_uses_compressed_features(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1],
        gen_ids=[10, 11, 12, 13],
        text="ref",
        q=1.0,
        features=_features(4),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=2,
        new_ids=[12, 99],
        text="hyb",
        q=0.25,
        dq=0.75,
        features=_features(2),
    )

    rows, summary = build_hybrid_token_dataset(tmp_path)

    assert summary["n_rows"] == 2
    assert len(rows) == 2
    assert rows[0]["switch_s"] == 2
    assert rows[0]["compressed_token_pos"] == 0
    assert rows[1]["compressed_token_pos"] == 1
    assert rows[1]["global_token_pos"] == 3
    assert rows[0]["q_ref"] == 1.0
    assert rows[0]["q_compressed"] == 0.25
    assert rows[0]["dq"] == 0.75
    assert rows[0]["damage_positive"] == 1
    assert rows[0]["damage_major"] == 1
    assert "feat__entropy" in rows[0]


def test_build_hybrid_token_dataset_skips_featureless_legacy_rows(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1],
        gen_ids=[10],
        text="ref",
        q=1.0,
        features=_features(1),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[10],
        text="hyb",
        q=1.0,
        dq=0.0,
    )

    rows, summary = build_hybrid_token_dataset(tmp_path)

    assert rows == []
    task_summary = summary["tasks"][0]
    assert task_summary["skipped_missing_features"] == 1


def test_build_hybrid_token_dataset_rejects_feature_length_mismatch(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1],
        gen_ids=[10, 11],
        text="ref",
        q=1.0,
        features=_features(2),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[10, 11],
        text="hyb",
        q=0.0,
        dq=1.0,
        features=_features(1),
    )

    import pytest

    with pytest.raises(ValueError, match="feature/token length mismatch"):
        build_hybrid_token_dataset(tmp_path)


def test_build_hybrid_token_dataset_skips_missing_reference(
    tmp_path: Path,
) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="missing",
        s=0,
        new_ids=[10],
        text="hyb",
        q=0.0,
        dq=1.0,
        features=_features(1),
    )

    rows, summary = build_hybrid_token_dataset(tmp_path)

    assert rows == []
    assert summary["tasks"][0]["skipped_missing_reference"] == 1
