from pathlib import Path

import numpy as np

from herald.features import FEATURE_NAMES
from herald.storage import append_hybrid, save_reference
from herald.switch_dataset import (
    build_switch_dataset,
    build_switch_rows,
    parse_hybrid_shard,
)


def _features(steps: int) -> np.ndarray:
    arr = np.zeros((steps, len(FEATURE_NAMES)), dtype=np.float32)
    for i, name in enumerate(FEATURE_NAMES):
        arr[:, i] = np.arange(steps, dtype=np.float32) + i
        if name == "kl_prev":
            arr[0, i] = np.nan
    return arr


def test_parse_hybrid_shard() -> None:
    assert parse_hybrid_shard(Path("snapkv__0.7500.jsonl")) == (
        "snapkv",
        0.75,
    )


def test_build_switch_rows_joins_hybrid_to_reference_features(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1, 2],
        gen_ids=[3, 4, 5],
        text="ref",
        q=1.0,
        features=_features(3),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=1,
        new_ids=[4, 5],
        text="hyb",
        q=0.25,
        dq=0.75,
    )

    rows, summary = build_switch_rows(
        tmp_path / "llama" / "gsm8k",
        model="llama",
        task="gsm8k",
    )

    assert summary["n_rows"] == 1
    row = rows[0]
    assert row["model"] == "llama"
    assert row["task"] == "gsm8k"
    assert row["prompt_id"] == "p0"
    assert row["compressor"] == "snapkv"
    assert row["ratio"] == 0.5
    assert row["s"] == 1
    assert row["q_ref"] == 1.0
    assert row["q_hybrid"] == 0.25
    assert row["dq"] == 0.75
    assert row["damaged"] == 1
    assert row["major_damage"] == 1
    assert row["relative_s"] == 1 / 3
    assert "feat__entropy" in row
    assert "feat__entropy_delta" in row


def test_build_switch_rows_counts_skips(tmp_path: Path) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[],
        gen_ids=[1],
        text="ref",
        q=0.0,
        features=_features(1),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=3,
        new_ids=[],
        text="bad",
        q=0.0,
        dq=0.0,
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="missing",
        s=0,
        new_ids=[],
        text="bad",
        q=0.0,
        dq=0.0,
    )

    rows, summary = build_switch_rows(
        tmp_path / "llama" / "gsm8k",
        model="llama",
        task="gsm8k",
    )

    assert rows == []
    assert summary["skipped_s_out_of_range"] == 1
    assert summary["skipped_missing_ref"] == 1


def test_build_switch_dataset_filters_tasks(tmp_path: Path) -> None:
    for task in ("gsm8k", "humaneval"):
        save_reference(
            tmp_path,
            "llama",
            task,
            prompt_id="p0",
            prompt_input_ids=[],
            gen_ids=[1, 2],
            text="ref",
            q=1.0,
            features=_features(2),
        )
        append_hybrid(
            tmp_path,
            "llama",
            task,
            "random",
            0.25,
            prompt_id="p0",
            s=0,
            new_ids=[],
            text="hyb",
            q=0.0,
            dq=1.0,
        )

    rows, summary = build_switch_dataset(tmp_path, tasks=["humaneval"])

    assert len(rows) == 1
    assert rows[0]["task"] == "humaneval"
    assert summary["n_tasks"] == 1
