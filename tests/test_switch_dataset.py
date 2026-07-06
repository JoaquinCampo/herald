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
    compressor, ratio = parse_hybrid_shard(Path("snapkv__0.7500.jsonl"))

    assert compressor == "snapkv"
    assert ratio == 0.75


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


def test_build_switch_rows_deduplicates_hybrid_retries(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[],
        gen_ids=[1, 2],
        text="ref",
        q=1.0,
        features=_features(2),
    )
    for q, dq in ((0.0, 1.0), (0.5, 0.5)):
        append_hybrid(
            tmp_path,
            "llama",
            "gsm8k",
            "snapkv",
            0.5,
            prompt_id="p0",
            s=1,
            new_ids=[],
            text="hyb",
            q=q,
            dq=dq,
        )

    rows, summary = build_switch_rows(
        tmp_path / "llama" / "gsm8k",
        model="llama",
        task="gsm8k",
    )

    assert summary["n_rows"] == 1
    assert summary["n_hybrids_seen"] == 1
    assert rows[0]["q_hybrid"] == 0.5
    assert rows[0]["dq"] == 0.5


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


def test_build_switch_rows_probe_and_press_columns(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1, 2],
        gen_ids=[3, 4, 5, 6],
        text="ref",
        q=1.0,
        features=_features(4),
    )
    hyb_feats = (_features(2) + 100.0).astype(np.float32)
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=1,
        new_ids=[4, 9],
        text="hyb",
        q=0.25,
        dq=0.75,
        features=hyb_feats,
        press_features={
            "press_evicted_reliance_lmean": 0.125,
            "press_evicted_reliance_lvar": 0.01,
        },
    )

    rows, _ = build_switch_rows(
        tmp_path / "llama" / "gsm8k",
        model="llama",
        task="gsm8k",
    )
    row = rows[0]
    # first hybrid token 4 == ref token at s=1; second diverges
    assert row["probe__token_match"] == 1.0
    assert row["probe__match_len4"] == 1.0
    # hybrid step-0 scalar and its delta vs the reference at s
    ent = FEATURE_NAMES.index("entropy")
    assert row["probe__h0_entropy"] == float(hyb_feats[0, ent])
    ref_at_s = _features(4)[1, ent]
    assert (
        abs(row["probe__d0_entropy"] - (hyb_feats[0, ent] - ref_at_s)) < 1e-4
    )
    assert row["press__evicted_reliance_lmean"] == 0.125
    assert row["press__evicted_reliance_lvar"] == 0.01


def test_build_switch_rows_widened_reference_features(
    tmp_path: Path,
) -> None:
    names = list(FEATURE_NAMES) + ["attn_entropy_lmean"]
    wide = np.concatenate(
        [_features(3), np.full((3, 1), 7.0, dtype=np.float32)],
        axis=1,
    )
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1, 2],
        gen_ids=[3, 4, 5],
        text="ref",
        q=1.0,
        features=wide,
        feature_names=names,
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=1,
        new_ids=[4],
        text="hyb",
        q=0.25,
        dq=0.75,
    )
    rows, _ = build_switch_rows(
        tmp_path / "llama" / "gsm8k",
        model="llama",
        task="gsm8k",
    )
    row = rows[0]
    assert row["feat__attn_entropy_lmean"] == 7.0
    assert "feat__entropy_delta" in row


def test_rows_to_table_keeps_late_columns(tmp_path: Path) -> None:
    """Columns appearing only in later rows must survive the write."""
    from herald.switch_dataset import rows_to_table

    rows = [
        {"task": "gsm8k", "dq": 0.0, "feat__entropy": 1.0},
        {
            "task": "ifeval",
            "dq": 1.0,
            "feat__entropy": 2.0,
            "probe__h0_entropy": 3.5,
        },
    ]
    table = rows_to_table(rows)
    assert "probe__h0_entropy" in table.column_names
    got = table.to_pylist()
    assert got[0]["probe__h0_entropy"] is None
    assert got[1]["probe__h0_entropy"] == 3.5
