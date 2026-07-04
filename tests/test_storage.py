"""Tests for src/herald/storage.py.

All tests are model-free and use pytest's tmp_path fixture.
"""

from pathlib import Path

import numpy as np
import pytest

from herald.storage import (
    append_hybrid,
    hybrid_done,
    hybrid_feature_path,
    legacy_safe_id,
    load_reference,
    reference_done,
    safe_id,
    save_reference,
)

# ---------------------------------------------------------------------------
# safe_id
# ---------------------------------------------------------------------------


def test_safe_id_no_slash() -> None:
    assert safe_id("abc123") == "abc123"


def test_safe_id_replaces_slash() -> None:
    assert safe_id("HumanEval/3") == "HumanEval%2F3"


def test_safe_id_multiple_slashes() -> None:
    assert safe_id("a/b/c") == "a%2Fb%2Fc"


def test_safe_id_is_injective_for_legacy_collision() -> None:
    assert legacy_safe_id("a/b") == legacy_safe_id("a_b")
    assert safe_id("a/b") != safe_id("a_b")


# ---------------------------------------------------------------------------
# reference_done / save_reference / load_reference
# ---------------------------------------------------------------------------


def _make_features(n: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random(n).astype(np.float32)


def test_reference_done_empty(tmp_path: Path) -> None:
    assert reference_done(tmp_path, "llama", "gsm8k") == set()


def test_save_and_reference_done(tmp_path: Path) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p0",
        prompt_input_ids=[1, 2, 3],
        gen_ids=[4, 5],
        text="hello",
        q=0.9,
        features=_make_features(),
    )
    done = reference_done(tmp_path, "llama", "gsm8k")
    assert "p0" in done


def test_load_reference_roundtrip(tmp_path: Path) -> None:
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p1",
        prompt_input_ids=[10, 20],
        gen_ids=[30],
        text="world",
        q=0.5,
        features=_make_features(),
    )
    rec = load_reference(tmp_path, "llama", "gsm8k", "p1")
    assert rec["prompt_input_ids"] == [10, 20]
    assert rec["gen_ids"] == [30]
    assert rec["text"] == "world"
    assert rec["q"] == pytest.approx(0.5)


def test_npy_roundtrip_float16(tmp_path: Path) -> None:
    feats = _make_features(16)
    save_reference(
        tmp_path,
        "llama",
        "gsm8k",
        prompt_id="p2",
        prompt_input_ids=[],
        gen_ids=[],
        text="",
        q=0.0,
        features=feats,
    )
    sid = safe_id("p2")
    npy_path = tmp_path / "llama" / "gsm8k" / "references" / f"{sid}.npy"
    loaded = np.load(npy_path)
    # Stored as float16; allow rounding error
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(loaded.astype(np.float32), feats, atol=1e-3)


def test_prompt_id_with_slash(tmp_path: Path) -> None:
    pid = "HumanEval/3"
    save_reference(
        tmp_path,
        "llama",
        "humaneval",
        prompt_id=pid,
        prompt_input_ids=[1],
        gen_ids=[2],
        text="code",
        q=0.7,
        features=_make_features(),
    )
    done = reference_done(tmp_path, "llama", "humaneval")
    assert pid in done
    rec = load_reference(tmp_path, "llama", "humaneval", pid)
    assert rec["text"] == "code"


def test_load_reference_falls_back_to_legacy_safe_id(tmp_path: Path) -> None:
    pid = "HumanEval/7"
    save_reference(
        tmp_path,
        "llama",
        "humaneval",
        prompt_id=pid,
        prompt_input_ids=[1],
        gen_ids=[2],
        text="code",
        q=0.7,
        features=_make_features(),
    )
    ref_dir = tmp_path / "llama" / "humaneval" / "references"
    encoded = ref_dir / f"{safe_id(pid)}.json"
    legacy = ref_dir / f"{legacy_safe_id(pid)}.json"
    encoded.rename(legacy)

    rec = load_reference(tmp_path, "llama", "humaneval", pid)

    assert rec["text"] == "code"


def test_load_reference_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_reference(tmp_path, "llama", "gsm8k", "missing")


def test_save_reference_idempotent(tmp_path: Path) -> None:
    kwargs: dict[str, object] = dict(
        prompt_id="p_idem",
        prompt_input_ids=[1],
        gen_ids=[2],
        text="x",
        q=0.1,
        features=_make_features(),
    )
    save_reference(tmp_path, "llama", "gsm8k", **kwargs)  # type: ignore[arg-type]
    save_reference(tmp_path, "llama", "gsm8k", **kwargs)  # type: ignore[arg-type]
    done = reference_done(tmp_path, "llama", "gsm8k")
    assert "p_idem" in done


# ---------------------------------------------------------------------------
# hybrid_done / append_hybrid
# ---------------------------------------------------------------------------


def test_hybrid_done_empty(tmp_path: Path) -> None:
    assert hybrid_done(tmp_path, "llama", "gsm8k", "snapkv", 0.5) == set()


def test_append_and_hybrid_done(tmp_path: Path) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[1, 2],
        text="a",
        q=0.8,
        dq=0.1,
    )
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=16,
        new_ids=[3],
        text="b",
        q=0.7,
        dq=0.05,
    )
    done = hybrid_done(tmp_path, "llama", "gsm8k", "snapkv", 0.5)
    assert ("p0", 0) in done
    assert ("p0", 16) in done


def test_hybrid_done_torn_line_tolerance(tmp_path: Path) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "knorm",
        0.25,
        prompt_id="px",
        s=0,
        new_ids=[9],
        text="ok",
        q=0.6,
        dq=0.02,
    )
    # Manually append a torn (invalid) JSON line to simulate a kill mid-write
    shard = tmp_path / "llama" / "gsm8k" / "hybrids" / "knorm__0.2500.jsonl"
    with shard.open("a") as f:
        f.write('{"prompt_id": "px", "s": 1, "new_ids": [1\n')

    done = hybrid_done(tmp_path, "llama", "gsm8k", "knorm", 0.25)
    assert ("px", 0) in done
    assert ("px", 1) not in done  # torn line ignored


def test_hybrid_different_ratios_isolated(tmp_path: Path) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.25,
        prompt_id="q0",
        s=0,
        new_ids=[],
        text="",
        q=0.0,
        dq=0.0,
    )
    done_50 = hybrid_done(tmp_path, "llama", "gsm8k", "snapkv", 0.5)
    assert ("q0", 0) not in done_50


def test_append_hybrid_with_features_marks_feature_done(
    tmp_path: Path,
) -> None:
    feats = np.ones((3, 4), dtype=np.float32)
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=16,
        new_ids=[1, 2, 3],
        text="hyb",
        q=0.25,
        dq=0.75,
        features=feats,
    )

    done = hybrid_done(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        require_features=True,
    )
    fpath = hybrid_feature_path(
        tmp_path, "llama", "gsm8k", "snapkv", 0.5, "p0", 16
    )

    assert ("p0", 16) in done
    assert fpath.exists()
    loaded = np.load(fpath)
    assert loaded.dtype == np.float16
    np.testing.assert_allclose(loaded.astype(np.float32), feats)


def test_featureless_hybrid_not_done_when_features_required(
    tmp_path: Path,
) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[1],
        text="hyb",
        q=1.0,
        dq=0.0,
    )

    assert ("p0", 0) in hybrid_done(tmp_path, "llama", "gsm8k", "snapkv", 0.5)
    assert ("p0", 0) not in hybrid_done(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        require_features=True,
    )


def test_orphan_feature_does_not_make_legacy_row_done(
    tmp_path: Path,
) -> None:
    append_hybrid(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[1],
        text="hyb",
        q=1.0,
        dq=0.0,
    )
    fpath = hybrid_feature_path(
        tmp_path, "llama", "gsm8k", "snapkv", 0.5, "p0", 0
    )
    fpath.parent.mkdir(parents=True, exist_ok=True)
    np.save(fpath, np.ones((1, 4), dtype=np.float16))

    assert ("p0", 0) not in hybrid_done(
        tmp_path,
        "llama",
        "gsm8k",
        "snapkv",
        0.5,
        require_features=True,
    )
