from pathlib import Path

from herald.phase0 import (
    build_random_manifest,
    load_manifest,
)


def test_build_random_manifest_is_deterministic(tmp_path: Path):
    out = tmp_path / "phase0-random-manifest.json"
    m1 = build_random_manifest(num_prompts=20, seed=42, out_path=out)
    m2 = build_random_manifest(num_prompts=20, seed=42, out_path=out)
    assert [e.prompt_id for e in m1.entries] == [
        e.prompt_id for e in m2.entries
    ]
    assert len(m1.entries) == 20
    for e in m1.entries:
        assert e.prompt_id and e.prompt_hash
        assert len(e.prompt_hash) == 64


def test_load_manifest_round_trip(tmp_path: Path):
    out = tmp_path / "manifest.json"
    m = build_random_manifest(num_prompts=5, seed=1, out_path=out)
    loaded = load_manifest(out)
    assert [e.prompt_id for e in loaded.entries] == [
        e.prompt_id for e in m.entries
    ]
