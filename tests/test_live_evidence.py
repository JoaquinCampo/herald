from pathlib import Path

import pytest

from herald.deployment_evidence import (
    END_TO_END_RETAINED_KV_CACHE,
    candidate_id,
    create_live_run_manifest,
    directory_sha256,
    initialize_live_run,
    verify_live_run,
)


def _config() -> dict[str, object]:
    one_shot = candidate_id("expected_attention_stats", 0.25, None)
    return {
        "task": "ifeval",
        "prompt_ids": ["p0"],
        "candidate_ids": [one_shot],
        "candidate_prompt_ids": {one_shot: ["p0"]},
    }


def _manifest() -> dict[str, object]:
    return create_live_run_manifest(_config())


def _baseline(run_id: str) -> dict[str, object]:
    return {
        "prompt_id": "p0",
        "run_id": run_id,
        "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
    }


def _episode(run_id: str) -> dict[str, object]:
    return {
        "key": (f"{candidate_id('expected_attention_stats', 0.25, None)}|p0"),
        "prompt_id": "p0",
        "compressor": "expected_attention_stats",
        "ratio": 0.25,
        "sustain_interval": None,
        "candidate_id": candidate_id("expected_attention_stats", 0.25, None),
        "run_id": run_id,
        "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
    }


def test_candidate_id_separates_one_shot_and_sustained_modes() -> None:
    one_shot = candidate_id("expected_attention_stats", 0.25, None)
    sustained = candidate_id("expected_attention_stats", 0.25, 32)

    assert one_shot == "expected_attention_stats|0.25|one_shot"
    assert sustained == "expected_attention_stats|0.25|sustained_every_32"
    assert one_shot != sustained
    assert one_shot != candidate_id("expected_attention_stats", 0.25001, None)


def test_verify_live_run_accepts_one_fresh_candidate_per_prompt() -> None:
    manifest = _manifest()
    run_id = str(manifest["run_id"])

    assert (
        verify_live_run(
            manifest,
            [_baseline(run_id)],
            [_episode(run_id)],
            require_complete=True,
        )
        == []
    )


def test_verify_live_run_rejects_stale_or_mixed_mode_records() -> None:
    manifest = _manifest()
    run_id = str(manifest["run_id"])
    stale = _episode("other-run")
    mixed_mode = _episode(run_id)
    mixed_mode["sustain_interval"] = 32

    errors = verify_live_run(
        manifest,
        [_baseline(run_id)],
        [stale, mixed_mode],
    )

    assert "run_id mismatch for episode p0" in errors
    assert "candidate_id mismatch for episode p0" in errors


def test_verify_live_run_requires_complete_manifest_coverage() -> None:
    manifest = _manifest()
    run_id = str(manifest["run_id"])

    errors = verify_live_run(
        manifest,
        [_baseline(run_id)],
        [],
        require_complete=True,
    )

    assert (
        "missing episode for candidate prompt: "
        "expected_attention_stats|0.25|one_shot|p0" in errors
    )


def test_directory_sha256_changes_when_bundle_content_changes(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "alarm.json"
    artifact.write_text('{"threshold": 0.1}\n')
    before = directory_sha256(bundle)
    artifact.write_text('{"threshold": 0.2}\n')

    assert directory_sha256(bundle) != before


def test_initialize_live_run_rejects_bundle_digest_drift_on_resume(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    artifact = bundle / "alarm.json"
    artifact.write_text('{"threshold": 0.1}\n')
    run_dir = tmp_path / "live"
    config = {**_config(), "bundle_sha256": directory_sha256(bundle)}
    initialize_live_run(run_dir, config, resume=False)
    artifact.write_text('{"threshold": 0.2}\n')
    changed_config = {
        **_config(),
        "bundle_sha256": directory_sha256(bundle),
    }

    with pytest.raises(ValueError, match="configuration does not match"):
        initialize_live_run(run_dir, changed_config, resume=True)


def test_initialize_live_run_rejects_reuse_without_explicit_resume(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "live"
    config = _config()

    manifest = initialize_live_run(run_dir, config, resume=False)

    assert (run_dir / "run_manifest.json").is_file()
    assert manifest == create_live_run_manifest(config)
    with pytest.raises(ValueError, match="fresh empty output directory"):
        initialize_live_run(run_dir, config, resume=False)
    assert initialize_live_run(run_dir, config, resume=True) == manifest


def test_verify_live_run_rejects_noncanonical_episode_keys() -> None:
    manifest = _manifest()
    run_id = str(manifest["run_id"])
    episode = _episode(run_id)
    episode["key"] = "stale-key"

    errors = verify_live_run(
        manifest,
        [_baseline(run_id)],
        [episode],
    )

    assert "episode key mismatch for p0" in errors


def test_verify_live_run_rejects_duplicate_prompt_evidence() -> None:
    manifest = _manifest()
    run_id = str(manifest["run_id"])

    errors = verify_live_run(
        manifest,
        [_baseline(run_id), _baseline(run_id)],
        [_episode(run_id), _episode(run_id)],
    )

    assert "duplicate baseline prompt_id: p0" in errors
    assert (
        "duplicate episode key: expected_attention_stats|0.25|one_shot|p0"
        in errors
    )
