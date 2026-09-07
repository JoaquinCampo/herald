"""Tests for the outcome-free, resumable lookahead collector."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from tests.test_engine import _tiny_model
from tests.test_runner import _manifest, _Tokenizer

from herald_v3.engineering import lookahead_collection
from herald_v3.engineering.lookahead_collection import (
    ACTION_RATIOS,
    MAX_STEPS,
    SCHEMA_VERSION,
    LookaheadCollectionError,
    collect_lookahead,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = _manifest()
    manifest_path = tmp_path / "prompts.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": "fixture.lookahead_lock.v1",
        "manifests": {
            "test": {
                "prompt_ids": [manifest.prompts[0].prompt_id],
                "manifest_sha256": manifest.fingerprint,
            }
        },
        "configuration": {
            "seed": 0,
            "decision_tokens": 32,
            "max_lookahead_steps": MAX_STEPS,
            "actions": [
                {"name": "knorm", "removal_fraction": ratio}
                for ratio in ACTION_RATIOS
            ],
        },
    }
    lock_path = tmp_path / "protocol-lock.json"
    lock_path.write_text(
        json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path, lock_path


def test_collects_ordered_pair_without_outcomes_and_resumes(
    tmp_path: Path,
) -> None:
    manifest_path, lock_path = _write_inputs(tmp_path)
    output = tmp_path / "collection"
    summary = collect_lookahead(
        _tiny_model(),
        _Tokenizer(),
        _manifest(),
        manifest_path,
        lock_path,
        output,
        phase="test",
        expected_prompt_count=1,
    )

    assert summary["status"] == "completed"
    assert summary["action_row_count"] == 2
    assert summary["ledger"][0]["source_hash"] == (
        _manifest().prompts[0].prompt_text_sha256
    )
    record_path = output / "ifeval_100/record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "eligible"
    assert len(record["action_rows"]) == 2
    assert record["actions"].keys() == {
        "knorm:0.25",
        "knorm:0.5",
    }
    assert record["checks"]["outcomes_collected"] is False
    assert "outcomes" not in record
    assert "logits" not in json.dumps(record)

    resumed = collect_lookahead(
        _tiny_model(),
        _Tokenizer(),
        _manifest(),
        manifest_path,
        lock_path,
        output,
        phase="test",
        expected_prompt_count=1,
    )
    assert resumed["status"] == "completed"
    assert resumed["processed"] == 1


def test_protocol_lock_distinguishes_raw_file_and_manifest_hash(
    tmp_path: Path,
) -> None:
    manifest_path, lock_path = _write_inputs(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    lock["manifests"]["test"]["file_sha256"] = raw_hash
    lock_path.write_text(
        json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    summary = collect_lookahead(
        _tiny_model(),
        _Tokenizer(),
        _manifest(),
        manifest_path,
        lock_path,
        tmp_path / "collection",
        phase="test",
        expected_prompt_count=1,
    )
    assert summary["schema_version"] == SCHEMA_VERSION


def test_existing_unlocked_output_fails_closed(tmp_path: Path) -> None:
    manifest_path, lock_path = _write_inputs(tmp_path)
    output = tmp_path / "collection"
    output.mkdir()
    (output / "unexpected.json").write_text("{}", encoding="utf-8")

    try:
        collect_lookahead(
            _tiny_model(),
            _Tokenizer(),
            _manifest(),
            manifest_path,
            lock_path,
            output,
            phase="test",
            expected_prompt_count=1,
        )
    except LookaheadCollectionError as error:
        assert "no lock" in str(error)
    else:
        raise AssertionError("unlocked output unexpectedly accepted")


def test_collector_checks_generation_versions_only() -> None:
    generation_package = "torch"
    generation_version = lookahead_collection.importlib.metadata.version(
        generation_package
    )
    lookahead_collection._validate_software(
        {
            "generation": {generation_package: generation_version},
            "evaluation": {
                "numpy": "mac-host-version",
                "scikit-learn": "mac-host-version",
            },
        }
    )

    try:
        lookahead_collection._validate_software(
            {
                "generation": {generation_package: "wrong-generation"},
                "evaluation": {"numpy": "mac-host-version"},
            }
        )
    except LookaheadCollectionError as error:
        assert "software version mismatch" in str(error)
    else:
        raise AssertionError("generation version mismatch was accepted")


def test_resume_rejects_ledger_source_hash_drift(tmp_path: Path) -> None:
    manifest_path, lock_path = _write_inputs(tmp_path)
    output = tmp_path / "collection"
    collect_lookahead(
        _tiny_model(),
        _Tokenizer(),
        _manifest(),
        manifest_path,
        lock_path,
        output,
        phase="test",
        expected_prompt_count=1,
    )
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["ledger"][0]["source_hash"] = "drifted"
    checkpoint_path.write_text(
        json.dumps(checkpoint, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        LookaheadCollectionError, match="source hash mismatch"
    ):
        collect_lookahead(
            _tiny_model(),
            _Tokenizer(),
            _manifest(),
            manifest_path,
            lock_path,
            output,
            phase="test",
            expected_prompt_count=1,
        )


def test_reuse_boundary_identity_is_portable_but_cache_content_strict(
    tmp_path: Path,
) -> None:
    model = _tiny_model()
    tokenizer = _Tokenizer()
    manifest = _manifest()
    prompt = manifest.prompts[0]
    input_ids = lookahead_collection.runner.tokenize_prompt(
        tokenizer, prompt
    )
    boundary = lookahead_collection.engine.build_boundary(
        model,
        input_ids,
        decision_tokens=lookahead_collection.DECISION_TOKENS,
        eos_ids=frozenset({tokenizer.eos_token_id}),
    )

    artifact_directory = tmp_path / "artifacts"
    artifact_directory.mkdir()
    artifacts_index = artifact_directory / "artifacts.json"
    run_file = artifact_directory / "run.json"
    artifacts_index.write_text("{}\n", encoding="utf-8")
    run_file.write_text("{}\n", encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source_file = Path(lookahead_collection.engine.__file__).resolve()
    entry = {
        "prompt_id": prompt.prompt_id,
        "prompt": prompt.to_dict(),
        "boundary": {
            **boundary.to_dict(),
            "validation_seconds": 9.0,
            "model_state_fingerprint": "producer-process",
            "state_fingerprint": "producer-process",
        },
        "boundary_cache_fingerprint": (
            lookahead_collection.engine.cache_fingerprint(boundary.cache)
        ),
        "tokenization": {
            "input_ids": boundary.prompt_ids[0].tolist(),
        },
        "artifact_directory": str(artifact_directory),
        "artifacts_sha256": digest(artifacts_index),
        "files_sha256": {
            "artifacts.json": digest(artifacts_index),
            "run.json": digest(run_file),
        },
        "configuration": lookahead_collection._configuration(
            frozenset({tokenizer.eos_token_id})
        ),
        "model": lookahead_collection.engine.model_signature(model),
        "environment": {
            "tokenizer_name_or_path": tokenizer.name_or_path,
        },
        "source_manifest": {
            "sha256": {source_file.name: digest(source_file)},
        },
    }
    reuse = {"rows": [entry]}

    lookahead_collection._validate_reuse_row(
        reuse,
        prompt,
        manifest,
        boundary,
        model_signature=lookahead_collection.engine.model_signature(model),
        tokenizer_name=tokenizer.name_or_path,
        eos_ids=frozenset({tokenizer.eos_token_id}),
    )

    changed_cache = deepcopy(reuse)
    changed_cache["rows"][0]["boundary_cache_fingerprint"] = "0" * 64
    with pytest.raises(
        LookaheadCollectionError, match="cache fingerprint mismatch"
    ):
        lookahead_collection._validate_reuse_row(
            changed_cache,
            prompt,
            manifest,
            boundary,
            model_signature=lookahead_collection.engine.model_signature(model),
            tokenizer_name=tokenizer.name_or_path,
            eos_ids=frozenset({tokenizer.eos_token_id}),
        )

    changed_tokens = deepcopy(reuse)
    changed_tokens["rows"][0]["tokenization"]["input_ids"] = [1, 2, 3]
    with pytest.raises(
        LookaheadCollectionError, match="tokenization mismatch"
    ):
        lookahead_collection._validate_reuse_row(
            changed_tokens,
            prompt,
            manifest,
            boundary,
            model_signature=lookahead_collection.engine.model_signature(model),
            tokenizer_name=tokenizer.name_or_path,
            eos_ids=frozenset({tokenizer.eos_token_id}),
        )

    changed_rng = deepcopy(reuse)
    changed_rng["rows"][0]["boundary"]["rng_fingerprint"] = "f" * 64
    with pytest.raises(LookaheadCollectionError, match="boundary mismatch"):
        lookahead_collection._validate_reuse_row(
            changed_rng,
            prompt,
            manifest,
            boundary,
            model_signature=lookahead_collection.engine.model_signature(model),
            tokenizer_name=tokenizer.name_or_path,
            eos_ids=frozenset({tokenizer.eos_token_id}),
        )


def test_cli_rejects_lock_digest_mismatch_before_model_load(
    tmp_path: Path,
) -> None:
    manifest_path, lock_path = _write_inputs(tmp_path)
    from scripts import collect_lookahead as cli

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--model",
                str(tmp_path / "missing-model"),
                "--prompts",
                str(manifest_path),
                "--protocol-lock",
                str(lock_path),
                "--expected-lock-sha256",
                "0" * 64,
                "--output",
                str(tmp_path / "collection"),
                "--phase",
                "test",
            ]
        )
    assert error.value.code == 2
