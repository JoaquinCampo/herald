"""Focused tests for sealed lookahead outcome collection."""

import copy
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from scripts import collect_lookahead_outcomes as outcomes
from tests.test_runner import _manifest

from herald_v3.engineering import runner as engineering_runner
from herald_v3.engineering.prompts import EngineeringPrompt, PromptManifest


def _lock() -> dict[str, object]:
    return {
        "configuration": {"max_new_tokens": 1024, "seed": 17},
        "software": {"generation": {"torch": "2.10.0"}},
        "scorer_provenance": {
            "files": {"scorer.py": {"sha256": "a" * 64, "bytes": 1}}
        },
        "source_manifest": {"src/herald_v3/engineering/runner.py": "b" * 64},
        "model": {"checkpoint_fingerprint": "c" * 64},
    }


def _collection_context(
    manifest: PromptManifest,
    prompts: list[EngineeringPrompt],
    output: Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    lock = _lock()
    row_values = rows or []
    return {
        "output_path": output,
        "existing_index": None,
        "existing_rows": row_values,
        "lock": lock,
        "seal": {"runtime": {"python": "3.12", "packages": {}}},
        "lock_file_sha256": "d" * 64,
        "seal_file_sha256": "e" * 64,
        "seal_sha256": "f" * 64,
        "test_manifest_sha256": "1" * 64,
        "test_manifest": manifest,
        "eligible_prompts": prompts,
        "h8_by_prompt": {prompt.prompt_id: {} for prompt in prompts},
    }


def _row(prompt_id: str) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "artifact_directory": f"raw-runs/{prompt_id}",
        "artifacts_sha256": "2" * 64,
        "run_sha256": "3" * 64,
        "actions": {
            action_id: {
                "q_reference": {"loose": 0.8, "strict": 0.6},
                "q_action": {"loose": 0.7, "strict": 0.5},
                "d": {"loose": 0.1, "strict": 0.1},
            }
            for action_id in outcomes.ACTION_IDS
        },
    }


def _stable_boundary() -> dict[str, object]:
    return {
        "prompt_token_ids": [1, 2],
        "prompt_length": 2,
        "generated_token_ids": list(range(32)),
        "generated_count": 32,
        "pending_token_id": 31,
        "pending_generated_index": 31,
        "logical_position": 33,
        "attention_mask": [1, 1],
        "cache_lengths": [33],
        "cache_bytes": 1024,
        "rng_fingerprint": "a" * 64,
        "model_tensor_count": 1,
    }


def _boundary_run_fixture(
    *,
    cache_fingerprint: str = "c" * 64,
    pending_token_id: int = 31,
    state_fingerprint: str = "d" * 64,
    model_state_fingerprint: str = "e" * 64,
) -> tuple[
    dict[str, object], dict[str, object], dict[str, dict[str, object]]
]:
    stable = _stable_boundary()
    boundary = dict(stable)
    boundary.update(
        {
            "state_fingerprint": state_fingerprint,
            "model_state_fingerprint": model_state_fingerprint,
            "validation_seconds": 0.001,
            "pending_token_id": pending_token_id,
        }
    )
    generated = list(range(32))
    h8_actions: dict[str, dict[str, object]] = {}
    for action_id in outcomes.ACTION_IDS:
        h8_actions[action_id] = {
            "input_token_ids": [31],
            "reference_argmax_token_ids": [32],
            "realized_steps": 1,
            "has_delayed": 0,
            "reference_eos_position": None,
            "reference_state_fingerprint": "b" * 64,
            "boundary_cache_fingerprint": cache_fingerprint,
            "boundary_stable": stable,
            "reference_probe_hash": "f" * 64,
            "probe_sha256": "1" * 64,
            "probe_path": "/tmp/probe.json",
        }
    action_arms = [
        {
            "action": {
                "name": "knorm",
                "removal_fraction": ratio,
                "action_id": action_id,
            },
            "compression": {"before_fingerprint": cache_fingerprint},
        }
        for action_id, ratio in (
            ("knorm:0.25", 0.25),
            ("knorm:0.5", 0.5),
        )
    ]
    result = {
        "tokenization": {"input_ids": [1, 2]},
        "outputs": {
            "uninterrupted": {"token_ids": generated + [32, 99]},
            "actions": {
                action_id: {
                    "action": {
                        "name": "knorm",
                        "removal_fraction": ratio,
                        "action_id": action_id,
                    },
                    "token_ids": generated + [98],
                }
                for action_id, ratio in (
                    ("knorm:0.25", 0.25),
                    ("knorm:0.5", 0.5),
                )
            },
        },
    }
    acceptance = {
        "eligibility": {
            "eligible": True,
            "requested_decision_tokens": 32,
        },
        "boundary": boundary,
        "action_arms": action_arms,
    }
    return result, acceptance, h8_actions


def _score(loose_count: int, strict_count: int) -> dict[str, object]:
    instruction_count = 10
    return {
        "instruction_count": instruction_count,
        "loose": loose_count / instruction_count,
        "loose_pass": [True] * loose_count
        + [False] * (instruction_count - loose_count),
        "strict": strict_count / instruction_count,
        "strict_pass": [True] * strict_count
        + [False] * (instruction_count - strict_count),
    }


def _score_pair(
    reference: dict[str, object], action: dict[str, object]
) -> dict[str, object]:
    return {
        "reference": reference,
        "action": action,
        "d_loose": cast(float, reference["loose"])
        - cast(float, action["loose"]),
        "d_strict": cast(float, reference["strict"])
        - cast(float, action["strict"]),
    }


def _score_document() -> dict[str, object]:
    reference = _score(8, 6)
    action = _score(7, 5)
    pair = _score_pair(reference, action)
    noop = _score_pair(reference, reference)
    return {
        "scores": {
            "reference": reference,
            "noop_forks": [noop, noop],
            "actions": {action_id: pair for action_id in outcomes.ACTION_IDS},
        }
    }


def _runner_environment_fixture() -> tuple[
    dict[str, object], dict[str, object]
]:
    resources = {
        "modules": {
            "absl": "2.4.0",
            "immutabledict": "4.3.1",
            "langdetect": None,
            "nltk": "3.9.4",
        },
        "nltk_resources": ["tokenizers/punkt", "tokenizers/punkt_tab"],
    }
    lock = _lock()
    lock["tokenizer"] = {"name_or_path": "fixture-tokenizer"}
    lock["generation_environment"] = {"resources": resources}
    run: dict[str, object] = {
        "environment": {
            "python": "3.12.11",
            "platform": "linux",
            "tokenizer_name_or_path": "fixture-tokenizer",
            "packages": {"torch": "2.10.0"},
            "resources": copy.deepcopy(resources),
        }
    }
    return run, lock


def test_derive_scores_returns_signed_quality_differences() -> None:
    result = _score_document()

    derived = outcomes._derive_scores(result, "prompt")

    assert derived["knorm:0.25"]["d"] == {
        "loose": pytest.approx(0.1),
        "strict": pytest.approx(0.1),
    }

    tampered = copy.deepcopy(result)
    scores = cast(dict[str, object], tampered["scores"])
    reference = cast(dict[str, object], scores["reference"])
    reference["loose"] = 1.1
    with pytest.raises(
        outcomes.OutcomeCollectionError, match="between 0 and 1"
    ):
        outcomes._derive_scores(tampered, "prompt")


def test_score_schema_validates_vectors_counts_and_signed_pairs() -> None:
    valid = _score_document()

    bad_vector = copy.deepcopy(valid)
    bad_scores = cast(dict[str, object], bad_vector["scores"])
    bad_reference = cast(dict[str, object], bad_scores["reference"])
    bad_reference["loose_pass"] = [True]
    with pytest.raises(
        outcomes.OutcomeCollectionError, match="pass vector is malformed"
    ):
        outcomes._derive_scores(bad_vector, "prompt")

    bad_count = copy.deepcopy(valid)
    count_scores = cast(dict[str, object], bad_count["scores"])
    count_reference = cast(dict[str, object], count_scores["reference"])
    count_reference["instruction_count"] = 0
    with pytest.raises(
        outcomes.OutcomeCollectionError, match="must be positive"
    ):
        outcomes._derive_scores(bad_count, "prompt")

    bad_difference = copy.deepcopy(valid)
    difference_scores = cast(dict[str, object], bad_difference["scores"])
    difference_actions = cast(dict[str, object], difference_scores["actions"])
    difference_pair = cast(
        dict[str, object], difference_actions[outcomes.ACTION_IDS[0]]
    )
    difference_pair["d_loose"] = 0.0
    with pytest.raises(
        outcomes.OutcomeCollectionError,
        match="d_loose differs from scores",
    ):
        outcomes._derive_scores(bad_difference, "prompt")


def test_runner_environment_requires_locked_resources_and_packages() -> None:
    run, lock = _runner_environment_fixture()
    outcomes._validate_runner_environment(run, lock, "prompt")

    bad_resources = copy.deepcopy(run)
    environment = cast(dict[str, object], bad_resources["environment"])
    resources = cast(dict[str, object], environment["resources"])
    modules = cast(dict[str, object], resources["modules"])
    modules["nltk"] = "wrong"
    with pytest.raises(
        outcomes.OutcomeCollectionError,
        match="raw scorer resource set differs",
    ):
        outcomes._validate_runner_environment(bad_resources, lock, "prompt")

    bad_package = copy.deepcopy(run)
    package_environment = cast(dict[str, object], bad_package["environment"])
    packages = cast(dict[str, object], package_environment["packages"])
    packages["torch"] = "wrong"
    with pytest.raises(
        outcomes.OutcomeCollectionError,
        match="raw generation dependencies differ",
    ):
        outcomes._validate_runner_environment(bad_package, lock, "prompt")


def test_boundary_validation_accepts_portable_cross_process_identity() -> (
    None
):
    prompt = _manifest().prompts[0]
    result, acceptance, h8_actions = _boundary_run_fixture()

    # The process-local state and model pointers can differ when identical
    # weights are loaded in another process, while the boundary cache bytes
    # and portable boundary evidence remain the same.
    outcomes._validate_boundary_and_tokens(
        result, acceptance, prompt, h8_actions
    )

    tampered_cache = copy.deepcopy(acceptance)
    tampered_arms = cast(
        list[dict[str, object]], tampered_cache["action_arms"]
    )
    tampered_compression = cast(
        dict[str, object], tampered_arms[0]["compression"]
    )
    tampered_compression["before_fingerprint"] = "9" * 64
    with pytest.raises(
        outcomes.OutcomeCollectionError,
        match="boundary cache fingerprint differs",
    ):
        outcomes._validate_boundary_and_tokens(
            result, tampered_cache, prompt, h8_actions
        )

    tampered_token = copy.deepcopy(acceptance)
    tampered_boundary = cast(dict[str, object], tampered_token["boundary"])
    tampered_boundary["pending_token_id"] = 30
    with pytest.raises(
        outcomes.OutcomeCollectionError, match="raw pending token differs"
    ):
        outcomes._validate_boundary_and_tokens(
            result, tampered_token, prompt, h8_actions
        )


def test_generation_versions_are_checked_against_the_installed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {"torch": "2.10.0", "transformers": "4.57.6"}
    monkeypatch.setattr(
        importlib.metadata, "version", lambda package: versions[package]
    )

    software = {"generation": dict(versions), "evaluation": {}}
    outcomes._validate_generation_software(software)

    generation = software["generation"]
    assert isinstance(generation, dict)
    generation["torch"] = "wrong"
    with pytest.raises(
        outcomes.OutcomeCollectionError,
        match="generation package version differs: torch",
    ):
        outcomes._validate_generation_software(software)


def test_collection_runs_one_manifest_prompt_at_a_time_and_writes_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base_manifest = _manifest()
    first = base_manifest.prompts[0]
    second = replace(first, prompt_id="ifeval_101")
    prompts = [first, second]
    context = _collection_context(base_manifest, prompts, tmp_path / "out")
    monkeypatch.setattr(outcomes, "_preflight", lambda **kwargs: context)
    monkeypatch.setattr(
        engineering_runner,
        "load_offline_model",
        lambda path: ("model", "tokenizer"),
    )
    monkeypatch.setattr(
        outcomes, "_validate_loaded_model", lambda *args: None
    )
    calls: list[tuple[object, Path, dict[str, object]]] = []

    def fake_run(
        model: object,
        tokenizer: object,
        manifest: object,
        output: str | Path,
        **kwargs: object,
    ) -> None:
        del model, tokenizer
        calls.append((manifest, Path(output), dict(kwargs)))
        Path(output).mkdir(parents=True)

    monkeypatch.setattr(engineering_runner, "run_engineering", fake_run)
    monkeypatch.setattr(
        outcomes,
        "_validate_raw_run",
        lambda raw_root, prompt, h8_actions, lock: _row(prompt.prompt_id),
    )

    result = outcomes.collect_lookahead_outcomes(
        "lock",
        "a" * 64,
        "seal",
        "b" * 64,
        "manifest",
        "model",
        tmp_path / "out",
    )

    assert result["status"] == "completed"
    assert len(calls) == 2
    manifests = [cast(PromptManifest, call[0]) for call in calls]
    assert [manifest.prompts[0].prompt_id for manifest in manifests] == [
        first.prompt_id,
        second.prompt_id,
    ]
    assert all(
        call[2] == {"max_new_tokens": 1024, "ratios": (0.25, 0.5), "seed": 17}
        for call in calls
    )
    document = json.loads(
        (tmp_path / "out" / "outcome-index.json").read_text(encoding="utf-8")
    )
    assert document["status"] == "completed"
    assert len(document["ordered_rows"]) == 2


def test_preflight_rejection_happens_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def reject(**kwargs: object) -> dict[str, object]:
        del kwargs
        raise outcomes.OutcomeCollectionError("seal rejected")

    monkeypatch.setattr(outcomes, "_preflight", reject)
    monkeypatch.setattr(
        engineering_runner,
        "load_offline_model",
        lambda path: pytest.fail("model loaded before preflight"),
    )

    with pytest.raises(
        outcomes.OutcomeCollectionError, match="seal rejected"
    ):
        outcomes.collect_lookahead_outcomes(
            "lock",
            "a" * 64,
            "seal",
            "b" * 64,
            "manifest",
            "model",
            tmp_path / "out",
        )


def test_completed_index_cannot_cover_only_a_prefix(tmp_path: Path) -> None:
    manifest = _manifest()
    second = replace(manifest.prompts[0], prompt_id="ifeval_101")
    eligible = [manifest.prompts[0], second]
    lock = _lock()
    context = {
        "lock": lock,
        "seal": {"runtime": {"python": "3.12", "packages": {}}},
        "lock_file_sha256": "d" * 64,
        "seal_file_sha256": "e" * 64,
        "seal_sha256": "f" * 64,
        "test_manifest_sha256": "1" * 64,
    }
    index = outcomes._index_document(
        context, [_row("ifeval_100")], status="completed"
    )
    output = tmp_path / "output"
    (output / "raw-runs" / "ifeval_100").mkdir(parents=True)
    (output / "outcome-index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    with pytest.raises(outcomes.OutcomeCollectionError, match="cover"):
        outcomes._preflight_output(
            output,
            "d" * 64,
            "e" * 64,
            "f" * 64,
            "1" * 64,
            {"python": "3.12", "packages": {}},
            manifest,
            eligible,
            {"ifeval_100": {}, "ifeval_101": {}},
            lock,
        )


def test_running_index_validates_and_resumes_a_verified_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    second = replace(manifest.prompts[0], prompt_id="ifeval_101")
    eligible = [manifest.prompts[0], second]
    first_row = _row(manifest.prompts[0].prompt_id)
    lock = _lock()
    context = {
        "lock": lock,
        "seal": {"runtime": {"python": "3.12", "packages": {}}},
        "lock_file_sha256": "d" * 64,
        "seal_file_sha256": "e" * 64,
        "seal_sha256": "f" * 64,
        "test_manifest_sha256": "1" * 64,
    }
    index = outcomes._index_document(context, [first_row], status="running")
    output = tmp_path / "output"
    (output / "raw-runs" / manifest.prompts[0].prompt_id).mkdir(parents=True)
    (output / "outcome-index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    monkeypatch.setattr(
        outcomes,
        "_validate_raw_run",
        lambda raw_root, prompt, h8_actions, lock: _row(prompt.prompt_id),
    )

    loaded_index, rows = outcomes._preflight_output(
        output,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "1" * 64,
        {"python": "3.12", "packages": {}},
        manifest,
        eligible,
        {prompt.prompt_id: {} for prompt in eligible},
        lock,
    )

    assert loaded_index is not None
    assert rows == [first_row]


def test_running_index_rejects_reordered_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest = _manifest()
    second = replace(manifest.prompts[0], prompt_id="ifeval_101")
    eligible = [manifest.prompts[0], second]
    reversed_rows = [
        _row(second.prompt_id),
        _row(manifest.prompts[0].prompt_id),
    ]
    lock = _lock()
    context = {
        "lock": lock,
        "seal": {"runtime": {"python": "3.12", "packages": {}}},
        "lock_file_sha256": "d" * 64,
        "seal_file_sha256": "e" * 64,
        "seal_sha256": "f" * 64,
        "test_manifest_sha256": "1" * 64,
    }
    index = outcomes._index_document(context, reversed_rows, status="running")
    output = tmp_path / "output"
    (output / "raw-runs" / manifest.prompts[0].prompt_id).mkdir(parents=True)
    (output / "raw-runs" / second.prompt_id).mkdir()
    (output / "outcome-index.json").write_text(
        json.dumps(index), encoding="utf-8"
    )
    monkeypatch.setattr(
        outcomes,
        "_validate_raw_run",
        lambda raw_root, prompt, h8_actions, lock: _row(prompt.prompt_id),
    )

    with pytest.raises(
        outcomes.OutcomeCollectionError, match="rows are reordered"
    ):
        outcomes._preflight_output(
            output,
            "d" * 64,
            "e" * 64,
            "f" * 64,
            "1" * 64,
            {"python": "3.12", "packages": {}},
            manifest,
            eligible,
            {prompt.prompt_id: {} for prompt in eligible},
            lock,
        )
