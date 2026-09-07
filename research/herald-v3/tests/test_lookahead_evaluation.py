"""Focused acceptance tests for the fixed lookahead evaluator."""

import copy
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate_lookahead as evaluation  # type: ignore[import-not-found]


def _row(
    prompt_id: str, action_id: str, index: int
) -> evaluation.LookaheadRow:
    ratio = evaluation.ACTION_RATIOS[action_id]
    return evaluation.LookaheadRow(
        prompt_id=prompt_id,
        action_id=action_id,
        removal_fraction=ratio,
        prompt_token_count=40.0 + index,
        decision_index32=32.0,
        pre_action_cache_size=1000.0 + index,
        reference_entropy=1.0 + index / 100.0,
        reference_top2_margin=0.2,
        action_reference_entropy_delta=ratio / 10.0,
        action_reference_margin_delta=-ratio / 10.0,
        argmax_match=action_id == "knorm:0.25",
        immediate_js=ratio / 10.0,
        lookahead_steps=2 if index % 2 else 1,
        has_delayed=int(index % 2 == 1),
        mean_delayed_js=0.01 if index % 2 else 0.0,
        reference_input_token_ids=(10, 11),
        reference_argmax_token_ids=(20, 21),
        reference_eos_position=1,
        reference_state_fingerprint="state",
        boundary_cache_fingerprint="a" * 64,
        boundary_stable={
            "prompt_token_ids": [10, 11],
            "prompt_length": 2,
            "generated_token_ids": [20, 21],
            "generated_count": 2,
            "pending_token_id": 21,
            "pending_generated_index": 1,
            "logical_position": 3,
            "attention_mask": [1, 1],
            "cache_lengths": [3],
            "cache_bytes": 1000,
            "rng_fingerprint": "rng",
            "model_tensor_count": 1,
        },
        reference_probe_hash="reference",
        probe_sha256=f"probe-{prompt_id}",
        probe_path=f"/tmp/{prompt_id}.json",
    )


def _training_data() -> tuple[
    list[evaluation.LookaheadRow], dict[str, evaluation.TrainingOutcome]
]:
    rows: list[evaluation.LookaheadRow] = []
    outcomes: dict[str, evaluation.TrainingOutcome] = {}
    for index in range(evaluation.TRAIN_PROMPT_COUNT):
        prompt_id = f"p{index:03d}"
        values: dict[str, dict[str, float]] = {}
        for action_id in evaluation.ACTION_IDS:
            rows.append(_row(prompt_id, action_id, index))
            values[action_id] = {
                "loose": (index % 11 - 5) / 10.0,
                "strict": (index % 7 - 3) / 10.0,
            }
        outcomes[prompt_id] = evaluation.TrainingOutcome(
            targets=values,
            source_hashes={},
        )
    return rows, outcomes


def _seal_for_scoring(
    predictions: dict[str, dict[str, dict[str, dict[str, float]]]],
) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": evaluation.SEAL_SCHEMA_VERSION,
        "stage": "fit_prediction_seal",
        "prediction_ids": list(predictions),
        "predictions": predictions,
        "integrity": {"test_outcomes_read": False},
    }
    body["seal_sha256"] = evaluation.canonical_sha256(body)
    return body


def test_fixed_models_fit_once_on_all_training_rows() -> None:
    train, outcomes = _training_data()
    test = [
        _row("test", action_id, 999) for action_id in evaluation.ACTION_IDS
    ]

    models, predictions = evaluation.fit_models_and_predict(
        train, test, outcomes
    )

    assert models["loose"]["B4_L"]["scaler"]["n_samples_seen"] == 240
    assert models["strict"]["B4_L"]["ridge"]["alpha"] == 1.0
    assert set(predictions["test"]) == set(evaluation.ACTION_IDS)
    for action_id in evaluation.ACTION_IDS:
        assert set(predictions["test"][action_id]) == set(
            evaluation.MODEL_NAMES
        )
        assert set(predictions["test"][action_id]["B4_L"]) == {
            "loose",
            "strict",
        }


def test_scaler_does_not_include_test_values() -> None:
    train, outcomes = _training_data()
    test = [
        _row("test", action_id, 1_000_000)
        for action_id in evaluation.ACTION_IDS
    ]
    models, _ = evaluation.fit_models_and_predict(train, test, outcomes)
    expected_mean = np.mean(
        [
            row.feature_values(evaluation.BLOCK_FEATURES["B4_L"])
            for row in train
        ],
        axis=0,
    )
    observed = np.asarray(models["loose"]["B4_L"]["scaler"]["mean"])
    np.testing.assert_allclose(observed, expected_mean)


def test_h8_pair_rejects_different_shared_reference() -> None:
    left = _row("p", evaluation.ACTION_IDS[0], 1)
    right = replace(
        _row("p", evaluation.ACTION_IDS[1], 1),
        reference_input_token_ids=(99, 11),
    )
    with pytest.raises(evaluation.EvaluationError, match="shared H8"):
        evaluation._validate_pair((left, right))


def test_portable_evidence_normalizes_flat_and_nested_serialization() -> None:
    row = _row("p", evaluation.ACTION_IDS[0], 1)
    stable = dict(row.boundary_stable)
    stable["rng_fingerprint"] = "a" * 64
    full = dict(
        stable,
        state_fingerprint="b" * 64,
        model_state_fingerprint="c" * 64,
        validation_seconds=0.1,
    )
    nested = {
        "input_ids": [stable["prompt_token_ids"]],
        "input_length": stable["prompt_length"],
        "chat_template_verified": True,
    }

    assert evaluation._portable_boundary(full, "p") == stable
    assert evaluation._portable_tokenization(nested, "p") == {
        "input_ids": stable["prompt_token_ids"],
        "input_length": stable["prompt_length"],
        "chat_template_verified": True,
    }


def test_r2_production_record_requires_and_hashes_cache_field(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = evaluation.load_prompt_manifest(
        root / "data/lookahead-v1-revision-2/train-prompts.json"
    )
    collection = root / "results/lookahead-probes-r2/train"
    record_path = collection / "ifeval_2779/record.json"
    record = evaluation.load_json(record_path)
    collection_lock = evaluation.load_json(
        collection / "collection-lock.json"
    )
    checkpoint = evaluation.load_json(collection / "checkpoint.json")

    evaluation._validate_probe_record_shape(
        record,
        "ifeval_2779",
        manifest,
        collection_lock,
        checkpoint,
    )

    missing = dict(record)
    missing.pop("boundary_cache_fingerprint")
    with pytest.raises(evaluation.EvaluationError, match="record shape"):
        evaluation._validate_probe_record_shape(
            missing,
            "ifeval_2779",
            manifest,
            collection_lock,
            checkpoint,
        )

    changed = dict(record)
    changed["boundary_cache_fingerprint"] = "changed"
    with pytest.raises(
        evaluation.EvaluationError, match="boundary cache fingerprint"
    ):
        evaluation._validate_probe_record_shape(
            changed,
            "ifeval_2779",
            manifest,
            collection_lock,
            checkpoint,
        )

    copied_collection = tmp_path / "train"
    shutil.copytree(collection, copied_collection)
    copied_record_path = copied_collection / "ifeval_2779/record.json"
    copied_record = evaluation.load_json(copied_record_path)
    copied_record["boundary_cache_fingerprint"] = "0" * 64
    copied_record_path.write_text(
        json.dumps(
            copied_record, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation.EvaluationError, match="record hash"):
        evaluation.load_probe_collection(copied_collection, manifest)


@pytest.mark.parametrize("phase", ["training", "test"])
def test_fit_collection_lock_must_match_active_protocol(
    tmp_path: Path, phase: str
) -> None:
    collection = tmp_path / phase
    collection.mkdir()
    (collection / "collection-lock.json").write_text(
        json.dumps({"protocol_lock": {"sha256": "0" * 64}}),
        encoding="utf-8",
    )

    with pytest.raises(
        evaluation.EvaluationError,
        match=f"{phase} probe collection protocol lock differs",
    ):
        evaluation._validate_probe_collection_protocol_lock(
            collection, "1" * 64, phase
        )


def test_seal_tampering_is_rejected() -> None:
    predictions = {
        "p": {
            action_id: {
                model_name: {"loose": 0.0, "strict": 0.0}
                for model_name in evaluation.MODEL_NAMES
            }
            for action_id in evaluation.ACTION_IDS
        }
    }
    seal = _seal_for_scoring(predictions)
    tampered = copy.deepcopy(seal)
    sealed_predictions = cast(dict[str, object], tampered["predictions"])
    prompt_predictions = cast(dict[str, object], sealed_predictions["p"])
    action_predictions = cast(
        dict[str, object], prompt_predictions[evaluation.ACTION_IDS[0]]
    )
    b4_predictions = cast(dict[str, object], action_predictions["B4_L"])
    b4_predictions["loose"] = 0.5
    path = Path("/tmp/lookahead-tampered-seal.json")
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError, match="seal hash"):
        evaluation.verify_prediction_seal(path)


def test_prediction_ids_define_manifest_order() -> None:
    def action_map() -> dict[str, dict[str, dict[str, float]]]:
        return {
            action_id: {
                model_name: {"loose": 0.0, "strict": 0.0}
                for model_name in evaluation.MODEL_NAMES
            }
            for action_id in evaluation.ACTION_IDS
        }

    seal = _seal_for_scoring({"p10": action_map(), "p2": action_map()})
    seal["prediction_ids"] = ["p2", "p10"]
    assert list(evaluation._sealed_predictions(seal)) == ["p2", "p10"]


def test_bootstrap_uses_shared_prompt_resamples() -> None:
    values = evaluation.paired_bootstrap(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) + 1,
        },
        replicates=100,
        seed=0,
    )
    np.testing.assert_allclose(values["b"] - values["a"], 1.0)


def test_zero_comparator_has_null_skill_and_never_beats() -> None:
    predictions = {
        "p": {
            action_id: {
                model_name: {"loose": 0.0, "strict": 0.0}
                for model_name in evaluation.MODEL_NAMES
            }
            for action_id in evaluation.ACTION_IDS
        }
    }
    seal = _seal_for_scoring(predictions)
    labels = {
        "p": {
            action_id: {"loose": 0.0, "strict": 0.0}
            for action_id in evaluation.ACTION_IDS
        }
    }
    report = evaluation.score_predictions(
        seal, labels, bootstrap_replicates=100
    )
    comparison = report["targets"]["loose"]["comparisons"]["action_mean"]
    assert comparison["relative_skill"] is None
    assert comparison["B4_L_beats"] is False
