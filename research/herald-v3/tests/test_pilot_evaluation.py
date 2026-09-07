"""Focused tests for the fixed offline pilot evaluator."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import evaluate_pilot as pilot  # noqa: E402


def _manifest(prompt_count: int = 10) -> dict[str, object]:
    prompts = [
        {"prompt_id": f"ifeval_{index}", "fold": index % 5}
        for index in range(prompt_count)
    ]
    return {
        "manifest_sha256": "fixture-manifest-fingerprint",
        "schema_version": "herald_v3.pilot_manifest.v1",
        "prompts": prompts,
    }


def _row_document(
    prompt_id: str,
    fold: int,
    *,
    d_loose: tuple[float, float] = (0.25, -0.125),
    d_strict: tuple[float, float] = (0.125, -0.25),
) -> dict[str, object]:
    ratios = (0.25, 0.5)
    reference_loose = 0.75
    reference_strict = 0.5
    arms: list[dict[str, object]] = []
    action_scores: dict[str, object] = {}
    for index, (ratio, loose, strict) in enumerate(
        zip(ratios, d_loose, d_strict, strict=True)
    ):
        action_id = f"knorm:{ratio:.6g}"
        action_loose = reference_loose - loose
        action_strict = reference_strict - strict
        action_scores[action_id] = {
            "action": {
                "loose": action_loose,
                "strict": action_strict,
            },
            "d_loose": loose,
            "d_strict": strict,
        }
        arms.append(
            {
                "action": {
                    "name": "knorm",
                    "removal_fraction": ratio,
                    "action_id": action_id,
                },
                "probe_enabled": True,
                "probe": {
                    "action": {"action_id": action_id},
                    "reference_entropy": 1.0,
                    "action_entropy": 1.2 + index,
                    "reference_top2_margin": 0.4,
                    "action_top2_margin": 0.3 - index * 0.01,
                    "argmax_match": index == 0,
                    "js_divergence": 0.01 * (index + 1),
                    "finite": True,
                },
            }
        )
    return {
        "schema_version": 1,
        "configuration": {
            "decision_tokens": 32,
            "actions": [
                {"name": "knorm", "removal_fraction": ratio}
                for ratio in ratios
            ],
        },
        "results": [
            {
                "status": "accepted",
                "prompt": {
                    "prompt_id": prompt_id,
                    "fold": fold,
                    "fold_hash": f"fold-{fold}",
                },
                "tokenization": {"input_length": 20},
                "acceptance": {
                    "passed": True,
                    "boundary": {
                        "cache_bytes": 1000 + fold,
                        "generated_count": 32,
                    },
                    "action_arms": arms,
                },
                "scores": {
                    "reference": {
                        "loose": reference_loose,
                        "strict": reference_strict,
                    },
                    "actions": action_scores,
                },
            }
        ],
    }


def _write_collection(
    tmp_path: Path,
    prompt_count: int = 120,
    *,
    early_before_target: bool = True,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "collection"
    root.mkdir()
    manifest_path = tmp_path / "manifest.json"
    manifest = _manifest(160)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source_manifest = {"root": "fixture-source", "sha256": "source-hash"}
    environment = {"runtime": "fixture"}
    model = {
        "name_or_path": "/models/Qwen2.5-7B-Instruct/" + pilot.MODEL_SNAPSHOT,
        "checkpoint_revision": pilot.MODEL_SNAPSHOT,
    }
    configuration = {
        "target_eligible": 120,
        "max_new_tokens": 1024,
        "decision_tokens": 32,
        "actions": [
            {"name": "knorm", "removal_fraction": ratio}
            for ratio in (0.25, 0.5)
        ],
        "seed": 0,
        "eos_ids": [151643, 151645],
    }
    run_configuration = {**configuration, "decode_skip_special_tokens": True}
    design_lock = json.loads(
        (Path(__file__).parents[1] / "data/pilot-v1/lock.json").read_text()
    )
    design_lock["pilot_roster"]["prompt_manifest"] = str(
        manifest_path.resolve()
    )
    design_lock["pilot_roster"]["prompt_manifest_sha256"] = pilot.file_sha256(
        manifest_path
    )
    design_lock["pilot_roster"]["manifest_fingerprint"] = manifest[
        "manifest_sha256"
    ]
    design_lock_path = tmp_path / "design-lock.json"
    design_lock_path.write_text(json.dumps(design_lock), encoding="utf-8")
    summary = {
        "schema_version": "herald_v3.pilot_checkpoint.v1",
        "status": "completed",
        "passed": True,
        "target_eligible": 120,
        "accepted_eligible": 120,
        "ineligible_early_eos": 40,
        "processed": 160,
        "roster_count": 160,
        "ledger": [],
    }
    lock = {
        "schema_version": "herald_v3.pilot_lock.v1",
        "model_argument": (
            "/models/Qwen2.5-7B-Instruct/" + pilot.MODEL_SNAPSHOT
        ),
        "prompt_manifest": {
            "path": str(manifest_path.resolve()),
            "file_sha256": pilot.file_sha256(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "prompt_ids": [
                entry["prompt_id"] for entry in manifest["prompts"]
            ],
        },
        "configuration": configuration,
        "model": model,
        "source_manifest": source_manifest,
        "environment": environment,
        "wrapper": {"path": "fixture-wrapper", "sha256": "wrapper-hash"},
    }
    lock_path = root / "experiment-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    lock_sha = pilot.file_sha256(lock_path)
    for index in range(160):
        accepted = index >= 40 if early_before_target else index < 120
        document = _row_document(f"ifeval_{index}", index % 5)
        document.update(
            {
                "configuration": run_configuration,
                "model": model,
                "source_manifest": source_manifest,
                "environment": environment,
                "status": (
                    "completed" if accepted else "completed_with_failures"
                ),
                "passed": accepted,
                "stopped_on_failure": False,
            }
        )
        result = document["results"][0]
        assert isinstance(result, dict)
        if not accepted:
            result["status"] = "ineligible"
            result["acceptance"] = {
                "eligibility": {
                    "eligible": False,
                    "reason": "eos_at_or_before_pending_boundary",
                }
            }
            result.pop("scores", None)
            result.pop("outputs", None)
        run_dir = root / f"ifeval_{index}"
        run_dir.mkdir()
        (run_dir / "prompt-manifest.json").write_text(
            json.dumps({"prompt_id": f"ifeval_{index}"}), encoding="utf-8"
        )
        (run_dir / "run.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
        (run_dir / "run.log").write_text("fixture\n", encoding="utf-8")
        artifacts = {
            "schema_version": 1,
            "sha256": {
                name: pilot.file_sha256(run_dir / name)
                for name in pilot.REQUIRED_ARTIFACTS
            },
        }
        (run_dir / "artifacts.json").write_text(
            json.dumps(artifacts), encoding="utf-8"
        )
        summary["ledger"].append(
            {
                "prompt_id": f"ifeval_{index}",
                "status": "accepted" if accepted else "ineligible_early_eos",
                **(
                    {
                        "reason": "eos_at_or_before_pending_boundary",
                        "generated_token_ids": [],
                    }
                    if not accepted
                    else {}
                ),
                "artifact_directory": f"/remote/fixture/ifeval_{index}",
                "artifacts_sha256": pilot.file_sha256(
                    run_dir / "artifacts.json"
                ),
            }
        )
    summary["early_eos_ledger"] = [
        entry
        for entry in summary["ledger"]
        if entry["status"] == "ineligible_early_eos"
    ]
    summary["experiment_lock_sha256"] = lock_sha
    (root / "checkpoint.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return root, manifest_path, design_lock_path


def test_load_rows_retains_signed_targets_and_two_actions(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)

    loaded = pilot.load_collection(
        root,
        manifest_path,
        design_lock_path=design_lock_path,
        minimum_eligible=0,
    )
    assert len(loaded.rows) == 240
    assert loaded.rows[0].target_loose == 0.25
    assert loaded.rows[1].target_loose == -0.125
    assert loaded.rows[0].target_strict == 0.125
    assert loaded.rows[0].action_reference_entropy_delta == pytest.approx(0.2)
    assert loaded.rows[1].action_reference_entropy_delta == pytest.approx(1.2)


def test_oof_predictions_do_not_use_test_fold_targets() -> None:
    rows = [
        pilot.PilotRow(
            prompt_id=f"p{index}",
            fold=index % 5,
            action_id="knorm:0.25",
            removal_fraction=0.25,
            prompt_token_count=20.0 + index,
            decision_index32=32.0,
            pre_action_cache_size=1000.0,
            reference_entropy=1.0,
            reference_top2_margin=0.5,
            action_reference_entropy_delta=0.1,
            action_reference_margin_delta=-0.1,
            argmax_match=True,
            js_divergence=0.01,
            target_loose=float(index),
            target_strict=float(index),
        )
        for index in range(5)
    ]

    first = pilot.fit_oof_predictions(rows, "B0", "loose")
    changed = [
        pilot.PilotRow(**{**row.__dict__, "target_loose": 1000.0})
        if row.fold == 0
        else row
        for row in rows
    ]
    second = pilot.fit_oof_predictions(changed, "B0", "loose")

    np.testing.assert_allclose(first[0], second[0])


def test_bootstrap_preserves_cluster_multiplicity() -> None:
    cluster_errors = {
        "B3": np.array([0.0, 1.0, 2.0]),
        "B0": np.array([1.0, 1.0, 1.0]),
    }
    observed = pilot.paired_bootstrap(cluster_errors, replicates=4, seed=0)
    rng = np.random.default_rng(0)
    sampled = rng.integers(0, 3, size=(4, 3))
    expected = np.mean(cluster_errors["B3"][sampled], axis=1)

    np.testing.assert_allclose(observed["B3"], expected)


def test_evaluation_reports_prompt_equal_oof_metrics_and_comparisons(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    loaded = pilot.load_collection(
        root,
        manifest_path,
        design_lock_path=design_lock_path,
        minimum_eligible=0,
    )

    report = pilot.evaluate_rows(
        loaded.rows, minimum_eligible=0, bootstrap_replicates=16
    )

    assert report["status"] == "evaluated_exploratory"
    assert report["claims"]["scope"] == "exploratory_only"
    assert set(report["metrics"]["loose"]) == set(pilot.MODEL_NAMES)
    assert set(report["comparisons"]["loose"]["against"]) == {
        "B0",
        "B1",
        "B2",
        "actionmean",
    }
    assert len(report["oof_rows"]) == 240
    assert report["coverage"]["rows_per_prompt"] == 2


def test_collection_rejects_duplicate_prompt(tmp_path: Path) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    duplicate_path = root / "ifeval_45" / "run.json"
    duplicate_document = json.loads(
        duplicate_path.read_text(encoding="utf-8")
    )
    duplicate_document["results"][0]["prompt"]["prompt_id"] = "ifeval_40"
    duplicate_path.write_text(
        json.dumps(duplicate_document), encoding="utf-8"
    )
    artifacts_path = duplicate_path.with_name("artifacts.json")
    artifacts = json.loads(artifacts_path.read_text())
    artifacts["sha256"]["run.json"] = pilot.file_sha256(duplicate_path)
    artifacts_path.write_text(json.dumps(artifacts), encoding="utf-8")
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["ledger"][45]["artifacts_sha256"] = pilot.file_sha256(
        artifacts_path
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(pilot.EvaluationError, match="does not match ledger"):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_collection_rejects_tampered_run_before_parsing(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    run_path = root / "ifeval_0" / "run.json"
    document = json.loads(run_path.read_text())
    document["results"][0]["tokenization"]["input_length"] = 999
    run_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(pilot.EvaluationError, match="artifact hash mismatch"):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_collection_requires_every_ineligible_ledger_directory(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    (root / "ifeval_159" / "run.json").unlink()

    with pytest.raises(
        pilot.EvaluationError, match="missing local run artifact"
    ):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_collection_rejects_entries_after_target_reached(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(
        tmp_path, early_before_target=False
    )

    with pytest.raises(
        pilot.EvaluationError, match="after the target was reached"
    ):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_collection_requires_complete_experiment_lock(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    lock_path = root / "experiment-lock.json"
    lock = json.loads(lock_path.read_text())
    del lock["source_manifest"]
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["experiment_lock_sha256"] = pilot.file_sha256(lock_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(pilot.EvaluationError, match="fields are incomplete"):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_collection_allows_relocated_manifest_but_rejects_changed_content(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    lock_path = root / "experiment-lock.json"
    lock = json.loads(lock_path.read_text())
    lock["prompt_manifest"]["path"] = (
        "/clustergpu/home/jcampo/herald-v3/data/pilot-v1/prompts.json"
    )
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["experiment_lock_sha256"] = pilot.file_sha256(lock_path)
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    loaded = pilot.load_collection(
        root,
        manifest_path,
        design_lock_path=design_lock_path,
        minimum_eligible=0,
    )
    assert len(loaded.eligible_ids) == 120

    manifest_path.write_text(
        manifest_path.read_text() + "\n", encoding="utf-8"
    )
    with pytest.raises(pilot.EvaluationError, match="prompt manifest hash"):
        pilot.load_collection(
            root,
            manifest_path,
            design_lock_path=design_lock_path,
            minimum_eligible=0,
        )


def test_run_evaluation_writes_once_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    root, manifest_path, design_lock_path = _write_collection(tmp_path)
    output = tmp_path / "report.json"
    report = pilot.run_evaluation(
        root,
        manifest_path,
        output,
        design_lock_path=design_lock_path,
    )

    assert output.is_file()
    assert "design_lock" in report["inputs"]["hashes"]["sha256"]
    with pytest.raises(pilot.EvaluationError, match="overwrite"):
        pilot.run_evaluation(
            root,
            manifest_path,
            output,
            design_lock_path=design_lock_path,
        )
