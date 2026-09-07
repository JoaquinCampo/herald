"""End-to-end producer and consumer coverage for the lookahead study."""

# ruff: noqa: E402, I001

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import pytest
from tests.test_engine import _tiny_model
from tests.test_runner import _Tokenizer

from herald_v3.engineering import lookahead_collection
from herald_v3.engineering.prompts import (
    PromptManifest,
    load_prompt_manifest,
    write_prompt_manifest,
)

import collect_lookahead_outcomes as outcomes  # noqa: E402
import evaluate_lookahead as evaluator  # noqa: E402
import freeze_lookahead as freezer  # noqa: E402


class _ManifestTokenizer(_Tokenizer):
    def __init__(self, manifest: PromptManifest) -> None:
        self._prompt_texts = {
            prompt.user_prompt: prompt.prompt_text
            for prompt in manifest.prompts
        }

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_tensors: str | None = None,
    ) -> object:
        if tokenize:
            return super().apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                return_tensors=return_tensors,
            )
        return self._prompt_texts[messages[-1]["content"]]


def _write_tiny_protocol_inputs(
    tmp_path: Path,
    manifest: PromptManifest,
) -> tuple[Path, Path]:
    manifest_path = tmp_path / "tiny-test-prompts.json"
    write_prompt_manifest(manifest, manifest_path)
    lock = {
        "manifests": {
            "test": {
                "prompt_ids": [
                    prompt.prompt_id for prompt in manifest.prompts
                ],
                "manifest_sha256": manifest.fingerprint,
            }
        },
        "configuration": {
            "seed": 0,
            "decision_tokens": 32,
            "max_lookahead_steps": lookahead_collection.MAX_STEPS,
            "actions": [
                {
                    "name": "knorm",
                    "removal_fraction": ratio,
                }
                for ratio in lookahead_collection.ACTION_RATIOS
            ],
        },
        "outcomes_collected": False,
    }
    lock_path = tmp_path / "tiny-protocol-lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, lock_path


def _collector_rows(
    collection_root: Path,
    manifest: evaluator.LoadedPrompts,
) -> tuple[tuple[evaluator.LookaheadRow, ...], list[dict[str, object]]]:
    rows: list[evaluator.LookaheadRow] = []
    for prompt_id in manifest.ids:
        record_path = collection_root / prompt_id / "record.json"
        record = evaluator.load_json(record_path)
        for action_id in evaluator.ACTION_IDS:
            rows.append(
                evaluator._row_from_probe(
                    prompt_id,
                    action_id,
                    cast(dict[str, object], record["actions"])[action_id],
                    record,
                    record_path,
                )
            )
    checkpoint = evaluator.load_json(collection_root / "checkpoint.json")
    return tuple(rows), cast(list[dict[str, object]], checkpoint["ledger"])


def _synthetic_probe_collection(
    manifest: evaluator.LoadedPrompts,
    base_rows: tuple[evaluator.LookaheadRow, ...],
    collector_ledger: list[dict[str, object]],
) -> evaluator.LoadedProbeCollection:
    by_action = {row.action_id: row for row in base_rows}
    rows: list[evaluator.LookaheadRow] = []
    ledger: list[dict[str, object]] = []
    probe_hashes: dict[str, str] = {}
    for index, prompt_id in enumerate(manifest.ids):
        for action_id in evaluator.ACTION_IDS:
            rows.append(
                replace(
                    by_action[action_id],
                    prompt_id=prompt_id,
                )
            )
        if index < len(collector_ledger):
            entry = dict(collector_ledger[index])
        else:
            entry = {
                "prompt_id": prompt_id,
                "status": "eligible",
                "record": f"generated/{prompt_id}.json",
                "record_sha256": "a" * 64,
                "action_row_count": 2,
                "source_hash": manifest.records[prompt_id][
                    "prompt_text_utf8_sha256"
                ],
            }
        if index >= len(collector_ledger):
            entry["prompt_id"] = prompt_id
            entry["source_hash"] = manifest.records[prompt_id][
                "prompt_text_utf8_sha256"
            ]
        ledger.append(entry)
        probe_hashes[prompt_id] = by_action[
            evaluator.ACTION_IDS[0]
        ].probe_sha256
    return evaluator.LoadedProbeCollection(
        rows=tuple(rows),
        ledger=tuple(ledger),
        probe_hashes=probe_hashes,
    )


def _training_outcomes(
    manifest: evaluator.LoadedPrompts,
    boundary_stable: dict[str, object],
    boundary_cache_fingerprint: str,
) -> dict[str, evaluator.TrainingOutcome]:
    result: dict[str, evaluator.TrainingOutcome] = {}
    for index, prompt_id in enumerate(manifest.ids):
        result[prompt_id] = evaluator.TrainingOutcome(
            targets={
                action_id: {
                    "loose": 0.05 + index / 1_000 + action_index / 100,
                    "strict": 0.03 + index / 1_000 + action_index / 100,
                }
                for action_index, action_id in enumerate(evaluator.ACTION_IDS)
            },
            source_hashes={},
            boundary_cache_fingerprint=boundary_cache_fingerprint,
            boundary_stable=boundary_stable,
        )
    return result


def test_real_collector_seal_preflight_and_score_preserve_manifest_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frozen_root = tmp_path / "frozen"
    freezer.freeze(frozen_root)
    train_path = frozen_root / "train-prompts.json"
    test_path = frozen_root / "test-prompts.json"
    reuse_path = frozen_root / "training-reuse.json"
    lock_path = frozen_root / "protocol-lock.json"

    full_test = load_prompt_manifest(test_path, limit=76)
    tiny_manifest = replace(full_test, prompts=full_test.prompts[:2])
    tiny_manifest_path, tiny_lock_path = _write_tiny_protocol_inputs(
        tmp_path, tiny_manifest
    )
    tiny_collection_root = tmp_path / "tiny-test-probes"
    tiny_summary = lookahead_collection.collect_lookahead(
        _tiny_model(),
        _ManifestTokenizer(tiny_manifest),
        tiny_manifest,
        tiny_manifest_path,
        tiny_lock_path,
        tiny_collection_root,
        phase="test",
        expected_prompt_count=2,
    )
    assert tiny_summary["status"] == "completed"
    assert tiny_summary["ledger"]

    train_manifest = evaluator.load_prompt_manifest(train_path)
    test_manifest = evaluator.load_prompt_manifest(test_path)
    tiny_rows, tiny_ledger = _collector_rows(
        tiny_collection_root,
        evaluator.load_prompt_manifest(tiny_manifest_path),
    )
    train_probe = _synthetic_probe_collection(
        train_manifest,
        tiny_rows,
        tiny_ledger,
    )
    test_probe = _synthetic_probe_collection(
        test_manifest,
        tiny_rows,
        tiny_ledger,
    )
    training = _training_outcomes(
        train_manifest,
        tiny_rows[0].boundary_stable,
        tiny_rows[0].boundary_cache_fingerprint,
    )
    monkeypatch.setattr(
        evaluator,
        "load_training_outcomes",
        lambda *args, **kwargs: training,
    )
    monkeypatch.setattr(
        evaluator,
        "load_probe_collection",
        lambda root, manifest: (
            train_probe if len(manifest.ids) == 120 else test_probe
        ),
    )
    for collection_root in (
        tmp_path / "train-probes",
        tmp_path / "test-probes",
    ):
        collection_root.mkdir()
        (collection_root / "collection-lock.json").write_text(
            json.dumps(
                {
                    "protocol_lock": {
                        "sha256": evaluator.file_sha256(lock_path),
                    }
                }
            ),
            encoding="utf-8",
        )

    seal_path = tmp_path / "prediction-seal.json"
    evaluator.fit_prediction_seal(
        train_path,
        test_path,
        reuse_path,
        tmp_path / "train-probes",
        tmp_path / "test-probes",
        lock_path,
        seal_path,
        expected_lock_sha256=evaluator.file_sha256(lock_path),
    )
    reloaded_seal = evaluator.load_json(seal_path)
    expected_ids = list(test_manifest.ids)
    assert expected_ids != sorted(expected_ids)
    assert reloaded_seal["prediction_ids"] == expected_ids

    lock = evaluator.load_json(lock_path)
    monkeypatch.setattr(
        outcomes.importlib.metadata,
        "version",
        lambda package: cast(dict[str, str], lock["software"]["generation"])[
            package
        ],
    )
    context = outcomes._preflight(
        lock_path=lock_path,
        expected_lock_sha256=outcomes.file_sha256(lock_path),
        seal_path=seal_path,
        expected_seal_sha256=outcomes.file_sha256(seal_path),
        test_manifest_path=test_path,
        output_path=tmp_path / "outcomes",
    )
    assert [prompt.prompt_id for prompt in context["eligible_prompts"]] == (
        expected_ids
    )
    assert context["seal"]["prediction_ids"] == expected_ids

    labels = {
        prompt_id: {
            action_id: {"loose": 0.0, "strict": 0.0}
            for action_id in evaluator.ACTION_IDS
        }
        for prompt_id in expected_ids
    }
    report = evaluator.score_predictions(
        reloaded_seal,
        labels,
        bootstrap_replicates=4,
    )
    assert report["coverage"]["eligible_prompt_count"] == 76
