#!/usr/bin/env python3
"""Collect the sealed full-answer outcomes for the HERALD v3 test set.

The prediction seal is the only authority for the eligible population and
the predictions.  This wrapper validates that seal and the frozen lock before
loading the model, then delegates every full paired run to the unchanged
engineering runner.  It stores only the raw runner artifacts and labels
derived from their raw reference and action scores.
"""

# The imports intentionally follow the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_lookahead as evaluator
from herald_v3.engineering import runner
from herald_v3.engineering.prompts import (
    EngineeringPrompt,
    PromptManifest,
    load_prompt_manifest,
)


SCHEMA_VERSION = "herald_v3.lookahead_outcomes.v1"
LOCK_SCHEMA_VERSION = "herald_v3.lookahead_protocol_lock.v1"
SEAL_SCHEMA_VERSION = "herald_v3.lookahead_prediction_seal.v1"
ACTION_IDS = ("knorm:0.25", "knorm:0.5")
ACTION_RATIOS = {"knorm:0.25": 0.25, "knorm:0.5": 0.5}
MODEL_NAMES = (
    "action_mean",
    "B0_L",
    "B1_L",
    "B2_L",
    "B3_L",
    "B4_L",
)
TARGET_NAMES = ("loose", "strict")
TEST_PROMPT_COUNT = 76
BOUNDARY_INDEX = 32
MAX_NEW_TOKENS = 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RAW_ARTIFACT_NAMES = {
    "prompt-manifest.json",
    "run.json",
    "run.log",
}


class OutcomeCollectionError(ValueError):
    """Raised when sealed outcome collection cannot proceed safely."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of exact file bytes."""
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise OutcomeCollectionError(f"cannot read {path}") from error
    return digest.hexdigest()


def collect_lookahead_outcomes(
    lock_path: str | Path,
    expected_lock_sha256: str,
    seal_path: str | Path,
    expected_seal_sha256: str,
    test_manifest_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Collect or safely resume all sealed eligible test outcomes.

    All input and existing-output validation happens before
    ``runner.load_offline_model``.  A completed verified index is returned
    without loading the model again.
    """
    context = _preflight(
        lock_path=lock_path,
        expected_lock_sha256=expected_lock_sha256,
        seal_path=seal_path,
        expected_seal_sha256=expected_seal_sha256,
        test_manifest_path=test_manifest_path,
        output_path=output_path,
    )
    output = cast(Path, context["output_path"])
    existing = cast(dict[str, object] | None, context["existing_index"])
    if existing is not None and existing.get("status") == "completed":
        return existing

    lock = _object(context["lock"], "protocol lock")
    configuration = _object(lock["configuration"], "lock configuration")
    model, tokenizer = runner.load_offline_model(str(model_path))
    _validate_loaded_model(model, tokenizer, lock, configuration)

    output.mkdir(parents=True, exist_ok=True)
    rows = list(cast(list[dict[str, object]], context["existing_rows"]))
    index = _index_document(context, rows, status="running")
    _write_index(output, index)

    eligible_prompts = cast(
        list[EngineeringPrompt], context["eligible_prompts"]
    )
    manifest = cast(PromptManifest, context["test_manifest"])
    h8_by_prompt = cast(
        dict[str, dict[str, dict[str, object]]], context["h8_by_prompt"]
    )
    for prompt in eligible_prompts[len(rows) :]:
        prompt_dir = output / "raw-runs" / prompt.prompt_id
        if prompt_dir.exists():
            raise OutcomeCollectionError(
                f"unindexed raw run already exists: {prompt.prompt_id}"
            )
        single_manifest = _single_prompt_manifest(manifest, prompt)
        runner.run_engineering(
            model,
            tokenizer,
            single_manifest,
            prompt_dir,
            max_new_tokens=_integer(
                configuration.get("max_new_tokens"), "max_new_tokens"
            ),
            ratios=(0.25, 0.5),
            seed=_integer(configuration.get("seed"), "seed"),
        )
        row = _validate_raw_run(
            prompt_dir,
            prompt,
            h8_by_prompt[prompt.prompt_id],
            _object(context["lock"], "protocol lock"),
        )
        rows.append(row)
        _write_index(
            output,
            _index_document(context, rows, status="running"),
        )

    final = _index_document(context, rows, status="completed")
    _write_index(output, final)
    return final


def main(argv: Sequence[str] | None = None) -> int:
    """Run sealed outcome collection from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", required=True, type=Path)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--seal", required=True, type=Path)
    parser.add_argument("--expected-seal-sha256", required=True)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = collect_lookahead_outcomes(
            args.lock,
            args.expected_lock_sha256,
            args.seal,
            args.expected_seal_sha256,
            args.test_manifest,
            args.model,
            args.output,
        )
    except (OutcomeCollectionError, OSError, TypeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("status") == "completed" else 1


def _preflight(
    *,
    lock_path: str | Path,
    expected_lock_sha256: str,
    seal_path: str | Path,
    expected_seal_sha256: str,
    test_manifest_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Validate every external input and existing raw artifact first."""
    lock_file = Path(lock_path).resolve()
    seal_file = Path(seal_path).resolve()
    manifest_file = Path(test_manifest_path).resolve()
    output = Path(output_path).resolve()
    lock_file_sha = _verify_external_digest(
        lock_file, expected_lock_sha256, "protocol lock"
    )
    seal_file_sha = _verify_external_digest(
        seal_file, expected_seal_sha256, "prediction seal"
    )

    lock = evaluator.load_json(lock_file)
    _validate_lock_shape(lock, lock_file)
    manifests = _object(lock["manifests"], "protocol lock manifests")
    train_ref = _object(manifests["train"], "training manifest reference")
    reuse_ref = _object(lock["training_reuse"], "training reuse reference")
    train_file = _resolve_locked_path(lock_file, train_ref["path"])
    reuse_file = _resolve_locked_path(lock_file, reuse_ref["path"])
    train_manifest = load_prompt_manifest(train_file, limit=120)
    test_manifest = load_prompt_manifest(
        manifest_file, limit=TEST_PROMPT_COUNT
    )
    _validate_manifest_ref(
        _object(manifests["train"], "training manifest reference"),
        train_manifest,
        train_file,
        "train",
    )
    _validate_manifest_ref(
        _object(manifests["test"], "test manifest reference"),
        test_manifest,
        manifest_file,
        "test",
    )
    train_ids = _manifest_ids(train_manifest)
    test_ids = _manifest_ids(test_manifest)
    if set(train_ids) & set(test_ids):
        raise OutcomeCollectionError("training and test manifests overlap")
    expected_reuse_hash = _sha256_value(
        reuse_ref.get("sha256"), "training reuse hash"
    )
    if file_sha256(reuse_file) != expected_reuse_hash:
        raise OutcomeCollectionError("training reuse file hash differs")
    if len(test_ids) != TEST_PROMPT_COUNT:
        raise OutcomeCollectionError(
            "test manifest must contain exactly 76 IDs"
        )

    try:
        seal = evaluator.verify_prediction_seal(seal_file)
        seal_body = dict(seal)
        seal_body.pop("seal_sha256", None)
        canonical_hash = evaluator.canonical_sha256(seal_body)
    except (
        evaluator.EvaluationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise OutcomeCollectionError(
            f"prediction seal validation failed: {error}"
        ) from error
    declared_seal_hash = _string(seal.get("seal_sha256"), "seal_sha256")
    if canonical_hash != declared_seal_hash:
        raise OutcomeCollectionError("prediction seal canonical hash differs")

    eligible_prompts, h8_by_prompt = _validate_seal(
        seal,
        lock,
        lock_file_sha,
        seal_file_sha,
        test_manifest,
        manifest_file,
        lock_file,
    )
    existing_index, existing_rows = _preflight_output(
        output,
        lock_file_sha,
        seal_file_sha,
        declared_seal_hash,
        file_sha256(manifest_file),
        _object(seal, "prediction seal")["runtime"],
        test_manifest,
        eligible_prompts,
        h8_by_prompt,
        lock,
    )
    return {
        "lock": lock,
        "seal": seal,
        "lock_path": str(lock_file),
        "seal_path": str(seal_file),
        "test_manifest": test_manifest,
        "test_manifest_path": str(manifest_file),
        "test_manifest_sha256": file_sha256(manifest_file),
        "output_path": output,
        "lock_file_sha256": lock_file_sha,
        "seal_file_sha256": seal_file_sha,
        "seal_sha256": declared_seal_hash,
        "eligible_prompts": eligible_prompts,
        "h8_by_prompt": h8_by_prompt,
        "existing_index": existing_index,
        "existing_rows": existing_rows,
    }


def _validate_lock_shape(lock: Mapping[str, object], lock_file: Path) -> None:
    """Require the exact protocol-lock builder shape used by the evaluator."""
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise OutcomeCollectionError("unsupported protocol lock schema")
    required = {
        "manifests",
        "training_reuse",
        "boundary",
        "actions",
        "configuration",
        "features",
        "estimator",
        "prediction_seal_schema",
        "scorer_provenance",
        "outcome_collection_schema",
        "source_manifest",
        "software",
        "model",
        "tokenizer",
        "outcomes_collected",
    }
    missing = sorted(required - set(lock))
    if missing:
        raise OutcomeCollectionError(
            "protocol lock missing required fields: " + ", ".join(missing)
        )
    if lock.get("outcomes_collected") is not False:
        raise OutcomeCollectionError(
            "protocol lock already contains outcomes"
        )
    manifests = _object(lock["manifests"], "protocol lock manifests")
    for name in ("train", "test"):
        ref = _object(manifests.get(name), f"{name} manifest reference")
        for key in ("path", "file_sha256", "manifest_sha256", "prompt_ids"):
            if key not in ref:
                raise OutcomeCollectionError(
                    f"{name} manifest reference missing {key}"
                )
        _string(ref["path"], f"{name} manifest path")
        _sha256_value(ref["file_sha256"], f"{name} manifest file hash")
        _sha256_value(ref["manifest_sha256"], f"{name} manifest hash")
        if not isinstance(ref["prompt_ids"], list):
            raise OutcomeCollectionError(f"{name} manifest IDs are malformed")
    reuse = _object(lock["training_reuse"], "training reuse reference")
    for key in ("path", "sha256"):
        if key not in reuse:
            raise OutcomeCollectionError(
                f"training reuse reference missing {key}"
            )
    _sha256_value(reuse["sha256"], "training reuse hash")

    try:
        evaluator._validate_locked_actions(lock["actions"])
        evaluator._validate_locked_configuration(lock["configuration"])
        evaluator._validate_locked_features(lock["features"])
        evaluator._validate_locked_estimator(lock["estimator"])
        evaluator._validate_source_manifest(lock["source_manifest"])
        _validate_software_shape(lock["software"])
    except (
        evaluator.EvaluationError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        raise OutcomeCollectionError(
            f"protocol lock shape validation failed: {error}"
        ) from error
    if lock["prediction_seal_schema"] != SEAL_SCHEMA_VERSION:
        raise OutcomeCollectionError(
            "protocol lock prediction seal schema differs"
        )
    if lock["outcome_collection_schema"] != SCHEMA_VERSION:
        raise OutcomeCollectionError("protocol lock outcome schema differs")
    boundary = _object(lock["boundary"], "protocol lock boundary")
    for key, expected in (
        ("committed_output_tokens", 32),
        ("first_affected_prediction_output_index", 32),
        ("pending_token_output_index", 31),
    ):
        if boundary.get(key) != expected:
            raise OutcomeCollectionError(
                f"protocol lock boundary differs: {key}"
            )
    scorer = _object(lock["scorer_provenance"], "scorer provenance")
    _validate_scorer_provenance(scorer)
    _validate_generation_software(lock["software"])
    model = _object(lock["model"], "locked model")
    if not model or not _string(
        model.get("checkpoint_fingerprint"), "model fingerprint"
    ):
        raise OutcomeCollectionError("locked model identity is incomplete")
    tokenizer = _object(lock["tokenizer"], "locked tokenizer")
    _string(tokenizer.get("name_or_path"), "locked tokenizer name")
    source = _object(lock["source_manifest"], "source manifest")
    for name in (
        "scripts/collect_lookahead_outcomes.py",
        "scripts/evaluate_lookahead.py",
        "src/herald_v3/engineering/runner.py",
    ):
        if name not in source:
            raise OutcomeCollectionError(
                f"protocol lock source manifest lacks {name}"
            )
    if file_sha256(lock_file) == "":
        raise OutcomeCollectionError("protocol lock hash is empty")


def _validate_manifest_ref(
    reference: Mapping[str, object],
    manifest: Any,
    path: Path,
    name: str,
) -> None:
    if _sha256_value(
        reference.get("file_sha256"), f"{name} file hash"
    ) != file_sha256(path):
        raise OutcomeCollectionError(f"{name} manifest file hash differs")
    if (
        _sha256_value(
            reference.get("manifest_sha256"), f"{name} manifest hash"
        )
        != manifest.fingerprint
    ):
        raise OutcomeCollectionError(f"{name} manifest semantic hash differs")
    if reference.get("prompt_ids") != _manifest_ids(manifest):
        raise OutcomeCollectionError(f"{name} manifest IDs differ")
    expected_count = 120 if name == "train" else TEST_PROMPT_COUNT
    if len(_manifest_ids(manifest)) != expected_count:
        raise OutcomeCollectionError(
            f"{name} manifest must contain exactly {expected_count} IDs"
        )


def _manifest_ids(manifest: PromptManifest) -> list[str]:
    return [prompt.prompt_id for prompt in manifest.prompts]


def _validate_software_shape(value: object) -> None:
    software = _object(value, "protocol lock software")
    generation = _object(software.get("generation"), "generation software")
    evaluation = _object(software.get("evaluation"), "evaluation software")
    if not generation or not evaluation:
        raise OutcomeCollectionError("protocol lock software is incomplete")
    expected_generation = {
        "torch",
        "transformers",
        "kvpress",
        "datasets",
        "nltk",
        "langdetect",
        "immutabledict",
        "absl-py",
    }
    if set(generation) != expected_generation:
        raise OutcomeCollectionError(
            "protocol lock generation package set differs"
        )
    if set(evaluation) != {"numpy", "scikit-learn"}:
        raise OutcomeCollectionError(
            "protocol lock evaluation package set differs"
        )
    for package, version in generation.items():
        _string(package, "generation package")
        _string(version, f"generation version {package}")
    for package in ("numpy", "scikit-learn"):
        _string(evaluation.get(package), f"evaluation version {package}")


def _validate_generation_software(value: object) -> None:
    """Require this host's generation packages to match the frozen lock."""
    software = _object(value, "protocol lock software")
    generation = _object(software.get("generation"), "generation software")
    for package, expected in generation.items():
        package_name = _string(package, "generation package")
        expected_version = _string(
            expected, f"generation version {package_name}"
        )
        try:
            actual_version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as error:
            raise OutcomeCollectionError(
                f"generation package missing: {package_name}"
            ) from error
        if actual_version != expected_version:
            raise OutcomeCollectionError(
                "generation package version differs: " + package_name
            )


def _validate_loaded_model(
    model: Any,
    tokenizer: Any,
    lock: Mapping[str, object],
    configuration: Mapping[str, object],
) -> None:
    """Reject model/tokenizer drift before creating any new output."""
    try:
        observed = runner.model_signature(model)
    except (AttributeError, StopIteration, TypeError, ValueError) as error:
        raise OutcomeCollectionError(
            "loaded model identity is unavailable"
        ) from error
    if observed != lock.get("model"):
        raise OutcomeCollectionError(
            "loaded model identity differs from lock"
        )
    tokenizer_name = str(getattr(tokenizer, "name_or_path", ""))
    expected_name = _string(
        _object(lock["tokenizer"], "locked tokenizer").get("name_or_path"),
        "locked tokenizer name",
    )
    if tokenizer_name != expected_name:
        raise OutcomeCollectionError(
            "loaded tokenizer identity differs from lock"
        )
    observed_eos = sorted(runner._eos_ids(model, tokenizer))
    if observed_eos != configuration.get("eos_ids"):
        raise OutcomeCollectionError("loaded EOS IDs differ from lock")


def _validate_seal(
    seal: Mapping[str, object],
    lock: Mapping[str, object],
    lock_file_sha: str,
    seal_file_sha: str,
    manifest: Any,
    manifest_file: Path,
    lock_file: Path | None = None,
) -> tuple[list[EngineeringPrompt], dict[str, dict[str, dict[str, object]]]]:
    """Validate seal provenance, ordered eligibility, and H8 records."""
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise OutcomeCollectionError("unsupported prediction seal schema")
    if seal.get("stage") != "fit_prediction_seal":
        raise OutcomeCollectionError("prediction seal is not a fit seal")
    required = {
        "schema_version",
        "stage",
        "protocol_lock",
        "locked_scorer_provenance",
        "inputs",
        "configuration",
        "models",
        "predictions",
        "eligibility_ledger",
        "h8_records",
        "probe_hashes",
        "integrity",
        "runtime",
        "seal_sha256",
        "locked_model",
        "locked_tokenizer",
        "locked_configuration",
        "locked_source_manifest",
        "locked_software",
    }
    missing = sorted(required - set(seal))
    if missing:
        raise OutcomeCollectionError(
            "prediction seal missing required fields: " + ", ".join(missing)
        )
    protocol_ref = _object(seal.get("protocol_lock"), "sealed protocol lock")
    # Paths in a seal describe the producer's filesystem.  Collection may
    # resume from a relocated copy, so the externally pinned file digest is
    # the identity authority and the recorded path is provenance only.
    _string(protocol_ref.get("path"), "sealed protocol lock path")
    if (
        _string(protocol_ref.get("sha256"), "sealed protocol lock hash")
        != lock_file_sha
    ):
        raise OutcomeCollectionError(
            "prediction seal names a different protocol lock"
        )
    inputs = _object(seal.get("inputs"), "sealed inputs")
    for name in (
        "train_manifest",
        "test_manifest",
        "training_reuse",
        "probe_collection_root",
    ):
        if name not in inputs:
            raise OutcomeCollectionError(f"sealed inputs missing {name}")
    if lock_file is not None:
        locked_manifests = _object(
            lock.get("manifests"), "protocol lock manifests"
        )
        locked_train = _object(
            locked_manifests.get("train"), "locked training manifest"
        )
        sealed_train = _object(
            inputs.get("train_manifest"), "sealed training manifest"
        )
        _string(sealed_train.get("path"), "sealed training manifest path")
        if (
            sealed_train.get("file_sha256") != locked_train.get("file_sha256")
            or sealed_train.get("manifest_sha256")
            != locked_train.get("manifest_sha256")
            or sealed_train.get("prompt_ids")
            != locked_train.get("prompt_ids")
        ):
            raise OutcomeCollectionError(
                "sealed training manifest differs from lock"
            )
    test_ref = _object(inputs.get("test_manifest"), "sealed test manifest")
    manifest_file_sha = file_sha256(manifest_file)
    _string(test_ref.get("path"), "sealed test manifest path")
    if lock_file is not None:
        locked_test = _object(
            _object(lock.get("manifests"), "protocol lock manifests").get(
                "test"
            ),
            "locked test manifest",
        )
        if (
            test_ref.get("file_sha256") != locked_test.get("file_sha256")
            or test_ref.get("manifest_sha256")
            != locked_test.get("manifest_sha256")
            or test_ref.get("prompt_ids") != locked_test.get("prompt_ids")
        ):
            raise OutcomeCollectionError(
                "sealed test manifest differs from lock"
            )
    if manifest_file_sha != _string(
        test_ref.get("file_sha256"), "sealed test manifest file hash"
    ):
        raise OutcomeCollectionError("sealed test manifest file hash differs")
    if (
        _string(test_ref.get("manifest_sha256"), "sealed manifest hash")
        != manifest.fingerprint
    ):
        raise OutcomeCollectionError(
            "sealed test manifest semantic hash differs"
        )
    if test_ref.get("prompt_ids") != _manifest_ids(manifest):
        raise OutcomeCollectionError("sealed test manifest IDs differ")
    reuse_ref = _object(inputs["training_reuse"], "sealed training reuse")
    locked_reuse = _object(lock["training_reuse"], "locked training reuse")
    _string(reuse_ref.get("path"), "sealed training reuse path")
    if _sha256_value(
        reuse_ref.get("sha256"), "sealed training reuse hash"
    ) != _sha256_value(
        locked_reuse.get("sha256"), "locked training reuse hash"
    ):
        raise OutcomeCollectionError(
            "sealed training reuse differs from lock"
        )
    _string(inputs["probe_collection_root"], "sealed probe collection root")

    if seal.get("locked_scorer_provenance") != lock.get("scorer_provenance"):
        raise OutcomeCollectionError(
            "sealed scorer provenance differs from lock"
        )
    for seal_key, lock_key in (
        ("locked_model", "model"),
        ("locked_tokenizer", "tokenizer"),
        ("locked_configuration", "configuration"),
        ("locked_source_manifest", "source_manifest"),
        ("locked_software", "software"),
    ):
        if seal.get(seal_key) != lock.get(lock_key):
            raise OutcomeCollectionError(
                f"sealed {seal_key} differs from lock"
            )
    configuration = _object(seal.get("configuration"), "sealed configuration")
    _validate_seal_configuration(configuration, lock)
    integrity = _object(seal.get("integrity"), "sealed integrity")
    if integrity.get("labels_embedded_in_probes") is not False:
        raise OutcomeCollectionError("sealed probes contain trusted labels")
    if integrity.get("test_outcomes_read") is not False:
        raise OutcomeCollectionError(
            "prediction seal was written after outcomes"
        )

    runtime = _object(seal.get("runtime"), "sealed runtime")
    _string(runtime.get("python"), "sealed runtime Python")
    _string(runtime.get("platform"), "sealed runtime platform")
    packages = _object(runtime.get("packages"), "sealed runtime packages")
    eval_packages = _object(
        _object(lock["software"], "lock software").get("evaluation"),
        "lock evaluation packages",
    )
    if packages != eval_packages:
        raise OutcomeCollectionError(
            "sealed evaluation runtime differs from lock"
        )

    models = seal.get("models")
    if not isinstance(models, dict) or set(models) != set(TARGET_NAMES):
        raise OutcomeCollectionError(
            "sealed model parameter targets are incomplete"
        )
    for target in TARGET_NAMES:
        model_map = models[target]
        if not isinstance(model_map, dict) or set(model_map) != set(
            MODEL_NAMES
        ):
            raise OutcomeCollectionError(
                f"sealed model parameters are incomplete: {target}"
            )

    ledger = seal.get("eligibility_ledger")
    if not isinstance(ledger, list) or len(ledger) != TEST_PROMPT_COUNT:
        raise OutcomeCollectionError(
            "sealed eligibility ledger must contain 76 IDs"
        )
    ledger_ids: list[str] = []
    eligible_ids: list[str] = []
    ledger_by_id: dict[str, dict[str, object]] = {}
    for prompt, raw in zip(manifest.prompts, ledger, strict=True):
        entry = _object(raw, "sealed eligibility entry")
        prompt_id = _string(entry.get("prompt_id"), "sealed ledger prompt ID")
        if prompt_id != prompt.prompt_id or prompt_id in ledger_by_id:
            raise OutcomeCollectionError(
                "sealed eligibility order differs from manifest"
            )
        if (
            _string(entry.get("source_hash"), "sealed ledger source hash")
            != prompt.prompt_text_sha256
        ):
            raise OutcomeCollectionError(
                f"sealed source hash differs: {prompt_id}"
            )
        status = entry.get("status")
        if status not in {
            "eligible",
            "accepted",
            "ineligible",
            "ineligible_early_eos",
        }:
            raise OutcomeCollectionError(
                f"unsupported sealed eligibility status: {prompt_id}"
            )
        if status in {"eligible", "accepted"}:
            if entry.get("action_row_count") != 2:
                raise OutcomeCollectionError(
                    f"sealed eligible pair is incomplete: {prompt_id}"
                )
            _string(entry.get("record"), f"sealed probe record {prompt_id}")
            _sha256_value(
                entry.get("record_sha256"),
                f"sealed probe record hash {prompt_id}",
            )
            eligible_ids.append(prompt_id)
        else:
            if "reason" not in entry or "position" not in entry:
                raise OutcomeCollectionError(
                    f"sealed early-EOS ledger is incomplete: {prompt_id}"
                )
            if entry.get("action_row_count") != 0:
                raise OutcomeCollectionError(
                    f"ineligible prompt has action rows: {prompt_id}"
                )
        ledger_ids.append(prompt_id)
        ledger_by_id[prompt_id] = entry
    if ledger_ids != _manifest_ids(manifest):
        raise OutcomeCollectionError(
            "sealed eligibility ledger order differs from manifest"
        )
    if not eligible_ids:
        raise OutcomeCollectionError(
            "sealed prediction set contains no eligible prompts"
        )

    predictions = seal.get("predictions")
    if not isinstance(predictions, dict):
        raise OutcomeCollectionError("sealed predictions are malformed")
    prediction_ids = seal.get("prediction_ids")
    if prediction_ids is not None and prediction_ids != eligible_ids:
        raise OutcomeCollectionError(
            "sealed prediction IDs are not ordered eligible IDs"
        )
    if set(predictions) != set(eligible_ids):
        raise OutcomeCollectionError(
            "sealed predictions do not cover every eligible ID"
        )
    normalized_predictions = evaluator._sealed_predictions(seal)
    if set(normalized_predictions) != set(eligible_ids):
        raise OutcomeCollectionError(
            "sealed predictions do not cover every eligible ID"
        )
    for prompt_id in eligible_ids:
        action_map = predictions[prompt_id]
        if not isinstance(action_map, dict) or set(action_map) != set(
            ACTION_IDS
        ):
            raise OutcomeCollectionError(
                f"sealed prediction action pair incomplete: {prompt_id}"
            )
        for action_id in ACTION_IDS:
            model_map = action_map[action_id]
            if not isinstance(model_map, dict) or set(model_map) != set(
                MODEL_NAMES
            ):
                raise OutcomeCollectionError(
                    "sealed predictions missing model: "
                    f"{prompt_id} {action_id}"
                )
            for model_name in MODEL_NAMES:
                targets = model_map[model_name]
                if not isinstance(targets, dict) or set(targets) != set(
                    TARGET_NAMES
                ):
                    raise OutcomeCollectionError(
                        "sealed predictions missing target: "
                        f"{prompt_id} {action_id} {model_name}"
                    )
                for target in TARGET_NAMES:
                    _finite(
                        targets[target],
                        "sealed prediction "
                        f"{prompt_id} {action_id} {model_name} {target}",
                    )

    h8_raw = seal.get("h8_records")
    if not isinstance(h8_raw, list) or len(h8_raw) != len(eligible_ids) * 2:
        raise OutcomeCollectionError("sealed H8 records are incomplete")
    h8_by_prompt: dict[str, dict[str, dict[str, object]]] = {}
    expected_order = [
        (prompt_id, action_id)
        for prompt_id in eligible_ids
        for action_id in ACTION_IDS
    ]
    for raw, (prompt_id, action_id) in zip(
        h8_raw, expected_order, strict=True
    ):
        item = _validate_h8_record(raw, prompt_id, action_id)
        h8_by_prompt.setdefault(prompt_id, {})[action_id] = item
    for prompt_id in eligible_ids:
        left = h8_by_prompt[prompt_id][ACTION_IDS[0]]
        right = h8_by_prompt[prompt_id][ACTION_IDS[1]]
        for key in (
            "input_token_ids",
            "reference_argmax_token_ids",
            "realized_steps",
            "has_delayed",
            "reference_eos_position",
            "reference_state_fingerprint",
            "boundary_cache_fingerprint",
            "boundary_stable",
            "reference_probe_hash",
            "probe_sha256",
        ):
            if left[key] != right[key]:
                raise OutcomeCollectionError(
                    f"sealed H8 pair differs: {prompt_id}/{key}"
                )
        expected_reference_hash = evaluator.canonical_sha256(
            {
                "forced_input_token_ids": left["input_token_ids"],
                "reference_argmax_token_ids": left[
                    "reference_argmax_token_ids"
                ],
                "reference_eos_position": left["reference_eos_position"],
                "reference_state_fingerprint": left[
                    "reference_state_fingerprint"
                ],
            }
        )
        if left["reference_probe_hash"] != expected_reference_hash:
            raise OutcomeCollectionError(
                f"sealed H8 reference hash differs: {prompt_id}"
            )
    probe_hashes = seal.get("probe_hashes")
    if not isinstance(probe_hashes, dict) or set(probe_hashes) != set(
        eligible_ids
    ):
        raise OutcomeCollectionError(
            "sealed probe hashes are incomplete or reordered"
        )
    for prompt_id in eligible_ids:
        _sha256_value(
            probe_hashes.get(prompt_id), f"sealed probe hash {prompt_id}"
        )
        for action_id in ACTION_IDS:
            if (
                h8_by_prompt[prompt_id][action_id]["probe_sha256"]
                != probe_hashes[prompt_id]
            ):
                raise OutcomeCollectionError(
                    f"sealed probe hash differs: {prompt_id}"
                )
    if integrity.get("eligible_test_action_rows") != len(eligible_ids) * 2:
        raise OutcomeCollectionError("sealed eligible action count differs")
    del seal_file_sha
    prompts_by_id = {prompt.prompt_id: prompt for prompt in manifest.prompts}
    return [
        prompts_by_id[prompt_id] for prompt_id in eligible_ids
    ], h8_by_prompt


def _validate_seal_configuration(
    configuration: Mapping[str, object], lock: Mapping[str, object]
) -> None:
    expected = {
        "actions": list(ACTION_IDS),
        "boundary_index": 32,
        "first_output_index": 32,
        "max_lookahead_steps": 8,
        "targets": ["loose", "strict"],
        "scaler": "StandardScaler",
        "regressor": "Ridge",
        "ridge_alpha": 1.0,
        "prediction_clip": [-1.0, 1.0],
        "fit_rows": 240,
        "fit_prompts": 120,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise OutcomeCollectionError(
                f"sealed configuration differs: {key}"
            )
    lock_config = _object(lock["configuration"], "lock configuration")
    if configuration.get("max_lookahead_steps") != lock_config.get(
        "max_lookahead_steps"
    ):
        raise OutcomeCollectionError("sealed lookahead cap differs from lock")


def _validate_h8_record(
    raw: object, prompt_id: str, action_id: str
) -> dict[str, object]:
    item = _object(raw, "sealed H8 record")
    required = {
        "prompt_id",
        "action_id",
        "removal_fraction",
        "input_token_ids",
        "reference_argmax_token_ids",
        "realized_steps",
        "has_delayed",
        "reference_eos_position",
        "reference_state_fingerprint",
        "boundary_cache_fingerprint",
        "boundary_stable",
        "reference_probe_hash",
        "probe_sha256",
        "probe_path",
    }
    missing = sorted(required - set(item))
    if missing:
        raise OutcomeCollectionError(
            f"sealed H8 record missing fields: {prompt_id} {action_id}"
        )
    if item["prompt_id"] != prompt_id or item["action_id"] != action_id:
        raise OutcomeCollectionError(
            f"sealed H8 lineage differs: {prompt_id} {action_id}"
        )
    if not math.isclose(
        _finite(item["removal_fraction"], "H8 ratio"),
        ACTION_RATIOS[action_id],
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise OutcomeCollectionError(
            f"sealed H8 action differs: {prompt_id} {action_id}"
        )
    input_ids = _integer_list(item["input_token_ids"], "H8 input token IDs")
    ref_ids = _integer_list(
        item["reference_argmax_token_ids"], "H8 reference token IDs"
    )
    realized = _integer(item["realized_steps"], "H8 realized steps")
    if (
        not 1 <= realized <= 8
        or len(input_ids) != realized
        or len(ref_ids) != realized
        or (len(input_ids) > 1 and input_ids[1:] != ref_ids[:-1])
    ):
        raise OutcomeCollectionError(
            f"sealed H8 step count differs: {prompt_id} {action_id}"
        )
    has_delayed = _integer(item["has_delayed"], "H8 delayed flag")
    if has_delayed != int(realized > 1):
        raise OutcomeCollectionError(
            f"sealed H8 delayed flag differs: {prompt_id} {action_id}"
        )
    eos_position = item["reference_eos_position"]
    if eos_position is not None and (
        _integer(eos_position, "H8 EOS position") != realized - 1
    ):
        raise OutcomeCollectionError(
            f"sealed H8 EOS position differs: {prompt_id} {action_id}"
        )
    for name in (
        "reference_state_fingerprint",
        "reference_probe_hash",
        "probe_sha256",
    ):
        _sha256_value(item[name], f"H8 {name}")
    _sha256_value(
        item["boundary_cache_fingerprint"],
        f"H8 boundary cache fingerprint {prompt_id} {action_id}",
    )
    boundary_stable = _object(
        item["boundary_stable"],
        f"H8 boundary stable {prompt_id} {action_id}",
    )
    try:
        normalized_boundary = evaluator._portable_boundary(
            boundary_stable, f"H8 {prompt_id} {action_id}"
        )
    except (evaluator.EvaluationError, TypeError, ValueError) as error:
        raise OutcomeCollectionError(
            f"sealed H8 boundary stable evidence is malformed: "
            f"{prompt_id} {action_id}"
        ) from error
    if normalized_boundary != boundary_stable:
        raise OutcomeCollectionError(
            f"sealed H8 boundary stable evidence is not normalized: "
            f"{prompt_id} {action_id}"
        )
    _string(item["probe_path"], "H8 probe_path")
    return item


def _preflight_output(
    output: Path,
    lock_file_sha: str,
    seal_file_sha: str,
    seal_sha: str,
    test_manifest_sha: str,
    runtime: object,
    manifest: Any,
    eligible_prompts: Sequence[EngineeringPrompt],
    h8_by_prompt: Mapping[str, Mapping[str, Mapping[str, object]]],
    lock: Mapping[str, object],
) -> tuple[dict[str, object] | None, list[dict[str, object]]]:
    """Verify resumable raw files before loading a model or writing output."""
    if not output.exists():
        return None, []
    if not output.is_dir():
        raise OutcomeCollectionError("outcome output is not a directory")
    index_path = output / "outcome-index.json"
    children = {item.name for item in output.iterdir()}
    if not index_path.is_file():
        if children:
            raise OutcomeCollectionError(
                "existing outcome output lacks its index"
            )
        return None, []
    if children - {"outcome-index.json", "raw-runs"}:
        raise OutcomeCollectionError(
            "outcome output contains an unexpected file"
        )
    try:
        index = evaluator.load_json(index_path)
    except evaluator.EvaluationError as error:
        raise OutcomeCollectionError(
            f"cannot read outcome index: {error}"
        ) from error
    _validate_index_metadata(
        index,
        lock_file_sha,
        seal_file_sha,
        seal_sha,
        test_manifest_sha,
        runtime,
        lock,
    )
    raw_rows = index.get("ordered_rows")
    if not isinstance(raw_rows, list):
        raise OutcomeCollectionError(
            "outcome index ordered_rows is malformed"
        )
    if len(raw_rows) > len(eligible_prompts):
        raise OutcomeCollectionError(
            "outcome index contains replacement rows"
        )
    if index.get("status") == "completed" and len(raw_rows) != len(
        eligible_prompts
    ):
        raise OutcomeCollectionError(
            "completed outcome index does not cover every eligible prompt"
        )
    rows: list[dict[str, object]] = []
    # A running index intentionally contains only the verified prefix.  Keep
    # strict pairing while limiting the expected side to that prefix, then
    # the caller can continue with the next unmaterialized prompt.
    for expected_prompt, raw in zip(
        eligible_prompts[: len(raw_rows)], raw_rows, strict=True
    ):
        row = _object(raw, "outcome index row")
        if row.get("prompt_id") != expected_prompt.prompt_id:
            raise OutcomeCollectionError("outcome index rows are reordered")
        artifact_directory = _relative_artifact_directory(
            row.get("artifact_directory")
        )
        raw_root = output / artifact_directory
        validated = _validate_raw_run(
            raw_root,
            expected_prompt,
            h8_by_prompt[expected_prompt.prompt_id],
            lock,
        )
        if validated != row:
            raise OutcomeCollectionError(
                "outcome index row differs from raw artifacts: "
                f"{expected_prompt.prompt_id}"
            )
        rows.append(row)
    raw_root = output / "raw-runs"
    if raw_root.exists():
        if not raw_root.is_dir():
            raise OutcomeCollectionError("raw-runs is not a directory")
        observed = {item.name for item in raw_root.iterdir()}
        indexed = {
            Path(_relative_artifact_directory(row["artifact_directory"])).name
            for row in rows
        }
        if observed != indexed:
            raise OutcomeCollectionError(
                "raw artifacts are not exactly indexed"
            )
    elif rows:
        raise OutcomeCollectionError(
            "indexed raw artifacts directory is missing"
        )
    return index, rows


def _validate_index_metadata(
    index: Mapping[str, object],
    lock_file_sha: str,
    seal_file_sha: str,
    seal_sha: str,
    test_manifest_sha: str,
    runtime: object,
    lock: Mapping[str, object],
) -> None:
    required = {
        "schema_version",
        "status",
        "protocol_lock_sha256",
        "prediction_seal_sha256",
        "prediction_seal_file_sha256",
        "test_manifest_sha256",
        "scorer_provenance",
        "dependencies",
        "source_manifest",
        "model",
        "runtime",
        "code_sha256",
        "runner_code_sha256",
        "ordered_rows",
    }
    if set(index) != required:
        raise OutcomeCollectionError("outcome index metadata is malformed")
    if index.get("schema_version") != SCHEMA_VERSION:
        raise OutcomeCollectionError("unsupported outcome index schema")
    if index.get("protocol_lock_sha256") != lock_file_sha:
        raise OutcomeCollectionError("outcome index protocol lock differs")
    if index.get("prediction_seal_sha256") != seal_sha:
        raise OutcomeCollectionError("outcome index prediction seal differs")
    if index.get("prediction_seal_file_sha256") != seal_file_sha:
        raise OutcomeCollectionError(
            "outcome index prediction seal file differs"
        )
    if index.get("test_manifest_sha256") != test_manifest_sha:
        raise OutcomeCollectionError("outcome index test manifest differs")
    if index.get("scorer_provenance") != lock.get("scorer_provenance"):
        raise OutcomeCollectionError(
            "outcome index scorer provenance differs"
        )
    if index.get("code_sha256") != file_sha256(__file__):
        raise OutcomeCollectionError("outcome collector code changed")
    expected_runner = PROJECT_ROOT / "src/herald_v3/engineering/runner.py"
    if index.get("runner_code_sha256") != file_sha256(expected_runner):
        raise OutcomeCollectionError("outcome runner source changed")
    if index.get("source_manifest") != lock.get("source_manifest"):
        raise OutcomeCollectionError("outcome source manifest differs")
    if index.get("dependencies") != _object(
        _object(lock["software"], "lock software").get("generation"),
        "lock generation packages",
    ):
        raise OutcomeCollectionError("outcome generation dependencies differ")
    if index.get("model") != lock.get("model"):
        raise OutcomeCollectionError("outcome model identity differs")
    if index.get("runtime") != runtime:
        raise OutcomeCollectionError("outcome runtime differs")
    if index.get("status") not in {"running", "completed"}:
        raise OutcomeCollectionError("outcome index status is malformed")


def _validate_raw_run(
    raw_root: Path,
    prompt: EngineeringPrompt,
    h8_actions: Mapping[str, Mapping[str, object]],
    lock: Mapping[str, object],
) -> dict[str, object]:
    """Validate one runner directory and derive signed quality differences."""
    if not raw_root.is_dir():
        raise OutcomeCollectionError(
            f"raw artifact directory is missing: {prompt.prompt_id}"
        )
    if (
        raw_root.parent.name != "raw-runs"
        or raw_root.name != prompt.prompt_id
    ):
        raise OutcomeCollectionError(
            f"raw artifact directory lineage differs: {prompt.prompt_id}"
        )
    expected_names = set(_RAW_ARTIFACT_NAMES)
    children = list(raw_root.iterdir())
    observed_names = {item.name for item in children}
    # The runner writes artifacts.json alongside the three listed files.
    # It is part of the collection directory but deliberately not in the
    # runner's own hash map.
    artifact_manifest = raw_root / "artifacts.json"
    if not artifact_manifest.is_file():
        raise OutcomeCollectionError(
            f"raw artifacts.json is missing: {prompt.prompt_id}"
        )
    if observed_names != expected_names | {"artifacts.json"}:
        raise OutcomeCollectionError(
            f"raw artifact set contains unexpected files: {prompt.prompt_id}"
        )
    if not all(item.is_file() for item in children):
        raise OutcomeCollectionError(
            "raw artifact directory contains a nested path: "
            f"{prompt.prompt_id}"
        )
    try:
        artifacts = evaluator.load_json(artifact_manifest)
        run = evaluator.load_json(raw_root / "run.json")
        prompt_manifest = evaluator.load_json(
            raw_root / "prompt-manifest.json"
        )
    except evaluator.EvaluationError as error:
        raise OutcomeCollectionError(
            f"raw artifact JSON is malformed: {prompt.prompt_id}"
        ) from error
    if artifacts.get("schema_version") != 1:
        raise OutcomeCollectionError(
            f"raw artifacts schema differs: {prompt.prompt_id}"
        )
    hashes = _object(artifacts.get("sha256"), "runner artifact hashes")
    if set(hashes) != expected_names:
        raise OutcomeCollectionError(
            f"runner artifact hash set differs: {prompt.prompt_id}"
        )
    for name in expected_names:
        if file_sha256(raw_root / name) != _string(
            hashes[name], f"runner artifact hash {name}"
        ):
            raise OutcomeCollectionError(
                f"runner artifact hash differs: {prompt.prompt_id}/{name}"
            )
    prompt_entries = prompt_manifest.get("prompts")
    if not isinstance(prompt_entries, list) or prompt_entries != [
        prompt.to_dict()
    ]:
        raise OutcomeCollectionError(
            f"raw prompt manifest differs: {prompt.prompt_id}"
        )
    if run.get("status") != "completed" or run.get("passed") is not True:
        raise OutcomeCollectionError(
            f"raw runner did not pass: {prompt.prompt_id}"
        )
    if run.get("prompt_manifest") != prompt_manifest:
        raise OutcomeCollectionError(
            f"raw run prompt manifest differs: {prompt.prompt_id}"
        )
    _validate_runner_configuration(run, lock, prompt.prompt_id)
    if run.get("model") != lock.get("model"):
        raise OutcomeCollectionError(
            f"raw model identity differs: {prompt.prompt_id}"
        )
    _validate_runner_environment(run, lock, prompt.prompt_id)
    _validate_runner_source_manifest(run, lock, prompt.prompt_id)
    results = run.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise OutcomeCollectionError(
            f"raw runner result count differs: {prompt.prompt_id}"
        )
    result = _object(results[0], "raw runner result")
    if result.get("status") != "accepted":
        raise OutcomeCollectionError(
            f"raw prompt was not accepted: {prompt.prompt_id}"
        )
    if result.get("prompt") != prompt.to_dict():
        raise OutcomeCollectionError(
            f"raw result prompt differs: {prompt.prompt_id}"
        )
    acceptance = _object(result.get("acceptance"), "raw acceptance")
    if acceptance.get("passed") is not True:
        raise OutcomeCollectionError(
            f"raw acceptance did not pass: {prompt.prompt_id}"
        )
    _validate_boundary_and_tokens(result, acceptance, prompt, h8_actions)
    actions = _derive_scores(result, prompt.prompt_id)
    return {
        "prompt_id": prompt.prompt_id,
        "artifact_directory": str(raw_root.relative_to(raw_root.parents[1])),
        "artifacts_sha256": file_sha256(artifact_manifest),
        "run_sha256": file_sha256(raw_root / "run.json"),
        "actions": actions,
    }


def _validate_runner_configuration(
    run: Mapping[str, object], lock: Mapping[str, object], prompt_id: str
) -> None:
    configuration = _object(run.get("configuration"), "runner configuration")
    lock_configuration = _object(lock["configuration"], "lock configuration")
    for key, expected in (
        ("max_new_tokens", MAX_NEW_TOKENS),
        ("decision_tokens", BOUNDARY_INDEX),
        ("seed", lock_configuration.get("seed")),
        (
            "actions",
            [
                {"name": "knorm", "removal_fraction": 0.25},
                {"name": "knorm", "removal_fraction": 0.5},
            ],
        ),
        ("eos_ids", lock_configuration.get("eos_ids")),
        ("decode_skip_special_tokens", True),
    ):
        if configuration.get(key) != expected:
            raise OutcomeCollectionError(
                f"raw runner configuration differs: {prompt_id}/{key}"
            )


def _validate_runner_environment(
    run: Mapping[str, object], lock: Mapping[str, object], prompt_id: str
) -> None:
    environment = _object(run.get("environment"), "runner environment")
    _string(environment.get("python"), "runner Python version")
    _string(environment.get("platform"), "runner platform")
    lock_tokenizer = _object(lock["tokenizer"], "locked tokenizer")
    if environment.get("tokenizer_name_or_path") != lock_tokenizer.get(
        "name_or_path"
    ):
        raise OutcomeCollectionError(
            f"raw tokenizer identity differs: {prompt_id}"
        )
    generation = _object(
        _object(lock["software"], "lock software").get("generation"),
        "lock generation packages",
    )
    if environment.get("packages") != generation:
        raise OutcomeCollectionError(
            f"raw generation dependencies differ: {prompt_id}"
        )
    resources = _object(
        environment.get("resources"), "runner scorer resources"
    )
    locked_environment = _object(
        lock.get("generation_environment"),
        "locked generation environment",
    )
    locked_resources = _object(
        locked_environment.get("resources"),
        "locked scorer resources",
    )
    if resources != locked_resources:
        raise OutcomeCollectionError(
            f"raw scorer resource set differs: {prompt_id}"
        )


def _validate_runner_source_manifest(
    run: Mapping[str, object], lock: Mapping[str, object], prompt_id: str
) -> None:
    raw = _object(run.get("source_manifest"), "runner source manifest")
    hashes = _object(raw.get("sha256"), "runner source hashes")
    locked = _object(lock.get("source_manifest"), "lock source manifest")
    expected: dict[str, object] = {}
    prefix = "src/herald_v3/engineering/"
    for name, digest in locked.items():
        if isinstance(name, str) and name.startswith(prefix):
            expected[name[len(prefix) :]] = digest
    if hashes != expected:
        raise OutcomeCollectionError(
            f"raw engineering source manifest differs: {prompt_id}"
        )


def _validate_boundary_and_tokens(
    result: Mapping[str, object],
    acceptance: Mapping[str, object],
    prompt: EngineeringPrompt,
    h8_actions: Mapping[str, Mapping[str, object]],
) -> None:
    eligibility = _object(acceptance.get("eligibility"), "runner eligibility")
    if (
        eligibility.get("eligible") is not True
        or eligibility.get("requested_decision_tokens") != BOUNDARY_INDEX
    ):
        raise OutcomeCollectionError(
            f"raw boundary eligibility differs: {prompt.prompt_id}"
        )
    boundary = _object(acceptance.get("boundary"), "runner boundary")
    tokenization = _object(result.get("tokenization"), "runner tokenization")
    input_ids = _integer_list(
        tokenization.get("input_ids"), "runner input IDs"
    )
    if (
        boundary.get("prompt_token_ids") != input_ids
        or boundary.get("generated_count") != BOUNDARY_INDEX
    ):
        raise OutcomeCollectionError(
            f"raw boundary tokenization differs: {prompt.prompt_id}"
        )
    generated = _integer_list(
        boundary.get("generated_token_ids"), "runner boundary generated IDs"
    )
    if (
        len(generated) != BOUNDARY_INDEX
        or boundary.get("pending_generated_index") != 31
    ):
        raise OutcomeCollectionError(
            f"raw boundary length differs: {prompt.prompt_id}"
        )
    if boundary.get("pending_token_id") != generated[31]:
        raise OutcomeCollectionError(
            f"raw pending token differs: {prompt.prompt_id}"
        )
    expected_position = len(input_ids) + 31
    if boundary.get("logical_position") != expected_position:
        raise OutcomeCollectionError(
            f"raw boundary position differs: {prompt.prompt_id}"
        )
    try:
        boundary_stable = evaluator._portable_boundary(
            boundary, f"raw {prompt.prompt_id}"
        )
    except (evaluator.EvaluationError, TypeError, ValueError) as error:
        raise OutcomeCollectionError(
            f"raw boundary stable evidence is malformed: {prompt.prompt_id}"
        ) from error
    outputs = _object(result.get("outputs"), "runner outputs")
    uninterrupted = _object(
        outputs.get("uninterrupted"), "runner reference output"
    )
    reference_ids = _integer_list(
        uninterrupted.get("token_ids"), "runner reference output IDs"
    )
    if len(reference_ids) < BOUNDARY_INDEX:
        raise OutcomeCollectionError(
            "raw reference output is shorter than boundary: "
            f"{prompt.prompt_id}"
        )
    if reference_ids[:BOUNDARY_INDEX] != generated:
        raise OutcomeCollectionError(
            f"raw reference boundary prefix differs: {prompt.prompt_id}"
        )
    actions_output = outputs.get("actions")
    if not isinstance(actions_output, dict) or set(actions_output) != set(
        ACTION_IDS
    ):
        raise OutcomeCollectionError(
            f"raw action outputs are incomplete: {prompt.prompt_id}"
        )
    acceptance_arms = acceptance.get("action_arms")
    if not isinstance(acceptance_arms, list) or len(acceptance_arms) != 2:
        raise OutcomeCollectionError(
            f"raw action arms are incomplete: {prompt.prompt_id}"
        )
    action_arm_ids: list[str] = []
    action_arm_by_id: dict[str, Mapping[str, object]] = {}
    for arm in acceptance_arms:
        arm_object = _object(arm, "raw action arm")
        arm_action = _object(arm_object.get("action"), "raw arm action")
        action_id = _string(arm_action.get("action_id"), "raw action ID")
        action_arm_ids.append(action_id)
        if action_id in action_arm_by_id:
            raise OutcomeCollectionError(
                f"duplicate raw action arm: {prompt.prompt_id} {action_id}"
            )
        action_arm_by_id[action_id] = arm_object
    if action_arm_ids != list(ACTION_IDS):
        raise OutcomeCollectionError(
            f"raw action arm order differs: {prompt.prompt_id}"
        )
    for action_id in ACTION_IDS:
        h8 = h8_actions[action_id]
        sealed_boundary = _object(
            h8.get("boundary_stable"),
            f"sealed H8 boundary stable {prompt.prompt_id} {action_id}",
        )
        if boundary_stable != sealed_boundary:
            raise OutcomeCollectionError(
                "raw boundary stable evidence differs: "
                f"{prompt.prompt_id} {action_id}"
            )
        expected_cache_fingerprint = _sha256_value(
            h8.get("boundary_cache_fingerprint"),
            f"sealed H8 boundary cache fingerprint "
            f"{prompt.prompt_id} {action_id}",
        )
        compression = _object(
            action_arm_by_id[action_id].get("compression"),
            f"raw action compression {prompt.prompt_id} {action_id}",
        )
        cache_fingerprint = _sha256_value(
            compression.get("before_fingerprint"),
            f"raw boundary cache fingerprint {prompt.prompt_id} {action_id}",
        )
        if cache_fingerprint != expected_cache_fingerprint:
            raise OutcomeCollectionError(
                "raw boundary cache fingerprint differs: "
                f"{prompt.prompt_id} {action_id}"
            )
        forced_ids = _integer_list(
            h8["input_token_ids"], "sealed H8 input IDs"
        )
        ref_argmax = _integer_list(
            h8["reference_argmax_token_ids"], "sealed H8 reference IDs"
        )
        if forced_ids[0] != generated[31]:
            raise OutcomeCollectionError(
                "raw pending token disagrees with H8: "
                f"{prompt.prompt_id} {action_id}"
            )
        steps = _integer(h8["realized_steps"], "sealed H8 steps")
        if (
            reference_ids[31 : 31 + steps] != forced_ids
            or reference_ids[32 : 32 + steps] != ref_argmax
        ):
            raise OutcomeCollectionError(
                "raw reference trajectory disagrees with H8: "
                f"{prompt.prompt_id} {action_id}"
            )
        output = _object(actions_output[action_id], "runner action output")
        action_meta = _object(output.get("action"), "runner output action")
        expected_meta = {
            "name": "knorm",
            "removal_fraction": ACTION_RATIOS[action_id],
            "action_id": action_id,
        }
        if action_meta != expected_meta:
            raise OutcomeCollectionError(
                f"raw action metadata differs: {prompt.prompt_id} {action_id}"
            )
        action_ids = _integer_list(
            output.get("token_ids"), "runner action output IDs"
        )
        if action_ids[:BOUNDARY_INDEX] != generated:
            raise OutcomeCollectionError(
                "raw action boundary prefix differs: "
                f"{prompt.prompt_id} {action_id}"
            )


def _derive_scores(
    result: Mapping[str, object], prompt_id: str
) -> dict[str, dict[str, dict[str, float]]]:
    scores = _object(result.get("scores"), "runner scores")
    if set(scores) != {"reference", "noop_forks", "actions"}:
        raise OutcomeCollectionError(
            f"raw score document shape differs: {prompt_id}"
        )
    reference = _score_values(scores.get("reference"), "reference score")
    noop_forks = scores.get("noop_forks")
    if not isinstance(noop_forks, list) or len(noop_forks) != 2:
        raise OutcomeCollectionError(
            f"raw no-op score controls are incomplete: {prompt_id}"
        )
    for index, noop in enumerate(noop_forks):
        noop_reference, noop_action = _validate_score_pair(
            noop, f"no-op score {prompt_id}/{index}"
        )
        if noop_reference != reference or noop_action != reference:
            raise OutcomeCollectionError(
                f"raw no-op score differs from reference: {prompt_id}/{index}"
            )
    action_scores = scores.get("actions")
    if not isinstance(action_scores, dict) or set(action_scores) != set(
        ACTION_IDS
    ):
        raise OutcomeCollectionError(
            f"raw action scores are incomplete: {prompt_id}"
        )
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for action_id in ACTION_IDS:
        pair_reference, action = _validate_score_pair(
            action_scores[action_id], f"action score {prompt_id}/{action_id}"
        )
        if pair_reference != reference:
            raise OutcomeCollectionError(
                f"raw reference score differs: {prompt_id} {action_id}"
            )
        rows[action_id] = {
            "q_reference": dict(reference),
            "q_action": dict(action),
            "d": {
                target: reference[target] - action[target]
                for target in TARGET_NAMES
            },
        }
    return rows


def _score_values(value: object, label: str) -> dict[str, float]:
    score = _object(value, label)
    if set(score) != {
        "instruction_count",
        "loose",
        "loose_pass",
        "strict",
        "strict_pass",
    }:
        raise OutcomeCollectionError(f"{label} schema is malformed")
    instruction_count = _integer(
        score.get("instruction_count"), f"{label} instruction count"
    )
    if instruction_count <= 0:
        raise OutcomeCollectionError(
            f"{label} instruction count must be positive"
        )
    result: dict[str, float] = {}
    for target in TARGET_NAMES:
        score_value = _bounded_score(score.get(target), f"{label} {target}")
        vector_value = score.get(target + "_pass")
        if (
            not isinstance(vector_value, list)
            or len(vector_value) != instruction_count
            or not all(isinstance(item, bool) for item in vector_value)
        ):
            raise OutcomeCollectionError(
                f"{label} {target} pass vector is malformed"
            )
        expected = sum(1 for item in vector_value if item) / instruction_count
        if not math.isclose(
            score_value, expected, abs_tol=1e-12, rel_tol=0.0
        ):
            raise OutcomeCollectionError(
                f"{label} {target} score does not match pass vector"
            )
        result[target] = score_value
    return result


def _validate_score_pair(
    value: object, label: str
) -> tuple[dict[str, float], dict[str, float]]:
    pair = _object(value, label)
    if set(pair) != {"reference", "action", "d_loose", "d_strict"}:
        raise OutcomeCollectionError(f"{label} schema is malformed")
    reference = _score_values(pair["reference"], f"{label} reference")
    action = _score_values(pair["action"], f"{label} action")
    for target in TARGET_NAMES:
        declared = _finite(pair["d_" + target], f"{label} d_{target}")
        if not -1.0 <= declared <= 1.0:
            raise OutcomeCollectionError(
                f"{label} d_{target} must be between -1 and 1"
            )
        expected = reference[target] - action[target]
        if not math.isclose(declared, expected, abs_tol=1e-12, rel_tol=0.0):
            raise OutcomeCollectionError(
                f"{label} d_{target} differs from scores"
            )
    return reference, action


def _bounded_score(value: object, label: str) -> float:
    score = _finite(value, label)
    if not 0.0 <= score <= 1.0:
        raise OutcomeCollectionError(f"{label} must be between 0 and 1")
    return score


def _index_document(
    context: Mapping[str, object],
    rows: Sequence[dict[str, object]],
    *,
    status: str,
) -> dict[str, object]:
    lock = _object(context["lock"], "protocol lock")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "protocol_lock_sha256": context["lock_file_sha256"],
        "prediction_seal_sha256": context["seal_sha256"],
        "prediction_seal_file_sha256": context["seal_file_sha256"],
        "test_manifest_sha256": context["test_manifest_sha256"],
        "scorer_provenance": lock["scorer_provenance"],
        "dependencies": _object(
            _object(lock["software"], "lock software").get("generation"),
            "lock generation packages",
        ),
        "source_manifest": lock["source_manifest"],
        "model": lock["model"],
        "runtime": _object(context["seal"], "prediction seal")["runtime"],
        "code_sha256": file_sha256(__file__),
        "runner_code_sha256": file_sha256(
            PROJECT_ROOT / "src/herald_v3/engineering/runner.py"
        ),
        "ordered_rows": list(rows),
    }


def _write_index(output: Path, document: Mapping[str, object]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    target = output / "outcome-index.json"
    temporary = output / ".outcome-index.json.tmp"
    text = (
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    )
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, target)


def _single_prompt_manifest(
    manifest: PromptManifest, prompt: EngineeringPrompt
) -> PromptManifest:
    return PromptManifest(
        source_path=manifest.source_path,
        source_sha256=manifest.source_sha256,
        source_bytes=manifest.source_bytes,
        official_dataset_path=manifest.official_dataset_path,
        official_dataset_sha256=manifest.official_dataset_sha256,
        official_dataset_bytes=manifest.official_dataset_bytes,
        official_dataset_rows=manifest.official_dataset_rows,
        selection_rule=manifest.selection_rule,
        prompts=(prompt,),
    )


def _resolve_locked_path(lock_file: Path, value: object) -> Path:
    path_text = _string(value, "locked path")
    path = Path(path_text)
    return path if path.is_absolute() else (lock_file.parent / path).resolve()


def _verify_external_digest(path: Path, expected: object, label: str) -> str:
    expected_text = _sha256_value(expected, f"expected {label} hash")
    actual = file_sha256(path)
    if actual != expected_text:
        raise OutcomeCollectionError(f"{label} file hash differs")
    return actual


def _validate_scorer_provenance(value: Mapping[str, object]) -> None:
    if not value:
        raise OutcomeCollectionError("scorer provenance is empty")
    files = value.get("files")
    if not isinstance(files, dict) or not files:
        raise OutcomeCollectionError("scorer provenance files are missing")
    for name, raw in files.items():
        _string(name, "scorer file name")
        item = _object(raw, "scorer file provenance")
        _sha256_value(item.get("sha256"), "scorer file hash")
        _integer(item.get("bytes"), "scorer file byte count")


def _relative_artifact_directory(value: object) -> str:
    text = _string(value, "artifact directory")
    path = Path(text)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.parts[:1] != ("raw-runs",)
    ):
        raise OutcomeCollectionError(
            "artifact directory must be relative to raw-runs"
        )
    return text


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise OutcomeCollectionError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OutcomeCollectionError(f"{label} must be a non-empty string")
    return value


def _sha256_value(value: object, label: str) -> str:
    text = _string(value, label)
    if _SHA256_RE.fullmatch(text) is None:
        raise OutcomeCollectionError(f"{label} is not a SHA-256 digest")
    return text


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise OutcomeCollectionError(f"{label} must be an integer")
    return value


def _integer_list(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise OutcomeCollectionError(f"{label} must be an integer list")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OutcomeCollectionError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OutcomeCollectionError(f"{label} must be finite")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
