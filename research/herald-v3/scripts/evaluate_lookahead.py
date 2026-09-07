#!/usr/bin/env python3
"""Fit, seal, and score the fixed HERALD v3 lookahead study.

The ``fit`` mode consumes probe-only H8 records and the frozen training
reuse manifest.  It writes one prediction seal before test outcomes are
available.  The ``score`` mode opens that seal once, verifies outcome-run
provenance, and computes the prespecified prompt-cluster analysis.

Probe records contain no quality labels.  Training labels are read from the
immutable pilot runs named by ``training-reuse.json``.  The collector may
store a prompt record in a directory or in a JSON collection file, but its
action values must be the serialized ``LookaheadResult`` objects produced by
the engineering module.
"""

# ruff: noqa: E402, I001

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "herald_v3.lookahead_evaluation.v1"
SEAL_SCHEMA_VERSION = "herald_v3.lookahead_prediction_seal.v1"
PROBE_SCHEMA_VERSION = "herald_v3.engineering.lookahead.v1"
REUSE_SCHEMA_VERSION = "herald_v3.lookahead_training_reuse.v1"
LOCK_SCHEMA_VERSION = "herald_v3.lookahead_protocol_lock.v1"
LOCK_STATUS = "frozen_implementation_pending_owner_collection_approval"
ACTION_IDS = ("knorm:0.25", "knorm:0.5")
ACTION_RATIOS = {"knorm:0.25": 0.25, "knorm:0.5": 0.5}
BOUNDARY_INDEX = 32
FIRST_OUTPUT_INDEX = 32
MAX_LOOKAHEAD_STEPS = 8
TRAIN_PROMPT_COUNT = 120
TEST_PROMPT_MAX = 76
TEST_PROMPT_COUNT = 76
TRAIN_ACTION_ROW_COUNT = 240
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0
NONZERO_PROMPT_FLOOR = 20

BASE_FEATURES = (
    "removal_fraction",
    "prompt_token_count",
    "decision_index32",
    "pre_action_cache_size",
)
BLOCK_FEATURES: dict[str, tuple[str, ...]] = {
    "B0_L": BASE_FEATURES + ("lookahead_steps", "has_delayed"),
    "B1_L": BASE_FEATURES
    + (
        "lookahead_steps",
        "has_delayed",
        "reference_entropy",
        "reference_top2_margin",
    ),
    "B2_L": BASE_FEATURES
    + (
        "lookahead_steps",
        "has_delayed",
        "reference_entropy",
        "reference_top2_margin",
        "action_reference_entropy_delta",
        "action_reference_margin_delta",
        "argmax_match",
    ),
    "B3_L": BASE_FEATURES
    + (
        "lookahead_steps",
        "has_delayed",
        "reference_entropy",
        "reference_top2_margin",
        "action_reference_entropy_delta",
        "action_reference_margin_delta",
        "argmax_match",
        "immediate_js",
    ),
    "B4_L": BASE_FEATURES
    + (
        "lookahead_steps",
        "has_delayed",
        "reference_entropy",
        "reference_top2_margin",
        "action_reference_entropy_delta",
        "action_reference_margin_delta",
        "argmax_match",
        "immediate_js",
        "mean_delayed_js",
    ),
}
MODEL_NAMES = ("action_mean",) + tuple(BLOCK_FEATURES)
TARGET_FIELDS = {"loose": "target_loose", "strict": "target_strict"}
_JSON_FILE_NAMES = {
    "checkpoint.json",
    "eligibility-ledger.json",
    "early-eos-ledger.json",
    "protocol-lock.json",
    "train-prompts.json",
    "test-prompts.json",
    "training-reuse.json",
}


class EvaluationError(ValueError):
    """Raised when the frozen study cannot be evaluated safely."""


@dataclass(frozen=True)
class LookaheadRow:
    """One action row extracted from a probe-only H8 record."""

    prompt_id: str
    action_id: str
    removal_fraction: float
    prompt_token_count: float
    decision_index32: float
    pre_action_cache_size: float
    reference_entropy: float
    reference_top2_margin: float
    action_reference_entropy_delta: float
    action_reference_margin_delta: float
    argmax_match: bool
    immediate_js: float
    lookahead_steps: int
    has_delayed: int
    mean_delayed_js: float
    reference_input_token_ids: tuple[int, ...]
    reference_argmax_token_ids: tuple[int, ...]
    reference_eos_position: int | None
    reference_state_fingerprint: str
    boundary_cache_fingerprint: str
    boundary_stable: dict[str, object]
    reference_probe_hash: str
    probe_sha256: str
    probe_path: str

    def feature_values(self, names: Sequence[str]) -> list[float]:
        return [float(getattr(self, name)) for name in names]

    def h8_dict(self) -> dict[str, object]:
        """Return the complete normalized H8 evidence bound into a seal."""
        return {
            "prompt_id": self.prompt_id,
            "action_id": self.action_id,
            "removal_fraction": self.removal_fraction,
            "input_token_ids": list(self.reference_input_token_ids),
            "reference_argmax_token_ids": list(
                self.reference_argmax_token_ids
            ),
            "realized_steps": self.lookahead_steps,
            "has_delayed": self.has_delayed,
            "reference_eos_position": self.reference_eos_position,
            "reference_state_fingerprint": self.reference_state_fingerprint,
            "boundary_cache_fingerprint": self.boundary_cache_fingerprint,
            "boundary_stable": self.boundary_stable,
            "reference_probe_hash": self.reference_probe_hash,
            "probe_sha256": self.probe_sha256,
            "probe_path": self.probe_path,
        }


@dataclass(frozen=True)
class LoadedPrompts:
    ids: tuple[str, ...]
    records: dict[str, dict[str, object]]
    file_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class LoadedProbeCollection:
    rows: tuple[LookaheadRow, ...]
    ledger: tuple[dict[str, object], ...]
    probe_hashes: dict[str, str]


@dataclass(frozen=True)
class TrainingOutcome:
    targets: dict[str, dict[str, float]]
    source_hashes: dict[str, str]
    boundary_cache_fingerprint: str | None = None
    boundary_stable: dict[str, object] | None = None


def file_sha256(path: str | Path) -> str:
    """Hash exact file bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON without allowing NaN or platform formatting."""
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise EvaluationError("value is not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path) -> dict[str, object]:
    """Load one JSON object and reject malformed top-level values."""
    file_path = Path(path)
    try:
        value = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"cannot read JSON: {file_path}") from error
    if not isinstance(value, dict):
        raise EvaluationError(f"JSON object required: {file_path}")
    return value


def load_prompt_manifest(path: str | Path) -> LoadedPrompts:
    """Validate one frozen ordered prompt manifest."""
    file_path = Path(path)
    document = load_json(file_path)
    prompts = document.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise EvaluationError(f"manifest has no prompts: {file_path}")
    records: dict[str, dict[str, object]] = {}
    ordered: list[str] = []
    for entry in prompts:
        if not isinstance(entry, dict):
            raise EvaluationError("prompt manifest entry is not an object")
        prompt_id = _string(entry.get("prompt_id"), "prompt_id")
        if prompt_id in records:
            raise EvaluationError(f"duplicate prompt ID: {prompt_id}")
        text = _string(entry.get("prompt_text"), f"prompt_text {prompt_id}")
        declared_hash = _string(
            entry.get("prompt_text_utf8_sha256"),
            f"prompt_text_utf8_sha256 {prompt_id}",
        )
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != declared_hash:
            raise EvaluationError(f"prompt text hash mismatch: {prompt_id}")
        records[prompt_id] = entry
        ordered.append(prompt_id)
    manifest_hash = _string(
        document.get("manifest_sha256"), "manifest_sha256"
    )
    content = dict(document)
    content.pop("manifest_sha256", None)
    if canonical_sha256(content) != manifest_hash:
        raise EvaluationError(f"manifest fingerprint mismatch: {file_path}")
    return LoadedPrompts(
        ids=tuple(ordered),
        records=records,
        file_sha256=file_sha256(file_path),
        manifest_sha256=manifest_hash,
    )


def validate_protocol_lock(
    path: str | Path,
    train: LoadedPrompts,
    test: LoadedPrompts,
    reuse_path: str | Path,
) -> dict[str, object]:
    """Validate protocol constants and immutable input hash bindings."""
    lock_path = Path(path)
    lock = load_json(lock_path)
    if lock.get("schema_version") != LOCK_SCHEMA_VERSION:
        raise EvaluationError(f"unsupported protocol lock: {lock_path}")
    if len(train.ids) != TRAIN_PROMPT_COUNT:
        raise EvaluationError(
            "training manifest must contain exactly 120 IDs"
        )
    if len(test.ids) != TEST_PROMPT_COUNT:
        raise EvaluationError("test manifest must contain exactly 76 IDs")
    if set(train.ids) & set(test.ids):
        raise EvaluationError("training and test manifests overlap")
    required = {
        "schema_version",
        "status",
        "manifests",
        "training_reuse",
        "boundary",
        "actions",
        "configuration",
        "features",
        "estimator",
        "evaluation",
        "prediction_seal_schema",
        "scorer_provenance",
        "outcome_collection_schema",
        "source_manifest",
        "software",
        "model",
        "tokenizer",
        "generation_environment",
        "outcomes_collected",
        "proposal_sha256",
        "roster_derivation_sha256",
    }
    if set(lock) != required:
        missing = sorted(required - set(lock))
        extra = sorted(set(lock) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("extra " + ", ".join(extra))
        raise EvaluationError(
            "protocol lock keys are not exact"
            + (": " + "; ".join(details) if details else "")
        )
    if lock.get("status") != LOCK_STATUS:
        raise EvaluationError("protocol lock status differs")
    if lock.get("outcomes_collected") is not False:
        raise EvaluationError("protocol lock already contains outcomes")
    manifests = _object(lock["manifests"], "protocol lock manifests")
    _validate_lock_manifest_ref(manifests, "train", train, lock_path)
    _validate_lock_manifest_ref(manifests, "test", test, lock_path)
    reuse_sha = file_sha256(reuse_path)
    _validate_file_ref(
        _object(lock["training_reuse"], "protocol lock training_reuse"),
        reuse_path,
        "training_reuse",
    )
    boundary = _object(lock["boundary"], "protocol lock boundary")
    for key, expected in (
        ("committed_output_tokens", 32),
        ("first_affected_prediction_output_index", 32),
        ("pending_token_output_index", 31),
    ):
        _expect_int(boundary, key, expected, required=True)
    _validate_locked_actions(lock["actions"])
    _validate_locked_configuration(lock["configuration"])
    _validate_locked_features(lock["features"])
    _validate_locked_estimator(lock["estimator"])
    if lock["prediction_seal_schema"] != SEAL_SCHEMA_VERSION:
        raise EvaluationError("protocol lock prediction seal schema differs")
    if lock["outcome_collection_schema"] != (
        "herald_v3.lookahead_outcomes.v1"
    ):
        raise EvaluationError("protocol lock outcome schema differs")
    _validate_scorer_provenance(lock["scorer_provenance"])
    _validate_source_manifest(lock["source_manifest"])
    _validate_software(lock["software"])
    for name in ("proposal_sha256", "roster_derivation_sha256"):
        _sha256(lock[name], f"protocol lock {name}")
    evaluation = _object(lock["evaluation"], "protocol lock evaluation")
    if not evaluation:
        raise EvaluationError("protocol lock evaluation is empty")
    if not isinstance(lock["model"], dict) or not lock["model"]:
        raise EvaluationError("protocol lock model identity is malformed")
    tokenizer = _object(lock["tokenizer"], "protocol lock tokenizer")
    _validate_locked_tokenizer(tokenizer)
    _validate_generation_environment(lock["generation_environment"])
    return {
        "path": str(lock_path.resolve()),
        "sha256": file_sha256(lock_path),
        "document": lock,
        "train_manifest_sha256": train.file_sha256,
        "test_manifest_sha256": test.file_sha256,
        "training_reuse_sha256": reuse_sha,
    }


def load_probe_collection(
    collection_root: str | Path,
    manifest: LoadedPrompts,
    *,
    require_full_manifest: bool = True,
    require_ledger: bool = True,
) -> LoadedProbeCollection:
    """Load paired probe records and the immutable eligibility ledger."""
    root = Path(collection_root)
    if not root.is_dir():
        raise EvaluationError(f"probe collection is not a directory: {root}")
    collection_lock_path = root / "collection-lock.json"
    checkpoint_path = root / "checkpoint.json"
    if not collection_lock_path.is_file() or not checkpoint_path.is_file():
        raise EvaluationError("probe collection lacks locked checkpoint")
    collection_lock = load_json(collection_lock_path)
    checkpoint = load_json(checkpoint_path)
    if collection_lock.get("schema_version") != (
        "herald_v3.lookahead_collection.v1"
    ) or checkpoint.get("schema_version") != (
        "herald_v3.lookahead_collection.v1"
    ):
        raise EvaluationError("probe collection schema is unsupported")
    if (
        collection_lock.get("outcomes_collected") is not False
        or checkpoint.get("outcomes_collected") is not False
    ):
        raise EvaluationError("probe collection contains outcomes")
    if (
        checkpoint.get("status") != "completed"
        or checkpoint.get("passed") is not True
    ):
        raise EvaluationError("probe collection is not complete")
    if checkpoint.get("lock_sha256") != file_sha256(collection_lock_path):
        raise EvaluationError("probe collection lock hash differs")
    phase = collection_lock.get("phase")
    expected_count = (
        TRAIN_PROMPT_COUNT if phase == "train" else TEST_PROMPT_COUNT
    )
    if phase not in {"train", "test"} or len(manifest.ids) != expected_count:
        raise EvaluationError("probe collection population size differs")
    lock_manifest = _object(
        collection_lock.get("manifest"), "collection manifest"
    )
    if lock_manifest.get("manifest_sha256") != manifest.manifest_sha256:
        raise EvaluationError("probe collection manifest fingerprint differs")
    if lock_manifest.get("file_sha256") != manifest.file_sha256:
        raise EvaluationError("probe collection manifest file hash differs")
    if lock_manifest.get("prompt_ids") != list(manifest.ids):
        raise EvaluationError("probe collection manifest IDs differ")
    if lock_manifest.get("expected_prompt_count") != expected_count:
        raise EvaluationError("probe collection expected count differs")
    if checkpoint.get("expected_prompt_count") != expected_count:
        raise EvaluationError("probe checkpoint expected count differs")
    if checkpoint.get("manifest_sha256") != manifest.manifest_sha256:
        raise EvaluationError("probe checkpoint manifest fingerprint differs")
    if checkpoint.get("prompt_ids") != list(manifest.ids):
        raise EvaluationError("probe checkpoint manifest IDs differ")
    ledger = _load_eligibility_ledger(
        root, manifest, require_ledger=require_ledger
    )
    eligible_ids = tuple(
        entry["prompt_id"]
        for entry in ledger
        if entry.get("status") == "eligible"
    )
    rows: list[LookaheadRow] = []
    hashes: dict[str, str] = {}
    for prompt_id in manifest.ids:
        entry = next(
            item for item in ledger if item["prompt_id"] == prompt_id
        )
        if entry.get("status") == "ineligible_early_eos":
            if _record_path_from_ledger(root, entry, prompt_id) is not None:
                raise EvaluationError(
                    f"ineligible prompt has a probe record: {prompt_id}"
                )
            continue
        record_path = _record_path_from_ledger(root, entry, prompt_id)
        if record_path is None:
            raise EvaluationError(f"missing probe record: {prompt_id}")
        expected_record_hash = _string(
            entry.get("record_sha256"), f"ledger record hash {prompt_id}"
        )
        if file_sha256(record_path) != expected_record_hash:
            raise EvaluationError(f"probe record hash differs: {prompt_id}")
        raw = load_json(record_path)
        _reject_embedded_labels(raw, prompt_id)
        record_id = _record_prompt_id(raw)
        if record_id != prompt_id:
            raise EvaluationError(f"probe prompt ID mismatch: {prompt_id}")
        _validate_prompt_lineage(raw, manifest.records[prompt_id], prompt_id)
        _validate_probe_record_shape(
            raw, prompt_id, manifest, collection_lock, checkpoint
        )
        hashes[prompt_id] = file_sha256(record_path)
        action_map = _action_map(raw, prompt_id)
        if set(action_map) != set(ACTION_IDS):
            raise EvaluationError(
                f"prompt {prompt_id} does not contain both action probes"
            )
        prompt_rows = [
            _row_from_probe(
                prompt_id,
                action_id,
                action_map[action_id],
                raw,
                record_path,
            )
            for action_id in ACTION_IDS
        ]
        _validate_shared_reference(raw, prompt_rows, prompt_id)
        _validate_action_rows(raw, prompt_rows, prompt_id)
        _validate_pair(prompt_rows)
        rows.extend(prompt_rows)
    if require_full_manifest and len(eligible_ids) != len(
        {row.prompt_id for row in rows}
    ):
        raise EvaluationError("eligibility ledger and probe rows disagree")
    if not rows:
        raise EvaluationError("probe collection contains no eligible rows")
    return LoadedProbeCollection(
        rows=tuple(rows),
        ledger=tuple(ledger),
        probe_hashes=hashes,
    )


def load_training_outcomes(
    reuse_path: str | Path,
    train_manifest: LoadedPrompts,
    *,
    pilot_collection_root: str | Path | None = None,
    protocol_lock: Mapping[str, object] | None = None,
) -> dict[str, TrainingOutcome]:
    """Read labels only from the exact old runs named by training reuse."""
    path = Path(reuse_path)
    reuse = load_json(path)
    if reuse.get("schema_version") != REUSE_SCHEMA_VERSION:
        raise EvaluationError("unsupported training-reuse schema")
    if reuse.get("labels_embedded") is not False:
        raise EvaluationError(
            "training-reuse must declare labels_embedded false"
        )
    if reuse.get("expected_prompt_count") != TRAIN_PROMPT_COUNT:
        raise EvaluationError("training-reuse prompt count differs")
    if reuse.get("expected_action_rows") != TRAIN_ACTION_ROW_COUNT:
        raise EvaluationError("training-reuse action row count differs")
    ids = reuse.get("training_prompt_ids")
    if ids != list(train_manifest.ids):
        raise EvaluationError("training-reuse IDs differ from train manifest")
    raw_rows = reuse.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != TRAIN_PROMPT_COUNT:
        raise EvaluationError("training-reuse rows must contain exactly 120")
    collection_root = _resolve_pilot_root(path, reuse, pilot_collection_root)
    outcomes: dict[str, TrainingOutcome] = {}
    for raw in raw_rows:
        row = _object(raw, "training-reuse row")
        prompt_id = _string(row.get("prompt_id"), "reuse prompt_id")
        if prompt_id not in train_manifest.records or prompt_id in outcomes:
            raise EvaluationError(
                f"invalid training-reuse prompt: {prompt_id}"
            )
        _validate_reuse_prompt(
            row, train_manifest.records[prompt_id], prompt_id
        )
        if protocol_lock is not None:
            _validate_reuse_lock_identity(row, protocol_lock, prompt_id)
        artifact_directory = _string(
            row.get(
                "artifact_directory", row.get("portable_artifact_directory")
            ),
            f"artifact_directory {prompt_id}",
        )
        artifact_root = collection_root / artifact_directory
        artifacts = artifact_root / "artifacts.json"
        run_path = artifact_root / "run.json"
        if not artifacts.is_file() or not run_path.is_file():
            raise EvaluationError(
                f"missing reused pilot artifact: {prompt_id}"
            )
        expected_artifact_sha = _string(
            row.get("artifacts_sha256"), f"artifacts_sha256 {prompt_id}"
        )
        if file_sha256(artifacts) != expected_artifact_sha:
            raise EvaluationError(
                f"reused artifact hash mismatch: {prompt_id}"
            )
        files_sha = _object(row.get("files_sha256"), "reuse files_sha256")
        for name, expected in files_sha.items():
            name_text = _string(name, "reuse artifact name")
            if Path(name_text).name != name_text:
                raise EvaluationError(
                    f"reused artifact name is not local: {prompt_id}"
                )
            expected_text = _string(expected, "reuse file hash")
            artifact = artifact_root / name_text
            if (
                not artifact.is_file()
                or file_sha256(artifact) != expected_text
            ):
                raise EvaluationError(
                    "reused source file hash mismatch: "
                    f"{prompt_id}/{name_text}"
                )
        document = load_json(run_path)
        outcomes[prompt_id] = _outcome_from_run(
            document, prompt_id, row, run_path, artifacts
        )
    if tuple(outcomes) != train_manifest.ids:
        raise EvaluationError("training-reuse rows are not in manifest order")
    return outcomes


def _validate_reuse_lock_identity(
    row: Mapping[str, object],
    lock: Mapping[str, object],
    prompt_id: str,
) -> None:
    """Bind every reused run to the frozen generation identities."""
    if row.get("model") != lock.get("model"):
        raise EvaluationError(
            f"reused model differs from protocol lock: {prompt_id}"
        )
    environment = _object(row.get("environment"), "reuse environment")
    locked_environment = _object(
        lock.get("generation_environment"),
        "protocol lock generation environment",
    )
    if environment != locked_environment:
        raise EvaluationError(
            f"reused generation environment differs: {prompt_id}"
        )
    tokenizer_name = _string(
        environment.get("tokenizer_name_or_path"),
        f"reuse tokenizer {prompt_id}",
    )
    tokenizer = _object(lock.get("tokenizer"), "protocol lock tokenizer")
    if tokenizer_name != tokenizer.get("name_or_path"):
        raise EvaluationError(
            f"reused tokenizer differs from protocol lock: {prompt_id}"
        )
    generation_versions = _object(
        _object(lock.get("software"), "protocol lock software").get(
            "generation"
        ),
        "generation software",
    )
    observed_versions = _object(
        environment.get("packages"), f"reuse packages {prompt_id}"
    )
    for package, expected in generation_versions.items():
        if observed_versions.get(package) != expected:
            raise EvaluationError(
                f"reused generation package differs: {prompt_id}/{package}"
            )
    source_manifest = row.get("source_manifest")
    if source_manifest is None:
        raise EvaluationError(f"reused source manifest missing: {prompt_id}")
    _validate_reuse_source_manifest(source_manifest, lock, prompt_id)


def _validate_reuse_source_manifest(
    value: object, lock: Mapping[str, object], prompt_id: str
) -> None:
    observed = _object(value, f"reuse source manifest {prompt_id}")
    hashes = observed.get("sha256")
    if not isinstance(hashes, dict):
        raise EvaluationError(f"reused source hashes missing: {prompt_id}")
    expected = _object(lock.get("source_manifest"), "protocol lock source")
    # Run provenance names the engineering package files relative to its
    # source root, while the lock names files relative to the project root.
    expected_engineering = {
        str(name).removeprefix("src/herald_v3/engineering/"): digest
        for name, digest in expected.items()
        if str(name).startswith("src/herald_v3/engineering/")
    }
    for name, digest in hashes.items():
        filename = _string(name, "reused source filename")
        expected_digest = expected_engineering.get(filename)
        if expected_digest is None or digest != expected_digest:
            raise EvaluationError(
                f"reused source hash differs from protocol lock: {prompt_id}"
            )


def fit_prediction_seal(
    train_manifest_path: str | Path,
    test_manifest_path: str | Path,
    training_reuse_path: str | Path,
    train_probe_collection_root: str | Path,
    test_probe_collection_root: str | Path,
    protocol_lock_path: str | Path,
    output_path: str | Path,
    *,
    expected_lock_sha256: str,
    pilot_collection_root: str | Path | None = None,
) -> dict[str, object]:
    """Fit fixed models and write a signed prediction seal."""
    output = Path(output_path)
    if output.exists():
        raise EvaluationError(f"refusing to overwrite seal: {output}")
    expected_lock_sha256 = _sha256(
        expected_lock_sha256, "expected protocol lock hash"
    )
    if file_sha256(protocol_lock_path) != expected_lock_sha256:
        raise EvaluationError("protocol lock file hash differs from expected")
    train_manifest = load_prompt_manifest(train_manifest_path)
    test_manifest = load_prompt_manifest(test_manifest_path)
    lock = validate_protocol_lock(
        protocol_lock_path,
        train_manifest,
        test_manifest,
        training_reuse_path,
    )
    outcomes = load_training_outcomes(
        training_reuse_path,
        train_manifest,
        pilot_collection_root=pilot_collection_root,
        protocol_lock=cast(Mapping[str, object], lock["document"]),
    )
    _validate_probe_collection_protocol_lock(
        train_probe_collection_root, expected_lock_sha256, "training"
    )
    _validate_probe_collection_protocol_lock(
        test_probe_collection_root, expected_lock_sha256, "test"
    )
    train_probes = load_probe_collection(
        train_probe_collection_root, train_manifest
    )
    test_probes = load_probe_collection(
        test_probe_collection_root, test_manifest
    )
    train_rows = _ordered_rows(train_probes.rows, train_manifest.ids)
    test_rows = _ordered_rows(test_probes.rows, test_manifest.ids)
    _validate_train_rows(train_rows, train_manifest.ids, outcomes)
    models, predictions = fit_models_and_predict(
        train_rows, test_rows, outcomes
    )
    prediction_ids = [
        _string(entry.get("prompt_id"), "eligible prompt ID")
        for entry in test_probes.ledger
        if entry.get("status") == "eligible"
    ]
    if tuple(prediction_ids) != tuple(predictions):
        raise EvaluationError(
            "prediction IDs differ from eligible manifest order"
        )
    seal_body: dict[str, object] = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "stage": "fit_prediction_seal",
        "protocol_lock": {
            "path": lock["path"],
            "sha256": lock["sha256"],
        },
        "prediction_ids": prediction_ids,
        "locked_scorer_provenance": _find_value(
            _object(lock["document"], "protocol lock document"),
            "scorer_provenance",
        ),
        "inputs": {
            "train_manifest": _manifest_seal_value(
                train_manifest_path, train_manifest
            ),
            "test_manifest": _manifest_seal_value(
                test_manifest_path, test_manifest
            ),
            "training_reuse": {
                "path": str(Path(training_reuse_path).resolve()),
                "sha256": file_sha256(training_reuse_path),
            },
            # The outcome collector uses this field as the sealed probe
            # collection lineage marker.  Keep the phase-specific roots
            # below as the evaluator's authoritative inputs as well.
            "probe_collection_root": str(
                Path(test_probe_collection_root).resolve()
            ),
            "train_probe_collection_root": str(
                Path(train_probe_collection_root).resolve()
            ),
            "test_probe_collection_root": str(
                Path(test_probe_collection_root).resolve()
            ),
        },
        "configuration": {
            "actions": list(ACTION_IDS),
            "boundary_index": BOUNDARY_INDEX,
            "first_output_index": FIRST_OUTPUT_INDEX,
            "max_lookahead_steps": MAX_LOOKAHEAD_STEPS,
            "feature_columns": {
                name: list(columns)
                for name, columns in BLOCK_FEATURES.items()
            },
            "targets": ["loose", "strict"],
            "scaler": "StandardScaler",
            "regressor": "Ridge",
            "ridge_alpha": 1.0,
            "prediction_clip": [-1.0, 1.0],
            "fit_rows": len(train_rows),
            "fit_prompts": len(train_manifest.ids),
        },
        "models": models,
        "predictions": predictions,
        "eligibility_ledger": list(test_probes.ledger),
        "h8_records": [row.h8_dict() for row in test_rows],
        "probe_hashes": test_probes.probe_hashes,
        "locked_model": _object(lock["document"], "protocol lock").get(
            "model"
        ),
        "locked_tokenizer": _object(lock["document"], "protocol lock").get(
            "tokenizer"
        ),
        "locked_configuration": _object(
            lock["document"], "protocol lock"
        ).get("configuration"),
        "locked_source_manifest": _object(
            lock["document"], "protocol lock"
        ).get("source_manifest"),
        "locked_software": _object(lock["document"], "protocol lock").get(
            "software"
        ),
        "integrity": {
            "labels_embedded_in_probes": False,
            "test_outcomes_read": False,
            "training_outcomes_source": "training-reuse old run hashes",
            "eligible_test_action_rows": len(test_rows),
        },
        "runtime": _runtime_versions(),
    }
    seal = dict(seal_body)
    seal["seal_sha256"] = canonical_sha256(seal_body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return seal


def fit_models_and_predict(
    train_rows: Sequence[LookaheadRow],
    test_rows: Sequence[LookaheadRow],
    outcomes: Mapping[str, TrainingOutcome],
) -> tuple[dict[str, object], dict[str, dict[str, dict[str, float]]]]:
    """Fit each fixed learned comparator on all 240 training rows."""
    train = list(train_rows)
    test = list(test_rows)
    if len(train) != TRAIN_ACTION_ROW_COUNT:
        raise EvaluationError("fit requires exactly 240 training action rows")
    models: dict[str, object] = {}
    ordered_test_ids: list[str] = []
    seen_test_ids: set[str] = set()
    for row in test:
        if row.prompt_id not in seen_test_ids:
            ordered_test_ids.append(row.prompt_id)
            seen_test_ids.add(row.prompt_id)
    predictions: dict[str, dict[str, dict[str, float]]] = {
        prompt_id: {} for prompt_id in ordered_test_ids
    }
    for target_name in TARGET_FIELDS:
        y = np.asarray(
            [
                outcomes[row.prompt_id].targets[row.action_id][target_name]
                for row in train
            ],
            dtype=float,
        )
        target_models: dict[str, object] = {}
        for model_name in MODEL_NAMES:
            if model_name == "action_mean":
                action_means = {
                    action_id: float(
                        np.mean(
                            [
                                outcomes[row.prompt_id].targets[
                                    row.action_id
                                ][target_name]
                                for row in train
                                if row.action_id == action_id
                            ]
                        )
                    )
                    for action_id in ACTION_IDS
                }
                target_models[model_name] = {
                    "kind": "action_mean",
                    "means": action_means,
                }
                for row in test:
                    predictions[row.prompt_id].setdefault(
                        row.action_id, {}
                    ).setdefault(model_name, {})[target_name] = _clip(
                        action_means[row.action_id]
                    )
                continue
            columns = BLOCK_FEATURES[model_name]
            x_train = np.asarray(
                [row.feature_values(columns) for row in train], dtype=float
            )
            x_test = np.asarray(
                [row.feature_values(columns) for row in test], dtype=float
            )
            if not np.isfinite(x_train).all() or not np.isfinite(y).all():
                raise EvaluationError("nonfinite value reached model fitting")
            scaler = StandardScaler()
            x_train_scaled = scaler.fit_transform(x_train)
            ridge = Ridge(alpha=1.0)
            ridge.fit(x_train_scaled, y)
            predicted = np.clip(
                ridge.predict(scaler.transform(x_test)), -1, 1
            )
            target_models[model_name] = {
                "kind": "StandardScaler+Ridge",
                "features": list(columns),
                "scaler": {
                    "mean": [float(value) for value in scaler.mean_],
                    "scale": [float(value) for value in scaler.scale_],
                    "var": [float(value) for value in scaler.var_],
                    "n_samples_seen": int(scaler.n_samples_seen_),
                },
                "ridge": {
                    "alpha": 1.0,
                    "coef": [float(value) for value in ridge.coef_],
                    "intercept": float(ridge.intercept_),
                    "n_features_in": int(ridge.n_features_in_),
                },
                "prediction_clip": [-1.0, 1.0],
            }
            for row, value in zip(test, predicted, strict=True):
                predictions[row.prompt_id].setdefault(
                    row.action_id, {}
                ).setdefault(model_name, {})[target_name] = float(value)
        models[target_name] = target_models
    return models, predictions


def _validate_probe_collection_protocol_lock(
    collection_root: str | Path, expected_lock_sha256: str, phase: str
) -> None:
    """Require each probe collection to name the active protocol lock."""
    lock_path = Path(collection_root) / "collection-lock.json"
    collection_lock = load_json(lock_path)
    protocol_lock = _object(
        collection_lock.get("protocol_lock"),
        f"{phase} probe collection protocol lock",
    )
    observed = _sha256(
        protocol_lock.get("sha256"),
        f"{phase} probe collection protocol lock hash",
    )
    if observed != expected_lock_sha256:
        raise EvaluationError(
            f"{phase} probe collection protocol lock differs from active lock"
        )


def verify_prediction_seal(path: str | Path) -> dict[str, object]:
    """Read a seal and verify its exact content hash before scoring."""
    seal = load_json(path)
    expected = _string(seal.get("seal_sha256"), "seal_sha256")
    body = dict(seal)
    body.pop("seal_sha256", None)
    if canonical_sha256(body) != expected:
        raise EvaluationError("prediction seal hash does not match content")
    if seal.get("schema_version") != SEAL_SCHEMA_VERSION:
        raise EvaluationError("unsupported prediction seal schema")
    if seal.get("stage") != "fit_prediction_seal":
        raise EvaluationError("prediction seal is not a fit seal")
    if seal.get("integrity", {}).get("test_outcomes_read") is not False:
        raise EvaluationError("prediction seal was written after outcomes")
    predictions = seal.get("predictions")
    if not isinstance(predictions, dict) or not predictions:
        raise EvaluationError("prediction seal has no predictions")
    _sealed_predictions(seal)
    return seal


def score_sealed_predictions(
    seal_path: str | Path,
    outcomes_path: str | Path,
    output_path: str | Path,
    *,
    protocol_lock_path: str | Path,
    expected_lock_sha256: str,
    expected_seal_sha256: str,
) -> dict[str, object]:
    """Verify one sealed prediction set and score its signed outcomes."""
    if file_sha256(protocol_lock_path) != expected_lock_sha256:
        raise EvaluationError("protocol lock file hash differs from expected")
    if file_sha256(seal_path) != expected_seal_sha256:
        raise EvaluationError(
            "prediction seal file hash differs from expected"
        )
    seal = verify_prediction_seal(seal_path)
    expected_lock = _string(
        _object(seal["protocol_lock"], "sealed protocol lock").get("sha256"),
        "sealed protocol lock hash",
    )
    if file_sha256(protocol_lock_path) != expected_lock:
        raise EvaluationError("protocol lock changed after prediction")
    outcomes = load_json(outcomes_path)
    outcome_seal_hash = outcomes.get("prediction_seal_sha256")
    if outcome_seal_hash != seal["seal_sha256"]:
        raise EvaluationError("outcomes do not name this prediction seal")
    _validate_outcome_provenance(outcomes, seal)
    labels = _load_test_outcomes(
        outcomes,
        seal,
        outcomes_path=Path(outcomes_path),
        expected_seal_file_sha256=expected_seal_sha256,
    )
    report = score_predictions(seal, labels)
    report["inputs"] = {
        "seal_path": str(Path(seal_path).resolve()),
        "seal_sha256": seal["seal_sha256"],
        "outcomes_path": str(Path(outcomes_path).resolve()),
        "outcomes_sha256": file_sha256(outcomes_path),
    }
    output = Path(output_path)
    if output.exists():
        raise EvaluationError(f"refusing to overwrite score report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def score_predictions(
    seal: Mapping[str, object],
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Compute fixed prompt-equal metrics and paired cluster intervals."""
    predictions = _sealed_predictions(seal)
    prompt_ids = tuple(predictions)
    if tuple(labels) != prompt_ids:
        raise EvaluationError(
            "outcome IDs differ from sealed prediction order"
        )
    report_targets: dict[str, object] = {}
    for target in ("loose", "strict"):
        model_metrics: dict[str, dict[str, object]] = {}
        errors_by_model: dict[str, np.ndarray] = {}
        for model_name in MODEL_NAMES:
            row_errors: dict[str, list[float]] = defaultdict(list)
            flat_errors: list[float] = []
            values: list[float] = []
            for prompt_id in prompt_ids:
                for action_id in ACTION_IDS:
                    target_value = labels[prompt_id][action_id][target]
                    predicted_value = predictions[prompt_id][action_id][
                        model_name
                    ][target]
                    error = predicted_value - target_value
                    row_errors[prompt_id].append(error)
                    flat_errors.append(error)
                    values.append(target_value)
            cluster_errors = np.asarray(
                [
                    np.mean(np.asarray(row_errors[prompt_id]) ** 2)
                    for prompt_id in prompt_ids
                ],
                dtype=float,
            )
            mean_errors = np.asarray(
                [
                    np.mean(np.asarray(row_errors[prompt_id]))
                    for prompt_id in prompt_ids
                ],
                dtype=float,
            )
            model_metrics[model_name] = _metrics_from_errors(
                flat_errors,
                values,
                cluster_errors,
                mean_errors,
                prompt_ids,
            )
            errors_by_model[model_name] = cluster_errors
        bootstrap = paired_bootstrap(
            errors_by_model,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        comparisons = _comparisons(model_metrics, bootstrap)
        report_targets[target] = {
            "metrics": model_metrics,
            "bootstrap": {
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "cluster_count": len(prompt_ids),
                "mse_by_model": {
                    name: {
                        "lower": float(np.quantile(values, 0.025)),
                        "upper": float(np.quantile(values, 0.975)),
                    }
                    for name, values in bootstrap.items()
                },
                "mse_difference_vs_B4_L": {
                    name: {
                        "lower": float(
                            np.quantile(
                                bootstrap[name] - bootstrap["B4_L"], 0.025
                            )
                        ),
                        "upper": float(
                            np.quantile(
                                bootstrap[name] - bootstrap["B4_L"], 0.975
                            )
                        ),
                    }
                    for name in MODEL_NAMES
                    if name != "B4_L"
                },
            },
            "comparisons": comparisons,
        }
    nonzero_prompts = len(
        {
            prompt_id
            for prompt_id in prompt_ids
            if any(
                labels[prompt_id][action_id]["loose"] != 0.0
                for action_id in ACTION_IDS
            )
        }
    )
    loose_comparisons = _object(
        _object(report_targets["loose"], "loose report").get("comparisons"),
        "loose comparisons",
    )
    status = _status_from_comparisons(
        nonzero_prompts,
        loose_comparisons,
        len(prompt_ids),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "claims": {
            "scope": "positive_exploratory_only"
            if status == "positive_exploratory"
            else "fixed_exploratory_analysis",
            "statement": (
                "This result is conditional on the frozen prompts, model, "
                "actions, probe, and scorer provenance."
            ),
        },
        "coverage": {
            "eligible_prompt_count": len(prompt_ids),
            "action_row_count": len(prompt_ids) * 2,
            "nonzero_loose_prompt_count": nonzero_prompts,
            "nonzero_prompt_floor": NONZERO_PROMPT_FLOOR,
            "information_floor_met": nonzero_prompts >= NONZERO_PROMPT_FLOOR,
            "target_counts": _target_counts(labels),
        },
        "configuration": {
            "models": list(MODEL_NAMES),
            "prompt_weighting": "equal_prompt_then_mean_two_actions",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "prediction_clip": [-1.0, 1.0],
        },
        "targets": report_targets,
    }


def paired_bootstrap(
    cluster_errors: Mapping[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Use one shared prompt resample matrix for every model."""
    if replicates <= 0:
        raise EvaluationError("bootstrap replicates must be positive")
    if not cluster_errors:
        raise EvaluationError("bootstrap requires at least one model")
    arrays = {
        name: np.asarray(values, dtype=float)
        for name, values in cluster_errors.items()
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
        raise EvaluationError(
            "bootstrap clusters must have equal nonzero lengths"
        )
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise EvaluationError("bootstrap clusters must be finite")
    count = next(iter(lengths))
    sampled = np.random.default_rng(seed).integers(
        0, count, size=(replicates, count)
    )
    return {
        name: np.mean(values[sampled], axis=1)
        for name, values in arrays.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run either the pre-outcome fit/seal or post-outcome score stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    fit_parser = subparsers.add_parser("fit")
    fit_parser.add_argument("--train-manifest", required=True)
    fit_parser.add_argument("--test-manifest", required=True)
    fit_parser.add_argument("--training-reuse", required=True)
    fit_parser.add_argument("--train-probes", required=True)
    fit_parser.add_argument("--test-probes", required=True)
    fit_parser.add_argument("--protocol-lock", required=True)
    fit_parser.add_argument("--expected-lock-sha256", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--pilot-collection-root", default=None)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--seal", required=True)
    score_parser.add_argument("--outcomes", required=True)
    score_parser.add_argument("--protocol-lock", required=True)
    score_parser.add_argument("--expected-lock-sha256", required=True)
    score_parser.add_argument("--expected-seal-sha256", required=True)
    score_parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.mode == "fit":
            fit_prediction_seal(
                args.train_manifest,
                args.test_manifest,
                args.training_reuse,
                args.train_probes,
                args.test_probes,
                args.protocol_lock,
                args.output,
                expected_lock_sha256=args.expected_lock_sha256,
                pilot_collection_root=args.pilot_collection_root,
            )
        else:
            score_sealed_predictions(
                args.seal,
                args.outcomes,
                args.output,
                protocol_lock_path=args.protocol_lock,
                expected_lock_sha256=args.expected_lock_sha256,
                expected_seal_sha256=args.expected_seal_sha256,
            )
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _load_eligibility_ledger(
    root: Path,
    manifest: LoadedPrompts,
    *,
    require_ledger: bool,
) -> list[dict[str, object]]:
    checkpoint_path = root / "checkpoint.json"
    if not checkpoint_path.is_file():
        if require_ledger:
            raise EvaluationError("probe collection lacks eligibility ledger")
        raise EvaluationError("eligibility ledger cannot be skipped")
    document = load_json(checkpoint_path)
    raw = document.get("ledger")
    if not isinstance(raw, list):
        raise EvaluationError("eligibility ledger is not a list")
    ledger: list[dict[str, object]] = []
    for value in raw:
        entry = _object(value, "eligibility ledger entry")
        prompt_id = _string(entry.get("prompt_id"), "ledger prompt_id")
        if prompt_id in {item["prompt_id"] for item in ledger}:
            raise EvaluationError(f"duplicate ledger prompt: {prompt_id}")
        status = entry.get("status")
        if status not in {"eligible", "ineligible_early_eos"}:
            raise EvaluationError(f"unsupported eligibility status: {status}")
        if status == "ineligible_early_eos":
            _validate_early_eos_entry(
                entry,
                prompt_id,
                expected_source_hash=_manifest_prompt_source_hash(
                    manifest, prompt_id
                ),
            )
        else:
            required_entry = {
                "prompt_id",
                "status",
                "record",
                "record_sha256",
                "source_hash",
                "action_row_count",
            }
            if set(entry) != required_entry or entry["action_row_count"] != 2:
                raise EvaluationError(
                    f"eligible ledger entry is incomplete: {prompt_id}"
                )
            _string(entry["record"], f"ledger record path {prompt_id}")
            _string(entry["record_sha256"], f"ledger record hash {prompt_id}")
            _validate_source_hash(
                entry["source_hash"],
                _manifest_prompt_source_hash(manifest, prompt_id),
                prompt_id,
            )
        ledger.append(entry)
    observed_ids = [item["prompt_id"] for item in ledger]
    expected_tuple = manifest.ids
    if tuple(observed_ids) != expected_tuple:
        raise EvaluationError(
            "eligibility ledger order differs from manifest"
        )
    return ledger


def _validate_early_eos_entry(
    entry: Mapping[str, object],
    prompt_id: str,
    *,
    expected_source_hash: str | None = None,
) -> None:
    required = {
        "prompt_id",
        "status",
        "reason",
        "position",
        "source_hash",
        "action_row_count",
    }
    if set(entry) != required or entry.get("action_row_count") != 0:
        raise EvaluationError(
            f"early-EOS ledger entry is incomplete: {prompt_id}"
        )
    _string(entry["reason"], f"early-EOS reason {prompt_id}")
    _integer(entry["position"], f"early-EOS position {prompt_id}")
    _validate_source_hash(
        entry["source_hash"], expected_source_hash, prompt_id
    )


def _manifest_prompt_source_hash(
    manifest: LoadedPrompts, prompt_id: str
) -> str:
    record = manifest.records.get(prompt_id)
    if record is None:
        raise EvaluationError(f"manifest prompt is missing: {prompt_id}")
    return _string(
        record.get("prompt_text_utf8_sha256"),
        f"manifest source hash {prompt_id}",
    )


def _validate_source_hash(
    value: object, expected: str | None, prompt_id: str
) -> None:
    observed = _sha256(value, f"ledger source hash {prompt_id}")
    if expected is not None and observed != expected:
        raise EvaluationError(f"ledger source hash differs: {prompt_id}")


def _find_prompt_record(root: Path, prompt_id: str) -> Path | None:
    direct = [
        root / prompt_id / "probe.json",
        root / prompt_id / "lookahead.json",
        root / prompt_id / "record.json",
        root / prompt_id / "result.json",
        root / f"{prompt_id}.json",
    ]
    for candidate in direct:
        if candidate.is_file():
            return candidate
    for candidate in sorted(root.glob("*.json")):
        if candidate.name in _JSON_FILE_NAMES:
            continue
        try:
            document = load_json(candidate)
        except EvaluationError:
            continue
        if _record_prompt_id(document) == prompt_id:
            return candidate
    directory = root / prompt_id
    if directory.is_dir():
        for candidate in sorted(directory.glob("*.json")):
            if candidate.name == "artifacts.json":
                continue
            document = load_json(candidate)
            if _record_prompt_id(document) == prompt_id:
                return candidate
    return None


def _record_path_from_ledger(
    root: Path, entry: Mapping[str, object], prompt_id: str
) -> Path | None:
    record = entry.get("record")
    if not isinstance(record, str):
        return None
    candidate = Path(record)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvaluationError(f"ledger record path is unsafe: {prompt_id}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise EvaluationError(
            f"ledger record path escapes root: {prompt_id}"
        ) from error
    if not path.is_file():
        return None
    return path


def _record_prompt_id(raw: Mapping[str, object]) -> str | None:
    direct = raw.get("prompt_id")
    if isinstance(direct, str):
        return direct
    prompt = raw.get("prompt")
    if isinstance(prompt, dict) and isinstance(prompt.get("prompt_id"), str):
        return prompt["prompt_id"]
    return None


def _action_map(
    raw: Mapping[str, object], prompt_id: str
) -> dict[str, dict[str, object]]:
    value = raw.get("actions")
    if not isinstance(value, dict):
        raise EvaluationError(f"missing action map: {prompt_id}")
    action_map: dict[str, dict[str, object]] = {}
    for key, item in value.items():
        if not isinstance(item, dict):
            raise EvaluationError(f"action entry malformed: {prompt_id}")
        action_id = _action_id_from_payload(item, str(key))
        if action_id != key:
            raise EvaluationError(f"action map key differs: {prompt_id}")
        if action_id in action_map:
            raise EvaluationError(
                f"duplicate action probe: {prompt_id} {action_id}"
            )
        action_map[action_id] = item
    return action_map


def _action_id_from_payload(
    payload: Mapping[str, object], fallback: str | None
) -> str:
    action = payload.get("action")
    if isinstance(action, dict) and isinstance(action.get("action_id"), str):
        action_id = action["action_id"]
    elif isinstance(payload.get("action_id"), str):
        action_id = payload["action_id"]
    elif fallback is not None:
        action_id = fallback
    else:
        raise EvaluationError("action payload has no action ID")
    if action_id not in ACTION_IDS:
        raise EvaluationError(f"unknown action ID: {action_id}")
    return action_id


def _portable_boundary(
    value: Mapping[str, object], prompt_id: str
) -> dict[str, object]:
    """Normalize boundary evidence and exclude process-local fingerprints."""
    stable_keys = {
        "prompt_token_ids",
        "prompt_length",
        "generated_token_ids",
        "generated_count",
        "pending_token_id",
        "pending_generated_index",
        "logical_position",
        "attention_mask",
        "cache_lengths",
        "cache_bytes",
        "rng_fingerprint",
        "model_tensor_count",
    }
    excluded = {
        "state_fingerprint",
        "model_state_fingerprint",
        "validation_seconds",
    }
    keys = set(value)
    if keys != stable_keys and keys != (stable_keys | excluded):
        raise EvaluationError(f"boundary shape differs: {prompt_id}")
    normalized: dict[str, object] = {}
    for key in ("prompt_token_ids", "attention_mask"):
        normalized[key] = _portable_int_list(
            value[key], f"boundary {key} {prompt_id}"
        )
    for key in ("generated_token_ids", "cache_lengths"):
        normalized[key] = _portable_int_list(
            value[key], f"boundary {key} {prompt_id}"
        )
    for key in (
        "prompt_length",
        "generated_count",
        "pending_token_id",
        "pending_generated_index",
        "logical_position",
        "cache_bytes",
        "model_tensor_count",
    ):
        normalized[key] = _integer(value[key], f"boundary {key} {prompt_id}")
    normalized["rng_fingerprint"] = _sha256(
        value["rng_fingerprint"], f"boundary RNG fingerprint {prompt_id}"
    )
    return normalized


def _portable_int_list(value: object, label: str) -> list[int]:
    """Accept flat or one-level nested token arrays from old/new runners."""
    if not isinstance(value, list):
        raise EvaluationError(f"{label} must be an integer list")
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not value or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise EvaluationError(f"{label} must be an integer list")
    return list(value)


def _portable_tokenization(
    value: object, prompt_id: str
) -> dict[str, object]:
    """Normalize token IDs and retain locked tokenization metadata."""
    tokenization = _object(value, f"tokenization {prompt_id}")
    expected_keys = {"input_ids", "input_length", "chat_template_verified"}
    if set(tokenization) != expected_keys:
        raise EvaluationError(f"tokenization shape differs: {prompt_id}")
    input_ids = _portable_int_list(
        tokenization["input_ids"], f"tokenization input IDs {prompt_id}"
    )
    input_length = _integer(
        tokenization["input_length"], f"tokenization input length {prompt_id}"
    )
    if input_length != len(input_ids):
        raise EvaluationError(f"tokenization length differs: {prompt_id}")
    if tokenization["chat_template_verified"] is not True:
        raise EvaluationError(f"tokenization template differs: {prompt_id}")
    return {
        "input_ids": input_ids,
        "input_length": input_length,
        "chat_template_verified": True,
    }


def _row_from_probe(
    prompt_id: str,
    action_id: str,
    payload: Mapping[str, object],
    record: Mapping[str, object],
    path: Path,
) -> LookaheadRow:
    result = payload
    if isinstance(payload.get("lookahead"), dict):
        result = payload["lookahead"]
    if result.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise EvaluationError(f"unsupported H8 probe schema: {prompt_id}")
    if result.get("passed") is not True:
        raise EvaluationError(
            f"H8 probe did not pass: {prompt_id} {action_id}"
        )
    checks = result.get("checks")
    if not isinstance(checks, dict) or not all(
        value is True for value in checks.values()
    ):
        raise EvaluationError(
            f"H8 probe checks failed: {prompt_id} {action_id}"
        )
    steps = result.get("steps")
    if (
        not isinstance(steps, list)
        or not 1 <= len(steps) <= MAX_LOOKAHEAD_STEPS
    ):
        raise EvaluationError(f"H8 steps malformed: {prompt_id} {action_id}")
    realized = result.get("realized_steps", len(steps))
    if realized != len(steps):
        raise EvaluationError(f"H8 realized step count differs: {prompt_id}")
    first = _object(steps[0], "H8 step")
    probe = _object(first.get("probe"), "H8 distribution probe")
    probe_action = _object(probe.get("action"), "H8 probe action")
    if probe_action.get("action_id") != action_id:
        raise EvaluationError(
            f"H8 probe action lineage mismatch: {prompt_id}"
        )
    for index, raw_step in enumerate(steps):
        step = _object(raw_step, "H8 step")
        if step.get("output_index") != FIRST_OUTPUT_INDEX + index:
            raise EvaluationError(f"H8 output index mismatch: {prompt_id}")
        if not isinstance(step.get("reference_argmax_is_eos"), bool):
            raise EvaluationError(f"H8 EOS flag is malformed: {prompt_id}")
        if (
            step.get("reference_argmax_is_eos") is True
            and index != len(steps) - 1
        ):
            raise EvaluationError(
                f"H8 continues after reference EOS: {prompt_id}"
            )
        if not isinstance(step.get("probe"), dict):
            raise EvaluationError(f"H8 step probe missing: {prompt_id}")
    action = _object(result.get("action"), "H8 action")
    if action.get("action_id") != action_id:
        raise EvaluationError(f"H8 action lineage mismatch: {prompt_id}")
    ratio = _finite(action.get("removal_fraction"), "removal fraction")
    if not math.isclose(ratio, ACTION_RATIOS[action_id], abs_tol=1e-12):
        raise EvaluationError(f"H8 action ratio mismatch: {prompt_id}")
    token_ids = _int_tuple(
        result.get("forced_input_token_ids"),
        [
            _integer(
                _object(step, "H8 step").get("input_token_id"), "input token"
            )
            for step in steps
        ],
        "forced input token IDs",
    )
    ref_ids = _int_tuple(
        result.get("reference_argmax_token_ids"),
        [
            _integer(
                _object(step, "H8 step").get("reference_argmax"),
                "reference argmax",
            )
            for step in steps
        ],
        "reference argmax IDs",
    )
    step_input_ids = tuple(
        _integer(
            _object(step, "H8 step").get("input_token_id"), "input token"
        )
        for step in steps
    )
    step_ref_ids = tuple(
        _integer(
            _object(step, "H8 step").get("reference_argmax"),
            "reference argmax",
        )
        for step in steps
    )
    if token_ids != step_input_ids or ref_ids != step_ref_ids:
        raise EvaluationError(f"H8 token summaries differ: {prompt_id}")
    input_positions = result.get("input_positions")
    if input_positions is not None:
        observed_positions = _int_tuple(
            input_positions, (), "input positions"
        )
        step_positions = tuple(
            _integer(
                _object(step, "H8 step").get("input_position"),
                "input position",
            )
            for step in steps
        )
        if observed_positions != step_positions:
            raise EvaluationError(f"H8 input positions differ: {prompt_id}")
        boundary = record.get("boundary")
        if (
            isinstance(boundary, dict)
            and boundary.get("logical_position") is not None
        ):
            logical_position = _integer(
                boundary["logical_position"], "boundary logical position"
            )
            if observed_positions != tuple(
                logical_position + index for index in range(len(steps))
            ):
                raise EvaluationError(
                    f"H8 boundary positions differ: {prompt_id}"
                )
    output_indices = result.get("output_indices")
    if output_indices is not None and output_indices != [
        FIRST_OUTPUT_INDEX + index for index in range(len(steps))
    ]:
        raise EvaluationError(f"H8 output summary differs: {prompt_id}")
    eos_positions = [
        index
        for index, step in enumerate(steps)
        if _object(step, "H8 step").get("reference_argmax_is_eos") is True
    ]
    eos_position = eos_positions[0] if eos_positions else None
    explicit_eos = result.get("reference_eos_position")
    if explicit_eos is not None and explicit_eos != eos_position:
        raise EvaluationError(f"H8 EOS position differs: {prompt_id}")
    source = result.get("source")
    source_object = _object(source, "H8 source")
    state = _string(
        source_object.get("boundary_state_fingerprint_before"),
        "reference state fingerprint",
    )
    if (
        source_object.get("source_cache_preserved") is not True
        or source_object.get("source_rng_preserved") is not True
        or source_object.get("model_state_preserved") is not True
    ):
        raise EvaluationError(f"H8 source preservation failed: {prompt_id}")
    boundary = _object(record.get("boundary"), "probe boundary")
    boundary_stable = _portable_boundary(boundary, prompt_id)
    boundary_cache_fingerprint = _sha256(
        record.get("boundary_cache_fingerprint"),
        f"probe boundary cache fingerprint {prompt_id}",
    )
    prompt_token_count, cache_size = _boundary_features(record, payload)
    reference_entropy = _finite(
        probe.get("reference_entropy"), "reference entropy"
    )
    reference_margin = _finite(
        probe.get("reference_top2_margin"), "reference margin"
    )
    action_entropy = _finite(probe.get("action_entropy"), "action entropy")
    action_margin = _finite(probe.get("action_top2_margin"), "action margin")
    argmax_match = probe.get("argmax_match")
    if not isinstance(argmax_match, bool):
        raise EvaluationError(f"argmax_match is not boolean: {prompt_id}")
    immediate_js = _finite(
        probe.get("js_divergence"), "immediate JS", lower=0.0
    )
    delayed_values = [
        _finite(
            _object(_object(step, "H8 step").get("probe"), "step probe").get(
                "js_divergence"
            ),
            "delayed JS",
            lower=0.0,
        )
        for step in steps[1:]
    ]
    mean_delayed = float(np.mean(delayed_values)) if delayed_values else 0.0
    has_delayed = int(bool(delayed_values))
    explicit_mean = result.get("mean_delayed_js")
    if explicit_mean is not None and not math.isclose(
        _finite(explicit_mean, "mean delayed JS", lower=0.0),
        mean_delayed,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise EvaluationError(f"mean delayed JS mismatch: {prompt_id}")
    reference_hash = canonical_sha256(
        {
            "forced_input_token_ids": list(token_ids),
            "reference_argmax_token_ids": list(ref_ids),
            "reference_eos_position": eos_position,
            "reference_state_fingerprint": state,
        }
    )
    explicit_reference_hash = result.get("reference_probe_hash")
    if explicit_reference_hash is None:
        explicit_reference_hash = record.get("reference_probe_hash")
    if explicit_reference_hash is not None:
        declared_reference_hash = _string(
            explicit_reference_hash, "reference probe hash"
        )
        if declared_reference_hash != reference_hash:
            raise EvaluationError(
                f"reference probe hash differs: {prompt_id}"
            )
    return LookaheadRow(
        prompt_id=prompt_id,
        action_id=action_id,
        removal_fraction=ratio,
        prompt_token_count=prompt_token_count,
        decision_index32=float(BOUNDARY_INDEX),
        pre_action_cache_size=cache_size,
        reference_entropy=reference_entropy,
        reference_top2_margin=reference_margin,
        action_reference_entropy_delta=action_entropy - reference_entropy,
        action_reference_margin_delta=action_margin - reference_margin,
        argmax_match=argmax_match,
        immediate_js=immediate_js,
        lookahead_steps=len(steps),
        has_delayed=has_delayed,
        mean_delayed_js=mean_delayed,
        reference_input_token_ids=token_ids,
        reference_argmax_token_ids=ref_ids,
        reference_eos_position=eos_position,
        reference_state_fingerprint=state,
        boundary_cache_fingerprint=boundary_cache_fingerprint,
        boundary_stable=boundary_stable,
        reference_probe_hash=reference_hash,
        probe_sha256=file_sha256(path),
        probe_path=str(path.resolve()),
    )


def _validate_pair(rows: Sequence[LookaheadRow]) -> None:
    if len(rows) != 2 or {row.action_id for row in rows} != set(ACTION_IDS):
        raise EvaluationError(
            f"prompt {rows[0].prompt_id} is not a full pair"
        )
    first = rows[0]
    for row in rows[1:]:
        if (
            row.reference_input_token_ids != first.reference_input_token_ids
            or row.lookahead_steps != first.lookahead_steps
            or row.reference_eos_position != first.reference_eos_position
            or row.reference_state_fingerprint
            != first.reference_state_fingerprint
            or row.boundary_cache_fingerprint
            != first.boundary_cache_fingerprint
            or row.boundary_stable != first.boundary_stable
            or row.reference_probe_hash != first.reference_probe_hash
        ):
            raise EvaluationError(
                f"shared H8 boundary differs: {first.prompt_id}"
            )


def _ordered_rows(
    rows: Sequence[LookaheadRow], prompt_ids: Sequence[str]
) -> list[LookaheadRow]:
    grouped: dict[str, list[LookaheadRow]] = defaultdict(list)
    for row in rows:
        grouped[row.prompt_id].append(row)
    ordered: list[LookaheadRow] = []
    for prompt_id in prompt_ids:
        group = grouped.get(prompt_id)
        if group is None:
            continue
        _validate_pair(group)
        ordered.extend(
            sorted(group, key=lambda row: ACTION_IDS.index(row.action_id))
        )
    return ordered


def _validate_train_rows(
    rows: Sequence[LookaheadRow],
    ids: Sequence[str],
    outcomes: Mapping[str, TrainingOutcome],
) -> None:
    if len(rows) != TRAIN_ACTION_ROW_COUNT or {
        row.prompt_id for row in rows
    } != set(ids):
        raise EvaluationError(
            "training probes do not contain exactly 120 paired prompts"
        )
    if set(outcomes) != set(ids):
        raise EvaluationError("training outcome IDs do not match probes")
    for row in rows:
        if row.prompt_id not in outcomes:
            raise EvaluationError(
                f"missing training outcome: {row.prompt_id}"
            )
        outcome = outcomes[row.prompt_id]
        if (
            outcome.boundary_stable is None
            or outcome.boundary_cache_fingerprint is None
            or row.boundary_stable != outcome.boundary_stable
            or row.boundary_cache_fingerprint
            != outcome.boundary_cache_fingerprint
        ):
            raise EvaluationError(
                "training boundary portability proof differs: "
                f"{row.prompt_id}"
            )
        for value in row.feature_values(BLOCK_FEATURES["B4_L"]):
            if not math.isfinite(value):
                raise EvaluationError(
                    f"nonfinite training feature: {row.prompt_id}"
                )


def _boundary_features(
    record: Mapping[str, object], payload: Mapping[str, object]
) -> tuple[float, float]:
    tokenization = record.get("tokenization")
    boundary = record.get("boundary")
    if (
        isinstance(tokenization, dict)
        and tokenization.get("input_length") is not None
    ):
        prompt_count = _finite(
            tokenization.get("input_length"), "prompt token count", lower=1.0
        )
    elif isinstance(boundary, dict):
        prompt_count = _finite(
            boundary.get("prompt_length"), "prompt token count", lower=1.0
        )
    else:
        prompt_count = _finite(
            payload.get("prompt_token_count"), "prompt token count", lower=1.0
        )
    if isinstance(boundary, dict):
        cache_size = _finite(
            boundary.get("cache_bytes"), "pre-action cache size", lower=0.0
        )
    else:
        memory = payload.get("memory")
        memory_object = (
            _object(memory, "probe memory")
            if isinstance(memory, dict)
            else {}
        )
        cache_size = _finite(
            memory_object.get("reference_cache_bytes"),
            "pre-action cache size",
            lower=0.0,
        )
    return prompt_count, cache_size


def _validate_prompt_lineage(
    raw: Mapping[str, object],
    manifest_record: Mapping[str, object],
    prompt_id: str,
) -> None:
    embedded = raw.get("prompt")
    if embedded is None:
        return
    embedded_object = _object(embedded, "probe prompt")
    if embedded_object.get("prompt_id") != prompt_id:
        raise EvaluationError(f"probe prompt lineage differs: {prompt_id}")
    if embedded_object.get("prompt_text_utf8_sha256") != manifest_record.get(
        "prompt_text_utf8_sha256"
    ):
        raise EvaluationError(
            f"probe source prompt hash differs: {prompt_id}"
        )


def _validate_probe_record_shape(
    raw: Mapping[str, object],
    prompt_id: str,
    manifest: LoadedPrompts,
    collection_lock: Mapping[str, object],
    checkpoint: Mapping[str, object],
) -> None:
    required = {
        "schema_version",
        "phase",
        "prompt_id",
        "status",
        "prompt",
        "prompt_manifest_sha256",
        "prompt_manifest_file_sha256",
        "provenance",
        "configuration",
        "boundary",
        "boundary_cache_fingerprint",
        "lookahead",
        "actions",
        "action_rows",
        "checks",
        "collection_seconds",
        "outcomes_collected",
    }
    if set(raw) != required:
        raise EvaluationError(f"probe record shape differs: {prompt_id}")
    if (
        raw.get("schema_version")
        != "herald_v3.lookahead_collection_record.v1"
    ):
        raise EvaluationError(f"probe record schema differs: {prompt_id}")
    if raw.get("status") != "eligible" or raw.get("phase") != checkpoint.get(
        "phase"
    ):
        raise EvaluationError(f"probe record status differs: {prompt_id}")
    if raw.get("prompt_manifest_sha256") != manifest.manifest_sha256:
        raise EvaluationError(f"probe record manifest differs: {prompt_id}")
    lock_manifest = _object(
        collection_lock.get("manifest"), "collection manifest"
    )
    if raw.get("prompt_manifest_file_sha256") != lock_manifest.get(
        "file_sha256"
    ):
        raise EvaluationError(
            f"probe record manifest file differs: {prompt_id}"
        )
    provenance = _object(raw.get("provenance"), "probe provenance")
    if provenance.get("prompt_id") != prompt_id or provenance.get(
        "prompt_text_utf8_sha256"
    ) != manifest.records[prompt_id].get("prompt_text_utf8_sha256"):
        raise EvaluationError(f"probe provenance differs: {prompt_id}")
    if raw.get("configuration") != collection_lock.get("configuration"):
        raise EvaluationError(f"probe configuration differs: {prompt_id}")
    if raw.get("outcomes_collected") is not False:
        raise EvaluationError(f"probe record contains outcomes: {prompt_id}")
    checks = _object(raw.get("checks"), "probe checks")
    if (
        not checks
        or checks.get("outcomes_collected") is not False
        or any(
            value is not True
            for key, value in checks.items()
            if key != "outcomes_collected"
        )
    ):
        raise EvaluationError(f"probe checks failed: {prompt_id}")
    _finite(raw.get("collection_seconds"), "collection seconds", lower=0.0)
    _object(raw.get("boundary"), "probe boundary")
    _sha256(
        raw.get("boundary_cache_fingerprint"),
        f"probe boundary cache fingerprint {prompt_id}",
    )
    lookahead = _object(raw.get("lookahead"), "probe lookahead")
    if set(lookahead) != {"actions", "noop_controls", "shared_reference"}:
        raise EvaluationError(f"probe lookahead shape differs: {prompt_id}")
    if not isinstance(lookahead["noop_controls"], dict) or set(
        lookahead["noop_controls"]
    ) != {"knorm:0"}:
        raise EvaluationError(f"probe no-op control differs: {prompt_id}")
    if lookahead["actions"] != raw.get("actions"):
        raise EvaluationError(f"probe action map differs: {prompt_id}")


def _validate_shared_reference(
    raw: Mapping[str, object], rows: Sequence[LookaheadRow], prompt_id: str
) -> None:
    lookahead = _object(raw["lookahead"], "probe lookahead")
    shared = _object(lookahead.get("shared_reference"), "shared H8 reference")
    required = {
        "output_indices",
        "forced_input_token_ids",
        "input_positions",
        "reference_argmax_token_ids",
        "reference_argmax_is_eos",
        "L",
        "reference_eos_output_index",
        "reference_state_fingerprint",
        "reference_probe_sha256",
    }
    if set(shared) != required:
        raise EvaluationError(
            f"shared H8 reference shape differs: {prompt_id}"
        )
    first = rows[0]
    if shared["L"] != first.lookahead_steps:
        raise EvaluationError(f"shared H8 L differs: {prompt_id}")
    if shared["forced_input_token_ids"] != list(
        first.reference_input_token_ids
    ):
        raise EvaluationError(f"shared H8 input tokens differ: {prompt_id}")
    if shared["reference_argmax_token_ids"] != list(
        first.reference_argmax_token_ids
    ):
        raise EvaluationError(
            f"shared H8 reference tokens differ: {prompt_id}"
        )
    if (
        shared["reference_state_fingerprint"]
        != first.reference_state_fingerprint
    ):
        raise EvaluationError(f"shared H8 state differs: {prompt_id}")
    eos_output = (
        FIRST_OUTPUT_INDEX + first.reference_eos_position
        if first.reference_eos_position is not None
        else None
    )
    if shared["reference_eos_output_index"] != eos_output:
        raise EvaluationError(f"shared H8 EOS differs: {prompt_id}")
    if shared["output_indices"] != [
        FIRST_OUTPUT_INDEX + index for index in range(first.lookahead_steps)
    ]:
        raise EvaluationError(f"shared H8 output indices differ: {prompt_id}")
    positions = shared["input_positions"]
    if (
        not isinstance(positions, list)
        or len(positions) != first.lookahead_steps
    ):
        raise EvaluationError(
            f"shared H8 input positions differ: {prompt_id}"
        )
    if shared["reference_argmax_is_eos"] != [
        index == first.reference_eos_position
        for index in range(first.lookahead_steps)
    ]:
        raise EvaluationError(f"shared H8 EOS flags differ: {prompt_id}")
    _string(shared["reference_probe_sha256"], "shared H8 reference hash")
    if any(
        row.reference_input_token_ids != first.reference_input_token_ids
        or row.reference_argmax_token_ids != first.reference_argmax_token_ids
        or row.reference_eos_position != first.reference_eos_position
        or row.reference_state_fingerprint
        != first.reference_state_fingerprint
        or row.boundary_cache_fingerprint != first.boundary_cache_fingerprint
        or row.boundary_stable != first.boundary_stable
        for row in rows[1:]
    ):
        raise EvaluationError(f"shared H8 pair differs: {prompt_id}")


def _validate_action_rows(
    raw: Mapping[str, object], rows: Sequence[LookaheadRow], prompt_id: str
) -> None:
    values = raw.get("action_rows")
    if not isinstance(values, list) or len(values) != len(ACTION_IDS):
        raise EvaluationError(
            f"probe action_rows are incomplete: {prompt_id}"
        )
    by_action = {
        _string(
            _object(value, "probe action row").get("action_id"), "action ID"
        ): value
        for value in values
    }
    if set(by_action) != set(ACTION_IDS):
        raise EvaluationError(f"probe action_rows differ: {prompt_id}")
    for row in rows:
        item = _object(by_action[row.action_id], "probe action row")
        expected = {
            "action_id",
            "removal_fraction",
            "prompt_token_count",
            "decision_index32",
            "pre_action_cache_size",
            "immediate_js",
            "js_divergence",
            "mean_delayed_js",
            "L",
            "has_delayed",
            "reference_entropy",
            "reference_top2_margin",
            "action_reference_entropy_delta",
            "action_reference_margin_delta",
            "argmax_match",
            "reference_argmax",
            "action_argmax",
            "reference_state_fingerprint",
            "reference_probe_sha256",
        }
        if set(item) != expected:
            raise EvaluationError(
                f"probe action row shape differs: {prompt_id}"
            )
        checks = {
            "removal_fraction": row.removal_fraction,
            "prompt_token_count": row.prompt_token_count,
            "decision_index32": row.decision_index32,
            "pre_action_cache_size": row.pre_action_cache_size,
            "immediate_js": row.immediate_js,
            "js_divergence": row.immediate_js,
            "mean_delayed_js": row.mean_delayed_js,
            "L": row.lookahead_steps,
            "has_delayed": row.has_delayed,
            "reference_entropy": row.reference_entropy,
            "reference_top2_margin": row.reference_top2_margin,
            "action_reference_entropy_delta": (
                row.action_reference_entropy_delta
            ),
            "action_reference_margin_delta": (
                row.action_reference_margin_delta
            ),
            "argmax_match": row.argmax_match,
            "reference_state_fingerprint": row.reference_state_fingerprint,
        }
        for key, actual in checks.items():
            if isinstance(actual, float):
                if not math.isclose(
                    _finite(item[key], key),
                    actual,
                    abs_tol=1e-12,
                    rel_tol=0.0,
                ):
                    raise EvaluationError(
                        f"probe action row value differs: {prompt_id}"
                    )
            elif item[key] != actual:
                raise EvaluationError(
                    f"probe action row value differs: {prompt_id}"
                )
        _string(item["reference_probe_sha256"], "action reference hash")


def _reject_embedded_labels(
    value: object, prompt_id: str, key_path: str = ""
) -> None:
    forbidden = {
        "target_loose",
        "target_strict",
        "d_loose",
        "d_strict",
        "scores",
        "outcome",
        "labels",
    }
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in forbidden:
                raise EvaluationError(
                    f"probe embeds outcome labels: {prompt_id}"
                )
            _reject_embedded_labels(nested, prompt_id, f"{key_path}.{key}")
    elif isinstance(value, list):
        for nested in value:
            _reject_embedded_labels(nested, prompt_id, key_path)


def _resolve_pilot_root(
    path: Path, reuse: Mapping[str, object], explicit: str | Path | None
) -> Path:
    if explicit is not None:
        root = Path(explicit)
    else:
        relative = _string(
            reuse.get("collection_root_relative_to_project"),
            "collection root",
        )
        root = path.parent.parent / relative
        if not root.is_dir():
            root = PROJECT_ROOT / relative
    if not root.is_dir():
        raise EvaluationError(
            f"pilot collection root is not a directory: {root}"
        )
    return root


def _validate_reuse_prompt(
    row: Mapping[str, object], manifest: Mapping[str, object], prompt_id: str
) -> None:
    prompt = _object(row.get("prompt"), "reuse prompt")
    if prompt.get("prompt_id") != prompt_id or prompt.get(
        "prompt_text_utf8_sha256"
    ) != manifest.get("prompt_text_utf8_sha256"):
        raise EvaluationError(
            f"training reuse prompt lineage differs: {prompt_id}"
        )
    tokenization = _portable_tokenization(row.get("tokenization"), prompt_id)
    input_ids = tokenization["input_ids"]
    boundary = _object(row.get("boundary"), "reuse boundary")
    _portable_boundary(boundary, prompt_id)
    _sha256(
        row.get("boundary_cache_fingerprint"),
        f"reuse boundary cache fingerprint {prompt_id}",
    )
    if (
        boundary.get("generated_count") != 32
        or boundary.get("pending_generated_index") != 31
    ):
        raise EvaluationError(f"training reuse boundary differs: {prompt_id}")
    portable_boundary = _portable_boundary(boundary, prompt_id)
    if portable_boundary["prompt_token_ids"] != input_ids:
        raise EvaluationError(
            f"training reuse prompt token IDs differ: {prompt_id}"
        )


def _outcome_from_run(
    document: Mapping[str, object],
    prompt_id: str,
    reuse: Mapping[str, object],
    run_path: Path,
    artifacts: Path,
) -> TrainingOutcome:
    if (
        document.get("passed") is not True
        or document.get("status") != "completed"
    ):
        raise EvaluationError(
            f"reused pilot run is not accepted: {prompt_id}"
        )
    if (
        document.get("prompt_manifest") is not None
        and reuse.get("prompt") is None
    ):
        raise EvaluationError(
            f"reused pilot metadata is incomplete: {prompt_id}"
        )
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise EvaluationError(
            f"reused pilot result count differs: {prompt_id}"
        )
    result = _object(results[0], "reused pilot result")
    prompt = _object(result.get("prompt"), "reused result prompt")
    if prompt.get("prompt_id") != prompt_id:
        raise EvaluationError(f"reused result prompt differs: {prompt_id}")
    for key in ("configuration", "model", "environment", "source_manifest"):
        if document.get(key) != reuse.get(key):
            raise EvaluationError(
                f"reused {key} differs from training-reuse metadata: "
                f"{prompt_id}"
            )
    run_tokenization = _portable_tokenization(
        result.get("tokenization"), prompt_id
    )
    reuse_tokenization = _portable_tokenization(
        reuse.get("tokenization"), prompt_id
    )
    if (
        prompt != reuse.get("prompt")
        or run_tokenization != reuse_tokenization
    ):
        raise EvaluationError(
            f"reused prompt/tokenization differs: {prompt_id}"
        )
    acceptance = _object(result.get("acceptance"), "reused acceptance")
    run_boundary = _object(acceptance.get("boundary"), "reused boundary")
    reuse_boundary = _object(reuse.get("boundary"), "reuse boundary")
    run_stable = _portable_boundary(run_boundary, prompt_id)
    reuse_stable = _portable_boundary(reuse_boundary, prompt_id)
    if run_stable != reuse_stable:
        raise EvaluationError(
            f"reused portable boundary differs: {prompt_id}"
        )
    expected_cache = _sha256(
        reuse.get("boundary_cache_fingerprint"),
        f"reuse boundary cache fingerprint {prompt_id}",
    )
    actual_cache = _boundary_cache_from_action_arms(
        acceptance.get("action_arms"), prompt_id
    )
    if actual_cache != expected_cache:
        raise EvaluationError(f"reused boundary cache differs: {prompt_id}")
    scores = _object(result.get("scores"), "reused scores")
    reference = _object(scores.get("reference"), "reused reference score")
    reference_loose = _finite(
        reference.get("loose"), "reference loose score", lower=0.0, upper=1.0
    )
    reference_strict = _finite(
        reference.get("strict"),
        "reference strict score",
        lower=0.0,
        upper=1.0,
    )
    actions = _object(scores.get("actions"), "reused action scores")
    targets: dict[str, dict[str, float]] = {}
    for action_id in ACTION_IDS:
        score = _object(actions.get(action_id), f"reused score {action_id}")
        action_score = _object(score.get("action"), "reused action score")
        action_loose = _finite(
            action_score.get("loose"),
            "action loose score",
            lower=0.0,
            upper=1.0,
        )
        action_strict = _finite(
            action_score.get("strict"),
            "action strict score",
            lower=0.0,
            upper=1.0,
        )
        d_loose = _finite(score.get("d_loose"), "signed loose target")
        d_strict = _finite(score.get("d_strict"), "signed strict target")
        if not math.isclose(
            d_loose,
            reference_loose - action_loose,
            abs_tol=1e-12,
            rel_tol=0.0,
        ) or not math.isclose(
            d_strict,
            reference_strict - action_strict,
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise EvaluationError(
                f"signed target mismatch: {prompt_id} {action_id}"
            )
        targets[action_id] = {
            "loose": d_loose,
            "strict": d_strict,
        }
    return TrainingOutcome(
        targets=targets,
        source_hashes={
            "run.json": file_sha256(run_path),
            "artifacts.json": file_sha256(artifacts),
        },
        boundary_cache_fingerprint=expected_cache,
        boundary_stable=reuse_stable,
    )


def _boundary_cache_from_action_arms(value: object, prompt_id: str) -> str:
    if not isinstance(value, list) or len(value) != len(ACTION_IDS):
        raise EvaluationError(f"action arms are incomplete: {prompt_id}")
    found: dict[str, str] = {}
    for arm_value in value:
        arm = _object(arm_value, "action arm")
        action = _object(arm.get("action"), "action arm action")
        action_id = _string(action.get("action_id"), "action arm ID")
        if action_id not in ACTION_IDS or action_id in found:
            raise EvaluationError(f"action arms are malformed: {prompt_id}")
        compression = _object(arm.get("compression"), "action compression")
        found[action_id] = _sha256(
            compression.get("before_fingerprint"),
            f"action cache fingerprint {prompt_id}",
        )
    if set(found) != set(ACTION_IDS) or len(set(found.values())) != 1:
        raise EvaluationError(
            f"action arms cache fingerprints differ: {prompt_id}"
        )
    return found[ACTION_IDS[0]]


def _manifest_seal_value(
    path: str | Path, manifest: LoadedPrompts
) -> dict[str, object]:
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": manifest.file_sha256,
        "manifest_sha256": manifest.manifest_sha256,
        "prompt_ids": list(manifest.ids),
    }


def _runtime_versions() -> dict[str, object]:
    versions: dict[str, str] = {}
    for package in ("numpy", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _sealed_predictions(
    seal: Mapping[str, object],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    raw = seal.get("predictions")
    if not isinstance(raw, dict):
        raise EvaluationError("sealed predictions are malformed")
    prediction_ids = seal.get("prediction_ids")
    if (
        not isinstance(prediction_ids, list)
        or not prediction_ids
        or any(
            not isinstance(item, str) or not item for item in prediction_ids
        )
        or len(set(prediction_ids)) != len(prediction_ids)
    ):
        raise EvaluationError("sealed prediction IDs are malformed")
    if set(raw) != set(prediction_ids):
        raise EvaluationError(
            "sealed predictions do not match prediction ID set"
        )
    normalized: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    # JSON output is written with sort_keys=True, so the predictions mapping
    # cannot carry the manifest order.  prediction_ids is the explicit order
    # authority for every fit, collector, and scoring path.
    for prompt_id in prediction_ids:
        action_map = raw[prompt_id]
        if not isinstance(action_map, dict) or set(action_map) != set(
            ACTION_IDS
        ):
            raise EvaluationError(
                f"sealed action pair malformed: {prompt_id}"
            )
        normalized[prompt_id] = {}
        for action_id, model_map in action_map.items():
            if not isinstance(model_map, dict) or set(model_map) != set(
                MODEL_NAMES
            ):
                raise EvaluationError(
                    f"sealed model map is incomplete: {prompt_id}"
                )
            normalized[prompt_id][action_id] = {}
            for model_name, targets in model_map.items():
                if not isinstance(targets, dict) or set(targets) != {
                    "loose",
                    "strict",
                }:
                    raise EvaluationError(
                        f"sealed model malformed: {prompt_id}"
                    )
                normalized[prompt_id][action_id][model_name] = {
                    target: _finite(targets.get(target), "sealed prediction")
                    for target in ("loose", "strict")
                }
    return normalized


def _validate_outcome_provenance(
    outcomes: Mapping[str, object], seal: Mapping[str, object]
) -> None:
    if outcomes.get("protocol_lock_sha256") != _string(
        _object(seal["protocol_lock"], "sealed protocol lock").get("sha256"),
        "sealed protocol lock hash",
    ):
        raise EvaluationError("outcomes do not name the sealed protocol lock")
    if outcomes.get("prediction_seal_sha256") != _string(
        seal.get("seal_sha256"), "prediction seal hash"
    ):
        raise EvaluationError("outcomes do not name this prediction seal")
    provenance = outcomes.get("scorer_provenance")
    if not isinstance(provenance, dict):
        raise EvaluationError("outcomes lack scorer provenance")
    lock_doc = seal.get("locked_scorer_provenance")
    if lock_doc is not None and provenance != lock_doc:
        raise EvaluationError("outcome scorer provenance differs from lock")


def _load_test_outcomes(
    outcomes: Mapping[str, object],
    seal: Mapping[str, object],
    *,
    outcomes_path: Path,
    expected_seal_file_sha256: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if outcomes.get("schema_version") != ("herald_v3.lookahead_outcomes.v1"):
        raise EvaluationError("unsupported outcomes schema")
    required_metadata = {
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
    }
    if not required_metadata <= set(outcomes):
        raise EvaluationError("outcome index metadata is incomplete")
    if outcomes.get("status") != "completed":
        raise EvaluationError("outcome index is not complete")
    if (
        outcomes.get("prediction_seal_file_sha256")
        != expected_seal_file_sha256
    ):
        raise EvaluationError("outcome index prediction seal file differs")
    seal_lock = _object(seal.get("protocol_lock"), "sealed protocol lock")
    lock_sha = _string(seal_lock.get("sha256"), "sealed protocol lock hash")
    if outcomes.get("protocol_lock_sha256") != lock_sha:
        raise EvaluationError("outcome index protocol lock differs")
    inputs = _object(seal.get("inputs"), "sealed inputs")
    test_ref = _object(inputs.get("test_manifest"), "sealed test manifest")
    if outcomes.get("test_manifest_sha256") != test_ref.get("file_sha256"):
        raise EvaluationError("outcome index test manifest differs")
    if outcomes.get("scorer_provenance") != seal.get(
        "locked_scorer_provenance"
    ):
        raise EvaluationError("outcome index scorer provenance differs")
    if outcomes.get("model") != seal.get("locked_model"):
        raise EvaluationError("outcome index model differs")
    source_manifest = _object(
        seal.get("locked_source_manifest"), "sealed source manifest"
    )
    if outcomes.get("source_manifest") != source_manifest:
        raise EvaluationError("outcome index source manifest differs")
    generation = _object(
        _object(seal.get("locked_software"), "sealed software").get(
            "generation"
        ),
        "sealed generation software",
    )
    if outcomes.get("dependencies") != generation:
        raise EvaluationError("outcome index dependencies differ")
    if outcomes.get("code_sha256") != source_manifest.get(
        "scripts/collect_lookahead_outcomes.py"
    ):
        raise EvaluationError("outcome collector source hash differs")
    if outcomes.get("runner_code_sha256") != source_manifest.get(
        "src/herald_v3/engineering/runner.py"
    ):
        raise EvaluationError("outcome runner source hash differs")
    raw = outcomes.get("ordered_rows")
    if not isinstance(raw, list):
        raise EvaluationError("outcomes ordered_rows must be a list")
    predictions = _sealed_predictions(seal)
    prediction_ids = seal["prediction_ids"]
    if not isinstance(prediction_ids, list):
        raise EvaluationError("sealed prediction IDs are malformed")
    h8_index = _sealed_h8_index(seal)
    if len(raw) != len(prediction_ids):
        raise EvaluationError("outcome rows do not cover sealed predictions")
    _validate_official_parity(outcomes, outcomes_path.parent)
    labels: dict[str, dict[str, dict[str, float]]] = {}
    for value in raw:
        row = _object(value, "outcome row")
        prompt_id = _string(row.get("prompt_id"), "outcome prompt_id")
        if prompt_id not in predictions or prompt_id in labels:
            raise EvaluationError(
                f"unexpected or duplicate outcome: {prompt_id}"
            )
        required_row = {
            "prompt_id",
            "artifact_directory",
            "artifacts_sha256",
            "run_sha256",
            "actions",
        }
        if set(row) != required_row:
            raise EvaluationError(f"outcome row shape differs: {prompt_id}")
        action_value = row["actions"]
        if not isinstance(action_value, dict) or set(action_value) != set(
            ACTION_IDS
        ):
            raise EvaluationError(
                f"outcome action pair malformed: {prompt_id}"
            )
        artifact_directory = _string(
            row["artifact_directory"],
            f"outcome artifact directory {prompt_id}",
        )
        artifact_root = _safe_outcome_artifact_directory(
            outcomes_path.parent, artifact_directory, prompt_id
        )
        artifacts_path = artifact_root / "artifacts.json"
        if not artifacts_path.is_file():
            raise EvaluationError(
                f"outcome artifacts are missing: {prompt_id}"
            )
        artifacts_hash = _string(
            row["artifacts_sha256"], f"outcome artifacts hash {prompt_id}"
        )
        if file_sha256(artifacts_path) != artifacts_hash:
            raise EvaluationError(
                f"outcome artifacts hash differs: {prompt_id}"
            )
        artifacts = load_json(artifacts_path)
        artifact_hashes = artifacts.get("sha256")
        if not isinstance(artifact_hashes, dict):
            raise EvaluationError(
                f"outcome artifact index is malformed: {prompt_id}"
            )
        run_hash = _string(row["run_sha256"], f"outcome run hash {prompt_id}")
        if artifact_hashes.get("run.json") != run_hash:
            raise EvaluationError(
                f"outcome run hash is not artifact-bound: {prompt_id}"
            )
        for name, expected_hash in artifact_hashes.items():
            filename = _string(name, f"outcome artifact filename {prompt_id}")
            if Path(filename).name != filename:
                raise EvaluationError(
                    f"unsafe outcome artifact filename: {prompt_id}"
                )
            expected_text = _string(
                expected_hash, f"outcome artifact hash {prompt_id}"
            )
            artifact = artifact_root / filename
            if (
                not artifact.is_file()
                or file_sha256(artifact) != expected_text
            ):
                raise EvaluationError(
                    f"outcome artifact hash differs: {prompt_id}/{filename}"
                )
        run_path = artifact_root / "run.json"
        if file_sha256(run_path) != run_hash:
            raise EvaluationError(f"outcome run hash differs: {prompt_id}")
        run_document = load_json(run_path)
        _verify_official_run(run_path, prompt_id, seal)
        raw_scores = _validate_raw_outcome_run(
            run_document, prompt_id, seal, h8_index
        )
        labels[prompt_id] = {}
        for action_id in ACTION_IDS:
            item = _object(action_value[action_id], "outcome action")
            if set(item) != {"q_reference", "q_action", "d"}:
                raise EvaluationError(
                    f"outcome action shape differs: {prompt_id}"
                )
            q_reference = _score_pair(
                item["q_reference"], "q_reference", prompt_id
            )
            q_action = _score_pair(item["q_action"], "q_action", prompt_id)
            declared_d = _score_pair(item["d"], "d", prompt_id, lower=-1.0)
            observed = raw_scores[action_id]
            if (
                q_reference != observed["q_reference"]
                or q_action != observed["q_action"]
            ):
                raise EvaluationError(
                    f"outcome q values differ from raw run: {prompt_id}"
                )
            observed_reference = cast(
                dict[str, float], observed["q_reference"]
            )
            observed_action = cast(dict[str, float], observed["q_action"])
            d_loose = observed_reference["loose"] - observed_action["loose"]
            d_strict = (
                observed_reference["strict"] - observed_action["strict"]
            )
            derived = {"loose": d_loose, "strict": d_strict}
            if declared_d != derived:
                raise EvaluationError(
                    f"supplied signed outcome differs: {prompt_id}"
                )
            labels[prompt_id][action_id] = {
                "loose": d_loose,
                "strict": d_strict,
            }
    if tuple(labels) != tuple(prediction_ids):
        raise EvaluationError(
            "outcome prompt IDs are incomplete or reordered"
        )
    return labels


def _sealed_h8_index(
    seal: Mapping[str, object],
) -> dict[tuple[str, str], Mapping[str, object]]:
    raw = seal.get("h8_records")
    if not isinstance(raw, list):
        raise EvaluationError("prediction seal lacks H8 records")
    index: dict[tuple[str, str], Mapping[str, object]] = {}
    for value in raw:
        record = _object(value, "sealed H8 record")
        prompt_id = _string(record.get("prompt_id"), "sealed H8 prompt ID")
        action_id = _string(record.get("action_id"), "sealed H8 action ID")
        if action_id not in ACTION_IDS or (prompt_id, action_id) in index:
            raise EvaluationError("sealed H8 record pair is malformed")
        index[(prompt_id, action_id)] = record
    predictions = _sealed_predictions(seal)
    expected = {
        (prompt_id, action_id)
        for prompt_id in predictions
        for action_id in ACTION_IDS
    }
    if set(index) != expected:
        raise EvaluationError("sealed H8 records do not match predictions")
    return index


def _safe_outcome_artifact_directory(
    base: Path, relative: str, prompt_id: str
) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise EvaluationError(f"outcome artifact path is unsafe: {prompt_id}")
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as error:
        raise EvaluationError(
            f"outcome artifact path escapes output: {prompt_id}"
        ) from error
    return resolved


def _score_pair(
    value: object, label: str, prompt_id: str, *, lower: float = 0.0
) -> dict[str, float]:
    pair = _object(value, f"{label} {prompt_id}")
    if set(pair) != {"loose", "strict"}:
        raise EvaluationError(f"{label} shape differs: {prompt_id}")
    return {
        target: _finite(
            pair[target], f"{label} {target}", lower=lower, upper=1.0
        )
        for target in ("loose", "strict")
    }


def _validate_raw_outcome_run(
    document: Mapping[str, object],
    prompt_id: str,
    seal: Mapping[str, object],
    h8_index: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if (
        document.get("passed") is not True
        or document.get("status") != "completed"
    ):
        raise EvaluationError(f"outcome run is not complete: {prompt_id}")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise EvaluationError(
            f"outcome run result count differs: {prompt_id}"
        )
    result = _object(results[0], "outcome run result")
    prompt = _object(result.get("prompt"), "outcome run prompt")
    if prompt.get("prompt_id") != prompt_id:
        raise EvaluationError(f"outcome run prompt differs: {prompt_id}")
    _validate_outcome_prompt_lineage(prompt, seal, prompt_id)
    _validate_outcome_run_identity(document, seal, prompt_id)
    _validate_outcome_configuration(document, seal, prompt_id)
    acceptance = _object(result.get("acceptance"), "outcome acceptance")
    boundary = _object(acceptance.get("boundary"), "outcome boundary")
    portable_boundary = _portable_boundary(boundary, prompt_id)
    actual_cache = _boundary_cache_from_action_arms(
        acceptance.get("action_arms"), prompt_id
    )
    generated = boundary.get("generated_token_ids")
    if (
        boundary.get("generated_count") != 32
        or boundary.get("pending_generated_index") != 31
        or not isinstance(generated, list)
        or len(generated) != 32
    ):
        raise EvaluationError(f"outcome boundary differs: {prompt_id}")
    if boundary.get("pending_token_id") != generated[31]:
        raise EvaluationError(f"outcome pending token differs: {prompt_id}")
    for action_id in ACTION_IDS:
        sealed = h8_index[(prompt_id, action_id)]
        token_ids = sealed.get("input_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise EvaluationError(
                f"sealed H8 input IDs are missing: {prompt_id}"
            )
        if generated[31] != token_ids[0]:
            raise EvaluationError(
                f"outcome boundary token differs: {prompt_id}"
            )
        if portable_boundary != sealed.get("boundary_stable"):
            raise EvaluationError(
                f"outcome portable boundary differs: {prompt_id}"
            )
        if actual_cache != sealed.get("boundary_cache_fingerprint"):
            raise EvaluationError(
                f"outcome boundary cache differs: {prompt_id}"
            )
        steps = _integer(sealed.get("realized_steps"), "sealed H8 steps")
        outputs = _object(result.get("outputs"), "outcome outputs")
        uninterrupted = _object(
            outputs.get("uninterrupted"), "outcome uninterrupted output"
        )
        reference_ids = uninterrupted.get("token_ids")
        if (
            not isinstance(reference_ids, list)
            or reference_ids[31 : 31 + steps] != token_ids
            or reference_ids[32 : 32 + steps]
            != sealed.get("reference_argmax_token_ids")
        ):
            raise EvaluationError(
                f"outcome reference trajectory differs: {prompt_id}"
            )
        action_outputs = outputs.get("actions")
        if not isinstance(action_outputs, dict) or set(action_outputs) != set(
            ACTION_IDS
        ):
            raise EvaluationError(
                f"outcome action outputs differ: {prompt_id}"
            )
        action_output = _object(
            action_outputs[action_id], "outcome action output"
        )
        action_meta = _object(
            action_output.get("action"), "outcome action metadata"
        )
        if action_meta != {
            "action_id": action_id,
            "name": "knorm",
            "removal_fraction": ACTION_RATIOS[action_id],
        }:
            raise EvaluationError(
                f"outcome action metadata differs: {prompt_id}"
            )
        action_ids = action_output.get("token_ids")
        if not isinstance(action_ids, list) or action_ids[:32] != generated:
            raise EvaluationError(
                f"outcome action boundary differs: {prompt_id}"
            )
    scores = _object(result.get("scores"), "outcome scores")
    reference = _score_summary(
        scores.get("reference"), "reference", prompt_id
    )
    actions = scores.get("actions")
    if not isinstance(actions, dict) or set(actions) != set(ACTION_IDS):
        raise EvaluationError(
            f"outcome raw action scores malformed: {prompt_id}"
        )
    observed: dict[str, dict[str, object]] = {}
    for action_id in ACTION_IDS:
        action_score = _object(
            actions[action_id], f"raw action score {prompt_id}"
        )
        action = _score_summary(
            action_score.get("action"), "action", prompt_id
        )
        observed[action_id] = {
            "q_reference": reference,
            "q_action": action,
        }
        for target in ("loose", "strict"):
            expected_d = reference[target] - action[target]
            declared_d = _finite(
                action_score.get("d_" + target),
                f"raw signed {target}",
                lower=-1.0,
                upper=1.0,
            )
            if not math.isclose(
                expected_d, declared_d, abs_tol=1e-12, rel_tol=0.0
            ):
                raise EvaluationError(
                    f"raw signed score differs: {prompt_id}"
                )
    return observed


def _validate_outcome_prompt_lineage(
    prompt: Mapping[str, object], seal: Mapping[str, object], prompt_id: str
) -> None:
    inputs = _object(seal.get("inputs"), "sealed inputs")
    reference = _object(inputs.get("test_manifest"), "sealed test manifest")
    manifest_path = Path(
        _string(reference.get("path"), "sealed test manifest path")
    )
    if file_sha256(manifest_path) != _string(
        reference.get("file_sha256"), "sealed test manifest file hash"
    ):
        raise EvaluationError(f"sealed test manifest changed: {prompt_id}")
    manifest = load_prompt_manifest(manifest_path)
    record = manifest.records.get(prompt_id)
    if record is None or prompt != record:
        raise EvaluationError(f"outcome prompt lineage differs: {prompt_id}")


def _score_summary(
    value: object, label: str, prompt_id: str
) -> dict[str, float]:
    score = _object(value, f"raw {label} score")
    result: dict[str, float] = {}
    for target in ("loose", "strict"):
        score_value = _finite(
            score.get(target), f"raw {label} {target}", lower=0.0, upper=1.0
        )
        vector = score.get(target + "_pass")
        if (
            not isinstance(vector, list)
            or not vector
            or not all(isinstance(item, bool) for item in vector)
        ):
            raise EvaluationError(
                f"raw {label} {target} vector malformed: {prompt_id}"
            )
        mean = float(np.mean(np.asarray(vector, dtype=float)))
        if not math.isclose(score_value, mean, abs_tol=1e-12, rel_tol=0.0):
            raise EvaluationError(
                f"raw {label} {target} vector mismatch: {prompt_id}"
            )
        instruction_count = score.get("instruction_count")
        if instruction_count != len(vector):
            raise EvaluationError(
                f"raw {label} instruction count differs: {prompt_id}"
            )
        result[target] = score_value
    return result


def _validate_outcome_configuration(
    document: Mapping[str, object], seal: Mapping[str, object], prompt_id: str
) -> None:
    observed = _object(document.get("configuration"), "outcome configuration")
    locked = _object(seal.get("locked_configuration"), "sealed configuration")
    for key in ("decision_tokens", "max_new_tokens", "seed", "eos_ids"):
        if observed.get(key) != locked.get(key):
            raise EvaluationError(
                f"outcome configuration differs: {prompt_id}"
            )
    if observed.get("actions") != locked.get("actions"):
        raise EvaluationError(
            f"outcome action configuration differs: {prompt_id}"
        )


def _validate_outcome_run_identity(
    document: Mapping[str, object], seal: Mapping[str, object], prompt_id: str
) -> None:
    locked_model = seal.get("locked_model")
    if document.get("model") != locked_model:
        raise EvaluationError(f"outcome model differs: {prompt_id}")
    environment = _object(document.get("environment"), "outcome environment")
    locked_software = _object(seal.get("locked_software"), "sealed software")
    packages = _object(environment.get("packages"), "outcome packages")
    generation = _object(
        locked_software.get("generation"), "generation software"
    )
    for name, expected in generation.items():
        if packages.get(name) != expected:
            raise EvaluationError(
                f"outcome software differs: {prompt_id}/{name}"
            )
    tokenizer = _object(seal.get("locked_tokenizer"), "sealed tokenizer")
    if environment.get("tokenizer_name_or_path") != tokenizer.get(
        "name_or_path"
    ):
        raise EvaluationError(f"outcome tokenizer differs: {prompt_id}")
    source = _object(
        document.get("source_manifest"), "outcome source manifest"
    )
    _validate_reuse_source_manifest(
        source,
        {"source_manifest": seal.get("locked_source_manifest")},
        prompt_id,
    )


def _validate_official_parity(
    value: Mapping[str, object], base: Path
) -> None:
    parity = value.get("official_parity")
    if parity is None:
        return
    item = _object(parity, "official parity")
    if set(item) != {"path", "sha256"}:
        raise EvaluationError("official parity reference shape differs")
    relative = _string(item["path"], "official parity path")
    path = _safe_outcome_artifact_directory(base, relative, "official parity")
    expected = _string(item["sha256"], "official parity hash")
    if not path.is_file() or file_sha256(path) != expected:
        raise EvaluationError("official parity hash differs")


def _verify_official_run(
    run_path: Path, prompt_id: str, seal: Mapping[str, object]
) -> None:
    """Recompute every saved response with the pinned official scorer.

    ``verify_official_scores.py`` is the single parity implementation used by
    the study.  Run it with the frozen scorer directory first on PYTHONPATH so
    a system-installed package cannot silently replace the pinned code.
    """
    verifier = PROJECT_ROOT / "scripts/verify_official_scores.py"
    scorer_root = PROJECT_ROOT / "results/engineering/official-scorer"
    if not verifier.is_file() or not scorer_root.is_dir():
        raise EvaluationError(
            f"pinned official scorer is unavailable: {prompt_id}"
        )
    with tempfile.TemporaryDirectory(prefix="herald-official-parity-") as tmp:
        report_path = Path(tmp) / "parity.json"
        environment = os.environ.copy()
        inherited = environment.get("PYTHONPATH")
        paths = [
            str(scorer_root),
            str(PROJECT_ROOT / "src"),
            str(PROJECT_ROOT / "scripts"),
        ]
        if inherited:
            paths.append(inherited)
        environment["PYTHONPATH"] = os.pathsep.join(paths)
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(verifier),
                    str(run_path),
                    "--output",
                    str(report_path),
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as error:
            raise EvaluationError(
                f"official scorer could not run: {prompt_id}"
            ) from error
        if result.returncode != 0 or not report_path.is_file():
            detail = result.stderr.strip().splitlines()
            suffix = f": {detail[-1][:240]}" if detail else ""
            raise EvaluationError(
                f"official scorer parity failed: {prompt_id}{suffix}"
            )
        report = load_json(report_path)
    if report.get("passed") is not True:
        raise EvaluationError(f"official scorer parity failed: {prompt_id}")
    if report.get("run_sha256") != file_sha256(run_path):
        raise EvaluationError(
            f"official scorer run hash differs: {prompt_id}"
        )
    if report.get("official_source") != seal.get("locked_scorer_provenance"):
        raise EvaluationError(
            f"official scorer provenance differs: {prompt_id}"
        )


def _validate_outcome_hashes(
    item: Mapping[str, object],
    row: Mapping[str, object],
    prompt_id: str,
) -> None:
    found = False
    for source in (item, row):
        for key in ("run_sha256", "run_hash", "outcome_run_sha256"):
            if key in source:
                _string(source[key], f"outcome run hash {prompt_id}")
                found = True
        run = source.get("run")
        if isinstance(run, dict) and any(
            key in run for key in ("sha256", "hash")
        ):
            hash_value = run.get("sha256", run.get("hash"))
            _string(hash_value, f"outcome run hash {prompt_id}")
            found = True
    if not found:
        raise EvaluationError(f"outcome run hash missing: {prompt_id}")


def _metrics_from_errors(
    flat_errors: Sequence[float],
    values: Sequence[float],
    cluster_mse: np.ndarray,
    cluster_bias: np.ndarray,
    prompt_ids: Sequence[str],
) -> dict[str, object]:
    errors = np.asarray(flat_errors, dtype=float)
    targets = np.asarray(values, dtype=float)
    if len(errors) != len(targets) or not np.isfinite(errors).all():
        raise EvaluationError("metric values are malformed")
    sign_subset: dict[str, dict[str, object]] = {}
    for name, mask in (
        ("positive", targets > 0),
        ("zero", targets == 0),
        ("negative", targets < 0),
    ):
        subset = errors[mask]
        entry: dict[str, object] = {"count": int(mask.sum())}
        if len(subset):
            entry.update(
                {
                    "mse": float(np.mean(subset**2)),
                    "mae": float(np.mean(np.abs(subset))),
                    "bias": float(np.mean(subset)),
                }
            )
        sign_subset[name] = entry
    return {
        "prompt_count": len(prompt_ids),
        "row_count": len(errors),
        "mse": float(np.mean(cluster_mse)),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(cluster_bias)),
        "sign_subset": sign_subset,
    }


def _comparisons(
    metrics: Mapping[str, Mapping[str, object]],
    bootstrap: Mapping[str, np.ndarray],
) -> dict[str, object]:
    b4 = float(metrics["B4_L"]["mse"])
    result: dict[str, object] = {}
    for name in ("action_mean", "B0_L", "B1_L", "B2_L", "B3_L"):
        comparator = float(metrics[name]["mse"])
        difference = bootstrap[name] - bootstrap["B4_L"]
        if comparator == 0.0:
            result[name] = {
                "comparator_mse": comparator,
                "B4_L_mse": b4,
                "relative_skill": None,
                "B4_L_beats": False,
                "mse_difference_comparator_minus_B4_L": {
                    "lower": float(np.quantile(difference, 0.025)),
                    "upper": float(np.quantile(difference, 0.975)),
                },
            }
        else:
            relative = (comparator - b4) / comparator
            lower = float(np.quantile(difference, 0.025))
            result[name] = {
                "comparator_mse": comparator,
                "B4_L_mse": b4,
                "relative_skill": float(relative),
                "B4_L_beats": bool(relative >= 0.05 and lower > 0.0),
                "mse_difference_comparator_minus_B4_L": {
                    "lower": lower,
                    "upper": float(np.quantile(difference, 0.975)),
                },
            }
    return result


def _status_from_comparisons(
    nonzero: int, comparisons: Mapping[str, object], prompt_count: int
) -> str:
    if prompt_count == 0:
        return "operational_integrity_failure"
    if nonzero < NONZERO_PROMPT_FLOOR:
        return "inconclusive_information_floor"
    if all(
        _object(value, "comparison").get("B4_L_beats") is True
        for value in comparisons.values()
    ):
        return "positive_exploratory"
    return "negative"


def _target_counts(
    labels: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, int]]:
    return {
        target: {
            sign: sum(
                value[target] > 0
                if sign == "positive"
                else value[target] < 0
                if sign == "negative"
                else value[target] == 0
                for action_map in labels.values()
                for value in action_map.values()
            )
            for sign in ("positive", "zero", "negative")
        }
        for target in ("loose", "strict")
    }


def _find_value(mapping: Mapping[str, object], *names: str) -> object | None:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _find_mapping(
    mapping: Mapping[str, object], *names: str
) -> dict[str, object] | None:
    value = _find_value(mapping, *names)
    return value if isinstance(value, dict) else None


def _find_hash_ref(lock: Mapping[str, object], *names: str) -> str | None:
    value = _find_value(lock, *names)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("sha256", "file_sha256", "hash"):
            if isinstance(value.get(key), str):
                return value[key]
    return None


def _validate_lock_manifest_ref(
    lock: Mapping[str, object], name: str, manifest: LoadedPrompts, path: Path
) -> None:
    reference = lock.get(name)
    if reference is None:
        raise EvaluationError(f"protocol lock lacks {name} manifest")
    item = _object(reference, f"{name} manifest lock reference")
    required = {"path", "file_sha256", "manifest_sha256", "prompt_ids"}
    missing = sorted(required - set(item))
    if missing:
        raise EvaluationError(
            f"{name} manifest lock is missing: {', '.join(missing)}"
        )
    _string(item["path"], f"{name} manifest path")
    declared_file = _string(item["file_sha256"], f"{name} manifest file hash")
    if declared_file != manifest.file_sha256:
        raise EvaluationError(f"{name} manifest file hash differs")
    declared_semantic = _string(
        item["manifest_sha256"], f"{name} manifest fingerprint"
    )
    if declared_semantic != manifest.manifest_sha256:
        raise EvaluationError(f"{name} manifest semantic hash differs")
    ids = item["prompt_ids"]
    if ids != list(manifest.ids):
        raise EvaluationError(f"{name} manifest IDs differ")


def _validate_file_ref(
    reference: Mapping[str, object], path: str | Path, label: str
) -> None:
    """Require the canonical path/hash pair used by frozen inputs."""
    _string(reference.get("path"), f"{label} path")
    expected = _string(reference.get("sha256"), f"{label} sha256")
    if file_sha256(path) != expected:
        raise EvaluationError(f"protocol lock {label} hash differs")


def _validate_locked_actions(value: object) -> None:
    if value != list(ACTION_IDS):
        raise EvaluationError("protocol lock actions differ")


def _validate_locked_configuration(value: object) -> None:
    configuration = _object(value, "protocol lock configuration")
    required = {
        "decision_tokens",
        "max_lookahead_steps",
        "max_new_tokens",
        "actions",
        "seed",
        "eos_ids",
        "decode_skip_special_tokens",
    }
    missing = sorted(required - set(configuration))
    if missing:
        raise EvaluationError(
            "protocol lock configuration is missing: " + ", ".join(missing)
        )
    if configuration["decision_tokens"] != 32:
        raise EvaluationError("protocol lock decision boundary differs")
    if configuration["max_lookahead_steps"] != MAX_LOOKAHEAD_STEPS:
        raise EvaluationError("protocol lock lookahead cap differs")
    if configuration["max_new_tokens"] != 1024:
        raise EvaluationError("protocol lock generation budget differs")
    if configuration["decode_skip_special_tokens"] is not True:
        raise EvaluationError("protocol lock decoding configuration differs")
    if configuration["seed"] != 0 or not isinstance(
        configuration["seed"], int
    ):
        raise EvaluationError("protocol lock seed differs")
    if not _actions_match_strict(configuration["actions"]):
        raise EvaluationError("protocol lock configuration actions differ")
    if not isinstance(configuration["eos_ids"], list) or not all(
        isinstance(item, int) and not isinstance(item, bool)
        for item in configuration["eos_ids"]
    ):
        raise EvaluationError("protocol lock EOS IDs are malformed")


def _actions_match_strict(value: object) -> bool:
    if not isinstance(value, list) or len(value) != len(ACTION_IDS):
        return False
    expected = (0.25, 0.5)
    for item, ratio in zip(value, expected, strict=True):
        if item != {"name": "knorm", "removal_fraction": ratio}:
            return False
    return True


def _validate_locked_features(value: object) -> None:
    features = _object(value, "protocol lock features")
    columns = features.get("columns")
    if columns != list(BLOCK_FEATURES["B4_L"]):
        raise EvaluationError("protocol lock feature columns differ")
    dtypes = features.get("dtypes")
    if not isinstance(dtypes, dict) or set(dtypes) != set(columns):
        raise EvaluationError("protocol lock feature dtypes differ")
    # Current producer writes all scalar feature columns as float64. L and
    # has_delayed are represented as numeric features by the fixed estimator.
    if any(dtype != "float64" for dtype in dtypes.values()):
        raise EvaluationError("protocol lock feature dtype differs")
    model_columns = features.get("model_columns")
    if not isinstance(model_columns, dict) or set(model_columns) != set(
        BLOCK_FEATURES
    ):
        raise EvaluationError("protocol lock model feature sets differ")
    for name, expected in BLOCK_FEATURES.items():
        if model_columns[name] != list(expected):
            raise EvaluationError(f"protocol lock {name} columns differ")


def _validate_locked_estimator(value: object) -> None:
    estimator = _object(value, "protocol lock estimator")
    if estimator.get("scaler") != "StandardScaler":
        raise EvaluationError("protocol lock scaler differs")
    if estimator.get("regressor") != "Ridge":
        raise EvaluationError("protocol lock regressor differs")
    if estimator.get("ridge_alpha") != 1.0:
        raise EvaluationError("protocol lock Ridge alpha differs")
    if estimator.get("prediction_clip") != [-1.0, 1.0]:
        raise EvaluationError("protocol lock prediction clip differs")
    if estimator.get("targets") != ["loose", "strict"]:
        raise EvaluationError("protocol lock targets differ")
    if estimator.get("train_prompt_count") != TRAIN_PROMPT_COUNT:
        raise EvaluationError("protocol lock training prompt count differs")
    if estimator.get("train_action_rows") != TRAIN_ACTION_ROW_COUNT:
        raise EvaluationError("protocol lock training action rows differ")


def _validate_locked_tokenizer(value: Mapping[str, object]) -> None:
    """Validate captured tokenizer metadata without touching its files."""
    required = {
        "name_or_path",
        "revision",
        "chat_template_sha256",
        "files_sha256",
    }
    if set(value) != required:
        raise EvaluationError("protocol lock tokenizer keys are not exact")
    _string(value.get("name_or_path"), "protocol lock tokenizer name")
    _string(value.get("revision"), "protocol lock tokenizer revision")
    _sha256(
        value.get("chat_template_sha256"),
        "protocol lock tokenizer chat template hash",
    )
    files = _object(
        value.get("files_sha256"), "protocol lock tokenizer files"
    )
    expected_files = {
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if set(files) != expected_files:
        raise EvaluationError("protocol lock tokenizer files are not exact")
    for name, digest in files.items():
        _sha256(digest, f"protocol lock tokenizer file hash {name}")


def _validate_generation_environment(value: object) -> None:
    """Validate captured old-run environment as metadata only."""
    environment = _object(value, "protocol lock generation environment")
    expected_keys = {
        "attention_backend",
        "cuda_available",
        "cuda_version",
        "cwd",
        "device",
        "dtype",
        "packages",
        "platform",
        "python",
        "python_executable",
        "resources",
        "seed",
        "tokenizer_name_or_path",
        "torch_version",
    }
    if set(environment) != expected_keys:
        raise EvaluationError(
            "protocol lock generation environment keys are not exact"
        )
    for key in (
        "attention_backend",
        "cuda_version",
        "cwd",
        "device",
        "dtype",
        "platform",
        "python",
        "python_executable",
        "tokenizer_name_or_path",
        "torch_version",
    ):
        _string(environment.get(key), f"generation environment {key}")
    if not isinstance(environment.get("cuda_available"), bool):
        raise EvaluationError("generation environment CUDA flag is malformed")
    _expect_int(environment, "seed", 0, required=True)
    packages = _object(
        environment.get("packages"), "generation environment packages"
    )
    if not packages or any(
        not isinstance(name, str)
        or not name
        or not isinstance(version, str)
        or not version
        for name, version in packages.items()
    ):
        raise EvaluationError("generation environment packages are malformed")
    resources = _object(
        environment.get("resources"), "generation environment resources"
    )
    modules = _object(
        resources.get("modules"), "generation environment modules"
    )
    if set(modules) != {"absl", "immutabledict", "langdetect", "nltk"}:
        raise EvaluationError("generation environment modules are not exact")
    for name, version in modules.items():
        if version is not None and (
            not isinstance(version, str) or not version
        ):
            raise EvaluationError(
                f"generation environment module is malformed: {name}"
            )
    if resources.get("nltk_resources") != [
        "tokenizers/punkt",
        "tokenizers/punkt_tab",
    ]:
        raise EvaluationError("generation environment resources differ")


def _validate_source_manifest(value: object) -> None:
    source_manifest = _object(value, "protocol lock source_manifest")
    if not source_manifest:
        raise EvaluationError("protocol lock source manifest is empty")
    for relative, expected in source_manifest.items():
        relative_name = _string(relative, "protocol lock source path")
        source_path = Path(relative_name)
        if source_path.is_absolute() or ".." in source_path.parts:
            raise EvaluationError("protocol lock source path is unsafe")
        expected_hash = _string(expected, f"source hash {relative_name}")
        actual_path = PROJECT_ROOT / source_path
        if (
            not actual_path.is_file()
            or file_sha256(actual_path) != expected_hash
        ):
            raise EvaluationError(
                f"protocol lock source hash differs: {relative_name}"
            )


def _validate_software(value: object) -> None:
    software = _object(value, "protocol lock software")
    generation = _object(software.get("generation"), "generation software")
    evaluation = _object(software.get("evaluation"), "evaluation software")
    if not generation or not evaluation:
        raise EvaluationError("protocol lock software is incomplete")
    for package, version in generation.items():
        _string(package, "generation package")
        _string(version, f"generation version {package}")
    for package in ("numpy", "scikit-learn"):
        expected = _string(evaluation.get(package), f"evaluation {package}")
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as error:
            raise EvaluationError(
                f"evaluation package missing: {package}"
            ) from error
        if actual != expected:
            raise EvaluationError(
                f"evaluation package version differs: {package}"
            )


def _validate_scorer_provenance(value: object) -> None:
    provenance = _object(value, "protocol lock scorer provenance")
    files = provenance.get("files")
    if not isinstance(files, dict) or not files:
        raise EvaluationError("protocol lock scorer files are missing")
    root = PROJECT_ROOT / "results/engineering/official-scorer"
    for name, entry_value in files.items():
        filename = _string(name, "scorer filename")
        entry = _object(entry_value, "scorer file entry")
        expected = _string(entry.get("sha256"), f"scorer hash {filename}")
        path = (
            root / filename
            if filename == "input_data.jsonl"
            else root / "instruction_following_eval" / filename
        )
        if not path.is_file() or file_sha256(path) != expected:
            raise EvaluationError(f"scorer source hash differs: {filename}")


def _action_ids_from_value(value: object) -> list[str]:
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict) and isinstance(
                item.get("action_id"), str
            ):
                result.append(item["action_id"])
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                result.append(item["id"])
        return result
    if isinstance(value, dict):
        return list(value)
    return []


def _expect_int(
    mapping: Mapping[str, object],
    key: str,
    expected: int,
    *,
    required: bool = False,
) -> None:
    if required and key not in mapping:
        raise EvaluationError(f"protocol lock {key} is missing")
    if mapping.get(key) is not None and mapping.get(key) != expected:
        raise EvaluationError(f"protocol lock {key} differs")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be an object")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{label} must be a nonempty string")
    return value


def _sha256(value: object, label: str) -> str:
    result = _string(value, label)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise EvaluationError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{label} must be an integer")
    return value


def _finite(
    value: object,
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label} must be finite")
    result = float(value)
    if (
        not math.isfinite(result)
        or (lower is not None and result < lower)
        or (upper is not None and result > upper)
    ):
        raise EvaluationError(f"{label} is outside its allowed range")
    return result


def _int_tuple(
    value: object, fallback: Sequence[int], label: str
) -> tuple[int, ...]:
    chosen = value if value is not None else fallback
    if not isinstance(chosen, list) or not chosen:
        raise EvaluationError(f"{label} is missing")
    return tuple(_integer(item, label) for item in chosen)


def _clip(value: float) -> float:
    return float(np.clip(value, -1.0, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
