#!/usr/bin/env python3
"""Evaluate the frozen HERALD v3 pilot from offline runner evidence.

The input contract is deliberately small. ``collection_root`` contains one
per-prompt ``run.json`` written by ``run_engineering.py``, a
``checkpoint.json`` ledger, and an ``experiment-lock.json``. Ledger prompt IDs
must be the prefix
of the frozen manifest in the same order, and accepted rows must be the first
120 eligible prompts. An eligible prompt must contribute exactly the two
frozen Knorm actions.

This script does not load a model or score text. It only consumes the saved
decision-time probes and paired signed IFEval scores, then fits the fixed
fold-local regressions specified in the research plan.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCHEMA_VERSION = "herald_v3.pilot_evaluation.v1"
COLLECTION_SUMMARY_NAME = "checkpoint.json"
EXPERIMENT_LOCK_NAME = "experiment-lock.json"
DESIGN_LOCK_DEFAULT = (
    Path(__file__).resolve().parents[1] / "data/pilot-v1/lock.json"
)
REQUIRED_ARTIFACTS = ("prompt-manifest.json", "run.json", "run.log")
ACTION_RATIOS = (0.25, 0.5)
ACTION_IDS = tuple(f"knorm:{ratio:.6g}" for ratio in ACTION_RATIOS)
FOLD_COUNT = 5
ROSTER_COUNT = 160
TARGET_ELIGIBLE = 120
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SNAPSHOT = "a09a35458c702b33eeacc393d103063234e8bc28"
MAX_NEW_TOKENS = 1024
SEED = 0
MIN_ELIGIBLE_PROMPTS = 120
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 0
NONZERO_PROMPT_FLOOR = 20

BLOCK_FEATURES: dict[str, tuple[str, ...]] = {
    "B0": (
        "removal_fraction",
        "prompt_token_count",
        "decision_index32",
        "pre_action_cache_size",
    ),
    "B1": (
        "removal_fraction",
        "prompt_token_count",
        "decision_index32",
        "pre_action_cache_size",
        "reference_entropy",
        "reference_top2_margin",
    ),
    "B2": (
        "removal_fraction",
        "prompt_token_count",
        "decision_index32",
        "pre_action_cache_size",
        "reference_entropy",
        "reference_top2_margin",
        "action_reference_entropy_delta",
        "action_reference_margin_delta",
        "argmax_match",
    ),
    "B3": (
        "removal_fraction",
        "prompt_token_count",
        "decision_index32",
        "pre_action_cache_size",
        "reference_entropy",
        "reference_top2_margin",
        "action_reference_entropy_delta",
        "action_reference_margin_delta",
        "argmax_match",
        "js_divergence",
    ),
}
MODEL_NAMES = tuple(BLOCK_FEATURES) + ("actionmean",)
TARGET_NAMES = {"loose": "target_loose", "strict": "target_strict"}


class EvaluationError(ValueError):
    """Raised when collection evidence cannot support the fixed analysis."""


@dataclass(frozen=True)
class PilotRow:
    """One action row with decision-time features and signed outcomes."""

    prompt_id: str
    fold: int
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
    js_divergence: float
    target_loose: float
    target_strict: float

    def feature_values(self, names: Sequence[str]) -> list[float]:
        return [float(getattr(self, name)) for name in names]


@dataclass(frozen=True)
class LoadedCollection:
    """Validated rows and provenance needed by the report writer."""

    rows: tuple[PilotRow, ...]
    manifest_ids: tuple[str, ...]
    eligible_ids: tuple[str, ...]
    input_hashes: dict[str, object]
    source_manifests: tuple[dict[str, object], ...]


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_collection(
    collection_root: str | Path,
    manifest_path: str | Path,
    *,
    design_lock_path: str | Path = DESIGN_LOCK_DEFAULT,
    minimum_eligible: int = MIN_ELIGIBLE_PROMPTS,
) -> LoadedCollection:
    """Load, validate, and order all eligible action rows in a collection."""
    root = Path(collection_root)
    if not root.is_dir():
        raise EvaluationError(f"collection root is not a directory: {root}")
    manifest_file = Path(manifest_path)
    if not manifest_file.is_file():
        raise EvaluationError(f"manifest does not exist: {manifest_file}")
    manifest = _load_manifest(manifest_file)
    design_lock_file = Path(design_lock_path)
    if not design_lock_file.is_file():
        raise EvaluationError(
            f"design lock does not exist: {design_lock_file}"
        )
    design_lock = _load_json(design_lock_file)
    _validate_design_lock(
        design_lock, design_lock_file, manifest_file, manifest
    )
    summary_path = root / COLLECTION_SUMMARY_NAME
    if not summary_path.is_file():
        raise EvaluationError(
            f"collection summary is required at {summary_path}"
        )
    summary = _load_json(summary_path)
    if (
        summary.get("schema_version") != "herald_v3.pilot_checkpoint.v1"
        or summary.get("status") != "completed"
        or summary.get("passed") is not True
    ):
        raise EvaluationError(
            "checkpoint is not a completed passing collection"
        )
    manifest_sha = file_sha256(manifest_file)
    experiment_lock_path = root / EXPERIMENT_LOCK_NAME
    if not experiment_lock_path.is_file():
        raise EvaluationError(
            f"experiment lock is required at {experiment_lock_path}"
        )
    experiment_lock = _load_json(experiment_lock_path)
    _validate_experiment_lock(
        experiment_lock,
        experiment_lock_path,
        manifest_sha,
        manifest,
        design_lock,
    )
    checkpoint_lock_sha = summary.get("experiment_lock_sha256")
    if not isinstance(checkpoint_lock_sha, str):
        raise EvaluationError(
            "checkpoint experiment_lock_sha256 must be a string"
        )
    if checkpoint_lock_sha != file_sha256(experiment_lock_path):
        raise EvaluationError(
            "checkpoint experiment lock hash does not match lock"
        )
    ledger = summary.get("ledger")
    if not isinstance(ledger, list) or not ledger:
        raise EvaluationError(
            "checkpoint must contain a nonempty ledger list"
        )
    ledger_ids: list[str] = []
    eligible_ids_list: list[str] = []
    for entry in ledger:
        entry_object = _object(entry, "checkpoint ledger entry")
        prompt_id = _string(entry_object.get("prompt_id"), "ledger prompt_id")
        if prompt_id in ledger_ids:
            raise EvaluationError(
                f"checkpoint contains duplicate prompt {prompt_id}"
            )
        ledger_ids.append(prompt_id)
        status = entry_object.get("status")
        common_keys = {
            "prompt_id",
            "status",
            "artifact_directory",
            "artifacts_sha256",
        }
        expected_keys = common_keys
        if status == "accepted":
            eligible_ids_list.append(prompt_id)
        elif status == "ineligible_early_eos":
            expected_keys = common_keys | {"reason", "generated_token_ids"}
        else:
            raise EvaluationError(
                f"unsupported checkpoint status for {prompt_id}: {status!r}"
            )
        if set(entry_object) != expected_keys:
            raise EvaluationError(
                f"checkpoint ledger fields are malformed for {prompt_id}"
            )
    if tuple(ledger_ids) != tuple(manifest["ids"][: len(ledger_ids)]):
        raise EvaluationError(
            "checkpoint ledger membership/order differs from frozen "
            "manifest prefix"
        )
    eligible_ids = tuple(eligible_ids_list)
    accepted_eligible = summary.get("accepted_eligible")
    if not isinstance(accepted_eligible, int) or accepted_eligible != len(
        eligible_ids
    ):
        raise EvaluationError(
            "checkpoint accepted_eligible does not match its ledger"
        )
    target_eligible = summary.get("target_eligible")
    if (
        not isinstance(target_eligible, int)
        or target_eligible != TARGET_ELIGIBLE
        or target_eligible != len(eligible_ids)
    ):
        raise EvaluationError(
            "checkpoint target_eligible does not match its ledger"
        )
    accepted_seen = 0
    for entry in ledger:
        if entry.get("status") == "accepted":
            accepted_seen += 1
        elif accepted_seen >= TARGET_ELIGIBLE:
            raise EvaluationError(
                "checkpoint contains entries after the target was reached"
            )
    processed = summary.get("processed")
    if not isinstance(processed, int) or processed != len(ledger):
        raise EvaluationError(
            "checkpoint processed does not match its ledger"
        )
    roster_count = summary.get("roster_count")
    if (
        not isinstance(roster_count, int)
        or roster_count != ROSTER_COUNT
        or roster_count != len(manifest["ids"])
    ):
        raise EvaluationError(
            "checkpoint roster_count does not match manifest"
        )
    early_eos = summary.get("ineligible_early_eos")
    if not isinstance(early_eos, int) or early_eos != len(ledger) - len(
        eligible_ids
    ):
        raise EvaluationError(
            "checkpoint early-EOS count does not match its ledger"
        )

    run_paths: list[Path] = []
    ledger_prompt_ids = set(ledger_ids)
    for child in root.iterdir():
        if child.is_dir() and child.name not in ledger_prompt_ids:
            raise EvaluationError(
                f"untracked prompt artifact directory: {child}"
            )
    if summary.get("early_eos_ledger") != [
        entry
        for entry in ledger
        if entry.get("status") == "ineligible_early_eos"
    ]:
        raise EvaluationError(
            "checkpoint early_eos_ledger does not match ledger"
        )
    for entry in ledger:
        prompt_id = _string(entry.get("prompt_id"), "ledger prompt_id")
        artifact_directory = _string(
            entry.get("artifact_directory"),
            f"artifact_directory for {prompt_id}",
        )
        expected_artifact_hash = entry.get("artifacts_sha256")
        if not isinstance(expected_artifact_hash, str):
            raise EvaluationError(
                f"ledger artifact hash is missing for {prompt_id}"
            )
        path = root / prompt_id / "run.json"
        if not path.is_file():
            raise EvaluationError(
                f"missing local run artifact for {prompt_id}, expected {path}"
            )
        artifact_path = root / prompt_id / "artifacts.json"
        if not artifact_path.is_file():
            raise EvaluationError(
                f"missing local artifact index for {prompt_id}, "
                f"expected {artifact_path}"
            )
        if file_sha256(artifact_path) != expected_artifact_hash:
            raise EvaluationError(f"artifact hash mismatch for {prompt_id}")
        if Path(artifact_directory).name != prompt_id:
            raise EvaluationError(
                f"ledger artifact directory does not name {prompt_id}"
            )
        _validate_artifact_index(artifact_path)
        _validate_prompt_run_shape(
            path,
            prompt_id,
            entry["status"],
            manifest["folds"],
            experiment_lock,
        )
        if entry["status"] == "accepted":
            run_paths.append(path)
    documents = [(_load_json(path), path) for path in run_paths]
    rows: list[PilotRow] = []
    seen_prompt_ids: set[str] = set()
    source_manifests: list[dict[str, object]] = []
    file_hashes: dict[str, str] = {
        "manifest": manifest_sha,
        "collection_summary": file_sha256(summary_path),
        "experiment_lock": file_sha256(experiment_lock_path),
        "design_lock": file_sha256(design_lock_file),
    }
    for document, path in documents:
        artifact_path = path.with_name("artifacts.json")
        for name in REQUIRED_ARTIFACTS:
            artifact_file = path.with_name(name)
            file_hashes[str(artifact_file.relative_to(root))] = file_sha256(
                artifact_file
            )
        file_hashes[str(artifact_path.relative_to(root))] = file_sha256(
            artifact_path
        )
        source_manifest = document.get("source_manifest")
        if isinstance(source_manifest, dict):
            source_manifests.append(
                _json_object(source_manifest, "source_manifest")
            )
        rows.extend(
            _rows_from_run(
                document,
                path,
                manifest["folds"],
                seen_prompt_ids,
                experiment_lock,
            )
        )

    observed_ids = tuple(
        prompt_id
        for prompt_id in manifest["ids"]
        if prompt_id in seen_prompt_ids and prompt_id in set(eligible_ids)
    )
    if observed_ids != eligible_ids:
        missing = [
            prompt_id
            for prompt_id in eligible_ids
            if prompt_id not in seen_prompt_ids
        ]
        extra = [
            prompt_id
            for prompt_id in seen_prompt_ids
            if prompt_id not in set(eligible_ids)
        ]
        details = f"missing={missing[:3]} extra={extra[:3]}"
        raise EvaluationError(
            "run artifacts do not match checkpoint eligible "
            "membership/order, " + details
        )
    if len(eligible_ids) < minimum_eligible:
        raise EvaluationError(
            f"only {len(eligible_ids)} eligible prompts, need at least "
            f"{minimum_eligible}"
        )
    ordered_rows = sorted(
        rows,
        key=lambda row: (
            manifest["order"][row.prompt_id],
            ACTION_RATIOS.index(row.removal_fraction),
        ),
    )
    _validate_rows(ordered_rows, manifest["folds"], minimum_eligible)
    return LoadedCollection(
        rows=tuple(ordered_rows),
        manifest_ids=tuple(manifest["ids"]),
        eligible_ids=eligible_ids,
        input_hashes={"sha256": file_hashes},
        source_manifests=tuple(source_manifests),
    )


def evaluate_rows(
    rows: Sequence[PilotRow],
    *,
    minimum_eligible: int = MIN_ELIGIBLE_PROMPTS,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Fit the fixed fold-local models and return the exploratory report."""
    materialized = list(rows)
    _validate_rows(materialized, None, minimum_eligible)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, dict[str, object]]] = {}
    comparisons: dict[str, dict[str, object]] = {}
    bootstrap_report: dict[str, dict[str, object]] = {}
    oof_rows: list[dict[str, object]] = [
        {
            "prompt_id": row.prompt_id,
            "fold": row.fold,
            "action_id": row.action_id,
            "removal_fraction": row.removal_fraction,
            "target_loose": row.target_loose,
            "target_strict": row.target_strict,
            "oof_predictions": {},
        }
        for row in materialized
    ]
    for target_name, target_field in TARGET_NAMES.items():
        predictions[target_name] = {}
        metrics[target_name] = {}
        for model_name in MODEL_NAMES:
            if model_name == "actionmean":
                predicted = fit_action_mean_predictions(
                    materialized, target_name
                )
            else:
                predicted = fit_oof_predictions(
                    materialized, model_name, target_name
                )
            predictions[target_name][model_name] = predicted
            metrics[target_name][model_name] = _metrics(
                materialized, predicted, target_field
            )
            for index, value in enumerate(predicted):
                oof_rows[index]["oof_predictions"][
                    target_name + ":" + model_name
                ] = float(value)

        cluster_errors = {
            model_name: _prompt_error_means(
                materialized,
                predictions[target_name][model_name],
                target_field,
            )
            for model_name in MODEL_NAMES
        }
        bootstrap_values = paired_bootstrap(
            cluster_errors, replicates=bootstrap_replicates, seed=seed
        )
        bootstrap_report[target_name] = {
            "replicates": bootstrap_replicates,
            "seed": seed,
            "cluster_count": len(next(iter(cluster_errors.values()))),
            "mse_by_model": {
                model_name: {
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                }
                for model_name, values in bootstrap_values.items()
            },
        }
        comparisons[target_name] = _comparisons(
            metrics[target_name], bootstrap_values, materialized, target_field
        )

    loose_nonzero = len(
        {row.prompt_id for row in materialized if row.target_loose != 0.0}
    )
    prompt_count = len({row.prompt_id for row in materialized})
    coverage = {
        "eligible_prompt_count": prompt_count,
        "action_row_count": len(materialized),
        "rows_per_prompt": 2,
        "nonzero_loose_d_prompt_count": loose_nonzero,
        "nonzero_prompt_floor": NONZERO_PROMPT_FLOOR,
        "information_floor_met": loose_nonzero >= NONZERO_PROMPT_FLOOR,
        "minimum_eligible_prompts": minimum_eligible,
        "minimum_eligible_met": prompt_count >= minimum_eligible,
        "analysis_inconclusive": (
            prompt_count < MIN_ELIGIBLE_PROMPTS
            or loose_nonzero < NONZERO_PROMPT_FLOOR
        ),
        "action_counts": {
            action_id: sum(row.action_id == action_id for row in materialized)
            for action_id in ACTION_IDS
        },
        "loose_target_distribution": _distribution(
            [row.target_loose for row in materialized]
        ),
        "strict_target_distribution": _distribution(
            [row.target_strict for row in materialized]
        ),
    }
    analysis_status = (
        "inconclusive"
        if coverage["analysis_inconclusive"]
        else "evaluated_exploratory"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": analysis_status,
        "claims": {
            "scope": "exploratory_only",
            "statement": (
                "OOF errors and paired intervals are conditional on the "
                "fixed fold models and do not establish deployment or "
                "general quality."
            ),
        },
        "configuration": {
            "features": {
                name: list(values) for name, values in BLOCK_FEATURES.items()
            },
            "scaler": "StandardScaler",
            "regressor": "Ridge",
            "ridge_alpha": 1.0,
            "clip": [-1.0, 1.0],
            "folds": FOLD_COUNT,
            "prompt_weighting": "equal_prompt_then_mean_two_actions",
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
            "no_tuning": True,
            "target_primary": "loose",
            "target_sensitivity": "strict",
        },
        "coverage": coverage,
        "oof_rows": oof_rows,
        "metrics": metrics,
        "bootstrap": bootstrap_report,
        "comparisons": comparisons,
    }


def fit_oof_predictions(
    rows: Sequence[PilotRow], block: str, target: str
) -> np.ndarray:
    """Fit one fixed block in five folds and return clipped OOF values."""
    if block not in BLOCK_FEATURES:
        raise EvaluationError(f"unknown feature block: {block}")
    target_field = TARGET_NAMES.get(target)
    if target_field is None:
        raise EvaluationError(f"unknown target: {target}")
    materialized = list(rows)
    _validate_fold_assignments(materialized)
    x = np.asarray(
        [row.feature_values(BLOCK_FEATURES[block]) for row in materialized],
        dtype=float,
    )
    y = np.asarray(
        [getattr(row, target_field) for row in materialized], dtype=float
    )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise EvaluationError(
            "nonfinite feature or target reached model fitting"
        )
    folds = np.asarray([row.fold for row in materialized], dtype=int)
    predictions = np.empty(len(materialized), dtype=float)
    for fold in range(FOLD_COUNT):
        test_mask = folds == fold
        train_mask = ~test_mask
        if not test_mask.any() or not train_mask.any():
            raise EvaluationError(f"fold {fold} has no train or test rows")
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=1.0)),
            ]
        )
        model.fit(x[train_mask], y[train_mask])
        predictions[test_mask] = np.clip(
            np.asarray(model.predict(x[test_mask]), dtype=float), -1.0, 1.0
        )
    return predictions


def fit_action_mean_predictions(
    rows: Sequence[PilotRow], target: str
) -> np.ndarray:
    """Return OOF predictions from action-wise training means."""
    target_field = TARGET_NAMES.get(target)
    if target_field is None:
        raise EvaluationError(f"unknown target: {target}")
    materialized = list(rows)
    _validate_fold_assignments(materialized)
    folds = np.asarray([row.fold for row in materialized], dtype=int)
    y = np.asarray(
        [getattr(row, target_field) for row in materialized], dtype=float
    )
    predictions = np.empty(len(materialized), dtype=float)
    for fold in range(FOLD_COUNT):
        train_mask = folds != fold
        test_mask = folds == fold
        means: dict[str, float] = {}
        for action_id in ACTION_IDS:
            action_mask = train_mask & np.asarray(
                [row.action_id == action_id for row in materialized],
                dtype=bool,
            )
            if not action_mask.any():
                raise EvaluationError(
                    "action-wise baseline has no training rows for "
                    f"{action_id}"
                )
            means[action_id] = float(np.mean(y[action_mask]))
        for index in np.flatnonzero(test_mask):
            predictions[index] = np.clip(
                means[materialized[index].action_id], -1.0, 1.0
            )
    return predictions


def paired_bootstrap(
    cluster_errors: Mapping[str, np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, np.ndarray]:
    """Bootstrap cluster means with one shared sampled index matrix."""
    if replicates <= 0:
        raise EvaluationError("bootstrap replicates must be positive")
    if not cluster_errors:
        raise EvaluationError("bootstrap requires at least one model")
    names = list(cluster_errors)
    arrays = {
        name: np.asarray(cluster_errors[name], dtype=float) for name in names
    }
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1:
        raise EvaluationError(
            "bootstrap clusters must have equal nonzero lengths"
        )
    cluster_count = next(iter(lengths))
    if cluster_count == 0:
        raise EvaluationError(
            "bootstrap clusters must have equal nonzero lengths"
        )
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise EvaluationError("bootstrap errors must be finite")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, cluster_count, size=(replicates, cluster_count))
    return {
        name: np.mean(values[sampled], axis=1)
        for name, values in arrays.items()
    }


def run_evaluation(
    collection_root: str | Path,
    manifest_path: str | Path,
    output: str | Path,
    *,
    design_lock_path: str | Path = DESIGN_LOCK_DEFAULT,
) -> dict[str, object]:
    """Run the fixed CLI analysis and write one immutable JSON report."""
    output_path = Path(output)
    if output_path.exists():
        raise EvaluationError(
            f"refusing to overwrite existing output: {output_path}"
        )
    loaded = load_collection(
        collection_root,
        manifest_path,
        design_lock_path=design_lock_path,
    )
    report = evaluate_rows(loaded.rows)
    report["inputs"] = {
        "collection_root": str(Path(collection_root).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "design_lock_path": str(Path(design_lock_path).resolve()),
        "manifest_prompt_ids": list(loaded.manifest_ids),
        "eligible_prompt_ids": list(loaded.eligible_ids),
        "hashes": loaded.input_hashes,
        "source_manifests": list(loaded.source_manifests),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _json_value(report), ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for the offline pilot evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--design-lock", default=str(DESIGN_LOCK_DEFAULT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        run_evaluation(
            args.collection_root,
            args.manifest,
            args.output,
            design_lock_path=args.design_lock,
        )
    except EvaluationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def _validate_artifact_index(path: Path) -> None:
    """Verify the exact runner artifact set before consuming run.json."""
    artifact_index = _load_json(path)
    if artifact_index.get("schema_version") != 1:
        raise EvaluationError(f"artifact index schema is unsupported: {path}")
    hashes = artifact_index.get("sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(REQUIRED_ARTIFACTS):
        raise EvaluationError(
            f"artifact index file set is incomplete: {path}"
        )
    for name in REQUIRED_ARTIFACTS:
        expected = hashes.get(name)
        if not isinstance(expected, str):
            raise EvaluationError(
                f"artifact hash is missing for {name}: {path}"
            )
        artifact = path.parent / name
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise EvaluationError(
                f"artifact hash mismatch for {name}: {path}"
            )


def _validate_prompt_run_shape(
    path: Path,
    prompt_id: str,
    status: object,
    manifest_folds: Mapping[str, int],
    experiment_lock: Mapping[str, object],
) -> None:
    """Check one ledger directory, including an early-EOS row."""
    document = _load_json(path)
    for key in ("source_manifest", "model", "environment"):
        expected = experiment_lock.get(key)
        if not isinstance(expected, dict) or document.get(key) != expected:
            raise EvaluationError(
                f"{key} mismatch with experiment lock: {path}"
            )
    configuration = document.get("configuration")
    locked_configuration = experiment_lock.get("configuration")
    if not isinstance(configuration, dict) or not isinstance(
        locked_configuration, dict
    ):
        raise EvaluationError(f"run configuration mismatch: {path}")
    for key in (
        "max_new_tokens",
        "decision_tokens",
        "actions",
        "seed",
        "eos_ids",
    ):
        if configuration.get(key) != locked_configuration.get(key):
            raise EvaluationError(f"run configuration mismatch: {path}")
    if configuration.get("decode_skip_special_tokens") is not True:
        raise EvaluationError(f"run decoding configuration mismatch: {path}")
    results = document.get("results")
    if not isinstance(results, list) or len(results) != 1:
        raise EvaluationError(
            f"run must contain exactly one result row: {path}"
        )
    row = _object(results[0], "prompt run result")
    prompt = _object(row.get("prompt"), "prompt run prompt")
    if prompt.get("prompt_id") != prompt_id:
        raise EvaluationError(f"run prompt does not match ledger: {path}")
    fold = prompt.get("fold")
    if prompt_id not in manifest_folds or fold != manifest_folds[prompt_id]:
        raise EvaluationError(
            f"run fold does not match frozen manifest: {path}"
        )
    if status == "accepted":
        if (
            document.get("status") != "completed"
            or document.get("passed") is not True
            or document.get("stopped_on_failure") is not False
            or row.get("status") != "accepted"
        ):
            raise EvaluationError(
                f"accepted ledger row is not accepted: {path}"
            )
        return
    if status != "ineligible_early_eos" or row.get("status") != "ineligible":
        raise EvaluationError(f"ledger status does not match run row: {path}")
    acceptance = _object(row.get("acceptance"), "ineligible acceptance")
    eligibility = _object(
        acceptance.get("eligibility"), "ineligible eligibility"
    )
    if (
        eligibility.get("eligible") is not False
        or eligibility.get("reason") != "eos_at_or_before_pending_boundary"
        or "scores" in row
        or "outputs" in row
        or document.get("status") != "completed_with_failures"
        or document.get("passed") is not False
        or document.get("stopped_on_failure") is not False
    ):
        raise EvaluationError(
            f"ineligible prompt evidence is malformed: {path}"
        )


def _load_manifest(path: Path) -> dict[str, object]:
    document = _load_json(path)
    entries = document.get("prompts")
    if not isinstance(entries, list) or not entries:
        raise EvaluationError(
            "frozen manifest must contain a nonempty prompts list"
        )
    ids: list[str] = []
    folds: dict[str, int] = {}
    order: dict[str, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise EvaluationError(
                "frozen manifest prompt entry must be an object"
            )
        prompt_id = entry.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise EvaluationError(
                "frozen manifest prompt_id must be a nonempty string"
            )
        if prompt_id in folds:
            raise EvaluationError(
                f"frozen manifest contains duplicate prompt {prompt_id}"
            )
        fold = _integer(entry.get("fold"), f"manifest fold for {prompt_id}")
        if fold < 0 or fold >= FOLD_COUNT:
            raise EvaluationError(
                f"manifest fold outside 0..4 for {prompt_id}"
            )
        ids.append(prompt_id)
        folds[prompt_id] = fold
        order[prompt_id] = index
    return {
        "ids": tuple(ids),
        "folds": folds,
        "order": order,
        "declared_sha256": document.get("manifest_sha256"),
    }


def _validate_design_lock(
    lock: Mapping[str, object],
    path: Path,
    manifest_path: Path,
    manifest: Mapping[str, object],
) -> None:
    """Validate the frozen protocol that defines the collection contract."""
    if lock.get("schema_version") != "herald_v3.pilot_lock.v1":
        raise EvaluationError(f"unsupported design lock schema: {path}")
    roster = _object(lock.get("pilot_roster"), "design pilot_roster")
    if roster.get("candidate_count") != ROSTER_COUNT:
        raise EvaluationError("design lock roster count is not 160")
    if roster.get("primary_target_count") != TARGET_ELIGIBLE:
        raise EvaluationError("design lock target count is not 120")
    if roster.get("fold_count") != FOLD_COUNT:
        raise EvaluationError("design lock fold count is not 5")
    if roster.get("prompt_manifest") != str(manifest_path.resolve()):
        raise EvaluationError("design lock prompt manifest path differs")
    if roster.get("prompt_manifest_sha256") != file_sha256(manifest_path):
        raise EvaluationError("design lock prompt manifest hash differs")
    declared_manifest_sha = manifest.get("declared_sha256")
    if (
        not isinstance(declared_manifest_sha, str)
        or roster.get("manifest_fingerprint") != declared_manifest_sha
    ):
        raise EvaluationError("design lock manifest fingerprint differs")
    if len(manifest["ids"]) != ROSTER_COUNT:
        raise EvaluationError("frozen prompt manifest roster is not 160")
    generation = _object(lock.get("generation"), "design generation")
    if (
        generation.get("model_id") != MODEL_ID
        or generation.get("checkpoint_snapshot") != MODEL_SNAPSHOT
        or generation.get("tokenizer_snapshot") != MODEL_SNAPSHOT
        or generation.get("dtype") != "bfloat16"
        or generation.get("attention_backend") != "sdpa"
        or generation.get("decoding") != "greedy"
        or generation.get("seed") != SEED
        or generation.get("total_new_token_budget") != MAX_NEW_TOKENS
    ):
        raise EvaluationError("design lock generation contract differs")
    boundary = _object(generation.get("decision_boundary"), "design boundary")
    if (
        boundary.get("committed_output_tokens") != 32
        or boundary.get("pending_token_output_index") != 31
        or boundary.get("first_affected_prediction_output_index") != 32
        or boundary.get("action_once_before_pending_token_forward")
        is not True
    ):
        raise EvaluationError("design lock decision boundary differs")
    actions = _object(lock.get("actions"), "design actions")
    candidates = actions.get("candidates")
    if not isinstance(candidates, list) or {
        _finite_number(
            _object(item, "design action").get("removal_fraction"),
            "design action ratio",
        )
        for item in candidates
    } != set(ACTION_RATIOS):
        raise EvaluationError("design lock actions differ from frozen roster")
    evaluation = _object(lock.get("evaluation"), "design evaluation")
    bootstrap = _object(evaluation.get("bootstrap"), "design bootstrap")
    if (
        evaluation.get("folds") != FOLD_COUNT
        or evaluation.get("model") != "Ridge(alpha=1)"
        or evaluation.get("tuning") != "none"
        or evaluation.get("prediction_clip") != [-1.0, 1.0]
        or bootstrap.get("replicates") != BOOTSTRAP_REPLICATES
        or bootstrap.get("seed") != BOOTSTRAP_SEED
        or bootstrap.get("unit") != "prompt cluster"
        or bootstrap.get("preserve_cluster_multiplicity") is not True
    ):
        raise EvaluationError("design lock evaluation contract differs")


def _validate_experiment_lock(
    lock: Mapping[str, object],
    path: Path,
    manifest_sha: str,
    manifest: Mapping[str, object],
    design_lock: Mapping[str, object],
) -> None:
    """Validate the complete collection lock against the design lock."""
    if lock.get("schema_version") != "herald_v3.pilot_lock.v1":
        raise EvaluationError(f"unsupported experiment lock schema: {path}")
    if set(lock) != {
        "schema_version",
        "model_argument",
        "model",
        "prompt_manifest",
        "configuration",
        "source_manifest",
        "environment",
        "wrapper",
    }:
        raise EvaluationError(
            f"experiment lock fields are incomplete: {path}"
        )
    model_argument = lock.get("model_argument")
    if not isinstance(model_argument, str) or not model_argument.endswith(
        MODEL_SNAPSHOT
    ):
        raise EvaluationError("experiment lock model argument is missing")
    prompt_manifest = _object(
        lock.get("prompt_manifest"), "lock prompt_manifest"
    )
    recorded_manifest_path = prompt_manifest.get("path")
    if set(prompt_manifest) != {
        "path",
        "file_sha256",
        "manifest_sha256",
        "prompt_ids",
    } or (
        not isinstance(recorded_manifest_path, str)
        or not recorded_manifest_path
        or prompt_manifest.get("file_sha256") != manifest_sha
        or prompt_manifest.get("manifest_sha256")
        != manifest["declared_sha256"]
        or prompt_manifest.get("prompt_ids") != list(manifest["ids"])
    ):
        raise EvaluationError("experiment lock prompt manifest differs")
    configuration = _object(lock.get("configuration"), "lock configuration")
    if set(configuration) != {
        "target_eligible",
        "max_new_tokens",
        "decision_tokens",
        "actions",
        "seed",
        "eos_ids",
    } or (
        configuration.get("target_eligible") != TARGET_ELIGIBLE
        or configuration.get("max_new_tokens") != MAX_NEW_TOKENS
        or configuration.get("decision_tokens") != 32
        or configuration.get("seed") != SEED
        or configuration.get("eos_ids") != [151643, 151645]
    ):
        raise EvaluationError("experiment lock configuration differs")
    actions = configuration.get("actions")
    if not isinstance(actions, list) or {
        _finite_number(
            _object(item, "locked action").get("removal_fraction"),
            "locked action ratio",
        )
        for item in actions
    } != set(ACTION_RATIOS):
        raise EvaluationError("experiment lock actions differ from design")
    model = _object(lock.get("model"), "lock model")
    model_path = model.get("name_or_path")
    if (
        not isinstance(model_path, str)
        or not model_path.endswith(MODEL_SNAPSHOT)
        or model.get("checkpoint_revision") != MODEL_SNAPSHOT
    ):
        raise EvaluationError("experiment lock model snapshot differs")
    for key in ("source_manifest", "environment", "wrapper"):
        if not isinstance(lock.get(key), dict):
            raise EvaluationError(f"experiment lock {key} is missing")


def _rows_from_run(
    document: Mapping[str, object],
    path: Path,
    manifest_folds: Mapping[str, int],
    seen_prompt_ids: set[str],
    experiment_lock: Mapping[str, object] | None = None,
) -> list[PilotRow]:
    if experiment_lock is not None:
        for key in ("source_manifest", "model", "environment"):
            expected = experiment_lock.get(key)
            if expected is not None and document.get(key) != expected:
                raise EvaluationError(
                    f"{key} mismatch with experiment lock: {path}"
                )
    results = document.get("results")
    if not isinstance(results, list):
        raise EvaluationError(f"run artifact has no results list: {path}")
    configuration = document.get("configuration")
    if isinstance(configuration, dict) and configuration.get(
        "decision_tokens"
    ) not in (
        None,
        32,
    ):
        raise EvaluationError(
            f"run artifact decision boundary is not 32: {path}"
        )
    parsed: list[PilotRow] = []
    for raw_row in results:
        if not isinstance(raw_row, dict):
            raise EvaluationError(f"run result row is not an object: {path}")
        prompt = raw_row.get("prompt")
        if not isinstance(prompt, dict):
            raise EvaluationError(f"run result has no prompt object: {path}")
        prompt_id = _string(prompt.get("prompt_id"), "prompt_id")
        if prompt_id not in manifest_folds:
            raise EvaluationError(
                f"run prompt absent from frozen manifest: {prompt_id}"
            )
        if prompt_id in seen_prompt_ids:
            raise EvaluationError(
                f"duplicate prompt in collection: {prompt_id}"
            )
        seen_prompt_ids.add(prompt_id)
        status = raw_row.get("status")
        if status == "ineligible":
            continue
        if status != "accepted":
            raise EvaluationError(
                f"prompt {prompt_id} is not an accepted run row, "
                f"status={status!r}"
            )
        fold = _integer(prompt.get("fold"), f"fold for {prompt_id}")
        if fold != manifest_folds[prompt_id]:
            raise EvaluationError(
                f"fold leakage or mismatch for prompt {prompt_id}"
            )
        acceptance = _object(raw_row.get("acceptance"), "acceptance")
        if acceptance.get("passed") is not True:
            raise EvaluationError(
                f"accepted row did not pass acceptance: {prompt_id}"
            )
        boundary = _object(acceptance.get("boundary"), "boundary")
        if boundary.get("generated_count") not in (None, 32):
            raise EvaluationError(
                f"unexpected boundary length for prompt {prompt_id}"
            )
        cache_bytes = _finite_number(
            boundary.get("cache_bytes"),
            f"pre-action cache size for {prompt_id}",
            lower=0.0,
        )
        tokenization = _object(raw_row.get("tokenization"), "tokenization")
        prompt_tokens = _finite_number(
            tokenization.get("input_length"),
            f"prompt token count for {prompt_id}",
            lower=1.0,
        )
        scores = _object(raw_row.get("scores"), "scores")
        reference_score = _object(scores.get("reference"), "reference score")
        reference_loose = _finite_number(
            reference_score.get("loose"),
            f"reference loose score for {prompt_id}",
            lower=0.0,
            upper=1.0,
        )
        reference_strict = _finite_number(
            reference_score.get("strict"),
            f"reference strict score for {prompt_id}",
            lower=0.0,
            upper=1.0,
        )
        action_scores = _object(scores.get("actions"), "action scores")
        arms = acceptance.get("action_arms")
        if not isinstance(arms, list) or len(arms) != 2:
            raise EvaluationError(
                f"prompt {prompt_id} must have exactly two action arms"
            )
        action_rows: dict[str, PilotRow] = {}
        for arm in arms:
            arm_object = _object(arm, "action arm")
            action = _object(arm_object.get("action"), "action")
            ratio = _finite_number(
                action.get("removal_fraction"),
                "action removal fraction",
                lower=0.0,
                upper=1.0,
            )
            action_id = _string(action.get("action_id"), "action_id")
            expected_id = f"knorm:{ratio:.6g}"
            if action_id != expected_id or action_id not in ACTION_IDS:
                raise EvaluationError(
                    f"unexpected action id for prompt {prompt_id}: "
                    f"{action_id}"
                )
            if action_id in action_rows:
                raise EvaluationError(
                    f"duplicate action for prompt {prompt_id}: {action_id}"
                )
            probe_enabled = arm_object.get("probe_enabled")
            probe = arm_object.get("probe")
            if probe_enabled is not True or not isinstance(probe, dict):
                raise EvaluationError(
                    f"missing decision probe for {prompt_id} {action_id}"
                )
            if probe.get("finite") is not True:
                raise EvaluationError(
                    f"nonfinite decision probe for {prompt_id} {action_id}"
                )
            probe_action = _object(probe.get("action"), "probe action")
            if probe_action.get("action_id") != action_id:
                raise EvaluationError(
                    f"probe action lineage mismatch for {prompt_id} "
                    f"{action_id}"
                )
            reference_entropy = _finite_number(
                probe.get("reference_entropy"), "reference entropy"
            )
            reference_margin = _finite_number(
                probe.get("reference_top2_margin"), "reference top-two margin"
            )
            action_entropy = _finite_number(
                probe.get("action_entropy"), "action entropy"
            )
            action_margin = _finite_number(
                probe.get("action_top2_margin"), "action top-two margin"
            )
            argmax_match = probe.get("argmax_match")
            if not isinstance(argmax_match, bool):
                raise EvaluationError(
                    f"argmax_match must be boolean for {prompt_id} "
                    f"{action_id}"
                )
            js_divergence = _finite_number(
                probe.get("js_divergence"), "JS divergence", lower=0.0
            )
            score = action_scores.get(action_id)
            if not isinstance(score, dict):
                raise EvaluationError(
                    f"missing score for {prompt_id} {action_id}"
                )
            action_score = _object(score.get("action"), "action score")
            action_loose = _finite_number(
                action_score.get("loose"),
                "action loose score",
                lower=0.0,
                upper=1.0,
            )
            action_strict = _finite_number(
                action_score.get("strict"),
                "action strict score",
                lower=0.0,
                upper=1.0,
            )
            d_loose = _finite_number(
                score.get("d_loose"), "signed loose score"
            )
            d_strict = _finite_number(
                score.get("d_strict"), "signed strict score"
            )
            if not math.isclose(
                d_loose,
                reference_loose - action_loose,
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise EvaluationError(
                    f"bad signed loose score for {prompt_id} {action_id}"
                )
            if not math.isclose(
                d_strict,
                reference_strict - action_strict,
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise EvaluationError(
                    f"bad signed strict score for {prompt_id} {action_id}"
                )
            action_rows[action_id] = PilotRow(
                prompt_id=prompt_id,
                fold=fold,
                action_id=action_id,
                removal_fraction=ratio,
                prompt_token_count=prompt_tokens,
                decision_index32=32.0,
                pre_action_cache_size=cache_bytes,
                reference_entropy=reference_entropy,
                reference_top2_margin=reference_margin,
                action_reference_entropy_delta=action_entropy
                - reference_entropy,
                action_reference_margin_delta=action_margin
                - reference_margin,
                argmax_match=argmax_match,
                js_divergence=js_divergence,
                target_loose=d_loose,
                target_strict=d_strict,
            )
        if set(action_rows) != set(ACTION_IDS) or set(action_scores) != set(
            ACTION_IDS
        ):
            raise EvaluationError(
                f"prompt {prompt_id} is missing one frozen action"
            )
        reference_values = [
            (row.reference_entropy, row.reference_top2_margin)
            for row in action_rows.values()
        ]
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for values in reference_values[1:]
            for left, right in zip(values, reference_values[0], strict=True)
        ):
            raise EvaluationError(
                f"reference probe differs between actions for {prompt_id}"
            )
        parsed.extend(action_rows.values())
    return parsed


def _validate_rows(
    rows: Sequence[PilotRow],
    manifest_folds: Mapping[str, int] | None,
    minimum_eligible: int,
) -> None:
    if not rows:
        raise EvaluationError(
            "pilot collection contains no eligible action rows"
        )
    prompt_groups: dict[str, list[PilotRow]] = defaultdict(list)
    for row in rows:
        prompt_groups[row.prompt_id].append(row)
        if row.action_id not in ACTION_IDS:
            raise EvaluationError(f"unknown action id: {row.action_id}")
        if (
            manifest_folds is not None
            and manifest_folds.get(row.prompt_id) != row.fold
        ):
            raise EvaluationError(
                f"fold leakage or mismatch for prompt {row.prompt_id}"
            )
        for value in row.feature_values(BLOCK_FEATURES["B3"]):
            if not math.isfinite(value):
                raise EvaluationError(
                    f"nonfinite feature for prompt {row.prompt_id}"
                )
        for value in (row.target_loose, row.target_strict):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise EvaluationError(
                    f"bad signed target for prompt {row.prompt_id}"
                )
    prompt_count = len(prompt_groups)
    if prompt_count < minimum_eligible:
        raise EvaluationError(
            f"only {prompt_count} eligible prompts, need at least "
            f"{minimum_eligible}"
        )
    for prompt_id, group in prompt_groups.items():
        if len(group) != 2 or {row.action_id for row in group} != set(
            ACTION_IDS
        ):
            raise EvaluationError(
                f"prompt {prompt_id} must have exactly two action rows"
            )
        if len({row.fold for row in group}) != 1:
            raise EvaluationError(f"fold leakage within prompt {prompt_id}")
    _validate_fold_assignments(rows)


def _validate_fold_assignments(rows: Sequence[PilotRow]) -> None:
    folds = {row.fold for row in rows}
    if folds != set(range(FOLD_COUNT)):
        raise EvaluationError(
            "pilot rows must use all five prompt folds exactly"
        )
    by_prompt: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        by_prompt[row.prompt_id].add(row.fold)
    if any(len(values) != 1 for values in by_prompt.values()):
        raise EvaluationError("prompt appears in more than one fold")


def _metrics(
    rows: Sequence[PilotRow], predictions: np.ndarray, target_field: str
) -> dict[str, object]:
    values = np.asarray(
        [getattr(row, target_field) for row in rows], dtype=float
    )
    errors = np.asarray(predictions, dtype=float) - values
    prompt_mse = _prompt_value_means(rows, errors**2)
    prompt_mae = _prompt_value_means(rows, np.abs(errors))
    prompt_bias = _prompt_value_means(rows, errors)
    action_metrics: dict[str, dict[str, float]] = {}
    for action_id in ACTION_IDS:
        action_values = errors[[row.action_id == action_id for row in rows]]
        action_metrics[action_id] = {
            "mse": float(np.mean(action_values**2)),
            "mae": float(np.mean(np.abs(action_values))),
            "bias": float(np.mean(action_values)),
        }
    sign_metrics: dict[str, dict[str, float | int]] = {}
    for label, mask in (
        ("positive", values > 0.0),
        ("zero", values == 0.0),
        ("negative", values < 0.0),
    ):
        sign_metrics[label] = {"count": int(mask.sum())}
        if mask.any():
            subset = errors[mask]
            sign_metrics[label].update(
                {
                    "mse": float(np.mean(subset**2)),
                    "mae": float(np.mean(np.abs(subset))),
                    "bias": float(np.mean(subset)),
                }
            )
    return {
        "prompt_count": len(prompt_mse),
        "row_count": len(rows),
        "mse": float(np.mean(prompt_mse)),
        "mae": float(np.mean(prompt_mae)),
        "bias": float(np.mean(prompt_bias)),
        "action": action_metrics,
        "sign_subset": sign_metrics,
    }


def _comparisons(
    metrics: Mapping[str, Mapping[str, object]],
    bootstrap_values: Mapping[str, np.ndarray],
    rows: Sequence[PilotRow],
    target_field: str,
) -> dict[str, object]:
    b3_mse = float(metrics["B3"]["mse"])
    nonzero_prompt_count = len(
        {row.prompt_id for row in rows if getattr(row, target_field) != 0.0}
    )
    result: dict[str, object] = {
        "triage_floor": {
            "eligible_prompt_count": len({row.prompt_id for row in rows}),
            "minimum_eligible_prompts": MIN_ELIGIBLE_PROMPTS,
            "nonzero_prompt_count": nonzero_prompt_count,
            "minimum": NONZERO_PROMPT_FLOOR,
            "met": (
                len({row.prompt_id for row in rows}) >= MIN_ELIGIBLE_PROMPTS
                and nonzero_prompt_count >= NONZERO_PROMPT_FLOOR
            ),
        },
        "against": {},
    }
    against = result["against"]
    assert isinstance(against, dict)
    for baseline in ("B0", "B1", "B2", "actionmean"):
        baseline_mse = float(metrics[baseline]["mse"])
        point_gain = baseline_mse - b3_mse
        improvement = bootstrap_values[baseline] - bootstrap_values["B3"]
        lower = float(np.quantile(improvement, 0.025))
        upper = float(np.quantile(improvement, 0.975))
        relative = point_gain / baseline_mse if baseline_mse != 0.0 else None
        against[baseline] = {
            "mse_difference_baseline_minus_B3": point_gain,
            "relative_skill": relative,
            "paired_interval_95": [lower, upper],
            "interval_lower_bound_gt_zero": lower > 0.0,
            "gain_at_least_5_percent": (
                relative is not None and relative >= 0.05
            ),
            "triage_pass": (
                relative is not None
                and relative >= 0.05
                and lower > 0.0
                and len({row.prompt_id for row in rows})
                >= MIN_ELIGIBLE_PROMPTS
                and nonzero_prompt_count >= NONZERO_PROMPT_FLOOR
            ),
        }
    result["best_comparator_by_mse"] = min(
        ("B0", "B1", "B2", "actionmean"),
        key=lambda name: float(metrics[name]["mse"]),
    )
    result["all_comparators_triage_pass"] = all(
        bool(value["triage_pass"]) for value in against.values()
    )
    return result


def _prompt_error_arrays(
    rows: Sequence[PilotRow], errors: np.ndarray
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, error in zip(rows, errors, strict=True):
        grouped[row.prompt_id].append(float(error))
    return np.asarray(
        [np.mean(values) for values in grouped.values()], dtype=float
    )


def _prompt_value_means(
    rows: Sequence[PilotRow], values: np.ndarray
) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        grouped[row.prompt_id].append(float(value))
    return np.asarray(
        [np.mean(items) for items in grouped.values()], dtype=float
    )


def _prompt_error_means(
    rows: Sequence[PilotRow], predictions: np.ndarray, target_field: str
) -> np.ndarray:
    values = np.asarray(
        [getattr(row, target_field) for row in rows], dtype=float
    )
    return _prompt_error_arrays(rows, (predictions - values) ** 2)


def _distribution(values: Iterable[float]) -> dict[str, object]:
    array = np.asarray(list(values), dtype=float)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "positive_count": int(np.sum(array > 0.0)),
        "zero_count": int(np.sum(array == 0.0)),
        "negative_count": int(np.sum(array < 0.0)),
    }


def _load_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(
            f"cannot read JSON artifact {path}: {error}"
        ) from error
    return _json_object(value, str(path))


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be a JSON object")
    return value


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvaluationError(f"{label} must be an object")
    return value


def _required_id_list(
    document: Mapping[str, object], key: str, label: str
) -> tuple[str, ...]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise EvaluationError(f"{label} must contain a nonempty {key} list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise EvaluationError(f"{label} {key} entries must be strings")
        result.append(item)
    return tuple(result)


def _optional_string(document: Mapping[str, object], key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise EvaluationError(f"{key} must be a string")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{label} must be a nonempty string")
    return value


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvaluationError(f"{label} must be an integer")
    return value


def _finite_number(
    value: object,
    label: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise EvaluationError(f"{label} must be finite")
    if lower is not None and number < lower:
        raise EvaluationError(f"{label} below {lower}")
    if upper is not None and number > upper:
        raise EvaluationError(f"{label} above {upper}")
    return number


def _json_value(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    raise SystemExit(main())
