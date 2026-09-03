# pyright: reportMissingImports=false

"""Bind switch data to the frozen generation-sweep configuration."""

import json
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from herald.config import Config
from herald.features import FEATURE_NAMES
from herald.storage import (
    hybrid_record_keys,
    load_reference,
    reference_done,
    safe_id,
)
from herald.switch_dataset import parse_hybrid_shard

SWEEP_CONFIG_SHA256_METADATA_KEY = b"herald.sweep_config_sha256"
SOURCE_MANIFEST_SHA256_METADATA_KEY = b"herald.source_manifest_sha256"
VALIDATION_STATUS_METADATA_KEY = b"herald.validation_status"
ARTIFACT_AGGREGATE_SHA256_METADATA_KEY = b"herald.artifact_aggregate_sha256"


TABLE_CONTENT_SHA256_METADATA_KEY = b"herald.table_content_sha256"


def initialize_sweep_config(results_dir: Path, config: Config) -> Path:
    """Write a sweep config once, refusing to mix it with prior results."""
    config_path = results_dir / "config.json"
    expected = config.model_dump_json(indent=2)
    if config_path.is_file():
        try:
            actual = config_path.read_text()
        except OSError as error:
            raise ValueError(
                f"could not read existing sweep config: {config_path}"
            ) from error
        if actual != expected:
            raise ValueError(
                "existing sweep config does not match invocation"
            )
        return config_path
    if results_dir.exists() and any(results_dir.iterdir()):
        raise ValueError(
            "sweep results directory contains artifacts without config.json"
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(expected)
    return config_path


def _validated_hybrid_keys(path: Path) -> list[tuple[str, int]]:
    """Parse every hybrid row and reject records the dataset would drop."""
    keys: list[tuple[str, int]] = []
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"could not read hybrid shard {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("record is not an object")
            prompt_id = str(record["prompt_id"])
            raw_s = record["s"]
            if isinstance(raw_s, bool):
                raise ValueError("switch position is not an integer")
            switch_s = int(raw_s)
            if isinstance(raw_s, float) and (
                not np.isfinite(raw_s) or not raw_s.is_integer()
            ):
                raise ValueError("switch position is not an integer")
            q = float(record["q"])
            dq = float(record["dq"])
            if not prompt_id or not np.isfinite(q) or not np.isfinite(dq):
                raise ValueError("invalid prompt or score")
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
        ) as error:
            raise ValueError(
                f"invalid hybrid row {path.name}:{line_number}"
            ) from error
        keys.append((prompt_id, switch_s))
    return keys


def validate_sweep_completeness(
    results_dir: Path,
    config: Config,
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    require_hybrid_features: bool = True,
) -> None:
    """Reject missing references or expected hybrid cells before training."""
    selected_models = list(models) if models is not None else config.models
    selected_tasks = list(tasks) if tasks is not None else config.tasks
    unknown_models = set(selected_models) - set(config.models)
    if unknown_models:
        raise ValueError(
            "models not in sweep config: " + ", ".join(sorted(unknown_models))
        )
    unknown_tasks = set(selected_tasks) - set(config.tasks)
    if unknown_tasks:
        raise ValueError(
            "tasks not in sweep config: " + ", ".join(sorted(unknown_tasks))
        )
    errors: list[str] = []
    for model in selected_models:
        for task in selected_tasks:
            expected_shards = {
                (compressor, ratio)
                for compressor in config.compressors
                for ratio in config.ratios
            }
            shard_paths: dict[tuple[str, float], Path] = {}
            hybrid_dir = results_dir / model / task / "hybrids"
            for shard in hybrid_dir.glob("*.jsonl"):
                try:
                    shard_key = parse_hybrid_shard(shard)
                except ValueError:
                    errors.append(
                        f"{model}/{task}: invalid hybrid shard {shard.name}"
                    )
                    continue
                if shard_key not in expected_shards:
                    errors.append(
                        f"{model}/{task}: unexpected hybrid shards "
                        f"({shard.name})"
                    )
                elif shard_key in shard_paths:
                    errors.append(
                        f"{model}/{task}: duplicate hybrid shard "
                        f"for {shard_key}"
                    )
                else:
                    shard_paths[shard_key] = shard
            prompt_ids = reference_done(results_dir, model, task)
            if len(prompt_ids) != config.prompts_per_task:
                errors.append(
                    f"{model}/{task}: expected {config.prompts_per_task} "
                    f"references, found {len(prompt_ids)}"
                )
                continue
            reference_dir = results_dir / model / task / "references"
            expected_reference_names = {
                f"{safe_id(prompt_id)}.json" for prompt_id in prompt_ids
            }
            actual_reference_names = {
                path.name for path in reference_dir.glob("*.json")
            }
            unexpected_references = (
                actual_reference_names - expected_reference_names
            )
            if unexpected_references:
                errors.append(
                    f"{model}/{task}: unexpected reference artifacts "
                    f"({len(unexpected_references)})"
                )
            expected_by_prompt: dict[str, set[int]] = {}
            for prompt_id in prompt_ids:
                feature_path = (
                    results_dir
                    / model
                    / task
                    / "references"
                    / f"{safe_id(prompt_id)}.npy"
                )
                if not feature_path.is_file():
                    errors.append(
                        f"{model}/{task}/{prompt_id}: "
                        "missing reference features"
                    )
                    continue
                try:
                    reference = load_reference(
                        results_dir,
                        model,
                        task,
                        prompt_id,
                    )
                    gen_ids = reference["gen_ids"]
                except (FileNotFoundError, KeyError) as error:
                    errors.append(f"{model}/{task}/{prompt_id}: {error}")
                    continue
                if not isinstance(gen_ids, list) or not gen_ids:
                    errors.append(
                        f"{model}/{task}/{prompt_id}: "
                        "invalid reference gen_ids"
                    )
                    continue
                try:
                    feature_shape = np.load(
                        feature_path,
                        mmap_mode="r",
                        allow_pickle=False,
                    ).shape
                except (OSError, ValueError) as error:
                    errors.append(
                        f"{model}/{task}/{prompt_id}: "
                        f"invalid reference features ({error})"
                    )
                    continue
                expected_shape = (len(gen_ids), len(FEATURE_NAMES))
                if feature_shape != expected_shape:
                    errors.append(
                        f"{model}/{task}/{prompt_id}: reference feature "
                        f"shape {feature_shape} != {expected_shape}"
                    )
                    continue
                expected_by_prompt[prompt_id] = set(
                    range(0, len(gen_ids), config.switch_stride)
                )
            for compressor in config.compressors:
                for ratio in config.ratios:
                    expected_shard = shard_paths.get((compressor, ratio))
                    if expected_shard is None:
                        keys: list[tuple[str, int]] = []
                    else:
                        try:
                            keys = _validated_hybrid_keys(expected_shard)
                        except ValueError as error:
                            errors.append(
                                f"{model}/{task}/{compressor}/{ratio}: "
                                f"{error}"
                            )
                            keys = []
                    actual = set(keys)
                    expected = {
                        (prompt_id, switch_s)
                        for prompt_id, switch_positions in (
                            expected_by_prompt.items()
                        )
                        for switch_s in switch_positions
                    }
                    duplicate_count = len(keys) - len(actual)
                    if duplicate_count:
                        errors.append(
                            f"{model}/{task}/{compressor}/{ratio}: "
                            f"duplicate hybrid cells ({duplicate_count})"
                        )
                    unexpected = actual - expected
                    if unexpected:
                        errors.append(
                            f"{model}/{task}/{compressor}/{ratio}: "
                            f"unexpected hybrid cells ({len(unexpected)})"
                        )
                    missing = expected - actual
                    if missing:
                        errors.append(
                            f"{model}/{task}/{compressor}/{ratio}: "
                            f"missing {len(missing)} hybrid cells"
                        )
                    if require_hybrid_features:
                        feature_keys = set(
                            hybrid_record_keys(
                                results_dir,
                                model,
                                task,
                                compressor,
                                ratio,
                                require_features=True,
                            )
                        )
                        missing_features = expected - feature_keys
                        if missing_features:
                            errors.append(
                                f"{model}/{task}/{compressor}/{ratio}: "
                                "missing hybrid features "
                                f"({len(missing_features)})"
                            )
    if errors:
        raise ValueError("incomplete hybrid sweep: " + "; ".join(errors))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of one regular artifact file."""
    if not path.is_file():
        raise ValueError(f"artifact file does not exist: {path}")
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bind_table_to_sweep_config(table: Any, config_path: Path) -> Any:
    """Attach the exact sweep config digest to a parquet table schema."""
    config_sha256 = sha256_file(config_path)
    metadata = dict(table.schema.metadata or {})
    metadata[SWEEP_CONFIG_SHA256_METADATA_KEY] = config_sha256.encode()
    return table.replace_schema_metadata(metadata)


def table_content_sha256(table: Any) -> str:
    """Hash table rows independently of mutable schema metadata."""
    encoded = [
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        for row in table.to_pylist()
    ]
    digest = sha256()
    for row in sorted(encoded):
        digest.update(row.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def validate_intervention_manifest(
    manifest_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    """Verify a complete faithful-sweep manifest and its raw artifacts."""
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read intervention manifest: {manifest_path}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("intervention manifest must be a JSON object")
    if payload.get("schema_version") != "herald.intervention_manifest.v1":
        raise ValueError("unsupported intervention manifest schema")
    if payload.get("status") != "complete":
        raise ValueError("intervention manifest is not complete")
    config_digest = sha256_file(config_path)
    if payload.get("sweep_config_sha256") != config_digest:
        raise ValueError("intervention manifest config digest does not match")
    intervention_config = config_path.parent / "intervention_config.json"
    if payload.get("intervention_config_sha256") != sha256_file(
        intervention_config
    ):
        raise ValueError(
            "intervention manifest intervention config does not match"
        )

    results_root = config_path.parent.resolve()
    manifest_root = manifest_path.parent.resolve()
    if manifest_root.parent.parent != results_root:
        raise ValueError(
            "intervention manifest is outside the sweep results tree"
        )
    model = manifest_root.parent.name
    task = manifest_root.name
    sweep = load_sweep_config(config_path)
    validate_sweep_completeness(
        results_root,
        sweep,
        models=[model],
        tasks=[task],
        require_hybrid_features=False,
    )
    cumulative = {
        "references": len(reference_done(results_root, model, task)),
        "cells": sum(
            len(
                hybrid_record_keys(
                    results_root,
                    model,
                    task,
                    compressor,
                    ratio,
                )
            )
            for compressor in sweep.compressors
            for ratio in sweep.ratios
        ),
    }
    if payload.get("completion_counts") != cumulative:
        raise ValueError(
            "intervention manifest completion counts do not match artifacts"
        )

    manifest_root = manifest_path.parent
    reference_entries = _validate_manifest_artifacts(
        manifest_root / "references",
        payload.get("reference_artifacts"),
        label="reference",
    )
    hybrid_entries = _validate_manifest_artifacts(
        manifest_root / "hybrids",
        payload.get("hybrid_artifacts"),
        label="hybrid",
    )
    artifact_entries = reference_entries + hybrid_entries
    if payload.get("artifact_files") != artifact_entries:
        raise ValueError("intervention manifest artifact list does not match")
    aggregate = sha256(
        json.dumps(
            artifact_entries,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if payload.get("artifact_aggregate_sha256") != aggregate:
        raise ValueError(
            "intervention manifest aggregate digest does not match"
        )
    scorer = payload.get("scorer")
    if not isinstance(scorer, dict) or scorer.get("id") != (
        "herald.ifeval.instruction_level.v1"
    ):
        raise ValueError("intervention manifest scorer is not supported")
    return payload


def _validate_manifest_artifacts(
    root: Path,
    raw_entries: object,
    *,
    label: str,
) -> list[dict[str, Any]]:
    """Verify one manifest artifact group and reject unlisted files."""
    if not isinstance(raw_entries, list):
        raise ValueError(f"intervention manifest lacks {label} artifacts")
    entries: list[dict[str, Any]] = []
    named: set[str] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError(f"invalid {label} artifact entry")
        name = str(raw.get("path", ""))
        relative = Path(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or name in named
        ):
            raise ValueError(f"invalid {label} artifact path {name!r}")
        named.add(name)
        artifact = root / relative
        try:
            size = int(raw["size"])
            digest = str(raw["sha256"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {label} artifact entry") from error
        if not artifact.is_file() or artifact.stat().st_size != size:
            raise ValueError(f"{label} artifact size does not match: {name}")
        if sha256_file(artifact) != digest:
            raise ValueError(
                f"{label} artifact digest does not match: {name}"
            )
        entries.append({"path": name, "size": size, "sha256": digest})
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != named:
        raise ValueError(f"{label} artifact file set does not match manifest")
    return entries


def bind_table_to_validated_intervention(
    table: Any,
    config_path: Path,
    manifest_path: Path,
) -> Any:
    """Bind a table to a fully verified faithful intervention sweep."""
    results_root = config_path.parent.resolve()
    manifest_root = manifest_path.parent.resolve()
    if manifest_root.parent.parent != results_root:
        raise ValueError(
            "intervention manifest is outside the sweep results tree"
        )
    model = manifest_root.parent.name
    task = manifest_root.name
    rows = table.to_pylist()
    if not rows or any(
        str(row.get("model")) != model or str(row.get("task")) != task
        for row in rows
    ):
        raise ValueError(
            "switch table identities do not match intervention manifest"
        )
    manifest = validate_intervention_manifest(manifest_path, config_path)
    expected_cells = int(manifest["completion_counts"]["cells"])
    if len(rows) != expected_cells:
        raise ValueError(
            f"switch table row count {len(rows)} != {expected_cells}"
        )
    sweep = load_sweep_config(config_path)
    expected_keys = {
        (compressor, ratio, prompt_id, switch_s)
        for compressor in sweep.compressors
        for ratio in sweep.ratios
        for prompt_id, switch_s in hybrid_record_keys(
            results_root,
            model,
            task,
            compressor,
            ratio,
        )
    }
    try:
        actual_keys = {
            (
                str(row["compressor"]),
                float(row["ratio"]),
                str(row["prompt_id"]),
                int(row["s"]),
            )
            for row in rows
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("switch table cell identity is invalid") from error
    if len(actual_keys) != len(rows) or actual_keys != expected_keys:
        raise ValueError(
            "switch table cells do not exactly match intervention artifacts"
        )
    bound = bind_table_to_sweep_config(table, config_path)
    metadata = dict(bound.schema.metadata or {})
    metadata[SOURCE_MANIFEST_SHA256_METADATA_KEY] = sha256_file(
        manifest_path
    ).encode()
    metadata[TABLE_CONTENT_SHA256_METADATA_KEY] = table_content_sha256(
        bound
    ).encode()
    metadata[VALIDATION_STATUS_METADATA_KEY] = b"validated"
    metadata[ARTIFACT_AGGREGATE_SHA256_METADATA_KEY] = str(
        manifest["artifact_aggregate_sha256"]
    ).encode()
    return bound.replace_schema_metadata(metadata)


def validate_parquet_sweep_config(
    parquet_path: Path,
    config_path: Path,
    *,
    required_compressor: str | None = None,
    expected_statistics_sha256: str | None = None,
) -> Config:
    """Require a parquet table to name this exact compatible sweep config."""
    config_sha256 = sha256_file(config_path)
    config = _load_config(config_path)
    metadata = _parquet_metadata(parquet_path)
    bound_sha256 = metadata.get(SWEEP_CONFIG_SHA256_METADATA_KEY)
    if bound_sha256 is None:
        raise ValueError("switch parquet lacks sweep config sha256 metadata")
    if bound_sha256.decode() != config_sha256:
        raise ValueError("switch parquet sweep config sha256 does not match")
    if (
        required_compressor is not None
        and required_compressor not in config.compressors
    ):
        raise ValueError(
            f"sweep config does not include {required_compressor!r}"
        )
    statistics_match = (
        config.expected_attention_stats_sha256 == expected_statistics_sha256
    )
    if expected_statistics_sha256 is not None and not statistics_match:
        raise ValueError("sweep config statistics digest does not match")
    return config


def load_sweep_config(path: Path) -> Config:
    """Load and validate a serialized sweep configuration."""
    return _load_config(path)


def _load_config(path: Path) -> Config:
    try:
        return Config.model_validate_json(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read sweep config: {path}") from error


def _parquet_metadata(path: Path) -> dict[bytes, bytes]:
    try:
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata  # type: ignore[no-untyped-call]
    except (ImportError, OSError, ValueError) as error:
        raise ValueError(
            f"could not read switch parquet schema: {path}"
        ) from error
    return dict(metadata or {})
