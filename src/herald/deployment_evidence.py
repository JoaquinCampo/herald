"""Verification for hash-backed deployment evidence manifests."""

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

LIVE_EVIDENCE_SCHEMA_VERSION = 1
END_TO_END_RETAINED_KV_CACHE = "end_to_end_retained_kv_cache"


def candidate_id(
    compressor: str,
    ratio: float,
    sustain_interval: int | None,
) -> str:
    """Return the identity of one live deployment candidate."""
    if not compressor:
        raise ValueError("compressor must not be empty")
    if not isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("ratio must be finite and in (0, 1]")
    if sustain_interval is None:
        mode = "one_shot"
    else:
        if sustain_interval <= 0:
            raise ValueError("sustain_interval must be positive")
        mode = f"sustained_every_{sustain_interval}"
    return f"{compressor}|{ratio!r}|{mode}"


def directory_sha256(root: Path) -> str:
    """Hash regular files and contents in a directory deterministically."""
    if not root.is_dir():
        raise ValueError(f"artifact directory does not exist: {root}")
    resolved_root = root.resolve()
    digest = sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if not resolved.is_relative_to(resolved_root):
            raise ValueError(f"artifact path escapes root: {path}")
        digest.update(resolved.relative_to(resolved_root).as_posix().encode())
        digest.update(b"\0")
        with resolved.open("rb") as file:
            while chunk := file.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def create_live_run_manifest(
    run_config: Mapping[str, Any],
) -> dict[str, object]:
    """Bind a fresh live-run configuration to a deterministic identity."""
    config = _validate_live_run_config(run_config)
    config_bytes = _canonical_json(config)
    return {
        "schema_version": LIVE_EVIDENCE_SCHEMA_VERSION,
        "run_id": sha256(config_bytes).hexdigest(),
        "run_config": config,
    }


def initialize_live_run(
    run_dir: Path,
    run_config: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, object]:
    """Create fresh run identity or verify an explicitly resumed run."""
    expected = create_live_run_manifest(run_config)
    manifest_path = run_dir / "run_manifest.json"
    if run_dir.exists():
        if not run_dir.is_dir():
            raise ValueError(
                f"live output path is not a directory: {run_dir}"
            )
        if any(run_dir.iterdir()):
            if not resume:
                raise ValueError(
                    "live runs require a fresh empty output directory"
                )
            if not manifest_path.is_file():
                raise ValueError("resumed live run lacks run_manifest.json")
            actual = _read_live_manifest(manifest_path)
            if actual != expected:
                raise ValueError(
                    "resumed live run configuration does not match"
                )
            return actual
        if resume:
            raise ValueError("cannot resume an empty live output directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_canonical_json(expected).decode() + "\n")
    return expected


def verify_live_run(
    manifest: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    *,
    require_complete: bool = False,
) -> list[str]:
    """Return identity, scope, and uniqueness errors for live evidence."""
    config = _live_run_config(manifest)
    run_id = str(manifest["run_id"])
    prompt_ids = set(_unique_strings(config["prompt_ids"], "prompt_ids"))
    candidate_ids = set(
        _unique_strings(config["candidate_ids"], "candidate_ids")
    )
    candidate_prompt_ids = _candidate_prompt_ids(
        config["candidate_prompt_ids"],
        candidate_ids,
        prompt_ids,
    )
    errors: list[str] = []
    baseline_prompt_ids: set[str] = set()
    episode_keys: set[str] = set()
    episode_candidate_prompts: set[tuple[str, str]] = set()

    for baseline in baselines:
        prompt_id = _record_string(baseline, "prompt_id", "baseline", errors)
        if prompt_id is None:
            continue
        if prompt_id in baseline_prompt_ids:
            errors.append(f"duplicate baseline prompt_id: {prompt_id}")
        baseline_prompt_ids.add(prompt_id)
        _verify_record_identity(
            baseline,
            kind="baseline",
            prompt_id=prompt_id,
            run_id=run_id,
            errors=errors,
        )
        if prompt_id not in prompt_ids:
            errors.append(f"baseline prompt_id not in manifest: {prompt_id}")

    for episode in episodes:
        prompt_id = _record_string(episode, "prompt_id", "episode", errors)
        key = _record_string(episode, "key", "episode", errors)
        if prompt_id is None or key is None:
            continue
        if key in episode_keys:
            errors.append(f"duplicate episode key: {key}")
        episode_keys.add(key)
        _verify_record_identity(
            episode,
            kind="episode",
            prompt_id=prompt_id,
            run_id=run_id,
            errors=errors,
        )
        if prompt_id not in prompt_ids:
            errors.append(f"episode prompt_id not in manifest: {prompt_id}")
        actual_candidate_id = _record_candidate_id(episode, prompt_id, errors)
        stored_candidate_id = episode.get("candidate_id")
        if actual_candidate_id is not None:
            if key != f"{actual_candidate_id}|{prompt_id}":
                errors.append(f"episode key mismatch for {prompt_id}")
            candidate_prompt = (actual_candidate_id, prompt_id)
            if candidate_prompt in episode_candidate_prompts:
                errors.append(
                    "duplicate candidate evidence for prompt_id: "
                    f"{actual_candidate_id}|{prompt_id}"
                )
            episode_candidate_prompts.add(candidate_prompt)
            if stored_candidate_id != actual_candidate_id:
                errors.append(
                    f"candidate_id mismatch for episode {prompt_id}"
                )
            if actual_candidate_id not in candidate_ids:
                errors.append(
                    "episode candidate_id not in manifest: "
                    f"{actual_candidate_id}"
                )
        if prompt_id not in baseline_prompt_ids:
            errors.append(
                f"missing baseline for episode prompt_id: {prompt_id}"
            )
    if require_complete:
        for prompt_id in sorted(prompt_ids - baseline_prompt_ids):
            errors.append(f"missing baseline prompt_id: {prompt_id}")
        for (
            current_candidate_id,
            expected_prompt_ids,
        ) in candidate_prompt_ids.items():
            for prompt_id in expected_prompt_ids:
                candidate_prompt = (current_candidate_id, prompt_id)
                if candidate_prompt not in episode_candidate_prompts:
                    errors.append(
                        "missing episode for candidate prompt: "
                        f"{current_candidate_id}|{prompt_id}"
                    )
    return errors


def verify_manifest(manifest: Mapping[str, Any], root: Path) -> list[str]:
    """Return artifact errors, rejecting malformed manifest entries."""
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("manifest artifacts must be a list")

    resolved_root = root.resolve()
    errors: list[str] = []
    for entry in artifacts:
        path, expected_hash = _artifact_fields(entry)
        artifact = (resolved_root / path).resolve()
        if not artifact.is_relative_to(resolved_root):
            raise ValueError(f"artifact path escapes root: {path}")
        if not artifact.is_file():
            errors.append(f"missing artifact: {path}")
            continue
        if _sha256(artifact) != expected_hash:
            errors.append(f"sha256 mismatch: {path}")
    return errors


def load_live_run_manifest(path: Path) -> dict[str, object]:
    """Load and validate a previously created live-run manifest."""
    return _read_live_manifest(path)


def _read_live_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"could not read live evidence manifest: {path}"
        ) from error
    if not isinstance(value, Mapping):
        raise ValueError("live evidence manifest must be an object")
    _live_run_config(value)
    return dict(value)


def _validate_live_run_config(
    run_config: Mapping[str, Any],
) -> dict[str, object]:
    task = run_config.get("task")
    if not isinstance(task, str) or not task:
        raise ValueError("live run config needs a non-empty task")
    prompt_ids = _unique_strings(run_config.get("prompt_ids"), "prompt_ids")
    candidate_ids = _unique_strings(
        run_config.get("candidate_ids"),
        "candidate_ids",
    )
    candidate_prompt_ids = _candidate_prompt_ids(
        run_config.get("candidate_prompt_ids"),
        set(candidate_ids),
        set(prompt_ids),
    )
    return {
        **dict(run_config),
        "task": task,
        "prompt_ids": prompt_ids,
        "candidate_ids": candidate_ids,
        "candidate_prompt_ids": candidate_prompt_ids,
    }


def _live_run_config(manifest: Mapping[str, Any]) -> dict[str, object]:
    schema_version = manifest.get("schema_version")
    if schema_version != LIVE_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported live evidence schema_version")
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or len(run_id) != 64:
        raise ValueError("live evidence manifest needs a sha256 run_id")
    config = manifest.get("run_config")
    if not isinstance(config, Mapping):
        raise ValueError("live evidence manifest needs a run_config object")
    normalized = _validate_live_run_config(config)
    actual_run_id = sha256(_canonical_json(normalized)).hexdigest()
    if run_id != actual_run_id:
        raise ValueError(
            "live evidence manifest run_id does not match config"
        )
    return normalized


def _canonical_json(value: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "live run config must be JSON serializable"
        ) from error


def _unique_strings(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"live run config needs non-empty {name}")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"live run config {name} must contain strings")
    if len(set(value)) != len(value):
        raise ValueError(f"live run config {name} must be unique")
    return list(value)


def _candidate_prompt_ids(
    value: Any,
    candidate_ids: set[str],
    prompt_ids: set[str],
) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        raise ValueError("live run config needs candidate_prompt_ids object")
    if set(value) != candidate_ids:
        raise ValueError(
            "live run config candidate_prompt_ids must match candidate_ids"
        )
    result: dict[str, list[str]] = {}
    for candidate, value_prompt_ids in value.items():
        if not isinstance(candidate, str):
            raise ValueError(
                "live run config candidate_prompt_ids has invalid key"
            )
        current_prompt_ids = _unique_strings(
            value_prompt_ids,
            f"candidate_prompt_ids[{candidate}]",
        )
        if not set(current_prompt_ids).issubset(prompt_ids):
            raise ValueError(
                "live run config candidate_prompt_ids includes unknown prompt"
            )
        result[candidate] = current_prompt_ids
    return result


def _record_string(
    record: Mapping[str, Any],
    key: str,
    kind: str,
    errors: list[str],
) -> str | None:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        errors.append(f"missing {key} for {kind}")
        return None
    return value


def _verify_record_identity(
    record: Mapping[str, Any],
    *,
    kind: str,
    prompt_id: str,
    run_id: str,
    errors: list[str],
) -> None:
    if record.get("run_id") != run_id:
        errors.append(f"run_id mismatch for {kind} {prompt_id}")
    if record.get("kv_measurement_scope") != END_TO_END_RETAINED_KV_CACHE:
        errors.append(f"invalid kv_measurement_scope for {kind} {prompt_id}")


def _record_candidate_id(
    episode: Mapping[str, Any],
    prompt_id: str,
    errors: list[str],
) -> str | None:
    compressor = episode.get("compressor")
    ratio = episode.get("ratio")
    sustain_interval = episode.get("sustain_interval")
    if not isinstance(compressor, str) or not isinstance(ratio, int | float):
        errors.append(f"invalid candidate fields for episode {prompt_id}")
        return None
    if sustain_interval is not None and not isinstance(sustain_interval, int):
        errors.append(f"invalid sustain_interval for episode {prompt_id}")
        return None
    try:
        return candidate_id(compressor, float(ratio), sustain_interval)
    except ValueError as error:
        errors.append(
            f"invalid candidate_id for episode {prompt_id}: {error}"
        )
        return None


def _artifact_fields(entry: Any) -> tuple[str, str]:
    if not isinstance(entry, Mapping):
        raise ValueError("artifact entry must be an object")

    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("artifact entry needs a non-empty path")

    digest = entry.get("sha256")
    if not isinstance(digest, str):
        raise ValueError("artifact entry needs a sha256 string")
    if len(digest) != 64 or any(
        char not in "0123456789abcdef" for char in digest
    ):
        raise ValueError("artifact sha256 must be 64 hexadecimal characters")
    return path, digest


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
