"""Verification for hash-backed deployment evidence manifests."""

from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any


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
