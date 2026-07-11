import hashlib
from pathlib import Path

import pytest

from herald.deployment_evidence import (  # pyright: ignore[reportMissingImports]
    verify_manifest,
)


def _entry(path: Path) -> dict[str, str]:
    return {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_verify_manifest_accepts_matching_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"ok": true}\n')
    manifest = {"artifacts": [_entry(artifact)]}

    assert verify_manifest(manifest, tmp_path) == []


def test_verify_manifest_reports_missing_and_modified_artifacts(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"ok": true}\n')
    entry = _entry(artifact)
    artifact.write_text('{"ok": false}\n')
    manifest = {
        "artifacts": [
            entry,
            {
                "path": "missing.json",
                "sha256": "0" * 64,
            },
        ]
    }

    assert verify_manifest(manifest, tmp_path) == [
        "sha256 mismatch: artifact.json",
        "missing artifact: missing.json",
    ]


def test_verify_manifest_rejects_malformed_entries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path"):
        verify_manifest({"artifacts": [{"sha256": "0" * 64}]}, tmp_path)

    with pytest.raises(ValueError, match="sha256"):
        verify_manifest({"artifacts": [{"path": "artifact.json"}]}, tmp_path)

    with pytest.raises(ValueError, match="64 hexadecimal"):
        verify_manifest(
            {"artifacts": [{"path": "artifact.json", "sha256": "x"}]},
            tmp_path,
        )


def test_verify_manifest_rejects_paths_outside_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("outside\n")
    manifest = {
        "artifacts": [{"path": "../outside.json", "sha256": "0" * 64}]
    }

    with pytest.raises(ValueError, match="escapes root"):
        verify_manifest(manifest, tmp_path)


def test_verify_manifest_rejects_symlinks_outside_root(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("outside\n")
    link = tmp_path / "link.json"
    link.symlink_to(outside)
    manifest = {"artifacts": [{"path": link.name, "sha256": "0" * 64}]}

    with pytest.raises(ValueError, match="escapes root"):
        verify_manifest(manifest, tmp_path)
