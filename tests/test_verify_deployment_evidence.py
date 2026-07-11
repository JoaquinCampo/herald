import hashlib
import subprocess
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


def test_verify_manifest_reads_repository_artifacts_from_frozen_revision(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / "implementation.py"
    artifact.write_text("FROZEN = True\n")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", artifact.name], check=True
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=HERALD test",
            "-c",
            "user.email=herald@example.invalid",
            "commit",
            "-qm",
            "freeze evidence",
        ],
        check=True,
    )
    revision = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    entry = _entry(artifact)
    artifact.write_text("FROZEN = False\n")
    manifest = {
        "artifacts": [],
        "repository_revision": revision,
        "repository_artifacts": [entry],
    }

    assert verify_manifest(manifest, tmp_path) == []

    manifest["repository_artifacts"] = [_entry(artifact)]
    assert verify_manifest(manifest, tmp_path) == [
        "sha256 mismatch at repository revision: implementation.py"
    ]


def test_verify_manifest_still_checks_filesystem_artifacts_with_revision(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    artifact = tmp_path / "result.json"
    artifact.write_text('{"ok": true}\n')
    entry = _entry(artifact)
    artifact.write_text('{"ok": false}\n')
    manifest = {
        "artifacts": [entry],
        "repository_revision": "0" * 40,
        "repository_artifacts": [],
    }

    assert verify_manifest(manifest, tmp_path) == [
        "sha256 mismatch: result.json"
    ]


def test_verify_manifest_rejects_invalid_repository_provenance(
    tmp_path: Path,
) -> None:
    manifest = {
        "artifacts": [],
        "repository_revision": "not-a-revision",
        "repository_artifacts": [
            {"path": "implementation.py", "sha256": "0" * 64}
        ],
    }
    with pytest.raises(ValueError, match="repository_revision"):
        verify_manifest(manifest, tmp_path)

    manifest["repository_revision"] = "0" * 40
    manifest["repository_artifacts"] = [
        {"path": "../implementation.py", "sha256": "0" * 64}
    ]
    with pytest.raises(ValueError, match="escapes repository"):
        verify_manifest(manifest, tmp_path)
