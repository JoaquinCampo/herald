"""Tests for the file-processing fixture."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from file_processing_app.extract import extract_zip


def make_zip(path: Path, members: dict[str, str]) -> None:
    """Write a zip archive with text members."""
    with ZipFile(path, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)


def test_extracts_safe_member(tmp_path: Path) -> None:
    """Safe members are extracted under the output directory."""
    archive_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "out"
    make_zip(archive_path, {"nested/file.txt": "hello"})

    extracted = extract_zip(archive_path, output_dir)

    assert [path.relative_to(output_dir) for path in extracted] == [Path("nested/file.txt")]
    assert (output_dir / "nested" / "file.txt").read_text(encoding="utf-8") == "hello"


def test_rejects_path_traversal_member(tmp_path: Path) -> None:
    """Archive members cannot escape the output directory."""
    archive_path = tmp_path / "archive.zip"
    output_dir = tmp_path / "out"
    make_zip(archive_path, {"../escape.txt": "bad"})

    with pytest.raises(ValueError, match="escapes output directory"):
        extract_zip(archive_path, output_dir)
