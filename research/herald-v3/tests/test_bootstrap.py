"""Verify the editable package is importable from this workspace."""

from pathlib import Path

import herald_v3


def test_package_resolves_to_this_workspace() -> None:
    assert herald_v3.__file__ is not None
    expected = Path(__file__).resolve().parents[1] / "src" / "herald_v3"
    assert Path(herald_v3.__file__).resolve().parent == expected
