"""Tests for the CLI fixture."""

import json
from pathlib import Path

from demo_cli_app.cli import run


def test_cli_prints_normalized_json(tmp_path: Path, capsys) -> None:
    """The CLI writes machine-readable output to stdout."""
    config_path = tmp_path / "config.json"
    config_path.write_text('{"Name": "Ada"}', encoding="utf-8")

    assert run([str(config_path), "--env-prefix", "APP"]) == 0

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"env_prefix": "APP", "name": "Ada"}
    assert captured.err == ""


def test_cli_reports_config_errors_on_stderr(tmp_path: Path, capsys) -> None:
    """Malformed JSON returns a nonzero exit and stderr diagnostic."""
    config_path = tmp_path / "config.json"
    config_path.write_text("not json", encoding="utf-8")

    assert run([str(config_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid JSON config" in captured.err
