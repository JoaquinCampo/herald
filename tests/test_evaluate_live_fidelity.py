# pyright: reportMissingImports=false

"""The fidelity evaluator must follow the compressors in its frozen target."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol, cast

import pytest


class FidelityModule(Protocol):
    def compressors_from_targets(
        self, targets: dict[str, Any]
    ) -> tuple[str, ...]:
        raise NotImplementedError


def _module() -> FidelityModule:
    script = (
        Path(__file__).parents[1] / "scripts" / "evaluate_live_fidelity.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_live_fidelity", script
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load evaluate_live_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(FidelityModule, module)


def test_uses_only_compressors_in_frozen_target() -> None:
    module = _module()
    targets: dict[str, Any] = {
        "compressors": {
            "expected_attention_stats": {},
            "knorm": {},
        }
    }

    expected = ("expected_attention_stats", "knorm")
    assert module.compressors_from_targets(targets) == expected


def test_cli_uses_expected_attention_stats_target(
    tmp_path: Path,
) -> None:
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    (live_dir / "baseline.jsonl").write_text("")
    (live_dir / "episodes.jsonl").write_text("")
    targets = tmp_path / "targets.json"
    targets.write_text(
        json.dumps(
            {
                "compressors": {
                    "expected_attention_stats": {
                        "replay_test": {"n_groups": 5}
                    }
                }
            }
        )
    )
    script = (
        Path(__file__).parents[1] / "scripts" / "evaluate_live_fidelity.py"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--live-dir",
            str(live_dir),
            "--targets",
            str(targets),
            "--skip-static",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "expected_attention_stats: no live episodes yet" in result.stdout


def test_rejects_missing_frozen_compressor_map() -> None:
    module = _module()

    with pytest.raises(ValueError, match="compressors"):
        module.compressors_from_targets({})
