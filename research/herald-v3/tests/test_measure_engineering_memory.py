"""CPU verification for the standalone memory diagnostic."""

# The diagnostic is an executable script rather than an installed module.
# ruff: noqa: E402, I001

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.measure_engineering_memory import measure_memory_modes
from tests.test_engine import _input_ids, _tiny_model


def test_memory_modes_use_equivalent_state_action_and_continuation() -> None:
    result = measure_memory_modes(
        _tiny_model(),
        _input_ids(),
        eos_ids=frozenset(),
        max_new_tokens=34,
        require_cuda=False,
    )

    assert result["passed"] is True
    checks = result["checks"]
    assert isinstance(checks, dict)
    assert all(checks.values())
    modes = result["modes"]
    assert isinstance(modes, dict)
    direct = modes["live_no_reference_no_probe"]
    preserved = modes["preserved_reference_with_probe"]
    assert isinstance(direct, dict)
    assert isinstance(preserved, dict)
    assert direct["reference_cache_preserved"] is False
    assert direct["probe_enabled"] is False
    assert preserved["reference_cache_preserved"] is True
    assert preserved["probe_enabled"] is True
    assert direct["peak_allocated_bytes"] is None
    assert preserved["peak_allocated_bytes"] is None
