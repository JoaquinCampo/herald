import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import run_int8_cache  # type: ignore[import-not-found]  # noqa: E402


def test_runner_accepts_knorm_always_on_mechanism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_int8_cache.py",
            "--mechanism",
            "knorm",
            "--out-dir",
            "unused",
        ],
    )

    args = run_int8_cache.parse_args()

    assert args.mechanism == "knorm"
