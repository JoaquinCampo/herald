from pathlib import Path

import polars as pl
import pytest

from herald.metrics.sequence import build


def _make_runs(final: Path) -> None:
    df = pl.DataFrame(
        {
            "run_id": ["base", "comp"],
            "prompt_id": ["p1", "p1"],
            "press": ["none", "snapkv"],
            "compression_ratio": [0.0, 0.875],
            "baseline_run_id": ["base", "base"],
            "generated_text": [
                "The capital is Paris.",
                "The capital is Lyon.",
            ],
        }
    )
    df.write_parquet(final / "runs.parquet")


def test_build_sequence_metrics(tmp_path: Path):
    pytest.importorskip("sentence_transformers")
    final = tmp_path / "final"
    final.mkdir()
    _make_runs(final)
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "sequence_metrics.parquet")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["run_id"] == "comp"
    assert 0.0 <= row["edit_distance_ratio"] <= 1.0
