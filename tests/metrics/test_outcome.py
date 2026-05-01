from pathlib import Path

import polars as pl

from herald.metrics.outcome import build


def _runs(final: Path) -> None:
    df = pl.DataFrame(
        {
            "run_id": ["b1", "b2", "c1", "c2"],
            "prompt_id": ["p1", "p2", "p1", "p2"],
            "press": ["none", "none", "snapkv", "snapkv"],
            "compression_ratio": [0.0, 0.0, 0.875, 0.875],
            "baseline_run_id": ["b1", "b2", "b1", "b2"],
            "correct": [True, False, False, True],
        }
    )
    df.write_parquet(final / "runs.parquet")


def test_outcome_paired_cells(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    _runs(final)
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "outcome.parquet")
    row = df.row(0, named=True)
    assert row["press"] == "snapkv"
    assert row["compression_ratio"] == 0.875
    assert row["gross_harm"] == 0.5
    assert row["gross_help"] == 0.5
    assert row["net_delta"] == 0.0
