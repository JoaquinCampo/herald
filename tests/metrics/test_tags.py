from pathlib import Path

import polars as pl

from herald.metrics.tags import build


def test_tags_build_passes_through(tmp_path: Path):
    final = tmp_path / "final"
    final.mkdir()
    pl.DataFrame(
        {
            "run_id": ["r1"],
            "catastrophes": [["looping", "non_termination"]],
        }
    ).write_parquet(final / "runs.parquet")
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)
    df = pl.read_parquet(out / "tags.parquet")
    assert df.height == 1
    assert df["has_looping"][0] is True
    assert df["has_non_termination"][0] is True
