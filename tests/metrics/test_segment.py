"""Segment-level aggregator tests (CPU-only)."""

from pathlib import Path

import polars as pl
import pytest

from herald.metrics.segment import (
    DEFAULT_KS,
    SEGMENTS_SCHEMA,
    TIER0_FEATURES,
    aggregate_all_segments,
    aggregate_run_segments,
    write_segments,
)


def _make_tokens(
    run_id: str = "r1",
    n: int = 50,
) -> pl.DataFrame:
    rows = []
    for i in range(n):
        row = {
            "run_id": run_id,
            "token_pos": i,
            "token_id": 100 + i,
            "token_str": f"t{i}",
        }
        # Increasing entropy for visible aggregation effect.
        for k, col in enumerate(TIER0_FEATURES):
            row[col] = float(i + k * 0.1)
        row["delta_h_valid"] = True
        rows.append(row)
    return pl.DataFrame(rows)


class TestAggregateRunSegments:
    def test_empty_input(self) -> None:
        empty = pl.DataFrame(
            {
                "run_id": [],
                "token_pos": [],
                **{c: [] for c in TIER0_FEATURES},
            }
        )
        out = aggregate_run_segments(empty, K=8)
        assert out.is_empty()

    def test_window_count(self) -> None:
        tokens = _make_tokens(n=20)
        out = aggregate_run_segments(tokens, K=8)
        # 20 tokens / 8 = 3 windows (8, 8, 4)
        assert out.height == 3
        assert list(out["window_start"].to_list()) == [0, 8, 16]
        assert list(out["window_end"].to_list()) == [8, 16, 20]
        assert list(out["n_tokens"].to_list()) == [8, 8, 4]

    def test_K_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="K must be positive"):
            aggregate_run_segments(_make_tokens(n=10), K=0)

    def test_single_run_id_required(self) -> None:
        a = _make_tokens("r1", n=5)
        b = _make_tokens("r2", n=5)
        merged = pl.concat([a, b])
        with pytest.raises(ValueError, match="single run_id"):
            aggregate_run_segments(merged, K=4)

    def test_aggregates_match_expected(self) -> None:
        tokens = _make_tokens(n=8)
        out = aggregate_run_segments(tokens, K=8)
        # entropy column was set to float(i + 0*0.1) = i for i in [0..7].
        row = out.row(0, named=True)
        assert row["entropy_min"] == pytest.approx(0.0)
        assert row["entropy_max"] == pytest.approx(7.0)
        assert row["entropy_mean"] == pytest.approx(3.5)

    def test_K_column_set(self) -> None:
        tokens = _make_tokens(n=10)
        out = aggregate_run_segments(tokens, K=16)
        assert out["K"].to_list() == [16]
        assert out["window_idx"].to_list() == [0]


class TestAggregateAllSegments:
    def test_empty_directory(self, tmp_path: Path) -> None:
        out = aggregate_all_segments(tmp_path, Ks=(8,))
        assert out.is_empty()

    def test_concatenates_per_run_with_K_column(self, tmp_path: Path) -> None:
        for rid, n in (("r1", 20), ("r2", 12)):
            _make_tokens(run_id=rid, n=n).write_parquet(
                tmp_path / f"{rid}.parquet"
            )
        out = aggregate_all_segments(tmp_path, Ks=(8, 16))
        # r1: 8/8/4 (3 windows at K=8) + 16/4 (2 at K=16)
        # r2: 8/4 (2 at K=8) + 12 (1 at K=16)
        # 3 + 2 + 2 + 1 = 8
        assert out.height == 8
        assert set(out["K"].to_list()) == {8, 16}
        assert set(out["run_id"].to_list()) == {"r1", "r2"}


class TestSchema:
    def test_pa_schema_columns_match_polars(self) -> None:
        tokens = _make_tokens(n=8)
        out = aggregate_run_segments(tokens, K=8)
        # Every PA field is present in the polars output.
        for field in SEGMENTS_SCHEMA:
            assert field.name in out.columns

    def test_default_Ks_constant(self) -> None:
        assert DEFAULT_KS == (8, 16, 32)


class TestWrite:
    def test_round_trip(self, tmp_path: Path) -> None:
        tokens = _make_tokens(n=10)
        df = aggregate_run_segments(tokens, K=8)
        out_path = tmp_path / "segments.parquet"
        write_segments(df, out_path)
        loaded = pl.read_parquet(out_path)
        assert loaded.height == df.height
        assert loaded.columns == df.columns
