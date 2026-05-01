"""Paired outcome cells: gross_harm, gross_help, net_delta per cell."""

from pathlib import Path

import polars as pl


def build(final: Path, out: Path) -> None:
    runs = pl.read_parquet(final / "runs.parquet")
    base = (
        runs.filter(pl.col("press") == "none")
        .select(["run_id", "correct"])
        .rename(
            {
                "run_id": "baseline_run_id",
                "correct": "baseline_correct",
            }
        )
    )
    comp = runs.filter(pl.col("press") != "none").join(
        base, on="baseline_run_id", how="left"
    )

    cells = comp.group_by(["press", "compression_ratio"]).agg(
        [
            (
                (pl.col("baseline_correct") & ~pl.col("correct"))
                .cast(pl.Float64)
                .mean()
            ).alias("gross_harm"),
            (
                (~pl.col("baseline_correct") & pl.col("correct"))
                .cast(pl.Float64)
                .mean()
            ).alias("gross_help"),
            pl.len().alias("n_cells"),
        ]
    )
    cells = cells.with_columns(
        (pl.col("gross_harm") - pl.col("gross_help")).alias("net_delta")
    )
    cells.write_parquet(out / "outcome.parquet")
