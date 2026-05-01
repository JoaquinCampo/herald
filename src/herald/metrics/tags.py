"""Diagnostic tags per run, exploded into boolean columns."""

from pathlib import Path

import polars as pl

CATEGORIES = ("looping", "non_termination", "format_break", "drift")


def build(final: Path, out: Path) -> None:
    runs = pl.read_parquet(final / "runs.parquet").select(
        ["run_id", "catastrophes"]
    )
    rows = []
    for r in runs.iter_rows(named=True):
        cats = set(r["catastrophes"] or [])
        rows.append(
            {
                "run_id": r["run_id"],
                **{f"has_{c}": (c in cats) for c in CATEGORIES},
            }
        )
    pl.DataFrame(rows).write_parquet(out / "tags.parquet")
