"""`herald metrics ...` CLI subcommands."""

from pathlib import Path

import typer

from herald.metrics.io import finalize_dataset

app = typer.Typer(help="Phase 0 metrics pipeline.")


@app.command()
def finalize(
    root: Path = typer.Option(
        Path("results/phase0"),
        help="Root containing raw/ and final/.",
    ),
) -> None:
    """Concatenate per-run parquets under raw/ into final/."""
    finalize_dataset(root)


@app.command()
def build(
    root: Path = typer.Option(Path("results/phase0")),
    skip_segments: bool = typer.Option(
        False,
        help="Skip segment aggregates (the slowest step). Useful for "
        "rerunning the cheap families during iteration.",
    ),
) -> None:
    """Run all offline metric modules in order.

    Segment metrics are the heaviest step (per-run iteration over
    `tokens.parquet` for K in {8, 16, 32}); they run last so partial
    reruns of the cheap families don't recompute them.
    """
    from herald.metrics import alignment as _alignment
    from herald.metrics import outcome as _outcome
    from herald.metrics import sequence as _sequence
    from herald.metrics import tags as _tags
    from herald.metrics import token as _token
    from herald.metrics import trajectory as _trajectory

    final = root / "final"
    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    _token.build(final, out)
    _trajectory.build(final, out)
    _sequence.build(final, out)
    _outcome.build(final, out)
    _tags.build(final, out)
    _alignment.build(out, out)
    if not skip_segments:
        _build_segments(root=root, out=out)


def _build_segments(*, root: Path, out: Path) -> None:
    """Aggregate Tier-0 features into K-token windows for K in {8,16,32}.

    Walks every per-cell raw/tokens/ directory under `root` and writes a
    single `segments.parquet` to `out/`. Cells with no tokens.parquet
    files (e.g. baseline cells with `n_attempted=0`) are silently
    skipped.
    """
    import polars as pl

    from herald.metrics.io import _collect_raw_dirs
    from herald.metrics.segment import (
        DEFAULT_KS,
        aggregate_all_segments,
        write_segments,
    )

    token_dirs = _collect_raw_dirs(root, "tokens")
    if not token_dirs:
        return
    parts: list[pl.DataFrame] = []
    for d in token_dirs:
        if not d.is_dir():
            continue
        df = aggregate_all_segments(d, Ks=DEFAULT_KS)
        if df.is_empty():
            continue
        parts.append(df)
    if not parts:
        return
    combined = pl.concat(parts, how="vertical")
    write_segments(combined, out / "segments.parquet")


@app.command()
def repair(
    run_id: str = typer.Argument(...),
    root: Path = typer.Option(Path("results/phase0")),
) -> None:
    """Re-run a single failed (run_id, press, ratio) cell."""
    from herald.metrics.repair import repair_run

    repair_run(run_id=run_id, root=root)
