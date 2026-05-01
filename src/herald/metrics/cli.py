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
) -> None:
    """Run all offline metric modules in order."""
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


@app.command()
def repair(
    run_id: str = typer.Argument(...),
    root: Path = typer.Option(Path("results/phase0")),
) -> None:
    """Re-run a single failed (run_id, press, ratio) cell."""
    from herald.metrics.repair import repair_run

    repair_run(run_id=run_id, root=root)
