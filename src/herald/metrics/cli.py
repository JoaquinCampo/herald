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
    from herald.metrics import (
        alignment,
        outcome,
        sequence,
        tags,
        token,
        trajectory,
    )

    final = root / "final"
    out = root / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    token.build(final, out)
    trajectory.build(final, out)
    sequence.build(final, out)
    outcome.build(final, out)
    tags.build(final, out)
    alignment.build(out, out)


@app.command()
def repair(
    run_id: str = typer.Argument(...),
    root: Path = typer.Option(Path("results/phase0")),
) -> None:
    """Re-run a single failed (run_id, press, ratio) cell."""
    from herald.metrics.repair import repair_run

    repair_run(run_id=run_id, root=root)
