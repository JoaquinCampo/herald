"""CLI subapp wiring for the analysis subpackage.

Each analysis is a `herald analyze <name>` command. Defaults
are chosen so `herald analyze all` performs the full tier-2
bundle with no flags in the common case.
"""

from pathlib import Path

import typer

from herald.analysis.calibration import run_calibration
from herald.analysis.common import DEFAULT_NT_ONSET_FRAC
from herald.analysis.errors import DEFAULT_THRESHOLD, run_errors
from herald.analysis.importances import run_importances
from herald.analysis.inference_cost import (
    DEFAULT_N_ITERS,
    run_inference_cost,
)
from herald.analysis.nt_sensitivity_all import run_nt_sensitivity_all
from herald.analysis.per_press import run_per_press
from herald.analysis.per_ratio import run_per_ratio
from herald.analysis.qualitative import (
    DEFAULT_HORIZON as QUAL_DEFAULT_HORIZON,
)
from herald.analysis.qualitative import (
    DEFAULT_N_CATASTROPHIC,
    DEFAULT_N_CLEAN,
    run_qualitative,
)

app = typer.Typer(
    name="analyze",
    help="Post-training analyses: per-press, per-ratio, "
    "calibration, errors, importances, qualitative, cost.",
)

DEFAULT_MODEL_DIR = Path("models")
DEFAULT_RESULTS_DIR = Path("results")
DEFAULT_OUTPUT_DIR = Path("models/analysis")
DEFAULT_HORIZONS = [1, 5, 10, 25, 50]


@app.command("per-press")
def per_press_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "per_press.json"),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
) -> None:
    """Per-press AUROC/AUPRC/sequence metrics."""
    run_per_press(
        model_dir, results_dir, output_path, horizons, nt_onset_frac
    )


@app.command("per-ratio")
def per_ratio_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "per_ratio.json"),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
) -> None:
    """Per-compression-ratio breakdown."""
    run_per_ratio(
        model_dir, results_dir, output_path, horizons, nt_onset_frac
    )


@app.command("calibration")
def calibration_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "calibration.json"),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
    n_bins: int = typer.Option(15),
) -> None:
    """Reliability curves, Brier score, and ECE."""
    run_calibration(
        model_dir,
        results_dir,
        output_path,
        horizons,
        nt_onset_frac,
        n_bins,
    )


@app.command("errors")
def errors_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "errors.json"),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
    threshold: float = typer.Option(DEFAULT_THRESHOLD),
) -> None:
    """Sequence-level FP/FN breakdown by press, ratio, length."""
    run_errors(
        model_dir,
        results_dir,
        output_path,
        horizons,
        nt_onset_frac,
        threshold,
    )


@app.command("importances")
def importances_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "importances.json"),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
) -> None:
    """Feature importances (gain, weight, cover)."""
    run_importances(model_dir, output_path, horizons)


@app.command("qualitative")
def qualitative_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(DEFAULT_OUTPUT_DIR / "qualitative.json"),
    horizon: int = typer.Option(QUAL_DEFAULT_HORIZON),
    n_catastrophic: int = typer.Option(DEFAULT_N_CATASTROPHIC),
    n_clean: int = typer.Option(DEFAULT_N_CLEAN),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
) -> None:
    """Dump qualitative sequences with per-token hazards."""
    run_qualitative(
        model_dir,
        results_dir,
        output_path,
        horizon,
        n_catastrophic,
        n_clean,
        nt_onset_frac,
    )


@app.command("cost")
def cost_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "inference_cost.json"
    ),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    n_iters: int = typer.Option(DEFAULT_N_ITERS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
) -> None:
    """Predictor inference-cost microbenchmark."""
    run_inference_cost(
        model_dir,
        results_dir,
        output_path,
        horizons,
        n_iters,
        nt_onset_frac,
    )


@app.command("nt-sensitivity-all")
def nt_sensitivity_all_cmd(
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_path: Path = typer.Option(
        DEFAULT_OUTPUT_DIR / "nt_sensitivity_all.json"
    ),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    fracs: list[float] = typer.Option([0.5, 0.6, 0.7, 0.75, 0.8, 0.9]),
) -> None:
    """NT-sensitivity sweep across every horizon."""
    run_nt_sensitivity_all(results_dir, output_path, horizons, fracs)


@app.command("all")
def all_cmd(
    model_dir: Path = typer.Option(DEFAULT_MODEL_DIR),
    results_dir: Path = typer.Option(DEFAULT_RESULTS_DIR),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR),
    horizons: list[int] = typer.Option(DEFAULT_HORIZONS),
    nt_onset_frac: float = typer.Option(DEFAULT_NT_ONSET_FRAC),
    skip_nt_sensitivity: bool = typer.Option(
        False,
        help="Skip the NT-sensitivity sweep (it retrains "
        "per-fraction, adds most of the runtime).",
    ),
) -> None:
    """Run every tier-2 analysis in one pass."""
    run_per_press(
        model_dir,
        results_dir,
        output_dir / "per_press.json",
        horizons,
        nt_onset_frac,
    )
    run_per_ratio(
        model_dir,
        results_dir,
        output_dir / "per_ratio.json",
        horizons,
        nt_onset_frac,
    )
    run_calibration(
        model_dir,
        results_dir,
        output_dir / "calibration.json",
        horizons,
        nt_onset_frac,
    )
    run_errors(
        model_dir,
        results_dir,
        output_dir / "errors.json",
        horizons,
        nt_onset_frac,
    )
    run_importances(model_dir, output_dir / "importances.json", horizons)
    run_qualitative(
        model_dir,
        results_dir,
        output_dir / "qualitative.json",
        QUAL_DEFAULT_HORIZON,
        DEFAULT_N_CATASTROPHIC,
        DEFAULT_N_CLEAN,
        nt_onset_frac,
    )
    run_inference_cost(
        model_dir,
        results_dir,
        output_dir / "inference_cost.json",
        horizons,
        DEFAULT_N_ITERS,
        nt_onset_frac,
    )
    if not skip_nt_sensitivity:
        run_nt_sensitivity_all(
            results_dir,
            output_dir / "nt_sensitivity_all.json",
            horizons,
        )
