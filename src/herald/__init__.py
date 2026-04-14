"""HERALD: Hazard Estimation via Real-time Analysis of Logit Distributions.

Predicts catastrophic failures in KV-cache compressed LLM generation
from per-token logit signals.
"""

from pathlib import Path

import typer

app = typer.Typer(
    name="herald",
    help="Predict catastrophic failures in KV-cache "
    "compression from logit signals.",
)


@app.command()
def train(
    results_dir: Path = typer.Option(
        Path("results"),
        help="Directory with sweep result JSONs",
    ),
    output_dir: Path = typer.Option(
        Path("models"),
        help="Directory to save trained models",
    ),
    horizons: list[int] = typer.Option(
        [1, 5, 10, 25, 50],
        help="Prediction horizons in tokens",
    ),
) -> None:
    """Train XGBoost hazard predictors from sweep results."""
    from herald.train import train_predictor

    train_predictor(results_dir, output_dir, horizons)


@app.command()
def evaluate(
    model_dir: Path = typer.Option(
        Path("models"),
        help="Directory with trained model JSONs",
    ),
    results_dir: Path = typer.Option(
        Path("results"),
        help="Directory with sweep result JSONs",
    ),
    horizons: list[int] = typer.Option(
        [1, 5, 10, 25, 50],
        help="Prediction horizons to evaluate",
    ),
    nt_onset_frac: float = typer.Option(
        0.75,
        help="NT onset fraction for labeling",
    ),
    output_path: Path = typer.Option(
        Path("models/eval_metrics.json"),
        help="Path to save evaluation results",
    ),
) -> None:
    """Evaluate trained models with full metrics suite."""
    import json

    from herald.evaluate import evaluate_all_horizons

    results = evaluate_all_horizons(
        model_dir, results_dir, horizons, nt_onset_frac
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


@app.command(name="nt-sensitivity")
def nt_sensitivity_cmd(
    results_dir: Path = typer.Option(
        Path("results"),
        help="Directory with sweep result JSONs",
    ),
    horizon: int = typer.Option(
        10,
        help="Prediction horizon in tokens",
    ),
    fracs: list[float] = typer.Option(
        [0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
        help="NT onset fractions to test",
    ),
    output_path: Path = typer.Option(
        Path("models/nt_sensitivity.json"),
        help="Path to save results",
    ),
) -> None:
    """Run NT onset fraction sensitivity analysis."""
    import json

    from herald.evaluate import nt_sensitivity

    results = nt_sensitivity(results_dir, horizon, fracs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))


def main() -> None:
    app()
