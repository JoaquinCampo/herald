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
        Path("results"), help="Directory with sweep result JSONs"
    ),
    output_dir: Path = typer.Option(
        Path("models"), help="Directory to save trained models"
    ),
    horizons: list[int] = typer.Option(
        [1, 5, 10, 25, 50], help="Prediction horizons in tokens"
    ),
) -> None:
    """Train XGBoost hazard predictors from sweep results."""
    from herald.train import train_predictor

    train_predictor(results_dir, output_dir, horizons)


def main() -> None:
    app()
