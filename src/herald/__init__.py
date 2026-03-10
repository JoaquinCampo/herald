"""HERALD: Hazard Estimation via Real-time Analysis of Logit Distributions.

Predicts catastrophic failures in KV-cache compressed LLM generation
from per-token logit signals.
"""

import typer

app = typer.Typer(
    name="herald",
    help="Predict catastrophic failures in KV-cache compression from logit signals.",
)


def main() -> None:
    app()
