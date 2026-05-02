"""Phase 1 cost-budget contract.

Block 2 profiles a slice and writes a per-cell budget. Block 3 loads
that budget and constructs a `CostWatchdog` per cell so the sweep
self-aborts if any cell drifts >25% slower than predicted.

Schema (JSON):

    {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "max_new_tokens": 512,
      "cells": [
        {
          "task": "gsm8k",
          "press": "streaming_llm",
          "compression_ratio": 0.5,
          "predicted_s_per_token": 0.024,
          "predicted_n_tokens": 312.0,
          "n_profile_runs": 50
        },
        ...
      ]
    }

A cell key is `(task, press, ratio)`. The watchdog reads
`predicted_s_per_token`. `predicted_n_tokens` is informational (used
by the run-time estimator log line); `n_profile_runs` documents the
sample size that produced the prediction.

Block 1 ships only the schema + load/lookup helpers. Block 2 produces
the file; Block 3 wires the watchdog into the sweep loop.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CellBudget:
    task: str
    press: str
    compression_ratio: float
    predicted_s_per_token: float
    predicted_n_tokens: float
    n_profile_runs: int


@dataclass(frozen=True)
class CostBudget:
    model: str
    max_new_tokens: int
    cells: dict[tuple[str, str, float], CellBudget]

    def lookup(
        self, task: str, press: str, compression_ratio: float
    ) -> CellBudget | None:
        return self.cells.get(
            (task, press, _quantize_ratio(compression_ratio))
        )


def _quantize_ratio(r: float) -> float:
    """Round to 6dp so dict-key float comparisons survive JSON I/O."""
    return round(float(r), 6)


def load_cost_budget(path: Path) -> CostBudget:
    raw = json.loads(Path(path).read_text())
    cells: dict[tuple[str, str, float], CellBudget] = {}
    for c in raw["cells"]:
        ratio = _quantize_ratio(c["compression_ratio"])
        cell = CellBudget(
            task=c["task"],
            press=c["press"],
            compression_ratio=ratio,
            predicted_s_per_token=float(c["predicted_s_per_token"]),
            predicted_n_tokens=float(c["predicted_n_tokens"]),
            n_profile_runs=int(c["n_profile_runs"]),
        )
        cells[(cell.task, cell.press, cell.compression_ratio)] = cell
    return CostBudget(
        model=raw["model"],
        max_new_tokens=int(raw["max_new_tokens"]),
        cells=cells,
    )


def dump_cost_budget(budget: CostBudget, path: Path) -> None:
    payload = {
        "model": budget.model,
        "max_new_tokens": budget.max_new_tokens,
        "cells": [
            {
                "task": c.task,
                "press": c.press,
                "compression_ratio": c.compression_ratio,
                "predicted_s_per_token": c.predicted_s_per_token,
                "predicted_n_tokens": c.predicted_n_tokens,
                "n_profile_runs": c.n_profile_runs,
            }
            for c in budget.cells.values()
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
