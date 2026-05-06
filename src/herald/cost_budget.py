"""Phase 1 cost-budget contract.

Block 2 profiles a slice and writes a per-cell budget. Block 3 loads
that budget and constructs a `CostWatchdog` per cell so the sweep
self-aborts if any cell drifts >25% slower than predicted.

Schema (JSON):

    {
      "model": "Qwen/Qwen2.5-7B-Instruct",
      "max_new_tokens": 512,
      "longbench_subtask": "qasper",
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
      ],
      "task_fallbacks": [
        {
          "task": "longbench_single",
          "press": "*",
          "compression_ratio": -1.0,
          "predicted_s_per_token": 0.060,
          "predicted_n_tokens": 60.0,
          "n_profile_runs": 50
        }
      ]
    }

Cell key is `(task, press, ratio)`. The watchdog reads
`predicted_s_per_token`. `predicted_n_tokens` is informational (used
by the run-time estimator log line); `n_profile_runs` documents the
sample size that produced the prediction.

`resolve(task, press, ratio)` is the Block 3 lookup path:

1. Exact `(task, press, ratio)` if present.
2. Otherwise the cell with the same `(task, press)` whose ratio is
   closest to the requested one (ties broken by lower ratio).
3. Otherwise the explicitly-recorded `task_fallbacks[task]` entry.
4. Otherwise raise `BudgetTaskNotFound`.

Every non-exact match is logged at INFO via loguru so the sweep log
records what substitution was made.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


class BudgetTaskNotFound(KeyError):
    """The requested task has no exact, nearest-ratio, or task-level
    fallback budget entry. Raised by `CostBudget.resolve` so callers
    cannot silently substitute an unrelated task's budget."""


@dataclass(frozen=True)
class CellBudget:
    task: str
    press: str
    compression_ratio: float
    predicted_s_per_token: float
    predicted_n_tokens: float
    n_profile_runs: int


@dataclass(frozen=True)
class BudgetMatch:
    """Result of `CostBudget.resolve`. `match_kind` is one of
    `"exact"`, `"nearest_ratio"`, or `"task_fallback"`."""

    cell: CellBudget
    match_kind: str
    requested_task: str
    requested_press: str
    requested_ratio: float


@dataclass(frozen=True)
class CostBudget:
    model: str
    max_new_tokens: int
    cells: dict[tuple[str, str, float], CellBudget]
    task_fallbacks: dict[str, CellBudget] = field(default_factory=dict)

    def lookup(
        self, task: str, press: str, compression_ratio: float
    ) -> CellBudget | None:
        """Exact-only lookup (back-compat)."""
        return self.cells.get(
            (task, press, _quantize_ratio(compression_ratio))
        )

    def resolve(
        self, task: str, press: str, compression_ratio: float
    ) -> BudgetMatch:
        """Block 3 lookup. See module docstring for fallback order.

        Raises `BudgetTaskNotFound` if task is not present in cells or
        task_fallbacks. Logs every non-exact substitution.
        """
        ratio = _quantize_ratio(compression_ratio)
        exact = self.cells.get((task, press, ratio))
        if exact is not None:
            return BudgetMatch(
                cell=exact,
                match_kind="exact",
                requested_task=task,
                requested_press=press,
                requested_ratio=ratio,
            )

        same_press = sorted(
            (
                c
                for k, c in self.cells.items()
                if k[0] == task and k[1] == press
            ),
            key=lambda c: (
                abs(c.compression_ratio - ratio),
                c.compression_ratio,
            ),
        )
        if same_press:
            cell = same_press[0]
            logger.info(
                f"budget.resolve: nearest_ratio substitution "
                f"task={task} press={press} requested_ratio={ratio} "
                f"-> profiled_ratio={cell.compression_ratio} "
                f"(predicted_s_per_token={cell.predicted_s_per_token:.5f})"
            )
            return BudgetMatch(
                cell=cell,
                match_kind="nearest_ratio",
                requested_task=task,
                requested_press=press,
                requested_ratio=ratio,
            )

        if task in self.task_fallbacks:
            cell = self.task_fallbacks[task]
            logger.info(
                f"budget.resolve: task_fallback substitution "
                f"task={task} press={press} ratio={ratio} "
                f"-> fallback predicted_s_per_token="
                f"{cell.predicted_s_per_token:.5f}"
            )
            return BudgetMatch(
                cell=cell,
                match_kind="task_fallback",
                requested_task=task,
                requested_press=press,
                requested_ratio=ratio,
            )

        any_for_task = any(k[0] == task for k in self.cells)
        if not any_for_task:
            raise BudgetTaskNotFound(
                f"No budget entry for task={task!r}. "
                f"Cells cover tasks={sorted({k[0] for k in self.cells})}, "
                f"task_fallbacks={sorted(self.task_fallbacks)}."
            )
        raise BudgetTaskNotFound(
            f"Task {task!r} has cell entries but none for press={press!r} "
            f"and no task_fallback recorded. "
            f"Add a task_fallback to the budget JSON or profile this press."
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

    fallbacks: dict[str, CellBudget] = {}
    for c in raw.get("task_fallbacks", []):
        cell = CellBudget(
            task=c["task"],
            press=c["press"],
            compression_ratio=_quantize_ratio(c["compression_ratio"]),
            predicted_s_per_token=float(c["predicted_s_per_token"]),
            predicted_n_tokens=float(c["predicted_n_tokens"]),
            n_profile_runs=int(c["n_profile_runs"]),
        )
        fallbacks[cell.task] = cell

    return CostBudget(
        model=raw["model"],
        max_new_tokens=int(raw["max_new_tokens"]),
        cells=cells,
        task_fallbacks=fallbacks,
    )


def dump_cost_budget(budget: CostBudget, path: Path) -> None:
    payload: dict[str, object] = {
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
    if budget.task_fallbacks:
        payload["task_fallbacks"] = [
            {
                "task": c.task,
                "press": c.press,
                "compression_ratio": c.compression_ratio,
                "predicted_s_per_token": c.predicted_s_per_token,
                "predicted_n_tokens": c.predicted_n_tokens,
                "n_profile_runs": c.n_profile_runs,
            }
            for c in budget.task_fallbacks.values()
        ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2))
