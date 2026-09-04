"""Development-only quality-risk label audit.

Implements protocol section 2.1 (null-score ruling). Runs after
``results/recovered/quality-risk-v1/protocol_lock.json`` is frozen and
never projects confirmation prompts.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

SCORE_COLUMNS = (
    "baseline_quality_score",
    "compressed_quality_score",
    "quality_delta",
)
CATASTROPHE_COLUMNS = (
    "has_looping",
    "has_non_termination",
    "has_format_break",
    "has_drift",
)
TOKEN_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
    "baseline_run_id",
    *SCORE_COLUMNS,
    *CATASTROPHE_COLUMNS,
)
NONE_COLUMNS = (
    "prompt_id",
    "stop_reason",
    "catastrophes",
    "predicted_answer",
    "num_tokens_generated",
    "max_new_tokens",
)
TASK_MAXIMUM = 1.0
TASK_MINIMUM = 0.0


def _read_filtered(
    path: Path, columns: list[str], prompt_ids: set[str]
) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("prompt_id").isin(sorted(prompt_ids)),  # type: ignore[attr-defined,no-untyped-call]
    )
    return table.to_pandas()


def _prevalence(
    frame: pd.DataFrame, dimension: str
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for value, group in frame.groupby(dimension, observed=True):
        result[str(value)] = {
            "runs": int(len(group)),
            "damage_rate": float(group["damage"].mean()),
        }
    return result


def _reference_failure_indicators(none: pd.DataFrame) -> set[str]:
    """Prompt ids whose reference run shows a recorded failure indicator."""
    flagged: set[str] = set()
    for row in none.itertuples():
        try:
            has_catastrophe = len(row.catastrophes) > 0
        except TypeError:
            has_catastrophe = False
        empty_answer = (
            not isinstance(row.predicted_answer, str)
            or not row.predicted_answer.strip()
        )
        stop = str(row.stop_reason)
        hit_limit = bool(row.num_tokens_generated >= row.max_new_tokens)
        if has_catastrophe or stop != "eos" or empty_answer or hit_limit:
            flagged.add(str(row.prompt_id))
    return flagged


def apply_run_labels(
    runs: pd.DataFrame,
    *,
    gap_prompts: list[str],
    lift_prompts: list[str],
) -> pd.DataFrame:
    """Attach damage labels to one row-per-run frame (pure function).

    Implements protocol 2.1: gap prompts dropped, null compressed scores
    imputed to the task minimum, null references in lift prompts imputed
    to the task minimum.
    """
    labeled = runs[~runs["prompt_id"].isin(gap_prompts)].copy()
    lift_mask = labeled["prompt_id"].isin(lift_prompts).to_numpy()
    comp = labeled["compressed_quality_score"].to_numpy(dtype=np.float64)
    base = labeled["baseline_quality_score"].to_numpy(dtype=np.float64)
    base[lift_mask] = TASK_MINIMUM
    imputed = np.isnan(comp)
    comp[imputed] = TASK_MINIMUM
    if not np.isfinite(base).all() or not np.isfinite(comp).all():
        raise ValueError("unlabeled run reached apply_run_labels")
    labeled["damage"] = (base - comp > 0).astype(np.int8)
    labeled["imputed_compressed_zero"] = imputed.astype(np.int8)
    labeled["lift_reference_zero"] = lift_mask.astype(np.int8)
    return labeled


def checkpoint_eligible_lengths(
    lengths: pd.Series, *, checkpoint: int
) -> pd.Series:
    """A run contributes to checkpoint t iff it has a token row at t."""
    return (lengths > checkpoint).astype(bool)


def audit_development_labels(
    *,
    dataset_root: Path,
    development: list[str],
    confirmation: list[str],
    presses: list[str],
) -> dict[str, Any]:
    """Audit damage labels on development prompts only."""
    development_ids = set(development)
    confirmation_ids = set(confirmation)
    if development_ids & confirmation_ids:
        raise ValueError("development and confirmation prompts overlap")
    frames: list[pd.DataFrame] = []
    for press in presses:
        tokens = _read_filtered(
            dataset_root / "tokens" / f"{press}.parquet",
            list(TOKEN_COLUMNS),
            development_ids,
        )
        leaked = set(tokens["prompt_id"].unique()) - development_ids
        if leaked:
            raise ValueError(f"non-development prompt in {press}")
        if tokens.empty:
            raise ValueError(f"no development rows in {press}")
        constant = tokens.groupby("run_id", sort=False)[
            [*SCORE_COLUMNS, *CATASTROPHE_COLUMNS]
        ].nunique(dropna=False)
        if not (constant <= 1).all().all():
            raise ValueError(f"run-level columns vary within run in {press}")
        first = tokens.sort_values("token_pos").groupby("run_id").first()
        frames.append(first)
    frame = pd.concat(frames)
    none = _read_filtered(
        dataset_root / "sequences" / "none.parquet",
        list(NONE_COLUMNS),
        development_ids,
    )
    none_indicators = _reference_failure_indicators(none)
    prompt_runs = frame.groupby("prompt_id").size()
    prompt_base_null = (
        frame["baseline_quality_score"].isna().groupby(frame["prompt_id"])
    ).sum()
    prompt_comp_null = (
        frame["compressed_quality_score"].isna().groupby(frame["prompt_id"])
    ).sum()
    missing_prompts = sorted(
        development_ids - set(frame["prompt_id"].unique())
    )
    if missing_prompts:
        raise ValueError(f"development prompts absent: {missing_prompts[:5]}")
    gap_prompts: list[str] = []
    lift_prompts: list[str] = []
    for prompt in sorted(development_ids):
        total = int(prompt_runs.loc[prompt])
        if int(prompt_base_null.loc[prompt]) != total:
            continue
        if int(prompt_comp_null.loc[prompt]) == total:
            gap_prompts.append(prompt)
            continue
        if prompt not in none_indicators:
            raise ValueError(
                f"reference null with no failure indicator: {prompt}, "
                "data error, blocking"
            )
        lift_prompts.append(prompt)
    labeled = frame[~frame["prompt_id"].isin(gap_prompts)].copy()
    lift_mask = labeled["prompt_id"].isin(lift_prompts)
    labeled.loc[lift_mask, "baseline_quality_score"] = TASK_MINIMUM
    residual_null_base = labeled[
        labeled["compressed_quality_score"].notna()
        & labeled["baseline_quality_score"].isna()
    ]
    if len(residual_null_base):
        raise ValueError(
            "unclassified null reference: "
            f"{len(residual_null_base)} runs, blocking"
        )
    imputed = labeled["compressed_quality_score"].isna()
    labeled.loc[imputed, "compressed_quality_score"] = 0.0
    baseline = labeled["baseline_quality_score"].to_numpy(dtype=np.float64)
    compressed = labeled["compressed_quality_score"].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(baseline).all():
        raise ValueError("non-finite baseline on labeled development runs")
    scored_mask = labeled["quality_delta"].notna()
    delta = labeled.loc[scored_mask, "quality_delta"].to_numpy(
        dtype=np.float64
    )
    if not np.allclose(
        delta,
        baseline[scored_mask.to_numpy()] - compressed[scored_mask.to_numpy()],
    ):
        return {
            "pass": False,
            "reason": "quality_delta sign convention is ambiguous",
            "confirmation_prompts_projected": 0,
        }
    if float(compressed.min()) < 0.0:
        raise ValueError("negative compressed score, unruled pattern")
    labeled["damage"] = (baseline - compressed > 0).astype(np.int8)
    labeled["catastrophe"] = (
        labeled[list(CATASTROPHE_COLUMNS)].fillna(False).any(axis=True)
    ).astype(np.int8)
    labeled["imputed_compressed_zero"] = imputed.astype(np.int8)
    labeled["lift_reference_zero"] = lift_mask.astype(np.int8)
    task_max = (
        labeled.groupby("task")["baseline_quality_score"].max().to_dict()
    )
    if any(float(value) > TASK_MAXIMUM for value in task_max.values()):
        raise ValueError(f"baseline exceeds task maximum: {task_max}")
    prompt_base_value = labeled.groupby("prompt_id")[
        "baseline_quality_score"
    ].first()
    imperfect = (prompt_base_value < TASK_MAXIMUM).astype(int)
    zero = (prompt_base_value == 0.0).astype(int)
    imputed_rows = labeled[labeled["imputed_compressed_zero"] == 1]
    return {
        "pass": True,
        "convention_note": "label uses the two score columns only",
        "quality_delta_convention": "baseline_minus_compressed",
        "confirmation_prompts_projected": 0,
        "excluded_gap_prompts": gap_prompts,
        "excluded_gap_prompts_by_task": labeled_gap_tasks(frame, gap_prompts),
        "lift_reference_zero_prompts": lift_prompts,
        "total_runs": int(len(labeled)),
        "damaged_runs": int(labeled["damage"].sum()),
        "damage_rate": float(labeled["damage"].mean()),
        "imputed_compressed_zero_runs": int(imputed.sum()),
        "imputed_runs_catastrophe_fraction": (
            float(imputed_rows["catastrophe"].mean())
            if len(imputed_rows)
            else 0.0
        ),
        "imputed_runs_by_task_press_ratio": (
            imputed_rows.groupby(["task", "press", "compression_ratio"])
            .size()
            .rename("runs")
            .reset_index()
            .to_dict(orient="records")
        ),
        "task_maximum_baseline": {
            str(k): float(v) for k, v in task_max.items()
        },
        "reference_imperfect_fraction": float(imperfect.mean()),
        "reference_zero_fraction": float(zero.mean()),
        "reference_imperfect_by_task": imperfect.groupby(
            prompt_base_value.index.map(
                labeled.groupby("prompt_id")["task"].first()
            )
        )
        .mean()
        .to_dict(),
        "catastrophe_rate_given_damage": float(
            labeled.loc[labeled["damage"] == 1, "catastrophe"].mean()
        ),
        "prevalence_by_task": _prevalence(labeled, "task"),
        "prevalence_by_press": _prevalence(labeled, "press"),
        "prevalence_by_ratio": _prevalence(labeled, "compression_ratio"),
    }


def labeled_gap_tasks(
    frame: pd.DataFrame, gap_prompts: list[str]
) -> dict[str, int]:
    """Count excluded gap prompts per task."""
    tasks = (
        frame[frame["prompt_id"].isin(gap_prompts)]
        .groupby("prompt_id")["task"]
        .first()
    )
    return {
        str(task): int(count) for task, count in tasks.value_counts().items()
    }
