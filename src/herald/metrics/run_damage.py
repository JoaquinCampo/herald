"""Canonical per-run compression-damage table for Phase 1.

This module produces ``run_damage.parquet`` with one row per
compressed run, joined to its uncompressed baseline by
``baseline_run_id``. It rescores tasks that used placeholder graders
during the sweep (IFEval, LongBench/qasper) deterministically from the
saved generations and dataset metadata; it does not rerun generation.

The rescore replaces both compressed and baseline ``correct`` for the
two saturated tasks. GSM8K and HumanEval keep their sweep-time labels.

Quality-label sources (one per row):
  - gsm8k_exact: GSM8K answer extraction (sweep-time correct field).
  - humaneval_pass: HumanEval ast.parse + presence (sweep-time).
  - qasper_f1: post-hoc Qasper F1 with configurable threshold.
  - ifeval_constraints: post-hoc IFEval partial scorer; threshold on
    fraction of supported satisfied. Run is "unavailable" if all
    instructions fall in the unsupported family for our scorer.
  - unavailable: no usable label could be derived for this row (e.g.
    Qasper without gold answers, IFEval without metadata, IFEval with
    only unsupported instruction types).

``gross_harm_final`` is True iff
``baseline_correct_final is True AND compressed_correct_final is False``;
when either side is None (label undefined) ``gross_harm_final`` is
False (we cannot prove harm). The same convention applies to
``gross_help_final``. Callers that want a tri-state version should use
``baseline_correct_final`` and ``compressed_correct_final`` directly.
"""

from pathlib import Path
from typing import Any

import polars as pl

from herald.metrics import task_scoring as ts

LABEL_SOURCES: tuple[str, ...] = (
    "gsm8k_exact",
    "humaneval_pass",
    "qasper_f1",
    "ifeval_constraints",
    "unavailable",
)


def _read_optional_parquet(path: Path | None) -> pl.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pl.read_parquet(path)


def _rescore_qasper(
    runs: pl.DataFrame,
    qasper_golds: dict[str, list[str]] | None,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Score every Qasper/LongBench run; return run_id -> score dict."""
    out: dict[str, dict[str, Any]] = {}
    sub = runs.filter(pl.col("task") == "longbench_single")
    for r in sub.iter_rows(named=True):
        prompt_id = r["prompt_id"]
        if qasper_golds is not None and prompt_id in qasper_golds:
            golds = qasper_golds[prompt_id]
        else:
            # Fall back to the single-gold field that the sweep wrote.
            gt = r.get("ground_truth") or ""
            golds = [gt] if gt else []
        score = ts.qasper_score(
            generated_text=r.get("generated_text") or "",
            gold_answers=golds,
            threshold=threshold,
        )
        out[r["run_id"]] = score
    return out


def _rescore_ifeval(
    runs: pl.DataFrame,
    ifeval_meta: dict[str, dict[str, Any]] | None,
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Score every IFEval run; return run_id -> score dict."""
    out: dict[str, dict[str, Any]] = {}
    sub = runs.filter(pl.col("task") == "ifeval")
    for r in sub.iter_rows(named=True):
        prompt_id = r["prompt_id"]
        if ifeval_meta is not None and prompt_id in ifeval_meta:
            meta = ifeval_meta[prompt_id]
            score = ts.ifeval_score(
                generated_text=r.get("generated_text") or "",
                instruction_id_list=list(meta["instruction_id_list"]),
                kwargs_list=list(meta["kwargs_list"]),
                threshold=threshold,
            )
        else:
            score = {
                "ifeval_num_constraints": 0,
                "ifeval_num_supported": 0,
                "ifeval_num_unsupported": 0,
                "ifeval_num_satisfied": 0,
                "ifeval_score": None,
                "ifeval_correct": None,
                "ifeval_unsupported_types": [],
                "ifeval_threshold": threshold,
            }
        out[r["run_id"]] = score
    return out


def _final_correctness(
    task: str,
    sweep_correct: bool | None,
    qasper_score: dict[str, Any] | None,
    ifeval_score: dict[str, Any] | None,
) -> tuple[bool | None, str, float | None]:
    """Return (correct_final, label_source, quality_score).

    ``quality_score`` is a float in [0, 1] suitable for continuous
    ``quality_delta``: 1.0 / 0.0 for binary tasks, qasper_f1 for
    LongBench, ifeval_score for IFEval. ``None`` means undefined.
    """
    if task == "gsm8k":
        if sweep_correct is None:
            return None, "unavailable", None
        return (
            bool(sweep_correct),
            "gsm8k_exact",
            1.0 if sweep_correct else 0.0,
        )
    if task == "humaneval":
        if sweep_correct is None:
            return None, "unavailable", None
        return (
            bool(sweep_correct),
            "humaneval_pass",
            1.0 if sweep_correct else 0.0,
        )
    if task == "longbench_single":
        if qasper_score is None or qasper_score["qasper_correct"] is None:
            return None, "unavailable", None
        return (
            bool(qasper_score["qasper_correct"]),
            "qasper_f1",
            float(qasper_score["qasper_f1"]),
        )
    if task == "ifeval":
        if ifeval_score is None or ifeval_score["ifeval_correct"] is None:
            # Binary correctness undefined. If at least the supported
            # subset was non-empty, expose the partial score for
            # debugging, but quality_label_source is unavailable.
            return None, "unavailable", None
        return (
            bool(ifeval_score["ifeval_correct"]),
            "ifeval_constraints",
            float(ifeval_score["ifeval_score"])
            if ifeval_score["ifeval_score"] is not None
            else None,
        )
    return None, "unavailable", None


def build(
    runs_path: Path,
    trajectory_path: Path,
    severity_path: Path | None,
    tags_path: Path | None,
    out_path: Path,
    qasper_threshold: float = 0.5,
    ifeval_threshold: float = 1.0,
    qasper_golds: dict[str, list[str]] | None = None,
    ifeval_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build run_damage.parquet from finalized Phase 1 artifacts.

    Returns a summary dict for the CLI to print. The parquet is the
    durable artifact; the summary is for stdout / logs.
    """
    runs = pl.read_parquet(runs_path)
    traj = pl.read_parquet(trajectory_path)
    sev = _read_optional_parquet(severity_path)
    tags = _read_optional_parquet(tags_path)

    # Rescore the saturated tasks once, indexed by run_id, before we
    # split baseline vs compressed (we need to score baselines too).
    qasper_scores = _rescore_qasper(runs, qasper_golds, qasper_threshold)
    ifeval_scores = _rescore_ifeval(runs, ifeval_meta, ifeval_threshold)

    # Per-row final correctness + quality score for every run.
    final_rows: list[dict[str, Any]] = []
    for r in runs.iter_rows(named=True):
        rid = r["run_id"]
        task = r["task"]
        correct_final, label_source, qscore = _final_correctness(
            task=task,
            sweep_correct=r.get("correct"),
            qasper_score=qasper_scores.get(rid),
            ifeval_score=ifeval_scores.get(rid),
        )
        row: dict[str, Any] = {
            "run_id": rid,
            "correct_final": correct_final,
            "quality_score": qscore,
            "quality_label_source": label_source,
        }
        # Carry post-hoc detail columns when we have them.
        qs = qasper_scores.get(rid)
        if qs is not None:
            row.update(
                {
                    "qasper_em": qs.get("qasper_em"),
                    "qasper_f1": qs.get("qasper_f1"),
                    "qasper_threshold": qs.get("qasper_threshold"),
                    "qasper_n_golds": qs.get("qasper_n_golds"),
                }
            )
        ifs = ifeval_scores.get(rid)
        if ifs is not None:
            row.update(
                {
                    "ifeval_num_constraints": ifs.get(
                        "ifeval_num_constraints"
                    ),
                    "ifeval_num_supported": ifs.get("ifeval_num_supported"),
                    "ifeval_num_unsupported": ifs.get(
                        "ifeval_num_unsupported"
                    ),
                    "ifeval_num_satisfied": ifs.get("ifeval_num_satisfied"),
                    "ifeval_score": ifs.get("ifeval_score"),
                    "ifeval_threshold": ifs.get("ifeval_threshold"),
                    "ifeval_unsupported_types": ifs.get(
                        "ifeval_unsupported_types"
                    ),
                }
            )
        final_rows.append(row)
    final_df = pl.DataFrame(final_rows, infer_schema_length=None)

    # Project the baseline view from the rescored frame; we want
    # baseline_correct_final and baseline_quality_score.
    base_view = (
        runs.filter(pl.col("press") == "none")
        .select(["run_id"])
        .join(final_df, on="run_id", how="left")
        .rename(
            {
                "run_id": "baseline_run_id",
                "correct_final": "baseline_correct_final",
                "quality_score": "baseline_quality_score",
                "quality_label_source": "baseline_quality_label_source",
            }
        )
        .select(
            [
                "baseline_run_id",
                "baseline_correct_final",
                "baseline_quality_score",
                "baseline_quality_label_source",
            ]
        )
    )

    # Compressed slice with all metadata + final correctness + per-task
    # detail columns.
    compressed = runs.filter(pl.col("press") != "none").select(
        [
            "run_id",
            "prompt_id",
            "task",
            "press",
            "compression_ratio",
            "baseline_run_id",
            "catastrophes",
        ]
    )
    df = compressed.join(final_df, on="run_id", how="left").rename(
        {
            "correct_final": "compressed_correct_final",
            "quality_score": "compressed_quality_score",
            "quality_label_source": "quality_label_source",
        }
    )
    df = df.join(base_view, on="baseline_run_id", how="left")

    # Trajectory metrics.
    df = df.join(
        traj.select(
            [
                "run_id",
                "sum_kl",
                "sum_js",
                "nll_ratio",
                "first_divergence_point",
            ]
        ),
        on="run_id",
        how="left",
    )

    # Severity metrics.
    if sev is not None:
        df = df.join(
            sev.select(
                [
                    "run_id",
                    "rouge_l_drop",
                    "char_edit_ratio",
                    "length_diff_ratio",
                ]
            ),
            on="run_id",
            how="left",
        )
    else:
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("rouge_l_drop"),
                pl.lit(None, dtype=pl.Float64).alias("char_edit_ratio"),
                pl.lit(None, dtype=pl.Float64).alias("length_diff_ratio"),
            ]
        )

    # Embedding cosine drop is documented as a placeholder column; null
    # until the [metrics] extra is wired in.
    df = df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("embedding_cosine_drop")
    )

    # Tags.
    if tags is not None:
        df = df.join(
            tags.select(
                [
                    "run_id",
                    "has_looping",
                    "has_non_termination",
                    "has_format_break",
                    "has_drift",
                ]
            ),
            on="run_id",
            how="left",
        )
    else:
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Boolean).alias("has_looping"),
                pl.lit(None, dtype=pl.Boolean).alias("has_non_termination"),
                pl.lit(None, dtype=pl.Boolean).alias("has_format_break"),
                pl.lit(None, dtype=pl.Boolean).alias("has_drift"),
            ]
        )

    # Unified outcome columns. ``gross_harm_final`` only fires when both
    # sides are well-defined; ``None``-sided rows are NOT counted as
    # harm or help.
    df = df.with_columns(
        [
            (
                pl.col("baseline_correct_final").fill_null(False)
                & ~pl.col("compressed_correct_final").fill_null(True)
            ).alias("gross_harm_final"),
            (
                ~pl.col("baseline_correct_final").fill_null(True)
                & pl.col("compressed_correct_final").fill_null(False)
            ).alias("gross_help_final"),
            (
                pl.when(
                    pl.col("baseline_quality_score").is_not_null()
                    & pl.col("compressed_quality_score").is_not_null()
                )
                .then(
                    pl.col("baseline_quality_score")
                    - pl.col("compressed_quality_score")
                )
                .otherwise(None)
            ).alias("quality_delta"),
        ]
    )

    # Make undefined rows truly undefined for the binary cells: when
    # either side is None, set gross_harm_final/gross_help_final to None
    # so callers can filter them. We compute it via mask + cast to Boolean.
    null_mask = (
        pl.col("baseline_correct_final").is_null()
        | pl.col("compressed_correct_final").is_null()
    )
    df = df.with_columns(
        [
            pl.when(null_mask)
            .then(None)
            .otherwise(pl.col("gross_harm_final"))
            .alias("gross_harm_final"),
            pl.when(null_mask)
            .then(None)
            .otherwise(pl.col("gross_help_final"))
            .alias("gross_help_final"),
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    # Build the summary the CLI prints.
    per_task: list[dict[str, Any]] = []
    for task in sorted(df["task"].unique().to_list()):
        sub = df.filter(pl.col("task") == task)
        sources = sub["quality_label_source"].value_counts().to_dicts()
        # Pick the most common non-unavailable source as the task's source.
        sources_sorted = sorted(
            sources, key=lambda r: r["count"], reverse=True
        )
        primary_source = next(
            (
                s["quality_label_source"]
                for s in sources_sorted
                if s["quality_label_source"] != "unavailable"
            ),
            "unavailable",
        )
        n_pos = int(
            sub["gross_harm_final"].fill_null(False).cast(pl.Int64).sum()
        )
        n_neg = int(
            (sub["gross_harm_final"].is_not_null().cast(pl.Int64).sum())
            - n_pos
        )
        n_undef = int(sub["gross_harm_final"].is_null().cast(pl.Int64).sum())
        per_task.append(
            {
                "task": task,
                "label_source": primary_source,
                "label_source_counts": {
                    s["quality_label_source"]: int(s["count"])
                    for s in sources
                },
                "n_total": sub.height,
                "n_pos_gross_harm": n_pos,
                "n_neg_gross_harm": n_neg,
                "n_undefined": n_undef,
                "non_degenerate": n_pos > 0 and n_neg > 0,
            }
        )

    # Aggregate unsupported IFEval types across rows.
    unsupported: dict[str, int] = {}
    if "ifeval_unsupported_types" in df.columns:
        for r in df.iter_rows(named=True):
            for t in r.get("ifeval_unsupported_types") or []:
                unsupported[t] = unsupported.get(t, 0) + 1

    return {
        "n_rows": df.height,
        "per_task": per_task,
        "unsupported_ifeval_types": dict(
            sorted(unsupported.items(), key=lambda kv: kv[1], reverse=True)
        ),
    }
