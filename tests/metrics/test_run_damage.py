"""Tests for the canonical run-level damage table.

The builder takes the finalized Phase 1 artifacts plus rescoring
metadata (gold answers for Qasper, instruction lists + kwargs for
IFEval) and emits one row per compressed run, joined to its baseline
plus all available trajectory / severity / tag metrics.
"""

from pathlib import Path

import polars as pl

from herald.metrics import run_damage as rd

# Minimal columns the builder needs from runs.parquet to work in
# tests. The real schema has a lot more fields; the builder must not
# depend on any of them.
RUNS_COLS_REQUIRED = (
    "run_id",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
    "baseline_run_id",
    "generated_text",
    "ground_truth",
    "correct",
    "catastrophes",
)


def _write_runs(final: Path, rows: list[dict]) -> None:
    final.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(
        final / "runs.parquet"
    )


def _write_trajectory(metrics: Path, rows: list[dict]) -> None:
    metrics.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics / "trajectory_metrics.parquet")


def _write_severity(metrics: Path, rows: list[dict]) -> None:
    metrics.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics / "severity_phase1.parquet")


def _write_tags(metrics: Path, rows: list[dict]) -> None:
    metrics.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics / "tags.parquet")


def _baseline_row(run_id: str, prompt_id: str, task: str, **kw):
    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "task": task,
        "press": "none",
        "compression_ratio": 0.0,
        "baseline_run_id": run_id,
        "generated_text": kw.get("generated_text", ""),
        "ground_truth": kw.get("ground_truth", ""),
        "correct": kw.get("correct", True),
        "catastrophes": [],
    }


def _compressed_row(
    run_id: str,
    prompt_id: str,
    task: str,
    baseline_run_id: str,
    **kw,
):
    return {
        "run_id": run_id,
        "prompt_id": prompt_id,
        "task": task,
        "press": kw.get("press", "snapkv"),
        "compression_ratio": kw.get("compression_ratio", 0.875),
        "baseline_run_id": baseline_run_id,
        "generated_text": kw.get("generated_text", ""),
        "ground_truth": kw.get("ground_truth", ""),
        "correct": kw.get("correct", True),
        "catastrophes": kw.get("catastrophes", []),
    }


# ---------------------------------------------------------------------------
# Joins and shape
# ---------------------------------------------------------------------------


class TestRunDamageJoins:
    def test_one_row_per_compressed_run(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        runs = [
            _baseline_row("b1", "gsm8k_1", "gsm8k", correct=True),
            _compressed_row("c1", "gsm8k_1", "gsm8k", "b1", correct=False),
            _baseline_row("b2", "gsm8k_2", "gsm8k", correct=True),
            _compressed_row("c2", "gsm8k_2", "gsm8k", "b2", correct=True),
        ]
        _write_runs(final, runs)
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c1",
                    "sum_kl": 1.0,
                    "sum_js": 0.5,
                    "nll_ratio": -0.1,
                    "first_divergence_point": 3,
                },
                {
                    "run_id": "c2",
                    "sum_kl": 0.1,
                    "sum_js": 0.05,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 50,
                },
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        df = pl.read_parquet(out)
        assert df.height == 2
        assert set(df["run_id"].to_list()) == {"c1", "c2"}
        # Baselines are joined, not present as their own rows.
        assert "none" not in df["press"].unique().to_list()

    def test_baseline_correctness_joined_via_baseline_run_id(
        self, tmp_path: Path
    ):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        runs = [
            _baseline_row("b1", "p1", "gsm8k", correct=True),
            _compressed_row("c1", "p1", "gsm8k", "b1", correct=False),
        ]
        _write_runs(final, runs)
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c1",
                    "sum_kl": 1.0,
                    "sum_js": 0.5,
                    "nll_ratio": -0.1,
                    "first_divergence_point": 3,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["baseline_run_id"] == "b1"
        assert row["baseline_correct_final"] is True
        assert row["compressed_correct_final"] is False
        assert row["gross_harm_final"] is True
        assert row["gross_help_final"] is False


# ---------------------------------------------------------------------------
# Quality columns by label source
# ---------------------------------------------------------------------------


class TestRunDamageQualityColumns:
    def test_gsm8k_label_source(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row("b", "p", "gsm8k", correct=True),
                _compressed_row("c", "p", "gsm8k", "b", correct=False),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 1.0,
                    "sum_js": 0.5,
                    "nll_ratio": -0.1,
                    "first_divergence_point": 3,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "gsm8k_exact"
        assert row["baseline_quality_score"] == 1.0
        assert row["compressed_quality_score"] == 0.0
        assert abs(row["quality_delta"] - 1.0) < 1e-9

    def test_humaneval_label_source(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row("b", "p", "humaneval", correct=True),
                _compressed_row("c", "p", "humaneval", "b", correct=True),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.1,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 99,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "humaneval_pass"

    def test_qasper_label_source_uses_provided_golds(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row(
                    "b",
                    "longbench_q1",
                    "longbench_single",
                    generated_text="42",
                    correct=True,
                ),
                _compressed_row(
                    "c",
                    "longbench_q1",
                    "longbench_single",
                    "b",
                    generated_text="completely wrong",
                    correct=True,  # placeholder grader said True
                ),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 1.0,
                    "sum_js": 1.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 1,
                }
            ],
        )
        qasper_golds = {"longbench_q1": ["42", "forty-two"]}
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
            qasper_golds=qasper_golds,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "qasper_f1"
        assert row["baseline_correct_final"] is True
        assert row["compressed_correct_final"] is False
        assert row["gross_harm_final"] is True
        # qasper_f1 column should also be populated
        assert row["qasper_f1"] is not None
        assert row["qasper_f1"] < 0.5

    def test_ifeval_label_source_supported(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row(
                    "b",
                    "ifeval_1",
                    "ifeval",
                    generated_text="all lowercase here",
                    correct=True,
                ),
                _compressed_row(
                    "c",
                    "ifeval_1",
                    "ifeval",
                    "b",
                    generated_text="HAS CAPS HERE",
                    correct=True,
                ),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.5,
                    "sum_js": 0.3,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 2,
                }
            ],
        )
        ifeval_meta = {
            "ifeval_1": {
                "instruction_id_list": ["change_case:english_lowercase"],
                "kwargs_list": [{}],
            }
        }
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
            ifeval_meta=ifeval_meta,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "ifeval_constraints"
        assert row["baseline_correct_final"] is True
        assert row["compressed_correct_final"] is False
        assert row["gross_harm_final"] is True

    def test_ifeval_unsupported_only_is_unavailable(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row(
                    "b",
                    "ifeval_x",
                    "ifeval",
                    generated_text="ok",
                    correct=True,
                ),
                _compressed_row(
                    "c",
                    "ifeval_x",
                    "ifeval",
                    "b",
                    generated_text="ok",
                    correct=True,
                ),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.0,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 0,
                }
            ],
        )
        # Only an unsupported instruction → run-level correctness undefined.
        ifeval_meta = {
            "ifeval_x": {
                "instruction_id_list": ["language:response_language"],
                "kwargs_list": [{"language": "fr"}],
            }
        }
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
            ifeval_meta=ifeval_meta,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "unavailable"
        assert row["baseline_correct_final"] is None
        assert row["compressed_correct_final"] is None
        # gross_harm_final must NOT be True when label is undefined.
        assert row["gross_harm_final"] in (False, None)


# ---------------------------------------------------------------------------
# Severity, tags, embedding placeholder
# ---------------------------------------------------------------------------


class TestRunDamageOptionalColumns:
    def test_severity_columns_attach(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row("b", "p", "gsm8k", correct=True),
                _compressed_row("c", "p", "gsm8k", "b", correct=True),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.0,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 0,
                }
            ],
        )
        _write_severity(
            metrics,
            [
                {
                    "run_id": "c",
                    "rouge_l_drop": 0.42,
                    "char_edit_ratio": 0.3,
                    "length_diff_ratio": 0.1,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=metrics / "severity_phase1.parquet",
            tags_path=None,
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert abs(row["rouge_l_drop"] - 0.42) < 1e-9
        # Embedding cosine drop is the documented placeholder; null when
        # not provided.
        assert "embedding_cosine_drop" in row
        assert row["embedding_cosine_drop"] is None

    def test_tags_attach(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row("b", "p", "gsm8k", correct=True),
                _compressed_row("c", "p", "gsm8k", "b", correct=True),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.0,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 0,
                }
            ],
        )
        _write_tags(
            metrics,
            [
                {
                    "run_id": "c",
                    "has_looping": True,
                    "has_non_termination": False,
                    "has_format_break": False,
                    "has_drift": False,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=metrics / "tags.parquet",
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["has_looping"] is True
        assert row["has_non_termination"] is False


# ---------------------------------------------------------------------------
# Build summary
# ---------------------------------------------------------------------------


class TestRunDamageDegenerate:
    def test_no_qasper_golds_marks_unavailable(self, tmp_path: Path):
        """If we have no gold answers (e.g. dataset reload failed),
        Qasper rows must be reported as undefined, not silently True."""
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row(
                    "b",
                    "qx",
                    "longbench_single",
                    generated_text="anything",
                    correct=True,
                ),
                _compressed_row(
                    "c",
                    "qx",
                    "longbench_single",
                    "b",
                    generated_text="anything",
                    correct=True,
                ),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c",
                    "sum_kl": 0.0,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 0,
                }
            ],
        )
        out = tmp_path / "out.parquet"
        # No qasper_golds and ground_truth is empty in our fixture.
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        row = pl.read_parquet(out).row(0, named=True)
        assert row["quality_label_source"] == "unavailable"
        assert row["baseline_correct_final"] is None
        assert row["compressed_correct_final"] is None
        assert row["gross_harm_final"] is None
        assert row["gross_help_final"] is None
        assert row["quality_delta"] is None

    def test_all_baseline_compressed_match_no_harm_positives(
        self, tmp_path: Path
    ):
        """If every paired (baseline, compressed) has the same final
        label, gross_harm_final must be False everywhere (not
        accidentally True). The success criterion will then report
        the AUROC as undefined."""
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        rows = []
        traj = []
        for i in range(5):
            rows.append(
                _baseline_row(f"b{i}", f"p{i}", "gsm8k", correct=True)
            )
            rows.append(
                _compressed_row(
                    f"c{i}", f"p{i}", "gsm8k", f"b{i}", correct=True
                )
            )
            traj.append(
                {
                    "run_id": f"c{i}",
                    "sum_kl": float(i),
                    "sum_js": float(i) * 0.5,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 10,
                }
            )
        _write_runs(final, rows)
        _write_trajectory(metrics, traj)
        out = tmp_path / "out.parquet"
        rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        df = pl.read_parquet(out)
        assert df.height == 5
        # No harm: both correct everywhere.
        assert (df["gross_harm_final"] == False).sum() == 5  # noqa: E712
        assert (df["gross_help_final"] == False).sum() == 5  # noqa: E712
        # And the per-task summary marks the task as degenerate
        # (no positives) — caller should treat AUROC as undefined.
        # (Re-run build to grab the summary.)
        summary = rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        per_task = {r["task"]: r for r in summary["per_task"]}
        assert per_task["gsm8k"]["non_degenerate"] is False


class TestRunDamageSummary:
    def test_summary_reports_per_task_label_sources(self, tmp_path: Path):
        final = tmp_path / "final"
        metrics = tmp_path / "metrics"
        _write_runs(
            final,
            [
                _baseline_row("b1", "p1", "gsm8k", correct=True),
                _compressed_row("c1", "p1", "gsm8k", "b1", correct=False),
                _baseline_row("b2", "p2", "humaneval", correct=True),
                _compressed_row("c2", "p2", "humaneval", "b2", correct=True),
            ],
        )
        _write_trajectory(
            metrics,
            [
                {
                    "run_id": "c1",
                    "sum_kl": 1.0,
                    "sum_js": 0.5,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 1,
                },
                {
                    "run_id": "c2",
                    "sum_kl": 0.0,
                    "sum_js": 0.0,
                    "nll_ratio": 0.0,
                    "first_divergence_point": 50,
                },
            ],
        )
        out = tmp_path / "out.parquet"
        summary = rd.build(
            runs_path=final / "runs.parquet",
            trajectory_path=metrics / "trajectory_metrics.parquet",
            severity_path=None,
            tags_path=None,
            out_path=out,
        )
        # Summary contains per-task source counts and pos/neg counts.
        per_task = {row["task"]: row for row in summary["per_task"]}
        assert per_task["gsm8k"]["label_source"] == "gsm8k_exact"
        assert per_task["gsm8k"]["n_pos_gross_harm"] == 1
        assert per_task["humaneval"]["label_source"] == "humaneval_pass"
        assert per_task["humaneval"]["n_pos_gross_harm"] == 0
        assert "unsupported_ifeval_types" in summary
