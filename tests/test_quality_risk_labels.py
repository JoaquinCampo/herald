"""CPU tests for the quality-risk development label audit (post-lock)."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from herald.quality_risk_labels import audit_development_labels


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _fixture(root: Path) -> dict:
    sequences = root / "corpus" / "sequences"
    tokens = root / "corpus" / "tokens"
    _write_parquet(
        sequences / "none.parquet",
        [
            {
                "run_id": "none-p1",
                "prompt_id": "p1",
                "correct": True,
                "stop_reason": "eos",
                "catastrophes": [],
                "predicted_answer": "42",
                "num_tokens_generated": 10,
                "max_new_tokens": 512,
            },
            {
                "run_id": "none-p2",
                "prompt_id": "p2",
                "correct": False,
                "stop_reason": "eos",
                "catastrophes": [],
                "predicted_answer": "41",
                "num_tokens_generated": 10,
                "max_new_tokens": 512,
            },
            {
                "run_id": "none-p9",
                "prompt_id": "p9",
                "correct": True,
                "stop_reason": "eos",
                "catastrophes": [],
                "predicted_answer": "42",
                "num_tokens_generated": 10,
                "max_new_tokens": 512,
            },
            # p5 reference produced no extractable answer: lift case.
            {
                "run_id": "none-p5",
                "prompt_id": "p5",
                "correct": None,
                "stop_reason": "eos",
                "catastrophes": ["wrong_answer"],
                "predicted_answer": "",
                "num_tokens_generated": 300,
                "max_new_tokens": 512,
            },
        ],
    )
    _write_parquet(
        sequences / "knorm.parquet",
        [
            {
                "run_id": "run-p1",
                "prompt_id": "p1",
                "baseline_run_id": "none-p1",
                "task": "gsm8k",
                "press": "knorm",
                "compression_ratio": 0.5,
                "correct": False,
            },
            {
                "run_id": "run-p2",
                "prompt_id": "p2",
                "baseline_run_id": "none-p2",
                "task": "gsm8k",
                "press": "knorm",
                "compression_ratio": 0.5,
                "correct": False,
            },
        ],
    )
    token_row = {
        "run_id": "run-p1",
        "token_pos": 0,
        "prompt_id": "p1",
        "task": "gsm8k",
        "press": "knorm",
        "compression_ratio": 0.5,
        "baseline_run_id": "none-p1",
        "baseline_quality_score": 1.0,
        "compressed_quality_score": 0.0,
        "quality_delta": 1.0,
        "has_looping": False,
        "has_non_termination": False,
        "has_format_break": False,
        "has_drift": False,
    }
    _write_parquet(
        tokens / "knorm.parquet",
        [
            token_row,
            {
                **token_row,
                "token_pos": 1,
                "run_id": "run-p2",
                "prompt_id": "p2",
                "baseline_run_id": "none-p2",
                "baseline_quality_score": 0.0,
                "quality_delta": 0.0,
            },
            # Run-level null with finite reference: imputed to zero.
            {
                **token_row,
                "run_id": "run-p3",
                "prompt_id": "p3",
                "baseline_run_id": "none-p3",
                "compressed_quality_score": float("nan"),
                "quality_delta": float("nan"),
                "has_non_termination": True,
            },
            # Prompt-level scoring gap: excluded entirely.
            {
                **token_row,
                "run_id": "run-p4",
                "prompt_id": "p4",
                "baseline_run_id": "none-p4",
                "baseline_quality_score": float("nan"),
                "compressed_quality_score": float("nan"),
                "quality_delta": float("nan"),
            },
            # Reference null with finite compressed score and a failure
            # indicator: baseline imputed to zero, not damaged (lift).
            {
                **token_row,
                "run_id": "run-p5",
                "prompt_id": "p5",
                "baseline_run_id": "none-p5",
                "baseline_quality_score": float("nan"),
                "compressed_quality_score": 0.5,
                "quality_delta": float("nan"),
            },
        ],
    )
    development = [
        {"prompt_id": "p1", "task": "gsm8k", "fold": 0},
        {"prompt_id": "p2", "task": "gsm8k", "fold": 1},
        {"prompt_id": "p3", "task": "gsm8k", "fold": 2},
        {"prompt_id": "p4", "task": "gsm8k", "fold": 3},
        {"prompt_id": "p5", "task": "gsm8k", "fold": 4},
    ]
    confirmation = [{"prompt_id": "p9", "task": "gsm8k"}]
    (root / "development.json").write_text(json.dumps(development))
    (root / "confirmation.json").write_text(json.dumps(confirmation))
    return {"development": development, "confirmation": confirmation}


def test_damage_label_prevalence_and_reference_wrong(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    report = audit_development_labels(
        dataset_root=tmp_path / "corpus",
        development=[row["prompt_id"] for row in fixture["development"]],
        confirmation=[row["prompt_id"] for row in fixture["confirmation"]],
        presses=["knorm"],
    )
    assert report["pass"] is True
    assert report["confirmation_prompts_projected"] == 0
    # run-p1: 1 - 0 damaged; run-p2: 0 - 0 not damaged;
    # run-p3: 1 - null imputed to zero, damaged; run-p4: gap, excluded;
    # run-p5: null reference imputed to zero, 0.5 not damaged (lift).
    assert report["total_runs"] == 4
    assert report["damaged_runs"] == 2
    assert report["imputed_compressed_zero_runs"] == 1
    assert report["excluded_gap_prompts"] == ["p4"]
    assert report["lift_reference_zero_prompts"] == ["p5"]
    # p2 and imputed p5 references are imperfect and zero; p4 excluded.
    assert report["reference_zero_fraction"] == 2 / 4
    assert report["reference_imperfect_fraction"] == 2 / 4
    # quality_delta is baseline minus compressed (verified on corpus).
    assert report["quality_delta_convention"] == "baseline_minus_compressed"


def test_reference_null_without_indicator_blocks(tmp_path: Path) -> None:
    _fixture(tmp_path)
    none = tmp_path / "corpus" / "sequences" / "none.parquet"
    rows = pq.read_table(none).to_pylist()
    for row in rows:
        if row["prompt_id"] == "p5":
            row["catastrophes"] = []
            row["predicted_answer"] = "42"
    pq.write_table(pa.Table.from_pylist(rows), none)
    try:
        audit_development_labels(
            dataset_root=tmp_path / "corpus",
            development=["p1", "p2", "p3", "p4", "p5"],
            confirmation=["p9"],
            presses=["knorm"],
        )
    except ValueError as exc:
        assert "no failure indicator" in str(exc).lower()
    else:
        raise AssertionError("indicator-less reference null did not block")


def test_finite_compressed_with_null_reference_blocks(tmp_path: Path) -> None:
    _fixture(tmp_path)
    tokens = tmp_path / "corpus" / "tokens" / "knorm.parquet"
    rows = pq.read_table(tokens).to_pylist()
    for row in rows:
        if row["run_id"] == "run-p2":
            row["baseline_quality_score"] = float("nan")
            row["compressed_quality_score"] = 0.5
    pq.write_table(pa.Table.from_pylist(rows), tokens)
    try:
        audit_development_labels(
            dataset_root=tmp_path / "corpus",
            development=["p1", "p2", "p3", "p4", "p5"],
            confirmation=["p9"],
            presses=["knorm"],
        )
    except ValueError as exc:
        assert "no failure indicator" in str(exc).lower()
    else:
        raise AssertionError("null-reference data error was not blocked")


def test_confirmation_contamination_blocks(tmp_path: Path) -> None:
    _fixture(tmp_path)
    try:
        audit_development_labels(
            dataset_root=tmp_path / "corpus",
            development=["p1", "p2", "p9"],
            confirmation=["p9"],
            presses=["knorm"],
        )
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("confirmation contamination was not blocked")
