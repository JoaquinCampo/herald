from herald.switch_baselines import (
    LOCKED_BASELINE,
    BaselineSpec,
    compare_to_baseline_lock,
    duplicate_audit,
    evaluate_baselines,
    evaluate_switch_predictions,
    fit_mean_baseline,
    leave_one_compressor_splits,
    make_baseline_lock,
    predict_mean_baseline,
)


def _row(
    prompt_id: str,
    compressor: str,
    *,
    task: str = "ifeval",
    ratio: float = 0.5,
    s: int = 0,
    dq: float = 0.0,
) -> dict[str, object]:
    """Build a minimal switch row for baseline tests."""
    return {
        "model": "llama",
        "task": task,
        "prompt_id": prompt_id,
        "compressor": compressor,
        "ratio": ratio,
        "s": s,
        "dq": dq,
    }


def test_leave_one_compressor_splits_are_grouped() -> None:
    """Leave-one-compressor splits keep prompts disjoint."""
    rows = [
        _row(f"p{i}", compressor, dq=[0.0, 1.0, 2.0][i % 3])
        for i in range(40)
        for compressor in ("streaming_llm", "knorm", "expected_attention")
    ]

    splits = leave_one_compressor_splits(
        rows, seed=7, test_group_fraction=0.35
    )

    assert {s.heldout_compressor for s in splits} == {
        "expected_attention",
        "knorm",
        "streaming_llm",
    }
    for split in splits:
        train_groups = {
            (row["model"], row["task"], row["prompt_id"])
            for row in split.train
        }
        test_groups = {
            (row["model"], row["task"], row["prompt_id"])
            for row in split.test
        }
        assert train_groups.isdisjoint(test_groups)
        assert {row["compressor"] for row in split.test} == {
            split.heldout_compressor
        }
        assert split.heldout_compressor not in {
            row["compressor"] for row in split.train
        }


def test_mean_baseline_backs_off_to_broader_group() -> None:
    """Grouped means back off when a narrow test key is unseen."""
    train = [
        {"task": "ifeval", "ratio": 0.5, "position_bucket": 0, "dq": 2.0},
        {"task": "ifeval", "ratio": 0.5, "position_bucket": 0, "dq": 4.0},
    ]
    model = fit_mean_baseline(
        train,
        BaselineSpec("task_ratio_position_bucket", ("task", "ratio")),
    )

    pred = predict_mean_baseline(
        model,
        {"task": "ifeval", "ratio": 0.5, "position_bucket": 16},
    )

    assert pred == 3.0


def test_evaluate_baselines_reports_all_required_baselines() -> None:
    """Baseline evaluation returns the deployable baseline matrix."""
    rows = _baseline_rows()

    summary = evaluate_baselines(
        rows,
        seed=3,
        test_group_fraction=0.25,
        bootstrap_resamples=5,
        command=["evaluate_switch_baselines.py"],
    )

    assert summary["n_rows"] == len(rows)
    assert summary["n_splits"] == 3
    assert summary["splits"]
    assert len(summary["metadata"]["dataset_fingerprint"]) == 64
    assert summary["metadata"]["command"] == ["evaluate_switch_baselines.py"]
    assert summary["duplicate_audit"]["duplicate_row_count"] == 0
    names = {
        m["baseline"]
        for split in summary["splits"]
        for m in split["baselines"]
    }
    assert names == {
        "global_mean",
        "ratio",
        "task_ratio",
        LOCKED_BASELINE,
    }
    for split in summary["splits"]:
        assert split["leakage_audit"] == {
            "prompt_group_fields": ["model", "task", "prompt_id"],
            "prompt_group_overlap_count": 0,
            "row_overlap_count": 0,
            "train_heldout_compressor_rows": 0,
            "test_nonheldout_compressor_rows": 0,
        }
        locked = next(
            m for m in split["baselines"] if m["baseline"] == LOCKED_BASELINE
        )
        assert locked["relative_mae_improvement_vs_locked"] == 0.0
        assert locked["controller_metrics"]["dq_gt_0"]["top_decile_lift"]
        assert locked["controller_metrics"]["dq_ge_0_5"]["top_decile_lift"]
        assert locked["ranking_metrics"]["dq_gt_0"]["auprc"]
        assert locked["ranking_metrics"]["dq_gt_0"]["recall_at_10_fpr"]
        assert locked["bootstrap_ci"]["n_bootstrap"] == 5
        assert locked["bootstrap_ci"]["n_clusters"] > 0
        assert locked["calibration_bins"]


def test_evaluate_switch_predictions_scores_candidate_predictions() -> None:
    """Canonical evaluator scores precomputed continuous dq predictions."""
    rows = []
    for row in _baseline_rows():
        predicted = dict(row)
        predicted["predicted_dq"] = predicted["dq"]
        rows.append(predicted)

    summary = evaluate_switch_predictions(
        rows,
        model_name="perfect_probe",
        seed=3,
        test_group_fraction=0.25,
        bootstrap_resamples=3,
    )

    assert summary["n_rows"] == len(rows)
    assert summary["n_splits"] == 3
    assert summary["config"]["prediction_key"] == "predicted_dq"
    assert summary["splits"]
    for split in summary["splits"]:
        metric = split["prediction"]
        assert metric["baseline"] == "perfect_probe"
        assert metric["mae"] == 0.0
        assert metric["relative_mae_improvement_vs_locked"] == 1.0
        assert metric["bootstrap_ci"]["n_bootstrap"] == 3
        assert metric["controller_metrics"]["dq_gt_0"]["top_decile_lift"]


def test_duplicate_audit_counts_canonical_switch_duplicates() -> None:
    """Duplicate audit reports repeated switch identities."""
    row = _row("p0", "knorm", s=16, dq=0.25)

    audit = duplicate_audit([row, dict(row), _row("p1", "knorm")])

    assert audit["duplicate_key_count"] == 1
    assert audit["duplicate_row_count"] == 1
    assert audit["examples"] == [
        {"key": ["llama", "ifeval", "p0", "knorm", 0.5, 16], "count": 2}
    ]


def test_baseline_lock_comparison_detects_metric_drift() -> None:
    """Locked baseline comparison fails on meaningful MAE drift."""
    summary = evaluate_baselines(
        _baseline_rows(),
        seed=3,
        test_group_fraction=0.25,
        bootstrap_resamples=0,
    )
    lock = make_baseline_lock(summary, tolerance=0.0)

    assert compare_to_baseline_lock(summary, lock)["passed"]

    heldout = next(iter(lock["heldout"]))
    lock["heldout"][heldout] += 0.1

    comparison = compare_to_baseline_lock(summary, lock)

    assert not comparison["passed"]


def _baseline_rows() -> list[dict[str, object]]:
    """Build enough rows for deterministic compressor-held-out splits."""
    return [
        _row(
            f"p{i}",
            compressor,
            task="gsm8k" if i % 2 == 0 else "ifeval",
            ratio=0.25 if i % 2 == 0 else 0.5,
            s=(i % 4) * 16,
            dq=[0.0, 1.0, 2.0][(i + len(compressor)) % 3],
        )
        for i in range(60)
        for compressor in ("streaming_llm", "knorm", "expected_attention")
    ]
