from herald.switch_baselines import (
    BaselineSpec,
    evaluate_baselines,
    fit_mean_baseline,
    leave_one_compressor_splits,
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
    rows = [
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

    summary = evaluate_baselines(rows, seed=3, test_group_fraction=0.25)

    assert summary["n_rows"] == len(rows)
    assert summary["splits"]
    names = {
        m["baseline"]
        for split in summary["splits"]
        for m in split["baselines"]
    }
    assert names == {
        "global_mean",
        "ratio",
        "task_ratio",
        "task_ratio_position_bucket",
    }
