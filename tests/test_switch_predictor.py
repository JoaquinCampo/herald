import math

import pytest

from herald.switch_predictor import (
    fit_grouped_stat,
    internal_alpha_grid,
    mix_predictions,
    predict_grouped_stat,
    run_median_mean_mix,
    select_alpha,
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
    """Build a minimal switch row for predictor tests."""
    return {
        "model": "llama",
        "task": task,
        "prompt_id": prompt_id,
        "compressor": compressor,
        "ratio": ratio,
        "s": s,
        "dq": dq,
    }


def test_grouped_median_uses_bucket_key() -> None:
    """Grouped medians aggregate within (task, ratio, bucket) keys."""
    train = [
        _row("p1", "knorm", s=0, dq=0.0),
        _row("p2", "knorm", s=1, dq=0.0),
        _row("p3", "knorm", s=2, dq=1.0),
        _row("p4", "knorm", s=40, dq=1.0),
    ]
    model = fit_grouped_stat(train, stat="median", bucket_size=16)

    low = predict_grouped_stat(model, _row("q", "knorm", s=5))
    high = predict_grouped_stat(model, _row("q", "knorm", s=44))
    assert low == 0.0
    assert high == 1.0


def test_grouped_stat_backs_off_to_broader_key() -> None:
    """Unseen narrow keys back off to broader grouped statistics."""
    train = [
        _row("p1", "knorm", task="gsm8k", ratio=0.5, s=0, dq=1.0),
        _row("p2", "knorm", task="gsm8k", ratio=0.5, s=0, dq=1.0),
    ]
    model = fit_grouped_stat(train, stat="median", bucket_size=16)

    unseen_bucket = predict_grouped_stat(
        model, _row("q", "knorm", task="gsm8k", ratio=0.5, s=999)
    )
    unseen_task = predict_grouped_stat(
        model, _row("q", "knorm", task="humaneval", ratio=0.25, s=999)
    )
    assert unseen_bucket == 1.0
    assert unseen_task == 1.0


def test_grouped_stat_rejects_unknown_stat() -> None:
    """Only median and mean statistics are supported."""
    with pytest.raises(ValueError):
        fit_grouped_stat([_row("p1", "knorm")], stat="mode")


def test_mix_predictions_interpolates_median_and_mean() -> None:
    """The alpha mix moves linearly from median to mean."""
    train = [
        _row("p1", "knorm", s=0, dq=0.0),
        _row("p2", "knorm", s=1, dq=0.0),
        _row("p3", "knorm", s=2, dq=1.0),
    ]
    test = [_row("q", "knorm", s=3)]

    median_only = mix_predictions(train, test, alpha=0.0)
    mean_only = mix_predictions(train, test, alpha=1.0)
    mixed = mix_predictions(train, test, alpha=0.5)

    assert median_only == [0.0]
    assert mean_only == [pytest.approx(1.0 / 3.0)]
    assert mixed == [pytest.approx(1.0 / 6.0)]


def test_internal_alpha_grid_is_train_only() -> None:
    """Internal CV folds never touch the requested held-out rows."""
    train = []
    for i in range(30):
        train.append(_row(f"a{i}", "knorm", s=i % 4, dq=float(i % 2)))
        train.append(
            _row(f"b{i}", "streaming_llm", s=i % 4, dq=float(i % 3 == 0))
        )
    grid = internal_alpha_grid(train, alphas=[0.0, 0.5])

    assert set(grid) == {0.0, 0.5}
    for entry in grid.values():
        assert math.isfinite(entry.mean_relative_improvement)
        assert entry.min_top_decile_lift >= 0.0


def test_select_alpha_prefers_feasible_then_improvement() -> None:
    """Selection picks the best-improvement alpha among lift-feasible."""
    grid = {
        0.0: (0.20, 1.5),
        0.1: (0.15, 2.2),
        0.2: (0.10, 2.4),
    }
    assert select_alpha(grid, min_lift=2.0) == 0.1


def test_select_alpha_falls_back_to_max_lift() -> None:
    """Without feasible alphas, selection maximizes internal lift."""
    grid = {
        0.0: (0.20, 1.2),
        0.1: (0.15, 1.9),
        0.2: (0.10, 1.9),
    }
    assert select_alpha(grid, min_lift=2.0) == 0.1


def test_run_median_mean_mix_scores_all_test_rows() -> None:
    """The end-to-end runner predicts every canonical test row."""
    rows = []
    for i in range(60):
        for compressor in (
            "knorm",
            "streaming_llm",
            "expected_attention",
        ):
            rows.append(
                _row(
                    f"p{i}",
                    compressor,
                    s=i % 8,
                    dq=float((i + len(compressor)) % 3 == 0),
                )
            )

    result = run_median_mean_mix(
        rows,
        compressors=[
            "knorm",
            "streaming_llm",
            "expected_attention",
        ],
        seed=0,
        alphas=[0.0, 0.1],
    )

    assert set(result.alpha_by_heldout) == {
        "knorm",
        "streaming_llm",
        "expected_attention",
    }
    scored = [
        row for row in result.predicted_rows if row["scored_by_predictor"]
    ]
    assert scored
    for row in scored:
        assert math.isfinite(float(row["predicted_dq"]))
    forbidden = {
        "q_ref",
        "q_hybrid",
        "damaged",
        "major_damage",
        "relative_s",
        "ref_len",
        "prompt_id",
        "compressor",
    }
    assert forbidden.isdisjoint(result.model_input_fields)
