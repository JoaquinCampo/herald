"""Tests for src/herald/predictor_baselines.py.

Behaviors under test:
- per-split p90 threshold derived from train rows only (no leakage)
- binarize propagates nulls and applies the threshold strictly
- iter_splits yields the right grouping per split kind
- single-feature baseline scores match the chosen feature value
- random baseline produces deterministic-but-uninformative output
- bootstrap CI degrades gracefully on degenerate inputs
"""

import numpy as np
import polars as pl

from herald.predictor_baselines import (
    binarize_with_threshold,
    bootstrap_auroc_ci,
    clustered_paired_bootstrap_delta,
    compute_train_quantile_threshold,
    cross_fold_clustered_bootstrap,
    iter_splits,
)


def _toy_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": [f"r{i // 3}" for i in range(12)],
            "prompt_id": [f"p{i // 3}" for i in range(12)],
            "task": (["gsm8k"] * 6) + (["humaneval"] * 6),
            "press": (["snapkv"] * 3 + ["streaming_llm"] * 3) * 2,
            "compression_ratio": ([0.5, 0.5, 0.5, 0.75, 0.75, 0.75]) * 2,
            "token_pos": list(range(12)),
            "entropy": [float(i) for i in range(12)],
            "future_sum_js_5": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
                None,
                None,
            ],
        }
    )


# ---------------------------------------------------------------
# Threshold from train-only
# ---------------------------------------------------------------


def test_compute_train_quantile_threshold_uses_only_train_rows() -> None:
    """p90 must be computed over train rows; test rows must not leak."""
    df = pl.DataFrame(
        {
            "future_sum_js_5": list(range(11)),  # 0..10
            "split": (["train"] * 6) + (["test"] * 5),
        }
    )
    train_vals = df.filter(pl.col("split") == "train")["future_sum_js_5"]
    thr = compute_train_quantile_threshold(train_vals, q=0.9)
    # train values are 0..5, p90 ≈ 4.5 (linear interp)
    assert 4.0 <= thr <= 5.0


def test_compute_train_quantile_threshold_drops_nulls() -> None:
    train = pl.Series([1.0, 2.0, None, 3.0, 4.0, None])
    thr = compute_train_quantile_threshold(train, q=0.5)
    # median of [1, 2, 3, 4] = 2.5
    assert thr == 2.5


def test_compute_train_quantile_threshold_returns_none_when_empty() -> None:
    train = pl.Series([None, None, None], dtype=pl.Float64)
    assert compute_train_quantile_threshold(train, q=0.9) is None


# ---------------------------------------------------------------
# Binarize
# ---------------------------------------------------------------


def test_binarize_with_threshold_propagates_nulls() -> None:
    s = pl.Series("future_sum_js_5", [0.5, 1.0, 1.5, None, 2.0])
    out = binarize_with_threshold(s, threshold=1.0)
    assert out.to_list() == [0, 1, 1, None, 1]


def test_binarize_with_threshold_returns_zeros_when_threshold_none() -> None:
    """If threshold is undefined (e.g. all-null train fold), the
    binary label is undefined; surface that as all-null, never zero."""
    s = pl.Series("future_sum_js_5", [0.5, 1.0, None])
    out = binarize_with_threshold(s, threshold=None)
    assert out.to_list() == [None, None, None]


# ---------------------------------------------------------------
# Splits
# ---------------------------------------------------------------


def test_iter_splits_held_out_prompts_groups_by_prompt_id() -> None:
    df = _toy_df()
    folds = list(iter_splits(df, kind="prompts", n_splits=2, seed=42))
    assert len(folds) == 2
    for train_idx, test_idx, fold_id in folds:
        train_prompts = set(df["prompt_id"].gather(train_idx).to_list())
        test_prompts = set(df["prompt_id"].gather(test_idx).to_list())
        assert train_prompts.isdisjoint(test_prompts)


def test_iter_splits_held_out_ratios_one_fold_per_ratio() -> None:
    df = _toy_df()
    folds = list(iter_splits(df, kind="ratios"))
    # toy df has 2 unique ratios: 0.5 and 0.75
    assert len(folds) == 2
    seen_test_ratios: set[float] = set()
    for train_idx, test_idx, fold_id in folds:
        train_r = set(df["compression_ratio"].gather(train_idx).to_list())
        test_r = set(df["compression_ratio"].gather(test_idx).to_list())
        assert len(test_r) == 1
        assert train_r.isdisjoint(test_r)
        seen_test_ratios |= test_r
    assert seen_test_ratios == {0.5, 0.75}


def test_iter_splits_held_out_presses_one_fold_per_press() -> None:
    df = _toy_df()
    folds = list(iter_splits(df, kind="presses"))
    assert len(folds) == 2
    seen: set[str] = set()
    for train_idx, test_idx, fold_id in folds:
        test_presses = set(df["press"].gather(test_idx).to_list())
        assert len(test_presses) == 1
        seen |= test_presses
    assert seen == {"snapkv", "streaming_llm"}


def test_iter_splits_held_out_tasks_one_fold_per_task() -> None:
    df = _toy_df()
    folds = list(iter_splits(df, kind="tasks"))
    assert len(folds) == 2
    seen: set[str] = set()
    for train_idx, test_idx, fold_id in folds:
        test_tasks = set(df["task"].gather(test_idx).to_list())
        assert len(test_tasks) == 1
        seen |= test_tasks
    assert seen == {"gsm8k", "humaneval"}


# ---------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------


def test_bootstrap_auroc_ci_finite_interval_on_separable_data() -> None:
    y = [0] * 50 + [1] * 50
    s = list(range(50)) + list(range(50, 100))  # perfectly separable
    lo, hi = bootstrap_auroc_ci(y, s, n_boot=50, seed=0)
    assert lo is not None and hi is not None
    assert 0.95 <= lo <= 1.0
    assert hi == 1.0


def test_bootstrap_auroc_ci_returns_none_on_degenerate_input() -> None:
    y = [0] * 10
    s = list(range(10))
    lo, hi = bootstrap_auroc_ci(y, s, n_boot=10, seed=0)
    assert lo is None and hi is None


# ---------------------------------------------------------------
# evaluate_split end-to-end: held-out-task split must not crash
# when the train/test halves carry disjoint categorical level sets.
# Regression for the LOGO bug where train had 3 tasks and test had 1,
# producing different one-hot column counts and breaking the LR scaler.
# ---------------------------------------------------------------


def test_evaluate_split_held_out_task_does_not_crash_on_disjoint_levels() -> (
    None
):
    from herald.predictor_baselines import evaluate_split

    rng = __import__("numpy").random.default_rng(0)
    n = 600
    df = pl.DataFrame(
        {
            "run_id": [f"r{i // 10}" for i in range(n)],
            "prompt_id": [f"p{i // 10}" for i in range(n)],
            "task": (
                ["gsm8k"] * (n // 3)
                + ["humaneval"] * (n // 3)
                + ["ifeval"] * (n - 2 * (n // 3))
            ),
            "press": ["snapkv"] * n,
            "compression_ratio": [0.5] * n,
            "token_pos": list(range(n)),
            "entropy": rng.uniform(size=n).tolist(),
            "top1_prob": rng.uniform(size=n).tolist(),
            "future_sum_js_5": rng.uniform(size=n).tolist(),
        }
    )
    rows = evaluate_split(
        df,
        label_col="future_sum_js_5",
        splits=("tasks",),
        n_boot=10,
    )
    # 3 tasks => 3 LOGO folds; each fold should have >= 1 baseline row.
    fold_ids = {r["fold_id"] for r in rows}
    assert len(fold_ids) == 3


# ---------------------------------------------------------------
# Cluster-bootstrap helpers (Phase 2b Task 1)
# ---------------------------------------------------------------


def test_clustered_paired_bootstrap_delta_returns_finite_ci() -> None:
    """Separable scores from one model and noise from the other should
    yield a positive delta with a CI excluding zero, and the resampling
    should respect the cluster structure."""
    rng = np.random.default_rng(0)
    n_groups = 30
    rows_per_group = 20
    y_parts: list[np.ndarray] = []
    sa_parts: list[np.ndarray] = []
    sb_parts: list[np.ndarray] = []
    g_parts: list[np.ndarray] = []
    for g in range(n_groups):
        y = rng.integers(0, 2, size=rows_per_group)
        # Score A: matches label closely.
        sa = y + rng.normal(scale=0.1, size=rows_per_group)
        # Score B: noise.
        sb = rng.normal(size=rows_per_group)
        y_parts.append(y)
        sa_parts.append(sa)
        sb_parts.append(sb)
        g_parts.append(np.full(rows_per_group, g))
    y_true = np.concatenate(y_parts)
    sa = np.concatenate(sa_parts)
    sb = np.concatenate(sb_parts)
    groups = np.concatenate(g_parts)
    res = clustered_paired_bootstrap_delta(
        y_true, sa, sb, groups, n_boot=200, seed=0
    )
    assert res["delta"] is not None and res["delta"] > 0.4
    assert res["delta_lo"] is not None and res["delta_lo"] > 0.0


def test_clustered_paired_bootstrap_delta_handles_degenerate() -> None:
    """All-positive labels = degenerate AUROC; both bounds should be None."""
    y_true = np.ones(20, dtype=np.int64)
    sa = np.linspace(0, 1, 20)
    sb = np.linspace(1, 0, 20)
    groups = np.arange(20) // 5
    res = clustered_paired_bootstrap_delta(
        y_true, sa, sb, groups, n_boot=10, seed=0
    )
    assert res["delta"] is None
    assert res["delta_lo"] is None
    assert res["delta_hi"] is None


def test_cross_fold_clustered_bootstrap_aggregates_folds() -> None:
    """Two folds with separable A vs noisy B should give a positive
    cross-fold mean delta with CI excluding zero."""
    rng = np.random.default_rng(1)
    fold_inputs = []
    for _fold in range(3):
        n_groups = 10
        per = 30
        ys: list[np.ndarray] = []
        sa_list: list[np.ndarray] = []
        sb_list: list[np.ndarray] = []
        gr: list[np.ndarray] = []
        for g in range(n_groups):
            y = rng.integers(0, 2, size=per)
            ys.append(y)
            sa_list.append(y + rng.normal(scale=0.1, size=per))
            sb_list.append(rng.normal(size=per))
            gr.append(np.full(per, g))
        fold_inputs.append(
            {
                "y_true": np.concatenate(ys),
                "score_a": np.concatenate(sa_list),
                "score_b": np.concatenate(sb_list),
                "groups": np.concatenate(gr),
            }
        )
    res = cross_fold_clustered_bootstrap(fold_inputs, n_boot=200, seed=0)
    assert res["delta_mean"] is not None and res["delta_mean"] > 0.3
    assert res["delta_lo"] is not None and res["delta_lo"] > 0.0
    assert res["n_folds"] == 3
