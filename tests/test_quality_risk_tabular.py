"""CPU tests for quality-risk tabular training helpers."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_quality_risk_dev import (  # noqa: E402
    delong_auc_and_var,
    delong_diff_var,
    delong_stats,
    delong_stats_fast,
    fit_calibrator,
    group_ids_from_order,
    preorder_tie_groups,
    weighted_auc,
)
from train_quality_risk_tabular import (  # noqa: E402
    fit_baselines,
    loss_weights,
    xgb_parameters,
)


def _frame() -> pd.DataFrame:
    rows = []
    for prompt in ("p1", "p2"):
        for run in (f"{prompt}-a", f"{prompt}-b"):
            for _token in range(4):
                rows.append(
                    {
                        "run_id": run,
                        "prompt_id": prompt,
                        "action_code": 0 if run.endswith("a") else 1,
                        "fold": 0 if prompt == "p1" else 1,
                        "damage": 1 if prompt == "p1" else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_loss_weights_prompt_action_token_equal() -> None:
    frame = _frame()
    weights = loss_weights(frame)
    assert float(weights.mean()) == 1.0
    # p1 has 8 rows over 2 runs; each p1 run weight sums equally.
    run_sums = (
        pd.Series(weights).groupby(frame["run_id"]).sum().round(6).unique()
    )
    assert len(run_sums) == 1


def test_fit_baselines_global_and_action_means() -> None:
    frame = _frame()
    weights = loss_weights(frame)
    out_global = np.full(len(frame), np.nan)
    out_action = np.full(len(frame), np.nan)
    fit_baselines(frame, weights, 1, out_global, out_action)
    # Fold 1 (p2, damage 0) predicted from fold 0 (p1, damage 1).
    assert (out_global[frame["fold"].to_numpy() == 1] == 1.0).all()
    assert (out_action[frame["fold"].to_numpy() == 1] == 1.0).all()


def test_xgb_parameters_binary_logistic() -> None:
    params = xgb_parameters("cpu")
    assert params["objective"] == "binary:logistic"
    assert params["eval_metric"] == "logloss"
    assert params["seed"] == 2718


def test_weighted_auc_perfect_and_random() -> None:
    truth = np.array([0.0, 0.0, 1.0, 1.0])
    assert weighted_auc(truth, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert weighted_auc(truth, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert weighted_auc(truth, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5


def test_fit_calibrator_selects_valid_method() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0.05, 0.95, size=200)
    truth = (rng.uniform(size=200) < scores).astype(float)
    weights = np.ones(200)
    name, calibrator, loss = fit_calibrator(scores, truth, weights)
    assert name in ("isotonic", "platt")
    assert 0.0 < loss < 1.0
    _ = calibrator


def test_delong_matches_weighted_auc_and_covers() -> None:
    rng = np.random.default_rng(1)
    truth = (rng.uniform(size=400) < 0.4).astype(float)
    scores = rng.normal(size=400) + truth
    weights = np.ones(400)
    auc, var = delong_auc_and_var(truth, scores, weights)
    assert auc == weighted_auc(truth, scores)
    assert 0.5 < auc < 1.0
    assert 0.0 < var < 0.01
    # Frequency weights of two give the same AUC with quarter variance.
    auc2, var2 = delong_auc_and_var(truth, scores, 2 * weights)
    assert auc2 == auc
    assert var2 == var / 2
    # Ties agree with sklearn.
    tied = np.round(scores, 0)
    auc3, _ = delong_auc_and_var(truth, tied, weights)
    from sklearn.metrics import roc_auc_score

    assert auc3 == roc_auc_score(truth, tied)


def test_delong_paired_diff_var() -> None:
    rng = np.random.default_rng(2)
    truth = (rng.uniform(size=300) < 0.4).astype(float)
    score_a = rng.normal(size=300) + truth
    score_b = rng.normal(size=300) + truth
    weights = np.ones(300)
    order_a, groups_a = preorder_tie_groups(score_a)
    order_b, groups_b = preorder_tie_groups(score_b)
    stats_a = delong_stats(truth, weights, order_a, groups_a)
    stats_b = delong_stats(truth, weights, order_b, groups_b)
    assert stats_a[0] == delong_auc_and_var(truth, score_a, weights)[0]
    self_var = delong_diff_var(truth, weights, stats_a, stats_a)
    assert self_var == 0.0
    diff_var = delong_diff_var(truth, weights, stats_a, stats_b)
    assert diff_var > 0.0
    assert diff_var < stats_a[1] + stats_b[1]


def test_delong_fast_matches_exact() -> None:
    rng = np.random.default_rng(3)
    truth = (rng.uniform(size=500) < 0.4).astype(float)
    scores = np.round(rng.normal(size=500) + truth, 1)
    weights = rng.integers(0, 3, size=500).astype(float) + 0.5
    order, groups = preorder_tie_groups(scores)
    group_id = group_ids_from_order(scores, order)
    slow = delong_stats(truth, weights, order, groups)
    fast = delong_stats_fast(truth, weights, order, group_id)
    assert fast[0] == slow[0]
    assert fast[1] == slow[1]
    assert np.allclose(fast[2], slow[2], equal_nan=True)
    assert np.allclose(fast[3], slow[3], equal_nan=True)
