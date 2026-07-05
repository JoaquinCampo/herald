"""Locked controller metrics: savings at quality budget, safety recall.

Spec: ``docs/implementation/controller_metrics.md``. Rationale:
``docs/_why/4_controller_metrics.md``. The policy switches at the
earliest sampled position whose predicted dq clears a threshold that
is selected train-only; metrics are the realized memory savings under
a mean-cost budget, plus catastrophe recall at bounded FPR.
"""

import hashlib
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

EPSILON = 0.01
MAJOR_DAMAGE_THRESHOLD = 0.5
MAX_FPR = 0.10
GROUP_FIELDS: tuple[str, ...] = ("model", "task", "prompt_id", "ratio")
CLUSTER_FIELDS: tuple[str, ...] = ("model", "task", "prompt_id")
TAU_GRID_SIZE = 201


@dataclass(frozen=True)
class PolicyChoice:
    """Outcome of one decision group under a switch policy."""

    switched: bool
    s: int | None
    savings: float
    cost: float


def decision_groups(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[object, ...], list[dict[str, Any]]]:
    """Group rows into per-(prompt, ratio) switch decisions."""
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in GROUP_FIELDS)
        groups[key].append(row)
    for members in groups.values():
        members.sort(key=lambda row: int(row["s"]))
    return dict(groups)


def policy_choice(
    group: Sequence[dict[str, Any]],
    *,
    prediction_key: str,
    tau: float,
) -> PolicyChoice:
    """Switch at the earliest position predicted safe enough."""
    for row in sorted(group, key=lambda row: int(row["s"])):
        if float(row[prediction_key]) <= tau:
            return _switch_at(row)
    return PolicyChoice(switched=False, s=None, savings=0.0, cost=0.0)


def oracle_choice(group: Sequence[dict[str, Any]]) -> PolicyChoice:
    """Switch at the earliest truly zero-damage position."""
    for row in sorted(group, key=lambda row: int(row["s"])):
        if float(row["dq"]) <= 0.0:
            return _switch_at(row)
    return PolicyChoice(switched=False, s=None, savings=0.0, cost=0.0)


def always_s0_choice(group: Sequence[dict[str, Any]]) -> PolicyChoice:
    """Switch at the earliest sampled position unconditionally."""
    first = min(group, key=lambda row: int(row["s"]))
    return _switch_at(first)


def select_tau(
    rows: Sequence[dict[str, Any]],
    *,
    prediction_key: str,
    epsilon: float = EPSILON,
    grid_size: int = TAU_GRID_SIZE,
) -> float:
    """Pick tau train-only: max mean savings under the cost budget.

    The grid is quantiles of the train predictions plus one value
    below the minimum (the never-switch policy), so the feasible set
    is never empty. Ties break to the smaller tau.
    """
    groups = decision_groups(rows)
    predictions = np.asarray(
        [float(row[prediction_key]) for row in rows], dtype=np.float64
    )
    if predictions.size == 0:
        raise ValueError("cannot select tau without rows")
    quantiles = np.quantile(predictions, np.linspace(0.0, 1.0, grid_size))
    below_min = float(predictions.min()) - 1.0
    grid = sorted({below_min, *(float(q) for q in quantiles)})
    best_tau = below_min
    best_savings = -1.0
    for tau in grid:
        choices = [
            policy_choice(group, prediction_key=prediction_key, tau=tau)
            for group in groups.values()
        ]
        cost = float(np.mean([choice.cost for choice in choices]))
        savings = float(np.mean([choice.savings for choice in choices]))
        if cost <= epsilon and savings > best_savings:
            best_savings = savings
            best_tau = tau
    return best_tau


def evaluate_controller_split(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    prediction_key: str,
    epsilon: float = EPSILON,
    seed: int = 0,
    bootstrap_resamples: int = 200,
) -> dict[str, Any]:
    """Evaluate one held-out split under the locked controller metrics."""
    tau = select_tau(
        train_rows, prediction_key=prediction_key, epsilon=epsilon
    )
    groups = decision_groups(test_rows)
    choices = {
        key: policy_choice(group, prediction_key=prediction_key, tau=tau)
        for key, group in groups.items()
    }
    savings = np.asarray(
        [choice.savings for choice in choices.values()],
        dtype=np.float64,
    )
    costs = np.asarray(
        [choice.cost for choice in choices.values()], dtype=np.float64
    )
    mean_savings = float(savings.mean())
    mean_cost = float(costs.mean())

    labels = [
        1 if float(row["dq"]) >= MAJOR_DAMAGE_THRESHOLD else 0
        for row in test_rows
    ]
    scores = [float(row[prediction_key]) for row in test_rows]

    return {
        "tau": tau,
        "epsilon": epsilon,
        "n_test_groups": len(groups),
        "n_test_rows": len(test_rows),
        "mean_savings": mean_savings,
        "mean_cost": mean_cost,
        "budget_respected": mean_cost <= epsilon,
        "switch_rate": float(
            np.mean(
                [
                    1.0 if choice.switched else 0.0
                    for choice in choices.values()
                ]
            )
        ),
        "catastrophe_recall_at_10_fpr": recall_at_fpr(
            labels, scores, max_fpr=MAX_FPR
        ),
        "oracle": _reference_metrics(groups, oracle_choice),
        "never": {"mean_savings": 0.0, "mean_cost": 0.0},
        "always_s0": _reference_metrics(groups, always_s0_choice),
        "bootstrap_ci": _cluster_bootstrap(
            groups,
            choices,
            seed=seed,
            n_resamples=bootstrap_resamples,
        ),
    }


def recall_at_fpr(
    labels: Sequence[int],
    scores: Sequence[float],
    *,
    max_fpr: float,
) -> float | None:
    """Best recall with false-positive rate at most ``max_fpr``.

    Ranking by descending score; matches the canonical definition in
    ``herald.switch_baselines``.
    """
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(
        range(len(labels)), key=lambda idx: scores[idx], reverse=True
    )
    true_positives = 0
    false_positives = 0
    best_recall = 0.0
    for idx in ranked:
        if labels[idx]:
            true_positives += 1
        else:
            false_positives += 1
        if false_positives / negatives <= max_fpr:
            best_recall = max(best_recall, true_positives / positives)
    return best_recall


def _reference_metrics(
    groups: dict[tuple[object, ...], list[dict[str, Any]]],
    choose: Any,
) -> dict[str, float]:
    """Mean savings and cost for a prediction-free reference policy."""
    choices = [choose(group) for group in groups.values()]
    return {
        "mean_savings": float(
            np.mean([choice.savings for choice in choices])
        ),
        "mean_cost": float(np.mean([choice.cost for choice in choices])),
    }


def _cluster_bootstrap(
    groups: dict[tuple[object, ...], list[dict[str, Any]]],
    choices: dict[tuple[object, ...], PolicyChoice],
    *,
    seed: int,
    n_resamples: int,
) -> dict[str, Any]:
    """Bootstrap mean savings and cost by prompt cluster."""
    clusters: dict[tuple[object, ...], list[tuple[object, ...]]] = (
        defaultdict(list)
    )
    for key in groups:
        cluster = tuple(
            key[GROUP_FIELDS.index(field)] for field in CLUSTER_FIELDS
        )
        clusters[cluster].append(key)
    cluster_keys = list(clusters.values())
    result: dict[str, Any] = {
        "cluster_fields": list(CLUSTER_FIELDS),
        "n_clusters": len(cluster_keys),
        "n_bootstrap": n_resamples,
        "savings_low": None,
        "savings_high": None,
        "cost_low": None,
        "cost_high": None,
    }
    if n_resamples <= 0 or not cluster_keys:
        return result
    rng = np.random.default_rng(_stable_seed(seed, "controller"))
    savings_samples: list[float] = []
    cost_samples: list[float] = []
    for _ in range(n_resamples):
        draws = rng.integers(0, len(cluster_keys), size=len(cluster_keys))
        savings: list[float] = []
        costs: list[float] = []
        for draw in draws.tolist():
            for key in cluster_keys[int(draw)]:
                savings.append(choices[key].savings)
                costs.append(choices[key].cost)
        savings_samples.append(float(np.mean(savings)))
        cost_samples.append(float(np.mean(costs)))
    result["savings_low"] = float(np.quantile(savings_samples, 0.025))
    result["savings_high"] = float(np.quantile(savings_samples, 0.975))
    result["cost_low"] = float(np.quantile(cost_samples, 0.025))
    result["cost_high"] = float(np.quantile(cost_samples, 0.975))
    return result


def _switch_at(row: dict[str, Any]) -> PolicyChoice:
    """Build the outcome of switching at one sampled position."""
    s = int(row["s"])
    ref_len = float(row["ref_len"])
    if not math.isfinite(ref_len) or ref_len <= 0:
        raise ValueError(f"invalid ref_len {row.get('ref_len')!r}")
    savings = max(0.0, 1.0 - s / ref_len)
    return PolicyChoice(
        switched=True,
        s=s,
        savings=savings,
        cost=float(row["dq"]),
    )


def _stable_seed(seed: int, *parts: str) -> int:
    """Derive a deterministic NumPy seed from labels."""
    digest = hashlib.sha256("\0".join([str(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "big")
