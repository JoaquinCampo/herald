"""Evaluate the consensus-risk candidate (locked metrics).

Per canonical leave-one-compressor-out split: dedupe the split's
train compressors plus the training-only donors to grid points
labelled with the CONSENSUS damage (mean dq over the compressors
present), then fit two heads per seed: a classifier
P(consensus dq > 0) that drives the safe region, and a magnitude
regressor that grades the risky region for catastrophe recall. The
perfect-label ceiling of this label clears the 0.10 rung on all
three held-out compressors, unlike the worst-case label which is
nearly binary on knorm.

The probability cut is the only knob, selected by prompt-disjoint
internal cross-validation (hold one train primary out of the fit
pool, donors always stay; eval on held prompts of the held
compressor) under the one-SE feasibility rule. Internal folds use
the verified fast replica of the locked tau rule; final numbers come
from the unchanged ``herald.controller_metrics`` evaluator on the
canonical splits, written as a lock-shaped candidate summary for
``scripts/check_mission.py``.
"""

import argparse
import hashlib
import json
import math
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from herald.controller_metrics import (
    EPSILON,
    decision_groups,
    evaluate_controller_split,
    policy_choice,
)
from herald.fast_controller import fast_select_tau
from herald.switch_baselines import (
    _is_test_group,
    dataset_fingerprint,
    leave_one_compressor_splits,
)
from herald.switch_risk import (
    attach_cumulative_features,
    build_consensus_points,
    feature_names,
    fit_consensus_model,
    predict_risk,
    shape_scores,
    smooth_risk_causal,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
DONOR_COMPRESSORS = ["random", "snapkv"]
CUTS = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.25]
CONFIG = "deep"
ENSEMBLE_SEEDS = 3


def main() -> None:
    """Run the consensus-risk candidate through the locked metrics."""
    args = _parse_args()
    rows = _load_rows(args.dataset, PRIMARY_COMPRESSORS + DONOR_COMPRESSORS)
    primary = [
        row for row in rows if str(row["compressor"]) in PRIMARY_COMPRESSORS
    ]
    # Fingerprint BEFORE attaching any derived columns: the lock
    # hashes the pristine rows, and check_mission compares to it.
    fingerprint = dataset_fingerprint(primary)
    attach_cumulative_features(rows)
    features = feature_names(rows)
    splits = leave_one_compressor_splits(
        primary, compressors=PRIMARY_COMPRESSORS, seed=args.seed
    )
    donor_rows = [
        row
        for row in rows
        if str(row["compressor"]) in DONOR_COMPRESSORS
        and not _is_test_group(row, args.seed, 0.25)
    ]

    heldouts: dict[str, Any] = {}
    for split in splits:
        heldout = split.heldout_compressor
        fit_pool = list(split.train) + donor_rows
        fold_comps = sorted({str(row["compressor"]) for row in split.train})
        fold_metrics: dict[tuple[float, str], list[tuple[float, float]]] = {
            (cut, comp): [] for cut in CUTS for comp in fold_comps
        }
        for fold_comp in fold_comps:
            for half in (0, 1):
                _run_fold_cell(
                    args,
                    features,
                    fit_pool,
                    split.train,
                    fold_comp,
                    half,
                    fold_metrics,
                )
        cv_log: dict[str, Any] = {}
        outcomes = _aggregate_folds(
            args, heldout, fold_comps, fold_metrics, cv_log
        )
        cut = _select_cut(outcomes)

        models = _fit_ensemble(fit_pool, features, args.seed)
        result = evaluate_controller_split(
            _with_scores(models, split.train, features, cut),
            _with_scores(models, split.test, features, cut),
            prediction_key="predicted_dq",
            epsilon=args.epsilon,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        result["selected_knob"] = {"cut": cut}
        result["internal_cv"] = cv_log
        heldouts[heldout] = result
        oracle = result["oracle"]["mean_savings"]
        print(
            f"[{heldout}] cut={cut} "
            f"tau={result['tau']:.4f} "
            f"savings={result['mean_savings']:.4f} "
            f"cost={result['mean_cost']:.4f} "
            f"budget={result['budget_respected']} "
            f"recall={result['catastrophe_recall_at_10_fpr']} "
            f"oracle_gap={oracle - result['mean_savings']:.4f}",
            flush=True,
        )

    summary = {
        "candidate": "consensus_risk",
        "epsilon": args.epsilon,
        "config": {
            "seed": args.seed,
            "test_group_fraction": 0.25,
            "compressors": PRIMARY_COMPRESSORS,
            "donors": DONOR_COMPRESSORS,
            "cut_grid": CUTS,
            "label": "consensus_dq (mean over fit compressors)",
            "heads": ["classifier", "magnitude"],
            "model_config": CONFIG,
            "smoothing": "causal_prev_max",
            "fold_budget": args.epsilon,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "dataset_fingerprint": fingerprint,
        "n_rows": len(primary),
        "heldout": heldouts,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_path}")


def _run_fold_cell(
    args: argparse.Namespace,
    features: list[str],
    fit_pool: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    fold_comp: str,
    half: int,
    fold_metrics: dict[tuple[float, str], list[tuple[float, float]]],
) -> None:
    """One prompt-disjoint internal fold across the cut grid."""
    inner_fit = [
        row
        for row in fit_pool
        if str(row["compressor"]) != fold_comp and _prompt_half(row) != half
    ]
    fold_train = [
        row
        for row in train_rows
        if str(row["compressor"]) != fold_comp and _prompt_half(row) != half
    ]
    fold_eval = [
        row
        for row in train_rows
        if str(row["compressor"]) == fold_comp and _prompt_half(row) == half
    ]
    models = _fit_ensemble(inner_fit, features, args.seed)
    for cut in CUTS:
        tau = fast_select_tau(
            _with_scores(models, fold_train, features, cut),
            prediction_key="predicted_dq",
            epsilon=args.epsilon,
        )
        pairs = _policy_pairs(
            _with_scores(models, fold_eval, features, cut), tau
        )
        fold_metrics[(cut, fold_comp)].extend(pairs)


def _fit_ensemble(
    fit_rows: list[dict[str, Any]],
    features: list[str],
    seed: int,
) -> list[tuple[Any, Any]]:
    """Seed ensemble of (classifier, magnitude) head pairs."""
    points = build_consensus_points(fit_rows)
    return [
        (
            fit_consensus_model(
                points,
                features,
                kind="classifier",
                config=CONFIG,
                seed=seed + offset,
            ),
            fit_consensus_model(
                points,
                features,
                kind="magnitude",
                config=CONFIG,
                seed=seed + offset,
            ),
        )
        for offset in range(ENSEMBLE_SEEDS)
    ]


def _aggregate_folds(
    args: argparse.Namespace,
    heldout: str,
    fold_comps: list[str],
    fold_metrics: dict[tuple[float, str], list[tuple[float, float]]],
    cv_log: dict[str, Any],
) -> dict[float, tuple[bool, float]]:
    """Per-cut feasibility and worst fold savings."""
    outcomes: dict[float, tuple[bool, float]] = {}
    for cut in CUTS:
        feasible = True
        comp_savings: list[float] = []
        for comp in fold_comps:
            pairs = fold_metrics[(cut, comp)]
            savings = sum(s for s, _ in pairs) / len(pairs)
            costs = np.asarray([c for _, c in pairs], dtype=np.float64)
            cost = float(costs.mean())
            # One-SE rule against knob-selection winner's curse:
            # feasibility requires the fold cost plus its standard
            # error to clear the budget, so the margin scales with
            # fold noise instead of being hand-picked.
            se = float(costs.std() / np.sqrt(len(costs)))
            within = cost + se <= args.epsilon
            feasible = feasible and within
            comp_savings.append(savings if within else 0.0)
            key = f"cut={cut}/fold={comp}"
            cv_log[key] = {
                "savings": savings,
                "cost": cost,
                "within_budget": within,
            }
            print(
                f"  [{heldout}] {key} savings={savings:.4f} cost={cost:.4f}",
                flush=True,
            )
        outcomes[cut] = (feasible, min(comp_savings))
    return outcomes


def _select_cut(
    outcomes: dict[float, tuple[bool, float]],
) -> float:
    """Best feasible cut; most conservative fallback."""
    feasible = {cut: savings for cut, (ok, savings) in outcomes.items() if ok}
    if not feasible:
        return min(outcomes)
    return min(feasible, key=lambda c: (-feasible[c], c))


def _prompt_half(row: dict[str, Any]) -> int:
    """Deterministic prompt-group half for internal fold splits."""
    key = "\0".join(
        [
            "risk-fold",
            str(row.get("model")),
            str(row.get("task")),
            str(row.get("prompt_id")),
        ]
    )
    return hashlib.sha256(key.encode()).digest()[0] % 2


def _with_scores(
    models: list[tuple[Any, Any]],
    rows: list[dict[str, Any]],
    features: list[str],
    cut: float,
) -> list[dict[str, Any]]:
    risk = np.mean(
        [predict_risk(clf, rows, features) for clf, _ in models],
        axis=0,
    )
    magnitude = np.mean(
        [predict_risk(mag, rows, features) for _, mag in models],
        axis=0,
    )
    risk = smooth_risk_causal(rows, risk)
    positions = np.asarray(
        [float(row["feat__position"]) for row in rows],
        dtype=np.float64,
    )
    scores = shape_scores(risk, positions, cut=cut, magnitude=magnitude)
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, scores, strict=True):
        copied = dict(row)
        copied["predicted_dq"] = float(score)
        out.append(copied)
    return out


def _policy_pairs(
    rows: list[dict[str, Any]], tau: float
) -> list[tuple[float, float]]:
    """Per-group (savings, cost) of the earliest-below-tau policy."""
    groups = decision_groups(rows)
    return [
        (choice.savings, choice.cost)
        for choice in (
            policy_choice(group, prediction_key="predicted_dq", tau=tau)
            for group in groups.values()
        )
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("results/predictor/switch_dataset.parquet"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/experiments/consensus_risk"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    return parser.parse_args()


def _load_rows(path: Path, compressors: list[str]) -> list[dict[str, Any]]:
    """Load usable rows, mirroring the locked evaluator's filter."""
    parquet = import_module("pyarrow.parquet")
    table = cast(Any, parquet.read_table)(path)
    rows = cast(list[dict[str, Any]], cast(Any, table.to_pylist)())
    selected = set(compressors)
    usable: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("compressor")) not in selected:
            continue
        dq = row.get("dq")
        if dq is None:
            continue
        try:
            value = float(dq)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value):
            continue
        ref_len = row.get("ref_len")
        if ref_len is None or float(ref_len) <= 0:
            continue
        usable.append(row)
    if not usable:
        raise SystemExit("no usable rows after filtering")
    return usable


if __name__ == "__main__":
    main()
