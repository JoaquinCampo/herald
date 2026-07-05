"""Evaluate the multi-task severity candidate (locked metrics).

Per canonical leave-one-compressor-out split: fit dq as a function of
(task, ratio, position, causal features, compressor severity) on the
split's train compressors plus the training-only donors, one sample
per (grid point, compressor). The unseen compressor is scored at a
hypothetical severity level (a knob), and the safety cut is in dq
units. Knobs are selected by prompt-disjoint internal
cross-validation that holds one train primary out of the fit pool
(donors always stay) and requires the epsilon budget on it; the fold
fit is already conservatively biased (one compressor poorer than
deployment), so folds use the full epsilon budget, no extra factor.
Internal folds use the verified fast replica of the locked tau rule;
final numbers come from the unchanged ``herald.controller_metrics``
evaluator on the canonical splits, written as a lock-shaped candidate
summary for ``scripts/check_mission.py``.
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
    compressor_severity,
    feature_names,
    fit_severity_model,
    predict_severity_risk,
    severity_level_value,
    shape_scores,
    smooth_risk_causal,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
DONOR_COMPRESSORS = ["random", "snapkv"]
SEVERITY_LEVELS = ["max", "second"]
CUTS = [0.0025, 0.005, 0.01, 0.02, 0.04]
LABELS = ["dq"]
CONFIG = "deep"
ENSEMBLE_SEEDS = 3

Knob = tuple[str, str, float]


def main() -> None:
    """Run the severity candidate through the locked metrics."""
    args = _parse_args()
    rows = _load_rows(
        args.dataset, PRIMARY_COMPRESSORS + DONOR_COMPRESSORS
    )
    primary = [
        row
        for row in rows
        if str(row["compressor"]) in PRIMARY_COMPRESSORS
    ]
    # Fingerprint BEFORE attaching any derived columns: the lock
    # hashes the pristine rows, and check_mission compares to it.
    fingerprint = dataset_fingerprint(primary)
    _attach_suffix_labels(rows)
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
        fold_comps = sorted(
            {str(row["compressor"]) for row in split.train}
        )
        fold_metrics: dict[
            tuple[Knob, str], list[tuple[float, float]]
        ] = {
            (knob, comp): []
            for knob in _knobs(args)
            for comp in fold_comps
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
        label, level, cut = _select_knob(outcomes)

        severity = compressor_severity(fit_pool)
        models = _fit_ensemble(
            fit_pool, features, severity, label, args.seed
        )
        severity_value = severity_level_value(severity, level)
        result = evaluate_controller_split(
            _with_scores(
                models,
                split.train,
                features,
                severity_value,
                cut,
            ),
            _with_scores(
                models, split.test, features, severity_value, cut
            ),
            prediction_key="predicted_dq",
            epsilon=args.epsilon,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        result["selected_knob"] = {
            "label": label,
            "severity_level": level,
            "severity_value": severity_value,
            "cut": cut,
        }
        result["severity_by_compressor"] = severity
        result["internal_cv"] = cv_log
        heldouts[heldout] = result
        oracle = result["oracle"]["mean_savings"]
        print(
            f"[{heldout}] {label}/{level}/cut={cut} "
            f"tau={result['tau']:.4f} "
            f"savings={result['mean_savings']:.4f} "
            f"cost={result['mean_cost']:.4f} "
            f"budget={result['budget_respected']} "
            f"recall={result['catastrophe_recall_at_10_fpr']} "
            f"oracle_gap={oracle - result['mean_savings']:.4f}",
            flush=True,
        )

    summary = {
        "candidate": "severity_multitask",
        "epsilon": args.epsilon,
        "config": {
            "seed": args.seed,
            "test_group_fraction": 0.25,
            "compressors": PRIMARY_COMPRESSORS,
            "donors": DONOR_COMPRESSORS,
            "cut_grid": CUTS,
            "severity_levels": SEVERITY_LEVELS,
            "labels": LABELS,
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


def _knobs(args: argparse.Namespace) -> list[Knob]:
    return [
        (label, level, cut)
        for label in LABELS
        for level in SEVERITY_LEVELS
        for cut in CUTS
    ]


def _run_fold_cell(
    args: argparse.Namespace,
    features: list[str],
    fit_pool: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    fold_comp: str,
    half: int,
    fold_metrics: dict[
        tuple[Knob, str], list[tuple[float, float]]
    ],
) -> None:
    """One prompt-disjoint internal fold across the knob grid."""
    inner_fit = [
        row
        for row in fit_pool
        if str(row["compressor"]) != fold_comp
        and _prompt_half(row) != half
    ]
    fold_train = [
        row
        for row in train_rows
        if str(row["compressor"]) != fold_comp
        and _prompt_half(row) != half
    ]
    fold_eval = [
        row
        for row in train_rows
        if str(row["compressor"]) == fold_comp
        and _prompt_half(row) == half
    ]
    severity = compressor_severity(inner_fit)
    for label in LABELS:
        models = _fit_ensemble(
            inner_fit, features, severity, label, args.seed
        )
        for level in SEVERITY_LEVELS:
            value = severity_level_value(severity, level)
            for cut in CUTS:
                tau = fast_select_tau(
                    _with_scores(
                        models, fold_train, features, value, cut
                    ),
                    prediction_key="predicted_dq",
                    epsilon=args.epsilon,
                )
                pairs = _policy_pairs(
                    _with_scores(
                        models, fold_eval, features, value, cut
                    ),
                    tau,
                )
                fold_metrics[
                    ((label, level, cut), fold_comp)
                ].extend(pairs)


def _fit_ensemble(
    fit_rows: list[dict[str, Any]],
    features: list[str],
    severity: dict[str, float],
    label: str,
    seed: int,
) -> list[Any]:
    """Seed ensemble of severity models (prediction = mean)."""
    return [
        fit_severity_model(
            fit_rows,
            features,
            severity,
            config=CONFIG,
            seed=seed + offset,
            label_key=label,
        )
        for offset in range(ENSEMBLE_SEEDS)
    ]


def _aggregate_folds(
    args: argparse.Namespace,
    heldout: str,
    fold_comps: list[str],
    fold_metrics: dict[
        tuple[Knob, str], list[tuple[float, float]]
    ],
    cv_log: dict[str, Any],
) -> dict[Knob, tuple[bool, float]]:
    """Per-knob feasibility and worst fold savings."""
    outcomes: dict[Knob, tuple[bool, float]] = {}
    for knob in _knobs(args):
        feasible = True
        comp_savings: list[float] = []
        for comp in fold_comps:
            pairs = fold_metrics[(knob, comp)]
            savings = sum(s for s, _ in pairs) / len(pairs)
            costs = np.asarray(
                [c for _, c in pairs], dtype=np.float64
            )
            cost = float(costs.mean())
            # One-SE rule against knob-selection winner's curse:
            # feasibility requires the fold cost plus its standard
            # error to clear the budget, so the margin scales with
            # fold noise instead of being hand-picked.
            se = float(costs.std() / np.sqrt(len(costs)))
            within = cost + se <= args.epsilon
            feasible = feasible and within
            comp_savings.append(savings if within else 0.0)
            label, level, cut = knob
            key = f"{label}/{level}/cut={cut}/fold={comp}"
            cv_log[key] = {
                "savings": savings,
                "cost": cost,
                "within_budget": within,
            }
            print(
                f"  [{heldout}] {key} savings={savings:.4f} "
                f"cost={cost:.4f}",
                flush=True,
            )
        outcomes[knob] = (feasible, min(comp_savings))
    return outcomes


def _select_knob(
    outcomes: dict[Knob, tuple[bool, float]],
) -> Knob:
    """Best feasible knob; most conservative fallback."""
    feasible = {
        knob: savings
        for knob, (ok, savings) in outcomes.items()
        if ok
    }
    if not feasible:
        return min(outcomes, key=lambda k: (k[2], k[0], k[1]))
    return min(
        feasible,
        key=lambda k: (-feasible[k], k[2], k[0], k[1]),
    )


def _attach_suffix_labels(rows: list[dict[str, Any]]) -> None:
    """Add ``dq_suffix``: max dq at this or any later position.

    Computed per (compressor, decision group); a training label only,
    never a model input.
    """
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["compressor"]),
            row.get("model"),
            row.get("task"),
            row.get("prompt_id"),
            row.get("ratio"),
        )
        groups.setdefault(key, []).append(row)
    for members in groups.values():
        members.sort(key=lambda row: int(row["s"]))
        running = float("-inf")
        for row in reversed(members):
            running = max(running, float(row["dq"]))
            row["dq_suffix"] = running


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
    models: list[Any],
    rows: list[dict[str, Any]],
    features: list[str],
    severity_value: float,
    cut: float,
) -> list[dict[str, Any]]:
    risk = np.mean(
        [
            predict_severity_risk(
                model,
                rows,
                features,
                severity_value=severity_value,
            )
            for model in models
        ],
        axis=0,
    )
    risk = smooth_risk_causal(rows, risk)
    positions = np.asarray(
        [float(row["feat__position"]) for row in rows],
        dtype=np.float64,
    )
    scores = shape_scores(risk, positions, cut=cut)
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
            policy_choice(
                group, prediction_key="predicted_dq", tau=tau
            )
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
        default=Path(
            "results/predictor/experiments/severity_multitask"
        ),
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
