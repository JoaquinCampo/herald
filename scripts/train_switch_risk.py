"""Evaluate the per-row worst-case risk candidate (locked metrics).

Per canonical leave-one-compressor-out split: fit the quantile risk
model on the split's train compressors plus the training-only donors
(random, snapkv), never on the held-out compressor. The safety-cut
knob is selected by internal cross-validation that holds one of the
split's two train primaries out of the fit pool (donors always stay)
AND holds out half of the non-test prompt groups (evaluation rows are
disjoint from fit rows in both compressor and prompt, mirroring the
outer evaluation), runs the unchanged locked tau selection, and
requires half the epsilon budget on the held-out fold compressor
(pre-registered severity headroom for a harsher unseen compressor).
Final numbers come from the unchanged ``herald.controller_metrics``
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

from herald.controller_metrics import (
    EPSILON,
    decision_groups,
    evaluate_controller_split,
    policy_choice,
    select_tau,
)
from herald.switch_baselines import (
    _is_test_group,
    dataset_fingerprint,
    leave_one_compressor_splits,
)
from herald.switch_risk import (
    DEFAULT_CUTS,
    MODEL_CONFIGS,
    build_worst_points,
    feature_names,
    fit_risk_model,
    score_rows,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
DONOR_COMPRESSORS = ["random", "snapkv"]


def main() -> None:
    """Run the risk candidate through the locked controller metrics."""
    args = _parse_args()
    rows = _load_rows(
        args.dataset, PRIMARY_COMPRESSORS + DONOR_COMPRESSORS
    )
    primary = [
        row
        for row in rows
        if str(row["compressor"]) in PRIMARY_COMPRESSORS
    ]
    fingerprint = dataset_fingerprint(primary)
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

        cv_log: dict[str, Any] = {}
        fold_metrics: dict[
            tuple[str, float, str], list[tuple[float, float]]
        ] = {
            (config, cut, comp): []
            for config in args.configs
            for cut in args.cuts
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
        outcomes = _aggregate_folds(
            args, heldout, fold_comps, fold_metrics, cv_log
        )
        config, cut = _select_knob(outcomes)

        points = build_worst_points(fit_pool)
        model = fit_risk_model(
            points, features, config=config, seed=args.seed
        )
        result = evaluate_controller_split(
            _with_scores(model, split.train, features, cut),
            _with_scores(model, split.test, features, cut),
            prediction_key="predicted_dq",
            epsilon=args.epsilon,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        result["selected_cut"] = cut
        result["selected_config"] = config
        result["internal_cv"] = cv_log
        heldouts[heldout] = result
        oracle = result["oracle"]["mean_savings"]
        print(
            f"[{heldout}] cut={cut} tau={result['tau']:.4f} "
            f"savings={result['mean_savings']:.4f} "
            f"cost={result['mean_cost']:.4f} "
            f"budget={result['budget_respected']} "
            f"recall={result['catastrophe_recall_at_10_fpr']} "
            f"oracle_gap={oracle - result['mean_savings']:.4f}",
            flush=True,
        )

    summary = {
        "candidate": "worst_case_risk_xgb",
        "epsilon": args.epsilon,
        "config": {
            "seed": args.seed,
            "test_group_fraction": 0.25,
            "compressors": PRIMARY_COMPRESSORS,
            "donors": DONOR_COMPRESSORS,
            "cut_grid": list(args.cuts),
            "config_grid": list(args.configs),
            "fold_budget": args.epsilon / 2.0,
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
            "results/predictor/experiments/worst_case_risk"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument(
        "--cut",
        action="append",
        dest="cuts",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        choices=sorted(MODEL_CONFIGS),
        default=None,
    )
    args = parser.parse_args()
    if not args.cuts:
        args.cuts = list(DEFAULT_CUTS)
    if not args.configs:
        args.configs = sorted(MODEL_CONFIGS)
    return args


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


def _run_fold_cell(
    args: argparse.Namespace,
    features: list[str],
    fit_pool: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    fold_comp: str,
    half: int,
    fold_metrics: dict[
        tuple[str, float, str], list[tuple[float, float]]
    ],
) -> None:
    """One prompt-disjoint internal fold: fit, tau, evaluate."""
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
    points = build_worst_points(inner_fit)
    for config in args.configs:
        model = fit_risk_model(
            points, features, config=config, seed=args.seed
        )
        for cut in args.cuts:
            tau = select_tau(
                _with_scores(model, fold_train, features, cut),
                prediction_key="predicted_dq",
                epsilon=args.epsilon,
            )
            pairs = _policy_pairs(
                _with_scores(model, fold_eval, features, cut), tau
            )
            fold_metrics[(config, cut, fold_comp)].extend(pairs)


def _aggregate_folds(
    args: argparse.Namespace,
    heldout: str,
    fold_comps: list[str],
    fold_metrics: dict[
        tuple[str, float, str], list[tuple[float, float]]
    ],
    cv_log: dict[str, Any],
) -> dict[tuple[str, float], tuple[bool, float]]:
    """Per-knob feasibility and worst fold savings across folds."""
    outcomes: dict[tuple[str, float], tuple[bool, float]] = {}
    for config in args.configs:
        for cut in args.cuts:
            feasible = True
            comp_savings: list[float] = []
            for comp in fold_comps:
                pairs = fold_metrics[(config, cut, comp)]
                savings = sum(s for s, _ in pairs) / len(pairs)
                cost = sum(c for _, c in pairs) / len(pairs)
                # Pre-registered severity headroom: the unseen
                # compressor may be harsher than the fold stand-in,
                # so folds must clear half the deployment budget.
                within = cost <= args.epsilon / 2.0
                feasible = feasible and within
                comp_savings.append(savings if within else 0.0)
                cv_log[f"{config}/cut={cut}/fold={comp}"] = {
                    "savings": savings,
                    "cost": cost,
                    "within_budget": within,
                }
                print(
                    f"  [{heldout}] {config} cut={cut} fold={comp} "
                    f"savings={savings:.4f} cost={cost:.4f}",
                    flush=True,
                )
            outcomes[(config, cut)] = (feasible, min(comp_savings))
    return outcomes


def _select_knob(
    outcomes: dict[tuple[str, float], tuple[bool, float]],
) -> tuple[str, float]:
    """Best feasible (config, cut); most conservative fallback."""
    feasible = {
        knob: savings
        for knob, (ok, savings) in outcomes.items()
        if ok
    }
    if not feasible:
        return min(outcomes, key=lambda knob: (knob[1], knob[0]))
    return min(
        feasible,
        key=lambda knob: (-feasible[knob], knob[1], knob[0]),
    )


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
    model: Any,
    rows: list[dict[str, Any]],
    features: list[str],
    cut: float,
    magnitude_model: Any | None = None,
) -> list[dict[str, Any]]:
    scores = score_rows(
        model,
        rows,
        features,
        cut=cut,
        magnitude_model=magnitude_model,
    )
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


if __name__ == "__main__":
    main()
