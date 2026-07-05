"""Evaluate the cell position-gate candidate under the locked metrics.

Per canonical leave-one-compressor-out split: fit the position gate on
the split's train compressors plus the training-only donor compressors
(random, snapkv), never on the held-out compressor. The tolerance knob
is selected by internal cross-validation that mirrors deployment: hold
one of the split's two train PRIMARY compressors out of the fit pool
(the donors, including the degradation-floor random press, always stay
in fit, exactly as they will at deployment), run the unchanged locked
tau selection on the remaining train rows, and require the epsilon
budget on the held-out fold compressor's rows. The held-out primary
compressor of the outer split never influences anything. Final numbers
come from the unchanged ``herald.controller_metrics`` evaluator on the
canonical splits, written as a lock-shaped candidate summary for
``scripts/check_mission.py``.
"""

import argparse
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
from herald.switch_gate import (
    DEFAULT_TOLERANCES,
    fit_position_gate,
    score_rows,
    select_tolerance,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
DONOR_COMPRESSORS = ["random", "snapkv"]


def main() -> None:
    """Run the gate candidate through the locked controller metrics."""
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
    splits = leave_one_compressor_splits(
        primary, compressors=PRIMARY_COMPRESSORS, seed=args.seed
    )
    donor_train = {
        donor: [
            row
            for row in rows
            if str(row["compressor"]) == donor
            and not _is_test_group(row, args.seed, 0.25)
        ]
        for donor in DONOR_COMPRESSORS
    }

    heldouts: dict[str, Any] = {}
    for split in splits:
        heldout = split.heldout_compressor
        fit_pool = list(split.train)
        for donor_rows in donor_train.values():
            fit_pool.extend(donor_rows)

        fold_comps = sorted(
            {str(row["compressor"]) for row in split.train}
        )
        outcomes: dict[float, tuple[bool, float]] = {}
        cv_log: dict[str, Any] = {}
        for tolerance in args.tolerances:
            feasible = True
            fold_savings: list[float] = []
            for fold_comp in fold_comps:
                inner_fit = [
                    row
                    for row in fit_pool
                    if str(row["compressor"]) != fold_comp
                ]
                fold_eval = [
                    row
                    for row in split.train
                    if str(row["compressor"]) == fold_comp
                ]
                fold_train = [
                    row
                    for row in split.train
                    if str(row["compressor"]) != fold_comp
                ]
                gate = fit_position_gate(
                    inner_fit, tolerance=tolerance
                )
                tau = select_tau(
                    _with_scores(gate, fold_train),
                    prediction_key="predicted_dq",
                    epsilon=args.epsilon,
                )
                savings, cost = _policy_metrics(
                    _with_scores(gate, fold_eval), tau
                )
                within = cost <= args.epsilon
                feasible = feasible and within
                fold_savings.append(savings if within else 0.0)
                cv_log[f"tol={tolerance}/fold={fold_comp}"] = {
                    "tau": tau,
                    "savings": savings,
                    "cost": cost,
                    "within_budget": within,
                }
            outcomes[tolerance] = (feasible, min(fold_savings))
        tolerance = select_tolerance(outcomes)

        gate = fit_position_gate(fit_pool, tolerance=tolerance)
        result = evaluate_controller_split(
            _with_scores(gate, split.train),
            _with_scores(gate, split.test),
            prediction_key="predicted_dq",
            epsilon=args.epsilon,
            seed=args.seed,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        result["selected_tolerance"] = tolerance
        result["internal_cv"] = cv_log
        result["thresholds"] = {
            f"{task}/{ratio}": threshold
            for (task, ratio), threshold in sorted(
                gate.thresholds.items(), key=lambda kv: str(kv[0])
            )
        }
        heldouts[heldout] = result
        oracle = result["oracle"]["mean_savings"]
        print(
            f"[{heldout}] tol={tolerance} tau={result['tau']:.4f} "
            f"savings={result['mean_savings']:.4f} "
            f"cost={result['mean_cost']:.4f} "
            f"budget={result['budget_respected']} "
            f"recall={result['catastrophe_recall_at_10_fpr']} "
            f"oracle_gap={oracle - result['mean_savings']:.4f}",
            flush=True,
        )

    summary = {
        "candidate": "cell_position_gate",
        "epsilon": args.epsilon,
        "config": {
            "seed": args.seed,
            "test_group_fraction": 0.25,
            "compressors": PRIMARY_COMPRESSORS,
            "donors": DONOR_COMPRESSORS,
            "tolerance_grid": list(args.tolerances),
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
        default=Path("results/predictor/experiments/position_gate"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument(
        "--tolerance",
        action="append",
        dest="tolerances",
        type=float,
        default=None,
    )
    args = parser.parse_args()
    if not args.tolerances:
        args.tolerances = list(DEFAULT_TOLERANCES)
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


def _with_scores(
    gate: Any, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, score in zip(rows, score_rows(gate, rows), strict=True):
        copied = dict(row)
        copied["predicted_dq"] = float(score)
        out.append(copied)
    return out


def _policy_metrics(
    rows: list[dict[str, Any]], tau: float
) -> tuple[float, float]:
    """Mean savings and cost of the earliest-below-tau policy."""
    groups = decision_groups(rows)
    choices = [
        policy_choice(group, prediction_key="predicted_dq", tau=tau)
        for group in groups.values()
    ]
    savings = float(
        sum(choice.savings for choice in choices) / len(choices)
    )
    cost = float(sum(choice.cost for choice in choices) / len(choices))
    return savings, cost


if __name__ == "__main__":
    main()
