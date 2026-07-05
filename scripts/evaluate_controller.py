"""Evaluate reference policies under the locked controller metrics.

Writes ``docs/implementation/controller_metric_lock.json`` (with
``--write-lock``) and a Markdown report of savings-at-budget and
catastrophe recall for the reference policies defined in
``docs/implementation/controller_metrics.md``.
"""

import argparse
import json
import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from herald.controller_metrics import (
    EPSILON,
    evaluate_controller_split,
)
from herald.switch_baselines import (
    dataset_fingerprint,
    leave_one_compressor_splits,
)
from herald.switch_predictor import (
    fit_grouped_stat,
    internal_alpha_grid,
    mix_predictions,
    predict_grouped_stat,
    select_alpha,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]
POLICIES = ["locked_mean_baseline", "median_mean_mix"]


def main() -> None:
    """Score reference policies and optionally write the metric lock."""
    args = _parse_args()
    rows = _load_rows(args.dataset, args.compressors)
    splits = leave_one_compressor_splits(
        rows, compressors=args.compressors, seed=args.seed
    )
    results: dict[str, dict[str, Any]] = {}
    for split in splits:
        heldout = split.heldout_compressor
        predictions = _policy_predictions(split.train, split.test)
        per_policy: dict[str, Any] = {}
        for policy in POLICIES:
            train_pred, test_pred = predictions[policy]
            train_rows = _with_predictions(split.train, train_pred)
            test_rows = _with_predictions(split.test, test_pred)
            per_policy[policy] = evaluate_controller_split(
                train_rows,
                test_rows,
                prediction_key="predicted_dq",
                epsilon=args.epsilon,
                seed=args.seed,
                bootstrap_resamples=args.bootstrap_resamples,
            )
        results[heldout] = per_policy

    lock = _build_lock(args, rows, results)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "controller_reference_summary.json").write_text(
        json.dumps(lock, indent=2)
    )
    report = _markdown_report(lock)
    (out_dir / "controller_reference_report.md").write_text(report)
    if args.write_lock:
        args.lock_path.write_text(json.dumps(lock, indent=2))
        print(f"lock -> {args.lock_path}")
    print(report)


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
        default=Path("results/predictor/controller"),
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=Path(
            "docs/implementation/controller_metric_lock.json"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EPSILON)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    parser.add_argument(
        "--compressor",
        action="append",
        dest="compressors",
        default=None,
    )
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    if not args.compressors:
        args.compressors = list(PRIMARY_COMPRESSORS)
    return args


def _load_rows(path: Path, compressors: list[str]) -> list[dict[str, Any]]:
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


def _policy_predictions(
    train: list[dict[str, Any]], test: list[dict[str, Any]]
) -> dict[str, tuple[list[float], list[float]]]:
    """Predictions on train and test rows for each locked policy."""
    mean_model = fit_grouped_stat(train, stat="mean")
    baseline_train = [
        predict_grouped_stat(mean_model, row) for row in train
    ]
    baseline_test = [
        predict_grouped_stat(mean_model, row) for row in test
    ]
    grid = internal_alpha_grid(train)
    alpha = select_alpha(grid)
    mix_train = mix_predictions(train, train, alpha=alpha)
    mix_test = mix_predictions(train, test, alpha=alpha)
    return {
        "locked_mean_baseline": (baseline_train, baseline_test),
        "median_mean_mix": (mix_train, mix_test),
    }


def _with_predictions(
    rows: list[dict[str, Any]], predictions: list[float]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, value in zip(rows, predictions, strict=True):
        copied = dict(row)
        copied["predicted_dq"] = float(value)
        out.append(copied)
    return out


def _build_lock(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    heldouts: dict[str, Any] = {}
    for heldout, per_policy in sorted(results.items()):
        entry: dict[str, Any] = {}
        for policy, metrics in per_policy.items():
            entry[policy] = {
                "tau": metrics["tau"],
                "mean_savings": metrics["mean_savings"],
                "mean_cost": metrics["mean_cost"],
                "budget_respected": metrics["budget_respected"],
                "switch_rate": metrics["switch_rate"],
                "catastrophe_recall_at_10_fpr": metrics[
                    "catastrophe_recall_at_10_fpr"
                ],
                "bootstrap_ci": metrics["bootstrap_ci"],
            }
        first = next(iter(per_policy.values()))
        entry["oracle"] = first["oracle"]
        entry["always_s0"] = first["always_s0"]
        entry["never"] = first["never"]
        entry["n_test_groups"] = first["n_test_groups"]
        heldouts[heldout] = entry
    return {
        "version": 1,
        "metric": "savings_at_quality_budget",
        "epsilon": args.epsilon,
        "spec": "docs/implementation/controller_metrics.md",
        "config": {
            "seed": args.seed,
            "test_group_fraction": 0.25,
            "compressors": args.compressors,
            "bootstrap_resamples": args.bootstrap_resamples,
        },
        "dataset_fingerprint": dataset_fingerprint(rows),
        "n_rows": len(rows),
        "heldout": heldouts,
    }


def _markdown_report(lock: dict[str, Any]) -> str:
    lines = [
        "# Controller reference policies (locked metrics)",
        "",
        f"Spec: `{lock['spec']}`. Epsilon: {lock['epsilon']}.",
        f"Rows: {lock['n_rows']}.",
        "",
        "| Held-out | Policy | Tau | Savings | Cost | Budget | "
        "Switch rate | Catastrophe recall |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for heldout, entry in lock["heldout"].items():
        for policy in POLICIES:
            m = entry[policy]
            lines.append(
                f"| {heldout} | {policy} | {m['tau']:.4f} | "
                f"{m['mean_savings']:.4f} | {m['mean_cost']:.4f} | "
                f"{'yes' if m['budget_respected'] else 'NO'} | "
                f"{m['switch_rate']:.3f} | "
                f"{_fmt(m['catastrophe_recall_at_10_fpr'])} |"
            )
        oracle = entry["oracle"]
        always = entry["always_s0"]
        lines.append(
            f"| {heldout} | oracle | | "
            f"{oracle['mean_savings']:.4f} | "
            f"{oracle['mean_cost']:.4f} | yes | | |"
        )
        lines.append(
            f"| {heldout} | always_s0 | | "
            f"{always['mean_savings']:.4f} | "
            f"{always['mean_cost']:.4f} | "
            f"{'yes' if always['mean_cost'] <= lock['epsilon'] else 'NO'}"
            " | | |"
        )
    return "\n".join(lines) + "\n"


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


if __name__ == "__main__":
    main()
