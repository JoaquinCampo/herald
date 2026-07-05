"""Machine-checkable stop condition for the controller mission.

Reads a candidate controller summary JSON and the locked reference
file, verifies every success condition of
``docs/implementation/controller_metrics.md`` plus the mission rung,
prints one line per check, then PASS or FAIL. Exit code 0 on PASS.

Candidate summary contract (one policy, lock-shaped):

    {
      "config": {"seed": 0, "compressors": [...]},
      "epsilon": 0.01,
      "dataset_fingerprint": "...",
      "heldout": {
        "<compressor>": {
          "mean_savings": float,
          "mean_cost": float,
          "budget_respected": bool,
          "catastrophe_recall_at_10_fpr": float,
          "bootstrap_ci": {"savings_low": float, ...},
          "savings_delta_ci_low_vs_reference": float (optional)
        }
      }
    }

This file is part of the locked mission surface: do not change it to
make a candidate pass.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REFERENCE_POLICIES = ("locked_mean_baseline", "median_mean_mix")
RECALL_SLACK = 0.05


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path("docs/implementation/controller_metric_lock.json"),
    )
    parser.add_argument("--rung", type=float, default=0.10)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text())
    lock = json.loads(args.lock.read_text())
    checks: list[tuple[str, bool, str]] = []

    epsilon = float(lock["epsilon"])
    checks.append(
        (
            "epsilon matches lock",
            float(candidate.get("epsilon", -1.0)) == epsilon,
            f"lock={epsilon}",
        )
    )
    checks.append(
        (
            "seed matches lock",
            candidate.get("config", {}).get("seed") == lock["config"]["seed"],
            f"lock={lock['config']['seed']}",
        )
    )
    lock_comps = sorted(lock["config"]["compressors"])
    cand_comps = sorted(candidate.get("config", {}).get("compressors", []))
    checks.append(
        (
            "compressors match lock",
            cand_comps == lock_comps,
            f"lock={lock_comps}",
        )
    )
    checks.append(
        (
            "dataset fingerprint matches lock",
            candidate.get("dataset_fingerprint")
            == lock["dataset_fingerprint"],
            "primary rows must be unchanged",
        )
    )

    heldout = candidate.get("heldout", {})
    missing = [c for c in lock_comps if c not in heldout]
    checks.append(
        (
            "all held-out compressors reported",
            not missing,
            f"missing={missing}",
        )
    )

    if not missing:
        savings_values: list[float] = []
        beats_reference: list[bool] = []
        delta_ci_positive = 0
        recall_at_least_best = 0
        recall_never_far_below = True
        for comp in lock_comps:
            entry = heldout[comp]
            ref_entry = lock["heldout"][comp]
            budget_ok = (
                bool(entry.get("budget_respected"))
                and float(entry["mean_cost"]) <= epsilon
            )
            checks.append(
                (
                    f"budget respected on {comp}",
                    budget_ok,
                    f"cost={float(entry['mean_cost']):.4f} <= {epsilon}",
                )
            )
            savings = float(entry["mean_savings"])
            savings_values.append(savings if budget_ok else 0.0)
            ref_savings = _best_feasible_reference_savings(ref_entry, epsilon)
            beats_reference.append(savings > ref_savings)
            checks.append(
                (
                    f"savings beat feasible reference on {comp}",
                    savings > ref_savings,
                    f"{savings:.4f} > {ref_savings:.4f}",
                )
            )
            delta_low = entry.get("savings_delta_ci_low_vs_reference")
            if delta_low is None:
                delta_low = (
                    float(entry["bootstrap_ci"]["savings_low"]) - ref_savings
                )
            if float(delta_low) > 0.0:
                delta_ci_positive += 1
            ref_recall = _best_reference_recall(ref_entry)
            recall = float(entry["catastrophe_recall_at_10_fpr"])
            if recall >= ref_recall:
                recall_at_least_best += 1
            if recall < ref_recall - RECALL_SLACK:
                recall_never_far_below = False

        worst_case = min(savings_values)
        checks.append(
            (
                "savings delta CI > 0 on >= 2 compressors",
                delta_ci_positive >= 2,
                f"count={delta_ci_positive}",
            )
        )
        checks.append(
            (
                "recall >= best reference on >= 2 compressors",
                recall_at_least_best >= 2,
                f"count={recall_at_least_best}",
            )
        )
        checks.append(
            (
                f"recall never below best reference - {RECALL_SLACK}",
                recall_never_far_below,
                "",
            )
        )
        checks.append(
            (
                f"worst-case savings at budget >= rung {args.rung}",
                worst_case >= args.rung,
                f"worst_case={worst_case:.4f}",
            )
        )

    passed = all(ok for _, ok, _ in checks)
    for name, ok, detail in checks:
        marker = "ok " if ok else "FAIL"
        suffix = f" ({detail})" if detail else ""
        print(f"[{marker}] {name}{suffix}")
    print("PASS" if passed else "FAIL")
    sys.exit(0 if passed else 1)


def _best_feasible_reference_savings(
    ref_entry: dict[str, Any], epsilon: float
) -> float:
    """Best savings among budget-respecting locked references.

    A reference that busts the budget contributes 0 (the never
    policy), per the 2026-07-05 clarification in the metric spec.
    """
    best = 0.0
    for policy in REFERENCE_POLICIES:
        metrics = ref_entry[policy]
        if (
            bool(metrics["budget_respected"])
            and float(metrics["mean_cost"]) <= epsilon
        ):
            best = max(best, float(metrics["mean_savings"]))
    return best


def _best_reference_recall(ref_entry: dict[str, Any]) -> float:
    """Best catastrophe recall among the locked reference policies."""
    values = []
    for policy in REFERENCE_POLICIES:
        value = ref_entry[policy].get("catastrophe_recall_at_10_fpr")
        if value is not None:
            values.append(float(value))
    return max(values) if values else 0.0


if __name__ == "__main__":
    main()
