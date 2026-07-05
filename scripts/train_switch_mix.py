"""Run the canonical median-mean mix switch-predictor experiment."""

import argparse
import json
import math
import sys
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from herald.switch_baselines import (
    dataset_fingerprint,
    duplicate_audit,
    evaluate_switch_predictions,
)
from herald.switch_predictor import (
    DEFAULT_ALPHAS,
    MixResult,
    run_median_mean_mix,
)

PRIMARY_COMPRESSORS = ["streaming_llm", "expected_attention", "knorm"]


def main() -> None:
    """Run one tracked mix experiment through the canonical evaluator."""
    args = _parse_args()
    rows = _load_rows(args.dataset, args.compressors)
    result = run_median_mean_mix(
        rows,
        compressors=args.compressors,
        seed=args.seed,
        alphas=args.alphas,
        min_lift=args.min_lift,
    )
    summary = evaluate_switch_predictions(
        result.predicted_rows,
        prediction_key="predicted_dq",
        model_name=args.name,
        compressors=args.compressors,
        seed=args.seed,
        bootstrap_resamples=args.bootstrap_resamples,
        command=sys.argv,
    )
    artifact = _build_artifact(args, rows, summary, result)
    _write_outputs(args, result, summary, artifact)


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
        default=Path("results/predictor/experiments"),
    )
    parser.add_argument("--name", default="median_mean_mix_cv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--compressor",
        action="append",
        dest="compressors",
        default=None,
    )
    parser.add_argument(
        "--alpha",
        action="append",
        dest="alphas",
        type=float,
        default=None,
    )
    parser.add_argument("--min-lift", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    args = parser.parse_args()
    if not args.compressors:
        args.compressors = list(PRIMARY_COMPRESSORS)
    if not args.alphas:
        args.alphas = list(DEFAULT_ALPHAS)
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
        usable.append(row)
    if not usable:
        raise SystemExit("no usable rows after compressor and dq filtering")
    return usable


def _build_artifact(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    result: MixResult,
) -> dict[str, Any]:
    split_metrics = []
    for split in summary["splits"]:
        metric = split["prediction"]
        split_metrics.append(
            {
                "heldout_compressor": split["heldout_compressor"],
                "alpha": result.alpha_by_heldout[split["heldout_compressor"]],
                "mae": metric["mae"],
                "relative_mae_improvement_vs_locked": metric[
                    "relative_mae_improvement_vs_locked"
                ],
                "top_decile_lift_dq_gt_0": metric["controller_metrics"][
                    "dq_gt_0"
                ]["top_decile_lift"],
                "top_decile_lift_dq_ge_0_5": metric["controller_metrics"][
                    "dq_ge_0_5"
                ]["top_decile_lift"],
                "auprc_dq_gt_0": metric["ranking_metrics"]["dq_gt_0"][
                    "auprc"
                ],
                "recall_at_10_fpr_dq_gt_0": metric["ranking_metrics"][
                    "dq_gt_0"
                ]["recall_at_10_fpr"],
                "bootstrap_ci": metric["bootstrap_ci"],
                "leakage_audit": split["leakage_audit"],
            }
        )
    rel_values = [
        item["relative_mae_improvement_vs_locked"] for item in split_metrics
    ]
    return {
        "config": {
            "name": args.name,
            "dataset": str(args.dataset),
            "out_dir": str(args.out_dir),
            "compressors": args.compressors,
            "seed": args.seed,
            "alphas": args.alphas,
            "min_lift": args.min_lift,
            "bootstrap_resamples": args.bootstrap_resamples,
            "model_type": "median_mean_mix",
        },
        "model_input_fields": list(result.model_input_fields),
        "dataset_fingerprint": dataset_fingerprint(rows),
        "n_rows": len(rows),
        "duplicate_audit": duplicate_audit(rows),
        "alpha_by_heldout": result.alpha_by_heldout,
        "internal_grids": {
            heldout: {
                str(alpha): {
                    "mean_relative_improvement": (
                        entry.mean_relative_improvement
                    ),
                    "min_top_decile_lift": entry.min_top_decile_lift,
                }
                for alpha, entry in grid.items()
            }
            for heldout, grid in result.internal_grids.items()
        },
        "split_metrics": split_metrics,
        "mean_relative_mae_improvement": sum(rel_values) / len(rel_values),
        "all_heldouts_improved": all(value > 0.0 for value in rel_values),
        "ci_excludes_zero_count": sum(
            1
            for item in split_metrics
            if item["bootstrap_ci"]["low"] is not None
            and item["bootstrap_ci"]["low"] > 0.0
        ),
        "lift_ge_2_all_heldouts": all(
            max(
                item["top_decile_lift_dq_gt_0"],
                item["top_decile_lift_dq_ge_0_5"],
            )
            >= 2.0
            for item in split_metrics
        ),
    }


def _write_outputs(
    args: argparse.Namespace,
    result: MixResult,
    summary: dict[str, Any],
    artifact: dict[str, Any],
) -> None:
    out_dir = args.out_dir / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "artifact.json").write_text(json.dumps(artifact, indent=2))
    report = _markdown_report(args, artifact)
    (out_dir / "report.md").write_text(report)
    _write_parquet(out_dir / "predictions.parquet", result.predicted_rows)
    log_path = args.out_dir / "experiment_log.md"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + report)
    print(f"experiment={args.name}")
    print(f"rows={artifact['n_rows']}")
    print(f"alpha_by_heldout={artifact['alpha_by_heldout']}")
    print(
        "mean_relative_mae_improvement="
        f"{artifact['mean_relative_mae_improvement']:.6f}"
    )
    print(f"all_heldouts_improved={artifact['all_heldouts_improved']}")
    print(f"lift_ge_2_all_heldouts={artifact['lift_ge_2_all_heldouts']}")
    print(f"ci_excludes_zero_count={artifact['ci_excludes_zero_count']}")
    print(f"summary -> {out_dir / 'summary.json'}")
    print(f"report -> {out_dir / 'report.md'}")


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    pyarrow = import_module("pyarrow")
    table = cast(Any, pyarrow.Table).from_pylist(rows)
    pq = import_module("pyarrow.parquet")
    cast(Any, pq.write_table)(table, path)


def _markdown_report(
    args: argparse.Namespace, artifact: dict[str, Any]
) -> str:
    lines = [
        f"## EXECUTE, {args.name}",
        "",
        "- Model type: `median_mean_mix` "
        "(grouped train median plus alpha times mean-median gap, "
        "alpha chosen by internal leave-one-train-compressor-out CV).",
        f"- Model input fields: {artifact['model_input_fields']}.",
        f"- Dataset fingerprint: `{artifact['dataset_fingerprint']}`.",
        f"- Rows: {artifact['n_rows']}.",
        f"- Alpha grid: {artifact['config']['alphas']}.",
        f"- Alpha by held-out: {artifact['alpha_by_heldout']}.",
        f"- Mean relative MAE improvement: "
        f"{artifact['mean_relative_mae_improvement']:.4f}.",
        f"- Improved all held-outs: {artifact['all_heldouts_improved']}.",
        f"- Lift >= 2x on all held-outs: "
        f"{artifact['lift_ge_2_all_heldouts']}.",
        f"- CIs excluding zero: {artifact['ci_excludes_zero_count']}.",
        "",
        "| Held-out | Alpha | Rel MAE | MAE | Lift dq > 0 | "
        "Lift dq >= 0.5 | AUPRC | CI low | CI high |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in artifact["split_metrics"]:
        ci = item["bootstrap_ci"]
        lines.append(
            f"| {item['heldout_compressor']} | "
            f"{item['alpha']:.1f} | "
            f"{item['relative_mae_improvement_vs_locked']:.4f} | "
            f"{item['mae']:.4f} | "
            f"{item['top_decile_lift_dq_gt_0']:.4f} | "
            f"{item['top_decile_lift_dq_ge_0_5']:.4f} | "
            f"{_fmt(item['auprc_dq_gt_0'])} | "
            f"{_fmt(ci['low'])} | {_fmt(ci['high'])} |"
        )
    lines.extend(
        [
            "",
            "### REFLECT",
            "",
            "Canonical leave-one-compressor-out evaluation with "
            "prompt-group-disjoint splits. Alpha selection is train-only "
            "via internal leave-one-train-compressor-out CV; the "
            "held-out compressor never influences selection. Inputs are "
            "task, ratio, and position bucket, the same key structure "
            "as the locked baseline; no forbidden columns are used.",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return "n/a"


if __name__ == "__main__":
    main()
