"""Summarize a sweep's results: q, damage dq distributions, and the
trivial-baseline break-it check.

The break-it check asks whether dq is trivially explained by
(compressor, ratio, switch position s) alone. It decomposes the total
variance of dq into the part captured by per-cell means (a predictor
that knows only compressor, ratio, s) and the residual across prompts
within a cell. If the cell means explain almost everything, then
per-token logit features can add little and the thesis is in trouble;
a large within-cell residual is the signal the predictor must capture.

Reads the storage layout directly. Stdlib + numpy only.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _read_references(task_dir: Path) -> dict[str, float]:
    q: dict[str, float] = {}
    ref_dir = task_dir / "references"
    if not ref_dir.is_dir():
        return q
    for jf in ref_dir.glob("*.json"):
        d = json.loads(jf.read_text())
        q[str(d["prompt_id"])] = float(d["q"])
    return q


def _read_hybrids(task_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    hyb_dir = task_dir / "hybrids"
    if not hyb_dir.is_dir():
        return rows
    for shard in hyb_dir.glob("*.jsonl"):
        comp, _, ratio_s = shard.stem.partition("__")
        ratio = float(ratio_s)
        for line in shard.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r["compressor"] = comp
            r["ratio"] = ratio
            rows.append(r)
    return rows


def _quantiles(a: np.ndarray) -> dict[str, float]:
    if a.size == 0:
        return {}
    qs = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "max": float(qs[4]),
        "frac_damage": float((a > 0).mean()),
        "frac_lift": float((a < 0).mean()),
        "n": int(a.size),
    }


def _trivial_baseline_r2(rows: list[dict[str, object]]) -> float:
    """Fraction of dq variance explained by (compressor, ratio, s) cell
    means. 1.0 means dq is fully determined by position+ratio; lower
    means there is prompt-level residual for features to predict.
    """
    if len(rows) < 2:
        return float("nan")
    dq = np.array([float(r["dq"]) for r in rows])
    total_var = float(dq.var())
    if total_var == 0.0:
        return float("nan")
    cells: dict[tuple[str, float, int], list[float]] = defaultdict(list)
    for r in rows:
        key = (str(r["compressor"]), float(r["ratio"]), int(r["s"]))
        cells[key].append(float(r["dq"]))
    resid = []
    for vals in cells.values():
        m = float(np.mean(vals))
        resid.extend(v - m for v in vals)
    resid_var = float(np.var(resid))
    return 1.0 - resid_var / total_var


def analyze(results_dir: Path) -> dict[str, object]:
    summary: dict[str, object] = {}
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        if model_dir.name == "config.json":
            continue
        for task_dir in sorted(
            p for p in model_dir.iterdir() if p.is_dir()
        ):
            key = f"{model_dir.name}/{task_dir.name}"
            qref = _read_references(task_dir)
            rows = _read_hybrids(task_dir)
            block: dict[str, object] = {
                "n_references": len(qref),
                "ref_q_mean": (
                    float(np.mean(list(qref.values()))) if qref else None
                ),
                "n_hybrids": len(rows),
            }
            # Two distinct quantities (methodology sections 2.1 vs 2.2):
            #   s0 = the reported damage measure, fully-compressed run
            #        (s=0) vs reference, one comparison per run.
            #   curve = mean over all switch positions, a summary of the
            #        per-position training target, NOT the headline.
            block["dq_s0_by_compressor_ratio"] = _group_dq(
                [r for r in rows if int(r["s"]) == 0]
            )
            block["dq_curve_mean_by_compressor_ratio"] = _group_dq(rows)
            block["trivial_baseline_r2"] = _trivial_baseline_r2(rows)
            summary[key] = block
    return summary


def _group_dq(rows: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for r in rows:
        grouped[(str(r["compressor"]), float(r["ratio"]))].append(
            float(r["dq"])
        )
    return {
        f"{comp}@{ratio}": _quantiles(np.array(vals))
        for (comp, ratio), vals in sorted(grouped.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    summary = analyze(args.results_dir)
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
