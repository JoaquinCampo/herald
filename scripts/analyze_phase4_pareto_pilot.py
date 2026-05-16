"""Phase 4 Pareto pilot analysis (Mac-side).

Reads `controller_runs.parquet` and `controller_segments.parquet`
emitted by `scripts/run_phase4_pareto_pilot.py` (Orion), computes
ROUGE-L drops paired against the no_compression text per prompt,
and writes `pareto_summary.json`.

Outputs a per-policy table and per-prompt paired deltas:
  - HERALD vs random_matched
  - HERALD vs fixed_64
  - HERALD vs no_compression

ROUGE-L is computed locally because the rouge_score dependency is
in the [metrics] extra and is not installed on Orion.
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import polars as pl
from rouge_score import rouge_scorer

DEFAULT_INPUT = Path("results/phase4/pareto_pilot")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Pilot artifact dir (controller_runs.parquet, ...).",
    )
    ap.add_argument(
        "--output-summary",
        type=Path,
        default=None,
        help=(
            "Path to pareto_summary.json "
            "(default: input/pareto_summary.json)."
        ),
    )
    return ap.parse_args(argv)


def _rouge_l(scorer: Any, ref: str, hyp: str) -> float:
    if not isinstance(ref, str) or not isinstance(hyp, str):
        return float("nan")
    if not ref or not hyp:
        return 0.0
    try:
        return float(scorer.score(ref, hyp)["rougeL"].fmeasure)
    except Exception:  # noqa: BLE001
        return float("nan")


def _compute_rouge(runs: pl.DataFrame, ref_policy: str) -> pl.DataFrame:
    """Pair each non-ref run with its prompt's ref text, score ROUGE-L."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    refs = runs.filter(pl.col("policy") == ref_policy).select(
        ["prompt_id", "generated_text"]
    )
    refs = refs.rename({"generated_text": "_ref_text"})
    joined = runs.join(refs, on="prompt_id", how="left")
    rouge_l = []
    rouge_l_drop = []
    for row in joined.iter_rows(named=True):
        if row["policy"] == ref_policy:
            rouge_l.append(1.0)
            rouge_l_drop.append(0.0)
            continue
        score = _rouge_l(
            scorer,
            row["_ref_text"] or "",
            row["generated_text"] or "",
        )
        rouge_l.append(score)
        rouge_l_drop.append(
            1.0 - score if not math.isnan(score) else float("nan")
        )
    return joined.with_columns(
        [
            pl.Series("rouge_l_vs_ref", rouge_l),
            pl.Series("rouge_l_drop_vs_ref", rouge_l_drop),
        ]
    ).drop("_ref_text")


def _per_policy(joined: pl.DataFrame) -> list[dict[str, Any]]:
    """Aggregate per-policy means / counts from the joined table."""
    out: list[dict[str, Any]] = []
    for policy, sub in joined.group_by("policy"):
        sub_df = sub
        n = sub_df.height
        n_correct_field = sub_df["correct"].cast(pl.Boolean, strict=False)
        n_correct = int(n_correct_field.fill_null(False).sum())
        n_unscored = int(sub_df["correct"].is_null().sum())
        n_truncated = int(
            sub_df.filter(pl.col("stop_reason") == "max_tokens").height
        )
        accuracy = (
            n_correct / max(1, n - n_unscored) if n > n_unscored else None
        )
        truncation_rate = n_truncated / max(1, n)
        rl = sub_df["rouge_l_vs_ref"].drop_nulls().to_list()
        rl_drop = sub_df["rouge_l_drop_vs_ref"].drop_nulls().to_list()
        out.append(
            {
                "policy": (
                    policy[0] if isinstance(policy, tuple) else policy
                ),
                "n": n,
                "n_correct": n_correct,
                "n_unscored": n_unscored,
                "n_truncated": n_truncated,
                "accuracy": accuracy,
                "truncation_rate": truncation_rate,
                "mean_num_tokens_generated": float(
                    sub_df["num_tokens_generated"].mean() or 0.0
                ),
                "mean_wall_clock_per_token": float(
                    sub_df["wall_clock_per_token"].mean() or 0.0
                ),
                "mean_total_evicted_tokens": float(
                    sub_df["total_evicted_tokens"].mean() or 0.0
                ),
                "mean_decode_event_count": float(
                    sub_df["decode_event_count"].mean() or 0.0
                ),
                "mean_retained_cache_size": (
                    float(sub_df["mean_retained_cache_size"].mean() or 0.0)
                    if sub_df["mean_retained_cache_size"].drop_nulls().len()
                    else None
                ),
                "mean_n_relax": float(sub_df["n_relax"].mean() or 0.0),
                "mean_n_tighten": float(sub_df["n_tighten"].mean() or 0.0),
                "mean_n_keep": float(sub_df["n_keep"].mean() or 0.0),
                "mean_rouge_l_vs_ref": (sum(rl) / len(rl) if rl else None),
                "mean_rouge_l_drop_vs_ref": (
                    sum(rl_drop) / len(rl_drop) if rl_drop else None
                ),
            }
        )
    return sorted(out, key=lambda r: r["policy"])


def _paired_delta(
    joined: pl.DataFrame, a: str, b: str, metric: str
) -> dict[str, Any]:
    """Per-prompt paired delta a - b on `metric`."""
    sub = joined.filter(pl.col("policy").is_in([a, b]))
    pivot = sub.pivot(values=metric, index="prompt_id", on="policy")
    if a not in pivot.columns or b not in pivot.columns:
        return {
            "a": a,
            "b": b,
            "metric": metric,
            "n_prompts": 0,
            "mean_delta": None,
            "median_delta": None,
            "n_a_better": 0,
            "n_b_better": 0,
            "n_tie": 0,
        }
    pairs = pivot.drop_nulls(subset=[a, b])
    if pairs.height == 0:
        return {
            "a": a,
            "b": b,
            "metric": metric,
            "n_prompts": 0,
            "mean_delta": None,
            "median_delta": None,
            "n_a_better": 0,
            "n_b_better": 0,
            "n_tie": 0,
        }
    diffs = (pairs[a] - pairs[b]).to_list()
    diffs = [float(d) for d in diffs if d is not None and not math.isnan(d)]
    if not diffs:
        return {
            "a": a,
            "b": b,
            "metric": metric,
            "n_prompts": 0,
            "mean_delta": None,
            "median_delta": None,
            "n_a_better": 0,
            "n_b_better": 0,
            "n_tie": 0,
        }
    n_a = sum(1 for d in diffs if d > 0)
    n_b = sum(1 for d in diffs if d < 0)
    n_tie = sum(1 for d in diffs if d == 0)
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "n_prompts": len(diffs),
        "mean_delta": statistics.mean(diffs),
        "median_delta": statistics.median(diffs),
        "n_a_better": n_a,
        "n_b_better": n_b,
        "n_tie": n_tie,
    }


def _bootstrap_paired_ci(
    joined: pl.DataFrame,
    a: str,
    b: str,
    metric: str,
    n_boot: int = 1000,
    seed: int = 13,
) -> dict[str, Any]:
    """Paired bootstrap CI for mean(a - b) on `metric`."""
    import random  # noqa: PLC0415

    sub = joined.filter(pl.col("policy").is_in([a, b]))
    pivot = sub.pivot(values=metric, index="prompt_id", on="policy")
    if a not in pivot.columns or b not in pivot.columns:
        return {
            "a": a,
            "b": b,
            "metric": metric,
            "ci_lower": None,
            "ci_upper": None,
        }
    pairs = pivot.drop_nulls(subset=[a, b])
    diffs = [
        float(d)
        for d in (pairs[a] - pairs[b]).to_list()
        if d is not None and not math.isnan(d)
    ]
    if len(diffs) < 3:
        return {
            "a": a,
            "b": b,
            "metric": metric,
            "ci_lower": None,
            "ci_upper": None,
            "n_prompts": len(diffs),
        }
    rng = random.Random(seed)
    means = []
    n = len(diffs)
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {
        "a": a,
        "b": b,
        "metric": metric,
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "n_prompts": len(diffs),
        "n_boot": n_boot,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_path = args.input_dir / "controller_runs.parquet"
    if not runs_path.exists():
        print(f"runs parquet not found: {runs_path}", file=sys.stderr)
        return 2
    runs = pl.read_parquet(runs_path)
    print(f"loaded {runs.height} runs from {runs_path}")

    # Detect ref policy: prefer no_compression; else fixed_256.
    policies = set(runs["policy"].unique().to_list())
    if "no_compression" in policies:
        ref = "no_compression"
        ref_note = "no_compression run is the high-quality anchor"
    elif "fixed_256" in policies:
        ref = "fixed_256"
        ref_note = (
            "no_compression missing; used fixed_256 as fallback "
            "high-quality anchor (lower quality than true uncompressed)"
        )
    else:
        print(
            "ERROR: neither no_compression nor fixed_256 found",
            file=sys.stderr,
        )
        return 2
    print(f"reference policy for ROUGE-L: {ref}")

    joined = _compute_rouge(runs, ref_policy=ref)
    per_policy = _per_policy(joined)

    # Find the herald and random policy names dynamically.
    herald_name = next((p for p in policies if p.startswith("herald_")), None)
    random_name = next((p for p in policies if p.startswith("random_")), None)
    paired = []
    bootstrap = []
    if herald_name and random_name:
        for metric in (
            "task_score",
            "rouge_l_vs_ref",
            "wall_clock_per_token",
            "total_evicted_tokens",
        ):
            paired.append(
                _paired_delta(joined, herald_name, random_name, metric)
            )
            bootstrap.append(
                _bootstrap_paired_ci(joined, herald_name, random_name, metric)
            )
        for metric in ("task_score", "rouge_l_vs_ref"):
            paired.append(
                _paired_delta(joined, herald_name, "fixed_64", metric)
            )
            bootstrap.append(
                _bootstrap_paired_ci(joined, herald_name, "fixed_64", metric)
            )
            paired.append(_paired_delta(joined, herald_name, ref, metric))
            bootstrap.append(
                _bootstrap_paired_ci(joined, herald_name, ref, metric)
            )

    summary = {
        "input_dir": str(args.input_dir),
        "n_runs": int(runs.height),
        "n_prompts": int(runs["prompt_id"].n_unique()),
        "policies": sorted(policies),
        "reference_policy": ref,
        "reference_note": ref_note,
        "per_policy": per_policy,
        "paired_deltas": paired,
        "bootstrap_cis": bootstrap,
    }
    out_path = args.output_summary or (args.input_dir / "pareto_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out_path}")

    # Pretty print headline table.
    print("\nper-policy summary:")
    headers = (
        "policy",
        "n",
        "acc",
        "trunc%",
        "tokens",
        "wct/tok",
        "evicted",
        "ROUGE-L",
        "ROUGE-L drop",
        "relax",
    )
    fmt = (
        "{:24s} {:>3s} {:>7s} {:>6s} {:>7s} "
        "{:>9s} {:>9s} {:>9s} {:>11s} {:>5s}"
    )
    print(fmt.format(*headers))
    for p in per_policy:
        acc = f"{p['accuracy']:.3f}" if p["accuracy"] is not None else "n/a"
        rouge = (
            f"{p['mean_rouge_l_vs_ref']:.3f}"
            if p["mean_rouge_l_vs_ref"] is not None
            else "n/a"
        )
        rouge_drop = (
            f"{p['mean_rouge_l_drop_vs_ref']:.3f}"
            if p["mean_rouge_l_drop_vs_ref"] is not None
            else "n/a"
        )
        print(
            fmt.format(
                p["policy"],
                str(p["n"]),
                acc,
                f"{p['truncation_rate'] * 100:.1f}",
                f"{p['mean_num_tokens_generated']:.1f}",
                f"{p['mean_wall_clock_per_token']:.5f}",
                f"{p['mean_total_evicted_tokens']:.0f}",
                rouge,
                rouge_drop,
                f"{p['mean_n_relax']:.1f}",
            )
        )

    if paired:
        print("\npaired deltas (a - b):")
        for d in paired:
            mean = (
                f"{d['mean_delta']:+.4f}"
                if d["mean_delta"] is not None
                else "n/a"
            )
            print(
                f"  {d['a']:24s} - {d['b']:24s} "
                f"on {d['metric']:24s} "
                f"mean={mean} n={d['n_prompts']} "
                f"a>b={d['n_a_better']} b>a={d['n_b_better']} "
                f"tie={d['n_tie']}"
            )
        print("\nbootstrap 95% CIs on mean delta:")
        for c in bootstrap:
            if c["ci_lower"] is None:
                continue
            print(
                f"  {c['a']:24s} - {c['b']:24s} "
                f"on {c['metric']:24s} "
                f"95% CI [{c['ci_lower']:+.4f}, {c['ci_upper']:+.4f}] "
                f"n={c['n_prompts']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
