"""Continuous sequence-level severity proxies for Phase 1.

Phase 1's `correct` field is a presence check on ifeval and longbench
(see src/herald/tasks.py). For those tasks, the binary outcome-harm
label is degenerate (compressed_correct is True everywhere). The
research plan (gold/research-plan.md, Phase 2 targets) anticipates this
case and instructs us to use continuous sequence/trajectory severity
in place of a binary outcome label when correctness is saturated.

This script computes three continuous severity proxies per compressed
run, paired against the matched-prompt uncompressed baseline:

  - rouge_l_drop  = 1.0 - ROUGE-L(baseline, compressed)
                    where ROUGE-L is the standard LCS-based F1 over
                    whitespace tokens.
  - char_edit_ratio = Levenshtein-distance(baseline, compressed) /
                      max(len(baseline), len(compressed), 1).
  - length_diff_ratio = abs(len(compressed) - len(baseline)) /
                        max(len(baseline), len(compressed), 1).

All three are in [0, 1]. Higher = more sequence-level damage. They are
implemented in pure Python so they don't require the [metrics] extra,
which can't be installed on the Orion sandbox without a proxy.

Output:
  - <root>/metrics/severity_phase1.parquet — one row per compressed
    run, columns: run_id, rouge_l_drop, char_edit_ratio,
    length_diff_ratio.
"""

import argparse
import sys
from pathlib import Path

import polars as pl


def _lcs_len(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence of `a` and `b`."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def _rouge_l_f1(reference: str, candidate: str) -> float:
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = _lcs_len(ref_tokens, cand_tokens)
    if lcs == 0:
        return 0.0
    p = lcs / len(cand_tokens)
    r = lcs / len(ref_tokens)
    return 2 * p * r / (p + r)


def _edit_distance(a: str, b: str) -> int:
    """Iterative Levenshtein distance, O(n*m) time, O(min(n,m)) space.

    Truncates input pairs longer than 16 384 chars on each side to keep
    the long-context cells (LongBench) tractable. The truncation is
    declared in the source; if an analysis depends on full edit
    distance for very long strings, recompute with rouge-score's
    optimized implementation.
    """
    MAX = 16_384
    if len(a) > MAX:
        a = a[:MAX]
    if len(b) > MAX:
        b = b[:MAX]
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    args = ap.parse_args()

    runs_path = args.root / "final" / "runs.parquet"
    if not runs_path.exists():
        sys.exit(f"missing {runs_path}; run finalize first.")
    runs = pl.read_parquet(runs_path).select(
        ["run_id", "task", "prompt_id", "press", "generated_text"]
    )
    base = (
        runs.filter(pl.col("press") == "none")
        .select(["task", "prompt_id", "generated_text"])
        .rename({"generated_text": "baseline_text"})
    )
    comp = (
        runs.filter(pl.col("press") != "none")
        .join(base, on=["task", "prompt_id"], how="inner")
        .select(
            ["run_id", "task", "press", "generated_text", "baseline_text"]
        )
    )
    print(
        f"computing severity over {comp.height} compressed runs",
        flush=True,
    )

    rows: list[dict[str, float | str]] = []
    last_progress = 0
    for i, r in enumerate(comp.iter_rows(named=True)):
        ref = r["baseline_text"] or ""
        cand = r["generated_text"] or ""
        denom = max(len(ref), len(cand), 1)
        rouge = _rouge_l_f1(ref, cand)
        ed = _edit_distance(ref, cand)
        rows.append(
            {
                "run_id": r["run_id"],
                "rouge_l_drop": 1.0 - rouge,
                "char_edit_ratio": min(1.0, ed / denom),
                "length_diff_ratio": min(
                    1.0, abs(len(cand) - len(ref)) / denom
                ),
            }
        )
        if (i + 1) % 1000 == 0 and (i + 1) != last_progress:
            last_progress = i + 1
            print(f"  ... {i + 1}/{comp.height}", flush=True)

    out_dir = args.root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(out_dir / "severity_phase1.parquet")
    print(
        f"wrote {len(rows)} severity rows -> "
        f"{out_dir / 'severity_phase1.parquet'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
