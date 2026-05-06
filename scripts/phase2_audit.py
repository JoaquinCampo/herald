"""Phase 2 schema/row-count audit (CPU-only).

Runs once after rsyncing `results/phase1/final/` from Orion. Answers
the OPEN-1..7 schema questions in `gold/phase-2-dataset.md`. Writes
a JSON summary to `results/phase2/audit/phase2_audit.json` and prints
a Markdown block ready to paste under "## Audit results".

Does NOT modify Phase 1 artifacts. Streaming where possible.

Patched 2026-05-05: `audit_replay_stride` now reads parquets by hive
path via `_direct_run_path`. The earlier full-tree scan per sampled
run_id was ~100x slower and got killed before completing.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl


def _scan_parquet_dir(root: Path) -> pl.LazyFrame:
    """Scan a hive-partitioned parquet tree as a lazy frame."""
    return pl.scan_parquet(
        str(root / "**" / "*.parquet"),
        hive_partitioning=True,
    )


def _collect_unique(lf: pl.LazyFrame, col: str) -> set[str]:
    """Return the unique string values of `col` from a lazy frame."""
    df = lf.select(pl.col(col).unique()).collect(streaming=True)
    return set(df[col].to_list())


def audit_runs(runs_path: Path) -> dict[str, Any]:
    """Top-level row counts and replay_status distribution."""
    df = pl.read_parquet(runs_path)
    out: dict[str, Any] = {
        "n_rows": df.height,
        "n_unique_run_id": df["run_id"].n_unique(),
        "replay_status_counts": dict(
            df.group_by("replay_status")
            .agg(pl.len().alias("n"))
            .sort("n", descending=True)
            .iter_rows()
        ),
        "task_counts": dict(
            df.group_by("task")
            .agg(pl.len().alias("n"))
            .sort("task")
            .iter_rows()
        ),
        "press_counts": dict(
            df.group_by("press")
            .agg(pl.len().alias("n"))
            .sort("press")
            .iter_rows()
        ),
        "ratio_counts": dict(
            df.group_by("compression_ratio")
            .agg(pl.len().alias("n"))
            .sort("compression_ratio")
            .iter_rows()
        ),
        "n_baseline_runs": int(df.filter(pl.col("press") == "none").height),
        "n_compressed_runs": int(df.filter(pl.col("press") != "none").height),
    }
    return out


def audit_set_diffs(
    runs_path: Path,
    tokens_root: Path,
    replay_root: Path,
    run_damage_path: Path,
) -> dict[str, Any]:
    """Set-diffs: runs / tokens / replay / run_damage on run_id."""
    runs_df = pl.read_parquet(runs_path)
    runs_ok = set(
        runs_df.filter(pl.col("replay_status") == "ok")["run_id"].to_list()
    )
    runs_compressed_ok = set(
        runs_df.filter(
            (pl.col("replay_status") == "ok") & (pl.col("press") != "none")
        )["run_id"].to_list()
    )

    tokens_ids = _collect_unique(_scan_parquet_dir(tokens_root), "run_id")
    replay_ids = _collect_unique(_scan_parquet_dir(replay_root), "run_id")
    run_damage_ids = set(
        pl.read_parquet(run_damage_path, columns=["run_id"])[
            "run_id"
        ].to_list()
    )

    return {
        "n_runs_ok": len(runs_ok),
        "n_runs_compressed_ok": len(runs_compressed_ok),
        "n_tokens_unique": len(tokens_ids),
        "n_replay_unique": len(replay_ids),
        "n_run_damage": len(run_damage_ids),
        "runs_ok_minus_tokens": len(runs_ok - tokens_ids),
        "tokens_minus_runs_ok": len(tokens_ids - runs_ok),
        "runs_ok_minus_replay": len(runs_ok - replay_ids),
        "replay_minus_runs_ok": len(replay_ids - runs_ok),
        "runs_compressed_ok_minus_run_damage": len(
            runs_compressed_ok - run_damage_ids
        ),
        "run_damage_minus_runs_compressed_ok": len(
            run_damage_ids - runs_compressed_ok
        ),
        "sample_runs_ok_not_in_tokens": sorted(list(runs_ok - tokens_ids))[
            :5
        ],
        "sample_runs_ok_not_in_replay": sorted(list(runs_ok - replay_ids))[
            :5
        ],
    }


def _ratio_dir(ratio: float) -> str:
    """Hive-partition directory name for a compression ratio.

    Phase 1 uses 4-decimal rounding: 0.96875 → ratio=0.9688.
    """
    return f"ratio={ratio:.4f}"


def _direct_run_path(
    root: Path, press: str, ratio: float, run_id: str
) -> Path:
    return root / f"press={press}" / _ratio_dir(ratio) / f"{run_id}.parquet"


def audit_replay_stride(
    tokens_root: Path,
    replay_root: Path,
    runs_path: Path,
    sample_per_cell: int = 3,
) -> dict[str, Any]:
    """Replay stride distribution and token_pos alignment.

    For each (task, press, ratio) cell, sample `sample_per_cell`
    runs and:
      - count tokens.token_pos and replay.token_pos
      - measure the stride implied by replay.token_pos differences
      - check replay.token_pos ⊆ tokens.token_pos
      - record min/max replay.token_pos

    Reads parquet files directly by hive path rather than scanning
    the whole tree per lookup — the tree-scan version is ~100x slower.
    """
    runs = pl.read_parquet(
        runs_path,
        columns=[
            "run_id",
            "task",
            "press",
            "compression_ratio",
            "replay_status",
            "num_tokens_generated",
        ],
    )
    runs = runs.filter(pl.col("replay_status") == "ok")

    cells: dict[tuple[str, str, float], list[str]] = {}
    for row in runs.iter_rows(named=True):
        key = (row["task"], row["press"], round(row["compression_ratio"], 4))
        cells.setdefault(key, []).append(row["run_id"])

    cell_summaries: list[dict[str, Any]] = []
    stride_global: Counter[int] = Counter()
    n_misaligned_runs = 0
    n_runs_inspected = 0
    first_replay_pos: Counter[int] = Counter()
    last_replay_offset: Counter[int] = Counter()
    missing_files: list[str] = []

    for (task, press, ratio), run_ids in sorted(cells.items()):
        sample = run_ids[:sample_per_cell]
        cell_strides: list[int] = []
        cell_n_replay: list[int] = []
        cell_n_tokens: list[int] = []
        cell_misaligned = 0
        cell_first: list[int] = []
        cell_last_off: list[int] = []

        for rid in sample:
            tok_path = _direct_run_path(tokens_root, press, ratio, rid)
            rep_path = _direct_run_path(replay_root, press, ratio, rid)
            if not tok_path.exists():
                missing_files.append(str(tok_path))
                continue
            if not rep_path.exists():
                missing_files.append(str(rep_path))
                continue
            tp = pl.read_parquet(tok_path, columns=["token_pos"])[
                "token_pos"
            ].to_list()
            rp = pl.read_parquet(rep_path, columns=["token_pos"])[
                "token_pos"
            ].to_list()
            if not tp or not rp:
                continue
            tp_set = set(tp)
            rp_set = set(rp)
            n_runs_inspected += 1
            if not rp_set.issubset(tp_set):
                cell_misaligned += 1
                n_misaligned_runs += 1

            cell_n_tokens.append(len(tp))
            cell_n_replay.append(len(rp))

            rp_sorted = sorted(rp)
            cell_first.append(rp_sorted[0])
            first_replay_pos[rp_sorted[0]] += 1
            cell_last_off.append(max(tp) - rp_sorted[-1])
            last_replay_offset[max(tp) - rp_sorted[-1]] += 1

            if len(rp_sorted) >= 2:
                diffs = [
                    rp_sorted[i + 1] - rp_sorted[i]
                    for i in range(len(rp_sorted) - 1)
                ]
                if diffs:
                    mode = Counter(diffs).most_common(1)[0][0]
                    cell_strides.append(mode)
                    stride_global[mode] += 1

        if not sample:
            continue

        cell_summaries.append(
            {
                "task": task,
                "press": press,
                "compression_ratio": ratio,
                "n_runs_in_cell": len(run_ids),
                "n_runs_inspected": len(sample),
                "median_n_tokens": (
                    sorted(cell_n_tokens)[len(cell_n_tokens) // 2]
                    if cell_n_tokens
                    else None
                ),
                "median_n_replay": (
                    sorted(cell_n_replay)[len(cell_n_replay) // 2]
                    if cell_n_replay
                    else None
                ),
                "stride_modes": cell_strides,
                "first_replay_positions": cell_first,
                "last_replay_offsets": cell_last_off,
                "n_misaligned_runs": cell_misaligned,
            }
        )

    return {
        "n_cells": len(cell_summaries),
        "n_runs_inspected": n_runs_inspected,
        "n_misaligned_runs": n_misaligned_runs,
        "n_missing_files": len(missing_files),
        "missing_files_sample": missing_files[:10],
        "global_stride_distribution": dict(
            sorted(stride_global.items(), key=lambda kv: -kv[1])
        ),
        "first_replay_position_distribution": dict(first_replay_pos),
        "last_replay_offset_distribution": dict(last_replay_offset),
        "per_cell": cell_summaries,
    }


def audit_baseline_replay(
    replay_root: Path,
    sample_n: int = 5,
) -> dict[str, Any]:
    """Are baseline (press=none) runs replayed?"""
    none_dir = replay_root / "press=none"
    if not none_dir.exists():
        return {
            "press_none_replay_dir_exists": False,
            "n_replay_runs_press_none": 0,
        }

    n_files = 0
    samples: list[dict[str, Any]] = []
    for path in none_dir.rglob("*.parquet"):
        if n_files < sample_n:
            df = pl.read_parquet(path)
            samples.append(
                {
                    "path": str(path.relative_to(replay_root)),
                    "n_rows": df.height,
                    "run_id": (
                        df["run_id"][0]
                        if df.height and "run_id" in df.columns
                        else None
                    ),
                }
            )
        n_files += 1

    return {
        "press_none_replay_dir_exists": True,
        "n_replay_files_press_none": n_files,
        "samples": samples,
    }


def audit_segments(metrics_dir: Path, final_dir: Path) -> dict[str, Any]:
    """Segment-metrics presence."""
    candidates = [
        metrics_dir / "segments.parquet",
        metrics_dir / "segment_metrics.parquet",
        final_dir / "segments",
    ]
    found: list[dict[str, Any]] = []
    for c in candidates:
        if c.exists():
            if c.is_file():
                df = pl.read_parquet(c)
                found.append(
                    {
                        "path": str(c),
                        "kind": "file",
                        "n_rows": df.height,
                        "columns": df.columns,
                    }
                )
            else:
                children = sorted([p.name for p in c.iterdir()])[:10]
                found.append(
                    {"path": str(c), "kind": "dir", "children": children}
                )
    return {
        "segment_artifacts_found": found,
        "available": bool(found),
    }


def audit_nll_sign(run_damage_path: Path) -> dict[str, Any]:
    """Confirm nll_ratio sign convention vs sum_kl, sum_js."""
    df = pl.read_parquet(
        run_damage_path,
        columns=["task", "sum_kl", "sum_js", "nll_ratio", "gross_harm_final"],
    )
    rows: list[dict[str, Any]] = []
    for task, sub in df.group_by("task"):
        sub_clean = sub.drop_nulls(subset=["sum_kl", "sum_js", "nll_ratio"])
        if sub_clean.height < 10:
            continue
        rho_kl_js = float(
            sub_clean.select(
                pl.corr("sum_kl", "sum_js", method="spearman")
            ).item()
            or 0.0
        )
        rho_kl_nll = float(
            sub_clean.select(
                pl.corr("sum_kl", "nll_ratio", method="spearman")
            ).item()
            or 0.0
        )
        rho_js_nll = float(
            sub_clean.select(
                pl.corr("sum_js", "nll_ratio", method="spearman")
            ).item()
            or 0.0
        )
        rows.append(
            {
                "task": task[0] if isinstance(task, tuple) else task,
                "rho_sum_kl_sum_js": round(rho_kl_js, 4),
                "rho_sum_kl_nll_ratio": round(rho_kl_nll, 4),
                "rho_sum_js_nll_ratio": round(rho_js_nll, 4),
                "n": sub_clean.height,
            }
        )
    return {
        "per_task": rows,
        "interpretation": (
            "If rho_sum_kl_nll_ratio is consistently negative, "
            "flip the sign of nll_ratio for downstream use."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("results/phase1"),
        help="Phase 1 root containing final/ and metrics/.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/audit"),
    )
    ap.add_argument(
        "--sample-per-cell",
        type=int,
        default=3,
        help="Per (task,press,ratio) cell, runs to inspect for stride.",
    )
    args = ap.parse_args()

    root: Path = args.root
    final_dir = root / "final"
    metrics_dir = root / "metrics"
    runs_path = final_dir / "runs.parquet"
    tokens_root = final_dir / "tokens"
    replay_root = final_dir / "replay"
    run_damage_path = metrics_dir / "run_damage.parquet"

    blockers: list[str] = []
    for p in (runs_path, tokens_root, replay_root, run_damage_path):
        if not p.exists():
            blockers.append(str(p))
    if blockers:
        raise SystemExit(f"missing required artifacts: {blockers}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "root": str(root),
        "runs": audit_runs(runs_path),
    }
    summary["set_diffs"] = audit_set_diffs(
        runs_path, tokens_root, replay_root, run_damage_path
    )
    summary["replay_stride"] = audit_replay_stride(
        tokens_root,
        replay_root,
        runs_path,
        sample_per_cell=args.sample_per_cell,
    )
    summary["baseline_replay"] = audit_baseline_replay(replay_root)
    summary["segments"] = audit_segments(metrics_dir, final_dir)
    summary["nll_sign"] = audit_nll_sign(run_damage_path)

    out_json = args.output_dir / "phase2_audit.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))

    print("# Audit results")
    print()
    print(
        f"runs.parquet: {summary['runs']['n_rows']} rows, "
        f"{summary['runs']['n_unique_run_id']} unique run_id"
    )
    print(f"  baseline (press=none): {summary['runs']['n_baseline_runs']}")
    print(f"  compressed:            {summary['runs']['n_compressed_runs']}")
    print()
    sd = summary["set_diffs"]
    print("Set diffs (run_id):")
    print(f"  runs_ok                = {sd['n_runs_ok']}")
    print(f"  tokens_unique          = {sd['n_tokens_unique']}")
    print(f"  replay_unique          = {sd['n_replay_unique']}")
    print(f"  run_damage             = {sd['n_run_damage']}")
    print(f"  runs_ok - tokens       = {sd['runs_ok_minus_tokens']}")
    print(f"  tokens - runs_ok       = {sd['tokens_minus_runs_ok']}")
    print(f"  runs_ok - replay       = {sd['runs_ok_minus_replay']}")
    print(f"  replay - runs_ok       = {sd['replay_minus_runs_ok']}")
    print(
        f"  compressed_ok - run_damage = "
        f"{sd['runs_compressed_ok_minus_run_damage']}"
    )
    print(
        f"  run_damage - compressed_ok = "
        f"{sd['run_damage_minus_runs_compressed_ok']}"
    )
    print()
    rs = summary["replay_stride"]
    print(
        f"Replay stride: {rs['n_runs_inspected']} runs inspected, "
        f"{rs['n_misaligned_runs']} misaligned (replay not subset of tokens)"
    )
    print(f"  global stride distribution: {rs['global_stride_distribution']}")
    print(f"  first replay pos: {rs['first_replay_position_distribution']}")
    print(
        f"  last replay offset (max_token_pos - last_replay_pos): "
        f"{rs['last_replay_offset_distribution']}"
    )
    print()
    br = summary["baseline_replay"]
    print(
        f"Baseline (press=none) replay dir exists: "
        f"{br['press_none_replay_dir_exists']}, "
        f"n_files={br.get('n_replay_files_press_none', 0)}"
    )
    print()
    seg = summary["segments"]
    print(f"Segment metrics available: {seg['available']}")
    if seg["segment_artifacts_found"]:
        for s in seg["segment_artifacts_found"]:
            print(f"  {s}")
    print()
    print("nll_ratio sign per task:")
    for r in summary["nll_sign"]["per_task"]:
        print(
            f"  {r['task']:18s} rho(kl,nll)={r['rho_sum_kl_nll_ratio']:+.3f}, "  # noqa: E501
            f"rho(js,nll)={r['rho_sum_js_nll_ratio']:+.3f}, n={r['n']}"
        )
    print()
    print(f"Full JSON: {out_json}")


if __name__ == "__main__":
    main()
