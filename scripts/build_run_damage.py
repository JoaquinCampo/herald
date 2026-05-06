"""Build the canonical Phase 1 run_damage.parquet.

Reads finalized Phase 1 artifacts under ``--root`` and writes
``<root>/metrics/run_damage.parquet`` with one row per compressed run,
joined to its uncompressed baseline.

The script reloads IFEval and LongBench/qasper deterministically (same
``seed`` and ``num_prompts`` as the sweep, recovered from
``phase1_sweep_summary.json``) so the post-hoc scorer can use the
original ``instruction_id_list`` / ``kwargs`` and full Qasper gold-
answer lists. GSM8K and HumanEval keep their sweep-time correctness.

No GPU launch, no LLM judge, no generation rerun.
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import polars as pl

from herald.metrics.run_damage import build
from herald.tasks import LONGBENCH_SUBTASK


def _read_sweep_summary(root: Path) -> tuple[int, int]:
    """Return (num_prompts, seed) for the sweep that produced root."""
    summary_path = root / "phase1_sweep_summary.json"
    if not summary_path.exists():
        return (
            200,
            42,
        )  # locked Phase 1 defaults; matches gold/phase-1-results.md
    summary = json.loads(summary_path.read_text())
    return int(summary.get("num_prompts", 200)), int(summary.get("seed", 42))


def _load_ifeval_meta(
    num_prompts: int, seed: int
) -> dict[str, dict[str, Any]]:
    """Return prompt_id → {instruction_id_list, kwargs_list}.

    We load **every** row from the dataset and key by the same
    `ifeval_{key}` prompt_id format ``IFEvalTask.load`` uses; the
    sweep's shuffle never has to be reproduced because lookup is by
    prompt_id, not by index. ``num_prompts`` and ``seed`` are accepted
    for API parity with the sweep but unused here.
    """
    del num_prompts, seed  # unused; lookup is by prompt_id
    from datasets import load_dataset

    ds = load_dataset("google/IFEval", split="train")
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(ds):
        raw_key = row.get("key", i)
        try:
            key = int(raw_key)
        except (TypeError, ValueError):
            key = i
        prompt_id = f"ifeval_{key}"
        out[prompt_id] = {
            "instruction_id_list": list(row.get("instruction_id_list") or []),
            "kwargs_list": list(row.get("kwargs") or []),
        }
    return out


def _load_qasper_golds(num_prompts: int, seed: int) -> dict[str, list[str]]:
    """Return prompt_id → list[gold_answer_str] for LongBench/qasper.

    LongBench's HF script-based loader is broken under modern
    ``datasets``; we extract ``data/qasper.jsonl`` from the published
    ``data.zip`` archive on the Hub. ``_id`` is the stable identifier
    the sweep keyed on (``LongBenchSingleTask.load`` builds
    ``f"longbench_{row['_id']}"``), so we load every row and let the
    builder look up by prompt_id. ``num_prompts`` and ``seed`` are
    accepted for API parity but unused.
    """
    del num_prompts, seed  # unused; lookup is by prompt_id
    from huggingface_hub import hf_hub_download

    archive = hf_hub_download(
        repo_id="THUDM/LongBench", filename="data.zip", repo_type="dataset"
    )
    rows: list[dict[str, Any]] = []
    target = f"data/{LONGBENCH_SUBTASK}.jsonl"
    with zipfile.ZipFile(archive) as z:
        with z.open(target) as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))

    out: dict[str, list[str]] = {}
    for row in rows:
        prompt_id = f"longbench_{row['_id']}"
        out[prompt_id] = list(row.get("answers") or [])
    return out


def _load_qasper_golds_from_runs(runs_path: Path) -> dict[str, list[str]]:
    """Fallback: take the single-gold ground_truth saved per run.

    Used when the dataset cannot be reloaded (e.g. offline). Yields a
    list-of-1 per prompt_id, which still works for Qasper EM/F1 (max
    over the list) but is strictly weaker than reloading the full gold
    set.
    """
    runs = pl.read_parquet(runs_path).select(
        ["prompt_id", "task", "ground_truth"]
    )
    sub = (
        runs.filter(pl.col("task") == "longbench_single")
        .unique(subset=["prompt_id"])
        .filter(pl.col("ground_truth").is_not_null())
        .filter(pl.col("ground_truth") != "")
    )
    return {
        r["prompt_id"]: [r["ground_truth"]] for r in sub.iter_rows(named=True)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--qasper-threshold", type=float, default=0.5)
    ap.add_argument("--ifeval-threshold", type=float, default=1.0)
    ap.add_argument(
        "--no-reload-datasets",
        action="store_true",
        help=(
            "Skip reloading IFEval/LongBench from HF and use only the "
            "metadata stored in runs.parquet. IFEval becomes 'unavailable' "
            "everywhere (no instruction_id_list saved); Qasper falls back "
            "to the single-gold ground_truth field. Useful for quick "
            "sanity checks when offline."
        ),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Override output path. Defaults to "
            "<root>/metrics/run_damage.parquet."
        ),
    )
    args = ap.parse_args()

    runs_path = args.root / "final" / "runs.parquet"
    traj_path = args.root / "metrics" / "trajectory_metrics.parquet"
    sev_path = args.root / "metrics" / "severity_phase1.parquet"
    tags_path = args.root / "metrics" / "tags.parquet"
    if not runs_path.exists():
        sys.exit(f"missing {runs_path}; sync from Orion first.")
    if not traj_path.exists():
        sys.exit(
            f"missing {traj_path}; run scripts/build_phase1_alignment.py."
        )

    num_prompts, seed = _read_sweep_summary(args.root)
    print(f"Sweep: num_prompts={num_prompts}, seed={seed}", flush=True)

    if args.no_reload_datasets:
        print(
            "skipping dataset reload; IFEval rescore will be unavailable for "
            "every row (no saved instruction_id_list)."
        )
        ifeval_meta: dict[str, dict[str, Any]] = {}
        qasper_golds = _load_qasper_golds_from_runs(runs_path)
    else:
        print("loading IFEval metadata...", flush=True)
        try:
            ifeval_meta = _load_ifeval_meta(num_prompts, seed)
            print(f"  {len(ifeval_meta)} IFEval prompts loaded.")
        except Exception as exc:  # noqa: BLE001
            print(f"  IFEval reload failed: {exc!r}; falling back to empty.")
            ifeval_meta = {}
        print("loading Qasper gold answers...", flush=True)
        try:
            qasper_golds = _load_qasper_golds(num_prompts, seed)
            print(f"  {len(qasper_golds)} Qasper prompts loaded.")
        except Exception as exc:  # noqa: BLE001
            print(
                f"  Qasper reload failed: {exc!r}; "
                f"falling back to runs.parquet golds."
            )
            qasper_golds = _load_qasper_golds_from_runs(runs_path)

    out_path = args.out or args.root / "metrics" / "run_damage.parquet"
    summary = build(
        runs_path=runs_path,
        trajectory_path=traj_path,
        severity_path=sev_path if sev_path.exists() else None,
        tags_path=tags_path if tags_path.exists() else None,
        out_path=out_path,
        qasper_threshold=args.qasper_threshold,
        ifeval_threshold=args.ifeval_threshold,
        qasper_golds=qasper_golds,
        ifeval_meta=ifeval_meta,
    )
    print()
    print(f"wrote {summary['n_rows']} rows -> {out_path}")
    print()
    print("per-task label sources and counts:")
    for r in summary["per_task"]:
        print(
            f"  {r['task']:<18} source={r['label_source']:<20} "
            f"n={r['n_total']:>5} pos_harm={r['n_pos_gross_harm']:>5} "
            f"neg={r['n_neg_gross_harm']:>5} undef={r['n_undefined']:>5} "
            f"non_degen={r['non_degenerate']}"
        )
        if len(r["label_source_counts"]) > 1:
            print(f"      label sources: {r['label_source_counts']}")
    if summary["unsupported_ifeval_types"]:
        print()
        print("unsupported IFEval instruction types (rows touched):")
        for t, n in summary["unsupported_ifeval_types"].items():
            print(f"  {t}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
