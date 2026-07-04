"""Build token-level datasets from compressed hybrid feature artifacts."""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from herald.features import derive_features
from herald.storage import hybrid_feature_path, safe_id


def build_hybrid_token_dataset(
    results_dir: Path,
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    max_hybrids_per_task: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build rows for the true compressed-stream online predictor.

    Each output row is one compressed generated token. The continuous
    target is the run-level task-quality delta ``dq``.
    """
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for model, task, task_dir in _iter_task_dirs(
        results_dir, models=models, tasks=tasks
    ):
        task_rows, summary = _build_task_rows(
            results_dir,
            model=model,
            task=task,
            task_dir=task_dir,
            max_hybrids=max_hybrids_per_task,
        )
        rows.extend(task_rows)
        summaries.append(summary)
    return rows, {
        "results_dir": str(results_dir),
        "n_rows": len(rows),
        "n_tasks": len(summaries),
        "tasks": summaries,
    }


def _build_task_rows(
    results_dir: Path,
    *,
    model: str,
    task: str,
    task_dir: Path,
    max_hybrids: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    refs = _load_reference_lengths(task_dir)
    rows: list[dict[str, Any]] = []
    n_hybrids_seen = 0
    n_hybrids_used = 0
    skipped_missing_features = 0
    skipped_empty_features = 0
    skipped_missing_reference = 0

    for rec in _iter_hybrid_records(task_dir):
        n_hybrids_seen += 1
        ref_len = refs.get(rec["prompt_id"])
        if ref_len is None or not np.isfinite(float(rec["q_ref"])):
            skipped_missing_reference += 1
            continue
        if rec["features_path"] is None:
            skipped_missing_features += 1
            continue
        fpath = hybrid_feature_path(
            results_dir,
            model,
            task,
            rec["compressor"],
            rec["ratio"],
            rec["prompt_id"],
            rec["s"],
        )
        stored_path = Path(str(rec["features_path"]))
        stored_path = (
            stored_path
            if stored_path.is_absolute()
            else results_dir / stored_path
        )
        if stored_path != fpath:
            skipped_missing_features += 1
            continue
        if not fpath.exists():
            skipped_missing_features += 1
            continue
        raw = np.load(fpath).astype(np.float32)
        if raw.shape[0] == 0:
            skipped_empty_features += 1
            continue
        if raw.shape[0] != len(rec["new_ids"]):
            raise ValueError(
                "hybrid feature/token length mismatch: "
                f"{model}/{task} {rec['compressor']}@{rec['ratio']} "
                f"prompt_id={rec['prompt_id']} s={rec['s']} "
                f"features={raw.shape[0]} new_ids={len(rec['new_ids'])}"
            )
        derived, names = derive_features(raw)
        for token_pos in range(derived.shape[0]):
            global_pos = rec["s"] + token_pos
            row: dict[str, Any] = {
                "model": model,
                "task": task,
                "prompt_id": rec["prompt_id"],
                "compressor": rec["compressor"],
                "ratio": rec["ratio"],
                "switch_s": rec["s"],
                "compressed_token_pos": token_pos,
                "global_token_pos": global_pos,
                "ref_len": ref_len,
                "relative_global_pos": (
                    float(global_pos / ref_len)
                    if ref_len is not None and ref_len > 0
                    else np.nan
                ),
                "q_ref": rec["q_ref"],
                "q_compressed": rec["q"],
                "dq": rec["dq"],
                "damage_positive": int(rec["dq"] > 0.0),
                "damage_major": int(rec["dq"] >= 0.5),
            }
            for name, value in zip(names, derived[token_pos], strict=True):
                row[f"feat__{name}"] = float(value)
            rows.append(row)
        n_hybrids_used += 1
        if max_hybrids is not None and n_hybrids_used >= max_hybrids:
            break

    return rows, {
        "model": model,
        "task": task,
        "n_hybrids_seen": n_hybrids_seen,
        "n_hybrids_used": n_hybrids_used,
        "n_rows": len(rows),
        "skipped_missing_features": skipped_missing_features,
        "skipped_empty_features": skipped_empty_features,
        "skipped_missing_reference": skipped_missing_reference,
    }


def _iter_task_dirs(
    results_dir: Path,
    *,
    models: Sequence[str] | None,
    tasks: Sequence[str] | None,
) -> Iterator[tuple[str, str, Path]]:
    model_filter = set(models) if models is not None else None
    task_filter = set(tasks) if tasks is not None else None
    for model_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        model = model_dir.name
        if model_filter is not None and model not in model_filter:
            continue
        for task_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            task = task_dir.name
            if task_filter is not None and task not in task_filter:
                continue
            yield model, task, task_dir


def _load_reference_lengths(task_dir: Path) -> dict[str, int]:
    refs: dict[str, int] = {}
    ref_dir = task_dir / "references"
    if not ref_dir.is_dir():
        return refs
    for path in sorted(ref_dir.glob("*.json")):
        data = json.loads(path.read_text())
        prompt_id = str(data["prompt_id"])
        refs[prompt_id] = len(data.get("gen_ids", []))
    return refs


def _load_reference_q(task_dir: Path) -> dict[str, float]:
    refs: dict[str, float] = {}
    ref_dir = task_dir / "references"
    if not ref_dir.is_dir():
        return refs
    for path in sorted(ref_dir.glob("*.json")):
        data = json.loads(path.read_text())
        refs[str(data["prompt_id"])] = float(data["q"])
    return refs


def _iter_hybrid_records(task_dir: Path) -> Iterator[dict[str, Any]]:
    q_ref = _load_reference_q(task_dir)
    hyb_dir = task_dir / "hybrids"
    if not hyb_dir.is_dir():
        return
    for shard in sorted(hyb_dir.glob("*.jsonl")):
        compressor, _, ratio_text = shard.stem.partition("__")
        ratio = float(ratio_text)
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_id = str(rec["prompt_id"])
                yield {
                    "prompt_id": prompt_id,
                    "compressor": compressor,
                    "ratio": ratio,
                    "s": int(rec["s"]),
                    "new_ids": list(rec.get("new_ids", [])),
                    "q": float(rec["q"]),
                    "q_ref": q_ref.get(prompt_id, np.nan),
                    "dq": float(rec["dq"]),
                    "features_path": rec.get("features_path"),
                    "feature_stem": safe_id(prompt_id),
                }
