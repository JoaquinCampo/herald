"""Build switch-level predictor rows from HERALD sweep artifacts.

The sweep stores references as one JSON plus one NPY feature matrix per
prompt, and hybrids as JSONL shards per compressor and ratio. This
module joins them into rows where the observation is one possible switch
decision:

    (task, prompt_id, compressor, ratio, s) -> dq

Feature columns are causal: row ``s`` uses only the reference logit
features up to and including generated token position ``s``.
"""

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from herald.features import derive_features
from herald.storage import legacy_safe_id, safe_id

PROBE_SCALARS: tuple[str, ...] = (
    "entropy",
    "max_prob",
    "margin_prob",
    "chosen_logprob",
)


@dataclass(frozen=True)
class ReferenceArtifact:
    prompt_id: str
    q: float
    ref_len: int
    features_path: Path
    gen_ids: tuple[int, ...] = ()
    feature_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class HybridRecord:
    prompt_id: str
    compressor: str
    ratio: float
    s: int
    q: float
    dq: float
    new_ids: tuple[int, ...] = ()
    press_features: tuple[tuple[str, float], ...] = ()


def parse_hybrid_shard(path: Path) -> tuple[str, float]:
    """Return ``(compressor, ratio)`` from a hybrid shard filename."""
    compressor, sep, ratio_text = path.stem.partition("__")
    if not sep:
        raise ValueError(f"bad hybrid shard name: {path.name}")
    return compressor, _finite_float(ratio_text)


def load_references(task_dir: Path) -> dict[str, ReferenceArtifact]:
    """Load reference metadata for one ``{model}/{task}`` directory."""
    ref_dir = task_dir / "references"
    if not ref_dir.is_dir():
        return {}

    refs: dict[str, ReferenceArtifact] = {}
    for json_path in sorted(ref_dir.glob("*.json")):
        if json_path.name == "_done.jsonl":
            continue
        try:
            data = _read_json_object(json_path)
            prompt_id = str(data["prompt_id"])
            q = _finite_float(data["q"])
        except (KeyError, ValueError):
            continue
        gen_ids = data.get("gen_ids", [])
        features_path = ref_dir / f"{safe_id(prompt_id)}.npy"
        if not features_path.exists():
            legacy_path = ref_dir / f"{legacy_safe_id(prompt_id)}.npy"
            if legacy_path.exists():
                features_path = legacy_path
        raw_names = data.get("feature_names")
        feature_names = (
            tuple(str(n) for n in raw_names)
            if isinstance(raw_names, list)
            else None
        )
        refs[prompt_id] = ReferenceArtifact(
            prompt_id=prompt_id,
            q=q,
            ref_len=len(gen_ids) if isinstance(gen_ids, list) else 0,
            features_path=features_path,
            gen_ids=(
                tuple(int(i) for i in gen_ids)
                if isinstance(gen_ids, list)
                else ()
            ),
            feature_names=feature_names,
        )
    return refs


def iter_hybrids(task_dir: Path) -> Iterator[HybridRecord]:
    """Yield hybrid rows from all JSONL shards under one task directory."""
    hyb_dir = task_dir / "hybrids"
    if not hyb_dir.is_dir():
        return

    for shard in sorted(hyb_dir.glob("*.jsonl")):
        compressor, ratio = parse_hybrid_shard(shard)
        records: dict[tuple[str, int], HybridRecord] = {}
        with shard.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    prompt_id = str(data["prompt_id"])
                    s = _integer(data["s"])
                    raw_press = data.get("press_features")
                    press = (
                        tuple(
                            (str(k), float(v))
                            for k, v in sorted(raw_press.items())
                            if isinstance(v, (int, float))
                        )
                        if isinstance(raw_press, dict)
                        else ()
                    )
                    raw_new = data.get("new_ids")
                    new_ids = (
                        tuple(int(i) for i in raw_new[:4])
                        if isinstance(raw_new, list)
                        else ()
                    )
                    records[(prompt_id, s)] = HybridRecord(
                        prompt_id=prompt_id,
                        compressor=compressor,
                        ratio=ratio,
                        s=s,
                        q=_finite_float(data["q"]),
                        dq=_finite_float(data["dq"]),
                        new_ids=new_ids,
                        press_features=press,
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
        for record in records.values():  # noqa: UP028
            yield record


def build_switch_rows(
    task_dir: Path,
    *,
    model: str,
    task: str,
    feature_prefix: str = "feat__",
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build switch-level rows for one task directory.

    Rows with missing references, missing feature files, or switch
    positions outside the reference feature matrix are skipped and
    counted in the summary.
    """
    refs = load_references(task_dir)
    feature_cache: dict[str, tuple[np.ndarray, list[str]]] = {}
    rows: list[dict[str, Any]] = []
    skipped_missing_ref = 0
    skipped_missing_features = 0
    skipped_s_out_of_range = 0
    n_hybrids_seen = 0

    for hybrid in iter_hybrids(task_dir):
        n_hybrids_seen += 1
        ref = refs.get(hybrid.prompt_id)
        if ref is None:
            skipped_missing_ref += 1
            continue
        if not ref.features_path.exists():
            skipped_missing_features += 1
            continue
        if hybrid.prompt_id not in feature_cache:
            per_step = np.load(ref.features_path).astype(np.float32)
            derived, names = derive_features(
                per_step, names=ref.feature_names
            )
            feature_cache[hybrid.prompt_id] = (derived, names)
        derived, names = feature_cache[hybrid.prompt_id]
        if hybrid.s < 0 or hybrid.s >= derived.shape[0]:
            skipped_s_out_of_range += 1
            continue

        row: dict[str, Any] = {
            "model": model,
            "task": task,
            "prompt_id": hybrid.prompt_id,
            "compressor": hybrid.compressor,
            "ratio": hybrid.ratio,
            "s": hybrid.s,
            "relative_s": _relative_position(hybrid.s, ref.ref_len),
            "ref_len": ref.ref_len,
            "q_ref": ref.q,
            "q_hybrid": hybrid.q,
            "dq": hybrid.dq,
            "damaged": _flag(hybrid.dq > 0.0),
            "major_damage": _flag(hybrid.dq >= 0.5),
        }
        values = derived[hybrid.s]
        for name, value in zip(names, values, strict=True):
            row[f"{feature_prefix}{name}"] = _feature_float(value)
        _attach_probe_columns(row, task_dir, ref, hybrid)
        for key, value in hybrid.press_features:
            row[f"press__{key.removeprefix('press_')}"] = value
        rows.append(row)
        if max_rows is not None and len(rows) >= max_rows:
            break

    summary = {
        "model": model,
        "task": task,
        "n_references": len(refs),
        "n_hybrids_seen": n_hybrids_seen,
        "n_rows": len(rows),
        "n_feature_matrices_loaded": len(feature_cache),
        "skipped_missing_ref": skipped_missing_ref,
        "skipped_missing_features": skipped_missing_features,
        "skipped_s_out_of_range": skipped_s_out_of_range,
    }
    return rows, summary


def _attach_probe_columns(
    row: dict[str, Any],
    task_dir: Path,
    ref: ReferenceArtifact,
    hybrid: HybridRecord,
) -> None:
    """Attach 1-token-probe features reconstructed from artifacts.

    ``probe__token_match`` and ``probe__match_len4`` compare the
    hybrid's first generated tokens with the reference continuation
    at ``s``. The ``probe__h0_*``/``probe__d0_*`` scalars are the
    hybrid stream's step-0 logit features (the model's reaction to
    the freshly compressed cache) and their deltas vs the reference
    at the same position. In deployment these equal a 1-token probe:
    one extra forward pass with the compressed cache.
    """
    if hybrid.new_ids and hybrid.s < len(ref.gen_ids):
        window = ref.gen_ids[hybrid.s : hybrid.s + 4]
        match = 0
        for a, b in zip(hybrid.new_ids, window, strict=False):
            if a != b:
                break
            match += 1
        row["probe__token_match"] = float(
            hybrid.new_ids[0] == ref.gen_ids[hybrid.s]
        )
        row["probe__match_len4"] = float(match)

    from herald.features import FEATURE_NAMES

    hyb_path = (
        task_dir
        / "hybrid_features"
        / f"{hybrid.compressor}__{hybrid.ratio:.4f}"
        / f"{safe_id(hybrid.prompt_id)}__s{hybrid.s}.npy"
    )
    if not hyb_path.exists():
        return
    try:
        hyb_step0 = np.load(hyb_path).astype(np.float32)[0]
    except (OSError, ValueError, IndexError):
        return
    if hyb_step0.shape[0] < len(FEATURE_NAMES):
        return
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    ref_names = (
        list(ref.feature_names)
        if ref.feature_names is not None
        else list(FEATURE_NAMES)
    )
    ref_idx = {name: i for i, name in enumerate(ref_names)}
    try:
        ref_step = np.load(ref.features_path).astype(np.float32)[hybrid.s]
    except (OSError, ValueError, IndexError):
        ref_step = None
    for name in PROBE_SCALARS:
        h0 = float(hyb_step0[idx[name]])
        row[f"probe__h0_{name}"] = h0
        if ref_step is not None and name in ref_idx:
            row[f"probe__d0_{name}"] = h0 - float(ref_step[ref_idx[name]])


def rows_to_table(rows: Sequence[dict[str, Any]]) -> Any:
    """Arrow table over the union of row keys.

    ``pa.Table.from_pylist`` infers the schema from leading rows and
    silently drops columns that first appear later (e.g. probe
    columns present only on tasks with hybrid feature files).
    """
    import pyarrow as pa

    keys: dict[str, None] = {}
    for row in rows:
        for key in row:
            keys.setdefault(key)
    filled = [{key: row.get(key) for key in keys} for row in rows]
    return pa.Table.from_pylist(filled)


def _read_json_object(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk or raise ``ValueError``."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object from {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _finite_float(value: object) -> float:
    """Return ``value`` as a finite float."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected finite float, got {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"expected finite float, got {value!r}")
    return out


def _feature_float(value: object) -> float:
    """Return a feature value as float, preserving NaN feature sentinels."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected numeric feature, got {value!r}") from exc


def _integer(value: object) -> int:
    """Return ``value`` as an integer."""
    if isinstance(value, bool):
        raise ValueError(f"expected int, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"expected int, got {value!r}") from exc
    if (
        isinstance(value, float)
        and math.isfinite(value)
        and value.is_integer()
    ):
        try:
            return int(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"expected int, got {value!r}") from exc
    raise ValueError(f"expected int, got {value!r}")


def _relative_position(s: int, ref_len: int) -> float:
    """Return switch position as a fraction of reference length."""
    if ref_len <= 0:
        return np.nan
    return _finite_float(s / ref_len)


def _flag(condition: bool) -> int:
    """Return an integer indicator for a boolean condition."""
    return 1 if condition else 0


def iter_task_dirs(
    results_dir: Path,
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
) -> Iterator[tuple[str, str, Path]]:
    """Yield ``(model, task, task_dir)`` directories in storage layout."""
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


def build_switch_dataset(
    results_dir: Path,
    *,
    models: Sequence[str] | None = None,
    tasks: Sequence[str] | None = None,
    max_rows_per_task: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build switch rows across a sweep results directory."""
    rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    for model, task, task_dir in iter_task_dirs(
        results_dir, models=models, tasks=tasks
    ):
        task_rows, summary = build_switch_rows(
            task_dir,
            model=model,
            task=task,
            max_rows=max_rows_per_task,
        )
        rows.extend(task_rows)
        task_summaries.append(summary)

    summary = {
        "results_dir": str(results_dir),
        "n_rows": len(rows),
        "n_tasks": len(task_summaries),
        "tasks": task_summaries,
    }
    return rows, summary
