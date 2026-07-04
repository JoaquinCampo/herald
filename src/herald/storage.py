"""Atomic, idempotent, resumable on-disk storage for HERALD sweep results.

Layout under results_dir:
    {model}/{task}/references/{safe_id(prompt_id)}.json
    {model}/{task}/references/{safe_id(prompt_id)}.npy
    {model}/{task}/references/_done.jsonl          (append-only manifest)
    {model}/{task}/hybrids/{compressor}__{ratio:.4f}.jsonl
    {model}/{task}/hybrid_features/{compressor}__{ratio:.4f}/
        {safe_id(prompt_id)}__s{s}.npy

Atomicity: JSON and NPY files are written to a temp file in the same
directory, fsynced, then renamed via os.replace (atomic on POSIX). The
manifest line is only appended AFTER both renames succeed, so a half-written
pair never appears as complete.
"""

import json
import os
import tempfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import numpy.typing as npt


def safe_id(prompt_id: str) -> str:
    """Encode a prompt_id as an injective filename-safe stem."""
    return quote(prompt_id, safe="")


def legacy_safe_id(prompt_id: str) -> str:
    """Pre-feature-collection filename convention for old artifacts."""
    return prompt_id.replace("/", "_")


def _ref_dir(results_dir: Path, model: str, task: str) -> Path:
    return results_dir / model / task / "references"


def _hybrid_dir(results_dir: Path, model: str, task: str) -> Path:
    return results_dir / model / task / "hybrids"


def _hybrid_feature_dir(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
) -> Path:
    return (
        results_dir
        / model
        / task
        / "hybrid_features"
        / _shard_name(compressor, ratio).removesuffix(".jsonl")
    )


def _shard_name(compressor: str, ratio: float) -> str:
    return f"{compressor}__{ratio:.4f}.jsonl"


def hybrid_feature_path(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    prompt_id: str,
    s: int,
) -> Path:
    """Return the stored compressed-stream feature path for a hybrid."""
    return (
        _hybrid_feature_dir(results_dir, model, task, compressor, ratio)
        / f"{safe_id(prompt_id)}__s{s}.npy"
    )


def reference_done(results_dir: Path, model: str, task: str) -> set[str]:
    """Return the set of prompt_ids whose reference is fully written.

    Reads _done.jsonl; skips any torn final line. Returns an empty set
    when the manifest does not exist.
    """
    manifest = _ref_dir(results_dir, model, task) / "_done.jsonl"
    if not manifest.exists():
        return set()
    done: set[str] = set()
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def save_reference(
    results_dir: Path,
    model: str,
    task: str,
    *,
    prompt_id: str,
    prompt_input_ids: list[int],
    gen_ids: list[int],
    text: str,
    q: float,
    features: npt.NDArray[np.float32],
) -> None:
    """Atomically write reference files, then record in the manifest.

    Both the JSON metadata file and the NPY features file are written via
    temp-then-rename so a crash leaves no partial state that reads as done.
    The manifest append is the commit point.
    """
    ref_dir = _ref_dir(results_dir, model, task)
    ref_dir.mkdir(parents=True, exist_ok=True)

    sid = safe_id(prompt_id)
    json_path = ref_dir / f"{sid}.json"
    npy_path = ref_dir / f"{sid}.npy"

    # Write JSON atomically
    payload = json.dumps(
        {
            "prompt_id": prompt_id,
            "prompt_input_ids": prompt_input_ids,
            "gen_ids": gen_ids,
            "text": text,
            "q": q,
        }
    ).encode()
    with tempfile.NamedTemporaryFile(delete=False, dir=ref_dir) as tf:
        tf.write(payload)
        tf.flush()
        os.fsync(tf.fileno())
        json_tmp = tf.name
    os.replace(json_tmp, json_path)

    # Write NPY atomically via a file object so np.save does not
    # append ".npy" to the name (it only does that with string paths).
    f16: npt.NDArray[np.float16] = features.astype(np.float16)
    with tempfile.NamedTemporaryFile(delete=False, dir=ref_dir) as tf:
        np.save(tf, f16)
        tf.flush()
        os.fsync(tf.fileno())
        npy_tmp = tf.name
    os.replace(npy_tmp, npy_path)

    # Commit: append to manifest only after both renames succeeded.
    manifest = ref_dir / "_done.jsonl"
    line = json.dumps({"prompt_id": prompt_id}) + "\n"
    with manifest.open("a") as mf:
        mf.write(line)
        mf.flush()
        os.fsync(mf.fileno())


def load_reference(
    results_dir: Path,
    model: str,
    task: str,
    prompt_id: str,
) -> dict[str, object]:
    """Load JSON metadata for a reference (does not load .npy features).

    Raises FileNotFoundError if the JSON file is missing.
    """
    ref_dir = _ref_dir(results_dir, model, task)
    json_path = ref_dir / f"{safe_id(prompt_id)}.json"
    if not json_path.exists():
        legacy_path = ref_dir / f"{legacy_safe_id(prompt_id)}.json"
        if legacy_path.exists():
            json_path = legacy_path
    if not json_path.exists():
        raise FileNotFoundError(f"Reference not found: {json_path}")
    with json_path.open() as f:
        data: dict[str, object] = json.load(f)
        return data


def hybrid_done(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    require_features: bool = False,
) -> set[tuple[str, int]]:
    """Return (prompt_id, s) pairs already recorded in the shard JSONL.

    Tolerates a torn/invalid final line by skipping any line that fails
    json.loads. Returns an empty set when the shard does not exist.
    When `require_features` is true, a row is done only if its
    compressed-stream feature file also exists. This lets upgraded
    sweeps backfill featureless legacy label rows.
    """
    shard = _hybrid_dir(results_dir, model, task) / _shard_name(
        compressor, ratio
    )
    if not shard.exists():
        return set()
    done: set[tuple[str, int]] = set()
    with shard.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                prompt_id = str(rec["prompt_id"])
                s = int(rec["s"])
                if require_features:
                    if "features_path" not in rec:
                        continue
                    fpath = hybrid_feature_path(
                        results_dir,
                        model,
                        task,
                        compressor,
                        ratio,
                        prompt_id,
                        s,
                    )
                    stored = Path(str(rec["features_path"]))
                    stored_path = (
                        stored
                        if stored.is_absolute()
                        else (results_dir / stored)
                    )
                    if stored_path != fpath:
                        continue
                    if not fpath.exists():
                        continue
                done.add((prompt_id, s))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return done


def save_hybrid_features(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    prompt_id: str,
    s: int,
    features: npt.NDArray[np.float32],
) -> Path:
    """Atomically write compressed-stream features for one hybrid."""
    feat_dir = _hybrid_feature_dir(
        results_dir, model, task, compressor, ratio
    )
    feat_dir.mkdir(parents=True, exist_ok=True)
    out_path = hybrid_feature_path(
        results_dir, model, task, compressor, ratio, prompt_id, s
    )
    f16: npt.NDArray[np.float16] = features.astype(np.float16)
    with tempfile.NamedTemporaryFile(delete=False, dir=feat_dir) as tf:
        np.save(tf, f16)
        tf.flush()
        os.fsync(tf.fileno())
        tmp = tf.name
    os.replace(tmp, out_path)
    return out_path


def append_hybrid(
    results_dir: Path,
    model: str,
    task: str,
    compressor: str,
    ratio: float,
    *,
    prompt_id: str,
    s: int,
    new_ids: list[int],
    text: str,
    q: float,
    dq: float,
    features: npt.NDArray[np.float32] | None = None,
) -> None:
    """Append one JSON line to the shard JSONL for (compressor, ratio).

    Opens in append mode, writes line + newline, flushes, fsyncs.
    If compressed-stream features are supplied, writes them atomically
    before appending the label row; the JSONL append is the commit point.
    """
    hyb_dir = _hybrid_dir(results_dir, model, task)
    hyb_dir.mkdir(parents=True, exist_ok=True)

    shard = hyb_dir / _shard_name(compressor, ratio)
    rec: dict[str, object] = {
        "prompt_id": prompt_id,
        "s": s,
        "new_ids": new_ids,
        "text": text,
        "q": q,
        "dq": dq,
    }
    if features is not None:
        feat_path = save_hybrid_features(
            results_dir,
            model,
            task,
            compressor,
            ratio,
            prompt_id=prompt_id,
            s=s,
            features=features,
        )
        rec["features_path"] = str(feat_path.relative_to(results_dir))
    record = json.dumps(rec)
    with shard.open("a") as f:
        f.write(record + "\n")
        f.flush()
        os.fsync(f.fileno())
