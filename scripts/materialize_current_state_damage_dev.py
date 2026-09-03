"""Materialize the quarantined development rows with causal-only features."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

SCHEMA_VERSION = "herald.current_state_damage_development_data.v1"
HORIZONS = (5, 10, 25, 50)
TARGET_COLUMNS = tuple(f"future_sum_js_{horizon}" for horizon in HORIZONS)
IDENTITY_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
)
INSTANTANEOUS_SENSORS = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "delta_h_valid",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
)
HISTORY_BASES = (
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
)
WINDOWS = (8, 32)
HALF_LIVES = (8, 32)
PROJECTED_COLUMNS = IDENTITY_COLUMNS + INSTANTANEOUS_SENSORS + TARGET_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--target-audit", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def validate_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, int]]:
    lock = load_json(args.protocol_lock)
    metadata = load_json(args.metadata_audit)
    target = load_json(args.target_audit)
    development = load_json(args.development_manifest)
    protocol_hash = sha256_file(args.protocol_lock)
    manifest_hash = sha256_file(args.development_manifest)
    if metadata.get("pass") is not True or target.get("pass") is not True:
        raise ValueError("prerequisite audit failed")
    if metadata.get("protocol_lock_sha256") != protocol_hash:
        raise ValueError("metadata audit protocol mismatch")
    if target.get("protocol_lock_sha256") != protocol_hash:
        raise ValueError("target audit protocol mismatch")
    if metadata["split"]["development_manifest_sha256"] != manifest_hash:
        raise ValueError("development manifest hash mismatch")
    if target.get("development_manifest_sha256") != manifest_hash:
        raise ValueError("target audit manifest mismatch")
    if tuple(lock["dataset"]["horizons"]) != HORIZONS:
        raise ValueError("protocol horizons changed")
    if tuple(lock["targets"]["primary_columns"]) != TARGET_COLUMNS:
        raise ValueError("protocol targets changed")
    if (
        tuple(lock["features"]["instantaneous_sensors"])
        != INSTANTANEOUS_SENSORS
    ):
        raise ValueError("protocol sensor roster changed")
    if tuple(lock["features"]["causal_history"]["bases"]) != HISTORY_BASES:
        raise ValueError("protocol history roster changed")
    folds = {str(row["prompt_id"]): int(row["fold"]) for row in development}
    if len(folds) != metadata["split"]["development_prompts"]:
        raise ValueError("development prompt count mismatch")
    return lock, folds


def run_boundaries(run_ids: np.ndarray) -> list[tuple[int, int]]:
    if len(run_ids) == 0:
        return []
    changes = np.flatnonzero(run_ids[1:] != run_ids[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [len(run_ids)]))
    return [
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]


def trailing_stats(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(values)
    clean = np.where(valid, values, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(clean, dtype=np.float64)))
    squared = np.concatenate(
        ([0.0], np.cumsum(clean * clean, dtype=np.float64))
    )
    counts = np.concatenate(([0], np.cumsum(valid, dtype=np.int64)))
    right = np.arange(1, len(values) + 1)
    left = np.maximum(right - window, 0)
    count = counts[right] - counts[left]
    total = cumulative[right] - cumulative[left]
    total_squared = squared[right] - squared[left]
    mean = np.divide(
        total,
        count,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 0,
    )
    variance_numerator = total_squared - np.divide(
        total * total,
        count,
        out=np.zeros(len(values), dtype=np.float64),
        where=count > 0,
    )
    variance = np.divide(
        np.maximum(variance_numerator, 0.0),
        count - 1,
        out=np.full(len(values), np.nan, dtype=np.float64),
        where=count > 1,
    )
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def ewma(values: np.ndarray, half_life: int) -> np.ndarray:
    alpha = 1.0 - np.exp(np.log(0.5) / half_life)
    output = np.full(len(values), np.nan, dtype=np.float32)
    state = np.nan
    for index, value in enumerate(values):
        if np.isfinite(value):
            state = (
                value
                if np.isnan(state)
                else alpha * value + (1.0 - alpha) * state
            )
        output[index] = state
    return output


def add_causal_history(frame: pd.DataFrame) -> list[str]:
    boundaries = run_boundaries(frame["run_id"].to_numpy())
    feature_names: list[str] = []
    for base in HISTORY_BASES:
        values = frame[base].to_numpy(dtype=np.float64)
        for window in WINDOWS:
            mean_name = f"{base}_causal_mean_{window}"
            std_name = f"{base}_causal_std_{window}"
            mean = np.empty(len(frame), dtype=np.float32)
            std = np.empty(len(frame), dtype=np.float32)
            for start, end in boundaries:
                mean[start:end], std[start:end] = trailing_stats(
                    values[start:end], window
                )
            frame[mean_name] = mean
            frame[std_name] = std
            feature_names.extend((mean_name, std_name))
        for half_life in HALF_LIVES:
            name = f"{base}_causal_ewma_hl{half_life}"
            history = np.empty(len(frame), dtype=np.float32)
            for start, end in boundaries:
                history[start:end] = ewma(values[start:end], half_life)
            frame[name] = history
            feature_names.append(name)
    return feature_names


def add_targets(frame: pd.DataFrame) -> list[str]:
    prior = np.zeros(len(frame), dtype=np.float32)
    lower = 0
    names: list[str] = []
    for horizon, column in zip(HORIZONS, TARGET_COLUMNS, strict=True):
        name = f"band_rate_{lower}_{horizon}"
        current = frame[column].to_numpy(dtype=np.float32)
        frame[name] = (current - prior) / (horizon - lower)
        prior = current
        lower = horizon
        names.append(name)
    return names


def materialize_press(
    path: Path,
    folds: dict[str, int],
    output: Path,
) -> dict[str, Any]:
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    prompt_filter = ds.field("prompt_id").isin(  # type: ignore[attr-defined,no-untyped-call]
        sorted(folds)
    )
    table = dataset.to_table(
        columns=list(PROJECTED_COLUMNS), filter=prompt_filter
    )
    frame = table.to_pandas()
    if not set(frame["prompt_id"].unique()) <= set(folds):
        raise ValueError(f"confirmation prompt read from {path.name}")
    frame.sort_values(
        ["run_id", "token_pos"], inplace=True, ignore_index=True
    )
    frame["fold"] = frame["prompt_id"].map(folds).astype(np.int8)
    frame["log_token_clock"] = np.log1p(
        frame["token_pos"].to_numpy(dtype=np.float64) + 1.0
    ).astype(np.float32)
    history_names = add_causal_history(frame)
    band_names = add_targets(frame)
    float_columns = [
        *INSTANTANEOUS_SENSORS,
        *TARGET_COLUMNS,
        "log_token_clock",
        *history_names,
        *band_names,
    ]
    for column in float_columns:
        if column != "delta_h_valid":
            frame[column] = frame[column].astype(np.float32)
    frame["delta_h_valid"] = frame["delta_h_valid"].astype(bool)
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(  # type: ignore[no-untyped-call]
        pa.Table.from_pandas(frame, preserve_index=False),
        output,
        compression="zstd",
        use_dictionary=("run_id", "prompt_id", "task", "press"),
    )
    return {
        "rows": len(frame),
        "runs": int(frame["run_id"].nunique()),
        "prompts": int(frame["prompt_id"].nunique()),
        "sha256": sha256_file(output),
        "columns": list(frame.columns),
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.output_root / "manifest.json"
    if manifest_path.exists() or args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_root}")
    lock, folds = validate_inputs(args)
    args.output_root.mkdir(parents=True)
    files: dict[str, Any] = {}
    for source in sorted((args.dataset_root / "tokens").glob("*.parquet")):
        destination = args.output_root / source.name
        files[source.stem] = materialize_press(source, folds, destination)
    if set(files) != set(lock["dataset"]["presses"]):
        raise ValueError("token press roster changed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_causal_only_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "metadata_audit_sha256": sha256_file(args.metadata_audit),
        "target_audit_sha256": sha256_file(args.target_audit),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "source_columns_projected": list(PROJECTED_COLUMNS),
        "confirmation_prompts_projected": 0,
        "released_aggregate_columns_loaded": [],
        "oracle_columns_loaded": [],
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
