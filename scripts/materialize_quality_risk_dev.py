"""Materialize quarantined development rows with causal labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from materialize_current_state_damage_dev import (  # noqa: E402
    INSTANTANEOUS_SENSORS,
    add_causal_history,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herald.quality_risk_engineering import (  # noqa: E402
    ENGINEERED_NAMES,
    add_engineered_features,
)
from herald.quality_risk_labels import (  # noqa: E402
    CATASTROPHE_COLUMNS,
    SCORE_COLUMNS,
    apply_run_labels,
)

SCHEMA_VERSION = "herald.quality_risk_development_data.v1"
IDENTITY_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
    "baseline_run_id",
)
PROJECTED_COLUMNS = (
    *IDENTITY_COLUMNS,
    *INSTANTANEOUS_SENSORS,
    *SCORE_COLUMNS,
    *CATASTROPHE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], set[str]]:
    lock = load_json(args.protocol_lock)
    audit = load_json(args.label_audit)
    development = load_json(args.development_manifest)
    if lock.get("schema_version") != "herald.quality_risk.v1":
        raise ValueError("unexpected protocol schema")
    if audit.get("pass") is not True:
        raise ValueError("label audit did not pass")
    if audit.get("protocol_lock_sha256") != sha256_file(args.protocol_lock):
        raise ValueError("label audit belongs to another protocol")
    if audit.get("development_manifest_sha256") != sha256_file(
        args.development_manifest
    ):
        raise ValueError("development manifest hash mismatch")
    if (
        tuple(lock["inputs"]["instantaneous_sensors"])
        != INSTANTANEOUS_SENSORS
    ):
        raise ValueError("protocol sensor roster changed")
    if tuple(lock["dataset"]["presses"]) != (
        "expected_attention",
        "knorm",
        "random",
        "snapkv",
        "streaming_llm",
        "tova",
    ):
        raise ValueError("protocol press roster changed")
    folds = {str(row["prompt_id"]): int(row["fold"]) for row in development}
    gap = set(audit["excluded_gap_prompts"])
    labeled_prompts = set(folds) - gap
    if not labeled_prompts:
        raise ValueError("no labeled development prompts")
    return lock, audit, folds, labeled_prompts


def materialize_press(
    path: Path,
    folds: dict[str, int],
    labeled_prompts: set[str],
    audit: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    prompt_filter = ds.field("prompt_id").isin(  # type: ignore[attr-defined,no-untyped-call]
        sorted(labeled_prompts)
    )
    table = dataset.to_table(
        columns=list(PROJECTED_COLUMNS), filter=prompt_filter
    )
    frame = table.to_pandas()
    if not set(frame["prompt_id"].unique()) <= labeled_prompts:
        raise ValueError(f"unlabeled prompt read from {path.name}")
    frame.sort_values(
        ["run_id", "token_pos"], inplace=True, ignore_index=True
    )
    runs = (
        frame.sort_values("token_pos").groupby("run_id").first().reset_index()
    )
    labeled_runs = apply_run_labels(
        runs[
            [
                "run_id",
                "prompt_id",
                *SCORE_COLUMNS,
            ]
        ],
        gap_prompts=audit["excluded_gap_prompts"],
        lift_prompts=audit["lift_reference_zero_prompts"],
    )
    expected_runs = audit["prevalence_by_press"][path.stem]["runs"]
    if len(labeled_runs) != expected_runs:
        raise ValueError(
            f"labeled run count mismatch in {path.name}: "
            f"{len(labeled_runs)} != {expected_runs}"
        )
    run_labels = labeled_runs.set_index("run_id")[
        ["damage", "imputed_compressed_zero", "lift_reference_zero"]
    ]
    label_columns = (
        "damage",
        "imputed_compressed_zero",
        "lift_reference_zero",
    )
    for column in label_columns:
        frame[column] = (
            frame["run_id"].map(run_labels[column]).astype(np.int8)
        )
    if frame[list(label_columns)].isna().any().any():
        raise ValueError(f"unlabeled run in {path.name}")
    run_catastrophe = (
        frame[list(CATASTROPHE_COLUMNS)]
        .fillna(False)
        .any(axis=True)
        .groupby(frame["run_id"])
        .max()
        .astype(np.int8)
    )
    frame["catastrophe"] = (
        frame["run_id"].map(run_catastrophe).astype(np.int8)
    )
    run_length = frame.groupby("run_id")["token_pos"].max() + 1
    frame["run_length"] = frame["run_id"].map(run_length).astype(np.int32)
    frame["fold"] = frame["prompt_id"].map(folds).astype(np.int8)
    frame["log_token_clock"] = np.log1p(
        frame["token_pos"].to_numpy(dtype=np.float64) + 1.0
    ).astype(np.float32)
    history_names = add_causal_history(frame)
    add_engineered_features(frame)
    for column in [
        *INSTANTANEOUS_SENSORS,
        "log_token_clock",
        *history_names,
        *ENGINEERED_NAMES,
    ]:
        if column != "delta_h_valid":
            frame[column] = frame[column].astype(np.float32)
    frame["delta_h_valid"] = frame["delta_h_valid"].astype(bool)
    keep = [
        "run_id",
        "token_pos",
        "prompt_id",
        "task",
        "press",
        "compression_ratio",
        "fold",
        "run_length",
        "log_token_clock",
        *INSTANTANEOUS_SENSORS,
        *history_names,
        *ENGINEERED_NAMES,
        "damage",
        "catastrophe",
        "imputed_compressed_zero",
        "lift_reference_zero",
    ]
    frame = frame[keep]
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
    lock, audit, folds, labeled_prompts = validate_inputs(args)
    args.output_root.mkdir(parents=True)
    files: dict[str, Any] = {}
    for source in sorted((args.dataset_root / "tokens").glob("*.parquet")):
        destination = args.output_root / source.name
        files[source.stem] = materialize_press(
            source, folds, labeled_prompts, audit, destination
        )
    if set(files) != set(lock["dataset"]["presses"]):
        raise ValueError("token press roster changed")
    total_runs = sum(info["runs"] for info in files.values())
    if total_runs != audit["total_runs"]:
        raise ValueError("materialized run total disagrees with label audit")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_causal_labeled_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "label_audit_sha256": sha256_file(args.label_audit),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "engineering_module_sha256": sha256_file(
            Path(__file__).resolve().parent.parent
            / "src"
            / "herald"
            / "quality_risk_engineering.py"
        ),
        "engineered_features": list(ENGINEERED_NAMES),
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
