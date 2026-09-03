"""Audit public HERALD metadata and freeze prompt-group quarantine splits."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

SCHEMA_VERSION = "herald.current_state_damage_metadata_audit.v1"
SPLIT_SALT = "herald.current_state_damage.v1.split"
FOLD_SALT = "herald.current_state_damage.v1.fold"
SEQUENCE_COLUMNS = [
    "run_id",
    "prompt_id",
    "baseline_run_id",
    "task",
    "press",
    "compression_ratio",
]
TOKEN_METADATA_COLUMNS = [
    "run_id",
    "token_pos",
    "prompt_id",
    "task",
    "press",
    "compression_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(salt: str, revision: str, prompt_id: str) -> str:
    value = f"{salt}\0{revision}\0{prompt_id}"
    return hashlib.sha256(value.encode()).hexdigest()


def load_lock(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("protocol lock must be an object")
    if (
        value.get("schema_version")
        != "herald.current_state_compression_damage.v1"
    ):
        raise ValueError("unexpected protocol schema")
    if value.get("status") != "locked_before_bulk_target_projection":
        raise ValueError("protocol was not locked before target projection")
    return value


def validate_input_hashes(root: Path, lock: dict[str, Any]) -> dict[str, str]:
    expected = lock["dataset"]["file_sha256"]
    if not isinstance(expected, dict) or not expected:
        raise ValueError("protocol has no file hashes")
    actual = {name: sha256_file(root / name) for name in expected}
    if actual != expected:
        raise ValueError("dataset file hash mismatch")
    return actual


def read_sequence_metadata(root: Path) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in sorted((root / "sequences").glob("*.parquet")):
        frame = pq.read_table(  # type: ignore[no-untyped-call]
            path, columns=SEQUENCE_COLUMNS
        ).to_pandas()
        if set(frame["press"].unique()) != {path.stem}:
            raise ValueError(f"sequence press/file mismatch: {path.name}")
        parts.append(frame)
    if not parts:
        raise ValueError("no sequence files found")
    return pd.concat(parts, ignore_index=True)


def audit_sequences(
    frame: pd.DataFrame, lock: dict[str, Any]
) -> dict[str, Any]:
    dataset = lock["dataset"]
    expected_presses = set(dataset["presses"])
    expected_ratios = set(dataset["ratios"])
    if len(frame) != dataset["sequence_rows_expected"]:
        raise ValueError("unexpected sequence row count")
    if frame["run_id"].nunique() != len(frame):
        raise ValueError("sequence run IDs are not unique")
    if frame["prompt_id"].nunique() != dataset["prompt_groups_expected"]:
        raise ValueError("unexpected prompt count")
    if set(frame["task"].unique()) != set(dataset["tasks"]):
        raise ValueError("unexpected task roster")
    if frame.groupby("prompt_id")["task"].nunique().max() != 1:
        raise ValueError("a prompt belongs to multiple tasks")
    if set(frame["press"].unique()) != expected_presses | {"none"}:
        raise ValueError("unexpected press roster")
    if (
        set(frame.loc[frame["press"] != "none", "compression_ratio"].unique())
        != expected_ratios
    ):
        raise ValueError("unexpected compression-ratio roster")
    if set(
        frame.loc[frame["press"] == "none", "compression_ratio"].unique()
    ) != {0.0}:
        raise ValueError("none rows do not have ratio zero")

    expected_actions = {
        (press, ratio)
        for press in expected_presses
        for ratio in expected_ratios
    }
    for prompt_id, prompt in frame.groupby("prompt_id", sort=False):
        if len(prompt) != 1 + len(expected_actions):
            raise ValueError(f"incomplete sequence roster for {prompt_id}")
        none = prompt[prompt["press"] == "none"]
        if (
            len(none) != 1
            or none.iloc[0]["run_id"] != none.iloc[0]["baseline_run_id"]
        ):
            raise ValueError(f"invalid none sequence for {prompt_id}")
        compressed = prompt[prompt["press"] != "none"]
        actions = set(
            zip(
                compressed["press"],
                compressed["compression_ratio"],
                strict=True,
            )
        )
        if actions != expected_actions:
            raise ValueError(f"incomplete action roster for {prompt_id}")
        if set(compressed["baseline_run_id"]) != {none.iloc[0]["run_id"]}:
            raise ValueError(f"invalid baseline links for {prompt_id}")

    task_counts = (
        frame[["task", "prompt_id"]]
        .drop_duplicates()
        .groupby("task")["prompt_id"]
        .size()
        .sort_index()
    )
    return {
        "rows": len(frame),
        "unique_runs": int(frame["run_id"].nunique()),
        "unique_prompts": int(frame["prompt_id"].nunique()),
        "prompt_counts_by_task": {
            key: int(value) for key, value in task_counts.items()
        },
        "runs_per_prompt": int(frame.groupby("prompt_id").size().iloc[0]),
        "coverage_exact": True,
    }


def audit_token_metadata(
    root: Path, sequences: pd.DataFrame, lock: dict[str, Any]
) -> dict[str, Any]:
    compressed = sequences[sequences["press"] != "none"].set_index("run_id")
    all_token_runs: set[str] = set()
    total_rows = 0
    by_press: dict[str, Any] = {}
    for path in sorted((root / "tokens").glob("*.parquet")):
        frame = pq.read_table(  # type: ignore[no-untyped-call]
            path, columns=TOKEN_METADATA_COLUMNS
        ).to_pandas()
        total_rows += len(frame)
        if set(frame["press"].unique()) != {path.stem}:
            raise ValueError(f"token press/file mismatch: {path.name}")
        grouped = frame.groupby("run_id", sort=False)
        sizes = grouped.size()
        minima = grouped["token_pos"].min()
        maxima = grouped["token_pos"].max()
        unique_positions = grouped["token_pos"].nunique()
        if not (
            (minima == 0)
            & (sizes == maxima + 1)
            & (sizes == unique_positions)
        ).all():
            raise ValueError(f"noncontiguous token positions: {path.name}")
        run_rows = grouped.agg(
            prompt_id=("prompt_id", "first"),
            prompt_count=("prompt_id", "nunique"),
            task=("task", "first"),
            task_count=("task", "nunique"),
            press=("press", "first"),
            press_count=("press", "nunique"),
            compression_ratio=("compression_ratio", "first"),
            ratio_count=("compression_ratio", "nunique"),
        )
        if (
            (
                run_rows[
                    [
                        "prompt_count",
                        "task_count",
                        "press_count",
                        "ratio_count",
                    ]
                ]
                != 1
            )
            .any()
            .any()
        ):
            raise ValueError(
                f"token run metadata changes within run: {path.name}"
            )
        run_ids = set(str(value) for value in run_rows.index)
        if all_token_runs & run_ids:
            raise ValueError("a token run appears in multiple files")
        all_token_runs.update(run_ids)
        expected = compressed.loc[list(run_rows.index)]
        if not (
            (
                run_rows["prompt_id"].to_numpy()
                == expected["prompt_id"].to_numpy()
            ).all()
            and (
                run_rows["task"].to_numpy() == expected["task"].to_numpy()
            ).all()
            and (
                run_rows["press"].to_numpy() == expected["press"].to_numpy()
            ).all()
            and (
                run_rows["compression_ratio"].to_numpy()
                == expected["compression_ratio"].to_numpy()
            ).all()
        ):
            raise ValueError(f"token/sequence identity mismatch: {path.name}")
        by_press[path.stem] = {
            "rows": len(frame),
            "runs": len(run_rows),
            "min_run_tokens": int(sizes.min()),
            "max_run_tokens": int(sizes.max()),
        }
    if total_rows != lock["dataset"]["token_rows_expected"]:
        raise ValueError("unexpected token row count")
    if all_token_runs != set(str(value) for value in compressed.index):
        raise ValueError("token and compressed sequence run rosters differ")
    return {
        "rows": total_rows,
        "unique_runs": len(all_token_runs),
        "by_press": by_press,
        "positions_contiguous": True,
        "sequence_identity_exact": True,
    }


def freeze_splits(
    frame: pd.DataFrame, revision: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = frame[["prompt_id", "task"]].drop_duplicates()
    development: list[dict[str, Any]] = []
    confirmation: list[dict[str, Any]] = []
    for task, task_prompts in prompts.groupby("task", sort=True):
        rows: list[dict[str, Any]] = [
            {
                "prompt_id": str(prompt_id),
                "task": str(task),
                "split_hash": stable_hash(
                    SPLIT_SALT, revision, str(prompt_id)
                ),
            }
            for prompt_id in task_prompts["prompt_id"]
        ]
        rows.sort(key=lambda row: (row["split_hash"], row["prompt_id"]))
        n_development = 2 * len(rows) // 3
        development_rows = rows[:n_development]
        for row in development_rows:
            row["fold_hash"] = stable_hash(
                FOLD_SALT, revision, row["prompt_id"]
            )
        development_rows.sort(
            key=lambda row: (row["fold_hash"], row["prompt_id"])
        )
        for rank, row in enumerate(development_rows):
            row["fold"] = rank % 5
        development.extend(development_rows)
        confirmation.extend(rows[n_development:])
    development.sort(key=lambda row: (row["task"], row["prompt_id"]))
    confirmation.sort(key=lambda row: (row["task"], row["prompt_id"]))
    return development, confirmation


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "audit": args.output_root / "metadata_audit.json",
        "development": args.output_root / "development_prompts.json",
        "confirmation": args.output_root / "confirmation_prompts.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError(
            "refusing to overwrite metadata audit artifacts"
        )
    lock = load_lock(args.protocol_lock)
    hashes = validate_input_hashes(args.dataset_root, lock)
    sequences = read_sequence_metadata(args.dataset_root)
    sequence_audit = audit_sequences(sequences, lock)
    token_audit = audit_token_metadata(args.dataset_root, sequences, lock)
    development, confirmation = freeze_splits(
        sequences, lock["dataset"]["revision"]
    )
    write_json(output_paths["development"], development)
    write_json(output_paths["confirmation"], confirmation)
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "metadata_only_quarantine_frozen",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "dataset_hashes": hashes,
        "columns_loaded": {
            "sequences": SEQUENCE_COLUMNS,
            "tokens": TOKEN_METADATA_COLUMNS,
        },
        "target_columns_loaded": [],
        "sensor_columns_loaded": [],
        "sequence_audit": sequence_audit,
        "token_audit": token_audit,
        "split": {
            "development_prompts": len(development),
            "confirmation_prompts": len(confirmation),
            "development_by_task": pd.Series(
                [row["task"] for row in development]
            )
            .value_counts()
            .sort_index()
            .to_dict(),
            "confirmation_by_task": pd.Series(
                [row["task"] for row in confirmation]
            )
            .value_counts()
            .sort_index()
            .to_dict(),
            "fold_counts": pd.Series([row["fold"] for row in development])
            .value_counts()
            .sort_index()
            .to_dict(),
            "development_manifest_sha256": sha256_file(
                output_paths["development"]
            ),
            "confirmation_manifest_sha256": sha256_file(
                output_paths["confirmation"]
            ),
        },
        "pass": True,
    }
    write_json(output_paths["audit"], audit)
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
