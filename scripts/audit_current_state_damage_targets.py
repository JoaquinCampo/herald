"""Audit development-only cumulative-JS labels and horizon censoring."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

SCHEMA_VERSION = "herald.current_state_damage_target_audit.v1"
HORIZONS = (5, 10, 25, 50)
TARGET_COLUMNS = tuple(f"future_sum_js_{horizon}" for horizon in HORIZONS)
ALIGNMENT_RTOL = 2e-5
ALIGNMENT_ATOL = 2e-4
PROJECTED_COLUMNS = (
    "run_id",
    "token_pos",
    "prompt_id",
    "js_full",
    *TARGET_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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
) -> tuple[dict[str, Any], set[str]]:
    lock = load_json(args.protocol_lock)
    metadata = load_json(args.metadata_audit)
    development = load_json(args.development_manifest)
    if (
        lock.get("schema_version")
        != "herald.current_state_compression_damage.v1"
    ):
        raise ValueError("unexpected protocol schema")
    if metadata.get("pass") is not True:
        raise ValueError("metadata audit did not pass")
    if metadata.get("protocol_lock_sha256") != sha256_file(
        args.protocol_lock
    ):
        raise ValueError("metadata audit belongs to another protocol")
    if metadata.get("target_columns_loaded") != []:
        raise ValueError("metadata quarantine read target columns")
    expected_manifest_hash = metadata["split"]["development_manifest_sha256"]
    if sha256_file(args.development_manifest) != expected_manifest_hash:
        raise ValueError("development manifest hash mismatch")
    if not isinstance(development, list) or not development:
        raise ValueError("development manifest is empty")
    prompt_ids = {row["prompt_id"] for row in development}
    if len(prompt_ids) != len(development):
        raise ValueError("development manifest has duplicate prompts")
    if len(prompt_ids) != metadata["split"]["development_prompts"]:
        raise ValueError("development manifest count mismatch")
    if tuple(lock["dataset"]["horizons"]) != HORIZONS:
        raise ValueError("protocol horizon roster changed")
    if tuple(lock["targets"]["primary_columns"]) != TARGET_COLUMNS:
        raise ValueError("protocol target roster changed")
    return lock, prompt_ids


def forward_sum(values: np.ndarray, horizon: int) -> np.ndarray:
    prefix = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(values, dtype=np.float64)]
    )
    return (  # type: ignore[no-any-return]
        prefix[horizon + 1 :] - prefix[1 : len(values) - horizon + 1]
    )


def audit_press(path: Path, development: set[str]) -> dict[str, Any]:
    dataset = ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]
    prompt_filter = ds.field("prompt_id").isin(  # type: ignore[attr-defined,no-untyped-call]
        sorted(development)
    )
    table = dataset.to_table(
        columns=list(PROJECTED_COLUMNS),
        filter=prompt_filter,
    )
    frame = table.to_pandas()
    if not set(frame["prompt_id"].unique()) <= development:
        raise ValueError(f"confirmation prompt projected from {path.name}")
    if not set(development) <= set(frame["prompt_id"].unique()):
        raise ValueError(f"development prompt missing from {path.name}")

    target_matrix = frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
    if np.isinf(target_matrix).any() or (target_matrix < 0).any():
        raise ValueError(f"invalid cumulative-JS target in {path.name}")

    eligible_counts = {str(horizon): 0 for horizon in HORIZONS}
    censored_non_null_counts = {str(horizon): 0 for horizon in HORIZONS}
    maximum_alignment_error = {str(horizon): 0.0 for horizon in HORIZONS}
    monotonic_violations = 0
    run_count = 0
    for _, run in frame.groupby("run_id", sort=False):
        run = run.sort_values("token_pos")
        positions = run["token_pos"].to_numpy(dtype=np.int64)
        if not np.array_equal(positions, np.arange(len(run), dtype=np.int64)):
            raise ValueError(f"noncontiguous development run in {path.name}")
        js = run["js_full"].to_numpy(dtype=np.float64)
        if not np.isfinite(js).all() or (js < 0).any():
            raise ValueError(f"invalid JS oracle in {path.name}")
        targets = run[list(TARGET_COLUMNS)].to_numpy(dtype=np.float64)
        if len(run) > 50:
            common = targets[: len(run) - 50]
            monotonic_violations += int(
                (np.diff(common, axis=1) < -1e-7).any(axis=1).sum()
            )
        for column_index, horizon in enumerate(HORIZONS):
            eligible = max(len(run) - horizon, 0)
            eligible_counts[str(horizon)] += eligible
            if eligible:
                expected = forward_sum(js, horizon)
                actual = targets[:eligible, column_index]
                if not np.isfinite(actual).all():
                    raise ValueError(
                        f"eligible future_sum_js_{horizon} is null "
                        f"in {path.name}"
                    )
                errors = np.abs(expected - actual)
                maximum_alignment_error[str(horizon)] = max(
                    maximum_alignment_error[str(horizon)],
                    float(errors.max(initial=0.0)),
                )
                if not np.allclose(
                    actual,
                    expected,
                    rtol=ALIGNMENT_RTOL,
                    atol=ALIGNMENT_ATOL,
                ):
                    raise ValueError(
                        f"future_sum_js_{horizon} alignment mismatch "
                        f"in {path.name}"
                    )
            censored = targets[eligible:, column_index]
            non_null = int((~np.isnan(censored)).sum())
            censored_non_null_counts[str(horizon)] += non_null
            if non_null:
                raise ValueError(
                    f"right-censored future_sum_js_{horizon} is populated "
                    f"in {path.name}"
                )
        run_count += 1
    if monotonic_violations:
        raise ValueError(f"nonmonotone cumulative-JS labels in {path.name}")
    return {
        "rows_projected": len(frame),
        "runs": run_count,
        "prompts": int(frame["prompt_id"].nunique()),
        "eligible_rows": eligible_counts,
        "right_censored_rows_with_non_null_target": censored_non_null_counts,
        "maximum_alignment_error": maximum_alignment_error,
        "eligible_finite_nonnegative": True,
        "common_support_monotone": True,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock, development = validate_inputs(args)
    by_press = {
        path.stem: audit_press(path, development)
        for path in sorted((args.dataset_root / "tokens").glob("*.parquet"))
    }
    if set(by_press) != set(lock["dataset"]["presses"]):
        raise ValueError("token press roster changed")
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_targets_audited_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "metadata_audit_sha256": sha256_file(args.metadata_audit),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "columns_projected": list(PROJECTED_COLUMNS),
        "alignment_tolerance": {
            "absolute": ALIGNMENT_ATOL,
            "relative": ALIGNMENT_RTOL,
        },
        "confirmation_prompts_projected": 0,
        "future_sum_kl_loaded": False,
        "future_max_js_loaded": False,
        "final_outcomes_loaded": False,
        "by_press": by_press,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
