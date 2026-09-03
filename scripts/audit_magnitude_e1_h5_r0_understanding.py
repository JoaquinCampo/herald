"""Audit whether the historical AttentionTap sweep can seed H5-R0."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from herald.attention_features import tap_feature_names
from herald.features import FEATURE_NAMES
from herald.magnitude_v2 import hash_prompt_ids, sha256_file

SCHEMA_VERSION = "herald.magnitude_e1_h5_r0_understanding_audit.v1"
EXPECTED_HASHES = {
    "source_oof_sha256": (
        "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
    ),
    "legacy_config_sha256": (
        "8c228094df9f36cd582752070d578fbfe60f49cd53553b3b507076d17c874408"
    ),
    "legacy_manifest_sha256": (
        "5f3d65d72ff506cfdd8105dd4a26cb065c2234c4ede3263bacf0547b1dc5cf88"
    ),
    "historical_tap_config_sha256": (
        "da58d41702307002fde423feaac62fc33564f1e10e0cb9128d5cc343b0db7b2f"
    ),
    "legacy_prefix_sidecar_sha256": (
        "63de212f19352f16c0c1fbfe78d98d59ffe3d1d284dfad56e52889a23663183c"
    ),
    "legacy_prefix_manifest_sha256": (
        "f433ddd18eb2f0b0d4017a05374be6e5eba50b8134f5dd15142ac28ad57e9ebc"
    ),
    "historical_tap_log_sha256": (
        "4eff1341f29d01c3578bcc6b7d0cab3ddec5ae27a2b5741a93963cc9e0b47d02"
    ),
    "attention_source_sha256": (
        "5233b9bc13afcee585c3ba70df27240da9ad5bd4d4746141243404ac0e157c39"
    ),
    "generate_source_sha256": (
        "0c5f43e87ed4c508b12f977a2ca117441acce244700c592218f742daa833de1f"
    ),
    "ifeval_source_sha256": (
        "6a0901d7209664409c15b7e1c3329b92caf00ae2ccdab82efabb61f1b0767606"
    ),
}
EXPECTED_PROMPTS = 154
EXPECTED_RAW_WIDTH = 20
EXPECTED_TAP_WIDTH = 16
ROSTER_NAMESPACE = "H5-R0\0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--legacy-config", type=Path, required=True)
    parser.add_argument("--legacy-manifest", type=Path, required=True)
    parser.add_argument("--legacy-reference-dir", type=Path, required=True)
    parser.add_argument("--historical-tap-config", type=Path, required=True)
    parser.add_argument(
        "--historical-tap-reference-dir", type=Path, required=True
    )
    parser.add_argument("--legacy-prefix-sidecar", type=Path, required=True)
    parser.add_argument("--legacy-prefix-manifest", type=Path, required=True)
    parser.add_argument("--historical-tap-log", type=Path, required=True)
    parser.add_argument("--attention-source", type=Path, required=True)
    parser.add_argument("--generate-source", type=Path, required=True)
    parser.add_argument("--ifeval-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def verify_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths = {
        "source_oof_sha256": args.source_oof,
        "legacy_config_sha256": args.legacy_config,
        "legacy_manifest_sha256": args.legacy_manifest,
        "historical_tap_config_sha256": args.historical_tap_config,
        "legacy_prefix_sidecar_sha256": args.legacy_prefix_sidecar,
        "legacy_prefix_manifest_sha256": args.legacy_prefix_manifest,
        "historical_tap_log_sha256": args.historical_tap_log,
        "attention_source_sha256": args.attention_source,
        "generate_source_sha256": args.generate_source,
        "ifeval_source_sha256": args.ifeval_source,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    if actual != EXPECTED_HASHES:
        differences = {
            name: {"expected": EXPECTED_HASHES[name], "actual": value}
            for name, value in actual.items()
            if value != EXPECTED_HASHES[name]
        }
        raise ValueError(f"H5-R0 understanding input changed: {differences}")
    return actual


def development_roster(
    path: Path,
) -> tuple[tuple[str, ...], dict[str, int], tuple[str, ...]]:
    table = pq.read_table(path, columns=["prompt_id", "fold"])  # type: ignore[no-untyped-call]
    if table.column_names != ["prompt_id", "fold"]:
        raise ValueError("unexpected OOF column was materialized")
    frame = table.to_pandas().drop_duplicates()
    if int((frame.groupby("prompt_id")["fold"].nunique() != 1).sum()):
        raise ValueError("prompt fold is not unique")
    unique = frame.drop_duplicates("prompt_id")
    prompt_folds = {
        str(row.prompt_id): int(row.fold)
        for row in unique.itertuples(index=False)
    }
    prompts = tuple(sorted(prompt_folds))
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError("development prompt count changed")
    pilot = tuple(
        min(
            (prompt for prompt in prompts if prompt_folds[prompt] == fold),
            key=lambda prompt: hashlib.sha256(
                f"{ROSTER_NAMESPACE}{prompt}".encode()
            ).hexdigest(),
        )
        for fold in range(5)
    )
    return prompts, prompt_folds, pilot


def array_pair_audit(
    prompts: tuple[str, ...],
    pilot: tuple[str, ...],
    legacy_dir: Path,
    tapped_dir: Path,
) -> dict[str, Any]:
    row_mismatches: list[str] = []
    legacy_column_mismatches: list[str] = []
    tap_nonfinite: list[str] = []
    tap_constant: list[str] = []
    missing: list[str] = []
    max_first20_abs_difference = 0.0
    compared_equal_shape = 0
    artifact_digest = hashlib.sha256()
    details: dict[str, dict[str, Any]] = {}
    legacy_rows = 0
    tapped_rows = 0
    for prompt_id in prompts:
        legacy_path = legacy_dir / f"{prompt_id}.npy"
        tapped_path = tapped_dir / f"{prompt_id}.npy"
        if not legacy_path.is_file() or not tapped_path.is_file():
            missing.append(prompt_id)
            continue
        legacy_sha = sha256_file(legacy_path)
        tapped_sha = sha256_file(tapped_path)
        artifact_digest.update(
            f"{prompt_id}\0{legacy_sha}\0{tapped_sha}\n".encode()
        )
        legacy = np.load(legacy_path, allow_pickle=False)
        tapped = np.load(tapped_path, allow_pickle=False)
        if (
            legacy.ndim != 2
            or legacy.shape[1] != EXPECTED_RAW_WIDTH
            or legacy.dtype != np.float16
        ):
            raise ValueError(f"legacy array schema changed: {prompt_id}")
        if (
            tapped.ndim != 2
            or tapped.shape[1] != EXPECTED_RAW_WIDTH + EXPECTED_TAP_WIDTH
            or tapped.dtype != np.float16
        ):
            raise ValueError(f"historical tap schema changed: {prompt_id}")
        legacy_rows += int(legacy.shape[0])
        tapped_rows += int(tapped.shape[0])
        rows_equal = legacy.shape[0] == tapped.shape[0]
        if not rows_equal:
            row_mismatches.append(prompt_id)
        else:
            compared_equal_shape += 1
            raw_tapped = tapped[:, :EXPECTED_RAW_WIDTH]
            raw_exact = np.array_equal(legacy, raw_tapped, equal_nan=True)
            if not raw_exact:
                legacy_column_mismatches.append(prompt_id)
            finite = np.isfinite(legacy) & np.isfinite(raw_tapped)
            if np.any(finite):
                max_first20_abs_difference = max(
                    max_first20_abs_difference,
                    float(
                        np.max(np.abs(legacy[finite] - raw_tapped[finite]))
                    ),
                )
        tap = tapped[:, EXPECTED_RAW_WIDTH:]
        all_tap_finite = bool(np.isfinite(tap).all())
        if not all_tap_finite:
            tap_nonfinite.append(prompt_id)
        nonconstant_columns = int(
            sum(
                np.unique(tap[:, index]).size > 1
                for index in range(tap.shape[1])
            )
        )
        if nonconstant_columns == 0:
            tap_constant.append(prompt_id)
        if prompt_id in pilot:
            details[prompt_id] = {
                "fold": pilot.index(prompt_id),
                "legacy_shape": list(legacy.shape),
                "historical_tap_shape": list(tapped.shape),
                "row_count_equal": rows_equal,
                "tap_all_finite": all_tap_finite,
                "tap_nonconstant_columns": nonconstant_columns,
            }
    mismatch_hash = hash_prompt_ids(row_mismatches)
    raw_mismatch_hash = hash_prompt_ids(legacy_column_mismatches)
    reusable = (
        not missing
        and not row_mismatches
        and not legacy_column_mismatches
        and not tap_nonfinite
        and not tap_constant
    )
    return {
        "development_arrays_compared": EXPECTED_PROMPTS - len(missing),
        "legacy_rows": legacy_rows,
        "historical_tap_rows": tapped_rows,
        "equal_shape_prompts": compared_equal_shape,
        "row_count_mismatch_count": len(row_mismatches),
        "row_count_mismatch_ids_sha256": mismatch_hash,
        "first20_bit_mismatch_count_among_equal_shapes": len(
            legacy_column_mismatches
        ),
        "first20_bit_mismatch_ids_sha256": raw_mismatch_hash,
        "max_first20_abs_difference_among_equal_shapes": (
            max_first20_abs_difference
        ),
        "tap_nonfinite_prompt_count": len(tap_nonfinite),
        "tap_all_constant_prompt_count": len(tap_constant),
        "missing_prompt_count": len(missing),
        "paired_array_aggregate_sha256": artifact_digest.hexdigest(),
        "pilot_prompt_details": details,
        "historical_tap_reusable": reusable,
    }


def prefix_sidecar_audit(
    path: Path, pilot: tuple[str, ...]
) -> dict[str, Any]:
    selected = set(pilot)
    positions: dict[str, dict[int, str]] = {prompt: {} for prompt in pilot}
    disagreements = 0
    records = 0
    with path.open() as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            prompt_id = str(record.get("prompt_id", ""))
            if prompt_id not in selected:
                continue
            position = int(record["s"])
            prefix_hash = str(record["prefix_hash"])
            prior = positions[prompt_id].setdefault(position, prefix_hash)
            disagreements += int(prior != prefix_hash)
            records += 1
    return {
        "pilot_records": records,
        "prefix_hash_disagreements": disagreements,
        "positions_per_prompt": {
            prompt: len(values) for prompt, values in positions.items()
        },
        "all_pilot_prompts_covered": all(positions.values()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H5-R0 understanding audit: {args.output}"
        )
    hashes = verify_hashes(args)
    legacy_config = load_json(args.legacy_config)
    tapped_config = load_json(args.historical_tap_config)
    prompts, prompt_folds, pilot = development_roster(args.source_oof)
    arrays = array_pair_audit(
        prompts,
        pilot,
        args.legacy_reference_dir,
        args.historical_tap_reference_dir,
    )
    prefixes = prefix_sidecar_audit(args.legacy_prefix_sidecar, pilot)
    checks = {
        "input_hashes_exact": hashes == EXPECTED_HASHES,
        "development_roster_exact": len(prompts) == EXPECTED_PROMPTS,
        "pilot_one_per_fold": tuple(prompt_folds[prompt] for prompt in pilot)
        == tuple(range(5)),
        "feature_widths_exact": len(FEATURE_NAMES) == EXPECTED_RAW_WIDTH
        and len(tap_feature_names()) == EXPECTED_TAP_WIDTH,
        "legacy_source_batch_one": legacy_config.get("ref_batch_size") == 1,
        "historical_tap_batch_twelve": tapped_config.get("ref_batch_size")
        == 12,
        "historical_tap_enabled": tapped_config.get("tap_attention") is True,
        "historical_reference_only": tapped_config.get("compressors") == [],
        "prefix_sidecar_consistent": prefixes["prefix_hash_disagreements"]
        == 0,
        "no_outcomes_loaded": True,
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "understood_before_h5_r0_strategy",
        "pass": all(checks.values()),
        "checks": checks,
        "provenance": {
            **hashes,
            "development_prompt_ids_sha256": hash_prompt_ids(prompts),
            "pilot_prompt_ids_sha256": hash_prompt_ids(pilot),
            "implementation_sha256": sha256_file(Path(__file__)),
        },
        "source_access": {
            "oof_columns_loaded": ["prompt_id", "fold"],
            "reference_npy_arrays_loaded": 2 * EXPECTED_PROMPTS,
            "reference_json_files_loaded": 0,
            "outcome_or_quality_columns_loaded": [],
            "protected_rows_materialized": False,
        },
        "pilot_roster": [
            {"fold": prompt_folds[prompt], "prompt_id": prompt}
            for prompt in pilot
        ],
        "historical_array_audit": arrays,
        "legacy_prefix_sidecar_audit": prefixes,
        "decision": (
            "reuse_historical_tap"
            if arrays["historical_tap_reusable"]
            else "run_new_batch1_paired_pilot"
        ),
        "interpretation": (
            "The historical batch-12 sweep is admissible only if every "
            "development continuation has the legacy row count and exact "
            "persisted first-20 features. Tap finiteness/nonconstancy alone "
            "cannot establish reference parity."
        ),
    }
    if not output["pass"]:
        raise ValueError(f"H5-R0 understanding checks failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
