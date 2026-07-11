# pyright: reportMissingImports=false

"""Train a frozen feature-only irreversible pre-compression selector."""

import argparse
import hashlib
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, "src")

from herald.precompression_selector import (  # noqa: E402
    deterministic_selector_split,
    fit_selector_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path(
            "results/expected_stats_predictor_full_s0_v2/switch_dataset.parquet"
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "results/expected_stats_alarm_bundle_full_s0_v2/fidelity_targets.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/expected_stats_precommit_selector_s0_v1"),
    )
    parser.add_argument("--num-boost-round", type=int, default=300)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_parquet(path)
    raw_rows = frame.to_dict(orient="records")
    rows: list[dict[str, Any]] = [dict(row) for row in raw_rows]
    if not rows:
        raise ValueError("selector source parquet is empty")
    compressors = {str(row["compressor"]) for row in rows}
    if compressors != {"expected_attention_stats"}:
        raise ValueError(
            f"unexpected selector compressors: {sorted(compressors)}"
        )
    ratios = {float(row["ratio"]) for row in rows}
    if ratios != {0.25}:
        raise ValueError(f"unexpected selector ratios: {sorted(ratios)}")
    keys = [
        (
            str(row["prompt_id"]),
            int(row["s"]),
            str(row["compressor"]),
            float(row["ratio"]),
        )
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("selector source contains duplicate switch cells")
    return rows


def main() -> None:
    args = parse_args()
    rows = load_rows(args.parquet)
    targets = json.loads(args.targets.read_text())
    target_ids = list(
        targets["compressors"]["expected_attention_stats"]["test_prompt_ids"]
    )
    split = deterministic_selector_split(
        [str(row["prompt_id"]) for row in rows], target_ids
    )
    train_set = set(split.train_prompt_ids)
    calibration_set = set(split.calibration_prompt_ids)
    train_rows = [row for row in rows if str(row["prompt_id"]) in train_set]
    calibration_rows = [
        row for row in rows if str(row["prompt_id"]) in calibration_set
    ]
    meta = {
        "source_provenance": {
            "parquet_sha256": sha256(args.parquet),
            "targets_sha256": sha256(args.targets),
            "sweep_config_sha256": targets["source_provenance"][
                "sweep_config_sha256"
            ],
        },
        "split_rule": (
            "sorted non-target prompt IDs; every fifth is calibration"
        ),
        "train_prompt_ids": split.train_prompt_ids,
        "calibration_prompt_ids": split.calibration_prompt_ids,
        "target_prompt_ids": split.target_prompt_ids,
    }
    bundle = fit_selector_bundle(
        train_rows,
        calibration_rows,
        compressor="expected_attention_stats",
        meta=meta,
        num_boost_round=args.num_boost_round,
    )
    temporary = args.output_dir.with_name(
        f".{args.output_dir.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle.save(temporary)
        if args.output_dir.exists():
            raise FileExistsError(
                f"selector output already exists: {args.output_dir}"
            )
        temporary.rename(args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "threshold": bundle.g_tau,
                "calibration": bundle.meta["calibration"],
                "n_train_prompts": len(split.train_prompt_ids),
                "n_calibration_prompts": len(split.calibration_prompt_ids),
                "n_target_prompts": len(split.target_prompt_ids),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
