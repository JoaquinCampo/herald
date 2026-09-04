"""Audit materialized quality-risk features: roster, causality, labels."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds
from audit_current_state_damage_features import (
    history_columns as v1_history_columns,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herald.quality_risk_features import (  # noqa: E402
    EXPECTED_COLUMNS_V3,
    INSTANTANEOUS_SENSORS,
    V1_PRED_COLUMNS,
    check_columns,
    history_columns,
    verify_causal_sample,
    verify_divergence_sample,
    verify_engineered_sample,
)
from herald.quality_risk_labels import (  # noqa: E402
    CATASTROPHE_COLUMNS,
    SCORE_COLUMNS,
    apply_run_labels,
)

SCHEMA_VERSION = "herald.quality_risk_feature_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--v1-oof", type=Path, default=None)
    parser.add_argument("--v1-tabular-report", type=Path, default=None)
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


def audit_press(
    source: Path,
    materialized: Path,
    audit: dict[str, Any],
    v1_oof: Path | None,
    v1_scales: list[float] | None,
    expect_v3: bool,
) -> dict[str, Any]:
    produced = (
        ds.dataset(materialized, format="parquet")  # type: ignore[no-untyped-call]
        .to_table()
        .to_pandas()
    )
    if expect_v3:
        check_columns(list(produced.columns), EXPECTED_COLUMNS_V3)
    else:
        check_columns(list(produced.columns))
    run_table = produced[["run_id"]].drop_duplicates()
    run_id = sorted(run_table["run_id"].tolist())[len(run_table) // 2]
    sample = produced[produced["run_id"] == run_id].copy()
    raw = (
        ds.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
        .to_table(
            columns=["run_id", "token_pos", *INSTANTANEOUS_SENSORS],
            filter=ds.field("run_id") == run_id,  # type: ignore[attr-defined,no-untyped-call]
        )
        .to_pandas()
    )
    if len(raw) != len(sample):
        raise ValueError(f"sample row count mismatch in {source.name}")
    causal_error = verify_causal_sample(sample, raw)
    engineered_error = verify_engineered_sample(sample, raw)
    divergence_error = 0.0
    if v1_oof is not None:
        assert v1_scales is not None
        v1_sample = (
            ds.dataset(v1_oof, format="parquet")  # type: ignore[no-untyped-call]
            .to_table(
                columns=["run_id", "token_pos", *V1_PRED_COLUMNS],
                filter=ds.field("run_id") == run_id,  # type: ignore[attr-defined,no-untyped-call]
            )
            .to_pandas()
        )
        divergence_error = verify_divergence_sample(
            sample, v1_sample, v1_scales
        )
    score_table = (
        ds.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
        .to_table(
            columns=["run_id", "prompt_id", *SCORE_COLUMNS],
            filter=ds.field("run_id") == run_id,  # type: ignore[attr-defined,no-untyped-call]
        )
        .to_pandas()
        .groupby("run_id")
        .first()
        .reset_index()
    )
    relabeled = apply_run_labels(
        score_table,
        gap_prompts=audit["excluded_gap_prompts"],
        lift_prompts=audit["lift_reference_zero_prompts"],
    )
    expected_damage = int(relabeled["damage"].iloc[0])
    if set(sample["damage"].unique()) != {expected_damage}:
        raise ValueError(f"label mismatch on sample run in {source.name}")
    catastrophe_table = (
        ds.dataset(source, format="parquet")  # type: ignore[no-untyped-call]
        .to_table(
            columns=["run_id", *CATASTROPHE_COLUMNS],
            filter=ds.field("run_id") == run_id,  # type: ignore[attr-defined,no-untyped-call]
        )
        .to_pandas()
    )
    expected_catastrophe = int(
        catastrophe_table[list(CATASTROPHE_COLUMNS)]
        .fillna(False)
        .any(axis=True)
        .max()
    )
    if set(sample["catastrophe"].unique()) != {expected_catastrophe}:
        raise ValueError(
            f"catastrophe mismatch on sample run in {source.name}"
        )
    return {
        "rows": len(produced),
        "runs": int(produced["run_id"].nunique()),
        "sample_run_id": run_id,
        "sample_causal_max_error": causal_error,
        "sample_engineered_max_error": engineered_error,
        "sample_divergence_max_error": divergence_error,
        "sample_damage": expected_damage,
        "sample_catastrophe": expected_catastrophe,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if tuple(history_columns()) != tuple(v1_history_columns()):
        raise ValueError("causal history roster drifted from v1")
    lock = load_json(args.protocol_lock)
    audit = load_json(args.label_audit)
    manifest = load_json(args.materialized_root / "manifest.json")
    if lock.get("schema_version") != "herald.quality_risk.v1":
        raise ValueError("unexpected protocol schema")
    if audit.get("pass") is not True:
        raise ValueError("label audit did not pass")
    if manifest.get("protocol_lock_sha256") != sha256_file(
        args.protocol_lock
    ):
        raise ValueError("materialization belongs to another protocol")
    if manifest.get("label_audit_sha256") != sha256_file(args.label_audit):
        raise ValueError("materialization label audit mismatch")
    expect_v3 = args.v1_oof is not None
    if expect_v3 != (args.v1_tabular_report is not None):
        raise ValueError("v1 OOF and v1 tabular report go together")
    v1_scales: list[float] | None = None
    if expect_v3:
        assert args.v1_oof is not None and args.v1_tabular_report is not None
        v1_scales = [
            float(value)
            for value in load_json(args.v1_tabular_report)[
                "calibration_scales"
            ]
        ]
    by_press = {}
    for source in sorted((args.dataset_root / "tokens").glob("*.parquet")):
        materialized = args.materialized_root / source.name
        by_press[source.stem] = audit_press(
            source, materialized, audit, args.v1_oof, v1_scales, expect_v3
        )
    if set(by_press) != set(lock["dataset"]["presses"]):
        raise ValueError("press roster changed")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_features_audited_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "label_audit_sha256": sha256_file(args.label_audit),
        "materialization_manifest_sha256": sha256_file(
            args.materialized_root / "manifest.json"
        ),
        "history_roster_matches_v1": True,
        "confirmation_prompts_projected": 0,
        "by_press": by_press,
        "pass": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
