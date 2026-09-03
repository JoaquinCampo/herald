"""Compare frozen M1 with sensor-augmented M2 on development prompts."""

from __future__ import annotations

import argparse
from pathlib import Path

from herald.magnitude_v2 import (
    load_development_rows,
    validate_lock_and_evidence,
)
from herald.magnitude_v2_sensors import (
    expected_m2_freeze_payload,
    fit_development_sensors,
    join_sensor_features,
    load_frozen_m1,
    load_sensor_sidecar,
    sha256_file,
    validate_m2_freeze,
    write_oof_parquet,
    write_report,
)
from herald.storage import sensor_sidecar_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_parquet", type=Path)
    parser.add_argument("--m1-report", type=Path, required=True)
    parser.add_argument("--m1-oof", type=Path, required=True)
    parser.add_argument("--m1-freeze", type=Path, required=True)
    parser.add_argument("--sensor-manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--reference-results-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--sensor-lock", type=Path, default=None)
    parser.add_argument("--m2-freeze", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, lock = validate_lock_and_evidence(args.evidence, args.lock)
    base_rows, base_provenance = load_development_rows(
        args.base_parquet,
        evidence_path=args.evidence,
        lock_path=args.lock,
    )
    base_keys = {
        (
            str(row["prompt_id"]),
            str(row["compressor"]),
            float(row["ratio"]),
            int(row["s"]),
        )
        for row in base_rows
    }
    sensors, sensor_provenance = load_sensor_sidecar(
        args.sensor_root,
        args.sensor_manifest,
        evidence_path=args.evidence,
        lock_path=args.lock,
        sensor_lock_path=args.sensor_lock,
        reference_results_root=args.reference_results_root,
        expected_keys=base_keys,
    )
    m1_rows, m1_provenance = load_frozen_m1(
        args.m1_report,
        args.m1_oof,
        args.m1_freeze,
        base_rows=base_rows,
        base_provenance=base_provenance,
        evidence_path=args.evidence,
        lock_path=args.lock,
        sensor_lock_path=args.sensor_lock,
    )
    resolved_sensor_lock = args.sensor_lock or args.lock.with_name(
        "magnitude_v2_sensor_lock.json"
    )
    freeze = expected_m2_freeze_payload(
        base_parquet_path=args.base_parquet,
        m1_freeze_path=args.m1_freeze,
        m1_report_path=args.m1_report,
        m1_oof_path=args.m1_oof,
        sensor_manifest_path=args.sensor_manifest,
        sensor_sidecar_path=sensor_sidecar_path(
            args.sensor_root, "llama", "ifeval"
        ),
        evidence_path=args.evidence,
        lock_path=args.lock,
        sensor_lock_path=resolved_sensor_lock,
        reference_manifest_sha256=str(
            sensor_provenance["reference_manifest_sha256"]
        ),
        reference_inputs_sha256=str(
            sensor_provenance["reference_inputs_sha256"]
        ),
    )
    frozen_m2 = validate_m2_freeze(args.m2_freeze, freeze)
    m2_freeze_sha = sha256_file(args.m2_freeze)
    joined = join_sensor_features(base_rows, sensors, m1_rows)
    report, predictions = fit_development_sensors(
        joined,
        m1_rows,
        lock=lock,
        base_provenance=base_provenance,
        sensor_provenance=sensor_provenance,
        m1_provenance=m1_provenance,
    )
    report["m2_freeze"] = {
        "sha256": m2_freeze_sha,
        "schema_version": frozen_m2["schema_version"],
        "status": frozen_m2["status"],
    }
    for prediction in predictions:
        prediction["m2_freeze_sha256"] = m2_freeze_sha
    write_oof_parquet(predictions, args.oof)
    report["oof_sha256"] = sha256_file(args.oof)
    write_report(report, args.report)
    print(f"wrote {args.report}")
    print(f"wrote {args.oof}")


if __name__ == "__main__":
    main()
