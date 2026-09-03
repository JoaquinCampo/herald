"""Run the one locked M3 development attempt."""

import argparse
from pathlib import Path

from herald.magnitude_v2 import DEFAULT_LOCK
from herald.magnitude_v3 import (
    expected_m3_freeze_payload,
    fit_development,
    join_m3_features,
    load_band_sidecar,
    load_base_rows,
    load_frozen_m2,
    sha256_file,
    validate_m3_freeze,
    validate_m3_protocol_lock,
    write_oof_parquet,
    write_report,
)
from herald.sensor_replay_m3 import (
    m3_implementation_paths,
    validate_m3_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--m2-report", type=Path, required=True)
    parser.add_argument("--m2-oof", type=Path, required=True)
    parser.add_argument("--m2-freeze", type=Path, required=True)
    parser.add_argument("--m2-result-freeze", type=Path, required=True)
    parser.add_argument("--sensor-lock", type=Path, required=True)
    parser.add_argument("--bands-root", type=Path, required=True)
    parser.add_argument("--bands-manifest", type=Path, required=True)
    parser.add_argument("--bands-lock", type=Path, required=True)
    parser.add_argument("--reference-results-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    return parser.parse_args()


def validate_output_paths(args: argparse.Namespace) -> None:
    """Keep M3 outputs disjoint from every frozen or replay input."""
    outputs = {
        "report": args.report.resolve(),
        "oof": args.oof.resolve(),
    }
    if len(set(outputs.values())) != len(outputs):
        raise ValueError("--report and --oof must resolve to distinct paths")
    frozen_inputs = {
        getattr(args, name).resolve()
        for name in (
            "parquet",
            "evidence",
            "lock",
            "protocol_lock",
            "freeze",
            "m2_report",
            "m2_oof",
            "m2_freeze",
            "m2_result_freeze",
            "sensor_lock",
            "bands_manifest",
            "bands_lock",
        )
    }
    frozen_inputs.add(
        (
            args.bands_root / "sensor_sidecars" / "llama__ifeval.jsonl"
        ).resolve()
    )
    protected_roots = (
        args.reference_results_root.resolve(),
        args.bands_root.resolve(),
    )
    for name, path in outputs.items():
        if path in frozen_inputs or any(
            path.is_relative_to(root) for root in protected_roots
        ):
            raise ValueError(f"M3 {name} path overlaps a frozen input")


def main() -> None:
    args = parse_args()
    validate_output_paths(args)
    if args.protocol_lock.resolve() != args.bands_lock.resolve():
        raise ValueError(
            "--protocol-lock and --bands-lock must resolve identically"
        )
    lock = validate_m3_protocol_lock(args.protocol_lock)
    validate_m3_lock(
        args.protocol_lock,
        args.evidence,
        args.lock,
        args.sensor_lock,
        args.m2_result_freeze,
    )
    base, base_provenance = load_base_rows(
        args.parquet,
        evidence_path=args.evidence,
        lock_path=args.lock,
    )
    m2, m2_provenance = load_frozen_m2(
        args.m2_report,
        args.m2_oof,
        args.m2_freeze,
        result_freeze_path=args.m2_result_freeze,
        base_rows=base,
        base_provenance=base_provenance,
        evidence_path=args.evidence,
        protocol_lock_path=args.lock,
        sensor_lock_path=args.sensor_lock,
    )
    band_hash = sha256_file(args.bands_lock)
    bands, band_provenance = load_band_sidecar(
        args.bands_root,
        args.bands_manifest,
        expected_keys={
            (
                str(row["prompt_id"]),
                str(row["compressor"]),
                float(row["ratio"]),
                int(row["s"]),
            )
            for row in base
        },
        evidence_sha256=sha256_file(args.evidence),
        lock_sha256=sha256_file(args.lock),
        sensor_lock_sha256=band_hash,
        band_lock_path=args.bands_lock,
        reference_results_root=args.reference_results_root,
        evidence_path=args.evidence,
        reference_manifest_sha256=m2_provenance.get(
            "reference_manifest_sha256"
        ),
        reference_inputs_sha256=m2_provenance.get("reference_inputs_sha256"),
    )
    if band_provenance.get("reference_manifest_sha256") != m2_provenance.get(
        "reference_manifest_sha256"
    ) or band_provenance.get("reference_inputs_sha256") != m2_provenance.get(
        "reference_inputs_sha256"
    ):
        raise ValueError("band references differ from frozen M2 references")
    expected = expected_m3_freeze_payload(
        base_parquet_path=args.parquet,
        m2_report_path=args.m2_report,
        m2_oof_path=args.m2_oof,
        m2_prefit_freeze_path=args.m2_freeze,
        m2_result_freeze_path=args.m2_result_freeze,
        m2_sensor_lock_path=args.sensor_lock,
        band_manifest_path=args.bands_manifest,
        band_sidecar_path=args.bands_root
        / "sensor_sidecars"
        / "llama__ifeval.jsonl",
        evidence_path=args.evidence,
        protocol_lock_path=args.protocol_lock,
        reference_manifest_sha256=str(
            m2_provenance.get("reference_manifest_sha256", "")
        ),
        reference_inputs_sha256=str(
            m2_provenance.get("reference_inputs_sha256", "")
        ),
        implementation_paths=m3_implementation_paths(),
    )
    validate_m3_freeze(args.freeze, expected)
    joined = join_m3_features(base, m2, bands)
    report, oof = fit_development(
        joined,
        lock=lock,
        provenance=base_provenance,
        m2_provenance=m2_provenance,
    )
    freeze_sha = sha256_file(args.freeze)
    report["m3_freeze_sha256"] = freeze_sha
    report["band_provenance"] = band_provenance
    for row in oof:
        row["m3_freeze_sha256"] = freeze_sha
    write_oof_parquet(oof, args.oof)
    report["oof_sha256"] = sha256_file(args.oof)
    write_report(report, args.report)
    print(f"wrote {args.report}")
    print(f"wrote {args.oof}")


if __name__ == "__main__":
    main()
