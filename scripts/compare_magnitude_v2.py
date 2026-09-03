"""Run the locked development-only M0/M1 magnitude comparison."""

import argparse
from pathlib import Path

from herald.magnitude_v2 import (
    DEFAULT_EVIDENCE,
    DEFAULT_LOCK,
    fit_development,
    load_development_rows,
    validate_lock_and_evidence,
    write_oof_parquet,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "results/recovered/ifeval-intervention-v1/"
            "magnitude_v2_report.json"
        ),
    )
    parser.add_argument(
        "--oof",
        type=Path,
        default=Path(
            "results/recovered/ifeval-intervention-v1/"
            "magnitude_v2_oof.parquet"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, lock = validate_lock_and_evidence(args.evidence, args.lock)
    rows, provenance = load_development_rows(
        args.parquet,
        evidence_path=args.evidence,
        lock_path=args.lock,
    )
    report, predictions = fit_development(
        rows,
        lock=lock,
        provenance=provenance,
    )
    write_report(report, args.report)
    write_oof_parquet(predictions, args.oof)
    print(f"wrote {args.report}")
    print(f"wrote {args.oof}")


if __name__ == "__main__":
    main()
