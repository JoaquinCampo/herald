"""Audit development-only quality-risk damage labels (post lock)."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from herald.quality_risk_labels import (  # noqa: E402
    CATASTROPHE_COLUMNS,
    SCORE_COLUMNS,
    audit_development_labels,
)

SCHEMA_VERSION = "herald.quality_risk_label_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = load_json(args.protocol_lock)
    if lock.get("schema_version") != "herald.quality_risk.v1":
        raise ValueError("unexpected protocol schema")
    if lock.get("status") != "locked_before_first_label_read":
        raise ValueError("protocol was not locked before label reads")
    development = load_json(args.development_manifest)
    confirmation = load_json(args.confirmation_manifest)
    report = audit_development_labels(
        dataset_root=args.dataset_root,
        development=[str(row["prompt_id"]) for row in development],
        confirmation=[str(row["prompt_id"]) for row in confirmation],
        presses=list(lock["dataset"]["presses"]),
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_labels_audited_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "development_manifest_sha256": sha256_file(args.development_manifest),
        "confirmation_manifest_sha256": sha256_file(
            args.confirmation_manifest
        ),
        "score_columns": list(SCORE_COLUMNS),
        "catastrophe_columns": list(CATASTROPHE_COLUMNS),
        **report,
    }
    if report.get("pass") is not True:
        audit["pass"] = False
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        raise ValueError(f"label audit failed: {report.get('reason')}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
