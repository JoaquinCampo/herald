"""Open, audit, and materialize the frozen confirmation prompt set once."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from audit_current_state_damage_targets import audit_press
from materialize_current_state_damage_dev import materialize_press

SCHEMA_VERSION = "herald.current_state_damage_confirmation_open.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--prefit-lock", type=Path, required=True)
    parser.add_argument("--confirmation-manifest", type=Path, required=True)
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


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reopen {args.output_root}")
    protocol = load_json(args.protocol_lock)
    prefit = load_json(args.prefit_lock)
    prompts = load_json(args.confirmation_manifest)
    if prefit.get("status") != "frozen_confirmation_open_condition_satisfied":
        raise ValueError("confirmation open condition is not frozen")
    if prefit.get("protocol_lock_sha256") != sha256_file(args.protocol_lock):
        raise ValueError("prefit protocol mismatch")
    if prefit["confirmation"]["prompt_manifest_sha256"] != sha256_file(
        args.confirmation_manifest
    ):
        raise ValueError("confirmation manifest mismatch")
    prompt_ids = {str(row["prompt_id"]) for row in prompts}
    if len(prompt_ids) != prefit["confirmation"]["prompt_count"]:
        raise ValueError("confirmation prompt count mismatch")
    args.output_root.mkdir(parents=True)
    target_audit = {
        path.stem: audit_press(path, prompt_ids)
        for path in sorted((args.dataset_root / "tokens").glob("*.parquet"))
    }
    if set(target_audit) != set(protocol["dataset"]["presses"]):
        raise ValueError("confirmation press roster changed")
    data_root = args.output_root / "tokens"
    data_root.mkdir()
    folds = dict.fromkeys(prompt_ids, -1)
    files = {
        path.stem: materialize_press(path, folds, data_root / path.name)
        for path in sorted((args.dataset_root / "tokens").glob("*.parquet"))
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "confirmation_opened_once_targets_audited_causal_materialized"
        ),
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "prefit_lock_sha256": sha256_file(args.prefit_lock),
        "confirmation_manifest_sha256": sha256_file(
            args.confirmation_manifest
        ),
        "prompt_count": len(prompt_ids),
        "target_audit": target_audit,
        "files": files,
        "released_aggregate_columns_loaded": [],
        "forbidden_outcome_columns_loaded": [],
        "secondary_targets_loaded": [],
        "pass": True,
    }
    (args.output_root / "open_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
