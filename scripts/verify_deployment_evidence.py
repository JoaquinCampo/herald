"""Verify the local files named by a deployment evidence manifest."""

import argparse
import json
from pathlib import Path
from typing import Any

from herald.deployment_evidence import (  # pyright: ignore[reportMissingImports]
    verify_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("docs/implementation/deployment_evidence_manifest.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("evidence manifest must be a JSON object")
    return payload


def main() -> None:
    args = parse_args()
    try:
        errors = verify_manifest(load_manifest(args.manifest), args.root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid evidence manifest: {error}") from error

    if errors:
        message = "evidence verification failed:\n" + "\n".join(errors)
        raise SystemExit(message)
    print(f"verified {args.manifest}")


if __name__ == "__main__":
    main()
