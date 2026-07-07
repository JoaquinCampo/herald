"""Small argparse CLI with config precedence for skill evaluation."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def load_config(path: Path) -> dict[str, Any]:
    """Load a JSON object from a config path."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON config: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("config must be a JSON object")
    return data


def normalize_config(config: dict[str, Any], *, env_prefix: str | None = None) -> dict[str, Any]:
    """Return a normalized config with optional environment prefix override."""
    normalized = {str(key).lower(): value for key, value in config.items()}
    if env_prefix is not None:
        normalized["env_prefix"] = env_prefix
    return normalized


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(prog="demo-config")
    parser.add_argument("config", type=Path)
    parser.add_argument("--env-prefix", default=os.getenv("DEMO_ENV_PREFIX"))
    return parser


def run(argv: list[str]) -> int:
    """Run the CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        normalized = normalize_config(config, env_prefix=args.env_prefix)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(normalized, sort_keys=True))
    return 0


def main() -> None:
    """CLI entry point."""
    raise SystemExit(run(sys.argv[1:]))
