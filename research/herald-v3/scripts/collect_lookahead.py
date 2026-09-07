#!/usr/bin/env python3
"""Collect the locked outcome-free H8 lookahead probe ledger."""

# The import intentionally follows the local source-path bootstrap.
# ruff: noqa: E402, I001

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from herald_v3.engineering import runner
from herald_v3.engineering.lookahead_collection import (
    LookaheadCollectionError,
    collect_lookahead,
)
from herald_v3.engineering.prompts import load_prompt_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--protocol-lock", required=True, type=Path)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--train-reuse", type=Path)
    args = parser.parse_args(argv)

    if args.phase == "test" and args.train_reuse is not None:
        parser.error("--train-reuse is only valid for --phase train")
    if args.phase == "train" and args.train_reuse is None:
        parser.error("--train-reuse is required for --phase train")
    if re.fullmatch(r"[0-9a-f]{64}", args.expected_lock_sha256) is None:
        parser.error("--expected-lock-sha256 must be 64 lowercase hex digits")
    try:
        observed_lock_sha256 = hashlib.sha256(
            args.protocol_lock.read_bytes()
        ).hexdigest()
    except OSError as error:
        parser.error(f"cannot read --protocol-lock: {error}")
    if observed_lock_sha256 != args.expected_lock_sha256:
        parser.error("protocol lock does not match --expected-lock-sha256")
    if not runner._torch().cuda.is_available():
        parser.error("CUDA is required before loading the model")

    expected_count = 120 if args.phase == "train" else 76
    try:
        manifest = load_prompt_manifest(args.prompts, limit=expected_count)
        model, tokenizer = runner.load_offline_model(args.model)
        summary = collect_lookahead(
            model,
            tokenizer,
            manifest,
            args.prompts,
            args.protocol_lock,
            args.output,
            phase=args.phase,
            train_reuse=args.train_reuse,
        )
    except (
        LookaheadCollectionError,
        OSError,
        TypeError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
