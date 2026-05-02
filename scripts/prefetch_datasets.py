"""Explicit, auditable dataset prefetcher.

Phase 1 forbids implicit downloads inside experiment execution. This
script is the single sanctioned entry point for pulling benchmark
data onto a sweep host. It enumerates `herald.tasks.PHASE1_DATASET_SPECS`,
prints what it will do, optionally checks what is already cached, and
runs `datasets.load_dataset(...)` with the pinned coordinates.

Usage on Orion (proxy env required, see gold/orion-setup.md):

    .venv/bin/python scripts/prefetch_datasets.py --dry-run
    .venv/bin/python scripts/prefetch_datasets.py
    .venv/bin/python scripts/prefetch_datasets.py \
        --cache-dir /clustergpu/home/jcampo/.cache/huggingface

`--only` filters by repo-id substring for incremental runs.
`--list` prints the registry without touching the network.

Exit codes: 0 on success, 1 on partial failure (one or more datasets
failed; the rest still attempted), 2 on argument error.
"""

import argparse
import os
import sys
from pathlib import Path


def _format_spec(spec: object) -> str:
    trc = "  trust_remote_code=True" if spec.trust_remote_code else ""  # type: ignore[attr-defined]
    return (
        f"{spec.name}"  # type: ignore[attr-defined]
        f"  config={spec.config}"  # type: ignore[attr-defined]
        f"  split={spec.split}"  # type: ignore[attr-defined]
        f"  revision={spec.revision}"  # type: ignore[attr-defined]
        f"{trc}"
    )


def _print_env_warning_if_no_proxy() -> None:
    if "HTTPS_PROXY" not in os.environ and "https_proxy" not in os.environ:
        print(
            "WARNING: HTTPS_PROXY is not set. On Orion this download "
            "will fail (no direct internet). See gold/orion-setup.md.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prefetch Phase 1 benchmark datasets via the HF hub."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "HF cache root. Defaults to the HF library default "
            "(~/.cache/huggingface)."
        ),
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help=(
            "Substring filter on dataset name. Repeatable. "
            "Empty means all."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded; do not touch the network.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the dataset registry and exit.",
    )
    args = parser.parse_args()

    from herald.tasks import PHASE1_DATASET_SPECS

    specs = list(PHASE1_DATASET_SPECS)
    if args.only:
        specs = [
            s for s in specs if any(needle in s.name for needle in args.only)
        ]
    if not specs:
        print("No datasets matched the filter. Nothing to do.")
        return 0

    print("Phase 1 dataset registry:")
    for s in specs:
        print(f"  - {_format_spec(s)}")

    if args.list:
        return 0

    if args.dry_run:
        print()
        print("--dry-run: skipping all downloads.")
        return 0

    _print_env_warning_if_no_proxy()

    from datasets import load_dataset
    from loguru import logger

    failures: list[str] = []
    for s in specs:
        print()
        logger.info(f"Prefetching: {_format_spec(s)}")
        kwargs: dict[str, object] = {
            "split": s.split,
            "revision": s.revision,
        }
        if s.config is not None:
            kwargs["name"] = s.config
        if args.cache_dir is not None:
            kwargs["cache_dir"] = str(args.cache_dir)
        if s.trust_remote_code:
            kwargs["trust_remote_code"] = True
        try:
            ds = load_dataset(s.name, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"  FAILED: {exc!r}")
            failures.append(s.name)
            continue
        n_rows = len(ds)
        logger.info(f"  ok: {n_rows} rows cached")

    print()
    if failures:
        print(f"FAILED: {len(failures)}/{len(specs)} datasets")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"OK: {len(specs)}/{len(specs)} datasets cached")
    return 0


if __name__ == "__main__":
    sys.exit(main())
