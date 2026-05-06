"""Run the Phase 1 offline metric modules that don't require either:

  - the [metrics] extra (sentence-transformers, rouge_score,
    editdistance — not installed on Orion).
  - a multi-million-row token-by-token Python audit (token.py).

This driver runs trajectory, outcome, tags, segment, and the alignment
matrix. It's designed to be the smallest set of metrics needed for the
Phase 1 success criterion (intrinsic-to-extrinsic damage prediction).

Usage:
    .venv/bin/python scripts/phase1_metrics_partial.py \\
        --root results/phase1
"""

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("results/phase1"))
    ap.add_argument("--skip-segments", action="store_true")
    args = ap.parse_args()

    final = args.root / "final"
    out = args.root / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    if not (final / "runs.parquet").exists():
        sys.exit(f"missing {final / 'runs.parquet'}; run finalize first.")

    print("=== trajectory ===", flush=True)
    from herald.metrics import trajectory as _trajectory

    _trajectory.build(final, out)

    print("=== outcome ===", flush=True)
    from herald.metrics import outcome as _outcome

    _outcome.build(final, out)

    print("=== tags ===", flush=True)
    from herald.metrics import tags as _tags

    _tags.build(final, out)

    if not args.skip_segments:
        print("=== segments ===", flush=True)
        import polars as pl

        from herald.metrics.io import _collect_raw_dirs
        from herald.metrics.segment import (
            DEFAULT_KS,
            aggregate_all_segments,
            write_segments,
        )

        token_dirs = _collect_raw_dirs(args.root, "tokens")
        parts: list[pl.DataFrame] = []
        for i, d in enumerate(token_dirs, 1):
            print(f"  segments cell {i}/{len(token_dirs)}: {d}", flush=True)
            df = aggregate_all_segments(d, Ks=DEFAULT_KS)
            if not df.is_empty():
                parts.append(df)
        if parts:
            combined = pl.concat(parts, how="vertical")
            write_segments(combined, out / "segments.parquet")

    print("done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
