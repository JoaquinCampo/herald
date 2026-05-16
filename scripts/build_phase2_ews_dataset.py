"""Phase 2c: augment phase2_tokens.parquet with early-warning features.

CPU-only.  Reads `results/phase2/dataset/phase2_tokens.parquet`,
computes the EWS family per (signal, window) per run_id (causally),
and writes:

- results/phase2c_early_warning/phase2_dataset_ews.parquet
- results/phase2c_early_warning/ews_feature_audit.json

The audit JSON records:
- the deployable cheap-feature column set (existing Phase 2 + EWS)
- the explicitly-excluded label / validator / replay columns
- the EWS feature list (canonical names) and per-family null counts
- summary statistics (rows, runs, partitions)
"""

import argparse
import json
import time
from pathlib import Path

import polars as pl
from loguru import logger

from herald.early_warning_features import (
    DEFAULT_BASE_SIGNALS,
    DEFAULT_WINDOWS,
    NON_DEPLOYABLE_COLUMNS,
    add_ews_features,
    deployable_feature_columns,
    ews_feature_names,
)


def _ews_null_counts(df: pl.DataFrame) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in df.columns:
        if not c.startswith("ews_"):
            continue
        out[c] = int(df[c].null_count())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/phase2/dataset/phase2_tokens.parquet"),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2c_early_warning"),
    )
    ap.add_argument(
        "--signals",
        type=str,
        nargs="+",
        default=list(DEFAULT_BASE_SIGNALS),
    )
    ap.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS),
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "If set, read only the first ~10 partitions for a fast "
            "schema check. Final outputs go under output-dir/smoke/."
        ),
    )
    args = ap.parse_args()

    out_dir = args.output_dir / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("loading dataset {}", args.dataset)
    if args.smoke:
        # Read a slice covering ~5 runs end-to-end.
        full = pl.read_parquet(args.dataset)
        run_ids = full["run_id"].unique().to_list()[:50]
        df = full.filter(pl.col("run_id").is_in(run_ids))
        logger.info("smoke slice: {} rows / {} runs", df.height, len(run_ids))
    else:
        df = pl.read_parquet(args.dataset)
        logger.info(
            "full dataset: {} rows / {} cols", df.height, len(df.columns)
        )

    # Sanity: required columns present.
    for col in ("run_id", "token_pos"):
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    # Schema hygiene: never expose replay/label columns to the EWS module.
    forbidden_in_signals = set(args.signals) & NON_DEPLOYABLE_COLUMNS
    if forbidden_in_signals:
        raise SystemExit(
            f"refusing to compute EWS over non-deployable signals: "
            f"{sorted(forbidden_in_signals)}"
        )

    t0 = time.time()
    logger.info(
        "computing EWS features over signals={} windows={}",
        args.signals,
        args.windows,
    )
    out = add_ews_features(
        df, signals=tuple(args.signals), windows=tuple(args.windows)
    )
    logger.info(
        "EWS attached in {:.1f}s ({} cols)",
        time.time() - t0,
        len(out.columns),
    )

    out_path = out_dir / "phase2_dataset_ews.parquet"
    out.write_parquet(out_path)
    logger.info(
        "wrote {} ({} rows, {} cols)", out_path, out.height, len(out.columns)
    )

    # Audit.
    ews_cols = ews_feature_names(
        signals=tuple(args.signals), windows=tuple(args.windows)
    )
    deployable_existing = deployable_feature_columns(
        [c for c in out.columns if not c.startswith("ews_")]
    )
    deployable_with_ews = deployable_feature_columns(out.columns)

    audit = {
        "dataset_path": str(args.dataset),
        "output_path": str(out_path),
        "n_rows": int(out.height),
        "n_runs": int(out.select(pl.col("run_id").n_unique()).item()),
        "signals": list(args.signals),
        "windows": list(args.windows),
        "n_ews_features": len(ews_cols),
        "ews_features": ews_cols,
        "deployable_existing_count": len(deployable_existing),
        "deployable_with_ews_count": len(deployable_with_ews),
        "non_deployable_columns_present": sorted(
            set(out.columns) & NON_DEPLOYABLE_COLUMNS
        ),
        "future_label_columns_present": sorted(
            c for c in out.columns if c.startswith("future_")
        ),
        "ews_null_counts": _ews_null_counts(out),
        "wall_seconds": round(time.time() - t0, 2),
        "smoke": bool(args.smoke),
    }
    audit_path = out_dir / "ews_feature_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, default=str))
    logger.info("audit -> {}", audit_path)


if __name__ == "__main__":
    main()
