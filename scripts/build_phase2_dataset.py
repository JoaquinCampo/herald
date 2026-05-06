"""Build the Phase 2 per-token predictor dataset.

CPU-only. Reads `<root>/final/runs.parquet`, walks compressed runs,
joins per-token features with per-token replay metrics, computes
future-window labels per horizon, and joins run-level validators
from `<root>/metrics/run_damage.parquet`.

Outputs:
- `<output_dir>/phase2_tokens.parquet` — full table (concat of parts)
- `<output_dir>/parts/press={p}__ratio={r}.parquet` — one per (press, ratio)
- `<output_dir>/phase2_dataset_summary.json` — row counts, columns,
  flags including `nll_ratio_sign_flipped`

Spec: gold/phase-2-dataset.md.
"""

import argparse
import json
import time
from pathlib import Path

from loguru import logger

from herald.predictor_dataset import DEFAULT_HORIZONS, build_token_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("results/phase1"),
        help="Phase 1 root containing final/ and metrics/.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase2/dataset"),
    )
    ap.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(DEFAULT_HORIZONS),
        help="Future-window horizons in realized tokens.",
    )
    args = ap.parse_args()

    final_dir = args.root / "final"
    run_damage_path = args.root / "metrics" / "run_damage.parquet"
    output_path = args.output_dir / "phase2_tokens.parquet"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Phase 2 dataset build")
    logger.info("  final_dir       = {}", final_dir)
    logger.info("  run_damage_path = {}", run_damage_path)
    logger.info("  output_path     = {}", output_path)
    logger.info("  horizons        = {}", args.horizons)

    t0 = time.time()
    summary = build_token_dataset(
        final_dir=final_dir,
        run_damage_path=run_damage_path,
        output_path=output_path,
        horizons=tuple(args.horizons),
    )
    summary["wall_seconds"] = round(time.time() - t0, 2)
    summary_path = args.output_dir / "phase2_dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    logger.info(
        "done: {} rows from {} runs in {} partitions, "
        "{} runs skipped (missing files), wall={}s",
        summary["n_rows"],
        summary["n_runs"],
        summary["n_partitions"],
        summary["n_runs_skipped_missing_files"],
        summary["wall_seconds"],
    )
    logger.info("summary -> {}", summary_path)
    logger.info("table   -> {}", output_path)


if __name__ == "__main__":
    main()
