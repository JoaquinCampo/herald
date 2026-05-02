"""CPU-only CLI for the HERALD feature-information ceiling analysis.

Example
-------
.venv/bin/python scripts/build_information_ceiling.py \\
    --input results/phase0 \\
    --output results/analysis/information_ceiling \\
    --horizons 5,10,25,50
"""

import argparse
import json
import sys
from pathlib import Path

from herald.analysis.information_ceiling import (
    DEFAULT_HORIZONS,
    DEFAULT_MAX_SAMPLES,
    DEFAULT_MIN_RUNS_PER_PRESS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_SPLITS,
    DEFAULT_ONLINE_FEATURES,
    InformationCeilingConfig,
    run_information_ceiling,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the HERALD information-ceiling figure. CPU-only. "
            "Quantifies whether cheap online features add information "
            "about future compression damage beyond position + "
            "metadata baselines."
        )
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Input root. Expects <input>/final/runs.parquet and "
            "<input>/final/tokens/."
        ),
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for parquet, JSON, and PNG artifacts.",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HORIZONS),
        help="Comma-separated future-damage horizons in tokens.",
    )
    p.add_argument(
        "--online-features",
        type=str,
        default=",".join(DEFAULT_ONLINE_FEATURES),
        help=(
            "Comma-separated online feature names. Names not present "
            "in the tokens parquet are reported in the summary and "
            "dropped."
        ),
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            "Maximum token rows used for MI / CV scoring; larger "
            "datasets are uniformly subsampled."
        ),
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Run-level bootstrap resamples for the AUROC CI.",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="GroupKFold splits over run_id for CV AUROC.",
    )
    p.add_argument(
        "--min-runs-per-press",
        type=int,
        default=DEFAULT_MIN_RUNS_PER_PRESS,
        help=(
            "Minimum compressed runs per press before the run stops "
            "being flagged as Phase 0 smoke."
        ),
    )
    p.add_argument(
        "--nt-onset-frac",
        type=float,
        default=DEFAULT_NT_ONSET_FRAC,
        help="Fraction of max_new_tokens used as non-term onset proxy.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap + MI RNG seed.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = tuple(
        int(h.strip()) for h in str(args.horizons).split(",") if h.strip()
    )
    online_features = tuple(
        f.strip()
        for f in str(args.online_features).split(",")
        if f.strip()
    )
    cfg = InformationCeilingConfig(
        horizons=horizons,
        online_features=online_features,
        nt_onset_frac=args.nt_onset_frac,
        max_samples=args.max_samples,
        n_bootstrap=args.n_bootstrap,
        n_splits=args.n_splits,
        min_runs_per_press=args.min_runs_per_press,
        seed=args.seed,
    )
    result = run_information_ceiling(args.input, args.output, cfg)
    print(json.dumps(result.summary, indent=2, default=str))
    if result.summary.get("blockers"):
        return 2
    if result.summary.get("status") in {
        "no_compressed_runs",
        "no_token_rows",
    }:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
