"""CPU-only CLI for the HERALD cross-press transfer matrix.

Trains a simple future-damage predictor on tokens of one press and
scores it on every other press, building an (m x m) matrix per
(horizon, feature_set). Validates or falsifies the black-box
"compressor-agnostic" claim.

Example
-------
.venv/bin/python scripts/build_cross_press_transfer.py \\
    --input results/phase0 \\
    --output results/analysis/cross_press_transfer \\
    --horizons 5,10,25,50
"""

import argparse
import json
import sys
from pathlib import Path

from herald.analysis.cross_press_transfer import (
    DEFAULT_FEATURE_SETS,
    DEFAULT_MAX_TEST_SAMPLES,
    DEFAULT_MAX_TRAIN_SAMPLES,
    DEFAULT_MIN_NEG,
    DEFAULT_MIN_POS,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_N_SPLITS,
    CrossPressTransferConfig,
    run_cross_press_transfer,
)
from herald.analysis.information_ceiling import (
    DEFAULT_HORIZONS,
    DEFAULT_ONLINE_FEATURES,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the HERALD cross-press transfer matrix. CPU-only. "
            "Trains a logistic regression on one press and tests on "
            "every other press, per (horizon, feature_set)."
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
            "Comma-separated online feature names. Names absent "
            "from the tokens parquet are reported and dropped."
        ),
    )
    p.add_argument(
        "--feature-sets",
        type=str,
        default=",".join(DEFAULT_FEATURE_SETS),
        help=(
            "Comma-separated feature-set names from "
            "{position, metadata, position_metadata, online, all, "
            "entropy_only}."
        ),
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Test-run bootstrap resamples for off-diagonal CI.",
    )
    p.add_argument(
        "--n-splits",
        type=int,
        default=DEFAULT_N_SPLITS,
        help="GroupKFold splits over run_id for diagonal CV.",
    )
    p.add_argument(
        "--min-pos",
        type=int,
        default=DEFAULT_MIN_POS,
        help="Minimum positives required on each side of a cell.",
    )
    p.add_argument(
        "--min-neg",
        type=int,
        default=DEFAULT_MIN_NEG,
        help="Minimum negatives required on each side of a cell.",
    )
    p.add_argument(
        "--max-train-samples",
        type=int,
        default=DEFAULT_MAX_TRAIN_SAMPLES,
        help="Cap on per-cell train rows after filtering.",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=DEFAULT_MAX_TEST_SAMPLES,
        help="Cap on per-cell test rows after filtering.",
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
        help="RNG seed.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = tuple(
        int(h.strip()) for h in str(args.horizons).split(",") if h.strip()
    )
    online_features = tuple(
        f.strip() for f in str(args.online_features).split(",") if f.strip()
    )
    feature_sets = tuple(
        f.strip() for f in str(args.feature_sets).split(",") if f.strip()
    )
    cfg = CrossPressTransferConfig(
        horizons=horizons,
        online_features=online_features,
        feature_sets=feature_sets,
        nt_onset_frac=args.nt_onset_frac,
        n_bootstrap=args.n_bootstrap,
        n_splits=args.n_splits,
        min_pos=args.min_pos,
        min_neg=args.min_neg,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        seed=args.seed,
    )
    result = run_cross_press_transfer(args.input, args.output, cfg)
    print(json.dumps(result.summary, indent=2, default=str))
    if result.summary.get("blockers"):
        return 2
    if result.summary.get("status") in {
        "no_compressed_runs",
        "insufficient_presses",
        "no_token_rows",
        "no_feature_sets",
    }:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
