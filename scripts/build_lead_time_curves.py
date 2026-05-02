"""CPU-only CLI for the per-feature lead-time analysis.

Example
-------
.venv/bin/python scripts/build_lead_time_curves.py \\
    --input results/phase0 \\
    --output results/analysis/lead_time \\
    --window-before 200 --window-after 50
"""

import argparse
import json
import sys
from pathlib import Path

from herald.analysis.lead_time import (
    DEFAULT_AUROC_THRESHOLD,
    DEFAULT_CI_LOWER_THRESHOLD,
    DEFAULT_FEATURES,
    DEFAULT_MIN_PER_STRATUM,
    DEFAULT_MIN_WELL_COVERED_STRATA,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_PERSISTENCE,
    DEFAULT_POSITION_STRIDE,
    DEFAULT_WINDOW_AFTER,
    DEFAULT_WINDOW_BEFORE,
    LeadTimeConfig,
    run_lead_time,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the HERALD per-feature lead-time figure. "
            "CPU-only. Reads finalized run+token parquet artifacts."
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
        "--window-before",
        type=int,
        default=DEFAULT_WINDOW_BEFORE,
        help="Tokens before onset to include in the window.",
    )
    p.add_argument(
        "--window-after",
        type=int,
        default=DEFAULT_WINDOW_AFTER,
        help="Tokens after onset to include in the window.",
    )
    p.add_argument(
        "--position-stride",
        type=int,
        default=DEFAULT_POSITION_STRIDE,
        help=(
            "Evaluate AUROC every N tokens. Stride=1 evaluates every "
            "token; larger values trade resolution for compute."
        ),
    )
    p.add_argument(
        "--features",
        type=str,
        default=",".join(DEFAULT_FEATURES),
        help=(
            "Comma-separated feature names. Names not found in the "
            "tokens parquet are reported in the summary and dropped."
        ),
    )
    p.add_argument(
        "--nt-onset-frac",
        type=float,
        default=DEFAULT_NT_ONSET_FRAC,
        help="Fraction of max_new_tokens used as non-term onset proxy.",
    )
    p.add_argument(
        "--rouge-threshold",
        type=float,
        default=None,
        help=(
            "If set, additionally require rouge_l < threshold "
            "(or rouge_l null) when selecting catastrophic runs."
        ),
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Bootstrap resamples for the 95%% CI band.",
    )
    p.add_argument(
        "--auroc-threshold",
        type=float,
        default=DEFAULT_AUROC_THRESHOLD,
        help="AUROC threshold for the lead-time persistence test.",
    )
    p.add_argument(
        "--ci-lower-threshold",
        type=float,
        default=DEFAULT_CI_LOWER_THRESHOLD,
        help=(
            "Bootstrap CI lower bound must exceed this value at every "
            "position in the persistence window."
        ),
    )
    p.add_argument(
        "--persistence",
        type=int,
        default=DEFAULT_PERSISTENCE,
        help=(
            "Number of consecutive evaluated positions that must "
            "satisfy the AUROC + CI condition."
        ),
    )
    p.add_argument(
        "--min-per-stratum",
        type=int,
        default=DEFAULT_MIN_PER_STRATUM,
        help=(
            "Minimum cat & ctrl runs per (task, press, ratio) stratum "
            "before the run is considered well-powered."
        ),
    )
    p.add_argument(
        "--min-well-covered-strata",
        type=int,
        default=DEFAULT_MIN_WELL_COVERED_STRATA,
        help=(
            "Minimum number of well-powered strata required before "
            "the run stops being flagged as control-limited."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Bootstrap RNG seed.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    feats = tuple(
        f.strip() for f in str(args.features).split(",") if f.strip()
    )
    cfg = LeadTimeConfig(
        window_before=args.window_before,
        window_after=args.window_after,
        position_stride=args.position_stride,
        features=feats,
        nt_onset_frac=args.nt_onset_frac,
        n_bootstrap=args.n_bootstrap,
        rouge_threshold=args.rouge_threshold,
        auroc_threshold=args.auroc_threshold,
        ci_lower_threshold=args.ci_lower_threshold,
        persistence=args.persistence,
        min_per_stratum=args.min_per_stratum,
        min_well_covered_strata=args.min_well_covered_strata,
        seed=args.seed,
    )
    result = run_lead_time(args.input, args.output, cfg)
    print(json.dumps(result.summary, indent=2, default=str))
    if result.summary.get("blockers"):
        return 2
    if result.summary.get("status") == "insufficient_pairs":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
