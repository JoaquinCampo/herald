"""CPU-only CLI for the failure-onset event-study figure.

Example
-------
.venv/bin/python scripts/build_event_study.py \\
    --input results/phase0 \\
    --output results/analysis/event_study \\
    --window-before 200 --window-after 200
"""

import argparse
import json
import sys
from pathlib import Path

from herald.analysis.event_study import (
    DEFAULT_FEATURES,
    DEFAULT_N_BOOTSTRAP,
    DEFAULT_WINDOW_AFTER,
    DEFAULT_WINDOW_BEFORE,
    EventStudyConfig,
    run_event_study,
)
from herald.labeling import DEFAULT_NT_ONSET_FRAC


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build the HERALD failure-onset event-study figure. "
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
    cfg = EventStudyConfig(
        window_before=args.window_before,
        window_after=args.window_after,
        features=feats,
        nt_onset_frac=args.nt_onset_frac,
        n_bootstrap=args.n_bootstrap,
        rouge_threshold=args.rouge_threshold,
        seed=args.seed,
    )
    result = run_event_study(args.input, args.output, cfg)
    print(json.dumps(result.summary, indent=2, default=str))
    if result.summary.get("blockers"):
        return 2
    if not result.summary.get("n_runs_with_onset"):
        # Soft success: pipeline ran end-to-end but had no usable
        # catastrophic runs. Caller can inspect the summary.
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
