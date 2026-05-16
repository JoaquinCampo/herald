"""Compute catastrophe onset positions for Phase 2 v2.

CPU-only. For every compressed run in `runs.parquet`, applies
`detect_catastrophe_onsets(generated_token_ids, stop_reason, catastrophes)`
and persists `(run_id, looping_onset, non_termination_onset)`.

This is the foundation for the onset-anchored task: y=1 iff the next K
tokens contain an onset. Does NOT touch tokens/ or replay/ — pure
post-hoc detection on already-generated traces.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from loguru import logger

from herald.detectors import detect_catastrophe_onsets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        type=Path,
        default=Path("results/phase1/final/runs.parquet"),
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase2_v2/onsets.parquet"),
    )
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Reading {}", args.runs)
    df = pd.read_parquet(
        args.runs,
        columns=[
            "run_id",
            "press",
            "compression_ratio",
            "stop_reason",
            "catastrophes",
            "generated_token_ids",
            "num_tokens_generated",
        ],
    )
    logger.info("Loaded {} rows", len(df))

    compressed = df[df["press"] != "none"].reset_index(drop=True)
    logger.info("Compressed runs: {}", len(compressed))

    looping_onsets: list[int | None] = []
    nt_onsets: list[int | None] = []

    for row in compressed.itertuples(index=False):
        catastrophes = list(row.catastrophes) if row.catastrophes is not None else []
        token_ids = (
            list(row.generated_token_ids)
            if row.generated_token_ids is not None
            else []
        )
        onsets = detect_catastrophe_onsets(
            token_ids, row.stop_reason, catastrophes
        )
        looping_onsets.append(onsets.get("looping"))
        nt_onsets.append(onsets.get("non_termination"))

    out = pd.DataFrame(
        {
            "run_id": compressed["run_id"].values,
            "press": compressed["press"].values,
            "compression_ratio": compressed["compression_ratio"].values,
            "num_tokens_generated": compressed[
                "num_tokens_generated"
            ].values,
            "stop_reason": compressed["stop_reason"].values,
            "looping_onset": pd.array(looping_onsets, dtype="Int32"),
            "non_termination_onset": pd.array(nt_onsets, dtype="Int32"),
        }
    )

    looping_rate = out["looping_onset"].notna().mean()
    nt_rate = out["non_termination_onset"].notna().mean()
    any_rate = (
        out["looping_onset"].notna() | out["non_termination_onset"].notna()
    ).mean()

    logger.info("Looping onset rate: {:.3%}", looping_rate)
    logger.info("Non-termination onset rate: {:.3%}", nt_rate)
    logger.info("Either-onset rate: {:.3%}", any_rate)

    logger.info("Writing {}", args.output)
    out.to_parquet(args.output, index=False)

    summary = {
        "n_runs": int(len(out)),
        "looping_onset_rate": float(looping_rate),
        "non_termination_onset_rate": float(nt_rate),
        "either_onset_rate": float(any_rate),
        "looping_onset_quantiles": {
            q: (
                float(
                    out["looping_onset"].dropna().astype(float).quantile(q)
                )
                if out["looping_onset"].notna().any()
                else None
            )
            for q in [0.1, 0.25, 0.5, 0.75, 0.9]
        },
        "by_press": (
            out.assign(
                has_looping=out["looping_onset"].notna(),
                has_nt=out["non_termination_onset"].notna(),
            )
            .groupby("press")[["has_looping", "has_nt"]]
            .mean()
            .round(4)
            .to_dict(orient="index")
        ),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Summary -> {}", summary_path)


if __name__ == "__main__":
    main()
