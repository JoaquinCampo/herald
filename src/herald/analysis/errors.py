"""Sequence-level error breakdown at a fixed threshold.

A sequence is predicted catastrophic if max(hazard) over
pre-onset tokens exceeds the threshold. We then cross-
tabulate errors by press, compression ratio, and generated
length bucket to see where the predictor fails.

Useful for two paper claims:
  1. Errors are not concentrated in one press or one ratio
     (i.e., the model is uniformly reliable).
  2. Failures have an interpretable pattern (e.g., very
     short sequences).
"""

from collections import defaultdict
from pathlib import Path

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    DEFAULT_NT_ONSET_FRAC,
    iter_horizon_predictions,
    write_json,
)

DEFAULT_THRESHOLD = 0.5


def _parse_seq(seq_id: str) -> tuple[str, str, float]:
    prompt_id, press, ratio = seq_id.rsplit("__", 2)
    return prompt_id, press, float(ratio)


def _per_seq_scores(
    seq_ids: list[str], preds: list[float], pre: list[bool]
) -> tuple[dict[str, float], dict[str, int]]:
    scores: dict[str, float] = {}
    counts: dict[str, int] = defaultdict(int)
    for i, sid in enumerate(seq_ids):
        if not pre[i]:
            continue
        counts[sid] += 1
        p = preds[i]
        if sid not in scores or p > scores[sid]:
            scores[sid] = p
    return scores, counts


def _bucket_tokens(n: int) -> str:
    if n < 64:
        return "0-63"
    if n < 128:
        return "64-127"
    if n < 256:
        return "128-255"
    return "256+"


def run_errors(
    model_dir: Path,
    results_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "nt_onset_frac": nt_onset_frac,
        "threshold": threshold,
        "by_horizon": by_horizon,
    }

    for h, ds, preds in iter_horizon_predictions(
        model_dir, results_dir, horizons, nt_onset_frac
    ):
        seq_scores, pre_counts = _per_seq_scores(
            ds.seq_ids, preds, ds.pre_onset
        )

        conf = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        by_press: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        by_ratio: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        by_len: dict[str, dict[str, int]] = defaultdict(
            lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        )
        fn_examples: list[str] = []
        fp_examples: list[str] = []

        for sid, onset in ds.seq_onsets.items():
            score = seq_scores.get(sid, 0.0)
            label = 1 if onset is not None else 0
            pred = 1 if score >= threshold else 0

            if label == 1 and pred == 1:
                cell = "tp"
            elif label == 0 and pred == 1:
                cell = "fp"
            elif label == 1 and pred == 0:
                cell = "fn"
            else:
                cell = "tn"

            _, press, ratio = _parse_seq(sid)
            bucket = _bucket_tokens(pre_counts.get(sid, 0))

            conf[cell] += 1
            by_press[press][cell] += 1
            by_ratio[f"{ratio}"][cell] += 1
            by_len[bucket][cell] += 1

            if cell == "fn" and len(fn_examples) < 20:
                fn_examples.append(sid)
            if cell == "fp" and len(fp_examples) < 20:
                fp_examples.append(sid)

        by_horizon[f"H{h}"] = {
            "n_sequences": len(ds.seq_onsets),
            "confusion": conf,
            "by_press": dict(by_press),
            "by_ratio": dict(by_ratio),
            "by_length_bucket": dict(by_len),
            "false_negatives_sample": fn_examples,
            "false_positives_sample": fp_examples,
        }

    write_json(output_path, out)
    return out
