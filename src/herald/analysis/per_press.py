"""Per-press breakdown of hazard prediction quality.

Answers: does the predictor work equally well across
compression methods, or is performance dominated by
a subset (e.g., StreamingLLM)?

For each horizon and each press name, reports token AUROC/
AUPRC (all + pre-onset), sequence AUROC, and count of
catastrophic vs clean sequences.
"""

from pathlib import Path

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    DEFAULT_NT_ONSET_FRAC,
    group_tokens,
    iter_horizon_predictions,
    seq_onsets_for_keys,
    write_json,
)
from herald.evaluate import sequence_metrics, token_metrics


def run_per_press(
    model_dir: Path,
    results_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "nt_onset_frac": nt_onset_frac,
        "by_horizon": by_horizon,
    }

    for h, ds, preds in iter_horizon_predictions(
        model_dir, results_dir, horizons, nt_onset_frac
    ):
        buckets = group_tokens(ds, preds, key_fn=lambda i: ds.press_ids[i])

        per_press: dict[str, object] = {}
        for press, b in buckets.items():
            seq_keys = set(b["seq_ids"])
            onsets = seq_onsets_for_keys(ds, seq_keys)

            tok = token_metrics(b["y"], b["pred"], b["pre"])
            seq = sequence_metrics(b["pred"], b["seq_ids"], b["pre"], onsets)
            per_press[press] = {
                "n_tokens": len(b["y"]),
                "n_sequences": seq["n_sequences"],
                "n_catastrophic": seq["n_catastrophic"],
                "token": tok,
                "sequence": {
                    k: v for k, v in seq.items() if k != "per_threshold"
                },
            }

        by_horizon[f"H{h}"] = per_press

    write_json(output_path, out)
    return out
