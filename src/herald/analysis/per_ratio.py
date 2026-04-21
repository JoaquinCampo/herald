"""Per-compression-ratio breakdown of prediction quality.

Answers: does the predictor get easier as compression gets
more aggressive? Does it degrade at light compression where
catastrophes are rare?

Slices the eval by compression_ratio (parsed from seq_id)
within each horizon.
"""

from pathlib import Path

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    DEFAULT_NT_ONSET_FRAC,
    compression_ratio_from_seq_id,
    group_tokens,
    iter_horizon_predictions,
    seq_onsets_for_keys,
    write_json,
)
from herald.evaluate import sequence_metrics, token_metrics


def run_per_ratio(
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
        buckets = group_tokens(
            ds,
            preds,
            key_fn=lambda i: (
                f"{compression_ratio_from_seq_id(ds.seq_ids[i])}"
            ),
        )

        per_ratio: dict[str, object] = {}
        for ratio_str, b in buckets.items():
            seq_keys = set(b["seq_ids"])
            onsets = seq_onsets_for_keys(ds, seq_keys)

            tok = token_metrics(b["y"], b["pred"], b["pre"])
            seq = sequence_metrics(b["pred"], b["seq_ids"], b["pre"], onsets)
            per_ratio[ratio_str] = {
                "compression_ratio": float(ratio_str),
                "n_tokens": len(b["y"]),
                "n_sequences": seq["n_sequences"],
                "n_catastrophic": seq["n_catastrophic"],
                "token": tok,
                "sequence": {
                    k: v for k, v in seq.items() if k != "per_threshold"
                },
            }

        by_horizon[f"H{h}"] = per_ratio

    write_json(output_path, out)
    return out
