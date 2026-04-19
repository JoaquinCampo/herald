"""Feature engineering: TokenSignals → ML-ready feature vectors.

Produces 42 features per token:
  17 raw numeric (from TokenSignals)
  24 rolling statistics (8-token and 32-token causal windows)
   1 positional (token_position)
"""

import json
import math
from collections import deque
from pathlib import Path

from pydantic import BaseModel

from herald.config import RunResult, TokenSignals
from herald.labeling import create_horizon_labels, earliest_onset


class DatasetBundle(BaseModel):
    """ML-ready dataset with onset metadata for evaluation."""

    X: list[list[float]]
    y: list[int]
    run_ids: list[str]
    press_ids: list[str]
    feat_names: list[str]
    pre_onset: list[bool]
    seq_ids: list[str]
    seq_onsets: dict[str, int | None]


ROLLING_TARGETS = [
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
]
ROLLING_WINDOW = 8
ROLLING_WINDOW_LONG = 32
ROLLING_WINDOWS = (ROLLING_WINDOW, ROLLING_WINDOW_LONG)


def flatten_signals(
    signals: list[TokenSignals],
    max_new_tokens: int = 512,
) -> list[dict[str, float]]:
    """Unpack TokenSignals into flat feature dicts.

    Expands top5_logprobs into logprob_0..logprob_4.
    Replaces NaN temporal features at token 0 with 0.0.
    Adds normalized token_position (t / max_new_tokens).
    """
    denom = max(max_new_tokens, 1)
    rows: list[dict[str, float]] = []
    for t, sig in enumerate(signals):
        row: dict[str, float] = {
            "entropy": sig.entropy,
            "top1_prob": sig.top1_prob,
            "top5_prob": sig.top5_prob,
            "h_alts": sig.h_alts,
            "avg_logp": sig.avg_logp,
            "delta_h": sig.delta_h,
            "kl_div": sig.kl_div,
            "top10_jaccard": sig.top10_jaccard,
            "eff_vocab_size": sig.eff_vocab_size,
            "tail_mass": sig.tail_mass,
            "logit_range": sig.logit_range,
            "delta_h_valid": float(sig.delta_h_valid),
        }

        # Expand top-5 logprobs
        for i in range(5):
            if i < len(sig.top5_logprobs):
                row[f"logprob_{i}"] = sig.top5_logprobs[i]
            else:
                row[f"logprob_{i}"] = 0.0

        # Replace structural NaN at first token with 0.0
        if t == 0:
            for key in ("delta_h", "kl_div", "top10_jaccard"):
                if math.isnan(row[key]):
                    row[key] = 0.0

        # Normalized positional feature (avoids leaking sequence length)
        row["token_position"] = float(t) / denom
        rows.append(row)
    return rows


def add_rolling_features(
    rows: list[dict[str, float]],
) -> list[dict[str, float]]:
    """Add causal rolling mean and std for each ROLLING_WINDOWS.

    Adds {name}_mean_{w} and {name}_std_{w} for each target
    and each window size. First tokens use partial windows
    (min_periods=1). Short window catches local shocks;
    long window captures drift.
    """
    windows: dict[tuple[str, int], deque[float]] = {
        (name, w): deque(maxlen=w)
        for name in ROLLING_TARGETS
        for w in ROLLING_WINDOWS
    }

    for row in rows:
        for name in ROLLING_TARGETS:
            val = row[name]
            for w in ROLLING_WINDOWS:
                win = windows[(name, w)]
                win.append(val)

                vals = list(win)
                n = len(vals)
                mean = sum(vals) / n
                row[f"{name}_mean_{w}"] = mean

                if n < 2:
                    row[f"{name}_std_{w}"] = 0.0
                else:
                    variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
                    row[f"{name}_std_{w}"] = math.sqrt(variance)

    return rows


def feature_names(rows: list[dict[str, float]]) -> list[str]:
    """Return ordered list of feature names from a processed row."""
    if not rows:
        return []
    return list(rows[0].keys())


def build_dataset(
    results_dir: Path,
    horizon: int = 10,
    nt_onset_frac: float = 0.75,
) -> DatasetBundle:
    """Load sweep results and build ML-ready dataset.

    Returns a DatasetBundle with feature vectors, labels,
    grouping IDs, and onset metadata for evaluation.
    """
    all_x: list[list[float]] = []
    all_y: list[int] = []
    all_run_ids: list[str] = []
    all_press: list[str] = []
    all_pre_onset: list[bool] = []
    all_seq_ids: list[str] = []
    seq_onsets: dict[str, int | None] = {}
    names: list[str] = []

    for json_path in sorted(results_dir.rglob("*.json")):
        data = json.loads(json_path.read_text())
        results_list = data.get("results", [])

        for raw in results_list:
            result = RunResult.model_validate(raw)

            if not result.signals:
                continue

            rows = flatten_signals(
                result.signals,
                max_new_tokens=result.max_new_tokens,
            )
            rows = add_rolling_features(rows)

            onset = earliest_onset(
                result.catastrophe_onsets,
                result.catastrophes,
                max_new_tokens=result.max_new_tokens,
                nt_onset_frac=nt_onset_frac,
                n_tokens=len(result.signals),
            )
            labels = create_horizon_labels(
                len(result.signals), onset, horizon
            )

            seq_id = (
                f"{result.prompt_id}"
                f"__{result.press}"
                f"__{result.compression_ratio}"
            )
            seq_onsets[seq_id] = onset

            if not names and rows:
                names = list(rows[0].keys())

            for t, (row, label) in enumerate(zip(rows, labels)):
                all_x.append(list(row.values()))
                all_y.append(label)
                all_run_ids.append(result.prompt_id)
                all_press.append(result.press)
                all_seq_ids.append(seq_id)
                is_pre = onset is None or t < onset
                all_pre_onset.append(is_pre)

    return DatasetBundle(
        X=all_x,
        y=all_y,
        run_ids=all_run_ids,
        press_ids=all_press,
        feat_names=names,
        pre_onset=all_pre_onset,
        seq_ids=all_seq_ids,
        seq_onsets=seq_onsets,
    )
