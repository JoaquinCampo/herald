"""Feature engineering: TokenSignals → ML-ready feature vectors.

Produces 30 features per token:
  17 raw numeric (from TokenSignals)
  12 rolling statistics (8-token causal window)
   1 positional (token_position)
"""

import json
import math
from collections import deque
from pathlib import Path

from herald.config import RunResult, TokenSignals
from herald.labeling import create_horizon_labels, earliest_onset

ROLLING_TARGETS = [
    "entropy",
    "top1_prob",
    "h_alts",
    "delta_h",
    "kl_div",
    "top10_jaccard",
]
ROLLING_WINDOW = 8


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
    """Add 8-token causal rolling mean and std for key signals.

    Adds {name}_mean_8 and {name}_std_8 for each target.
    First tokens use partial windows (min_periods=1).
    """
    # Pre-build windows per target
    windows: dict[str, deque[float]] = {
        name: deque(maxlen=ROLLING_WINDOW) for name in ROLLING_TARGETS
    }

    for row in rows:
        for name in ROLLING_TARGETS:
            val = row[name]
            win = windows[name]
            win.append(val)

            vals = list(win)
            n = len(vals)
            mean = sum(vals) / n
            row[f"{name}_mean_{ROLLING_WINDOW}"] = mean

            if n < 2:
                row[f"{name}_std_{ROLLING_WINDOW}"] = 0.0
            else:
                variance = sum((v - mean) ** 2 for v in vals) / (n - 1)
                row[f"{name}_std_{ROLLING_WINDOW}"] = math.sqrt(variance)

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
) -> tuple[
    list[list[float]],
    list[int],
    list[str],
    list[str],
    list[str],
]:
    """Load sweep results and build ML-ready dataset.

    Returns (X, y, run_ids, press_ids, feature_names) where:
      X: list of feature vectors (one per token)
      y: list of binary labels
      run_ids: list of prompt_id per token (for GroupKFold)
      press_ids: list of press name per token (for LOCO CV)
      feature_names: ordered feature column names
    """
    all_x: list[list[float]] = []
    all_y: list[int] = []
    all_run_ids: list[str] = []
    all_press: list[str] = []
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

            if not names and rows:
                names = list(rows[0].keys())

            for row, label in zip(rows, labels):
                all_x.append(list(row.values()))
                all_y.append(label)
                all_run_ids.append(result.prompt_id)
                all_press.append(result.press)

    return all_x, all_y, all_run_ids, all_press, names
