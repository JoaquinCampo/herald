"""Calibration and reliability analysis.

AUROC measures ranking; it says nothing about whether a
predicted hazard of 0.8 corresponds to an 80% catastrophe
rate. Reviewers always ask about calibration for a
thresholded intervention system.

Reports, per horizon on pre-onset tokens:
  - Reliability curve (bin means, positive fractions,
    bin counts)
  - Brier score
  - Expected Calibration Error (ECE, uniform-width bins)
"""

from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    DEFAULT_NT_ONSET_FRAC,
    iter_horizon_predictions,
    write_json,
)

DEFAULT_N_BINS = 15


def _expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int
) -> float:
    """Uniform-width ECE.

    Sum over bins of (bin_count / N) * |bin_accuracy - bin_conf|.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y)
    if n == 0:
        return 0.0

    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1.0 else (p >= lo) & (p <= hi)
        k = int(mask.sum())
        if k == 0:
            continue
        ece += (k / n) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return round(ece, 4)


def run_calibration(
    model_dir: Path,
    results_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
    n_bins: int = DEFAULT_N_BINS,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "nt_onset_frac": nt_onset_frac,
        "n_bins": n_bins,
        "by_horizon": by_horizon,
    }

    for h, ds, preds in iter_horizon_predictions(
        model_dir, results_dir, horizons, nt_onset_frac
    ):
        y_pre = np.asarray(
            [ds.y[i] for i in range(len(preds)) if ds.pre_onset[i]]
        )
        p_pre = np.asarray(
            [preds[i] for i in range(len(preds)) if ds.pre_onset[i]],
            dtype=float,
        )

        entry: dict[str, object] = {
            "n_samples": int(len(y_pre)),
            "positive_rate": round(float(y_pre.mean()), 6)
            if len(y_pre)
            else None,
        }

        if len(set(y_pre.tolist())) < 2:
            entry["error"] = "degenerate labels"
            by_horizon[f"H{h}"] = entry
            continue

        frac_pos, mean_pred = calibration_curve(
            y_pre, p_pre, n_bins=n_bins, strategy="uniform"
        )

        bin_counts = []
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (
                (p_pre >= lo) & (p_pre < hi)
                if hi < 1.0
                else ((p_pre >= lo) & (p_pre <= hi))
            )
            bin_counts.append(int(mask.sum()))

        entry["brier"] = round(float(brier_score_loss(y_pre, p_pre)), 6)
        entry["ece"] = _expected_calibration_error(y_pre, p_pre, n_bins)
        entry["reliability"] = {
            "mean_predicted": [round(float(x), 6) for x in mean_pred],
            "fraction_positive": [round(float(x), 6) for x in frac_pos],
            "bin_counts": bin_counts,
            "bin_edges": [round(float(x), 6) for x in edges],
        }

        by_horizon[f"H{h}"] = entry

    write_json(output_path, out)
    return out
