"""Post-training evaluation: pre-onset metrics, sequence-level
metrics, baselines, and sensitivity analysis."""

from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from herald.features import build_dataset
from herald.train import _cv_metrics

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_ALPHA = 0.05  # 95% CI

# Features where lower value = more dangerous
# (need negation for AUROC to work correctly)
_INVERSE_FEATURES = {
    "top1_prob",
    "top10_jaccard",
    "top1_prob_mean_8",
    "top1_prob_mean_32",
    "top10_jaccard_mean_8",
    "top10_jaccard_mean_32",
}

BASELINE_FEATURES = [
    # Raw single-token features
    "entropy",
    "top1_prob",
    "kl_div",
    "top10_jaccard",
    # Short-window rolling (local shocks)
    "entropy_mean_8",
    "entropy_std_8",
    "top1_prob_mean_8",
    "top1_prob_std_8",
    "kl_div_mean_8",
    "h_alts_mean_8",
    # Long-window rolling (drift)
    "entropy_mean_32",
    "top1_prob_mean_32",
    "top10_jaccard_mean_32",
]

DEFAULT_THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

DEFAULT_NT_FRACS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]


def _safe_auroc(
    y_true: list[int],
    y_score: list[float],
) -> float | None:
    """AUROC that returns None on degenerate input."""
    if len(set(y_true)) < 2 or len(y_true) < 2:
        return None
    val: float = roc_auc_score(y_true, y_score)
    return round(val, 4)


def _safe_auprc(
    y_true: list[int],
    y_score: list[float],
) -> float | None:
    """AUPRC that returns None on degenerate input."""
    if len(set(y_true)) < 2 or len(y_true) < 2:
        return None
    val: float = average_precision_score(y_true, y_score)
    return round(val, 4)


def _bootstrap_ci(
    y_true: list[int],
    y_score: list[float],
    metric_fn: Callable[[list[int], list[float]], float],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = BOOTSTRAP_ALPHA,
) -> list[float] | None:
    """Percentile bootstrap CI for a ranking metric.

    Returns [lo, hi] at (1 - alpha) coverage, or None if the
    input is degenerate or too few resamples yield valid
    labels (e.g. both classes present).
    """
    n = len(y_true)
    if n < 2 or len(set(y_true)) < 2:
        return None

    rng = np.random.default_rng(seed)
    yt = np.asarray(y_true)
    ys = np.asarray(y_score, dtype=float)

    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample_t = yt[idx]
        if len(set(sample_t.tolist())) < 2:
            continue
        boots.append(float(metric_fn(sample_t.tolist(), ys[idx].tolist())))

    if len(boots) < max(10, n_boot // 10):
        return None

    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return [round(lo, 4), round(hi, 4)]


# -------------------------------------------------------
# Token-level metrics
# -------------------------------------------------------


def token_metrics(
    y_true: list[int],
    y_pred: list[float],
    pre_onset: list[bool] | None = None,
) -> dict[str, dict[str, float | list[float] | None]]:
    """Token-level AUROC/AUPRC with bootstrap CIs.

    Returns {"all": {auroc, auprc, auroc_ci, auprc_ci},
             "pre_onset": {...}}.
    Pre-onset excludes trivially-classifiable post-onset
    tokens to give a realistic picture of early-warning
    performance.
    """
    result: dict[str, dict[str, float | list[float] | None]] = {
        "all": {
            "auroc": _safe_auroc(y_true, y_pred),
            "auprc": _safe_auprc(y_true, y_pred),
            "auroc_ci": _bootstrap_ci(y_true, y_pred, roc_auc_score),
            "auprc_ci": _bootstrap_ci(
                y_true, y_pred, average_precision_score
            ),
        },
    }

    if pre_onset is not None:
        yt = [y_true[i] for i in range(len(y_true)) if pre_onset[i]]
        yp = [y_pred[i] for i in range(len(y_pred)) if pre_onset[i]]
        result["pre_onset"] = {
            "auroc": _safe_auroc(yt, yp),
            "auprc": _safe_auprc(yt, yp),
            "auroc_ci": _bootstrap_ci(yt, yp, roc_auc_score),
            "auprc_ci": _bootstrap_ci(yt, yp, average_precision_score),
        }

    return result


# -------------------------------------------------------
# Sequence-level metrics
# -------------------------------------------------------


def sequence_metrics(
    y_pred: list[float],
    seq_ids: list[str],
    pre_onset: list[bool],
    seq_onsets: dict[str, int | None],
    thresholds: list[float] | None = None,
) -> dict[str, object]:
    """Sequence-level prediction quality.

    Per sequence, score = max(y_pred[t]) over pre-onset
    tokens. Label = 1 if sequence has catastrophe.

    Returns sequence AUROC/AUPRC and per-threshold
    precision/recall/F1.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    # Group predictions by sequence
    seq_preds: dict[str, list[float]] = defaultdict(list)
    for i, sid in enumerate(seq_ids):
        if pre_onset[i]:
            seq_preds[sid].append(y_pred[i])

    seq_scores: list[float] = []
    seq_labels: list[int] = []

    for sid, onset in seq_onsets.items():
        preds = seq_preds.get(sid, [])
        score = max(preds) if preds else 0.0
        label = 1 if onset is not None else 0
        seq_scores.append(score)
        seq_labels.append(label)

    n_cat = sum(seq_labels)

    result: dict[str, object] = {
        "seq_auroc": _safe_auroc(seq_labels, seq_scores),
        "seq_auprc": _safe_auprc(seq_labels, seq_scores),
        "seq_auroc_ci": _bootstrap_ci(seq_labels, seq_scores, roc_auc_score),
        "seq_auprc_ci": _bootstrap_ci(
            seq_labels, seq_scores, average_precision_score
        ),
        "n_sequences": len(seq_labels),
        "n_catastrophic": n_cat,
    }

    # Per-threshold precision/recall/F1
    per_thresh: list[dict[str, object]] = []
    for thr in thresholds:
        tp = fp = fn = tn = 0
        for sc, lb in zip(seq_scores, seq_labels):
            pred_pos = sc >= thr
            if pred_pos and lb == 1:
                tp += 1
            elif pred_pos and lb == 0:
                fp += 1
            elif not pred_pos and lb == 1:
                fn += 1
            else:
                tn += 1

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = (
            2 * prec * rec / max(prec + rec, 1e-10)
            if (prec + rec) > 0
            else 0.0
        )

        per_thresh.append(
            {
                "threshold": thr,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    result["per_threshold"] = per_thresh
    return result


# -------------------------------------------------------
# Lead-time metrics
# -------------------------------------------------------


def lead_time_metrics(
    y_pred: list[float],
    seq_ids: list[str],
    pre_onset: list[bool],
    seq_onsets: dict[str, int | None],
    threshold: float = 0.5,
) -> dict[str, object]:
    """Distribution of detection lead times for catastrophic sequences.

    For each seq with onset != None, find the first pre-onset
    token where y_pred >= threshold. Lead time = onset - t.
    Sequences with no pre-onset crossing are "missed".

    Returns summary stats (n_catastrophic, n_detected,
    detection_rate, lead_time mean/median/p25/p75/p95) and
    the raw per-sequence lead times.
    """
    # Group (y_pred, pre_onset) by seq_id preserving order
    grouped: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for i, sid in enumerate(seq_ids):
        grouped[sid].append((y_pred[i], pre_onset[i]))

    leads: list[int] = []
    n_catastrophic = 0
    per_seq: list[dict[str, object]] = []

    for sid, onset in seq_onsets.items():
        if onset is None:
            continue
        n_catastrophic += 1
        tokens = grouped.get(sid, [])

        first_cross: int | None = None
        for t, (score, is_pre) in enumerate(tokens):
            if is_pre and score >= threshold:
                first_cross = t
                break

        if first_cross is not None:
            lead = onset - first_cross
            leads.append(lead)
            per_seq.append({"seq_id": sid, "lead": lead, "detected": True})
        else:
            per_seq.append({"seq_id": sid, "lead": None, "detected": False})

    if n_catastrophic == 0:
        return {
            "threshold": threshold,
            "n_catastrophic": 0,
            "n_detected": 0,
            "detection_rate": None,
            "lead_time_mean": None,
            "lead_time_median": None,
            "lead_time_p25": None,
            "lead_time_p75": None,
            "lead_time_p95": None,
            "per_sequence": per_seq,
        }

    n_detected = len(leads)
    rate = round(n_detected / n_catastrophic, 4)

    if leads:
        arr = np.asarray(leads, dtype=float)
        stats: dict[str, float | None] = {
            "lead_time_mean": round(float(arr.mean()), 4),
            "lead_time_median": round(float(np.median(arr)), 4),
            "lead_time_p25": round(float(np.percentile(arr, 25)), 4),
            "lead_time_p75": round(float(np.percentile(arr, 75)), 4),
            "lead_time_p95": round(float(np.percentile(arr, 95)), 4),
        }
    else:
        stats = {
            "lead_time_mean": None,
            "lead_time_median": None,
            "lead_time_p25": None,
            "lead_time_p75": None,
            "lead_time_p95": None,
        }

    return {
        "threshold": threshold,
        "n_catastrophic": n_catastrophic,
        "n_detected": n_detected,
        "detection_rate": rate,
        **stats,
        "per_sequence": per_seq,
    }


# -------------------------------------------------------
# Baseline comparisons
# -------------------------------------------------------


def baseline_feature_metrics(
    X: list[list[float]],
    y: list[int],
    feat_names: list[str],
    pre_onset: list[bool],
    seq_ids: list[str],
    seq_onsets: dict[str, int | None],
) -> dict[str, dict[str, object]]:
    """Single-feature threshold baselines.

    Uses each feature's raw value as the predicted score.
    Negates inverse-relationship features so higher = more
    dangerous for AUROC computation.
    """
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    results: dict[str, dict[str, object]] = {}

    for feat in BASELINE_FEATURES:
        if feat not in name_to_idx:
            continue

        idx = name_to_idx[feat]
        raw = [row[idx] for row in X]

        # Negate inverse features
        if feat in _INVERSE_FEATURES:
            scores = [-v for v in raw]
        else:
            scores = raw

        tok = token_metrics(y, scores, pre_onset)
        seq = sequence_metrics(scores, seq_ids, pre_onset, seq_onsets)

        results[feat] = {"token": tok, "sequence": seq}

    return results


# -------------------------------------------------------
# Full model evaluation
# -------------------------------------------------------


def evaluate_model(
    model_path: Path,
    results_dir: Path,
    horizon: int = 10,
    nt_onset_frac: float = 0.75,
) -> dict[str, object]:
    """Full evaluation of a trained model.

    Loads model, builds dataset with onset metadata,
    computes token, sequence, and baseline metrics.
    """
    ds = build_dataset(
        results_dir,
        horizon=horizon,
        nt_onset_frac=nt_onset_frac,
    )

    if not ds.X:
        return {"error": "no data"}

    model = xgb.Booster()
    model.load_model(str(model_path))
    dmat = xgb.DMatrix(ds.X, feature_names=ds.feat_names)
    y_pred = model.predict(dmat).tolist()

    tok = token_metrics(ds.y, y_pred, ds.pre_onset)
    seq = sequence_metrics(
        y_pred,
        ds.seq_ids,
        ds.pre_onset,
        ds.seq_onsets,
    )
    leads = {
        f"thr_{t}": lead_time_metrics(
            y_pred, ds.seq_ids, ds.pre_onset, ds.seq_onsets, threshold=t
        )
        for t in (0.3, 0.5, 0.7)
    }
    baselines = baseline_feature_metrics(
        ds.X,
        ds.y,
        ds.feat_names,
        ds.pre_onset,
        ds.seq_ids,
        ds.seq_onsets,
    )

    return {
        "horizon": horizon,
        "n_samples": len(ds.X),
        "token_metrics": tok,
        "sequence_metrics": seq,
        "lead_time": leads,
        "baselines": baselines,
    }


def evaluate_all_horizons(
    model_dir: Path,
    results_dir: Path,
    horizons: list[int] | None = None,
    nt_onset_frac: float = 0.75,
) -> dict[str, object]:
    """Evaluate all trained models across horizons."""
    if horizons is None:
        horizons = [1, 5, 10, 25, 50]

    results: dict[str, object] = {}

    for h in horizons:
        model_path = model_dir / f"hazard_H{h}.json"
        if not model_path.exists():
            logger.warning(f"H={h}: model not found at {model_path}")
            continue

        logger.info(f"Evaluating H={h}...")
        results[f"H{h}"] = evaluate_model(
            model_path,
            results_dir,
            horizon=h,
            nt_onset_frac=nt_onset_frac,
        )

    return results


# -------------------------------------------------------
# NT onset sensitivity analysis
# -------------------------------------------------------


def nt_sensitivity(
    results_dir: Path,
    horizon: int = 10,
    fracs: list[float] | None = None,
) -> dict[str, object]:
    """NT onset fraction sensitivity analysis.

    Retrains and evaluates at each nt_onset_frac value.
    Labels change with the fraction, so retraining is
    required (not just re-evaluation).
    """
    if fracs is None:
        fracs = DEFAULT_NT_FRACS

    results: dict[str, object] = {}

    for frac in fracs:
        logger.info(f"NT sensitivity: frac={frac}, H={horizon}")
        ds = build_dataset(
            results_dir,
            horizon=horizon,
            nt_onset_frac=frac,
        )

        if not ds.X:
            continue

        cv = _cv_metrics(ds.X, ds.y, ds.run_ids, ds.feat_names)

        results[str(frac)] = {
            "nt_onset_frac": frac,
            "n_samples": len(ds.X),
            "event_rate": round(sum(ds.y) / max(len(ds.y), 1), 4),
            **cv,
        }

        logger.info(
            f"  frac={frac}: "
            f"AUROC={cv.get('auroc_mean')}, "
            f"AUPRC={cv.get('auprc_mean')}"
        )

    return results
