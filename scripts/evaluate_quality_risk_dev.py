"""Evaluate development OOF quality-risk skill on quarantined prompts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

SCHEMA_VERSION = "herald.quality_risk_development_evaluation.v1"
CHECKPOINTS = (8, 16, 32, 64, 128)
BASELINES = ("global_mean", "action_only", "action_clock")
BASELINE_COLUMNS = {
    "global_mean": "pred_global_mean",
    "action_only": "pred_action_only",
    "action_clock": "pred_action_clock",
}
CANDIDATES = ("causal_xgb", "tcn")
CANDIDATE_COLUMNS = {
    "causal_xgb": "pred_causal_xgb",
    "tcn": "pred_tcn",
}
RESAMPLES = 10000
SEED = 314159


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--tabular-root", type=Path, required=True)
    parser.add_argument("--tcn-root", type=Path, default=None)
    parser.add_argument("--label-audit", type=Path, required=True)
    parser.add_argument("--feature-audit", type=Path, required=True)
    parser.add_argument(
        "--calibration-method",
        choices=("auto", "isotonic", "platt"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_predictions(
    tabular_root: Path, tcn_root: Path | None
) -> pd.DataFrame:
    tabular_columns = [
        "run_id",
        "token_pos",
        "prompt_id",
        "task",
        "press",
        "compression_ratio",
        "damage",
        "catastrophe",
        "run_length",
        "weight",
        "pred_global_mean",
        "pred_action_only",
        "pred_action_clock",
        "pred_causal_xgb",
    ]
    frame = (
        ds.dataset(  # type: ignore[no-untyped-call]
            tabular_root / "oof_predictions.parquet", format="parquet"
        )
        .to_table(columns=tabular_columns)
        .to_pandas()
    )
    if tcn_root is not None:
        tcn = (
            ds.dataset(  # type: ignore[no-untyped-call]
                tcn_root / "oof_predictions.parquet", format="parquet"
            )
            .to_table(columns=["run_id", "token_pos", "pred_tcn"])
            .to_pandas()
        )
        frame = frame.merge(
            tcn, on=["run_id", "token_pos"], how="left", validate="one_to_one"
        )
        if frame["pred_tcn"].isna().any():
            raise ValueError("TCN/tabular OOF identity mismatch")
    return frame


def checkpoint_weights(frame: pd.DataFrame) -> np.ndarray:
    """Prompt-equal, then action-equal weights (one row per run)."""
    actions_per_prompt = frame.groupby("prompt_id", sort=False)[
        "run_id"
    ].transform("size")
    raw = 1.0 / actions_per_prompt.to_numpy(dtype=np.float64)
    return (raw * len(raw) / raw.sum()).astype(np.float64)


def logloss_rows(truth: np.ndarray, prob: np.ndarray) -> np.ndarray:
    clipped = np.clip(prob, 1e-12, 1 - 1e-12)
    return -(truth * np.log(clipped) + (1 - truth) * np.log(1 - clipped))


def weighted_auc(truth: np.ndarray, score: np.ndarray) -> float:
    """Pooled AUROC via tie-averaged Mann-Whitney ranks."""
    order = np.argsort(score, kind="mergesort")
    ranked_truth = truth[order]
    ranked_score = score[order]
    n = len(ranked_truth)
    ranks = np.empty(n, dtype=np.float64)
    start = 0
    while start < n:
        end = start + 1
        while end < n and ranked_score[end] == ranked_score[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2
        start = end
    pos = ranked_truth == 1
    n_pos = int(pos.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(
        (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def preorder_tie_groups(
    score: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Fixed score order and tie groups for bootstrap reuse."""
    order = np.argsort(score, kind="mergesort")
    ranked_score = score[order]
    n = len(ranked_score)
    groups: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and ranked_score[end] == ranked_score[start]:
            end += 1
        groups.append((start, end))
        start = end
    return order, groups


def delong_stats(
    truth: np.ndarray,
    weight: np.ndarray,
    order: np.ndarray,
    groups: list[tuple[int, int]],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """AUROC, DeLong variance, and per-row placements (original row order).

    Returns placements as full-length arrays with NaN where inapplicable.
    """
    truth_o = truth[order].astype(np.float64)
    weight_o = weight[order].astype(np.float64)
    pos_weight = float(np.sum(weight_o[truth_o == 1]))
    neg_weight = float(np.sum(weight_o[truth_o == 0]))
    n = len(truth_o)
    empty = (
        float("nan"),
        float("nan"),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )
    if pos_weight == 0 or neg_weight == 0:
        return empty
    auc_num = 0.0
    cum_neg = 0.0
    v10_o = np.full(n, np.nan)
    for start, end in groups:
        segment_truth = truth_o[start:end]
        segment_weight = weight_o[start:end]
        group_pos = float(np.sum(segment_weight[segment_truth == 1]))
        group_neg = float(np.sum(segment_weight[segment_truth == 0]))
        auc_num += group_pos * (cum_neg + 0.5 * group_neg)
        placement = (cum_neg + 0.5 * group_neg) / neg_weight
        for index in range(start, end):
            if truth_o[index] == 1:
                v10_o[index] = placement
        cum_neg += group_neg
    v01_o = np.full(n, np.nan)
    cum_pos = 0.0
    for start, end in reversed(groups):
        segment_truth = truth_o[start:end]
        segment_weight = weight_o[start:end]
        group_pos = float(np.sum(segment_weight[segment_truth == 1]))
        group_neg = float(np.sum(segment_weight[segment_truth == 0]))
        placement = (cum_pos + 0.5 * group_pos) / pos_weight
        for index in range(start, end):
            if truth_o[index] == 0:
                v01_o[index] = placement
        cum_pos += group_pos
    auc = auc_num / (pos_weight * neg_weight)
    pos_mask = truth_o == 1
    neg_mask = truth_o == 0
    mean_10 = float(np.sum(v10_o[pos_mask] * weight_o[pos_mask]) / pos_weight)
    mean_01 = float(np.sum(v01_o[neg_mask] * weight_o[neg_mask]) / neg_weight)
    var_10 = float(
        np.sum(weight_o[pos_mask] * (v10_o[pos_mask] - mean_10) ** 2)
        / pos_weight
    )
    var_01 = float(
        np.sum(weight_o[neg_mask] * (v01_o[neg_mask] - mean_01) ** 2)
        / neg_weight
    )
    undo = np.empty(n, dtype=np.int64)
    undo[order] = np.arange(n, dtype=np.int64)
    return (
        float(auc),
        float(var_10 / pos_weight + var_01 / neg_weight),
        v10_o[undo],
        v01_o[undo],
    )


def group_ids_from_order(score: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Consecutive group ids along the fixed score order (ties share)."""
    ranked_score = score[order]
    change = np.ones(len(ranked_score), dtype=bool)
    change[1:] = ranked_score[1:] != ranked_score[:-1]
    return (np.cumsum(change) - 1).astype(np.int64)


def delong_stats_fast(
    truth: np.ndarray,
    weight: np.ndarray,
    order: np.ndarray,
    group_id: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Vectorized DeLong core for bootstrap loops (original row order)."""
    truth = truth.astype(np.float64)
    weight = weight.astype(np.float64)
    truth_o = truth[order]
    weight_o = weight[order]
    n_groups = int(group_id.max()) + 1
    pos_o = (truth_o == 1).astype(np.float64)
    group_pos = np.bincount(
        group_id, weights=weight_o * pos_o, minlength=n_groups
    )
    group_w = np.bincount(group_id, weights=weight_o, minlength=n_groups)
    group_neg = group_w - group_pos
    pos_total = float(group_pos.sum())
    neg_total = float(group_neg.sum())
    n = len(truth_o)
    empty = (
        float("nan"),
        float("nan"),
        np.full(n, np.nan),
        np.full(n, np.nan),
    )
    if pos_total == 0 or neg_total == 0:
        return empty
    cum_neg = np.concatenate([[0.0], np.cumsum(group_neg)])[:-1]
    place10_g = (cum_neg + 0.5 * group_neg) / neg_total
    auc = float(
        np.sum(group_pos * (cum_neg + 0.5 * group_neg))
        / (pos_total * neg_total)
    )
    cum_pos = np.concatenate([[0.0], np.cumsum(group_pos[::-1])])[:-1][::-1]
    place01_g = (cum_pos + 0.5 * group_pos) / pos_total
    v10_o = np.where(pos_o == 1, place10_g[group_id], np.nan)
    v01_o = np.where(pos_o == 0, place01_g[group_id], np.nan)
    pos_mask = pos_o == 1
    neg_mask = pos_o == 0
    mean_10 = float(np.sum(v10_o[pos_mask] * weight_o[pos_mask]) / pos_total)
    mean_01 = float(np.sum(v01_o[neg_mask] * weight_o[neg_mask]) / neg_total)
    var_10 = float(
        np.sum(weight_o[pos_mask] * (v10_o[pos_mask] - mean_10) ** 2)
        / pos_total
    )
    var_01 = float(
        np.sum(weight_o[neg_mask] * (v01_o[neg_mask] - mean_01) ** 2)
        / neg_total
    )
    undo = np.empty(n, dtype=np.int64)
    undo[order] = np.arange(n, dtype=np.int64)
    return (
        float(auc),
        float(var_10 / pos_total + var_01 / neg_total),
        v10_o[undo],
        v01_o[undo],
    )


def delong_diff_var(
    truth: np.ndarray,
    weight: np.ndarray,
    stats_a: tuple[float, float, np.ndarray, np.ndarray],
    stats_b: tuple[float, float, np.ndarray, np.ndarray],
) -> float:
    """DeLong variance of AUC(a) - AUC(b) on the same weighted rows.

    Placement arrays must share row order (as returned by delong_stats).
    """
    truth = truth.astype(np.float64)
    weight = weight.astype(np.float64)
    _, var_a, v10a, v01a = stats_a
    _, var_b, v10b, v01b = stats_b
    pos_mask = truth == 1
    neg_mask = truth == 0
    pos_weight = float(np.sum(weight[pos_mask]))
    neg_weight = float(np.sum(weight[neg_mask]))
    mean_a10 = np.sum(v10a[pos_mask] * weight[pos_mask]) / pos_weight
    mean_b10 = np.sum(v10b[pos_mask] * weight[pos_mask]) / pos_weight
    mean_a01 = np.sum(v01a[neg_mask] * weight[neg_mask]) / neg_weight
    mean_b01 = np.sum(v01b[neg_mask] * weight[neg_mask]) / neg_weight
    cov_10 = float(
        np.sum(
            weight[pos_mask]
            * (v10a[pos_mask] - mean_a10)
            * (v10b[pos_mask] - mean_b10)
        )
        / pos_weight
    )
    cov_01 = float(
        np.sum(
            weight[neg_mask]
            * (v01a[neg_mask] - mean_a01)
            * (v01b[neg_mask] - mean_b01)
        )
        / neg_weight
    )
    return float(
        var_a + var_b - 2 * (cov_10 / pos_weight + cov_01 / neg_weight)
    )


def delong_auc_and_var(
    truth: np.ndarray, score: np.ndarray, weight: np.ndarray
) -> tuple[float, float]:
    """Pooled AUROC and DeLong variance under frequency weights."""
    order, groups = preorder_tie_groups(score)
    auc, var, _, _ = delong_stats(truth, weight, order, groups)
    return auc, var


def stratified_draws(tasks: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    draws = np.empty((count, len(tasks)), dtype=np.int32)
    for bootstrap_index in range(count):
        offset = 0
        for group in groups:
            sample = rng.choice(group, size=len(group), replace=True)
            draws[bootstrap_index, offset : offset + len(group)] = sample
            offset += len(group)
    return draws


def simultaneous_bound(
    matrix: np.ndarray,
    draws: np.ndarray,
    side: Literal["upper", "lower", "two-sided"],
) -> dict[str, Any]:
    if not np.isfinite(matrix).all():
        raise ValueError("nonfinite prompt matrix")
    point = matrix.mean(axis=0)
    standard_error = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
    bootstrap_mean = matrix[draws].mean(axis=1)
    bootstrap_error = matrix[draws].std(axis=1, ddof=1) / np.sqrt(len(matrix))
    studentized = np.divide(
        bootstrap_mean - point,
        bootstrap_error,
        out=np.zeros_like(bootstrap_mean),
        where=bootstrap_error > 0,
    )
    if side == "upper":
        critical = float(np.quantile(studentized.max(axis=1), 0.95))
        bound = point + critical * standard_error
        return {
            "point": point.tolist(),
            "upper": bound.tolist(),
            "critical": critical,
        }
    if side == "lower":
        critical = float(np.quantile((-studentized).max(axis=1), 0.95))
        bound = point - critical * standard_error
        return {
            "point": point.tolist(),
            "lower": bound.tolist(),
            "critical": critical,
        }
    critical = float(np.quantile(np.abs(studentized).max(axis=1), 0.95))
    return {
        "point": point.tolist(),
        "lower": (point - critical * standard_error).tolist(),
        "upper": (point + critical * standard_error).tolist(),
        "critical": critical,
    }


def auroc_bootstrap(
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    prompt_idx: np.ndarray,
    n_prompts: int,
    draws: np.ndarray,
    pairs: list[tuple[str, str]],
) -> dict[str, Any]:
    """Pooled AUROC draws with DeLong studentization per draw.

    draws: (n_draws, n_prompts) prompt positions; pairs: (a, b) for a - b.
    """
    n_draws = draws.shape[0]
    counts = np.zeros((n_draws, n_prompts), dtype=np.float64)
    counts[np.arange(n_draws)[:, None], draws] += 1.0
    precomputed = {}
    for name, score in scores.items():
        order, _ = preorder_tie_groups(score)
        precomputed[name] = (order, group_ids_from_order(score, order))
    names = list(scores)
    auc_draws = {name: np.full(n_draws, np.nan) for name in names}
    var_draws = {name: np.full(n_draws, np.nan) for name in names}
    diff_draws = {
        f"{a}_minus_{b}": np.full(n_draws, np.nan) for a, b in pairs
    }
    diff_var_draws = {
        f"{a}_minus_{b}": np.full(n_draws, np.nan) for a, b in pairs
    }
    chunk = 1000
    for start in range(0, n_draws, chunk):
        block = counts[start : start + chunk]
        for offset in range(block.shape[0]):
            weight = block[offset][prompt_idx]
            stats = {}
            for name in names:
                order, group_id = precomputed[name]
                stats[name] = delong_stats_fast(
                    truth, weight, order, group_id
                )
                auc_draws[name][start + offset] = stats[name][0]
                var_draws[name][start + offset] = stats[name][1]
            for a, b in pairs:
                key = f"{a}_minus_{b}"
                diff_draws[key][start + offset] = stats[a][0] - stats[b][0]
                diff_var_draws[key][start + offset] = delong_diff_var(
                    truth, weight, stats[a], stats[b]
                )
    return {
        "auc_draws": auc_draws,
        "var_draws": var_draws,
        "diff_draws": diff_draws,
        "diff_var_draws": diff_var_draws,
    }


def studentized_max_t_bound(
    point: np.ndarray,
    se_point: np.ndarray,
    boot: np.ndarray,
    se_boot: np.ndarray,
    side: Literal["upper", "lower", "two-sided"],
) -> dict[str, Any]:
    """Simultaneous bound over dims from bootstrap draws (draws x dims).

    Degenerate draws (nonfinite or nonpositive SE) are dropped and counted.
    """
    if np.any(se_point <= 0) or not np.isfinite(se_point).all():
        raise ValueError("nonpositive AUROC point standard error")
    keep = np.isfinite(boot).all(axis=1) & np.all(se_boot > 0, axis=1)
    dropped = int(boot.shape[0] - keep.sum())
    if keep.sum() < 1000:
        raise ValueError("too few usable AUROC bootstrap draws")
    boot, se_boot = boot[keep], se_boot[keep]
    studentized = (boot - point) / se_boot
    if side == "upper":
        critical = float(np.quantile(studentized.max(axis=1), 0.95))
        return {
            "point": point.tolist(),
            "upper": (point + critical * se_point).tolist(),
            "critical": critical,
            "bootstrap_draws_kept": int(keep.sum()),
            "bootstrap_draws_dropped": dropped,
        }
    if side == "lower":
        critical = float(np.quantile((-studentized).max(axis=1), 0.95))
        return {
            "point": point.tolist(),
            "lower": (point - critical * se_point).tolist(),
            "critical": critical,
            "bootstrap_draws_kept": int(keep.sum()),
            "bootstrap_draws_dropped": dropped,
        }
    critical = float(np.quantile(np.abs(studentized).max(axis=1), 0.95))
    return {
        "point": point.tolist(),
        "lower": (point - critical * se_point).tolist(),
        "upper": (point + critical * se_point).tolist(),
        "critical": critical,
        "bootstrap_draws_kept": int(keep.sum()),
        "bootstrap_draws_dropped": dropped,
    }


def prompt_weighted_means(
    prompt_ids: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> pd.Series:
    """Weight-averaged values per prompt, sorted by prompt id."""
    selected = pd.DataFrame(
        {
            "prompt_id": prompt_ids,
            "weighted": values * weights,
            "weight": weights,
        }
    )
    grouped = selected.groupby("prompt_id", sort=True)
    return grouped["weighted"].sum() / grouped["weight"].sum()


def prompt_matrix(
    frame: pd.DataFrame,
    values: np.ndarray,
    prompt_ids: list[str],
) -> np.ndarray:
    selected = pd.DataFrame(
        {
            "prompt_id": frame["prompt_id"].to_numpy(),
            "value": values,
        }
    )
    grouped = selected.groupby("prompt_id", sort=True)["value"].mean()
    return np.column_stack(
        [grouped.loc[prompt_ids].to_numpy(dtype=np.float64)]
    )


def fit_calibrator(
    scores: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
    method: str = "auto",
) -> tuple[str, Any, float]:
    scores = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    candidates: dict[str, tuple[Any, float]] = {}
    if method in ("auto", "isotonic"):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso_pred = iso.fit_transform(scores, truth)
        candidates["isotonic"] = (
            iso,
            float(
                np.average(
                    logloss_rows(truth, np.clip(iso_pred, 1e-12, 1 - 1e-12)),
                    weights=weights,
                )
            ),
        )
    if method in ("auto", "platt"):
        logit = np.log(scores / (1 - scores)).reshape(-1, 1)
        platt = LogisticRegression(C=1e6, solver="lbfgs")
        platt_pred = platt.fit(logit, truth).predict_proba(logit)[:, 1]
        candidates["platt"] = (
            platt,
            float(
                np.average(
                    logloss_rows(
                        truth, np.clip(platt_pred, 1e-12, 1 - 1e-12)
                    ),
                    weights=weights,
                )
            ),
        )
    name = min(candidates, key=lambda key: candidates[key][1])
    return name, candidates[name][0], candidates[name][1]


def apply_calibrator(
    name: str, calibrator: Any, scores: np.ndarray
) -> np.ndarray:
    scores = np.clip(scores.astype(np.float64), 1e-12, 1 - 1e-12)
    if name == "isotonic":
        return np.clip(
            calibrator.predict(scores).astype(np.float64), 1e-12, 1 - 1e-12
        )
    logit = np.log(scores / (1 - scores)).reshape(-1, 1)
    return np.clip(
        calibrator.predict_proba(logit)[:, 1].astype(np.float64),
        1e-12,
        1 - 1e-12,
    )


def expected_calibration_error(
    truth: np.ndarray, prob: np.ndarray, bins: int = 10
) -> float:
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = (prob > low) & (prob <= high)
        if selected.sum() == 0:
            continue
        total += selected.mean() * abs(
            truth[selected].mean() - prob[selected].mean()
        )
    return float(total)


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    lock = load_json(args.protocol_lock)
    if lock.get("schema_version") != "herald.quality_risk.v1":
        raise ValueError("unexpected protocol schema")
    if tuple(lock["checkpoints"]) != CHECKPOINTS:
        raise ValueError("protocol checkpoints changed")
    label_audit = load_json(args.label_audit)
    feature_audit = load_json(args.feature_audit)
    audits_pass = (
        label_audit.get("pass") is True and feature_audit.get("pass") is True
    )
    if not audits_pass:
        raise ValueError("prerequisite audits did not pass")
    frame = load_predictions(args.tabular_root, args.tcn_root)
    candidates = [
        name
        for name in CANDIDATES
        if CANDIDATE_COLUMNS[name] in frame.columns
    ]
    truth_all = frame["damage"].to_numpy(dtype=np.float64)
    checkpoints: dict[str, Any] = {}
    checkpoint_row_masks: dict[str, np.ndarray] = {}
    for checkpoint in CHECKPOINTS:
        mask = (frame["token_pos"].to_numpy() == checkpoint) & (
            frame["run_length"].to_numpy() > checkpoint
        )
        checkpoint_row_masks[str(checkpoint)] = mask
        checkpoints[str(checkpoint)] = {"rows": int(mask.sum())}
    calibration: dict[str, Any] = {}
    calibrated_oof: dict[str, np.ndarray] = {}
    for candidate in candidates:
        scores = frame[CANDIDATE_COLUMNS[candidate]].to_numpy(
            dtype=np.float64
        )
        name, calibrator, oof_loss = fit_calibrator(
            scores,
            truth_all,
            frame["weight"].to_numpy(dtype=np.float64),
            args.calibration_method,
        )
        calibrated_oof[candidate] = apply_calibrator(name, calibrator, scores)
        calibration[candidate] = {
            "method": name,
            "oof_log_loss": oof_loss,
        }
    probabilities: dict[str, np.ndarray] = {
        name: frame[column].to_numpy(dtype=np.float64)
        for name, column in BASELINE_COLUMNS.items()
    }
    for name in candidates:
        probabilities[f"calibrated_{name}"] = calibrated_oof[name]
    results: dict[str, Any] = {}
    superiority: dict[str, Any] = {}
    hard_names: dict[str, str] = {}
    for checkpoint in CHECKPOINTS:
        key = str(checkpoint)
        mask = checkpoint_row_masks[key]
        sub = frame[mask].reset_index(drop=True)
        weights = checkpoint_weights(sub)
        truth = sub["damage"].to_numpy(dtype=np.float64)
        probs = {name: probabilities[name][mask] for name in probabilities}
        losses = {
            name: float(np.average(logloss_rows(truth, p), weights=weights))
            for name, p in probs.items()
        }
        aurocs = {name: weighted_auc(truth, p) for name, p in probs.items()}
        briers = {
            name: float(np.average((truth - p) ** 2, weights=weights))
            for name, p in probs.items()
        }
        hard = min(BASELINES, key=lambda name: losses[name])
        hard_names[key] = hard
        results[key] = {
            "rows": int(mask.sum()),
            "prompts": int(sub["prompt_id"].nunique()),
            "damage_rate": float(truth.mean()),
            "log_loss": losses,
            "auroc": aurocs,
            "brier": briers,
            "hard_comparator": hard,
        }
    prompt_ids_all = sorted(frame["prompt_id"].unique())
    tasks_all = (
        frame.groupby("prompt_id")["task"]
        .first()
        .loc[prompt_ids_all]
        .to_numpy()
    )
    draws = stratified_draws(tasks_all, RESAMPLES, SEED)
    universe = sorted(
        set.intersection(
            *(
                set(
                    frame.loc[
                        checkpoint_row_masks[str(checkpoint)], "prompt_id"
                    ].unique()
                )
                for checkpoint in CHECKPOINTS
            )
        )
    )
    universe_index = pd.Index(prompt_ids_all).get_indexer(universe)
    for candidate in [f"calibrated_{name}" for name in candidates]:
        short = candidate.replace("calibrated_", "")
        diffs: dict[str, Any] = {}
        for comparator in ("action_clock", "hard"):
            columns = []
            for checkpoint in CHECKPOINTS:
                key = str(checkpoint)
                mask = checkpoint_row_masks[key]
                sub = frame[mask].reset_index(drop=True)
                weights = checkpoint_weights(sub)
                truth = sub["damage"].to_numpy(dtype=np.float64)
                if comparator == "hard":
                    comp_probs = probabilities[hard_names[key]][mask]
                else:
                    comp_probs = probabilities[comparator][mask]
                cand_probs = probabilities[candidate][mask]
                row_diff = logloss_rows(truth, cand_probs) - logloss_rows(
                    truth, comp_probs
                )
                prompt_ids = sorted(sub["prompt_id"].unique())
                selected = pd.DataFrame(
                    {
                        "prompt_id": sub["prompt_id"].to_numpy(),
                        "value": row_diff,
                    }
                )
                grouped = (
                    selected.groupby("prompt_id", sort=True)["value"]
                    .mean()
                    .loc[prompt_ids]
                    .to_numpy(dtype=np.float64)
                )
                full = np.full(len(universe), np.nan)
                index = pd.Index(universe).get_indexer(prompt_ids)
                full[index] = grouped
                columns.append(full)
            matrix = np.column_stack(columns)
            diffs[comparator] = simultaneous_bound(
                matrix, draws[:, universe_index], "upper"
            )
        superiority[short] = diffs
    universe_draws = draws[:, universe_index]
    auroc_bounds: dict[str, Any] = {}
    for candidate in [f"calibrated_{name}" for name in candidates]:
        short = candidate.replace("calibrated_", "")
        boot_auc: list[np.ndarray] = []
        se_auc: list[np.ndarray] = []
        point_auc: list[float] = []
        se_point_auc: list[float] = []
        boot_diff_clock: list[np.ndarray] = []
        se_diff_clock: list[np.ndarray] = []
        boot_diff_hard: list[np.ndarray] = []
        se_diff_hard: list[np.ndarray] = []
        point_diff_clock: list[float] = []
        se_point_diff_clock: list[float] = []
        point_diff_hard: list[float] = []
        se_point_diff_hard: list[float] = []
        for checkpoint in CHECKPOINTS:
            key = str(checkpoint)
            mask = checkpoint_row_masks[key]
            sub_prompts = frame.loc[mask, "prompt_id"].to_numpy()
            prompt_idx = pd.Index(universe).get_indexer(sub_prompts)
            truth = frame.loc[mask, "damage"].to_numpy(dtype=np.float64)
            hard = hard_names[key]
            if hard == "global_mean":
                raise ValueError("hard comparator is degenerate at " + key)
            model_names = list(
                dict.fromkeys([candidate, "action_clock", hard])
            )
            scores = {name: probabilities[name][mask] for name in model_names}
            unit = np.ones(len(truth))
            boot = auroc_bootstrap(
                truth,
                scores,
                prompt_idx,
                len(universe),
                universe_draws,
                [
                    (candidate, "action_clock"),
                    (candidate, hard),
                ],
            )
            point_c, var_c = delong_auc_and_var(
                truth, scores[candidate], unit
            )
            point_a, var_a = delong_auc_and_var(
                truth, scores["action_clock"], unit
            )
            point_h, var_h = delong_auc_and_var(truth, scores[hard], unit)
            order_c, groups_c = preorder_tie_groups(scores[candidate])
            order_a, groups_a = preorder_tie_groups(scores["action_clock"])
            order_h, groups_h = preorder_tie_groups(scores[hard])
            _, _, v10c, v01c = delong_stats(truth, unit, order_c, groups_c)
            _, _, v10a, v01a = delong_stats(truth, unit, order_a, groups_a)
            _, _, v10h, v01h = delong_stats(truth, unit, order_h, groups_h)
            var_dc = delong_diff_var(
                truth,
                unit,
                (point_c, var_c, v10c, v01c),
                (point_a, var_a, v10a, v01a),
            )
            var_dh = delong_diff_var(
                truth,
                unit,
                (point_c, var_c, v10c, v01c),
                (point_h, var_h, v10h, v01h),
            )
            point_auc.append(point_c)
            se_point_auc.append(float(np.sqrt(var_c)))
            boot_auc.append(boot["auc_draws"][candidate])
            se_auc.append(np.sqrt(boot["var_draws"][candidate]))
            point_diff_clock.append(point_c - point_a)
            se_point_diff_clock.append(float(np.sqrt(max(var_dc, 0.0))))
            boot_diff_clock.append(
                boot["diff_draws"][f"{candidate}_minus_action_clock"]
            )
            se_diff_clock.append(
                np.sqrt(
                    np.maximum(
                        boot["diff_var_draws"][
                            f"{candidate}_minus_action_clock"
                        ],
                        0.0,
                    )
                )
            )
            point_diff_hard.append(point_c - point_h)
            se_point_diff_hard.append(float(np.sqrt(max(var_dh, 0.0))))
            boot_diff_hard.append(
                boot["diff_draws"][f"{candidate}_minus_{hard}"]
            )
            se_diff_hard.append(
                np.sqrt(
                    np.maximum(
                        boot["diff_var_draws"][f"{candidate}_minus_{hard}"],
                        0.0,
                    )
                )
            )
        auroc_bounds[short] = {
            "auroc_lower": studentized_max_t_bound(
                np.asarray(point_auc),
                np.asarray(se_point_auc),
                np.column_stack(boot_auc),
                np.column_stack(se_auc),
                "lower",
            ),
            "vs_action_clock": studentized_max_t_bound(
                np.asarray(point_diff_clock),
                np.asarray(se_point_diff_clock),
                np.column_stack(boot_diff_clock),
                np.column_stack(se_diff_clock),
                "lower",
            ),
            "vs_hard": studentized_max_t_bound(
                np.asarray(point_diff_hard),
                np.asarray(se_point_diff_hard),
                np.column_stack(boot_diff_hard),
                np.column_stack(se_diff_hard),
                "lower",
            ),
        }
    gates: dict[str, Any] = {}
    subgroups: dict[str, Any] = {}
    slopes: dict[str, Any] = {}
    eces: dict[str, Any] = {}
    earliness: dict[str, Any] = {}
    for candidate in [f"calibrated_{name}" for name in candidates]:
        short = candidate.replace("calibrated_", "")
        mask32 = checkpoint_row_masks["32"]
        sub32 = frame[mask32].reset_index(drop=True)
        truth32 = sub32["damage"].to_numpy(dtype=np.float64)
        cand32 = probabilities[candidate][mask32]
        clock32 = probabilities["action_clock"][mask32]
        subgroup_improvement: dict[str, Any] = {}
        for dimension in ("task", "press", "compression_ratio"):
            groups: dict[str, float] = {}
            for value, indices in sub32.groupby(
                dimension, observed=True
            ).indices.items():
                rows = np.asarray(indices, dtype=np.int64)
                improvement = weighted_auc(
                    truth32[rows], cand32[rows]
                ) - weighted_auc(truth32[rows], clock32[rows])
                groups[str(value)] = float(improvement)
            subgroup_improvement[dimension] = groups
        subgroups[short] = subgroup_improvement
        slope_columns = []
        for checkpoint in CHECKPOINTS:
            key = str(checkpoint)
            mask = checkpoint_row_masks[key]
            sub = frame[mask].reset_index(drop=True)
            weights = checkpoint_weights(sub)
            truth = sub["damage"].to_numpy(dtype=np.float64)
            prob = probabilities[candidate][mask]
            numerator = prompt_weighted_means(
                sub["prompt_id"].to_numpy(), truth * prob, weights
            )
            denominator = prompt_weighted_means(
                sub["prompt_id"].to_numpy(), prob * prob, weights
            )
            column = (
                numerator.loc[universe] / denominator.loc[universe]
            ).to_numpy(dtype=np.float64)
            slope_columns.append(column)
        slope_matrix = np.column_stack(slope_columns)
        slopes[short] = simultaneous_bound(
            slope_matrix, draws[:, universe_index], "two-sided"
        )
        ece_points = []
        ece_boots = []
        for checkpoint in CHECKPOINTS:
            key = str(checkpoint)
            mask = checkpoint_row_masks[key]
            sub_prompts = frame.loc[mask, "prompt_id"].to_numpy()
            prompt_idx = pd.Index(universe).get_indexer(sub_prompts)
            truth = frame.loc[mask, "damage"].to_numpy(dtype=np.float64)
            prob = probabilities[candidate][mask]
            edges = np.quantile(prob, np.linspace(0, 1, 11))
            edges[0], edges[-1] = -np.inf, np.inf
            point = expected_calibration_error(truth, prob)
            ece_points.append(point)
            counts = np.zeros(
                (universe_draws.shape[0], len(universe)), dtype=np.float64
            )
            counts[
                np.arange(universe_draws.shape[0])[:, None], universe_draws
            ] += 1.0
            draws_ece = np.full(universe_draws.shape[0], np.nan)
            for draw_index in range(universe_draws.shape[0]):
                repeat = counts[draw_index][prompt_idx].astype(np.int64)
                resampled_rows = np.repeat(np.arange(len(truth)), repeat)
                draws_ece[draw_index] = expected_calibration_error(
                    truth[resampled_rows], prob[resampled_rows]
                )
            ece_boots.append(draws_ece)
        ece_matrix = np.column_stack(ece_boots)
        ece_point = np.asarray(ece_points)
        ece_scale = ece_matrix.std(axis=0, ddof=1)
        ece_scale = np.where(ece_scale > 0, ece_scale, np.nan)
        studentized = (ece_matrix - ece_point) / ece_scale
        critical = float(np.nanquantile(np.nanmax(studentized, axis=1), 0.95))
        eces[short] = {
            "method": "max-T percentile-t with bootstrap scale",
            "point": ece_point.tolist(),
            "upper": (ece_point + critical * ece_scale).tolist(),
            "critical": critical,
        }
        earliness_curve: dict[str, Any] = {}
        for split_name, split_mask in (
            ("overall", np.ones(len(frame), dtype=bool)),
            ("catastrophe", frame["catastrophe"].to_numpy() == 1),
            ("no_catastrophe", frame["catastrophe"].to_numpy() == 0),
        ):
            curve = []
            for checkpoint in CHECKPOINTS:
                key = str(checkpoint)
                mask = checkpoint_row_masks[key] & split_mask
                sub = frame[mask]
                truth = sub["damage"].to_numpy(dtype=np.float64)
                cand_p = probabilities[candidate][mask]
                hard_p = probabilities[hard_names[key]][mask]
                brier_cand = float(np.mean((truth - cand_p) ** 2))
                brier_hard = float(np.mean((truth - hard_p) ** 2))
                curve.append(
                    {
                        "rows": int(mask.sum()),
                        "auroc_candidate": weighted_auc(truth, cand_p),
                        "auroc_action_clock": weighted_auc(
                            truth, probabilities["action_clock"][mask]
                        ),
                        "brier_skill_vs_hard": float(
                            1 - brier_cand / brier_hard
                        ),
                    }
                )
            earliness_curve[split_name] = curve
        earliness[short] = earliness_curve
        late = [str(t) for t in CHECKPOINTS if t >= 16]
        late_index = [list(map(str, CHECKPOINTS)).index(t) for t in late]
        gate2_logloss = all(
            superiority[short][comparator]["upper"][i] < 0
            for comparator in ("action_clock", "hard")
            for i in late_index
        )
        gate2_auroc = all(
            auroc_bounds[short][key]["lower"][i] > 0
            for key in ("vs_action_clock", "vs_hard")
            for i in late_index
        )
        gate2 = bool(gate2_logloss and gate2_auroc)
        gate3 = bool(
            auroc_bounds[short]["auroc_lower"]["lower"][
                list(map(str, CHECKPOINTS)).index("32")
            ]
            >= 0.70
        )
        gate4 = bool(
            all(
                value >= 0
                for dimension in subgroup_improvement.values()
                for value in dimension.values()
            )
        )
        slope_bound = slopes[short]
        gate5_slope = bool(
            all(
                lower <= 1 <= upper
                for lower, upper in zip(
                    slope_bound["lower"], slope_bound["upper"], strict=True
                )
            )
        )
        gate5_ece = bool(all(value <= 0.05 for value in eces[short]["upper"]))
        gates[short] = {
            "2_superiority": gate2,
            "3_usefulness_floor": gate3,
            "4_no_harm": gate4,
            "5_calibration": bool(gate5_slope and gate5_ece),
            "5_slope_contains_1": gate5_slope,
            "5_ece_upper_within_005": gate5_ece,
        }
    nominee = None
    qualified = [
        short
        for short in gates
        if all(
            gates[short][gate]
            for gate in (
                "2_superiority",
                "3_usefulness_floor",
                "4_no_harm",
                "5_calibration",
            )
        )
    ]
    if len(qualified) == 1:
        nominee = qualified[0]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_gates_except_loto_confirmation_unread",
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "tabular_report_sha256": sha256_file(
            args.tabular_root / "report.json"
        ),
        "label_audit_sha256": sha256_file(args.label_audit),
        "feature_audit_sha256": sha256_file(args.feature_audit),
        "checkpoints": results,
        "calibration": calibration,
        "superiority_log_loss": superiority,
        "superiority_auroc": auroc_bounds,
        "subgroup_auroc_improvement_t32": subgroups,
        "calibration_slope": slopes,
        "calibration_ece": eces,
        "earliness": earliness,
        "gates": gates,
        "qualified_before_loto": qualified,
        "nominee_before_loto": nominee,
        "bootstrap_prompt_universe": len(universe),
        "bootstrap_prompts_total": len(prompt_ids_all),
        "confirmation_prompts_projected": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
