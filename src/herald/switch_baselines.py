"""Switch-level prediction baselines and canonical evaluation metrics."""

import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOCKED_BASELINE = "task_ratio_position_bucket"
DUPLICATE_KEY_FIELDS: tuple[str, ...] = (
    "model",
    "task",
    "prompt_id",
    "compressor",
    "ratio",
    "s",
)
PROMPT_GROUP_FIELDS: tuple[str, ...] = ("model", "task", "prompt_id")


@dataclass(frozen=True)
class BaselineSpec:
    """A mean-prediction baseline keyed by train-row fields."""

    name: str
    keys: tuple[str, ...]


BASELINES: tuple[BaselineSpec, ...] = (
    BaselineSpec("global_mean", ()),
    BaselineSpec("ratio", ("ratio",)),
    BaselineSpec("task_ratio", ("task", "ratio")),
    BaselineSpec(
        LOCKED_BASELINE,
        ("task", "ratio", "position_bucket"),
    ),
)


@dataclass(frozen=True)
class SplitRows:
    """Rows for one held-out-compressor baseline evaluation."""

    heldout_compressor: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]


@dataclass(frozen=True)
class MeanBaseline:
    """A fitted train-only grouped-mean baseline."""

    spec: BaselineSpec
    global_mean: float
    means: dict[tuple[object, ...], float]


def leave_one_compressor_splits(
    rows: Sequence[dict[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_group_fraction: float = 0.25,
) -> list[SplitRows]:
    """Build compressor-held-out splits with prompt groups disjoint."""
    if not 0.0 < test_group_fraction < 1.0:
        raise ValueError("test_group_fraction must be in (0, 1)")

    selected = set(compressors) if compressors is not None else None
    all_compressors = sorted(
        {
            str(row["compressor"])
            for row in rows
            if selected is None or str(row.get("compressor")) in selected
        }
    )
    splits: list[SplitRows] = []
    for heldout in all_compressors:
        train: list[dict[str, Any]] = []
        test: list[dict[str, Any]] = []
        for row in rows:
            compressor = str(row.get("compressor"))
            if selected is not None and compressor not in selected:
                continue
            is_test_group = _is_test_group(row, seed, test_group_fraction)
            if compressor == heldout and is_test_group:
                test.append(row)
            elif compressor != heldout and not is_test_group:
                train.append(row)
        if train and test:
            splits.append(SplitRows(heldout, train, test))
    return splits


def evaluate_baselines(
    rows: Sequence[dict[str, Any]],
    *,
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_group_fraction: float = 0.25,
    position_bucket_size: int = 16,
    bootstrap_resamples: int = 200,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate train-only baselines on compressor-held-out splits."""
    selected = set(compressors) if compressors is not None else None
    prepared = [
        _with_position_bucket(row, position_bucket_size)
        for row in rows
        if _as_float(row.get("dq")) is not None
        and (selected is None or str(row.get("compressor")) in selected)
    ]
    splits = leave_one_compressor_splits(
        prepared,
        compressors=compressors,
        seed=seed,
        test_group_fraction=test_group_fraction,
    )

    split_summaries: list[dict[str, Any]] = []
    for split in splits:
        split_summaries.append(
            _evaluate_split(
                split,
                seed=seed,
                bootstrap_resamples=bootstrap_resamples,
            )
        )

    return {
        "config": {
            "seed": seed,
            "test_group_fraction": test_group_fraction,
            "position_bucket_size": position_bucket_size,
            "compressors": list(compressors)
            if compressors is not None
            else None,
            "locked_baseline": LOCKED_BASELINE,
            "bootstrap_resamples": bootstrap_resamples,
        },
        "metadata": _artifact_metadata(prepared, command),
        "n_rows": len(prepared),
        "n_splits": len(split_summaries),
        "row_counts": _row_counts(prepared),
        "duplicate_audit": duplicate_audit(prepared),
        "splits": split_summaries,
    }


def evaluate_switch_predictions(
    rows: Sequence[dict[str, Any]],
    *,
    prediction_key: str = "predicted_dq",
    model_name: str = "candidate",
    compressors: Sequence[str] | None = None,
    seed: int = 0,
    test_group_fraction: float = 0.25,
    position_bucket_size: int = 16,
    bootstrap_resamples: int = 200,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate precomputed switch-level dq predictions canonically."""
    selected = set(compressors) if compressors is not None else None
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if (
            selected is not None
            and str(row.get("compressor")) not in selected
        ):
            continue
        if _as_float(row.get("dq")) is None:
            continue
        prediction = _as_float(row.get(prediction_key))
        if prediction is None:
            raise ValueError(f"missing finite prediction {prediction_key!r}")
        out = _with_position_bucket(row, position_bucket_size)
        out[prediction_key] = prediction
        prepared.append(out)

    splits = leave_one_compressor_splits(
        prepared,
        compressors=compressors,
        seed=seed,
        test_group_fraction=test_group_fraction,
    )
    split_summaries = [
        _evaluate_prediction_split(
            split,
            prediction_key=prediction_key,
            model_name=model_name,
            seed=seed,
            bootstrap_resamples=bootstrap_resamples,
        )
        for split in splits
    ]
    return {
        "config": {
            "seed": seed,
            "test_group_fraction": test_group_fraction,
            "position_bucket_size": position_bucket_size,
            "compressors": list(compressors)
            if compressors is not None
            else None,
            "locked_baseline": LOCKED_BASELINE,
            "bootstrap_resamples": bootstrap_resamples,
            "prediction_key": prediction_key,
            "model_name": model_name,
        },
        "metadata": _artifact_metadata(prepared, command),
        "n_rows": len(prepared),
        "n_splits": len(split_summaries),
        "row_counts": _row_counts(prepared),
        "duplicate_audit": duplicate_audit(prepared),
        "splits": split_summaries,
    }


def make_baseline_lock(
    summary: Mapping[str, Any], *, tolerance: float = 1e-9
) -> dict[str, Any]:
    """Build a deterministic lock file from an evaluator summary."""
    return {
        "version": 1,
        "baseline": LOCKED_BASELINE,
        "metric": "mae",
        "tolerance": tolerance,
        "config": summary["config"],
        "n_rows": summary["n_rows"],
        "n_splits": summary["n_splits"],
        "dataset_fingerprint": summary["metadata"]["dataset_fingerprint"],
        "heldout": _locked_mae_by_heldout(summary),
    }


def compare_to_baseline_lock(
    summary: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    tolerance: float | None = None,
) -> dict[str, Any]:
    """Compare current locked-baseline MAEs to a stored lock."""
    allowed_delta = tolerance
    if allowed_delta is None:
        allowed_delta = _required_float(lock.get("tolerance", 1e-9))
    expected = lock.get("heldout")
    if not isinstance(expected, Mapping):
        raise ValueError("baseline lock must contain a heldout mapping")

    observed = _locked_mae_by_heldout(summary)
    comparisons: list[dict[str, Any]] = []
    passed = True
    for heldout in sorted(set(expected) | set(observed)):
        expected_value = expected.get(heldout)
        observed_value = observed.get(heldout)
        row = {
            "heldout_compressor": heldout,
            "expected_mae": expected_value,
            "observed_mae": observed_value,
            "absolute_delta": None,
            "passed": False,
        }
        if expected_value is not None and observed_value is not None:
            delta = abs(
                _required_float(expected_value)
                - _required_float(observed_value)
            )
            row["absolute_delta"] = delta
            row["passed"] = delta <= allowed_delta
        passed = passed and bool(row["passed"])
        comparisons.append(row)

    fingerprint = summary["metadata"]["dataset_fingerprint"]
    expected_fingerprint = lock.get("dataset_fingerprint")
    fingerprint_match = (
        expected_fingerprint is None or expected_fingerprint == fingerprint
    )
    n_rows_match = lock.get("n_rows") in (None, summary["n_rows"])
    n_splits_match = lock.get("n_splits") in (None, summary["n_splits"])
    passed = passed and fingerprint_match and n_rows_match and n_splits_match
    return {
        "passed": passed,
        "baseline": LOCKED_BASELINE,
        "metric": "mae",
        "tolerance": allowed_delta,
        "dataset_fingerprint_match": fingerprint_match,
        "n_rows_match": n_rows_match,
        "n_splits_match": n_splits_match,
        "comparisons": comparisons,
    }


def baseline_report(summary: Mapping[str, Any]) -> str:
    """Render a Markdown report for ``evaluate_baselines`` output."""
    lines = [
        "# Switch Predictor Baselines",
        "",
        "Train-only baselines for the pre-compression controller target.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(summary["config"], indent=2, sort_keys=True),
        "```",
        "",
        "## Artifact metadata",
        "",
        "```json",
        json.dumps(summary["metadata"], indent=2, sort_keys=True),
        "```",
        "",
        f"Rows: {summary['n_rows']}",
        f"Splits: {summary['n_splits']}",
        "",
        "## Dataset audits",
        "",
        "```json",
        json.dumps(
            {
                "row_counts": summary["row_counts"],
                "duplicate_audit": summary["duplicate_audit"],
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
    ]
    if "lock_comparison" in summary:
        lines.extend(
            [
                "## Baseline lock comparison",
                "",
                "```json",
                json.dumps(
                    summary["lock_comparison"], indent=2, sort_keys=True
                ),
                "```",
                "",
            ]
        )
    lines.extend(_results_table(summary))
    lines.append("")
    return "\n".join(lines)


def duplicate_audit(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count duplicate switch rows by canonical switch identity."""
    counts = Counter(_key(row, DUPLICATE_KEY_FIELDS) for row in rows)
    duplicate_items = [
        (key, count) for key, count in counts.items() if count > 1
    ]
    examples = [
        {"key": list(key), "count": count}
        for key, count in sorted(duplicate_items, key=lambda item: item[0])[
            :10
        ]
    ]
    return {
        "key_fields": list(DUPLICATE_KEY_FIELDS),
        "duplicate_key_count": len(duplicate_items),
        "duplicate_row_count": sum(count - 1 for _, count in duplicate_items),
        "examples": examples,
    }


def fit_mean_baseline(
    rows: Sequence[dict[str, Any]], spec: BaselineSpec
) -> MeanBaseline:
    """Fit a grouped train-mean baseline."""
    values = [_required_float(row["dq"]) for row in rows]
    global_mean = _mean(values)
    groups: dict[tuple[object, ...], list[float]] = defaultdict(list)
    for row in rows:
        groups[_key(row, spec.keys)].append(_required_float(row["dq"]))
    means = {key: _mean(vals) for key, vals in groups.items()}
    return MeanBaseline(spec, global_mean, means)


def predict_mean_baseline(model: MeanBaseline, row: dict[str, Any]) -> float:
    """Predict with grouped means, backing off to broader keys."""
    keys = model.spec.keys
    for width in range(len(keys), -1, -1):
        key = _key(row, keys[:width])
        if key in model.means:
            return model.means[key]
    return model.global_mean


def _evaluate_split(
    split: SplitRows, *, seed: int, bootstrap_resamples: int
) -> dict[str, Any]:
    """Evaluate all baseline specs for one split."""
    baselines = [fit_mean_baseline(split.train, spec) for spec in BASELINES]
    predictions = {
        baseline.spec.name: [
            predict_mean_baseline(baseline, row) for row in split.test
        ]
        for baseline in baselines
    }
    locked_predictions = predictions[LOCKED_BASELINE]
    y_true = [_required_float(row["dq"]) for row in split.test]
    metrics = [
        _metrics_for_predictions(
            spec=baseline.spec,
            y_pred=predictions[baseline.spec.name],
            y_true=y_true,
            locked_pred=locked_predictions,
            rows=split.test,
            seed=_stable_seed(
                seed, split.heldout_compressor, baseline.spec.name
            ),
            bootstrap_resamples=bootstrap_resamples,
        )
        for baseline in baselines
    ]
    train_groups = {_prompt_group(row) for row in split.train}
    test_groups = {_prompt_group(row) for row in split.test}
    return {
        "heldout_compressor": split.heldout_compressor,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_train_prompt_groups": len(train_groups),
        "n_test_prompt_groups": len(test_groups),
        "leakage_audit": leakage_audit(split),
        "baselines": metrics,
    }


def _evaluate_prediction_split(
    split: SplitRows,
    *,
    prediction_key: str,
    model_name: str,
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Evaluate one candidate prediction vector for a split."""
    locked_model = fit_mean_baseline(
        split.train,
        BaselineSpec(LOCKED_BASELINE, ("task", "ratio", "position_bucket")),
    )
    locked_predictions = [
        predict_mean_baseline(locked_model, row) for row in split.test
    ]
    y_true = [_required_float(row["dq"]) for row in split.test]
    y_pred = [_required_float(row[prediction_key]) for row in split.test]
    train_groups = {_prompt_group(row) for row in split.train}
    test_groups = {_prompt_group(row) for row in split.test}
    return {
        "heldout_compressor": split.heldout_compressor,
        "n_train": len(split.train),
        "n_test": len(split.test),
        "n_train_prompt_groups": len(train_groups),
        "n_test_prompt_groups": len(test_groups),
        "leakage_audit": leakage_audit(split),
        "prediction": _metrics_for_predictions(
            spec=BaselineSpec(model_name, ()),
            y_pred=y_pred,
            y_true=y_true,
            locked_pred=locked_predictions,
            rows=split.test,
            seed=_stable_seed(seed, split.heldout_compressor, model_name),
            bootstrap_resamples=bootstrap_resamples,
        ),
    }


def leakage_audit(split: SplitRows) -> dict[str, Any]:
    """Audit train/test split leakage for one held-out compressor split."""
    train_groups = {_prompt_group(row) for row in split.train}
    test_groups = {_prompt_group(row) for row in split.test}
    train_rows = {_key(row, DUPLICATE_KEY_FIELDS) for row in split.train}
    test_rows = {_key(row, DUPLICATE_KEY_FIELDS) for row in split.test}
    return {
        "prompt_group_fields": list(PROMPT_GROUP_FIELDS),
        "prompt_group_overlap_count": len(train_groups & test_groups),
        "row_overlap_count": len(train_rows & test_rows),
        "train_heldout_compressor_rows": sum(
            str(row.get("compressor")) == split.heldout_compressor
            for row in split.train
        ),
        "test_nonheldout_compressor_rows": sum(
            str(row.get("compressor")) != split.heldout_compressor
            for row in split.test
        ),
    }


def _metrics_for_predictions(
    *,
    spec: BaselineSpec,
    y_pred: Sequence[float],
    y_true: Sequence[float],
    locked_pred: Sequence[float],
    rows: Sequence[dict[str, Any]],
    seed: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    """Compute regression, controller, ranking, and bootstrap metrics."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    locked = np.asarray(locked_pred, dtype=np.float64)
    err = pred - y
    locked_err = locked - y
    mae = _mean_abs(err)
    locked_mae = _mean_abs(locked_err)
    return {
        "baseline": spec.name,
        "mae": mae,
        "rmse": _root_mean_square(err),
        "mean_y": _mean_array(y),
        "mean_pred": _mean_array(pred),
        "relative_mae_improvement_vs_locked": _relative_improvement(
            locked_mae, mae
        ),
        "controller_metrics": {
            "dq_gt_0": _controller_metrics(y_true, y_pred, 0.0, strict=True),
            "dq_ge_0_5": _controller_metrics(
                y_true, y_pred, 0.5, strict=False
            ),
        },
        "ranking_metrics": {
            "dq_gt_0": _ranking_metrics(y_true, y_pred, 0.0, strict=True)
        },
        "calibration_bins": _calibration_bins(y_true, y_pred),
        "bootstrap_ci": _cluster_bootstrap_ci(
            y_true=y_true,
            y_pred=y_pred,
            locked_pred=locked_pred,
            rows=rows,
            seed=seed,
            n_resamples=bootstrap_resamples,
        ),
    }


def _controller_metrics(
    y_true: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Compute top-decile enrichment for a damage threshold."""
    labels = _labels(y_true, threshold, strict=strict)
    n = len(labels)
    if n == 0:
        raise ValueError("cannot score empty predictions")
    top_n = max(1, math.ceil(0.10 * n))
    ranked = sorted(range(n), key=lambda idx: scores[idx], reverse=True)
    top_labels = [labels[idx] for idx in ranked[:top_n]]
    prevalence = _rate(labels)
    top_rate = _rate(top_labels)
    lift = None if prevalence == 0.0 else top_rate / prevalence
    return {
        "threshold": threshold,
        "operator": ">" if strict else ">=",
        "n": n,
        "top_decile_n": top_n,
        "prevalence": prevalence,
        "top_decile_rate": top_rate,
        "top_decile_lift": lift,
    }


def _ranking_metrics(
    y_true: Sequence[float],
    scores: Sequence[float],
    threshold: float,
    *,
    strict: bool,
) -> dict[str, Any]:
    """Compute binary ranking diagnostics for a damage threshold."""
    labels = _labels(y_true, threshold, strict=strict)
    return {
        "threshold": threshold,
        "operator": ">" if strict else ">=",
        "prevalence": _rate(labels),
        "auprc": _average_precision(labels, scores),
        "recall_at_10_fpr": _recall_at_fpr(labels, scores, max_fpr=0.10),
    }


def _calibration_bins(
    y_true: Sequence[float], scores: Sequence[float], *, bins: int = 10
) -> list[dict[str, Any]]:
    """Summarize observed damage within predicted-risk bins."""
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    n = len(y_true)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda idx: scores[idx])
    bin_count = min(bins, n)
    chunks = np.array_split(np.asarray(order, dtype=np.int64), bin_count)
    out: list[dict[str, Any]] = []
    labels = _labels(y_true, 0.0, strict=True)
    for idx, chunk in enumerate(chunks):
        indices = [_required_int(i) for i in chunk.tolist()]
        chunk_scores = [scores[i] for i in indices]
        chunk_y = [y_true[i] for i in indices]
        chunk_labels = [labels[i] for i in indices]
        out.append(
            {
                "bin": idx,
                "n": len(indices),
                "score_min": min(chunk_scores),
                "score_max": max(chunk_scores),
                "mean_pred": _mean(chunk_scores),
                "mean_dq": _mean(chunk_y),
                "damage_rate": _rate(chunk_labels),
            }
        )
    return out


def _average_precision(
    labels: Sequence[int], scores: Sequence[float]
) -> float | None:
    """Compute average precision for binary labels and risk scores."""
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(
        range(len(labels)), key=lambda idx: scores[idx], reverse=True
    )
    true_positives = 0
    precision_sum = 0.0
    for rank, idx in enumerate(ranked, start=1):
        if labels[idx]:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _recall_at_fpr(
    labels: Sequence[int], scores: Sequence[float], *, max_fpr: float
) -> float | None:
    """Return best recall with false-positive rate at most max_fpr."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(
        range(len(labels)), key=lambda idx: scores[idx], reverse=True
    )
    true_positives = 0
    false_positives = 0
    best_recall = 0.0
    for idx in ranked:
        if labels[idx]:
            true_positives += 1
        else:
            false_positives += 1
        fpr = false_positives / negatives
        if fpr <= max_fpr:
            best_recall = max(best_recall, true_positives / positives)
    return best_recall


def _cluster_bootstrap_ci(
    *,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    locked_pred: Sequence[float],
    rows: Sequence[dict[str, Any]],
    seed: int,
    n_resamples: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap relative MAE improvement by prompt cluster."""
    if n_resamples < 0:
        raise ValueError("n_resamples must be non-negative")
    groups_by_key: dict[tuple[object, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups_by_key[_prompt_group(row)].append(idx)
    groups = list(groups_by_key.values())
    observed = _relative_mae_for_indices(
        y_true, y_pred, locked_pred, list(range(len(y_true)))
    )
    result = {
        "metric": "relative_mae_improvement_vs_locked",
        "cluster_fields": list(PROMPT_GROUP_FIELDS),
        "confidence_level": confidence_level,
        "n_bootstrap": n_resamples,
        "n_clusters": len(groups),
        "observed": observed,
        "low": None,
        "high": None,
    }
    if n_resamples == 0 or not groups:
        return result

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        indices: list[int] = []
        draws = rng.integers(0, len(groups), size=len(groups))
        for draw in draws.tolist():
            indices.extend(groups[_required_int(draw)])
        value = _relative_mae_for_indices(
            y_true, y_pred, locked_pred, indices
        )
        if value is not None:
            samples.append(value)
    if samples:
        alpha = (1.0 - confidence_level) / 2.0
        result["low"] = _quantile(samples, alpha)
        result["high"] = _quantile(samples, 1.0 - alpha)
    return result


def _relative_mae_for_indices(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    locked_pred: Sequence[float],
    indices: Sequence[int],
) -> float | None:
    """Compute relative MAE improvement on a subset of row indices."""
    if not indices:
        return None
    candidate_errors = [abs(y_pred[idx] - y_true[idx]) for idx in indices]
    locked_errors = [abs(locked_pred[idx] - y_true[idx]) for idx in indices]
    locked_mae = _mean(locked_errors)
    candidate_mae = _mean(candidate_errors)
    return _relative_improvement(locked_mae, candidate_mae)


def _relative_improvement(
    baseline_mae: float, candidate_mae: float
) -> float | None:
    """Return relative MAE improvement over a baseline MAE."""
    if baseline_mae == 0.0:
        return None
    return (baseline_mae - candidate_mae) / baseline_mae


def _labels(
    values: Sequence[float], threshold: float, *, strict: bool
) -> list[int]:
    """Build binary damage labels from continuous dq values."""
    labels: list[int] = []
    for value in values:
        if strict:
            labels.append(_flag(value > threshold))
        else:
            labels.append(_flag(value >= threshold))
    return labels


def _rate(labels: Sequence[int]) -> float:
    """Return the positive fraction for binary labels."""
    if not labels:
        raise ValueError("cannot average empty labels")
    return sum(labels) / len(labels)


def _row_counts(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Count rows by compressor for auditability."""
    by_compressor = Counter(str(row.get("compressor")) for row in rows)
    return {
        "total": len(rows),
        "by_compressor": dict(sorted(by_compressor.items())),
    }


def _artifact_metadata(
    rows: Sequence[dict[str, Any]], command: Sequence[str] | None
) -> dict[str, Any]:
    """Build reproducibility metadata for evaluator artifacts."""
    metadata = _git_metadata()
    metadata.update(
        {
            "dataset_fingerprint": dataset_fingerprint(rows),
            "command": list(command) if command is not None else None,
        }
    )
    return metadata


def dataset_fingerprint(rows: Sequence[dict[str, Any]]) -> str:
    """Hash all evaluated rows in a deterministic JSON representation."""
    digest = hashlib.sha256()
    digest.update(str(len(rows)).encode())
    for row in sorted(rows, key=_row_sort_key):
        encoded = json.dumps(
            {key: _jsonable(row.get(key)) for key in sorted(row)},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(encoded.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    """Return git SHA and dirty flag when available."""
    sha = _run_git(["rev-parse", "HEAD"])
    unstaged_clean = _git_clean(["diff", "--quiet"])
    staged_clean = _git_clean(["diff", "--cached", "--quiet"])
    dirty = None
    if unstaged_clean is not None and staged_clean is not None:
        dirty = not (unstaged_clean and staged_clean)
    return {"git_sha": sha, "dirty": dirty}


def _run_git(args: Sequence[str]) -> str | None:
    """Run a read-only git command and return stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (
        subprocess.CalledProcessError,
        OSError,
        subprocess.TimeoutExpired,
    ):
        return None
    return result.stdout.strip()


def _git_clean(args: Sequence[str]) -> bool | None:
    """Return whether a git cleanliness check exits successfully."""
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode in (0, 1):
        return result.returncode == 0
    return None


def _locked_mae_by_heldout(summary: Mapping[str, Any]) -> dict[str, float]:
    """Extract locked-baseline MAE values from a summary."""
    values: dict[str, float] = {}
    splits = summary.get("splits")
    if not isinstance(splits, Sequence):
        raise ValueError("summary must contain splits")
    for split in splits:
        if not isinstance(split, Mapping):
            continue
        heldout = str(split.get("heldout_compressor"))
        baselines = split.get("baselines")
        if not isinstance(baselines, Sequence):
            continue
        for metric in baselines:
            if not isinstance(metric, Mapping):
                continue
            if metric.get("baseline") == LOCKED_BASELINE:
                values[heldout] = _required_float(metric.get("mae"))
    return values


def _results_table(summary: Mapping[str, Any]) -> list[str]:
    """Render the main metrics table."""
    lines = [
        "## Leave-One-Compressor-Out Results",
        "",
        (
            "| Held-out compressor | Baseline | Train rows | Test rows | "
            "MAE | Rel MAE vs locked | Lift dq > 0 | Lift dq >= 0.5 | "
            "AUPRC | Recall at 10% FPR |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    splits = summary.get("splits")
    if not isinstance(splits, Sequence):
        return lines
    for split in splits:
        if not isinstance(split, Mapping):
            continue
        heldout = split["heldout_compressor"]
        for metric in split["baselines"]:
            controller = metric["controller_metrics"]
            ranking = metric["ranking_metrics"]["dq_gt_0"]
            lines.append(
                "| "
                f"{heldout} | {metric['baseline']} | "
                f"{split['n_train']} | {split['n_test']} | "
                f"{_fmt(metric['mae'])} | "
                f"{_fmt(metric['relative_mae_improvement_vs_locked'])} | "
                f"{_fmt(controller['dq_gt_0']['top_decile_lift'])} | "
                f"{_fmt(controller['dq_ge_0_5']['top_decile_lift'])} | "
                f"{_fmt(ranking['auprc'])} | "
                f"{_fmt(ranking['recall_at_10_fpr'])} |"
            )
    return lines


def _with_position_bucket(
    row: dict[str, Any], bucket_size: int
) -> dict[str, Any]:
    """Return a copy with integer position bucket added."""
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    out = dict(row)
    s = _required_float(row["s"])
    out["position_bucket"] = _position_bucket(s, bucket_size)
    return out


def _is_test_group(
    row: dict[str, Any], seed: int, test_group_fraction: float
) -> bool:
    """Return whether a prompt group belongs to the deterministic test set."""
    key = "\0".join(
        [
            str(seed),
            str(row.get("model")),
            str(row.get("task")),
            str(row.get("prompt_id")),
        ]
    )
    digest = hashlib.sha256(key.encode()).digest()
    value = _unit_interval_from_digest(digest)
    return value < test_group_fraction


def _mean(values: Sequence[float]) -> float:
    """Return a finite mean for a non-empty float sequence."""
    if not values:
        raise ValueError("cannot average an empty sequence")
    return _mean_array(np.asarray(values, dtype=np.float64))


def _mean_array(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return a finite mean for a non-empty NumPy vector."""
    if values.size == 0:
        raise ValueError("cannot average an empty array")
    try:
        result = values.mean().item()
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute mean") from exc
    return _required_float(result)


def _mean_abs(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return the mean absolute value for a vector."""
    try:
        abs_values = np.abs(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute absolute errors") from exc
    return _mean_array(abs_values)


def _root_mean_square(values: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return root mean square for a vector."""
    try:
        squared = values * values
        mean_square = _mean_array(squared)
        result = np.sqrt(mean_square).item()
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute root mean square") from exc
    return _required_float(result)


def _position_bucket(s: float, bucket_size: int) -> int:
    """Return the lower edge of the position bucket containing ``s``."""
    try:
        return math.floor(s / bucket_size) * bucket_size
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"invalid position {s!r}") from exc


def _unit_interval_from_digest(digest: bytes) -> float:
    """Map the first eight digest bytes into [0, 1)."""
    try:
        numerator = int.from_bytes(digest[:8], "big")
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid digest") from exc
    return numerator / 18_446_744_073_709_551_616.0


def _key(row: dict[str, Any], keys: Sequence[str]) -> tuple[object, ...]:
    """Return a hashable row key for grouped means."""
    return tuple(row.get(key) for key in keys)


def _prompt_group(row: dict[str, Any]) -> tuple[object, ...]:
    """Return the prompt cluster key used for split and bootstrap audits."""
    return _key(row, PROMPT_GROUP_FIELDS)


def _row_sort_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable string key for row fingerprint ordering."""
    return tuple(str(row.get(field)) for field in DUPLICATE_KEY_FIELDS)


def _jsonable(value: object) -> object:
    """Return a stable JSON scalar for hashing."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_seed(seed: int, *parts: str) -> int:
    """Derive a NumPy-compatible deterministic seed from labels."""
    digest = hashlib.sha256("\0".join([str(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _required_int(value: object) -> int:
    """Return an integer or raise a clear error."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"expected integer, got {value!r}") from exc
    if isinstance(value, float) and value.is_integer():
        try:
            return int(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"expected integer, got {value!r}") from exc
    raise ValueError(f"expected integer, got {value!r}")


def _quantile(values: Sequence[float], q: float) -> float:
    """Return a finite quantile for numeric values."""
    try:
        result = np.quantile(values, q).item()
    except (TypeError, ValueError) as exc:
        raise ValueError("could not compute quantile") from exc
    return _required_float(result)


def _flag(condition: bool) -> int:
    """Return 1 for true and 0 for false."""
    return 1 if condition else 0


def _required_float(value: object) -> float:
    """Return a finite float or raise a clear error."""
    out = _as_float(value)
    if out is None:
        raise ValueError(f"expected finite float, got {value!r}")
    return out


def _as_float(value: object) -> float | None:
    """Return a finite float when conversion is possible."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _fmt(value: object) -> str:
    """Format report values compactly."""
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f}"
