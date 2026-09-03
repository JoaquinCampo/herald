"""Independently audit H5-U0 repair, timing, and label-blind metrics."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from herald.attention_features import AttentionTap, tap_feature_names
from herald.features import IncrementalDerived, derive_features
from herald.magnitude_v2 import hash_prompt_ids, sha256_file

SCHEMA_VERSION = "herald.magnitude_e1_h5_u0_independent_audit.v1"
EXPECTED_HASHES = {
    "protocol_lock_sha256": (
        "10ed71fabceb9743f3dec3d9d67efc3035274fccc3394fca279c1ab212ec3f9f"
    ),
    "repair_report_sha256": (
        "76a6a82d9cccd3af01fdfb1b37d315af7cafcba12c001b989f736c17369a9e72"
    ),
    "source_oof_sha256": (
        "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
    ),
    "sweep_config_sha256": (
        "8c228094df9f36cd582752070d578fbfe60f49cd53553b3b507076d17c874408"
    ),
    "reference_manifest_sha256": (
        "5f3d65d72ff506cfdd8105dd4a26cb065c2234c4ede3263bacf0547b1dc5cf88"
    ),
    "features_source_sha256": (
        "75b6f42ba622323cdfa22b816d61f7b790fb383f058087933c01b6a955124528"
    ),
    "attention_source_sha256": (
        "5233b9bc13afcee585c3ba70df27240da9ad5bd4d4746141243404ac0e157c39"
    ),
}
EXPECTED_PROMPTS = 154
EXPECTED_ROWS = 59_182
EXPECTED_FINITE_ROWS = 59_028
EXPECTED_COMPLETE_ROWS = 58_720
FOLDS = (0, 1, 2, 3, 4)
SPANS = (8, 32)
TARGETS = tuple(f"kl_prev_ewma_{span}" for span in SPANS)
PREDICTORS = (
    "kl_prev",
    "kl_prev_delta",
    "kl_prev_accel",
    "kl_prev_rmean_8",
    "kl_prev_rstd_8",
    "kl_prev_rmin_8",
    "kl_prev_rmax_8",
    "kl_prev_rmedian_8",
    "kl_prev_riqr_8",
    "kl_prev_slope_8",
    "kl_prev_rmean_32",
    "kl_prev_rstd_32",
    "kl_prev_rmin_32",
    "kl_prev_rmax_32",
    "kl_prev_rmedian_32",
    "kl_prev_riqr_32",
    "kl_prev_slope_32",
)
R2_THRESHOLD = 0.99
METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class AuditPrompt:
    prompt_id: str
    fold: int
    predictors: np.ndarray
    targets: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--features-source", type=Path, required=True)
    parser.add_argument("--attention-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def independent_ewma(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    output = np.full(values.shape, np.nan, dtype=np.float64)
    accumulator: float | None = None
    for position in range(len(values)):
        value = float(values[position])
        if np.isnan(value):
            continue
        if accumulator is None:
            accumulator = value
        else:
            accumulator *= 1.0 - alpha
            accumulator += alpha * value
        output[position] = accumulator
    return output.astype(np.float32)


def independent_diff(values: np.ndarray) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    output[0] = np.nan
    output[1:] = values[1:] - values[:-1]
    return output


def independent_rolling(
    values: np.ndarray, span: int
) -> dict[str, np.ndarray]:
    names = ("rmean", "rstd", "rmin", "rmax", "rmedian", "riqr", "slope")
    outputs = {
        name: np.full(len(values), np.nan, dtype=np.float64) for name in names
    }
    for stop in range(1, len(values) + 1):
        start = max(0, stop - span)
        window = values[start:stop]
        observed = ~np.isnan(window)
        finite = window[observed]
        if not len(finite):
            continue
        index = stop - 1
        outputs["rmean"][index] = float(np.mean(finite))
        outputs["rstd"][index] = float(np.std(finite))
        outputs["rmin"][index] = float(np.min(finite))
        outputs["rmax"][index] = float(np.max(finite))
        outputs["rmedian"][index] = float(np.median(finite))
        lower, upper = np.percentile(finite, [25, 75])
        outputs["riqr"][index] = float(upper - lower)
        if len(finite) == 1:
            outputs["slope"][index] = 0.0
        else:
            positions = np.arange(len(window), dtype=np.float64)[observed]
            outputs["slope"][index] = float(
                np.polyfit(positions, finite, 1)[0]
            )
    return outputs


def independent_kl_columns(kl: np.ndarray) -> dict[str, np.ndarray]:
    values = kl.astype(np.float64)
    columns: dict[str, np.ndarray] = {
        "kl_prev": values,
        "kl_prev_delta": independent_diff(values),
        "kl_prev_accel": independent_diff(independent_diff(values)),
    }
    for span in SPANS:
        columns[f"kl_prev_ewma_{span}"] = independent_ewma(kl, span)
        for stat, output in independent_rolling(values, span).items():
            columns[f"kl_prev_{stat}_{span}"] = output
    return {
        name: np.asarray(value, dtype=np.float32)
        for name, value in columns.items()
    }


def ordered_float32(values: np.ndarray) -> np.ndarray:
    bits = (
        values.astype(np.float32, copy=False)
        .view(np.uint32)
        .astype(np.uint64)
    )
    sign = np.uint64(1 << 31)
    return np.where(
        (bits & sign) != 0,
        np.uint64(0xFFFF_FFFF) - bits,
        bits + sign,
    )


def maximum_ulp(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape:
        raise ValueError("ULP arrays have different shapes")
    finite_left = np.isfinite(left)
    finite_right = np.isfinite(right)
    if not np.array_equal(finite_left, finite_right):
        raise ValueError("independent source missing masks differ")
    if not np.any(finite_left):
        return 0
    ordered_left = ordered_float32(left[finite_left])
    ordered_right = ordered_float32(right[finite_right])
    return int(
        np.max(
            np.where(
                ordered_left >= ordered_right,
                ordered_left - ordered_right,
                ordered_right - ordered_left,
            )
        )
    )


def source_columns(kl: np.ndarray) -> tuple[dict[str, np.ndarray], bool]:
    raw = kl.astype(np.float32).reshape(-1, 1)
    batch, names = derive_features(
        raw,
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    state = IncrementalDerived(
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    online_rows = [state.update(row)[0] for row in raw]
    online = np.stack(online_rows)
    by_name = {name: batch[:, index] for index, name in enumerate(names)}
    targets_exact = all(
        np.array_equal(
            by_name[target],
            online[:, names.index(target)],
            equal_nan=True,
        )
        for target in TARGETS
    )
    return by_name, targets_exact


def roster(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    table = pq.read_table(path, columns=["prompt_id", "fold"])  # type: ignore[no-untyped-call]
    if table.column_names != ["prompt_id", "fold"]:
        raise ValueError("outcome column was materialized")
    frame = table.to_pandas().drop_duplicates()
    if int((frame.groupby("prompt_id")["fold"].nunique() != 1).sum()):
        raise ValueError("prompt fold is not unique")
    unique = frame.drop_duplicates("prompt_id")
    prompt_folds = {
        str(row.prompt_id): int(row.fold)
        for row in unique.itertuples(index=False)
    }
    prompts = tuple(sorted(prompt_folds))
    if len(prompts) != EXPECTED_PROMPTS:
        raise ValueError("development roster changed")
    return prompts, prompt_folds


def array_digests(manifest: dict[str, Any]) -> dict[str, str]:
    entries = manifest.get("artifact_files")
    if not isinstance(entries, list):
        raise ValueError("reference manifest entries are missing")
    output: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("reference manifest entry is invalid")
        path = str(entry.get("path", ""))
        if path.endswith(".npy"):
            if path in output:
                raise ValueError(f"duplicate array manifest entry: {path}")
            output[path] = str(entry.get("sha256", ""))
    return output


def load_independent_data(
    prompts: tuple[str, ...],
    prompt_folds: dict[str, int],
    reference_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[AuditPrompt], dict[str, Any]]:
    digests = array_digests(manifest)
    data: list[AuditPrompt] = []
    total_rows = 0
    finite_rows = 0
    complete_rows = 0
    source_online_exact = True
    source_max_ulp = {name: 0 for name in (*PREDICTORS, *TARGETS)}
    for prompt_id in prompts:
        relative = f"{prompt_id}.npy"
        path = reference_dir / relative
        if sha256_file(path) != digests.get(relative):
            raise ValueError(f"reference array hash mismatch: {relative}")
        persisted = np.load(path, allow_pickle=False)
        if persisted.dtype != np.float16 or persisted.shape[1] != 20:
            raise ValueError(f"reference array schema mismatch: {relative}")
        kl = persisted[:, 13].astype(np.float32)
        independent = independent_kl_columns(kl)
        source, targets_exact = source_columns(kl)
        source_online_exact = source_online_exact and targets_exact
        for name in (*PREDICTORS, *TARGETS):
            source_max_ulp[name] = max(
                source_max_ulp[name],
                maximum_ulp(source[name], independent[name]),
            )
        predictor_matrix = np.column_stack(
            [independent[name] for name in PREDICTORS]
        )
        targets = {target: independent[target] for target in TARGETS}
        complete = np.isfinite(predictor_matrix).all(axis=1) & np.isfinite(
            targets[TARGETS[0]]
        )
        if not np.any(complete):
            raise ValueError(f"no independent complete rows: {prompt_id}")
        data.append(
            AuditPrompt(
                prompt_id=prompt_id,
                fold=prompt_folds[prompt_id],
                predictors=predictor_matrix[complete].astype(np.float64),
                targets={
                    name: values[complete].astype(np.float64)
                    for name, values in targets.items()
                },
            )
        )
        total_rows += len(kl)
        finite_rows += int(np.isfinite(kl).sum())
        complete_rows += int(complete.sum())
    return data, {
        "total_rows": total_rows,
        "finite_rows": finite_rows,
        "complete_rows": complete_rows,
        "source_online_exact": source_online_exact,
        "source_max_ulp": source_max_ulp,
    }


def flatten_training(
    prompts: list[AuditPrompt], target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([prompt.predictors for prompt in prompts], axis=0)
    y = np.concatenate([prompt.targets[target] for prompt in prompts])
    weights = np.concatenate(
        [
            np.repeat(
                1.0 / len(prompt.targets[target]), len(prompt.targets[target])
            )
            for prompt in prompts
        ]
    )
    return x, y, weights


def independent_fold_audit(
    data: list[AuditPrompt], target: str, fold: int
) -> dict[str, Any]:
    training = [prompt for prompt in data if prompt.fold != fold]
    held_out = [prompt for prompt in data if prompt.fold == fold]
    train_x, train_y, weights = flatten_training(training, target)
    denominator = float(np.sum(weights))
    mean_x = np.sum(train_x * weights[:, None], axis=0) / denominator
    variance_x = (
        np.sum((train_x - mean_x) ** 2 * weights[:, None], axis=0)
        / denominator
    )
    scale_x = np.sqrt(variance_x)
    active = scale_x != 0.0
    standardized = (train_x[:, active] - mean_x[active]) / scale_x[active]
    design = np.concatenate(
        [np.ones((len(train_x), 1)), standardized], axis=1
    )
    sqrt_weight = np.sqrt(weights)
    weighted_design = design * sqrt_weight[:, None]
    weighted_y = train_y * sqrt_weight
    coefficients, _, rank, _ = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None
    )
    q_matrix, r_matrix = np.linalg.qr(weighted_design, mode="reduced")
    qr_coefficients = np.linalg.solve(r_matrix, q_matrix.T @ weighted_y)
    training_mean = float(np.sum(train_y * weights) / denominator)

    model_errors: list[float] = []
    baseline_errors: list[float] = []
    qr_prediction_difference = 0.0
    for prompt in held_out:
        standardized_held = (
            prompt.predictors[:, active] - mean_x[active]
        ) / scale_x[active]
        held_design = np.concatenate(
            [np.ones((len(standardized_held), 1)), standardized_held], axis=1
        )
        prediction = held_design @ coefficients
        qr_prediction = held_design @ qr_coefficients
        qr_prediction_difference = max(
            qr_prediction_difference,
            float(np.max(np.abs(prediction - qr_prediction))),
        )
        target_values = prompt.targets[target]
        model_errors.append(float(np.mean((target_values - prediction) ** 2)))
        baseline_errors.append(
            float(np.mean((target_values - training_mean) ** 2))
        )
    mean_model = float(np.mean(model_errors))
    mean_baseline = float(np.mean(baseline_errors))
    residual_fraction = mean_model / mean_baseline
    leave_one_out_r2 = [
        1.0
        - (
            (sum(model_errors) - model_errors[index])
            / (sum(baseline_errors) - baseline_errors[index])
        )
        for index in range(len(held_out))
    ]
    top_five_share = float(
        sum(sorted(model_errors, reverse=True)[:5]) / sum(model_errors)
    )
    return {
        "fold": fold,
        "training_prompts": len(training),
        "held_out_prompts": len(held_out),
        "training_rows": int(len(train_y)),
        "held_out_rows": int(
            sum(len(prompt.targets[target]) for prompt in held_out)
        ),
        "active_predictors": [
            name
            for name, keep in zip(PREDICTORS, active, strict=True)
            if keep
        ],
        "design_rank": int(rank),
        "design_condition_number": float(np.linalg.cond(weighted_design)),
        "mean_prompt_model_mse": mean_model,
        "mean_prompt_baseline_mse": mean_baseline,
        "residual_fraction": residual_fraction,
        "r2": 1.0 - residual_fraction,
        "qr_max_prediction_difference": qr_prediction_difference,
        "held_out_leave_one_prompt_out_r2_min": float(min(leave_one_out_r2)),
        "held_out_leave_one_prompt_out_r2_max": float(max(leave_one_out_r2)),
        "all_held_out_leave_one_prompt_out_pass": all(
            value <= R2_THRESHOLD for value in leave_one_out_r2
        ),
        "top_five_prompt_model_error_share": top_five_share,
    }


def compare_primary_metrics(
    primary: dict[str, Any], independent: dict[str, list[dict[str, Any]]]
) -> float:
    primary_metrics = primary.get("fold_metrics")
    if not isinstance(primary_metrics, dict):
        raise ValueError("primary fold metrics are missing")
    maximum_difference = 0.0
    metric_names = (
        "mean_prompt_model_mse",
        "mean_prompt_baseline_mse",
        "residual_fraction",
        "r2",
    )
    for target in TARGETS:
        primary_rows = primary_metrics.get(target)
        if not isinstance(primary_rows, list):
            raise ValueError(f"primary target metrics are missing: {target}")
        by_fold = {int(row["fold"]): row for row in primary_rows}
        for row in independent[target]:
            expected = by_fold[row["fold"]]
            for name in (
                "training_prompts",
                "held_out_prompts",
                "training_rows",
                "held_out_rows",
                "design_rank",
                "active_predictors",
            ):
                if row[name] != expected[name]:
                    raise ValueError(
                        f"primary count differs for {target} fold "
                        f"{row['fold']} {name}"
                    )
            for name in metric_names:
                maximum_difference = max(
                    maximum_difference,
                    abs(float(row[name]) - float(expected[name])),
                )
    return maximum_difference


def synthetic_ewma_audit() -> dict[str, Any]:
    cases = {
        "all_nan": np.full(5, np.nan, dtype=np.float32),
        "leading_nan": np.asarray([np.nan, 0.2, 0.4, 0.1], dtype=np.float32),
        "internal_nan": np.asarray(
            [0.1, 0.2, np.nan, 0.4, 0.3], dtype=np.float32
        ),
        "one_finite_value": np.asarray([0.25], dtype=np.float32),
        "long_deterministic_stream": np.concatenate(
            [
                np.asarray([np.nan], dtype=np.float32),
                np.linspace(-0.25, 0.75, 257, dtype=np.float32),
            ]
        ),
    }
    results: dict[str, dict[str, int]] = {}
    for name, values in cases.items():
        source, online_exact = source_columns(values)
        if not online_exact:
            raise ValueError(f"synthetic batch/online mismatch: {name}")
        results[name] = {
            target: maximum_ulp(
                source[target],
                independent_ewma(values, span),
            )
            for target, span in zip(TARGETS, SPANS, strict=True)
        }
    reset_prompts = (
        np.asarray([np.nan, 0.9, 0.1, 0.3], dtype=np.float32),
        np.asarray([np.nan, 0.2, 0.4], dtype=np.float32),
    )
    reset_max_ulp = 0
    for values in reset_prompts:
        source, online_exact = source_columns(values)
        if not online_exact:
            raise ValueError("synthetic reset batch/online mismatch")
        for target, span in zip(TARGETS, SPANS, strict=True):
            reset_max_ulp = max(
                reset_max_ulp,
                maximum_ulp(source[target], independent_ewma(values, span)),
            )
    return {
        "case_max_ulp": results,
        "prompt_reset_max_ulp": reset_max_ulp,
        "pass": reset_max_ulp <= 1
        and all(
            value <= 1 for case in results.values() for value in case.values()
        ),
    }


def attention_timing_audit() -> dict[str, Any]:
    import torch
    from transformers import DynamicCache, LlamaConfig, LlamaForCausalLM

    config = cast(
        Any,
        LlamaConfig(  # type: ignore[no-untyped-call]
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
            attn_implementation="sdpa",
        ),
    )
    torch.manual_seed(0)
    model = cast(Any, LlamaForCausalLM)(config)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("H5-U0 timing audit must be CPU-only")
    prompt = torch.arange(9).unsqueeze(0) % 128

    tap = AttentionTap(model, layer_indices=[0, 2])

    def capture(tokens: list[int]) -> np.ndarray:
        tap.begin(prompt_lens=[int(prompt.shape[1])])
        cache = DynamicCache()
        with torch.no_grad():
            model(prompt, past_key_values=cache, use_cache=True)
            for token in tokens[:-1]:
                model(
                    torch.tensor([[token]], dtype=torch.long),
                    past_key_values=cache,
                    use_cache=True,
                )
        matrix, names = tap.matrix()
        if names != tap_feature_names():
            raise ValueError("AttentionTap feature roster changed")
        if matrix.shape != (1, len(tokens), len(names)):
            raise ValueError("AttentionTap row alignment changed")
        return cast(np.ndarray, matrix[0])

    causal_cut = 2
    left = capture([7, 11, 13, 17, 19])
    right = capture([7, 11, 97, 23, 29])
    reset = capture([7, 11, 13, 17, 19])
    tap.remove()
    prefix_difference = float(
        np.max(np.abs(left[: causal_cut + 1] - right[: causal_cut + 1]))
    )
    later_different_cells = int(
        np.count_nonzero(left[causal_cut + 1 :] != right[causal_cut + 1 :])
    )
    reset_difference = float(np.max(np.abs(left - reset)))
    checks = {
        "prefix_array_exact": np.array_equal(
            left[: causal_cut + 1], right[: causal_cut + 1]
        ),
        "later_divergence_observed": later_different_cells > 0,
        "begin_reset_array_exact": np.array_equal(left, reset),
        "row_count_exact": left.shape == (5, 16),
        "cpu_only": True,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "causal_cut": causal_cut,
        "rows": int(left.shape[0]),
        "columns": int(left.shape[1]),
        "prefix_max_abs_difference": prefix_difference,
        "later_different_cells": later_different_cells,
        "reset_max_abs_difference": reset_difference,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H5-U0 independent audit: {args.output}"
        )
    paths = {
        "protocol_lock_sha256": args.protocol_lock,
        "repair_report_sha256": args.repair_report,
        "source_oof_sha256": args.source_oof,
        "sweep_config_sha256": args.sweep_config,
        "reference_manifest_sha256": args.reference_manifest,
        "features_source_sha256": args.features_source,
        "attention_source_sha256": args.attention_source,
    }
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    if actual_hashes != EXPECTED_HASHES:
        raise ValueError("independent H5-U0 input hash changed")
    primary = load_json(args.repair_report)
    config = load_json(args.sweep_config)
    manifest = load_json(args.reference_manifest)
    prompts, prompt_folds = roster(args.source_oof)
    data, source_parity = load_independent_data(
        prompts,
        prompt_folds,
        args.reference_dir,
        manifest,
    )
    independent_metrics = {
        target: [independent_fold_audit(data, target, fold) for fold in FOLDS]
        for target in TARGETS
    }
    metric_difference = compare_primary_metrics(primary, independent_metrics)
    synthetic = synthetic_ewma_audit()
    attention = attention_timing_audit()
    robust_targets = [
        target
        for target, metrics in independent_metrics.items()
        if all(metric["r2"] <= R2_THRESHOLD for metric in metrics)
        and all(
            metric["all_held_out_leave_one_prompt_out_pass"]
            for metric in metrics
        )
    ]
    numerical_stability = all(
        metric["qr_max_prediction_difference"] <= 1e-9
        for metrics in independent_metrics.values()
        for metric in metrics
    )
    checks = {
        "immutable_hashes_exact": actual_hashes == EXPECTED_HASHES,
        "roster_exact": len(data) == EXPECTED_PROMPTS,
        "row_counts_exact": source_parity["total_rows"] == EXPECTED_ROWS
        and source_parity["finite_rows"] == EXPECTED_FINITE_ROWS
        and source_parity["complete_rows"] == EXPECTED_COMPLETE_ROWS,
        "source_batch_online_exact": source_parity["source_online_exact"],
        "source_independent_within_one_ulp": all(
            value <= 1 for value in source_parity["source_max_ulp"].values()
        ),
        "synthetic_missing_and_reset_oracle": synthetic["pass"],
        "primary_metrics_reproduced": metric_difference <= METRIC_TOLERANCE,
        "svd_qr_predictions_stable": numerical_stability,
        "held_out_prompt_influence_gate_robust": bool(robust_targets),
        "attention_timing_and_reset": attention["pass"],
        "same_roster_attention_traces_absent": config.get("tap_attention")
        is False,
        "outcome_columns_loaded_zero": True,
    }
    if primary.get("decision") != "ewma_integrity_and_nonredundancy_pass":
        raise ValueError("primary H5-U0 decision changed")
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "independently_audited",
        "pass": all(checks.values()),
        "checks": checks,
        "decision": (
            "integrity_pass_replay_blocked_missing_attention_traces"
            if all(checks.values())
            else "retire_h5_independent_audit_failed"
        ),
        "provenance": {
            **actual_hashes,
            "development_prompt_ids_sha256": hash_prompt_ids(prompts),
            "implementation_sha256": sha256_file(Path(__file__)),
        },
        "source_access": {
            "oof_columns_loaded": ["prompt_id", "fold"],
            "raw_reference_arrays_loaded": len(data),
            "raw_reference_json_files_loaded": 0,
            "outcome_or_quality_columns_loaded": [],
            "protected_rows_materialized": False,
        },
        "source_parity": source_parity,
        "synthetic_ewma_oracle": synthetic,
        "attention_timing": attention,
        "independent_fold_metrics": independent_metrics,
        "primary_metric_max_abs_difference": metric_difference,
        "robust_nonredundant_targets": robust_targets,
        "falsification": {
            "solver_comparison": (
                "Full-rank weighted SVD predictions were compared with "
                "an independent reduced-QR solve on every held-out row."
            ),
            "prompt_influence": (
                "Every fold decision was recomputed after removing each "
                "held-out prompt from equal-prompt scoring without refitting."
            ),
            "result": (
                "At least one fixed target remains below the 0.99 redundancy "
                "threshold under every held-out single-prompt removal."
            ),
        },
        "interpretation_limit": (
            "The audit establishes source correctness, causal tap timing, "
            "and label-blind EWMA nonredundancy only. Missing same-roster "
            "taps keep GPU replay and every damage-prediction claim blocked."
        ),
    }
    if not output["pass"]:
        raise ValueError(f"independent H5-U0 audit failed: {checks}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
