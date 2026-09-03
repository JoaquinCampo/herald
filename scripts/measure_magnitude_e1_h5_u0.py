"""Measure repaired KL-EWMA integrity and label-blind nonredundancy."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from herald.features import IncrementalDerived, derive_features
from herald.magnitude_v2 import hash_prompt_ids, sha256_file

SCHEMA_VERSION = "herald.magnitude_e1_h5_u0_repair_report.v1"
EXPECTED_PROTOCOL_SHA256 = (
    "10ed71fabceb9743f3dec3d9d67efc3035274fccc3394fca279c1ab212ec3f9f"
)
EXPECTED_OOF_SHA256 = (
    "cf0f0282fa605d0151683777870bc33f5848e22801ec175649e46866359e182e"
)
EXPECTED_CONFIG_SHA256 = (
    "8c228094df9f36cd582752070d578fbfe60f49cd53553b3b507076d17c874408"
)
EXPECTED_MANIFEST_SHA256 = (
    "5f3d65d72ff506cfdd8105dd4a26cb065c2234c4ede3263bacf0547b1dc5cf88"
)
EXPECTED_FEATURES_SOURCE_SHA256 = (
    "75b6f42ba622323cdfa22b816d61f7b790fb383f058087933c01b6a955124528"
)
EXPECTED_PROMPTS = 154
EXPECTED_ROWS = 59_182
EXPECTED_FINITE_KL_ROWS = 59_028
EXPECTED_RAW_WIDTH = 20
EXPECTED_RAW_DTYPE = np.dtype(np.float16)
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
REDUNDANCY_R2_THRESHOLD = 0.99


@dataclass(frozen=True)
class PromptData:
    prompt_id: str
    fold: int
    predictors: np.ndarray
    targets: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-lock", type=Path, required=True)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--sweep-config", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--features-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_roster(path: Path) -> tuple[tuple[str, ...], dict[str, int]]:
    table = pq.read_table(path, columns=["prompt_id", "fold"])  # type: ignore[no-untyped-call]
    if table.column_names != ["prompt_id", "fold"]:
        raise ValueError("unexpected OOF columns were materialized")
    frame = table.to_pandas().drop_duplicates()
    fold_counts = frame.groupby("prompt_id")["fold"].nunique()
    if int((fold_counts != 1).sum()):
        raise ValueError("prompt fold is not constant")
    roster = frame.drop_duplicates("prompt_id")
    if len(roster) != EXPECTED_PROMPTS:
        raise ValueError("development prompt count changed")
    prompt_folds = {
        str(row.prompt_id): int(row.fold)
        for row in roster.itertuples(index=False)
    }
    prompts = tuple(sorted(prompt_folds))
    if set(prompt_folds.values()) != set(FOLDS):
        raise ValueError("fixed fold roster changed")
    return prompts, prompt_folds


def declared_arrays(manifest: dict[str, Any]) -> dict[str, str]:
    entries = manifest.get("artifact_files")
    if not isinstance(entries, list):
        raise ValueError("reference artifact list is missing")
    arrays: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid reference artifact entry")
        relative = str(entry.get("path", ""))
        digest = str(entry.get("sha256", ""))
        if relative.endswith(".npy"):
            if relative in arrays:
                raise ValueError(f"duplicate array declaration: {relative}")
            arrays[relative] = digest
    return arrays


def independent_ewma(values: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    acc: float | None = None
    for index, raw_value in enumerate(values.astype(np.float64)):
        value = float(raw_value)
        if np.isnan(value):
            continue
        acc = value if acc is None else alpha * value + (1.0 - alpha) * acc
        out[index] = acc
    return out.astype(np.float32)


def max_ulp_nonnegative(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape != right.shape:
        raise ValueError("ULP inputs have different shapes")
    if left.size == 0:
        return 0
    if np.any(left < 0.0) or np.any(right < 0.0):
        raise ValueError("development KL EWMAs must be nonnegative")
    left_bits = left.astype(np.float32, copy=False).view(np.uint32)
    right_bits = right.astype(np.float32, copy=False).view(np.uint32)
    distance = np.abs(
        left_bits.astype(np.int64) - right_bits.astype(np.int64)
    )
    return int(distance.max())


def source_kl_features(
    kl: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    raw = np.asarray(kl, dtype=np.float32).reshape(-1, 1)
    batch, names = derive_features(
        raw,
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    engine = IncrementalDerived(
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    online_rows = [engine.update(row)[0] for row in raw]
    online = np.stack(online_rows)
    if engine.position != len(raw) - 1:
        raise ValueError("online EWMA row alignment failed")
    return batch, names, online


def load_prompt_data(
    prompts: tuple[str, ...],
    prompt_folds: dict[str, int],
    reference_dir: Path,
    manifest: dict[str, Any],
) -> tuple[list[PromptData], dict[str, Any]]:
    arrays = declared_arrays(manifest)
    prompt_data: list[PromptData] = []
    total_rows = 0
    finite_kl_rows = 0
    complete_rows = 0
    max_batch_oracle_ulp = {target: 0 for target in TARGETS}
    max_online_oracle_ulp = {target: 0 for target in TARGETS}
    batch_online_exact = {target: True for target in TARGETS}
    finite_target_rows = {target: 0 for target in TARGETS}
    for prompt_id in prompts:
        relative = f"{prompt_id}.npy"
        path = reference_dir / relative
        expected_digest = arrays.get(relative)
        if expected_digest is None or not path.is_file():
            raise ValueError(f"missing development array: {relative}")
        if sha256_file(path) != expected_digest:
            raise ValueError(f"development array hash changed: {relative}")
        persisted = np.load(path, allow_pickle=False)
        if (
            persisted.ndim != 2
            or persisted.shape[1] != EXPECTED_RAW_WIDTH
            or persisted.dtype != EXPECTED_RAW_DTYPE
        ):
            raise ValueError(f"development array schema changed: {relative}")
        raw = persisted.astype(np.float32)
        kl = raw[:, 13]
        batch, names, online = source_kl_features(kl)
        if (
            names
            != IncrementalDerived(
                names=("kl_prev",), bases=("kl_prev",)
            ).names()
        ):
            raise ValueError("batch and online KL feature names differ")
        name_index = {name: index for index, name in enumerate(names)}
        if not set(PREDICTORS) <= set(name_index):
            raise ValueError("fixed rolling-KL predictor is missing")
        if set(TARGETS) & set(PREDICTORS):
            raise ValueError("EWMA target leaked into fixed predictors")

        target_values: dict[str, np.ndarray] = {}
        for span, target in zip(SPANS, TARGETS, strict=True):
            target_index = name_index[target]
            batch_target = batch[:, target_index]
            online_target = online[:, target_index]
            oracle = independent_ewma(kl, span)
            finite = np.isfinite(oracle)
            if not np.array_equal(np.isnan(batch_target), np.isnan(oracle)):
                raise ValueError(
                    f"batch missing mask mismatch: {prompt_id} {target}"
                )
            if not np.array_equal(np.isnan(online_target), np.isnan(oracle)):
                raise ValueError(
                    f"online missing mask mismatch: {prompt_id} {target}"
                )
            max_batch_oracle_ulp[target] = max(
                max_batch_oracle_ulp[target],
                max_ulp_nonnegative(batch_target[finite], oracle[finite]),
            )
            max_online_oracle_ulp[target] = max(
                max_online_oracle_ulp[target],
                max_ulp_nonnegative(online_target[finite], oracle[finite]),
            )
            batch_online_exact[target] = batch_online_exact[
                target
            ] and np.array_equal(batch_target, online_target, equal_nan=True)
            if not np.array_equal(finite, np.isfinite(kl)):
                raise ValueError(
                    f"EWMA coverage mismatch: {prompt_id} {target}"
                )
            finite_target_rows[target] += int(finite.sum())
            target_values[target] = batch_target

        predictor_values = batch[:, [name_index[name] for name in PREDICTORS]]
        target_finite = np.isfinite(target_values[TARGETS[0]])
        complete = target_finite & np.isfinite(predictor_values).all(axis=1)
        if not int(complete.sum()):
            raise ValueError(f"no complete KL rows for {prompt_id}")
        prompt_data.append(
            PromptData(
                prompt_id=prompt_id,
                fold=prompt_folds[prompt_id],
                predictors=predictor_values[complete].astype(np.float64),
                targets={
                    target: values[complete].astype(np.float64)
                    for target, values in target_values.items()
                },
            )
        )
        total_rows += int(len(kl))
        finite_kl_rows += int(np.isfinite(kl).sum())
        complete_rows += int(complete.sum())

    parity = {
        "total_raw_rows": total_rows,
        "finite_kl_rows": finite_kl_rows,
        "finite_target_rows": finite_target_rows,
        "complete_comparison_rows": complete_rows,
        "max_batch_oracle_ulp": max_batch_oracle_ulp,
        "max_online_oracle_ulp": max_online_oracle_ulp,
        "batch_online_exact": batch_online_exact,
    }
    return prompt_data, parity


def weighted_training_arrays(
    prompts: list[PromptData], target: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictors: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for prompt in prompts:
        y = prompt.targets[target]
        predictors.append(prompt.predictors)
        targets.append(y)
        weights.append(np.full(len(y), 1.0 / len(y), dtype=np.float64))
    return (
        np.concatenate(predictors, axis=0),
        np.concatenate(targets),
        np.concatenate(weights),
    )


def fit_and_score_fold(
    prompt_data: list[PromptData], target: str, fold: int
) -> dict[str, Any]:
    train = [prompt for prompt in prompt_data if prompt.fold != fold]
    held_out = [prompt for prompt in prompt_data if prompt.fold == fold]
    train_x, train_y, weights = weighted_training_arrays(train, target)
    weight_sum = float(weights.sum())
    mean_x = (train_x * weights[:, None]).sum(axis=0) / weight_sum
    centered = train_x - mean_x
    variance_x = (centered**2 * weights[:, None]).sum(axis=0) / weight_sum
    std_x = np.sqrt(variance_x)
    active = std_x > 0.0
    if not np.any(active):
        raise ValueError("all fixed rolling-KL predictors are constant")
    standardized = centered[:, active] / std_x[active]
    design = np.column_stack(
        [np.ones(len(standardized), dtype=np.float64), standardized]
    )
    sqrt_weight = np.sqrt(weights)
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * sqrt_weight[:, None],
        train_y * sqrt_weight,
        rcond=None,
    )
    baseline = float((train_y * weights).sum() / weight_sum)
    model_errors: list[float] = []
    baseline_errors: list[float] = []
    held_out_rows: list[int] = []
    for prompt in held_out:
        x = (prompt.predictors[:, active] - mean_x[active]) / std_x[active]
        held_design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
        prediction = held_design @ coefficients
        y = prompt.targets[target]
        model_errors.append(float(np.mean((y - prediction) ** 2)))
        baseline_errors.append(float(np.mean((y - baseline) ** 2)))
        held_out_rows.append(len(y))
    mean_model_error = float(np.mean(model_errors))
    mean_baseline_error = float(np.mean(baseline_errors))
    if not mean_baseline_error > 0.0:
        raise ValueError("held-out EWMA baseline variance is zero")
    residual_fraction = mean_model_error / mean_baseline_error
    r2 = 1.0 - residual_fraction
    return {
        "fold": fold,
        "training_prompts": len(train),
        "held_out_prompts": len(held_out),
        "training_rows": int(len(train_y)),
        "held_out_rows": int(sum(held_out_rows)),
        "active_predictors": [
            name
            for name, keep in zip(PREDICTORS, active, strict=True)
            if keep
        ],
        "design_rank": int(rank),
        "mean_prompt_model_mse": mean_model_error,
        "mean_prompt_baseline_mse": mean_baseline_error,
        "residual_fraction": residual_fraction,
        "r2": r2,
        "passes_r2_le_0_99": r2 <= REDUNDANCY_R2_THRESHOLD,
    }


def nonconstancy(
    prompt_data: list[PromptData], target: str
) -> dict[str, Any]:
    global_values = np.concatenate(
        [prompt.targets[target] for prompt in prompt_data]
    )
    folds = {
        str(fold): int(
            np.unique(
                np.concatenate(
                    [
                        prompt.targets[target]
                        for prompt in prompt_data
                        if prompt.fold == fold
                    ]
                )
            ).size
        )
        for fold in FOLDS
    }
    return {
        "global_unique_finite_values": int(np.unique(global_values).size),
        "fold_unique_finite_values": folds,
        "passes": int(np.unique(global_values).size) > 1
        and all(count > 1 for count in folds.values()),
    }


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H5-U0 report: {args.output}"
        )
    actual_hashes = {
        "protocol_lock_sha256": sha256_file(args.protocol_lock),
        "source_oof_sha256": sha256_file(args.source_oof),
        "sweep_config_sha256": sha256_file(args.sweep_config),
        "reference_manifest_sha256": sha256_file(args.reference_manifest),
        "features_source_sha256": sha256_file(args.features_source),
    }
    expected_hashes = {
        "protocol_lock_sha256": EXPECTED_PROTOCOL_SHA256,
        "source_oof_sha256": EXPECTED_OOF_SHA256,
        "sweep_config_sha256": EXPECTED_CONFIG_SHA256,
        "reference_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "features_source_sha256": EXPECTED_FEATURES_SOURCE_SHA256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError("H5-U0 measurement input hash changed")
    protocol = load_json(args.protocol_lock)
    if protocol.get("status") != "locked_before_execution":
        raise ValueError("H5-U0 protocol is not locked")
    config = load_json(args.sweep_config)
    if config.get("tap_attention") is not False:
        raise ValueError("same-roster AttentionTap absence changed")
    manifest = load_json(args.reference_manifest)
    prompts, prompt_folds = load_roster(args.source_oof)
    prompt_data, parity = load_prompt_data(
        prompts,
        prompt_folds,
        args.reference_dir,
        manifest,
    )
    if parity["total_raw_rows"] != EXPECTED_ROWS:
        raise ValueError("development raw row count changed")
    if parity["finite_kl_rows"] != EXPECTED_FINITE_KL_ROWS:
        raise ValueError("development finite KL count changed")

    nonconstant = {
        target: nonconstancy(prompt_data, target) for target in TARGETS
    }
    fold_metrics = {
        target: [
            fit_and_score_fold(prompt_data, target, fold) for fold in FOLDS
        ]
        for target in TARGETS
    }
    qualifying_targets = [
        target
        for target, metrics in fold_metrics.items()
        if all(bool(metric["passes_r2_le_0_99"]) for metric in metrics)
    ]
    checks = {
        "source_batch_oracle_within_one_ulp": all(
            value <= 1 for value in parity["max_batch_oracle_ulp"].values()
        ),
        "source_online_oracle_within_one_ulp": all(
            value <= 1 for value in parity["max_online_oracle_ulp"].values()
        ),
        "source_batch_online_exact": all(
            parity["batch_online_exact"].values()
        ),
        "target_coverage_exact": all(
            count == EXPECTED_FINITE_KL_ROWS
            for count in parity["finite_target_rows"].values()
        ),
        "both_targets_nonconstant": all(
            value["passes"] for value in nonconstant.values()
        ),
        "fixed_nonredundancy_gate": bool(qualifying_targets),
        "same_roster_attention_traces_absent": config.get("tap_attention")
        is False,
        "outcome_columns_loaded_zero": True,
    }
    ewma_gate = all(
        value
        for name, value in checks.items()
        if name != "same_roster_attention_traces_absent"
    )
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "measured_label_blind",
        "pass": ewma_gate,
        "checks": checks,
        "decision": (
            "ewma_integrity_and_nonredundancy_pass"
            if ewma_gate
            else "retire_h5_ewma_gate_failed"
        ),
        "replay_status": "blocked_missing_same_roster_attention_traces",
        "provenance": {
            **actual_hashes,
            "development_prompt_ids_sha256": hash_prompt_ids(prompts),
            "implementation_sha256": sha256_file(Path(__file__)),
        },
        "source_access": {
            "oof_columns_loaded": ["prompt_id", "fold"],
            "raw_reference_arrays_loaded": len(prompt_data),
            "raw_reference_json_files_loaded": 0,
            "outcome_or_quality_columns_loaded": [],
            "protected_rows_materialized": False,
        },
        "fixed_targets": list(TARGETS),
        "fixed_predictors": list(PREDICTORS),
        "parity_and_coverage": parity,
        "nonconstancy": nonconstant,
        "fold_metrics": fold_metrics,
        "qualifying_nonredundant_targets": qualifying_targets,
        "threshold_interpretation": (
            "R2 <= 0.99 is a label-blind engineering redundancy gate, "
            "not the frozen 0.01 damage-prediction skill threshold."
        ),
        "attention_interpretation": (
            "The tiny CPU test can prove timing but current artifacts "
            "contain no same-roster AttentionTap columns, so attention "
            "variability, "
            "redundancy, cost, and predictive relevance remain unmeasured."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
