"""Validate sparse Phase B train sampling against dense gate scheduling.

CPU-only. Test rows stay dense; only train rows are subsampled per
(prompt_id, compressor) pool. Alarm and gate-A follow the locked
within-compressor protocols and evaluate on the dense held-out split.
"""

import hashlib
import argparse
import math
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "src")

from herald.grace_replay import (  # noqa: E402
    ReplayOutcome,
    calibrate_theta,
    gated_replay,
    replay,
    replay_matrices,
)
from herald.grace_window import hyb_feature_names  # noqa: E402
from herald.switch_baselines import leave_one_compressor_splits  # noqa: E402
from herald.switch_risk import featurize  # noqa: E402

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
DESIGNS = ("D1_uniform", "D2_stratified")
N_VALUES = (8, 12, 16)
SAMPLING_SEEDS = (0, 1, 2)
MODES = ("sparse", "dense-cal", "mixed25")
PARQUET = Path("results/predictor/switch_dataset_attn.parquet")
NPZ = Path("results/predictor/hybrid_streams_ifeval.npz")
DENSE_GATE_RESULTS = Path(
    "results/predictor/gate_scheduling/test_results.csv"
)
DENSE_GATE_THRESHOLDS = Path(
    "results/predictor/gate_scheduling/train_thresholds.csv"
)
OUT_DIR = Path("results/predictor/sparse_sampling")

SPLIT_SEED = 0
N_FOLDS = 5
MODEL_SEEDS = (0, 1, 2)
K = 2
EPSILON = 0.03
GATE_COST_LIMIT = 0.03
SAVINGS_FACTOR = 0.98
GATE_QUANTILES = 201
SAVINGS_TOL = 0.02
OVERHEAD_FACTOR_LIMIT = 1.5
BASE = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": -1,
}


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse sampling calibration experiments."
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="sparse",
        help="sparse: train and calibrate on subsampled rows; dense-cal: calibrate on full dense train rows; mixed25: 25 pct prompts full-row calibration.",
    )
    parser.add_argument(
        "--designs",
        default=",".join(DESIGNS),
        help="comma-separated designs",
    )
    parser.add_argument(
        "--n-values",
        default=",".join(str(value) for value in N_VALUES),
        help="comma-separated N values",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in SAMPLING_SEEDS),
        help="comma-separated sampling seeds",
    )
    return parser.parse_args()


def parse_str_list(value: str, valid: tuple[str, ...]) -> tuple[str, ...]:
    out = tuple(v.strip() for v in value.split(",") if v.strip())
    if not out:
        return valid
    invalid = [item for item in out if item not in valid]
    if invalid:
        raise ValueError(f"invalid values: {invalid}")
    return out


def parse_int_list(value: str, valid: tuple[int, ...]) -> tuple[int, ...]:
    out = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not out:
        return valid
    invalid = [item for item in out if item not in valid]
    if invalid:
        raise ValueError(f"invalid values: {invalid}")
    return out


def main() -> None:
    args = parse_cli()
    try:
        designs = parse_str_list(args.designs, DESIGNS)
        n_values = parse_int_list(args.n_values, N_VALUES)
        seeds = parse_int_list(args.seeds, SAMPLING_SEEDS)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    dense = load_dense_reference()
    dense_theta = load_dense_theta()
    split_data = build_split_data(data["rows"])
    dense_pool_rows = pool_row_counts(data["rows"], split_data)
    existing_summary = load_existing_summary(OUT_DIR / "summary.json")

    existing_config = existing_summary.get("config", {}) if existing_summary else {}
    existing_modes = tuple(existing_config.get("modes", ()))
    existing_designs = tuple(existing_config.get("run_designs", ()))
    existing_n_values = tuple(existing_config.get("run_n_values", ()))
    existing_sampling_seeds = tuple(existing_config.get("run_sampling_seeds", ()))

    table_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    for design in designs:
        for n_rows in n_values:
            for sampling_seed in seeds:
                print(
                    f"START design={design} n={n_rows} seed={sampling_seed}",
                    flush=True,
                )
                result = run_config(
                    design=design,
                    n_rows=n_rows,
                    sampling_seed=sampling_seed,
                    mode=args.mode,
                    data=data,
                    split_data=split_data,
                    dense=dense,
                    dense_theta=dense_theta,
                )
                run_rows.extend(result)
                print(
                    f"DONE design={design} n={n_rows} "
                    f"seed={sampling_seed} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    merged_runs = merge_runs(
        existing_runs=existing_summary.get("runs", []) if existing_summary else [],
        incoming_runs=run_rows,
    )
    requested_modes = sorted(set(existing_modes) | {args.mode})
    requested_designs = tuple(
        sorted(set(existing_designs).union(designs))
    )
    requested_n_values = tuple(
        sorted(set(existing_n_values).union(n_values))
    )
    requested_sampling_seeds = tuple(
        sorted(set(existing_sampling_seeds).union(seeds))
    )
    summary = build_summary(
        run_rows=merged_runs,
        dense=dense,
        dense_pool_rows=dense_pool_rows,
        elapsed_s=time.time() - t0,
        modes=tuple(requested_modes),
        designs=requested_designs,
        n_values=requested_n_values,
        sampling_seeds=requested_sampling_seeds,
    )
    table_rows = flatten_summary(summary)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(table_rows).to_csv(OUT_DIR / "tables.csv", index=False)


def load_inputs() -> dict[str, Any]:
    df = pd.read_parquet(PARQUET)
    df = df[df["task"] == "ifeval"].reset_index(drop=True)
    stream_data = np.load(NPZ, allow_pickle=True)
    mask = df["compressor"].isin(COMPRESSORS).to_numpy()
    df3 = df[mask].reset_index(drop=True)
    rows = df3.to_dict("records")
    feat_cols = sorted(
        col
        for col in df3.columns
        if col.startswith("feat__") and not col.startswith("feat__attn_")
    )
    hyb_mat, hyb_cols = hyb_summaries(
        stream_data["blocks"][mask],
        stream_data["lengths"][mask],
        stream_data["trailing"][mask],
        K,
    )
    return {
        "rows": rows,
        "feat_cols": feat_cols,
        "hyb_mat": hyb_mat,
        "hyb_cols": hyb_cols,
    }


def load_dense_reference() -> dict[str, dict[str, float]]:
    test = pd.read_csv(DENSE_GATE_RESULTS)
    train = pd.read_csv(DENSE_GATE_THRESHOLDS)
    gate_test = test[test["variant"] == "gate-A alarm-imitation"]
    gate_train = train[train["variant"] == "gate-A alarm-imitation"]
    out: dict[str, dict[str, float]] = {}
    for _, row in gate_test.iterrows():
        compressor = str(row["compressor"])
        out[compressor] = {
            "dense_savings": float(row["mean_savings"]),
            "dense_cost": float(row["mean_cost"]),
            "dense_overhead": float(row["mean_token_overhead"]),
            "dense_g_tau": float(row["g_tau"]),
        }
    for _, row in gate_train.iterrows():
        compressor = str(row["compressor"])
        out[compressor]["dense_train_g_tau"] = float(row["g_tau"])
    return out


def build_split_data(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    splits = leave_one_compressor_splits(
        rows,
        compressors=COMPRESSORS,
        seed=SPLIT_SEED,
        test_group_fraction=0.25,
    )
    row_index = {id(row): i for i, row in enumerate(rows)}
    out: dict[str, dict[str, Any]] = {}
    for split in splits:
        compressor = split.heldout_compressor
        donor_train_sel = [row_index[id(row)] for row in split.train]
        test_pids = {str(row["prompt_id"]) for row in split.test}
        own_sel = [
            i
            for i, row in enumerate(rows)
            if str(row["compressor"]) == compressor
        ]
        own_train_sel = [
            i for i in own_sel if str(rows[i]["prompt_id"]) not in test_pids
        ]
        own_test_sel = [
            i for i in own_sel if str(rows[i]["prompt_id"]) in test_pids
        ]
        out[compressor] = {
            "donor_train_sel": donor_train_sel,
            "own_train_sel": own_train_sel,
            "own_test_sel": own_test_sel,
            "test_prompt_ids": sorted(test_pids),
        }
    return out


def run_config(
    *,
    design: str,
    n_rows: int,
    sampling_seed: int,
    mode: str,
    data: dict[str, Any],
    split_data: dict[str, dict[str, Any]],
    dense: dict[str, dict[str, float]],
    dense_theta: dict[str, float],
) -> list[dict[str, Any]]:
    rows = data["rows"]
    out: list[dict[str, Any]] = []
    for compressor in COMPRESSORS:
        split = split_data[compressor]
        alarm_train_sel, calibration_sel = (
            build_train_and_calibration_indices(
                rows=rows,
                indices=split["own_train_sel"],
                design=design,
                n_rows=n_rows,
                sampling_seed=sampling_seed,
                mode=mode,
                compressor=compressor,
            )
        )
        if not alarm_train_sel:
            raise ValueError(
                f"empty alarm training rows for compressor={compressor}, "
                f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
            )
        if not calibration_sel:
            raise ValueError(
                f"empty calibration rows for compressor={compressor}, "
                f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
            )

        gate_train_sel = alarm_train_sel[:]
        test_sel = split["own_test_sel"]
        alarm_train_rows = [rows[i] for i in alarm_train_sel]
        gate_train_rows = [rows[i] for i in gate_train_sel]
        calibration_rows = [rows[i] for i in calibration_sel]
        test_rows = [rows[i] for i in test_sel]
        pool_sizes = per_prompt_train_sizes(rows, alarm_train_sel)
        pool_stats = describe_values(pool_sizes)

        (
            alarm_oof,
            alarm_pred_scores,
        ) = fit_alarm_scores(
            data=data,
            train_sel=alarm_train_sel,
            oof_sel=calibration_sel,
            pred_sels=[calibration_sel, gate_train_sel, test_sel],
            rows=rows,
        )
        alarm_calibration_scores = alarm_oof
        alarm_gate_train_scores = alarm_pred_scores[1]
        alarm_test_scores = alarm_pred_scores[2]
        theta_source_rows = calibration_rows
        theta_source_scores = alarm_calibration_scores
        finite_scores = np.asarray(theta_source_scores, dtype=np.float64)
        assert finite_scores.size > 0, (
            f"empty theta score array for compressor={compressor}, "
            f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
        )
        finite_scores = finite_scores[np.isfinite(finite_scores)]
        assert finite_scores.size > 0, (
            f"no finite theta score in calibration for compressor={compressor}, "
            f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
        )
        theta = calibrate_theta(
            replay_matrices(theta_source_rows, theta_source_scores),
            theta_source_scores,
            epsilon=EPSILON,
            k=K,
        )
        score_min = float(finite_scores.min())
        score_max = float(finite_scores.max())
        assert score_min <= theta <= score_max, (
            f"calibrated theta {theta:.6f} out of OOF score range "
            f"[{score_min:.6f}, {score_max:.6f}] for compressor={compressor}, "
            f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
        )
        theta_ref = dense_theta.get(compressor)
        if theta_ref is not None and theta_ref > 0 and theta > 3.0 * theta_ref:
            print(
                f"[warn] theta {theta:.6f} is >3x dense theta "
                f"{theta_ref:.6f} for compressor={compressor}, "
                f"design={design}, n={n_rows}, seed={sampling_seed}, mode={mode}"
            )

        alarm_train_rows_for_cal = (
            calibration_rows if mode != "sparse" else gate_train_rows
        )
        alarm_scores_for_cal = (
            alarm_calibration_scores
            if mode != "sparse"
            else alarm_gate_train_scores
        )
        alarm_train_for_calibration_mats = replay_matrices(
            alarm_train_rows_for_cal,
            alarm_scores_for_cal,
        )
        ungated_train = replay(
            alarm_train_for_calibration_mats,
            theta=theta,
            k=K,
        )

        test_mats = replay_matrices(test_rows, alarm_test_scores)
        ungated_test = replay(test_mats, theta=theta, k=K)
        gate_labels = (np.asarray(alarm_gate_train_scores) <= theta).astype(
            np.float32
        )

        (
            gate_oof,
            gate_pred_scores,
        ) = fit_gate_scores(
            data=data,
            train_rows=gate_train_rows,
            train_sel=gate_train_sel,
            oof_sel=calibration_sel,
            pred_sels=[calibration_sel, gate_train_sel, test_sel],
            labels=gate_labels,
            rows=rows,
        )
        gate_calibration_scores = gate_pred_scores[0]
        gate_gate_train_scores = gate_pred_scores[1]
        gate_test_scores = gate_pred_scores[2]
        gate_train_rows_for_cal = (
            calibration_rows if mode != "sparse" else gate_train_rows
        )
        gate_scores_for_cal = (
            gate_calibration_scores
            if mode != "sparse"
            else gate_gate_train_scores
        )
        gate_train_mats = replay_matrices(
            gate_train_rows_for_cal,
            gate_scores_for_cal,
        )
        g_tau, train_scan = select_gate_threshold(
            train_mats=gate_train_mats,
            alarm_scores=alarm_train_for_calibration_mats.pred,
            gate_scores=gate_train_mats.pred,
            theta=theta,
            ungated=ungated_train,
        )
        test_gate_mats = replay_matrices(test_rows, gate_test_scores)
        gated_test = gated_replay(
            test_mats,
            scores=test_mats.pred,
            gate_scores=test_gate_mats.pred,
            g_tau=g_tau,
            theta=theta,
            k=K,
        )
        attempts = int(gated_test.n_attempts.sum())
        base_attempts = float(ungated_test.n_attempts.sum())
        dense_ref = dense[compressor]
        mean_savings = float(gated_test.savings.mean())
        mean_cost = float(gated_test.cost.mean())
        mean_overhead = float(gated_test.overhead.mean())
        overhead_limit = dense_ref["dense_overhead"] * OVERHEAD_FACTOR_LIMIT
        passed = (
            abs(mean_savings - dense_ref["dense_savings"]) <= SAVINGS_TOL
            and mean_cost <= GATE_COST_LIMIT
            and mean_overhead <= overhead_limit
        )
        out.append(
            {
                "design": design,
                "n": n_rows,
                "sampling_seed": sampling_seed,
                "mode": mode,
                "compressor": compressor,
                "theta": float(theta),
                "g_tau": float(g_tau),
                "mean_savings": mean_savings,
                "mean_cost": mean_cost,
                "mean_token_overhead": mean_overhead,
                "attempts_kept_fraction": attempts / base_attempts
                if base_attempts
                else 0.0,
                "pass": passed,
                "dense_savings": dense_ref["dense_savings"],
                "dense_cost": dense_ref["dense_cost"],
                "dense_token_overhead": dense_ref["dense_overhead"],
                "dense_g_tau": dense_ref["dense_g_tau"],
                "dense_train_g_tau": dense_ref.get(
                    "dense_train_g_tau", float("nan")
                ),
                "savings_delta": mean_savings - dense_ref["dense_savings"],
                "cost_delta": mean_cost - dense_ref["dense_cost"],
                "overhead_ratio_vs_dense": mean_overhead
                / dense_ref["dense_overhead"]
                if dense_ref["dense_overhead"] > 0.0
                else float("inf"),
                "theta_delta_vs_dense": float("nan"),
                "g_tau_delta_vs_dense": float(g_tau)
                - dense_ref["dense_g_tau"],
                "alarm_train_rows": len(alarm_train_sel),
                "gate_train_rows": len(gate_train_sel),
                "calibration_rows": len(calibration_rows),
                "test_rows": len(test_sel),
                "train_rows_per_prompt_min": float(pool_stats["min"]),
                "train_rows_per_prompt_max": float(pool_stats["max"]),
                "train_rows_per_prompt_mean": float(pool_stats["mean"]),
                "train_pool_count": int(pool_stats["count"]),
                **train_scan,
            }
        )
    return out


def fit_alarm_scores(
    *,
    data: dict[str, Any],
    train_sel: list[int],
    oof_sel: list[int],
    pred_sels: list[list[int]],
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[list[float]]]:
    if not train_sel:
        raise ValueError("cannot fit alarm scores with empty training rows")
    train_rows = [rows[i] for i in train_sel]
    oof_rows = [rows[i] for i in oof_sel]
    x_train = alarm_matrix(data, train_sel, train_rows)
    x_oof = alarm_matrix(data, oof_sel, oof_rows)
    y_train = np.asarray(
        [float(row["dq"]) > 0.0 for row in train_rows], dtype=np.float32
    )
    train_prompt_ids = [str(row["prompt_id"]) for row in train_rows]
    oof_prompt_ids = [str(row["prompt_id"]) for row in oof_rows]
    train_pid_arr = np.asarray(train_prompt_ids)
    oof_pid_arr = np.asarray(oof_prompt_ids)
    oof_folds = prompt_folds(oof_prompt_ids)
    pred_mats = [
        alarm_matrix(
            data, pred_sels[index], [rows[i] for i in pred_sels[index]]
        )
        for index in range(len(pred_sels))
    ]
    oof_acc = np.zeros(len(oof_rows), dtype=np.float64)
    pred_acc = [
        np.zeros(x_pred.shape[0], dtype=np.float64) for x_pred in pred_mats
    ]
    for model_seed in MODEL_SEEDS:
        params = {**BASE, "seed": model_seed}
        oof = np.full(len(oof_rows), np.nan, dtype=np.float64)
        for fold_pids in oof_folds:
            test_mask = np.isin(oof_pid_arr, list(fold_pids))
            if not test_mask.any():
                continue
            train_mask = ~np.isin(train_pid_arr, list(fold_pids))
            if not train_mask.any():
                train_mask = np.ones(len(train_rows), dtype=bool)
            booster = fit_xgb(
                params=params,
                x=x_train[train_mask],
                y=y_train[train_mask],
            )
            oof[test_mask] = booster.predict(xgb.DMatrix(x_oof[test_mask]))
        final = fit_xgb(params=params, x=x_train, y=y_train)
        oof_acc += oof
        for acc, x_pred in zip(pred_acc, pred_mats, strict=True):
            acc += final.predict(xgb.DMatrix(x_pred))
    scale = float(len(MODEL_SEEDS))
    return (
        (oof_acc / scale).tolist(),
        [(acc / scale).tolist() for acc in pred_acc],
    )


def fit_gate_scores(
    *,
    data: dict[str, Any],
    train_rows: list[dict[str, Any]],
    train_sel: list[int],
    oof_sel: list[int],
    pred_sels: list[list[int]],
    labels: np.ndarray,
    rows: list[dict[str, Any]],
) -> tuple[list[float], list[list[float]]]:
    if not train_rows or not train_sel:
        raise ValueError("cannot fit gate scores with empty training rows")
    oof_rows = [rows[i] for i in oof_sel]
    x_train = gate_matrix(train_rows, data["feat_cols"])
    x_oof = gate_matrix(oof_rows, data["feat_cols"])
    prompt_ids = [str(row["prompt_id"]) for row in train_rows]
    oof_prompt_ids = [str(row["prompt_id"]) for row in oof_rows]
    folds = prompt_folds(oof_prompt_ids)
    pid_arr = np.asarray(prompt_ids)
    oof_pid_arr = np.asarray(oof_prompt_ids)
    pred_mats = [
        gate_matrix([rows[i] for i in pred_sels[index]], data["feat_cols"])
        for index in range(len(pred_sels))
    ]
    oof_acc = np.zeros(len(oof_rows), dtype=np.float64)
    pred_acc = [
        np.zeros(x_pred.shape[0], dtype=np.float64) for x_pred in pred_mats
    ]
    for model_seed in MODEL_SEEDS:
        params = {**BASE, "seed": model_seed}
        oof = np.full(len(oof_rows), np.nan, dtype=np.float64)
        for fold_pids in folds:
            test_mask = np.isin(oof_pid_arr, list(fold_pids))
            if not test_mask.any():
                continue
            train_mask = ~np.isin(pid_arr, list(fold_pids))
            if not train_mask.any():
                train_mask = np.ones(len(train_rows), dtype=bool)
            booster = fit_xgb(
                params=params,
                x=x_train[train_mask],
                y=labels[train_mask],
            )
            oof[test_mask] = booster.predict(xgb.DMatrix(x_oof[test_mask]))
        final = fit_xgb(params=params, x=x_train, y=labels)
        oof_acc += oof
        for acc, x_pred in zip(pred_acc, pred_mats, strict=True):
            acc += final.predict(xgb.DMatrix(x_pred))
    scale = float(len(MODEL_SEEDS))
    return (
        (oof_acc / scale).tolist(),
        [(acc / scale).tolist() for acc in pred_acc],
    )


def build_train_and_calibration_indices(
    *,
    rows: list[dict[str, Any]],
    indices: list[int],
    design: str,
    n_rows: int,
    sampling_seed: int,
    mode: str,
    compressor: str,
) -> tuple[list[int], list[int]]:
    sparse_train_sel = subsample_indices(
        rows=rows,
        indices=indices,
        design=design,
        n_rows=n_rows,
        sampling_seed=sampling_seed,
        compressor=compressor,
    )

    if mode == "sparse":
        return sparse_train_sel, sparse_train_sel

    if mode == "dense-cal":
        return sparse_train_sel, sorted(indices)

    if mode != "mixed25":
        raise ValueError(f"unknown mode: {mode}")

    prompt_ids = sorted({str(rows[idx]["prompt_id"]) for idx in indices})
    if not prompt_ids:
        return sparse_train_sel, []

    rng = np.random.default_rng(
        stable_seed("mixed25", design, n_rows, sampling_seed, compressor)
    )
    shuffled = list(prompt_ids)
    rng.shuffle(shuffled)
    n_cal_prompts = max(1, int(math.floor(0.25 * len(prompt_ids))))
    if len(prompt_ids) > 1:
        n_cal_prompts = min(n_cal_prompts, len(prompt_ids) - 1)
    cal_prompt_set = set(shuffled[:n_cal_prompts])
    calibration_indices = [
        idx
        for idx in indices
        if str(rows[idx]["prompt_id"]) in cal_prompt_set
    ]
    train_prompt_indices = [
        idx
        for idx in indices
        if str(rows[idx]["prompt_id"]) not in cal_prompt_set
    ]
    train_sparse_sel = (
        subsample_indices(
            rows=rows,
            indices=train_prompt_indices,
            design=design,
            n_rows=n_rows,
            sampling_seed=sampling_seed,
            compressor=compressor,
        )
        if train_prompt_indices
        else []
    )
    mixed_train_sel = sorted(set(train_sparse_sel).union(calibration_indices))

    return mixed_train_sel, sorted(calibration_indices)


def per_prompt_train_sizes(
    rows: list[dict[str, Any]],
    indices: list[int],
) -> list[int]:
    by_prompt: dict[str, int] = {}
    for idx in indices:
        prompt_id = str(rows[idx]["prompt_id"])
        by_prompt[prompt_id] = by_prompt.get(prompt_id, 0) + 1
    return [int(v) for v in by_prompt.values()]


def select_gate_threshold(
    *,
    train_mats: Any,
    alarm_scores: np.ndarray,
    gate_scores: np.ndarray,
    theta: float,
    ungated: ReplayOutcome,
) -> tuple[float, dict[str, Any]]:
    gate_values = np.asarray(gate_scores, dtype=np.float64)
    values = gate_values[np.isfinite(gate_values)]
    if values.size == 0:
        values = np.array([0.0], dtype=np.float64)
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, GATE_QUANTILES))
    grid = [float("-inf"), *sorted({float(value) for value in quantiles})]
    min_savings = float(ungated.savings.mean()) * SAVINGS_FACTOR
    best_tau: float | None = None
    best_out: ReplayOutcome | None = None
    best_key: tuple[float, float, float, float] | None = None
    feasible = True
    for tau in grid:
        out = gated_replay(
            train_mats,
            scores=alarm_scores,
            gate_scores=gate_scores,
            g_tau=tau,
            theta=theta,
            k=K,
        )
        savings = float(out.savings.mean())
        cost = float(out.cost.mean())
        attempts = float(out.n_attempts.sum())
        if savings < min_savings or cost > GATE_COST_LIMIT:
            continue
        key = (attempts, -savings, cost, -tau)
        if best_key is None or key < best_key:
            best_key = key
            best_tau = tau
            best_out = out
    if best_tau is None or best_out is None:
        feasible = False
        best_tau = float("-inf")
        best_out = gated_replay(
            train_mats,
            scores=alarm_scores,
            gate_scores=gate_scores,
            g_tau=best_tau,
            theta=theta,
            k=K,
        )
    ungated_attempts = float(ungated.n_attempts.sum())
    return best_tau, {
        "train_mean_savings": float(best_out.savings.mean()),
        "train_mean_cost": float(best_out.cost.mean()),
        "train_mean_overhead": float(best_out.overhead.mean()),
        "train_total_attempts": int(best_out.n_attempts.sum()),
        "train_total_reverts": int(total_reverts(best_out)),
        "train_commit_rate": commit_rate(best_out),
        "train_ungated_mean_savings": float(ungated.savings.mean()),
        "train_ungated_mean_cost": float(ungated.cost.mean()),
        "train_ungated_total_attempts": int(ungated_attempts),
        "train_attempt_fraction": float(best_out.n_attempts.sum())
        / ungated_attempts
        if ungated_attempts
        else 0.0,
        "selection_feasible": feasible,
        "n_thresholds": len(grid),
    }


def subsample_indices(
    *,
    rows: list[dict[str, Any]],
    indices: list[int],
    design: str,
    n_rows: int,
    sampling_seed: int,
    compressor: str,
) -> list[int]:
    by_pool: dict[tuple[str, str], list[int]] = {}
    for idx in indices:
        row = rows[idx]
        key = (str(row["prompt_id"]), str(row["compressor"]))
        by_pool.setdefault(key, []).append(idx)
    selected: list[int] = []
    for (prompt_id, pool_compressor), pool in sorted(by_pool.items()):
        rng = np.random.default_rng(
            stable_seed(
                design, n_rows, sampling_seed, prompt_id, pool_compressor
            )
        )
        ordered_pool = sorted(pool)
        if design == "D1_uniform":
            keep = sample_uniform(ordered_pool, n_rows, rng)
        elif design == "D2_stratified":
            keep = sample_stratified(rows, ordered_pool, n_rows, rng)
        else:
            raise ValueError(f"unknown design: {design}")
        selected.extend(keep)
    return sorted(selected)


def sample_uniform(
    pool: list[int],
    n_rows: int,
    rng: np.random.Generator,
) -> list[int]:
    if len(pool) <= n_rows:
        return pool[:]
    chosen = rng.choice(np.asarray(pool), size=n_rows, replace=False)
    return sorted(int(value) for value in chosen.tolist())


def sample_stratified(
    rows: list[dict[str, Any]],
    pool: list[int],
    n_rows: int,
    rng: np.random.Generator,
) -> list[int]:
    if len(pool) <= n_rows:
        return pool[:]
    by_s: dict[int, list[int]] = {}
    for idx in pool:
        by_s.setdefault(int(rows[idx]["s"]), []).append(idx)
    s_values = sorted(by_s)
    if len(s_values) < n_rows:
        return sample_uniform(pool, n_rows, rng)
    bins = np.array_split(np.asarray(s_values, dtype=np.int64), n_rows)
    chosen: list[int] = []
    for bin_values in bins:
        s_choice = int(rng.choice(bin_values))
        candidates = sorted(by_s[s_choice])
        chosen.append(int(rng.choice(np.asarray(candidates, dtype=np.int64))))
    return sorted(chosen)


def alarm_matrix(
    data: dict[str, Any],
    sel: list[int],
    base_rows: list[dict[str, Any]],
) -> np.ndarray:
    rows = inject_hybrid(data, sel, base_rows)
    cols = [*data["feat_cols"], *data["hyb_cols"]]
    return featurize(rows, cols)


def inject_hybrid(
    data: dict[str, Any],
    sel: list[int],
    base_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    out = [dict(row) for row in base_rows]
    for row, idx in zip(out, sel, strict=True):
        for j, name in enumerate(data["hyb_cols"]):
            value = float(data["hyb_mat"][idx, j])
            row[name] = None if np.isnan(value) else value
    return out


def gate_matrix(
    rows: list[dict[str, Any]],
    feat_cols: list[str],
) -> np.ndarray:
    cols = [*feat_cols, "ratio"]
    out = np.full((len(rows), len(cols)), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            value = row.get(col)
            out[i, j] = np.nan if value is None else float(value)
    return out


def fit_xgb(
    *,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> xgb.Booster:
    return xgb.train(
        params,
        xgb.DMatrix(x, label=y),
        num_boost_round=300,
    )


def prompt_folds(prompt_ids: list[str]) -> list[set[str]]:
    pids = sorted(set(prompt_ids))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pids)
    return [set(pids[fold::N_FOLDS]) for fold in range(N_FOLDS)]


def hyb_summaries(
    blocks: np.ndarray,
    lengths: np.ndarray,
    trailing: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    n = blocks.shape[0]
    b = blocks[:, :k].astype(np.float32)
    m = np.minimum(lengths, k)
    step = np.arange(k)[None, :, None]
    b = np.where(step < m[:, None, None], b, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(b, axis=1)
        mn = np.nanmin(b, axis=1)
        mx = np.nanmax(b, axis=1)
    step0 = b[:, 0]
    last = b[np.arange(n), np.maximum(m - 1, 0)]
    slope = (last - step0) / np.maximum(m - 1, 1).astype(np.float32)[:, None]
    dtrail = mean - trailing
    parts = [step0, mean, mn, mx, slope, dtrail]
    for part in parts:
        part[m == 0] = np.nan
    return np.concatenate(parts, axis=1), hyb_feature_names(k)


def total_reverts(outcome: ReplayOutcome) -> int:
    committed = (outcome.commit_index >= 0).astype(np.int64)
    return int((outcome.n_attempts - committed).sum())


def commit_rate(outcome: ReplayOutcome) -> float:
    return float((outcome.commit_index >= 0).mean())


def pool_row_counts(
    rows: list[dict[str, Any]],
    split_data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    all_counts: list[int] = []
    own_counts: list[int] = []
    train_counts_by_compressor: dict[str, dict[str, Any]] = {}
    for compressor, split in split_data.items():
        comp_own_counts: list[int] = []
        comp_donor_counts: list[int] = []
        for role in ("donor_train_sel", "own_train_sel"):
            by_pool: dict[tuple[str, str], int] = {}
            for idx in split[role]:
                row = rows[idx]
                key = (str(row["prompt_id"]), str(row["compressor"]))
                by_pool[key] = by_pool.get(key, 0) + 1
            values = list(by_pool.values())
            all_counts.extend(values)
            if role == "own_train_sel":
                comp_own_counts.extend(values)
                own_counts.extend(values)
            else:
                comp_donor_counts.extend(values)
        train_counts_by_compressor[compressor] = {
            "own_train": describe_values(comp_own_counts),
            "donor_train": describe_values(comp_donor_counts),
            "combined": describe_values(comp_own_counts + comp_donor_counts),
        }
    return {
        "all_train_prompt_compressor_pools": describe_values(all_counts),
        "own_train_prompt_compressor_pools": describe_values(own_counts),
        "by_heldout_compressor": train_counts_by_compressor,
    }


def build_summary(
    *,
    run_rows: list[dict[str, Any]],
    dense: dict[str, dict[str, float]],
    dense_pool_rows: dict[str, Any],
    elapsed_s: float,
    modes: tuple[str, ...],
    designs: tuple[str, ...],
    n_values: tuple[int, ...],
    sampling_seeds: tuple[int, ...],
) -> dict[str, Any]:
    runs = pd.DataFrame(run_rows)
    dense_theta = load_dense_theta()
    for compressor, theta in dense_theta.items():
        mask = runs["compressor"] == compressor
        runs.loc[mask, "dense_theta"] = theta
        runs.loc[mask, "theta_delta_vs_dense"] = (
            runs.loc[mask, "theta"] - theta
        )

    configs: dict[str, Any] = {}
    actual_cost: dict[str, Any] = {}
    for (run_mode, design, n_rows), group in runs.groupby(
        ["mode", "design", "n"]
    ):
        config_key = f"{run_mode}_{design}_N{int(n_rows)}"
        compressors: dict[str, Any] = {}
        config_pass = True
        actual_cost.setdefault(config_key, {})
        for compressor, comp_group in group.groupby("compressor"):
            compressor_pass = bool(comp_group["pass"].all())
            config_pass = config_pass and compressor_pass
            compressors[str(compressor)] = summarize_compressor(comp_group)
            dense_train_pool_mean = dense_pool_rows["by_heldout_compressor"][
                str(compressor)
            ]["own_train"]["mean"]
            actual_cost[config_key][str(compressor)] = {
                "vs_mean_dense_pool": float(
                    comp_group["train_rows_per_prompt_mean"].mean()
                )
                / dense_train_pool_mean
                if dense_train_pool_mean
                else float("inf"),
                "dense_mean_rows_per_prompt": float(dense_train_pool_mean),
                "actual_mean_rows_per_prompt": float(
                    comp_group["train_rows_per_prompt_mean"].mean()
                ),
            }
        configs[config_key] = {
            "mode": str(run_mode),
            "design": str(design),
            "n": int(n_rows),
            "pass": config_pass,
            "compressors": compressors,
        }

    smallest = find_smallest_passing(configs)
    return {
        "config": {
            "compressors": list(COMPRESSORS),
            "designs": list(DESIGNS),
            "modes": list(modes),
            "run_designs": list(designs),
            "run_n_values": list(n_values),
            "run_sampling_seeds": list(sampling_seeds),
            "requested_mode": modes[0] if modes else None,
            "split_seed": SPLIT_SEED,
            "n_folds": N_FOLDS,
            "model_seeds": list(MODEL_SEEDS),
            "k": K,
            "epsilon": EPSILON,
            "gate_cost_limit": GATE_COST_LIMIT,
            "savings_factor": SAVINGS_FACTOR,
            "savings_tolerance": SAVINGS_TOL,
            "overhead_factor_limit": OVERHEAD_FACTOR_LIMIT,
            "parquet": str(PARQUET),
            "npz": str(NPZ),
            "alarm_train_scope": "own_compressor",
            "gate_train_scope": "own_compressor",
        },
        "dense_reference": dense,
        "dense_theta": dense_theta,
        "dense_pool_rows": dense_pool_rows,
        "actual_cost_multiplier": actual_cost,
        "configs": configs,
        "smallest_passing": smallest,
        "runs": json.loads(runs.to_json(orient="records")),
        "elapsed_s": elapsed_s,
    }


def load_existing_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def run_row_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["mode"]),
        str(row["design"]),
        int(row["n"]),
        int(row["sampling_seed"]),
    )


def merge_runs(
    *,
    existing_runs: list[dict[str, Any]] | None,
    incoming_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = existing_runs or []
    if not incoming_runs:
        return existing[:]
    replace_keys = {run_row_key(row) for row in incoming_runs}
    merged = [
        row
        for row in existing
        if run_row_key(row) not in replace_keys
    ]
    merged.extend(incoming_runs)
    return merged


def summarize_compressor(group: pd.DataFrame) -> dict[str, Any]:
    fields = [
        "mean_savings",
        "mean_cost",
        "mean_token_overhead",
        "attempts_kept_fraction",
        "theta",
        "g_tau",
        "theta_delta_vs_dense",
        "g_tau_delta_vs_dense",
        "train_rows_per_prompt_min",
        "train_rows_per_prompt_max",
        "train_rows_per_prompt_mean",
        "train_pool_count",
    ]
    out: dict[str, Any] = {
        "pass": bool(group["pass"].all()),
        "seeds": sorted(int(value) for value in group["sampling_seed"]),
    }
    for field in fields:
        values = group[field].astype(float)
        out[field] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return out


def flatten_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in summary["configs"].values():
        for compressor, values in config["compressors"].items():
            row = {
                "mode": config["mode"],
                "design": config["design"],
                "n": config["n"],
                "compressor": compressor,
                "pass": values["pass"],
            }
            for metric, stats in values.items():
                if metric in {"pass", "seeds"}:
                    continue
                for stat_name, stat_value in stats.items():
                    row[f"{metric}_{stat_name}"] = stat_value
            rows.append(row)
    return rows


def find_smallest_passing(configs: dict[str, Any]) -> dict[str, Any] | None:
    passing = [config for config in configs.values() if config["pass"]]
    if not passing:
        return None
    passing.sort(
        key=lambda item: (
            str(item["mode"]),
            int(item["n"]),
            str(item["design"]),
        )
    )
    return {
        "mode": passing[0]["mode"],
        "design": passing[0]["design"],
        "n": int(passing[0]["n"]),
    }


def load_dense_theta() -> dict[str, float]:
    path = Path("results/predictor/alarm_bundle/fidelity_targets.json")
    if not path.exists():
        return {}
    targets = json.loads(path.read_text())
    return {
        compressor: float(value["theta"])
        for compressor, value in targets["compressors"].items()
    }


def describe_values(values: list[int]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def stable_seed(*parts: object) -> int:
    payload = "\0".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


if __name__ == "__main__":
    main()
