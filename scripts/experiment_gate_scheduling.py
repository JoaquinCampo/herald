"""Run scorer-gated attempt scheduling on recorded HERALD data."""

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
    bootstrap_group_ci,
    gated_replay,
    replay,
    replay_matrices,
)
from herald.grace_window import hyb_feature_names  # noqa: E402
from herald.switch_baselines import leave_one_compressor_splits  # noqa: E402
from herald.switch_risk import featurize  # noqa: E402

COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
PARQUET = Path("results/predictor/switch_dataset_attn.parquet")
NPZ = Path("results/predictor/hybrid_streams_ifeval.npz")
TARGETS = Path("results/predictor/alarm_bundle/fidelity_targets.json")
LIVE_EPISODES = Path("results/live_controller_v2/episodes.jsonl")
LIVE_BASELINE = Path("results/live_controller_v2/baseline.jsonl")
OUT_DIR = Path("results/predictor/gate_scheduling")

SPLIT_SEED = 0
N_FOLDS = 5
SEEDS = (0, 1, 2)
K = 2
EPSILON = 0.03
GATE_COST_LIMIT = 0.03
SAVINGS_FACTOR = 0.98
GATE_QUANTILES = 201
BASE = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": -1,
}


def ratio_key(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    targets = json.loads(TARGETS.read_text())
    episodes = load_jsonl(LIVE_EPISODES)
    baselines = {
        str(row["prompt_id"]): float(row["wall_s"])
        for row in load_jsonl(LIVE_BASELINE)
    }
    wall_models = fit_wall_models(episodes)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    sanity: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "config": {
            "compressors": list(COMPRESSORS),
            "split_seed": SPLIT_SEED,
            "n_folds": N_FOLDS,
            "seeds": list(SEEDS),
            "k": K,
            "epsilon": EPSILON,
            "gate_cost_limit": GATE_COST_LIMIT,
            "savings_factor": SAVINGS_FACTOR,
            "parquet": str(PARQUET),
            "npz": str(NPZ),
        },
        "compressors": {},
        "wall_models": wall_models,
    }

    splits = leave_one_compressor_splits(
        data["rows"],
        compressors=COMPRESSORS,
        seed=SPLIT_SEED,
        test_group_fraction=0.25,
    )
    row_index = {id(row): i for i, row in enumerate(data["rows"])}
    for split in splits:
        compressor = split.heldout_compressor
        print(f"START {compressor}", flush=True)
        theta = float(targets["compressors"][compressor]["theta"])
        split_data = make_split_data(data, split, row_index)
        result = run_compressor(
            compressor=compressor,
            theta=theta,
            data=data,
            split_data=split_data,
            target=targets["compressors"][compressor],
            episodes=episodes,
            baselines=baselines,
            wall_models=wall_models,
        )
        summary["compressors"][compressor] = result["summary"]
        sanity[compressor] = result["sanity"]
        train_rows.extend(result["train_rows"])
        test_rows.extend(result["test_rows"])
        frontier_rows.extend(result["frontier_rows"])
        wall_rows.extend(result["wall_rows"])
        print(f"DONE {compressor} {time.time() - t0:.1f}s", flush=True)

    write_outputs(
        summary=summary,
        sanity=sanity,
        train_rows=train_rows,
        test_rows=test_rows,
        frontier_rows=frontier_rows,
        wall_rows=wall_rows,
    )


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
    blocks = stream_data["blocks"][mask]
    lengths = stream_data["lengths"][mask]
    trailing = stream_data["trailing"][mask]
    hyb_mat, hyb_cols = hyb_summaries(blocks, lengths, trailing, K)
    return {
        "rows": rows,
        "feat_cols": feat_cols,
        "hyb_mat": hyb_mat,
        "hyb_cols": hyb_cols,
    }


def hyb_summaries(
    blocks: np.ndarray,
    lengths: np.ndarray,
    trailing: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """Match the alarm bundle hybrid summaries exactly."""
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


def make_split_data(
    data: dict[str, Any],
    split: Any,
    row_index: dict[int, int],
) -> dict[str, Any]:
    rows = data["rows"]
    compressor = split.heldout_compressor
    donor_train_sel = [row_index[id(row)] for row in split.train]
    test_pids = {str(row["prompt_id"]) for row in split.test}
    heldout_sel = [
        i
        for i, row in enumerate(rows)
        if str(row["compressor"]) == compressor
    ]
    heldout_train_sel = [
        i for i in heldout_sel if str(rows[i]["prompt_id"]) not in test_pids
    ]
    heldout_test_sel = [
        i for i in heldout_sel if str(rows[i]["prompt_id"]) in test_pids
    ]
    return {
        "donor_train_sel": donor_train_sel,
        "heldout_train_sel": heldout_train_sel,
        "heldout_test_sel": heldout_test_sel,
    }


def run_compressor(
    *,
    compressor: str,
    theta: float,
    data: dict[str, Any],
    split_data: dict[str, Any],
    target: dict[str, Any],
    episodes: list[dict[str, Any]],
    baselines: dict[str, float],
    wall_models: dict[str, Any],
) -> dict[str, Any]:
    rows = data["rows"]
    heldout_train_sel = split_data["heldout_train_sel"]
    heldout_test_sel = split_data["heldout_test_sel"]
    train = [rows[i] for i in heldout_train_sel]
    test = [rows[i] for i in heldout_test_sel]

    alarm_train = fit_alarm_final_scores(
        data=data,
        train_sel=heldout_train_sel,
        pred_sel=heldout_train_sel,
        rows=rows,
    )
    alarm_test = fit_alarm_final_scores(
        data=data,
        train_sel=heldout_train_sel,
        pred_sel=heldout_test_sel,
        rows=rows,
    )
    sanity = check_fidelity(
        compressor=compressor,
        theta=theta,
        test_rows=test,
        test_scores=alarm_test,
        target=target,
    )

    train_mats = replay_matrices(train, alarm_train)
    test_mats = replay_matrices(test, alarm_test)
    ungated_train = replay(train_mats, theta=theta, k=K)
    ungated_test = replay(test_mats, theta=theta, k=K)

    variants = {
        "gate-A alarm-imitation": (np.asarray(alarm_train) <= theta).astype(
            np.float32
        ),
        "gate-B dq-direct": np.asarray(
            [float(row["dq"]) <= 0.0 for row in train], dtype=np.float32
        ),
    }

    out_summary: dict[str, Any] = {
        "theta": theta,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "ungated_test": outcome_summary(
            compressor=compressor,
            variant="ungated",
            outcome=ungated_test,
            mats=test_mats,
            baseline=None,
        ),
    }

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    frontier_rows: list[dict[str, Any]] = []
    wall_rows: list[dict[str, Any]] = []
    gate_train_x = gate_matrix(train, data["feat_cols"])
    gate_test_x = gate_matrix(test, data["feat_cols"])
    train_prompt_ids = [str(row["prompt_id"]) for row in train]
    for variant, labels in variants.items():
        gate_oof, gate_test = fit_oof_and_test(
            x_train=gate_train_x,
            y_train=labels,
            x_test=gate_test_x,
            prompt_ids=train_prompt_ids,
        )
        train_gate_mats = replay_matrices(train, gate_oof)
        # Gate selection must use alarm scores aligned to the same matrix rows
        # as replay matrices. Replaying with row-ordered scores corrupts commit
        # semantics even when g_tau is -inf.
        g_tau, train_scan = select_gate_threshold(
            train_mats=train_mats,
            alarm_scores=train_mats.pred,
            gate_scores=train_gate_mats.pred,
            theta=theta,
            ungated=ungated_train,
        )
        test_gate_mats = replay_matrices(test, gate_test)
        gated_test = gated_replay(
            test_mats,
            scores=test_mats.pred,
            gate_scores=test_gate_mats.pred,
            g_tau=g_tau,
            theta=theta,
            k=K,
        )
        wall = project_wall(
            compressor=compressor,
            variant=variant,
            test_mats=test_mats,
            ungated=ungated_test,
            gate_scores=test_gate_mats.pred,
            g_tau=g_tau,
            theta=theta,
            episodes=episodes,
            baselines=baselines,
            wall_models=wall_models,
        )
        wall_rows.append(wall)
        train_rows.append(
            {
                "compressor": compressor,
                "variant": variant,
                "g_tau": g_tau,
                **train_scan,
            }
        )
        test_row = outcome_summary(
            compressor=compressor,
            variant=variant,
            outcome=gated_test,
            mats=test_mats,
            baseline=ungated_test,
        )
        moved_commit_groups = int(
            np.sum(ungated_test.commit_index != gated_test.commit_index)
        )
        test_row["g_tau"] = g_tau
        test_row["moved_commit_groups_frozen_vs_ungated"] = moved_commit_groups
        test_row["projected_revert_wall_overhead_primary"] = wall[
            "mean_share_primary"
        ]
        test_row["projected_revert_wall_overhead_secondary"] = wall[
            "mean_share_secondary"
        ]
        test_rows.append(test_row)
        frontier = build_frontier(
            compressor=compressor,
            variant=variant,
            test_mats=test_mats,
            ungated=ungated_test,
            gate_scores=test_gate_mats.pred,
            theta=theta,
            episodes=episodes,
            baselines=baselines,
            wall_models=wall_models,
            frozen_g_tau=g_tau,
        )
        frontier_rows.extend(frontier)
        out_summary[variant] = {
            "train": train_rows[-1],
            "test": test_row,
            "wall": wall,
        }
    return {
        "summary": out_summary,
        "sanity": sanity,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "frontier_rows": frontier_rows,
        "wall_rows": wall_rows,
    }


def fit_alarm_final_scores(
    *,
    data: dict[str, Any],
    train_sel: list[int],
    pred_sel: list[int],
    rows: list[dict[str, Any]],
) -> list[float]:
    train = [rows[i] for i in train_sel]
    pred_rows = [rows[i] for i in pred_sel]
    x_train = alarm_matrix(data, train_sel, train)
    x_pred = alarm_matrix(data, pred_sel, pred_rows)
    y_train = np.asarray(
        [float(row["dq"]) > 0.0 for row in train], dtype=np.float32
    )
    return fit_test(
        x_train=x_train,
        y_train=y_train,
        x_test=x_pred,
    )


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


def fit_oof_and_test(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    prompt_ids: list[str],
) -> tuple[list[float], list[float]]:
    oof_acc = np.zeros(len(y_train), dtype=np.float64)
    test_acc = np.zeros(x_test.shape[0], dtype=np.float64)
    folds = prompt_folds(prompt_ids)
    pid_arr = np.asarray(prompt_ids)
    for seed in SEEDS:
        params = {**BASE, "seed": seed}
        oof = np.full(len(y_train), np.nan, dtype=np.float64)
        for fold_pids in folds:
            test_mask = np.isin(pid_arr, list(fold_pids))
            train_mask = ~test_mask
            booster = train_xgb(
                params=params,
                x=x_train[train_mask],
                y=y_train[train_mask],
            )
            oof[test_mask] = booster.predict(xgb.DMatrix(x_train[test_mask]))
        final = train_xgb(params=params, x=x_train, y=y_train)
        oof_acc += oof
        test_acc += final.predict(xgb.DMatrix(x_test))
    return (
        (oof_acc / len(SEEDS)).tolist(),
        (test_acc / len(SEEDS)).tolist(),
    )


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


def fit_test(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
) -> list[float]:
    acc = np.zeros(x_test.shape[0], dtype=np.float64)
    for seed in SEEDS:
        params = {**BASE, "seed": seed}
        booster = fit_xgb(params=params, x=x_train, y=y_train)
        acc += booster.predict(xgb.DMatrix(x_test))
    return (acc / len(SEEDS)).tolist()


def train_xgb(
    *,
    params: dict[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> xgb.Booster:
    return fit_xgb(params=params, x=x, y=y)


def prompt_folds(prompt_ids: list[str]) -> list[set[str]]:
    pids = sorted(set(prompt_ids))
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pids)
    return [set(pids[fold::N_FOLDS]) for fold in range(N_FOLDS)]


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
    grid = [float("-inf"), *sorted({float(v) for v in quantiles})]
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
        / ungated_attempts,
        "selection_feasible": feasible,
        "n_thresholds": len(grid),
    }


def outcome_summary(
    *,
    compressor: str,
    variant: str,
    outcome: ReplayOutcome,
    mats: Any,
    baseline: ReplayOutcome | None,
) -> dict[str, Any]:
    ci = bootstrap_group_ci(outcome.savings, mats.prompt_ids)
    attempts = int(outcome.n_attempts.sum())
    reverts = int(total_reverts(outcome))
    out = {
        "compressor": compressor,
        "variant": variant,
        "mean_savings": float(outcome.savings.mean()),
        "ci95_savings_low": ci[0],
        "ci95_savings_high": ci[1],
        "mean_cost": float(outcome.cost.mean()),
        "mean_token_overhead": float(outcome.overhead.mean()),
        "total_attempts": attempts,
        "total_reverts": reverts,
        "commit_rate": commit_rate(outcome),
    }
    if baseline is not None:
        base_attempts = float(baseline.n_attempts.sum())
        out["attempts_kept_fraction"] = attempts / base_attempts
    else:
        out["attempts_kept_fraction"] = 1.0
    return out


def _commit_slots(mats: Any, outcome: ReplayOutcome) -> np.ndarray:
    slot_s = np.full_like(outcome.commit_index, np.nan, dtype=np.float64)
    for i, commit_index in enumerate(outcome.commit_index):
        if int(commit_index) >= 0:
            slot_s[i] = float(mats.s[i, int(commit_index)])
    return slot_s


def _n_commit_slot_mismatches(
    lhs: np.ndarray,
    rhs: np.ndarray,
) -> int:
    both = np.isnan(lhs) & np.isnan(rhs)
    diff = lhs != rhs
    diff[both] = False
    return int(diff.sum())


def total_reverts(outcome: ReplayOutcome) -> int:
    committed = (outcome.commit_index >= 0).astype(np.int64)
    return int((outcome.n_attempts - committed).sum())


def commit_rate(outcome: ReplayOutcome) -> float:
    return float((outcome.commit_index >= 0).mean())


def check_fidelity(
    *,
    compressor: str,
    theta: float,
    test_rows: list[dict[str, Any]],
    test_scores: list[float],
    target: dict[str, Any],
) -> dict[str, Any]:
    mats = replay_matrices(test_rows, test_scores)
    out = replay(mats, theta=theta, k=K)
    target_by_key = {
        (str(group["prompt_id"]), float(group["ratio"])): group
        for group in target["groups"]
    }
    max_score_delta = 0.0
    max_commit_mismatch = 0
    for i, key in enumerate(mats.keys):
        group = target_by_key[key]
        valid = int(mats.valid[i])
        target_scores = np.asarray(group["scores"], dtype=np.float64)
        deltas = np.abs(mats.pred[i, :valid] - target_scores)
        max_score_delta = max(max_score_delta, float(deltas.max()))
        commit_index = int(out.commit_index[i])
        commit_s = int(mats.s[i, commit_index]) if commit_index >= 0 else None
        max_commit_mismatch += int(commit_s != group["commit_s"])
    replay_target = target["replay_test"]
    return {
        "compressor": compressor,
        "theta_used": theta,
        "theta_target": float(target["theta"]),
        "theta_delta": abs(theta - float(target["theta"])),
        "score_max_abs_delta": max_score_delta,
        "commit_mismatches": max_commit_mismatch,
        "mean_savings": float(out.savings.mean()),
        "target_mean_savings": float(replay_target["mean_savings"]),
        "mean_savings_delta": abs(
            float(out.savings.mean()) - float(replay_target["mean_savings"])
        ),
        "mean_cost": float(out.cost.mean()),
        "target_mean_cost": float(replay_target["mean_cost"]),
        "mean_cost_delta": abs(
            float(out.cost.mean()) - float(replay_target["mean_cost"])
        ),
    }


def build_frontier(
    *,
    compressor: str,
    variant: str,
    test_mats: Any,
    ungated: ReplayOutcome,
    gate_scores: np.ndarray,
    theta: float,
    episodes: list[dict[str, Any]],
    baselines: dict[str, float],
    wall_models: dict[str, Any],
    frozen_g_tau: float,
) -> list[dict[str, Any]]:
    finite = gate_scores[np.isfinite(gate_scores)]
    grid = [float("-inf"), *np.quantile(finite, np.linspace(0.0, 1.0, 11))]
    grid.append(frozen_g_tau)
    rows: list[dict[str, Any]] = []
    seen: set[float] = set()
    ungated_commit_slots = _commit_slots(mats=test_mats, outcome=ungated)
    for tau in sorted({float(v) for v in grid}):
        if tau in seen:
            continue
        seen.add(tau)
        out = gated_replay(
            test_mats,
            scores=test_mats.pred,
            gate_scores=gate_scores,
            g_tau=tau,
            theta=theta,
            k=K,
        )
        gated_commit_slots = _commit_slots(mats=test_mats, outcome=out)
        wall = project_wall(
            compressor=compressor,
            variant=variant,
            test_mats=test_mats,
            ungated=ungated,
            gate_scores=gate_scores,
            g_tau=tau,
            theta=theta,
            episodes=episodes,
            baselines=baselines,
            wall_models=wall_models,
        )
        rows.append(
            {
                "compressor": compressor,
                "variant": variant,
                "g_tau": tau,
                "is_frozen": tau == frozen_g_tau,
                "commit_s_moved": _n_commit_slot_mismatches(
                    ungated_commit_slots,
                    gated_commit_slots,
                ),
                "attempt_fraction": (
                    float(out.n_attempts.sum())
                    / float(ungated.n_attempts.sum())
                ),
                "mean_savings": float(out.savings.mean()),
                "mean_cost": float(out.cost.mean()),
                "projected_revert_wall_overhead_primary": wall[
                    "mean_share_primary"
                ],
                "projected_revert_wall_delay_share": wall["mean_delay_share"],
                "projected_revert_wall_overhead_secondary": wall[
                    "mean_share_secondary"
                ],
                "commit_rate": commit_rate(out),
            }
        )
    return rows


def project_wall(
    *,
    compressor: str,
    variant: str,
    test_mats: Any,
    ungated: ReplayOutcome,
    gate_scores: np.ndarray,
    g_tau: float,
    theta: float,
    episodes: list[dict[str, Any]],
    baselines: dict[str, float],
    wall_models: dict[str, Any],
) -> dict[str, Any]:
    outcome = gated_replay(
        test_mats,
        scores=test_mats.pred,
        gate_scores=gate_scores,
        g_tau=g_tau,
        theta=theta,
        k=K,
    )
    by_group = group_slots(
        test_mats=test_mats,
        gate_scores=gate_scores,
        outcome=outcome,
        g_tau=g_tau,
    )
    primary_shares: list[float] = []
    secondary_shares: list[float] = []
    mismatch_details: list[dict[str, Any]] = []
    mismatch_counts = {
        "missing_group": 0,
        "missing_slot": 0,
        "missing_live_measurement": 0,
        "missing_live_after_commit": 0,
    }
    live_attempt_mismatches = 0
    n_measured = 0
    n_predicted_delay = 0
    kept_live = 0
    total_live = 0

    key_to_row = {key: i for i, key in enumerate(test_mats.keys)}
    for episode in episodes:
        if str(episode["compressor"]) != compressor:
            continue
        prompt = str(episode["prompt_id"])
        ratio = float(episode["ratio"])
        key = (prompt, ratio)
        row_index = key_to_row.get(key)
        slots = by_group.get(key)
        if slots is None or row_index is None:
            mismatch_counts["missing_group"] += 1
            mismatch_details.append(
                {
                    "prompt_id": prompt,
                    "ratio": ratio_key(ratio),
                    "reason": "missing_group",
                }
            )
            continue
        base_wall = baselines.get(prompt)
        if base_wall is None or base_wall <= 0.0:
            continue

        live_commit = episode.get("commit_s")
        live_commit = None if live_commit is None else int(live_commit)
        ci = int(outcome.commit_index[row_index])
        outcome_commit_slot = int(slots[ci]["s"]) if ci >= 0 else None
        ungated_ci = int(ungated.commit_index[row_index])
        delay_by_gate = False
        delay_start = 0
        delay_end = 0
        if g_tau != float("-inf") and live_commit is not None:
            if ungated_ci >= 0 and ci >= 0 and ci > ungated_ci:
                delay_by_gate = True
                delay_start = ungated_ci + 1
                delay_end = ci
            elif ungated_ci >= 0 and ci < 0:
                delay_by_gate = True
                delay_start = ungated_ci + 1
                delay_end = int(test_mats.valid[row_index])

        live_by_s = {
            int(a["s"]): float(a["wall_s"]) for a in episode["attempts"]
        }
        slot_by_s = {slot["s"]: slot for slot in slots}
        row_level_primary = 0.0
        row_level_secondary = 0.0

        for attempt in episode["attempts"]:
            total_live += 1
            attempt_s = int(attempt["s"])
            slot = slot_by_s.get(attempt_s)
            if slot is None:
                live_attempt_mismatches += 1
                mismatch_counts["missing_slot"] += 1
                mismatch_details.append(
                    {
                        "prompt_id": prompt,
                        "ratio": ratio_key(ratio),
                        "s": attempt_s,
                        "reason": "missing_slot",
                    }
                )
                continue
            kept_live += 1
            if not slot["kept"]:
                continue
            if not attempt["committed"]:
                row_level_primary += float(attempt["wall_s"])
                n_measured += 1

        for slot in slots:
            if not slot["kept"]:
                continue
            s = int(slot["s"])
            if s in live_by_s:
                continue
            if live_commit is None:
                mismatch_counts["missing_live_measurement"] += 1
                mismatch_details.append(
                    {
                        "prompt_id": prompt,
                        "ratio": ratio_key(ratio),
                        "s": s,
                        "reason": "missing_live_measurement",
                    }
                )
                continue
            if s <= live_commit:
                mismatch_counts["missing_live_measurement"] += 1
                mismatch_details.append(
                    {
                        "prompt_id": prompt,
                        "ratio": ratio_key(ratio),
                        "s": s,
                        "reason": "missing_live_measurement_before_live_commit",
                    }
                )
                continue
            if not delay_by_gate:
                mismatch_counts["missing_live_measurement"] += 1
                mismatch_details.append(
                    {
                        "prompt_id": prompt,
                        "ratio": ratio_key(ratio),
                        "s": s,
                        "reason": "missing_live_measurement_after_live_commit",
                    }
                )
                continue
            if ci >= 0 and slot["index"] >= ci:
                continue
            if not (delay_start <= slot["index"] < delay_end):
                continue
            mismatch_counts["missing_live_after_commit"] += 1
            row_level_secondary += predict_wall(
                wall_models[compressor],
                s=float(s),
                ref_len=float(episode.get("ref_len_live") or slot["ref_len"]),
                ratio=ratio,
            )
            n_predicted_delay += 1
            mismatch_details.append(
                {
                    "prompt_id": prompt,
                    "ratio": ratio_key(ratio),
                    "s": s,
                    "reason": "gated_delay_without_live_measurement",
                }
            )

        if row_level_secondary < 0.0:
            row_level_secondary = 0.0
        primary_shares.append(row_level_primary / base_wall)
        secondary_shares.append(row_level_secondary / base_wall)

    mean_primary = float(np.mean(primary_shares)) if primary_shares else 0.0
    mean_delay = float(np.mean(secondary_shares)) if secondary_shares else 0.0
    return {
        "compressor": compressor,
        "variant": variant,
        "g_tau": g_tau,
        "mean_share_primary": mean_primary,
        "mean_share_secondary": mean_primary + mean_delay,
        "mean_delay_share": mean_delay,
        "n_episodes": len(primary_shares),
        "live_attempt_mismatches": live_attempt_mismatches,
        "projected_attempts": n_predicted_delay,
        "kept_live_attempt_fraction": kept_live / total_live
        if total_live
        else 0.0,
        "n_measured_slots": n_measured,
        "n_predicted_slots": n_predicted_delay,
        "mismatch_counts": mismatch_counts,
        "mismatch_examples": mismatch_details[:200],
    }


def group_slots(
    test_mats: Any,
    gate_scores: np.ndarray,
    outcome: ReplayOutcome,
    g_tau: float,
) -> dict[tuple[str, float], list[dict[str, Any]]]:
    by_group: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for i, key in enumerate(test_mats.keys):
        valid = int(test_mats.valid[i])
        commit = int(outcome.commit_index[i])
        slots: list[dict[str, Any]] = []
        for j in range(valid):
            slot_score = float(gate_scores[i, j])
            slots.append(
                {
                    "index": j,
                    "s": int(test_mats.s[i, j]),
                    "ref_len": float(test_mats.ref_len[i]),
                    "gate_score": slot_score,
                    "kept": bool(slot_score >= g_tau),
                    "is_commit": j == commit,
                    "after_commit": commit >= 0 and j > commit,
                }
            )
        by_group[(key[0], float(key[1]))] = slots
    return by_group


def fit_wall_models(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for compressor in COMPRESSORS:
        x_rows: list[list[float]] = []
        y_rows: list[float] = []
        for episode in episodes:
            if str(episode["compressor"]) != compressor:
                continue
            ref_len = float(episode.get("ref_len_live") or 0.0)
            ratio = float(episode["ratio"])
            for attempt in episode["attempts"]:
                x_rows.append([1.0, float(attempt["s"]), ref_len, ratio])
                y_rows.append(float(attempt["wall_s"]))
        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        pred = x @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
        models[compressor] = {
            "features": ["intercept", "s", "ref_len_live", "ratio"],
            "coef": coef.tolist(),
            "r2": r2,
            "n_attempts": int(len(y_rows)),
        }
    return models


def predict_wall(
    model: dict[str, Any],
    *,
    s: float,
    ref_len: float,
    ratio: float,
) -> float:
    coef = np.asarray(model["coef"], dtype=np.float64)
    value = float(np.asarray([1.0, s, ref_len, ratio]) @ coef)
    return max(0.0, value)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_outputs(
    *,
    summary: dict[str, Any],
    sanity: dict[str, Any],
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    wall_rows: list[dict[str, Any]],
) -> None:
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT_DIR / "sanity_checks.json").write_text(json.dumps(sanity, indent=2))
    pd.DataFrame(train_rows).to_csv(
        OUT_DIR / "train_thresholds.csv", index=False
    )
    pd.DataFrame(test_rows).to_csv(OUT_DIR / "test_results.csv", index=False)
    pd.DataFrame(frontier_rows).to_csv(OUT_DIR / "frontier.csv", index=False)
    pd.DataFrame(wall_rows).to_csv(
        OUT_DIR / "wall_projection.csv", index=False
    )
    wall_rows_out = []
    for compressor, model in summary["wall_models"].items():
        row = {
            "compressor": compressor,
            "r2": model["r2"],
            "n_attempts": model["n_attempts"],
        }
        for name, coef in zip(model["features"], model["coef"], strict=True):
            row[f"coef_{name}"] = coef
        wall_rows_out.append(row)
    pd.DataFrame(wall_rows_out).to_csv(
        OUT_DIR / "wall_models.csv", index=False
    )


if __name__ == "__main__":
    main()
