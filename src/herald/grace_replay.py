"""Grace-window policy replay from single-switch records.

Rollback-to-reference semantics make sequential multi-attempt policies
replayable from recorded per-s hybrid streams: every reverted attempt
puts the run back on the reference trajectory, so the next attempt's
stream is exactly the recorded one. The policy attempts compression at
every recorded grid point in ascending s and commits at the first
alarm score <= theta; a revert costs the k observed tokens
(overhead = k * reverts / ref_len).

This is the funded AIMD feasibility spec, ported verbatim from the
feasibility scripts. The live controller must implement the same
semantics; `bootstrap_group_ci` provides the cluster-bootstrap CI the
live results are checked against (replay fidelity, mission goal 1).
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReplayMatrices:
    """Padded per-group attempt matrices, ascending s within a group.

    A group is one (prompt_id, ratio) episode. `pred` pads with +inf
    (a padded slot never commits); `valid` counts real grid points.
    """

    pred: np.ndarray  # (n_groups, width) alarm scores
    dq: np.ndarray  # (n_groups, width) recorded quality deltas
    sav: np.ndarray  # (n_groups, width) savings if committed there
    valid: np.ndarray  # (n_groups,) real attempts per group
    ref_len: np.ndarray  # (n_groups,)
    s: np.ndarray  # (n_groups, width) switch positions (-1 padding)
    keys: list[tuple[str, float]]  # (prompt_id, ratio) per group
    prompt_ids: list[str]  # cluster ids, aligned to rows


@dataclass
class ReplayOutcome:
    """Per-group results of one replay at a fixed theta."""

    savings: np.ndarray
    cost: np.ndarray
    overhead: np.ndarray
    n_attempts: np.ndarray
    # Index into the group's attempt list, -1 when never committed.
    commit_index: np.ndarray


def replay_matrices(
    rows: Sequence[dict[str, Any]], scores: Sequence[float]
) -> ReplayMatrices:
    """Group rows into padded per-episode attempt matrices."""
    order: dict[
        tuple[str, float], list[tuple[int, float, float, float, float]]
    ] = {}
    for row, sc in zip(rows, scores, strict=True):
        key = (str(row["prompt_id"]), float(row["ratio"]))
        order.setdefault(key, []).append(
            (
                int(row["s"]),
                float(sc),
                float(row["dq"]),
                max(0.0, 1.0 - int(row["s"]) / float(row["ref_len"])),
                float(row["ref_len"]),
            )
        )
    width = max(len(v) for v in order.values())
    n = len(order)
    pred = np.full((n, width), np.inf)
    dq = np.zeros((n, width))
    sav = np.zeros((n, width))
    s_mat = np.full((n, width), -1, dtype=np.int64)
    valid = np.zeros(n, dtype=np.int64)
    ref_len = np.zeros(n)
    keys: list[tuple[str, float]] = []
    for i, (key, members) in enumerate(order.items()):
        members.sort(key=lambda item: item[0])
        keys.append(key)
        valid[i] = len(members)
        ref_len[i] = members[0][4]
        for j, (s, p, d, savings, _) in enumerate(members):
            pred[i, j] = p
            dq[i, j] = d
            sav[i, j] = savings
            s_mat[i, j] = s
    return ReplayMatrices(
        pred=pred,
        dq=dq,
        sav=sav,
        valid=valid,
        ref_len=ref_len,
        s=s_mat,
        keys=keys,
        prompt_ids=[key[0] for key in keys],
    )


def replay(mats: ReplayMatrices, *, theta: float, k: int) -> ReplayOutcome:
    """Sequential alarm replay: commit at the first score <= theta."""
    hit = mats.pred <= theta
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    idx = np.arange(mats.pred.shape[0])
    savings = np.where(any_hit, mats.sav[idx, first], 0.0)
    cost = np.where(any_hit, mats.dq[idx, first], 0.0)
    reverts = np.where(any_hit, first, mats.valid)
    overhead = k * reverts / mats.ref_len
    n_attempts = np.where(any_hit, first + 1, mats.valid)
    commit_index = np.where(any_hit, first, -1)
    return ReplayOutcome(
        savings=savings,
        cost=cost,
        overhead=overhead,
        n_attempts=n_attempts,
        commit_index=commit_index,
    )


def gated_replay(
    mats: ReplayMatrices,
    scores: Sequence[float] | np.ndarray,
    gate_scores: Sequence[float] | np.ndarray,
    *,
    g_tau: float,
    theta: float,
    k: int,
) -> ReplayOutcome:
    """Sequential replay with an attempt gate before the alarm check."""
    alarm = _slot_matrix(scores, mats=mats, name="scores")
    gate = _slot_matrix(gate_scores, mats=mats, name="gate_scores")
    steps = np.arange(alarm.shape[1])[None, :]
    valid = steps < mats.valid[:, None]
    gate = np.nan_to_num(gate, nan=np.inf)
    attempted = valid & (gate >= g_tau)
    hit = attempted & (alarm <= theta)
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    idx = np.arange(alarm.shape[0])
    savings = np.where(any_hit, mats.sav[idx, first], 0.0)
    cost = np.where(any_hit, mats.dq[idx, first], 0.0)
    before_commit = steps < first[:, None]
    reverts = np.where(
        any_hit,
        (attempted & before_commit).sum(axis=1),
        attempted.sum(axis=1),
    )
    overhead = k * reverts / mats.ref_len
    n_attempts = np.where(any_hit, reverts + 1, reverts)
    commit_index = np.where(any_hit, first, -1)
    return ReplayOutcome(
        savings=savings,
        cost=cost,
        overhead=overhead,
        n_attempts=n_attempts,
        commit_index=commit_index,
    )


def oracle_replay(
    mats: ReplayMatrices, *, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Perfect-alarm ceiling: commit at the first undamaged point."""
    hit = mats.dq <= 0.0
    # Padded slots have dq == 0; mask them out with valid counts.
    steps = np.arange(mats.dq.shape[1])[None, :]
    hit = hit & (steps < mats.valid[:, None])
    any_hit = hit.any(axis=1)
    first = hit.argmax(axis=1)
    idx = np.arange(mats.dq.shape[0])
    savings = np.where(any_hit, mats.sav[idx, first], 0.0)
    reverts = np.where(any_hit, first, mats.valid)
    overhead = k * reverts / mats.ref_len
    return savings, overhead


def calibrate_theta(
    mats: ReplayMatrices,
    scores: Sequence[float],
    *,
    epsilon: float,
    k: int,
    grid_size: int = 201,
) -> float:
    """Point calibration: the theta maximizing mean replay savings
    subject to mean replay cost <= epsilon, over a quantile grid of the
    calibration scores (plus a below-all sentinel that never commits).
    """
    values = np.asarray(scores, dtype=np.float64)
    grid = np.quantile(values, np.linspace(0.0, 1.0, grid_size))
    thetas = sorted({float(values.min()) - 1.0, *map(float, grid)})
    best_theta, best_sav = thetas[0], -1.0
    for theta in thetas:
        out = replay(mats, theta=theta, k=k)
        if float(out.cost.mean()) <= epsilon and (
            float(out.savings.mean()) > best_sav
        ):
            best_sav = float(out.savings.mean())
            best_theta = theta
    return best_theta


def bootstrap_group_ci(
    values: np.ndarray,
    prompt_ids: Sequence[str],
    *,
    n_bootstraps: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster-bootstrap CI of the mean of per-group values.

    Clusters are prompts: all groups (ratios) of a resampled prompt
    enter together, matching the locked evaluator's clustering unit.
    """
    prompts = sorted(set(prompt_ids))
    by_prompt: dict[str, list[int]] = {p: [] for p in prompts}
    for i, p in enumerate(prompt_ids):
        by_prompt[p].append(i)
    rng = np.random.default_rng(_stable_seed(seed, "grace_ci"))
    n_prompts = len(prompts)
    means = np.empty(n_bootstraps)
    for b in range(n_bootstraps):
        chosen = rng.integers(0, n_prompts, size=n_prompts)
        idx = np.concatenate([by_prompt[prompts[c]] for c in chosen]).astype(
            np.int64
        )
        means[b] = float(values[idx].mean())
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _stable_seed(seed: int, *parts: str) -> int:
    """Process-independent seed (Python's hash() is salted)."""
    digest = hashlib.sha256("\0".join([str(seed), *parts]).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _slot_matrix(
    values: Sequence[float] | np.ndarray,
    *,
    mats: ReplayMatrices,
    name: str,
) -> np.ndarray:
    """Return a padded slot matrix aligned to ``mats``."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape == mats.pred.shape:
        return arr
    if arr.ndim != 1 or arr.size != int(mats.valid.sum()):
        raise ValueError(
            f"{name} must have shape {mats.pred.shape} or "
            f"{int(mats.valid.sum())} flat valid-slot values"
        )
    out = np.full(mats.pred.shape, np.inf, dtype=np.float64)
    offset = 0
    for i, count in enumerate(mats.valid.tolist()):
        width = int(count)
        out[i, :width] = arr[offset : offset + width]
        offset += width
    return out
