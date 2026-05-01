"""Phase 0 sampling-rate study (design Q4)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr


def _future_max_js(values: np.ndarray, h: int) -> np.ndarray:
    out = np.empty_like(values)
    n = len(values)
    for i in range(n):
        out[i] = values[i : min(i + h, n)].max(initial=0.0)
    return out


def _subset_and_interp(values: np.ndarray, rate: int) -> np.ndarray:
    if rate == 1:
        return values.copy()
    idx = np.arange(0, len(values), rate)
    sampled = values[idx]
    interp = np.interp(np.arange(len(values)), idx, sampled)
    return np.asarray(interp)


def run_study(
    final: Path,
    out: Path,
    horizons: tuple[int, ...] = (5, 10, 25, 50),
    rates: tuple[int, ...] = (1, 4, 8),
) -> None:
    parts = list((final / "replay").glob("press=*/ratio=*/*.parquet"))
    if not parts:
        out.mkdir(parents=True, exist_ok=True)
        (out / "sampling_rate_report.json").write_text(
            json.dumps({"runs": 0})
        )
        return
    df = pl.concat([pl.read_parquet(p) for p in parts]).sort(
        ["run_id", "token_pos"]
    )

    spearman: dict[int, dict[int, list[float]]] = {
        r: {h: [] for h in horizons} for r in rates if r != 1
    }
    dense_traj: list[float] = []
    rate_traj: dict[int, list[float]] = {r: [] for r in rates if r != 1}

    for _, sub in df.group_by("run_id"):
        v = sub["js_full"].to_numpy()
        if len(v) < max(horizons):
            continue
        dense = v
        dense_traj.append(float(dense.sum()))
        for r in rates:
            if r == 1:
                continue
            sub_v = _subset_and_interp(v, r)
            for h in horizons:
                a = _future_max_js(dense, h)
                b = _future_max_js(sub_v, h)
                rho = spearmanr(a, b).statistic
                if not np.isnan(rho):
                    spearman[r][h].append(float(rho))
            rate_traj[r].append(float(sub_v.sum()))

    report = {
        "runs": len(dense_traj),
        "spearman_future_max_js": {
            r: {
                h: float(np.median(spearman[r][h]))
                if spearman[r][h]
                else None
                for h in horizons
            }
            for r in spearman
        },
        "trajectory_rank_corr": {
            r: float(spearmanr(dense_traj, rate_traj[r]).statistic)
            if rate_traj[r]
            else None
            for r in rate_traj
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sampling_rate_report.json").write_text(
        json.dumps(report, indent=2)
    )
