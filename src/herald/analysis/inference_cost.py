"""Predictor inference-cost microbenchmark.

The "lightweight" claim needs numbers: how long does it take
to score one token's features? A batch of 512? On what
hardware?

This module times `booster.predict` on a DMatrix built from
real features (not synthetic noise) and reports p50/p95/p99
latencies for two access patterns:

  - single_token: one row per call (worst-case online use).
  - batch_512:    one 512-row call (amortized / offline).

A GPU step-time comparison is left as a TODO because it
requires a live LLM run and belongs in the sweep pipeline.
"""

import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from herald.analysis.common import (
    DEFAULT_HORIZONS,
    DEFAULT_NT_ONSET_FRAC,
    load_booster,
    model_path_for,
    write_json,
)
from herald.features import build_dataset

DEFAULT_N_ITERS = 500
WARMUP_ITERS = 20
BATCH_SIZE = 512


def _percentiles(times_us: list[float]) -> dict[str, float]:
    arr = np.asarray(times_us)
    return {
        "mean": round(float(arr.mean()), 3),
        "p50": round(float(np.percentile(arr, 50)), 3),
        "p95": round(float(np.percentile(arr, 95)), 3),
        "p99": round(float(np.percentile(arr, 99)), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "n": int(arr.size),
    }


def _time_single(
    booster: xgb.Booster,
    X: list[list[float]],
    feat_names: list[str],
    n_iters: int,
) -> dict[str, float]:
    dmats = [
        xgb.DMatrix([X[i]], feature_names=feat_names)
        for i in range(min(n_iters + WARMUP_ITERS, len(X)))
    ]
    times: list[float] = []
    for i, dm in enumerate(dmats):
        t0 = time.perf_counter()
        booster.predict(dm)
        dt = (time.perf_counter() - t0) * 1e6
        if i >= WARMUP_ITERS:
            times.append(dt)
    return _percentiles(times)


def _time_batch(
    booster: xgb.Booster,
    X: list[list[float]],
    feat_names: list[str],
    n_iters: int,
    batch_size: int,
) -> dict[str, float]:
    n = len(X)
    times: list[float] = []
    for i in range(n_iters + WARMUP_ITERS):
        start = (i * batch_size) % max(1, n - batch_size)
        batch = X[start : start + batch_size]
        if len(batch) < batch_size:
            batch = X[:batch_size]
        dm = xgb.DMatrix(batch, feature_names=feat_names)
        t0 = time.perf_counter()
        booster.predict(dm)
        dt = (time.perf_counter() - t0) * 1e6
        if i >= WARMUP_ITERS:
            times.append(dt)
    return _percentiles(times)


def run_inference_cost(
    model_dir: Path,
    results_dir: Path,
    output_path: Path,
    horizons: list[int] | None = None,
    n_iters: int = DEFAULT_N_ITERS,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
) -> dict[str, object]:
    horizons = horizons or DEFAULT_HORIZONS
    by_horizon: dict[str, dict[str, object]] = {}
    out: dict[str, object] = {
        "horizons": horizons,
        "batch_size": BATCH_SIZE,
        "n_iters": n_iters,
        "warmup_iters": WARMUP_ITERS,
        "units": "microseconds",
        "gpu_step_time_todo": (
            "Measure LLM decode-step latency on orion for "
            "an apples-to-apples overhead ratio. Not done here "
            "because it requires a live model."
        ),
        "by_horizon": by_horizon,
    }

    ds = build_dataset(
        results_dir, horizon=horizons[0], nt_onset_frac=nt_onset_frac
    )
    if not ds.X:
        raise RuntimeError("empty dataset for inference cost benchmark")

    for h in horizons:
        mp = model_path_for(model_dir, h)
        if not mp.exists():
            continue
        booster = load_booster(mp)

        by_horizon[f"H{h}"] = {
            "single_token": _time_single(
                booster, ds.X, ds.feat_names, n_iters
            ),
            f"batch_{BATCH_SIZE}": _time_batch(
                booster, ds.X, ds.feat_names, n_iters, BATCH_SIZE
            ),
            "n_trees": int(booster.num_boosted_rounds()),
        }

    write_json(output_path, out)
    return out
