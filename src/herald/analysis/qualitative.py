"""Qualitative example dump for paper figures.

Selects a small set of catastrophic and clean sequences
spanning presses and ratios, predicts per-token hazards at
one horizon, and saves everything needed to draw an
"intro figure": the prompt, the model output, the detected
catastrophe and its onset, and the hazard trajectory.

Selection strategy is deterministic (seeded sampling)
so the figure is reproducible from run to run.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import xgboost as xgb

from herald.analysis.common import (
    DEFAULT_NT_ONSET_FRAC,
    load_booster,
    model_path_for,
    write_json,
)
from herald.config import RunResult
from herald.features import (
    add_rolling_features,
    feature_names,
    flatten_signals,
)
from herald.labeling import earliest_onset

DEFAULT_HORIZON = 10
DEFAULT_N_CATASTROPHIC = 5
DEFAULT_N_CLEAN = 5
SELECTION_SEED = 42


def _iter_results(
    results_dir: Path,
) -> list[tuple[RunResult, Path]]:
    found: list[tuple[RunResult, Path]] = []
    for jp in sorted(results_dir.rglob("*.json")):
        data = json.loads(jp.read_text())
        for raw in data.get("results", []):
            found.append((RunResult.model_validate(raw), jp))
    return found


def _balanced_sample(
    pool: list[tuple[RunResult, Path]],
    n: int,
    rng: random.Random,
) -> list[tuple[RunResult, Path]]:
    """Pick `n` results spread across press families.

    Groups by press, round-robins until we have n, shuffling
    within each press for variety.
    """
    by_press: dict[str, list[tuple[RunResult, Path]]] = defaultdict(list)
    for rr, jp in pool:
        by_press[rr.press].append((rr, jp))
    for press in by_press:
        rng.shuffle(by_press[press])

    selected: list[tuple[RunResult, Path]] = []
    presses = list(by_press.keys())
    rng.shuffle(presses)
    i = 0
    while len(selected) < n and any(by_press.values()):
        press = presses[i % len(presses)]
        bucket = by_press[press]
        if bucket:
            selected.append(bucket.pop())
        i += 1
    return selected


def _predict_hazards(booster: xgb.Booster, rr: RunResult) -> list[float]:
    rows = flatten_signals(rr.signals, max_new_tokens=rr.max_new_tokens)
    rows = add_rolling_features(rows)
    names = feature_names(rows)
    X = [list(r.values()) for r in rows]
    dmat = xgb.DMatrix(X, feature_names=names)
    return [round(float(x), 6) for x in booster.predict(dmat).tolist()]


def _summarize(
    rr: RunResult,
    source: Path,
    hazards: list[float],
    nt_onset_frac: float,
) -> dict[str, object]:
    onset = earliest_onset(
        rr.catastrophe_onsets,
        rr.catastrophes,
        max_new_tokens=rr.max_new_tokens,
        nt_onset_frac=nt_onset_frac,
        n_tokens=len(rr.signals),
    )
    return {
        "prompt_id": rr.prompt_id,
        "source_file": str(source),
        "press": rr.press,
        "compression_ratio": rr.compression_ratio,
        "catastrophes": rr.catastrophes,
        "catastrophe_onsets": rr.catastrophe_onsets,
        "derived_onset": onset,
        "stop_reason": rr.stop_reason,
        "num_tokens_generated": rr.num_tokens_generated,
        "correct": rr.correct,
        "predicted_answer": rr.predicted_answer,
        "ground_truth": rr.ground_truth,
        "prompt_text": rr.prompt_text,
        "generated_text": rr.generated_text,
        "hazards": hazards,
    }


def run_qualitative(
    model_dir: Path,
    results_dir: Path,
    output_path: Path,
    horizon: int = DEFAULT_HORIZON,
    n_catastrophic: int = DEFAULT_N_CATASTROPHIC,
    n_clean: int = DEFAULT_N_CLEAN,
    nt_onset_frac: float = DEFAULT_NT_ONSET_FRAC,
) -> dict[str, object]:
    mp = model_path_for(model_dir, horizon)
    if not mp.exists():
        raise FileNotFoundError(f"No model at {mp}")
    booster = load_booster(mp)

    all_results = _iter_results(results_dir)
    cat_pool: list[tuple[RunResult, Path]] = []
    clean_pool: list[tuple[RunResult, Path]] = []
    for rr, jp in all_results:
        if not rr.signals:
            continue
        onset = earliest_onset(
            rr.catastrophe_onsets,
            rr.catastrophes,
            max_new_tokens=rr.max_new_tokens,
            nt_onset_frac=nt_onset_frac,
            n_tokens=len(rr.signals),
        )
        if onset is not None:
            cat_pool.append((rr, jp))
        else:
            clean_pool.append((rr, jp))

    rng = random.Random(SELECTION_SEED)
    cats = _balanced_sample(cat_pool, n_catastrophic, rng)
    cleans = _balanced_sample(clean_pool, n_clean, rng)

    out = {
        "horizon": horizon,
        "nt_onset_frac": nt_onset_frac,
        "n_candidates_catastrophic": len(cat_pool),
        "n_candidates_clean": len(clean_pool),
        "catastrophic": [
            _summarize(rr, jp, _predict_hazards(booster, rr), nt_onset_frac)
            for rr, jp in cats
        ],
        "clean": [
            _summarize(rr, jp, _predict_hazards(booster, rr), nt_onset_frac)
            for rr, jp in cleans
        ],
    }

    write_json(output_path, out)
    return out
