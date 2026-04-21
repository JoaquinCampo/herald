"""Shared helpers for analysis modules.

Every analysis needs the same three things:

1. Load a trained booster for a given horizon.
2. Build the matching dataset and produce per-token
   predictions.
3. Slice the resulting (dataset, predictions) pair by
   some attribute (press, compression ratio, sequence).

`iter_horizon_predictions` handles (1) and (2). The slice
helpers handle (3).
"""

import json
from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TypedDict

import xgboost as xgb
from loguru import logger

from herald.features import DatasetBundle, build_dataset


class TokenBucket(TypedDict):
    y: list[int]
    pred: list[float]
    pre: list[bool]
    seq_ids: list[str]

DEFAULT_HORIZONS = [1, 5, 10, 25, 50]
DEFAULT_NT_ONSET_FRAC = 0.75


def model_path_for(model_dir: Path, horizon: int) -> Path:
    return model_dir / f"hazard_H{horizon}.json"


def load_booster(model_path: Path) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    return booster


def predict_dataset(booster: xgb.Booster, ds: DatasetBundle) -> list[float]:
    dmat = xgb.DMatrix(ds.X, feature_names=ds.feat_names)
    preds: list[float] = booster.predict(dmat).tolist()
    return preds


def iter_horizon_predictions(
    model_dir: Path,
    results_dir: Path,
    horizons: list[int],
    nt_onset_frac: float,
) -> Iterator[tuple[int, DatasetBundle, list[float]]]:
    """Yield (horizon, dataset, predictions) for each horizon.

    Skips horizons with no model file or empty dataset,
    logging a warning. Labels depend on horizon, so the
    dataset is rebuilt per horizon; this is expensive but
    unavoidable.
    """
    for h in horizons:
        mp = model_path_for(model_dir, h)
        if not mp.exists():
            logger.warning(f"H={h}: model missing at {mp}")
            continue
        ds = build_dataset(
            results_dir, horizon=h, nt_onset_frac=nt_onset_frac
        )
        if not ds.X:
            logger.warning(f"H={h}: empty dataset, skipping")
            continue
        booster = load_booster(mp)
        preds = predict_dataset(booster, ds)
        yield h, ds, preds


def compression_ratio_from_seq_id(seq_id: str) -> float:
    """Seq IDs are `{prompt_id}__{press}__{ratio}`."""
    return float(seq_id.rsplit("__", 1)[-1])


def group_tokens(
    ds: DatasetBundle,
    preds: list[float],
    key_fn: Callable[[int], str],
) -> dict[str, TokenBucket]:
    """Group per-token (y, pred, pre_onset, seq_id) by key.

    `key_fn(i)` returns the bucket name for token `i`. The
    returned dict has one entry per bucket with aligned lists.
    """

    def _empty() -> TokenBucket:
        return {"y": [], "pred": [], "pre": [], "seq_ids": []}

    buckets: dict[str, TokenBucket] = defaultdict(_empty)
    for i in range(len(preds)):
        k = key_fn(i)
        b = buckets[k]
        b["y"].append(ds.y[i])
        b["pred"].append(preds[i])
        b["pre"].append(ds.pre_onset[i])
        b["seq_ids"].append(ds.seq_ids[i])
    return buckets


def seq_onsets_for_keys(
    ds: DatasetBundle, seq_keys: set[str]
) -> dict[str, int | None]:
    """Filter seq_onsets to just the sequences in `seq_keys`."""
    return {k: v for k, v in ds.seq_onsets.items() if k in seq_keys}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(f"Wrote {path}")
