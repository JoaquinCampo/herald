"""CPU tests for quality-risk TCN helpers (tiny synthetic runs)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from predict_quality_risk_tcn_oof import parse_selected_epochs
from train_quality_risk_tcn import (  # noqa: E402
    RiskTCN,
    batch_tensors,
    make_batches,
    run_slices,
)


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for run in ("r1", "r2", "r3"):
        for token in range(6):
            rows.append(
                {
                    "run_id": run,
                    "token_pos": token,
                    "damage": float(token % 2),
                    "value": float(rng.normal()),
                }
            )
    return pd.DataFrame(rows)


def test_run_slices_and_batches() -> None:
    frame = _frame()
    slices = run_slices(frame)
    assert slices == [(0, 6), (6, 12), (12, 18)]
    batches = make_batches(slices, [0, 1, 2], False, 0)
    assert sum(len(bucket) for bucket in batches) == 3


def test_batch_tensors_shapes_and_mask() -> None:
    frame = _frame()
    slices = run_slices(frame)
    continuous = frame[["value"]].to_numpy(dtype=np.float32)
    press = np.zeros(len(frame), dtype=np.int16)
    ratio = np.zeros(len(frame), dtype=np.int16)
    action = np.zeros(len(frame), dtype=np.int16)
    targets = frame["damage"].to_numpy(dtype=np.float32)
    weights = np.ones(len(frame), dtype=np.float32)
    mean = np.zeros(1, dtype=np.float32)
    scale = np.ones(1, dtype=np.float32)
    inputs, truth, weight, mask, locations = batch_tensors(
        [0, 2],
        slices,
        continuous,
        press,
        ratio,
        action,
        targets,
        weights,
        mean,
        scale,
    )
    assert inputs.shape == (2, 6, 6 + 7 + 42 + 1)
    assert truth.shape == (2, 6)
    assert mask.all()
    assert locations == [(0, 6), (12, 18)]
    model = RiskTCN(6 + 7 + 42 + 1)
    logits = model(inputs)
    assert logits.shape == (2, 6)
    assert np.isfinite(logits.detach().numpy()).all()


def test_parse_selected_epochs(tmp_path) -> None:
    log = tmp_path / "train.log"
    log.write_text(
        "outer=0 epoch=3 validation=0.5\n"
        "completed TCN outer fold=0 epochs=9\n"
        "completed TCN outer fold=1 epochs=13\n"
        "completed TCN outer fold=2 epochs=9\n"
        "completed TCN outer fold=3 epochs=7\n"
        "completed TCN outer fold=4 epochs=7\n"
    )
    assert parse_selected_epochs(log) == {
        "0": 9,
        "1": 13,
        "2": 9,
        "3": 7,
        "4": 7,
    }
