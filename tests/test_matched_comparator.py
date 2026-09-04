"""CPU tests for the matched action+task+clock comparator trainer."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_matched_comparator import (  # noqa: E402
    train_outer_model,
)


def _frame() -> pd.DataFrame:
    rows = []
    for task in ("t1", "t2"):
        for prompt in (f"{task}-p1", f"{task}-p2"):
            for token in range(6):
                rows.append(
                    {
                        "run_id": f"{prompt}-r",
                        "prompt_id": prompt,
                        "task": task,
                        "press_category": 0,
                        "ratio_category": 1,
                        "compression_ratio": 0.5,
                        "action_category": 3,
                        "task_category": 0 if task == "t1" else 1,
                        "token_pos": token,
                        "log_token_clock": float(token),
                        "fold": 0 if "p1" in prompt else 1,
                        "damage": 1 if task == "t1" else 0,
                    }
                )
    frame = pd.DataFrame(rows)
    for column in (
        "press_category",
        "ratio_category",
        "action_category",
        "task_category",
    ):
        frame[column] = pd.Categorical(frame[column])
    return frame


def test_train_outer_model_finite_oof(tmp_path: Path) -> None:
    import xgboost as xgb

    frame = _frame()
    weights = np.ones(len(frame), dtype=np.float32)
    features = [
        "press_category",
        "ratio_category",
        "compression_ratio",
        "action_category",
        "task_category",
        "token_pos",
        "log_token_clock",
    ]
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cpu",
        "eval_metric": "logloss",
        "max_bin": 16,
        "seed": 0,
        "nthread": 1,
    }
    predicted, rounds = train_outer_model(
        frame,
        features,
        weights,
        1,
        params,
        tmp_path / "model.json",
        maximum_rounds=10,
        early_stopping=2,
    )
    assert rounds >= 1
    assert np.isfinite(predicted).all()
    assert ((predicted >= 0) & (predicted <= 1)).all()
    _ = xgb
