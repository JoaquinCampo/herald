"""Streaming HERALD v1: O(1) per-token regression at inference.

Wraps a trained `HistGradientBoostingRegressor` (or any sklearn
estimator that accepts the 24-feature `CHEAP_ALL_FEATURE_ORDER`
vector with NaN-aware predict) and an `OnlineFeatureState`.

For each token emitted by the generator, `step(signals)` returns the
expected `future_sum_js_H` for the next H tokens. State update is
O(1) amortized in feature count. NaN inputs are passed through to
the model so HGB's native missing-value handling kicks in.
"""

import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from herald.config import TokenSignals
from herald.phase4_online_features import (
    CHEAP_ALL_FEATURE_ORDER,
    OnlineFeatureState,
)


@dataclass(slots=True)
class StreamingHeraldRegressor:
    """Per-token regression of future compression damage.

    Parameters
    ----------
    model :
        sklearn-compatible regressor trained on
        `np.log1p(future_sum_js_H)` with `CHEAP_ALL_FEATURE_ORDER`
        input columns and NaN-aware predict.
    press, compression_ratio, max_new_tokens :
        Run-level metadata identical to the offline pipeline.
    inverse_transform :
        Callable applied to model output (default `np.expm1` to undo
        the log1p target transform used in training).
    """

    model: Any
    press: str
    compression_ratio: float
    max_new_tokens: int
    inverse_transform: Any = None
    _state: OnlineFeatureState = field(init=False)
    _last_pred_raw: float = field(default=math.nan, init=False)
    _last_pred_model: float = field(default=math.nan, init=False)

    def __post_init__(self) -> None:
        self._state = OnlineFeatureState(
            press=self.press,
            compression_ratio=self.compression_ratio,
            max_new_tokens=self.max_new_tokens,
        )
        if self.inverse_transform is None:
            self.inverse_transform = np.expm1

    def step(self, sig: TokenSignals) -> float:
        """Advance one token and return the predicted future damage."""
        self._state.update(sig)
        vec = self._feature_array_nan_aware()
        y_log1p = float(self.model.predict(vec.reshape(1, -1))[0])
        self._last_pred_model = y_log1p
        y_raw = float(self.inverse_transform(y_log1p))
        self._last_pred_raw = y_raw
        return y_raw

    def _feature_array_nan_aware(self) -> np.ndarray:
        """Build a NaN-preserving 24-feature vector.

        HGB handles NaN natively, so we do not fill with zeros (which
        would corrupt the prediction at t=0 where delta_h / kl_div /
        top10_jaccard are legitimately undefined).
        """
        out = np.empty(len(CHEAP_ALL_FEATURE_ORDER), dtype=np.float32)
        for i, col in enumerate(CHEAP_ALL_FEATURE_ORDER):
            v = self._state._last_features.get(col, math.nan)
            out[i] = float(v)
        return out

    @property
    def token_pos(self) -> int:
        return self._state.token_pos

    @property
    def last_pred_raw(self) -> float:
        return self._last_pred_raw

    @property
    def last_pred_log1p(self) -> float:
        return self._last_pred_model


def load_regressor(path: Path) -> Any:
    """Load a pickled sklearn estimator."""
    with open(path, "rb") as fp:
        return pickle.load(fp)
