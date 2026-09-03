"""Bit-exact parity tests for incremental derived features."""

import numpy as np
import pytest

from herald.features import FEATURE_NAMES, IncrementalDerived, derive_features
from herald.grace_window import (
    assemble_alarm_row,
    assemble_alarm_row_from_state,
)

RNG = np.random.default_rng(19)
N_RAW = len(FEATURE_NAMES)


def random_stream(n_steps: int) -> np.ndarray:
    raw = RNG.normal(size=(n_steps, N_RAW)).astype(np.float32)
    raw[:, FEATURE_NAMES.index("max_prob")] = RNG.uniform(
        0.05, 0.95, size=n_steps
    ).astype(np.float32)
    raw[0, FEATURE_NAMES.index("kl_prev")] = np.nan
    if n_steps > 10:
        raw[10, FEATURE_NAMES.index("kl_prev")] = np.nan
    return raw


EWMA_CASES = (
    pytest.param(
        np.full(5, np.nan, dtype=np.float32),
        id="all_nan",
    ),
    pytest.param(
        np.asarray([np.nan, 0.2, 0.4, 0.1], dtype=np.float32),
        id="leading_nan",
    ),
    pytest.param(
        np.asarray([0.1, 0.2, np.nan, 0.4, 0.3], dtype=np.float32),
        id="internal_nan",
    ),
    pytest.param(
        np.asarray([0.25], dtype=np.float32),
        id="one_finite_value",
    ),
    pytest.param(
        np.concatenate(
            [
                np.asarray([np.nan], dtype=np.float32),
                np.linspace(-0.25, 0.75, 257, dtype=np.float32),
            ]
        ),
        id="long_deterministic_stream",
    ),
)


def independent_ewma(values: np.ndarray, span: int) -> np.ndarray:
    """Frozen missing-aware oracle, independent of production helpers."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(values.shape, np.nan, dtype=np.float64)
    acc: float | None = None
    for index, value in enumerate(values.astype(np.float64)):
        if np.isnan(value):
            continue
        acc = (
            float(value)
            if acc is None
            else alpha * float(value) + (1.0 - alpha) * acc
        )
        out[index] = acc
    return out.astype(np.float32)


def source_ewma(
    values: np.ndarray, span: int
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float32).reshape(-1, 1)
    batch, names = derive_features(
        raw,
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    target = names.index(f"kl_prev_ewma_{span}")
    engine = IncrementalDerived(
        names=("kl_prev",),
        bases=("kl_prev",),
    )
    online_rows = [engine.update(row)[0] for row in raw]
    online = np.stack(online_rows)
    return batch[:, target], online[:, target]


def assert_ewma_matches_oracle(values: np.ndarray, span: int) -> None:
    want = independent_ewma(values, span)
    for got in source_ewma(values, span):
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want))
        finite = np.isfinite(want)
        np.testing.assert_array_max_ulp(
            got[finite],
            want[finite],
            maxulp=1,
        )


@pytest.mark.parametrize("values", EWMA_CASES)
@pytest.mark.parametrize("span", [8, 32])
def test_ewma_matches_missing_aware_oracle(
    values: np.ndarray, span: int
) -> None:
    assert_ewma_matches_oracle(values, span)


@pytest.mark.parametrize("span", [8, 32])
def test_ewma_state_resets_between_prompts(span: int) -> None:
    prompts = (
        np.asarray([np.nan, 0.9, 0.1, 0.3], dtype=np.float32),
        np.asarray([np.nan, 0.2, 0.4], dtype=np.float32),
    )
    for prompt in prompts:
        assert_ewma_matches_oracle(prompt, span)


def test_incremental_matches_batch_at_every_step_bit_exact() -> None:
    lengths = [1, 2, 7, 8, 9, 31, 32, 33, 48]
    for n_steps in lengths:
        raw = random_stream(n_steps)
        engine = IncrementalDerived()
        for t in range(n_steps):
            got, got_names = engine.update(raw[t])
            want, want_names = derive_features(raw[: t + 1])
            assert got_names == want_names
            np.testing.assert_array_equal(got, want[t])
            np.testing.assert_array_equal(engine.latest_row(), want[t])


def test_assemble_alarm_row_from_state_matches_batch_bit_exact() -> None:
    raw = random_stream(49)
    block = random_stream(4)
    for s in [0, 1, 7, 8, 9, 31, 32, 33, 48]:
        engine = IncrementalDerived()
        for row in raw[: s + 1]:
            engine.update(row)
        got = assemble_alarm_row_from_state(
            state=engine,
            block=block,
            ratio=0.375,
            k=2,
        )
        want = assemble_alarm_row(
            ref_raw=raw,
            s=s,
            block=block,
            ratio=0.375,
            k=2,
        )
        assert got == want
