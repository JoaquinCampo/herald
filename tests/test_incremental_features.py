"""Bit-exact parity tests for incremental derived features."""

import numpy as np

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
