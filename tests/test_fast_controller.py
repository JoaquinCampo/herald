"""The fast tau selector must replicate the locked one exactly."""

import random

from herald.controller_metrics import select_tau
from herald.fast_controller import fast_select_tau


def _rows(seed: int, n_groups: int = 40) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []
    for g in range(n_groups):
        ref_len = rng.randrange(64, 512)
        for s in range(0, ref_len, 16):
            rows.append(
                {
                    "model": "m",
                    "task": rng.choice(["a", "b"]),
                    "prompt_id": f"p{g}",
                    "ratio": rng.choice([0.25, 0.5]),
                    "s": s,
                    "ref_len": ref_len,
                    "dq": rng.choice([0.0, 0.0, 0.0, -0.1, 0.3, 1.0]),
                    "predicted_dq": rng.uniform(-0.2, 1.2),
                }
            )
    return rows


def test_fast_select_tau_matches_locked() -> None:
    for seed in range(5):
        rows = _rows(seed)
        locked = select_tau(rows, prediction_key="predicted_dq")
        fast = fast_select_tau(rows, prediction_key="predicted_dq")
        assert fast == locked, f"seed={seed}: {fast} != {locked}"
