"""Unit tests for the logit feature superset and downstream dynamics.

Model-free: the collector is exercised with synthetic logits, so these
run fast without loading any model.
"""

import numpy as np
import torch

from herald.features import (
    FEATURE_NAMES,
    TOPK_VALUES,
    FeatureCollector,
    derive_features,
)


def _collect(logits_rows: list[list[float]]) -> np.ndarray:
    """Run the collector over a sequence of single-batch logit vectors."""
    fc = FeatureCollector()
    for row in logits_rows:
        fc(torch.empty(0), torch.tensor([row], dtype=torch.float32))
    return fc.stacked()[:, 0, :]  # (steps, n_features)


def _col(arr: np.ndarray, name: str) -> np.ndarray:
    return arr[:, FEATURE_NAMES.index(name)]


def test_superset_shape_and_ranges() -> None:
    vocab = [0.0, 1.0, 2.0, 3.0, 0.5] + [0.0] * 195  # 200-dim
    feats = _collect([vocab, [v + 0.1 for v in vocab]])
    assert feats.shape == (2, len(FEATURE_NAMES))
    assert (_col(feats, "entropy") >= -1e-5).all()
    assert (_col(feats, "varentropy") >= -1e-5).all()
    assert (_col(feats, "h_alts") >= -1e-5).all()
    mp = _col(feats, "max_prob")
    assert ((mp >= 0) & (mp <= 1 + 1e-6)).all()
    assert (_col(feats, "chosen_logprob") <= 1e-6).all()
    # top-k mass is non-decreasing in k and within [0, 1].
    masses = [_col(feats, f"topk_mass_{k}") for k in TOPK_VALUES]
    for a, b in zip(masses[:-1], masses[1:], strict=True):
        assert (b >= a - 1e-6).all()
    assert (masses[-1] <= 1 + 1e-6).all()


def test_kl_prev_first_step_nan_then_nonneg() -> None:
    feats = _collect([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kl = _col(feats, "kl_prev")
    assert np.isnan(kl[0])
    assert (kl[1:] >= -1e-6).all()


def test_known_distributions() -> None:
    # Near-uniform over 100 -> high entropy, tiny max_prob.
    uni = _collect([[0.0] * 100])
    assert abs(_col(uni, "entropy")[0] - np.log(100)) < 1e-3
    assert abs(_col(uni, "max_prob")[0] - 0.01) < 1e-3
    # Sharp peak -> near-zero entropy, max_prob ~1.
    peak = _collect([[50.0] + [0.0] * 99])
    assert _col(peak, "entropy")[0] < 1e-3
    assert _col(peak, "max_prob")[0] > 0.999


def test_derive_causality_no_future_leakage() -> None:
    # The defining streaming property: a derived value at step t must
    # equal the value computed from the prefix up to t. If any column
    # peeked at future steps, these would differ.
    rng = np.random.default_rng(0)
    per_step = rng.standard_normal((40, len(FEATURE_NAMES))).astype(
        np.float32
    )
    full, names = derive_features(per_step)
    for t in (5, 17, 39):
        prefix, _ = derive_features(per_step[: t + 1])
        assert np.allclose(prefix[t], full[t], equal_nan=True), (
            f"future leakage at t={t}"
        )
    assert "position" in names
    assert "entropy_delta" in names
    assert "kl_prev_rmean_8" in names


def test_derive_position_and_shapes() -> None:
    per_step = np.ones((10, len(FEATURE_NAMES)), dtype=np.float32)
    aug, names = derive_features(per_step)
    assert aug.shape[0] == 10
    assert aug.shape[1] == len(names)
    pos = aug[:, names.index("position")]
    assert np.array_equal(pos, np.arange(10))
