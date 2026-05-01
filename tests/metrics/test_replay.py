import math

import torch as _torch

from herald.metrics.replay import compute_replay_position


def test_compute_replay_position_identity_distribution():
    vocab = 100
    logits = _torch.randn(vocab)
    row = compute_replay_position(
        logits_compressed=logits,
        logits_uncompressed=logits.clone(),
        realized_token_id=int(logits.argmax().item()),
        top_k=16,
    )
    assert row["top1_match"] is True
    assert row["js_full"] < 1e-6
    assert row["kl_unc_comp_full"] < 1e-6
    assert row["kl_comp_unc_full"] < 1e-6
    assert row["top1_rank_comp_under_unc"] == 0
    assert row["top1_rank_unc_under_comp"] == 0
    assert len(row["union_top_k_token_ids"]) <= 32
    assert (
        row["realized_logprob_compressed"]
        == row["realized_logprob_uncompressed"]
    )


def test_compute_replay_position_disjoint_top1():
    vocab = 100
    a = _torch.full((vocab,), -10.0)
    a[3] = 5.0
    b = _torch.full((vocab,), -10.0)
    b[7] = 5.0
    row = compute_replay_position(
        logits_compressed=a,
        logits_uncompressed=b,
        realized_token_id=3,
        top_k=16,
    )
    assert row["top1_match"] is False
    assert row["js_full"] > 0.1
    assert row["top1_rank_unc_under_comp"] > 0
    assert row["top1_rank_comp_under_unc"] > 0
    assert row["realized_token_id"] == 3
    assert math.isfinite(row["realized_logprob_compressed"])
    assert math.isfinite(row["realized_logprob_uncompressed"])
