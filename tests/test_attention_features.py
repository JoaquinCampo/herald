"""Tests for the decode-time attention-reliance tap."""

from typing import Any, cast

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from herald.attention_features import (
    AttentionTap,
    tap_feature_names,
)


def _tiny_model(attn_implementation: str) -> LlamaForCausalLM:
    config = cast(Any, LlamaConfig)(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        attn_implementation=attn_implementation,
    )
    torch.manual_seed(0)
    model = cast(Any, LlamaForCausalLM)(config)
    model.eval()
    return cast(LlamaForCausalLM, model)


def _decode(model: Any, ids: torch.Tensor, steps: int) -> torch.Tensor:
    """Greedy decode manually to keep a live cache for the tap."""
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        seq = ids
        for _ in range(steps):
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, next_id], dim=1)
            out = model(next_id, past_key_values=cache, use_cache=True)
    return seq


def test_row_matches_eager_attention() -> None:
    model = _tiny_model("eager")
    ids = torch.arange(10).unsqueeze(0) % 128
    tap = AttentionTap(model, layer_indices=[1, 3])
    tap.begin(prompt_lens=[10])

    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        out = model(
            ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
    # ground truth: eager attention of the LAST query position,
    # averaged over heads, for each tapped layer
    for layer, got in zip([1, 3], tap.debug_last_rows(), strict=True):
        want = out.attentions[layer][0, :, -1, :].mean(dim=0).float().numpy()
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)

    # one decode step: row must cover the grown cache
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    with torch.no_grad():
        out2 = model(
            next_id,
            past_key_values=cache,
            use_cache=True,
            output_attentions=True,
        )
    for layer, got in zip([1, 3], tap.debug_last_rows(), strict=True):
        want = out2.attentions[layer][0, :, -1, :].mean(dim=0).float().numpy()
        assert got.shape == (11,)
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
    tap.remove()


def test_features_matrix_shapes_and_ranges() -> None:
    model = _tiny_model("sdpa")
    ids = torch.arange(12).unsqueeze(0) % 128
    tap = AttentionTap(model, layer_indices=[0, 2])
    tap.begin(prompt_lens=[12])
    _decode(model, ids, steps=5)
    matrix, names = tap.matrix()
    tap.remove()
    assert names == tap_feature_names()
    # prefill row + 5 decode rows = 6 steps observed
    assert matrix.shape == (1, 6, len(names))
    idx = {n: i for i, n in enumerate(names)}
    masses = [
        "attn_prompt_mass_lmean",
        "attn_sink_mass_lmean",
        "attn_local_mass_lmean",
        "attn_top8_mass_lmean",
        "attn_reliance_prompt_share_lmean",
    ]
    for name in masses:
        vals = matrix[0, :, idx[name]]
        assert np.all(vals >= -1e-6) and np.all(vals <= 1 + 1e-6)
    assert np.all(matrix[0, :, idx["attn_entropy_lmean"]] >= 0)
    assert np.all(np.isfinite(matrix[0]))


def test_column_accumulator_is_sum_of_rows() -> None:
    model = _tiny_model("eager")
    ids = torch.arange(8).unsqueeze(0) % 128
    tap = AttentionTap(model, layer_indices=[1])
    tap.begin(prompt_lens=[8])
    rows = []
    from transformers import DynamicCache

    cache = DynamicCache()
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        rows.append(tap.debug_last_rows()[0].copy())
        for _ in range(3):
            next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
            out = model(next_id, past_key_values=cache, use_cache=True)
            rows.append(tap.debug_last_rows()[0].copy())
    acc = tap.debug_accumulator(layer=1)
    tap.remove()
    want = np.zeros(len(rows[-1]))
    for row in rows:
        want[: len(row)] += row
    np.testing.assert_allclose(acc, want, rtol=1e-5, atol=1e-6)


def test_causality_prefix_invariance() -> None:
    """Step-t features must not change when more tokens follow."""
    model = _tiny_model("sdpa")
    ids = torch.arange(9).unsqueeze(0) % 128

    tap = AttentionTap(model, layer_indices=[0, 2])
    tap.begin(prompt_lens=[9])
    _decode(model, ids, steps=2)
    short, _ = tap.matrix()
    tap.remove()

    tap2 = AttentionTap(model, layer_indices=[0, 2])
    tap2.begin(prompt_lens=[9])
    _decode(model, ids, steps=6)
    long, _ = tap2.matrix()
    tap2.remove()

    np.testing.assert_allclose(
        short[0], long[0, : short.shape[1]], rtol=1e-5, atol=1e-6
    )


def test_padded_batch_matches_solo_run() -> None:
    """Left padding must not change a prompt's features."""
    model = _tiny_model("sdpa")
    short = (torch.arange(8) % 128).unsqueeze(0)
    long = (torch.arange(12) % 96 + 5).unsqueeze(0)
    pad_id = 0

    tap = AttentionTap(model, layer_indices=[1, 3])
    tap.begin(prompt_lens=[8])
    with torch.no_grad():
        model.generate(
            input_ids=short,
            attention_mask=torch.ones_like(short),
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=pad_id,
        )
    solo, _ = tap.matrix()
    tap.remove()

    padded = torch.cat([torch.full((1, 4), pad_id), short], dim=1)
    batch_ids = torch.cat([padded, long], dim=0)
    mask = torch.ones_like(batch_ids)
    mask[0, :4] = 0
    tap2 = AttentionTap(model, layer_indices=[1, 3])
    tap2.begin(prompt_lens=[8, 12])
    with torch.no_grad():
        model.generate(
            input_ids=batch_ids,
            attention_mask=mask,
            max_new_tokens=4,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=pad_id,
        )
    batched, _ = tap2.matrix()
    tap2.remove()

    assert solo.shape == (1, 4, batched.shape[2])
    assert batched.shape[0] == 2
    np.testing.assert_allclose(solo[0], batched[0], rtol=1e-3, atol=1e-4)
