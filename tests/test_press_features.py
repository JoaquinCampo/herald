"""Tests for press-score feature capture at prefill compression."""

from typing import Any, cast

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from herald.press_features import (
    PressScoreRecorder,
    press_feature_names,
)
from herald.presses import get_press


def _tiny_model() -> LlamaForCausalLM:
    config = cast(Any, LlamaConfig)(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = cast(Any, LlamaForCausalLM)(config)
    model.eval()
    return cast(LlamaForCausalLM, model)


def _prefill_then_step(
    model: Any, press: Any, ids: torch.Tensor
) -> tuple[torch.Tensor, tuple[int, ...]]:
    from transformers import DynamicCache

    cache = DynamicCache()
    torch.manual_seed(0)
    with torch.no_grad(), press(model):
        out = model(ids, past_key_values=cache, use_cache=True)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        out2 = model(next_id, past_key_values=cache, use_cache=True)
    keys, _ = cache[0]
    return out2.logits, tuple(keys.shape)


def test_recorder_does_not_change_evictions() -> None:
    model = _tiny_model()
    ids = (torch.arange(96) % 128).unsqueeze(0)

    plain_logits, plain_shape = _prefill_then_step(
        model, get_press("knorm", 0.5), ids
    )
    press = get_press("knorm", 0.5)
    recorder = PressScoreRecorder(press)
    recorder.begin(prompt_len=80)
    rec_logits, rec_shape = _prefill_then_step(model, press, ids)

    assert plain_shape == rec_shape
    np.testing.assert_allclose(
        rec_logits.float().numpy(),
        plain_logits.float().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


def test_recorder_features_finite_and_in_range() -> None:
    model = _tiny_model()
    ids = (torch.arange(96) % 128).unsqueeze(0)
    for name in ("knorm", "random", "streaming_llm"):
        press = get_press(name, 0.5)
        recorder = PressScoreRecorder(press)
        recorder.begin(prompt_len=80)
        _prefill_then_step(model, press, ids)
        feats = recorder.features()
        assert sorted(feats) == sorted(press_feature_names())
        for key, value in feats.items():
            assert np.isfinite(value), (name, key, value)
        for base in ("press_evicted_reliance", "press_evicted_share_prompt"):
            assert 0.0 <= feats[f"{base}_lmean"] <= 1.0, (name, base)
        assert -1.0 <= feats["press_score_reliance_align_lmean"] <= 1.0


def test_recorder_sees_every_layer_and_resets() -> None:
    model = _tiny_model()
    ids = (torch.arange(96) % 128).unsqueeze(0)
    press = get_press("knorm", 0.25)
    recorder = PressScoreRecorder(press)
    recorder.begin(prompt_len=80)
    _prefill_then_step(model, press, ids)
    assert recorder.layers_seen() == 4
    recorder.begin(prompt_len=80)
    assert recorder.layers_seen() == 0


def test_random_press_aligns_worse_than_reliance_press() -> None:
    """Random evictions should overlap reliance more than knorm's."""
    model = _tiny_model()
    ids = (torch.arange(120) % 128).unsqueeze(0)

    def align(name: str) -> float:
        press = get_press(name, 0.5)
        recorder = PressScoreRecorder(press)
        recorder.begin(prompt_len=100)
        _prefill_then_step(model, press, ids)
        return recorder.features()["press_score_reliance_align_lmean"]

    assert abs(align("random")) < 0.5
