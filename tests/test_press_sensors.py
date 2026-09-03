"""Focused contract tests for label-free compressor sensors."""

from typing import Any, cast

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from herald.press_sensors import (
    SENSOR_NAMES,
    SensorCaptureError,
    aggregate_layer_sensors,
    causal_recent_reliance,
    eviction_mask,
    layer_sensor_values,
    sensor_feature_names,
)


def test_knorm_evicts_largest_norms_and_floor_rule() -> None:
    scores = -torch.tensor([[[1.0, 2.0, 3.0, 4.0, 5.0]]])
    mask = eviction_mask(scores, 0.5)
    assert mask[0, 0].tolist() == [False, False, True, True, True]


def test_streaming_score_evicts_its_zero_middle_partition() -> None:
    scores = torch.ones(1, 2, 10)
    scores[..., 4:7] = 0
    mask = eviction_mask(scores, 0.25)
    assert not mask[..., :4].any()
    assert mask[..., 4:7].all()
    assert not mask[..., 7:].any()

    high_ratio_scores = torch.ones(1, 2, 10)
    high_ratio_scores[..., 4:] = 0
    high_ratio_mask = eviction_mask(high_ratio_scores, 0.875)
    assert high_ratio_mask.sum(dim=-1).tolist() == [[9, 9]]
    assert high_ratio_mask[..., :4].sum(dim=-1).tolist() == [[3, 3]]
    assert high_ratio_mask[..., 4:].all()


def test_sensor_formulas_and_full_schema() -> None:
    keys = torch.tensor([[[[1.0], [2.0], [3.0], [4.0]]]])
    values = torch.tensor([[[[2.0], [2.0], [2.0], [2.0]]]])
    scores = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
    evicted = torch.tensor([[[False, True, False, True]]])
    result = layer_sensor_values(
        keys=keys,
        values=values,
        scores=scores,
        evicted=evicted,
        prompt_len=2,
        reliance=torch.tensor([0.1, 0.2, 0.3, 0.4]),
    )
    assert result["removed_k_norm_mass_fraction"] == pytest.approx(6 / 10)
    assert result["removed_v_norm_mass_fraction"] == pytest.approx(0.5)
    assert result["prompt_region_eviction_share"] == pytest.approx(0.5)
    assert result["recent_window_eviction_share"] == pytest.approx(1.0)
    assert len(sensor_feature_names()) == len(SENSOR_NAMES) * 4 == 36
    aggregate = aggregate_layer_sensors([result, result])
    assert set(aggregate) == set(sensor_feature_names())
    assert all(np.isfinite(value) for value in aggregate.values())


def test_aggregation_fails_closed_on_missing_or_nonfinite_layers() -> None:
    with pytest.raises(SensorCaptureError):
        aggregate_layer_sensors([])
    with pytest.raises(SensorCaptureError):
        aggregate_layer_sensors(
            [{name: float("nan") for name in SENSOR_NAMES}]
        )


def test_recent_reliance_is_causal() -> None:
    config = cast(
        Any,
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=32,
        ),
    )
    model = LlamaForCausalLM(config)
    module = model.model.layers[0].self_attn
    hidden = torch.randn(1, 8, 16)
    cos, sin = model.model.rotary_emb(hidden, torch.arange(8).unsqueeze(0))
    keys = module.k_proj(hidden).view(1, 8, 1, 8).transpose(1, 2)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    _, keys = apply_rotary_pos_emb(torch.zeros_like(keys), keys, cos, sin)
    profile = causal_recent_reliance(
        module, hidden, keys, (cos, sin), recent_window=8
    )
    assert profile.shape == (1, 8)
    assert torch.isfinite(profile).all()
    assert profile[0, -1] >= 0
