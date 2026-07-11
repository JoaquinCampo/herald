# pyright: reportMissingImports=false, reportOperatorIssue=false

from typing import cast

import pytest
import torch

import herald.generate as G
from herald.generate import LoadedModel, generate_int8_cache, load_model
from herald.int8_cache import (
    Int8QuantizedCache,
    Int8QuantizedLayer,
    quantize_int8,
)
from herald.tasks import PromptRecord

pytestmark = pytest.mark.model
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def lm() -> LoadedModel:
    return load_model(
        "llama",
        dtype="float32",
        device="cpu",
        attn_implementation="sdpa",
        model_id=TINY,
    )


def test_int8_quantization_tracks_scales_and_error() -> None:
    tensor = torch.linspace(-2.0, 2.0, 256).reshape(1, 2, 1, 128)
    quantized = quantize_int8(tensor)
    restored = quantized.dequantize(torch.float32)

    assert quantized.values.dtype == torch.int8
    assert quantized.scales.dtype == torch.float32
    assert quantized.scales.shape == (1, 2, 1, 1)
    assert torch.max(torch.abs(restored - tensor)) <= quantized.scales.max()
    assert quantized.nbytes == (
        quantized.values.untyped_storage().nbytes()
        + quantized.scales.untyped_storage().nbytes()
    )


def test_int8_quantization_supports_bfloat16_scale_arithmetic() -> None:
    tensor = torch.linspace(
        -2.0,
        2.0,
        256,
        dtype=torch.bfloat16,
    ).reshape(1, 2, 1, 128)

    quantized = quantize_int8(tensor, scale_dtype=torch.bfloat16)
    restored = quantized.dequantize(torch.bfloat16)

    assert quantized.values.dtype == torch.int8
    assert quantized.scales.dtype == torch.bfloat16
    assert restored.dtype == torch.bfloat16
    assert (
        torch.max(torch.abs(restored - tensor)) <= 2 * quantized.scales.max()
    )


def test_real_generation_uses_int8_storage_and_measures_it(
    lm: LoadedModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = PromptRecord(
        task="ifeval",
        prompt_id="int8-real-path",
        messages=[{"role": "user", "content": "alpha beta gamma " * 96}],
        gold={"instruction_id_list": [], "kwargs": []},
    )

    def direct_input_ids(
        loaded: LoadedModel, prompt: PromptRecord
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            loaded.tokenizer(
                prompt.messages[-1]["content"], return_tensors="pt"
            ).input_ids[0],
        )

    monkeypatch.setattr(G, "build_input_ids", direct_input_ids)
    run, cache = generate_int8_cache(
        lm,
        record,
        8,
        residual_length=4,
        scale_dtype=torch.bfloat16,
    )

    assert run.gen_ids
    assert run.peak_kv_cache_bytes > 0
    assert cache.get_seq_length() > 0
    layers = [cast(Int8QuantizedLayer, layer) for layer in cache.layers]
    assert all(
        layer.cumulative_length == cache.get_seq_length() for layer in layers
    )
    assert all(
        layer.quantized_keys.values.dtype == torch.int8 for layer in layers
    )
    assert all(
        layer.quantized_values.values.dtype == torch.int8 for layer in layers
    )
    assert all(
        layer.quantized_keys.scales.dtype == torch.bfloat16
        for layer in layers
    )
    assert cache.retained_peak_nbytes() >= run.peak_kv_cache_bytes
    assert isinstance(cache, Int8QuantizedCache)
