"""Dependency-free symmetric int8 Transformers KV cache."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from transformers.cache_utils import Cache, CacheLayerMixin, QuantizedLayer


@dataclass(frozen=True)
class Int8Tensor:
    """An int8 tensor and its per-vector symmetric scale."""

    values: torch.Tensor
    scales: torch.Tensor

    @property
    def nbytes(self) -> int:
        return _unique_storage_nbytes((self.values, self.scales))

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.values.to(torch.float32) * self.scales).to(dtype)


def quantize_int8(tensor: torch.Tensor) -> Int8Tensor:
    """Quantize independently over each final-dimension KV vector."""
    if not tensor.is_floating_point():
        raise TypeError("int8 KV quantization requires a floating tensor")
    maxima = tensor.abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scales = torch.clamp(maxima / 127.0, min=torch.finfo(torch.float32).tiny)
    values = torch.round(tensor.to(torch.float32) / scales).clamp(-127, 127)
    return Int8Tensor(values.to(torch.int8), scales)


def _unique_storage_nbytes(tensors: tuple[torch.Tensor, ...]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if identity not in seen:
            seen.add(identity)
            total += storage.nbytes()
    return total


class Int8QuantizedLayer(QuantizedLayer):
    """QuantizedLayer backend using only native PyTorch int8 tensors."""

    def __init__(self, residual_length: int = 128) -> None:
        if residual_length <= 0:
            raise ValueError("residual length must be positive")
        super().__init__(
            nbits=8,
            axis_key=-1,
            axis_value=-1,
            q_group_size=0,
            residual_length=residual_length,
        )
        self._peak_retained_nbytes = 0

    @property
    def quantized_keys(self) -> Int8Tensor:
        return cast(Int8Tensor, self._quantized_keys)

    @property
    def quantized_values(self) -> Int8Tensor:
        return cast(Int8Tensor, self._quantized_values)

    def retained_tensors(self) -> tuple[torch.Tensor, ...]:
        tensors: list[torch.Tensor] = []
        for name in ("keys", "values"):
            tensor = getattr(self, name, None)
            if isinstance(tensor, torch.Tensor):
                tensors.append(tensor)
        for name in ("_quantized_keys", "_quantized_values"):
            quantized = getattr(self, name, None)
            if isinstance(quantized, Int8Tensor):
                tensors.extend((quantized.values, quantized.scales))
        return tuple(tensors)

    def retained_nbytes(self) -> int:
        return _unique_storage_nbytes(self.retained_tensors())

    @property
    def peak_retained_nbytes(self) -> int:
        return max(self._peak_retained_nbytes, self.retained_nbytes())

    def _quantize(self, tensor: torch.Tensor, axis: int) -> Int8Tensor:
        del axis
        quantized = quantize_int8(tensor)
        existing = self.retained_nbytes()
        self._peak_retained_nbytes = max(
            self._peak_retained_nbytes,
            existing + quantized.nbytes,
        )
        return quantized

    def _dequantize(self, quantized: Int8Tensor) -> torch.Tensor:
        return quantized.dequantize(self.dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        result = super().update(key_states, value_states, cache_kwargs)
        self._peak_retained_nbytes = max(
            self._peak_retained_nbytes,
            self.retained_nbytes(),
        )
        return result


class Int8QuantizedCache(Cache):
    """Per-layer dependency-free int8 cache for decoder-only models."""

    def __init__(self, config: Any, *, residual_length: int = 128) -> None:
        text_config = config.get_text_config(decoder=True)
        layers: list[CacheLayerMixin] = [
            Int8QuantizedLayer(residual_length=residual_length)
            for _ in range(text_config.num_hidden_layers)
        ]
        super().__init__(layers=layers)

    def retained_peak_nbytes(self) -> int:
        return sum(
            cast(Int8QuantizedLayer, layer).peak_retained_nbytes
            for layer in self.layers
        )
