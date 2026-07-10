"""Decode-time KV compression that maintains a logical cache ratio."""

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from kvpress.presses.base_press import BasePress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.utils import extract_keys_and_values

from herald.kv_metrics import kv_cache_nbytes


@dataclass
class SustainedRatioPress(BasePress):  # type: ignore[misc]
    """Periodically prune decode growth to a fraction of logical length."""

    base_press: StreamingLLMPress | KnormPress
    compression_ratio: float
    interval: int = 32
    peak_kv_cache_bytes: int = field(init=False, default=0)
    _steps: dict[int, int] = field(
        init=False, default_factory=lambda: defaultdict(int)
    )

    def __post_init__(self) -> None:
        if not 0 < self.compression_ratio < 1:
            raise ValueError("compression_ratio must be between zero and one")
        if self.interval <= 0:
            raise ValueError("interval must be positive")

    def post_init_from_model(self, model: Any) -> None:
        self.base_press.post_init_from_model(model)

    def compress(
        self,
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        logical_len = int(kwargs["cache_position"][-1].item()) + 1
        target = max(1, int(logical_len * (1 - self.compression_ratio)))
        physical_len = int(keys.shape[2])
        if physical_len <= target:
            return keys, values

        original_ratio = self.base_press.compression_ratio
        self.base_press.compression_ratio = 1 - target / physical_len
        try:
            return cast(
                tuple[Any, Any],
                self.base_press.compress(
                    module,
                    hidden_states,
                    keys,
                    values,
                    attentions,
                    kwargs,
                ),
            )
        finally:
            self.base_press.compression_ratio = original_ratio

    def forward_hook(
        self,
        module: Any,
        inputs: list[Any],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> list[Any]:
        del inputs
        hidden_states = kwargs["hidden_states"]
        if kwargs["cache_position"][-1] <= hidden_states.shape[1]:
            return output

        cache = kwargs["past_key_values"]
        layer_idx = int(module.layer_idx)
        self._steps[layer_idx] += 1
        if self._steps[layer_idx] < self.interval:
            return output
        if layer_idx == 0:
            self.peak_kv_cache_bytes = max(
                self.peak_kv_cache_bytes,
                kv_cache_nbytes(cache),
            )

        keys, values = extract_keys_and_values(cache, layer_idx)
        keys, values = self.compress(
            module,
            hidden_states,
            keys,
            values,
            output[1],
            kwargs,
        )
        cache.layers[layer_idx].keys = keys
        cache.layers[layer_idx].values = values
        self._steps[layer_idx] = 0
        return output

    @contextmanager
    def __call__(self, model: Any) -> Generator[None]:
        try:
            with super().__call__(model):
                yield
        finally:
            self._steps.clear()
