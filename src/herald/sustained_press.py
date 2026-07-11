# pyright: reportMissingImports=false

"""Decode-time KV compression that maintains a logical cache ratio."""

from collections import defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, cast

from kvpress.presses.base_press import BasePress
from kvpress.presses.expected_attention_with_stats import (
    ExpectedAttentionStatsPress,
)
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.utils import extract_keys_and_values

from herald.kv_metrics import kv_cache_nbytes


@dataclass
class PeakTrackingStreamingLLMPress(StreamingLLMPress):  # type: ignore[misc]
    """StreamingLLM prefill press with exact retained-cache peak tracking."""

    peak_kv_cache_bytes: int = field(init=False, default=0)

    def forward_hook(
        self,
        module: Any,
        input: list[Any],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> list[Any]:
        cache = kwargs["past_key_values"]
        hidden_states = kwargs["hidden_states"]
        is_prefill = kwargs["cache_position"][-1] <= hidden_states.shape[1]
        if is_prefill:
            self.peak_kv_cache_bytes = max(
                self.peak_kv_cache_bytes,
                kv_cache_nbytes(cache),
            )
        result = cast(
            list[Any], super().forward_hook(module, input, kwargs, output)
        )
        if is_prefill:
            self.peak_kv_cache_bytes = max(
                self.peak_kv_cache_bytes,
                kv_cache_nbytes(cache),
            )
        return result


@dataclass
class PrefillCachePeakObserver:
    """Track retained prefill cache before a later-registered press."""

    peak_kv_cache_bytes: int = field(init=False, default=0)

    def forward_hook(
        self,
        module: Any,
        input: list[Any],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> None:
        del module, input, output
        hidden_states = kwargs["hidden_states"]
        if kwargs["cache_position"][-1] <= hidden_states.shape[1]:
            self.peak_kv_cache_bytes = max(
                self.peak_kv_cache_bytes,
                kv_cache_nbytes(kwargs["past_key_values"]),
            )

    @contextmanager
    def __call__(self, model: Any) -> Generator[None]:
        language_model = (
            model.model.language_model
            if hasattr(model.model, "language_model")
            else model.model
        )
        hooks = [
            layer.self_attn.register_forward_hook(
                self.forward_hook,
                with_kwargs=True,
            )
            for layer in language_model.layers
        ]
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()


@dataclass
class SustainedRatioPress(BasePress):  # type: ignore[misc]
    """Periodically prune decode growth to a fraction of logical length."""

    base_press: StreamingLLMPress | KnormPress | ExpectedAttentionStatsPress
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
        input: list[Any],
        kwargs: dict[str, Any],
        output: list[Any],
    ) -> list[Any]:
        del input
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
