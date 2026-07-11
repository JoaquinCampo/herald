# pyright: reportMissingImports=false

"""KV-cache compressor factory.

Maps the five compressor names in the sweep to kvpress press objects.
All five are weight-free ScorerPresses that fire once at prefill (when
`q_len == k_len`) and skip during decode, which matches the design's
one-time-at-the-switch semantics directly.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from kvpress import (
    ExpectedAttentionPress,
    ExpectedAttentionStatsPress,
    KnormPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
)
from kvpress.presses.base_press import BasePress

from herald.expected_attention_stats import (
    StatisticsArtifact,
    make_expected_attention_stats_press,
)


@dataclass
class SafeSnapKVPress(SnapKVPress):  # type: ignore[misc]
    """SnapKV that leaves contexts no longer than its scoring window alone."""

    def compress(
        self,
        module: Any,
        hidden_states: Any,
        keys: Any,
        values: Any,
        attentions: Any,
        kwargs: dict[str, Any],
    ) -> tuple[Any, Any]:
        if int(keys.shape[2]) <= self.window_size:
            return keys, values
        return cast(
            tuple[Any, Any],
            super().compress(
                module,
                hidden_states,
                keys,
                values,
                attentions,
                kwargs,
            ),
        )


# compression_ratio is the fraction of KV pairs REMOVED (kvpress
# convention), so 0.875 keeps 12.5 percent.
PRESS_REGISTRY: dict[str, Callable[..., BasePress]] = {
    "streaming_llm": StreamingLLMPress,
    "snapkv": SafeSnapKVPress,
    "expected_attention": ExpectedAttentionPress,
    "expected_attention_stats": ExpectedAttentionStatsPress,
    "knorm": KnormPress,
    "random": RandomPress,
}


def get_press(
    name: str,
    compression_ratio: float,
    *,
    model: Any | None = None,
    statistics: StatisticsArtifact | None = None,
) -> BasePress:
    """Build a press, requiring frozen local statistics when needed."""
    if name not in PRESS_REGISTRY:
        raise ValueError(
            f"unknown compressor {name!r}; known: {sorted(PRESS_REGISTRY)}"
        )
    if name == "expected_attention_stats":
        if model is None or statistics is None:
            raise ValueError(
                "expected_attention_stats requires model and "
                "frozen statistics"
            )
        return make_expected_attention_stats_press(
            statistics,
            model,
            compression_ratio=compression_ratio,
        )
    return PRESS_REGISTRY[name](compression_ratio=compression_ratio)
