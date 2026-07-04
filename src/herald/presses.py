"""KV-cache compressor factory.

Maps the five compressor names in the sweep to kvpress press objects.
All five are weight-free ScorerPresses that fire once at prefill (when
`q_len == k_len`) and skip during decode, which matches the design's
one-time-at-the-switch semantics directly.
"""

from kvpress import (
    ExpectedAttentionPress,
    KnormPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
)
from kvpress.presses.base_press import BasePress

# compression_ratio is the fraction of KV pairs REMOVED (kvpress
# convention), so 0.875 keeps 12.5 percent.
PRESS_REGISTRY: dict[str, type[BasePress]] = {
    "streaming_llm": StreamingLLMPress,
    "snapkv": SnapKVPress,
    "expected_attention": ExpectedAttentionPress,
    "knorm": KnormPress,
    "random": RandomPress,
}


def get_press(name: str, compression_ratio: float) -> BasePress:
    """Build the named press at the given removal ratio."""
    if name not in PRESS_REGISTRY:
        raise ValueError(
            f"unknown compressor {name!r}; known: {sorted(PRESS_REGISTRY)}"
        )
    return PRESS_REGISTRY[name](compression_ratio=compression_ratio)
