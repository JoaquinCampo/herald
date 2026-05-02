"""Generation policies.

A `Policy` decides which kvpress press is active during a generation.
The default is `FixedRatioPolicy`: one press, one compression ratio,
applied for the whole generation. This matches Phase 0 byte-for-byte.

`SwitchAtOffsetPolicy` is the pre-registered probe arm from
`gold/phase-1-intervention-probe.md`. It is a stub here; the body
lands before Block 4.

The Policy abstraction exists so the headline Phase 1 sweep can keep
calling `model.generate(...)` inside a single press context, while the
probe can switch behavior mid-generation without forking the runner.
"""

from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


class Policy(ABC):
    """A schedule of press/ratio over a single generation."""

    @abstractmethod
    def initial_press(self) -> tuple[str, float]:
        """Press name and compression ratio active at token 0."""

    @abstractmethod
    def make_context(self, model: Any) -> tuple[Any, Any]:
        """Return `(press_obj, ctx_manager)` for `model.generate(...)`.

        For FixedRatioPolicy: `(press, press(model))` or `(None,
        nullcontext())`. Returning the press object lets the runner
        query press-side telemetry (retained KV bytes, etc) without
        constructing a second press.

        Policies that switch mid-generation instead drive a manual
        decode loop; the runner branches on `requires_manual_decode`.
        """

    @property
    @abstractmethod
    def requires_manual_decode(self) -> bool:
        """True iff the runner must abandon a single `model.generate(...)`."""


@dataclass(frozen=True)
class FixedRatioPolicy(Policy):
    """Constant (press, ratio) for the whole generation.

    Phase 0 default. The headline Phase 1 sweep uses this exclusively.
    """

    press_name: str
    compression_ratio: float

    def initial_press(self) -> tuple[str, float]:
        return self.press_name, self.compression_ratio

    def make_context(self, model: Any) -> tuple[Any, Any]:
        from herald.experiment import get_press

        press = get_press(self.press_name, self.compression_ratio)
        if press is None:
            return None, nullcontext()
        return press, press(model)

    @property
    def requires_manual_decode(self) -> bool:
        return False


@dataclass(frozen=True)
class SwitchAtOffsetPolicy(Policy):
    """Switch press/ratio at token offset T.

    Probe-only. See `gold/phase-1-intervention-probe.md`.

    History preservation depends on the press family:

    - Mask-based (StreamingLLM): mask relaxes, KV intact, history kept.
    - Continuous eviction (Knorm, TOVA): stops further eviction, KV at
      offset T retained, history kept.
    - Prompt-time eviction (SnapKV, ExpectedAttention): no deployable
      lift-pressure action after prefill; this policy raises for those
      presses unless `arm == "reprefill"` (oracle, history changes).

    Body lands before Block 4. Block 1 only locks in the interface so
    the runner has the right seams.
    """

    press_before: str
    ratio_before: float
    press_after: str
    ratio_after: float
    switch_at_token: int
    arm: str = "lift_pressure"  # or "reprefill" (oracle, unpaired)

    def initial_press(self) -> tuple[str, float]:
        return self.press_before, self.ratio_before

    def make_context(self, model: Any) -> tuple[Any, Any]:
        raise NotImplementedError(
            "SwitchAtOffsetPolicy body lands before Block 4; the "
            "headline Phase 1 sweep must use FixedRatioPolicy."
        )

    @property
    def requires_manual_decode(self) -> bool:
        return True
