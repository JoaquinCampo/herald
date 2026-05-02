"""Policy abstraction tests: FixedRatioPolicy preserves Phase 0
behavior; SwitchAtOffsetPolicy is a stub that raises until Block 4."""

from contextlib import nullcontext

import pytest

from herald.policy import FixedRatioPolicy, Policy, SwitchAtOffsetPolicy


class TestFixedRatioPolicy:
    def test_initial_press_round_trip(self) -> None:
        p = FixedRatioPolicy(
            press_name="streaming_llm", compression_ratio=0.5
        )
        assert p.initial_press() == ("streaming_llm", 0.5)

    def test_no_manual_decode(self) -> None:
        p = FixedRatioPolicy(press_name="none", compression_ratio=0.0)
        assert p.requires_manual_decode is False

    def test_none_press_returns_nullcontext(self) -> None:
        p = FixedRatioPolicy(press_name="none", compression_ratio=0.0)
        press, ctx = p.make_context(model=object())
        assert press is None
        assert isinstance(ctx, type(nullcontext()))

    def test_real_press_returns_press_and_ctx(self) -> None:
        kvpress = pytest.importorskip("kvpress")
        p = FixedRatioPolicy(
            press_name="streaming_llm", compression_ratio=0.5
        )
        press, ctx = p.make_context(model=object())
        assert isinstance(press, kvpress.StreamingLLMPress)
        assert ctx is not None

    def test_is_a_policy(self) -> None:
        assert isinstance(
            FixedRatioPolicy(press_name="none", compression_ratio=0.0),
            Policy,
        )


class TestSwitchAtOffsetPolicy:
    def test_initial_press_is_press_before(self) -> None:
        p = SwitchAtOffsetPolicy(
            press_before="streaming_llm",
            ratio_before=0.5,
            press_after="streaming_llm",
            ratio_after=0.0,
            switch_at_token=32,
        )
        assert p.initial_press() == ("streaming_llm", 0.5)

    def test_requires_manual_decode(self) -> None:
        p = SwitchAtOffsetPolicy(
            press_before="streaming_llm",
            ratio_before=0.5,
            press_after="streaming_llm",
            ratio_after=0.0,
            switch_at_token=32,
        )
        assert p.requires_manual_decode is True

    def test_make_context_raises_until_block_4(self) -> None:
        p = SwitchAtOffsetPolicy(
            press_before="streaming_llm",
            ratio_before=0.5,
            press_after="streaming_llm",
            ratio_after=0.0,
            switch_at_token=32,
        )
        with pytest.raises(NotImplementedError, match="Block 4"):
            p.make_context(model=object())

    def test_is_a_policy(self) -> None:
        assert isinstance(
            SwitchAtOffsetPolicy(
                press_before="streaming_llm",
                ratio_before=0.5,
                press_after="streaming_llm",
                ratio_after=0.0,
                switch_at_token=32,
            ),
            Policy,
        )
