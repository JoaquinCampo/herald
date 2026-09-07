"""CPU acceptance tests for the bounded reference-prefix lookahead."""

from typing import Any

import pytest
import torch
from scripts import measure_lookahead as measure_module
from scripts.measure_lookahead import measure_prompt
from tests.test_engine import _input_ids, _tiny_model
from tests.test_runner import _manifest, _Tokenizer

from herald_v3.engineering import engine
from herald_v3.engineering.lookahead import (
    FIRST_OUTPUT_INDEX,
    MAX_LOOKAHEAD_STEPS,
    run_lookahead,
)


def test_eight_steps_use_reference_prefix_and_preserve_source() -> None:
    model = _tiny_model()
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())
    source_state = boundary.state_fingerprint
    source_rng = boundary.rng_state.fingerprint

    result = run_lookahead(
        model,
        boundary,
        engine.ActionSpec("knorm", 0.0),
        eos_ids=frozenset(),
    )

    assert result.passed, result.to_dict()
    assert len(result.steps) == MAX_LOOKAHEAD_STEPS
    assert [step.output_index for step in result.steps] == list(range(32, 40))
    assert [step.input_position for step in result.steps] == list(
        range(boundary.logical_position, boundary.logical_position + 8)
    )
    assert result.steps[0].input_token_id == boundary.pending_token_id
    assert all(
        step.input_token_id == result.steps[index - 1].reference_argmax
        for index, step in enumerate(result.steps)
        if index > 0
    )
    assert all(
        step.probe.js_divergence <= 1e-6
        and step.probe.max_probability_difference <= 1e-6
        and step.probe.max_logit_difference <= 1e-6
        for step in result.steps
    )
    assert result.boundary_state_fingerprint_after == source_state
    assert result.boundary_rng_fingerprint_after == source_rng
    assert "logits" not in result.to_dict()


def test_step_zero_matches_original_probe() -> None:
    model = _tiny_model()
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())
    action = engine.ActionSpec("knorm", 0.5)

    lookahead = run_lookahead(
        model,
        boundary,
        action,
        eos_ids=frozenset(),
        max_steps=1,
    )
    original = engine.probe_action(model, boundary, action)
    observed = lookahead.steps[0].probe

    assert observed.js_divergence == pytest.approx(
        original.js_divergence, abs=1e-12
    )
    assert observed.reference_argmax == original.reference_argmax
    assert observed.action_argmax == original.action_argmax
    assert observed.max_probability_difference == pytest.approx(
        original.max_probability_difference, abs=1e-12
    )
    assert observed.max_logit_difference == pytest.approx(
        original.max_logit_difference, abs=1e-12
    )


def test_action_eos_does_not_stop_reference_prefix() -> None:
    model = _tiny_model(seed=12)
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())
    forced_action_eos = {51}
    active = False

    def force_action_eos(
        module: Any, inputs: tuple[Any, ...], output: Any
    ) -> Any:
        del module, inputs
        if not active:
            return output
        cache = output.past_key_values
        if cache.get_seq_length() < boundary.cache_lengths[0]:
            output.logits.fill_(-1_000_000.0)
            output.logits[:, :, 51] = 1_000_000.0
        return output

    handle = model.register_forward_hook(force_action_eos)
    try:
        active = True
        result = run_lookahead(
            model,
            boundary,
            engine.ActionSpec("knorm", 0.5),
            eos_ids=forced_action_eos,
        )
    finally:
        handle.remove()

    assert result.passed, result.to_dict()
    assert len(result.steps) == MAX_LOOKAHEAD_STEPS
    assert all(step.action_argmax == 51 for step in result.steps)
    assert all(not step.reference_argmax_is_eos for step in result.steps)


def test_reference_eos_stops_after_the_eos_distribution() -> None:
    model = _tiny_model(seed=12)
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())
    baseline = run_lookahead(
        model,
        boundary,
        engine.ActionSpec("knorm", 0.0),
        eos_ids=frozenset(),
        max_steps=1,
    )
    eos_id = baseline.steps[0].reference_argmax

    result = run_lookahead(
        model,
        boundary,
        engine.ActionSpec("knorm", 0.5),
        eos_ids={eos_id},
    )

    assert result.passed, result.to_dict()
    assert len(result.steps) == 1
    assert result.steps[0].output_index == FIRST_OUTPUT_INDEX
    assert result.steps[0].reference_argmax_is_eos


def test_lookahead_cap_is_hard() -> None:
    model = _tiny_model()
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())

    with pytest.raises(ValueError, match="max_steps"):
        run_lookahead(
            model,
            boundary,
            engine.ActionSpec("knorm", 0.5),
            max_steps=MAX_LOOKAHEAD_STEPS + 1,
        )


def test_rng_consumption_is_reported_and_caller_rng_is_restored() -> None:
    model = _tiny_model()
    boundary = engine.build_boundary(model, _input_ids(), eos_ids=frozenset())

    def consume_rng(module: Any, inputs: tuple[Any, ...]) -> None:
        del module, inputs
        torch.rand(())

    handle = model.register_forward_pre_hook(consume_rng)
    try:
        result = run_lookahead(
            model,
            boundary,
            engine.ActionSpec("knorm", 0.0),
            max_steps=1,
        )
    finally:
        handle.remove()

    assert not result.passed
    assert not result.source_rng_preserved
    assert (
        result.boundary_rng_fingerprint_after
        == result.boundary_rng_fingerprint_before
    )


def test_measure_prompt_records_action_and_noop_without_outcomes() -> None:
    measurement = measure_prompt(
        _tiny_model(),
        _Tokenizer(),
        _manifest().prompts[0],
        max_steps=2,
    )

    assert measurement["passed"] is True
    action = measurement["lookahead"]
    noop = measurement["noop_control"]
    configuration = measurement["configuration"]
    assert isinstance(action, dict)
    assert isinstance(noop, dict)
    assert isinstance(configuration, dict)
    assert action["realized_steps"] == 2
    assert noop["realized_steps"] == 2
    assert configuration["stored_outcomes"] is False
    assert measurement["checks"] == {
        "lookahead": True,
        "noop_control": True,
        "step0_parity": True,
        "continuation_transparency": True,
    }
    diagnostics = measurement["diagnostics"]
    assert isinstance(diagnostics, dict)
    transparency = diagnostics["continuation_transparency"]
    assert isinstance(transparency, dict)
    assert transparency["token_ids_equal"]
    assert transparency["termination_equal"]
    assert transparency["final_cache_fingerprint_equal"]


def test_measure_prompt_fails_when_step_zero_parity_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        measure_module,
        "_step_zero_parity",
        lambda *args: {
            "passed": False,
            "checks": {"forced_failure": False},
            "wall_seconds": 0.0,
        },
    )

    measurement = measure_prompt(
        _tiny_model(),
        _Tokenizer(),
        _manifest().prompts[0],
        max_steps=1,
    )

    assert measurement["passed"] is False
    assert measurement["checks"]["step0_parity"] is False  # type: ignore[index]


def test_measure_prompt_fails_when_continuations_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_short = measure_module._short_continuation
    calls = 0

    def mismatch_after_probe(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal calls
        calls += 1
        result = dict(original_short(*args, **kwargs))
        if calls == 2:
            result["token_ids"] = [999]
            result["sha256"] = "tampered"
        return result

    monkeypatch.setattr(
        measure_module, "_short_continuation", mismatch_after_probe
    )
    measurement = measure_prompt(
        _tiny_model(),
        _Tokenizer(),
        _manifest().prompts[0],
        max_steps=1,
    )

    assert measurement["passed"] is False
    assert (
        measurement["checks"]["continuation_transparency"] is False  # type: ignore[index]
    )
