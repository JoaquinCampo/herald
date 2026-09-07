"""CPU acceptance tests using a real randomly initialized decoder."""

import json
from typing import Any, cast

import pytest
import torch
from kvpress.presses.knorm_press import KnormPress
from transformers import (
    DynamicCache,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from herald_v3.engineering.engine import (
    ActionSpec,
    build_boundary,
    cache_fingerprint,
    cache_storage_independent,
    cache_tensors_equal,
    clone_cache,
    compress_knorm,
    decoder_state_fingerprint,
    full_vocabulary_js,
    run_acceptance,
)


def _tiny_model(seed: int = 7) -> LlamaForCausalLM:
    config = cast(
        Any,
        LlamaConfig(
            vocab_size=67,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=None,
            pad_token_id=0,
            attn_implementation="sdpa",
        ),
    )
    torch.manual_seed(seed)
    return cast(LlamaForCausalLM, LlamaForCausalLM(config).eval())


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 5, 9, 13, 17, 21]], dtype=torch.long)


def _filled_cache(model: LlamaForCausalLM) -> DynamicCache:
    with torch.no_grad():
        output = model(
            input_ids=torch.arange(1, 13).unsqueeze(0),
            use_cache=True,
            return_dict=True,
        )
    return cast(DynamicCache, output.past_key_values)


def test_boundary_is_exact_and_cache_copies_are_independent() -> None:
    model = _tiny_model()
    prompt = _input_ids()
    boundary = build_boundary(model, prompt, eos_ids=frozenset())

    assert len(boundary.generated_ids) == 32
    assert boundary.pending_token_id == boundary.generated_ids[31]
    assert boundary.logical_position == prompt.shape[1] + 31
    assert boundary.cache_lengths == (prompt.shape[1] + 31,)
    assert boundary.attention_mask.shape == (1, prompt.shape[1] + 31)
    assert boundary.attention_mask.all()
    assert boundary.to_dict()["attention_mask"] == [1] * (
        prompt.shape[1] + 31
    )

    recomputed = decoder_state_fingerprint(
        boundary.prompt_ids,
        boundary.generated_ids,
        boundary.cache,
        boundary.logical_position,
        attention_mask=boundary.attention_mask,
        pending_token_id=boundary.pending_token_id,
        rng_fingerprint=boundary.rng_state.fingerprint,
        model_state_fingerprint=boundary.model_state_fingerprint,
    )
    assert recomputed == boundary.state_fingerprint
    changed_mask = boundary.attention_mask.clone()
    changed_mask[0, 0] = 0
    assert (
        decoder_state_fingerprint(
            boundary.prompt_ids,
            boundary.generated_ids,
            boundary.cache,
            boundary.logical_position,
            attention_mask=changed_mask,
            pending_token_id=boundary.pending_token_id,
            rng_fingerprint=boundary.rng_state.fingerprint,
            model_state_fingerprint=boundary.model_state_fingerprint,
        )
        != boundary.state_fingerprint
    )

    cloned = clone_cache(boundary.cache)
    assert cache_storage_independent(boundary.cache, cloned)
    assert cache_tensors_equal(boundary.cache, cloned)
    source_fingerprint = cache_fingerprint(boundary.cache)
    with torch.no_grad():
        cloned.layers[0].keys.reshape(-1)[0].add_(1.0)
    assert cache_fingerprint(boundary.cache) == source_fingerprint
    assert not cache_tensors_equal(boundary.cache, cloned)


def test_knorm_matches_kvpress_and_keeps_smallest_key_norms() -> None:
    model = _tiny_model()
    source = _filled_cache(model)
    candidate = clone_cache(source)
    evidence = compress_knorm(candidate, 0.5)

    source_keys = cast(torch.Tensor, source.layers[0].keys)
    source_values = cast(torch.Tensor, source.layers[0].values)
    attention = model.model.layers[0].self_attn
    press = KnormPress(compression_ratio=0.5)
    expected_keys, expected_values = press.compress(
        attention,
        torch.empty((1, source_keys.shape[2], 0)),
        source_keys,
        source_values,
        torch.empty(0),
        {},
    )

    assert torch.equal(candidate.layers[0].keys, expected_keys)
    assert torch.equal(candidate.layers[0].values, expected_values)
    expected_indices = (-source_keys.norm(dim=-1)).topk(6, dim=-1).indices
    assert evidence.kept_indices[0][0] == tuple(
        int(value) for value in expected_indices[0, 0].tolist()
    )
    retained_norms = source_keys.norm(dim=-1).gather(2, expected_indices)
    evicted_mask = torch.ones_like(source_keys.norm(dim=-1), dtype=torch.bool)
    evicted_mask.scatter_(2, expected_indices, False)
    evicted_norms = source_keys.norm(dim=-1)[evicted_mask]
    assert retained_norms.max() <= evicted_norms.min()
    assert evidence.before_lengths == (12,)
    assert evidence.after_lengths == (6,)
    assert evidence.after_bytes == evidence.before_bytes // 2


def test_full_vocabulary_js_is_stable_symmetric_and_compact() -> None:
    reference = torch.tensor([1000.0, 999.0, -1000.0, -2000.0])
    action = torch.tensor([999.0, 1000.0, -1500.0, -2500.0])
    forward = full_vocabulary_js(reference, action)
    reverse = full_vocabulary_js(action, reference)
    noop = full_vocabulary_js(reference, reference)

    assert forward.finite
    assert forward.js_divergence == pytest.approx(
        reverse.js_divergence, abs=1e-15
    )
    assert forward.js_divergence > 0.0
    assert forward.js_divergence <= torch.log(torch.tensor(2.0)).item()
    assert noop.js_divergence == pytest.approx(0.0, abs=1e-15)
    assert noop.max_probability_difference == 0.0
    assert forward.reference_probability_sum == pytest.approx(1.0)
    assert forward.action_probability_sum == pytest.approx(1.0)
    assert "logits" not in forward.to_dict()
    assert "probabilities" not in forward.to_dict()


def test_real_tiny_model_acceptance_covers_parity_actions_and_order() -> None:
    result = run_acceptance(
        _tiny_model(),
        _input_ids(),
        max_new_tokens=36,
        eos_ids=frozenset(),
        ratios=(0.25, 0.5),
    )

    assert result.passed, [gate.to_dict() for gate in result.gates]
    assert result.uninterrupted is not None
    assert len(result.noop_forks) == 2
    assert len(result.action_arms) == 2
    assert [arm.action.removal_fraction for arm in result.action_arms] == [
        0.25,
        0.5,
    ]
    assert [
        arm.action.removal_fraction for arm in result.reverse_action_arms
    ] == [
        0.5,
        0.25,
    ]
    assert all(
        arm.continuation.token_ids[:32] == result.boundary.generated_ids  # type: ignore[union-attr]
        for arm in result.action_arms
    )
    gate_names = {gate.name for gate in result.gates}
    assert "noop_uninterrupted_token_parity" in gate_names
    assert "noop_distribution_parity" in gate_names
    assert "physical_eviction:knorm:0.25" in gate_names
    assert "physical_eviction:knorm:0.5" in gate_names

    payload = result.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    assert "checkpoint_fingerprint" in encoded
    assert "generated_token_ids" in encoded
    assert "vocab_distribution" not in encoded
    assert "past_key_values" not in encoded
    assert result.timing_seconds["validation_overhead"] > 0.0
    assert result.memory["reference_state_preserved_during_measurement"]
    assert (
        result.memory["without_reference_state_peak_allocated_bytes"] is None
    )
    assert (
        result.memory["without_reference_state_peak_status"]
        == "not_measured_requires_separate_run"
    )
    assert all(
        arm.compression.validation_seconds > 0.0
        and arm.continuation.validation_seconds > 0.0
        for arm in (*result.noop_forks, *result.action_arms)
    )


def test_noop_only_mode_still_runs_parity_and_js() -> None:
    result = run_acceptance(
        _tiny_model(),
        _input_ids(),
        max_new_tokens=34,
        eos_ids=frozenset(),
        ratios=(),
    )

    assert result.passed
    assert result.action_arms == ()
    assert result.reverse_action_arms == ()
    assert {gate.name for gate in result.gates} >= {
        "boundary_exact",
        "independent_equal_forks",
        "noop_uninterrupted_token_parity",
        "noop_distribution_parity",
    }


def test_tiny_qwen2_noop_matches_hugging_face_greedy_generate() -> None:
    config = cast(
        Any,
        Qwen2Config(
            vocab_size=71,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=None,
            pad_token_id=0,
            attn_implementation="sdpa",
        ),
    )
    torch.manual_seed(8)
    model = cast(Qwen2ForCausalLM, Qwen2ForCausalLM(config).eval())
    prompt = torch.tensor([[1, 3, 5, 7, 9]], dtype=torch.long)
    result = run_acceptance(
        model,
        prompt,
        max_new_tokens=34,
        eos_ids=frozenset(),
        ratios=(0.5,),
    )
    with torch.no_grad():
        generated_output = cast(
            Any,
            model.generate(
                prompt,
                max_new_tokens=34,
                do_sample=False,
                return_dict_in_generate=True,
            ),
        )
        generated = generated_output.sequences[0, prompt.shape[1] :]

    assert result.passed, [gate.to_dict() for gate in result.gates]
    assert result.uninterrupted is not None
    assert result.uninterrupted.token_ids == tuple(
        int(token) for token in generated.tolist()
    )


def test_early_eos_is_an_ineligibility_record() -> None:
    model = _tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = run_acceptance(
        model,
        _input_ids(),
        max_new_tokens=36,
        eos_ids=frozenset({0}),
    )

    assert not result.passed
    assert not result.eligibility.eligible
    assert result.eligibility.reason == "eos_at_or_before_pending_boundary"
    assert result.eligibility.generated_token_ids == (0,)
    assert result.boundary is None
    assert result.gates == ()


def test_action_validation_rejects_non_knorm_and_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="unsupported action"):
        ActionSpec("random", 0.5)
    with pytest.raises(ValueError, match="removal_fraction"):
        ActionSpec("knorm", 1.0)
    with pytest.raises(ValueError, match="decision_tokens=32"):
        run_acceptance(
            _tiny_model(),
            _input_ids(),
            max_new_tokens=8,
            decision_tokens=4,
        )


def test_acceptance_detects_model_and_rng_mutation() -> None:
    model = _tiny_model()
    mutation_counter = torch.zeros(())
    model.register_buffer("acceptance_mutation_counter", mutation_counter)

    def mutate_state(
        module: LlamaForCausalLM, inputs: tuple[Any, ...]
    ) -> None:
        del module
        del inputs
        torch.add(mutation_counter, 1, out=mutation_counter)
        torch.rand(())

    handle = model.register_forward_pre_hook(mutate_state)
    try:
        result = run_acceptance(
            model,
            _input_ids(),
            max_new_tokens=34,
            eos_ids=frozenset(),
            ratios=(),
        )
    finally:
        handle.remove()

    gates = {gate.name: gate for gate in result.gates}
    assert not result.passed
    assert not gates["model_state_unchanged_by_controls"].passed
    assert not gates["rng_state_unchanged_by_controls"].passed
