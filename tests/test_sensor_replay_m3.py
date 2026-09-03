"""Focused end-to-end contracts for M3 layer-band replay."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from herald import magnitude_v3
from herald.magnitude_v2 import DEFAULT_EVIDENCE, DEFAULT_LOCK
from herald.press_sensors_m3 import (
    LAYER_BAND_FEATURE_NAMES,
    LAYER_BAND_PROTOCOL,
)
from herald.sensor_replay_m3 import (
    ReplayProtocolError,
    ReplayReference,
    _exact_position,
    read_allowed_reference_m3,
    replay_prompt_m3,
    validate_m3_lock,
)


def test_m3_tiny_replay_covers_s0_s16_and_emits_twelve_records() -> None:
    config = cast(
        Any,
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=None,
            pad_token_id=0,
        ),
    )
    model = LlamaForCausalLM(config).eval()
    prompt = tuple(range(1, 13))
    with torch.no_grad():
        generated = model.generate(
            torch.tensor(prompt).unsqueeze(0),
            max_new_tokens=17,
            return_dict_in_generate=True,
            do_sample=False,
        ).sequences[0, len(prompt) :]
    reference = ReplayReference(
        "tiny-m3",
        prompt,
        tuple(int(token) for token in generated.tolist()),
        Path("unused"),
    )
    boundaries = 0
    reprefills = 0
    guarded_forwards = 0
    observed_forwards = 0

    def before_boundary() -> None:
        nonlocal boundaries
        boundaries += 1

    def before_reprefill() -> None:
        nonlocal reprefills
        reprefills += 1

    def before_model_forward() -> None:
        nonlocal guarded_forwards
        guarded_forwards += 1

    def observe_forward(
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
    ) -> None:
        nonlocal observed_forwards
        observed_forwards += 1

    handle = model.register_forward_pre_hook(
        observe_forward,
        with_kwargs=True,
    )
    try:
        records = replay_prompt_m3(
            cast(Any, SimpleNamespace(model=model)),
            reference,
            before_boundary=before_boundary,
            before_reprefill=before_reprefill,
            before_model_forward=before_model_forward,
        )
    finally:
        handle.remove()
    assert boundaries == 2
    assert reprefills == 2
    assert guarded_forwards == observed_forwards
    assert guarded_forwards > boundaries + reprefills
    assert len(records) == 24
    assert {record["s"] for record in records} == {0, 16}
    assert all(
        record["protocol_version"] == LAYER_BAND_PROTOCOL
        for record in records
    )
    assert all(
        record["feature_names"] == list(LAYER_BAND_FEATURE_NAMES)
        and set(cast(dict[str, float], record["sensors"]))
        == set(LAYER_BAND_FEATURE_NAMES)
        for record in records
    )
    assert {record["state_semantics"] for record in records} == {
        "herald.cache_native_pending_v1",
        "herald.matched_reprefill_v1",
    }


def test_m3_reference_allowlist_rejects_quarantine_before_model_access(
    tmp_path: Path,
) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text())
    quarantined = str(evidence["split"]["test_prompt_ids"][0])
    with pytest.raises(ReplayProtocolError, match="quarantined"):
        read_allowed_reference_m3(
            tmp_path, "llama", "ifeval", quarantined, {"development-only"}
        )


def test_m3_reference_rejects_malformed_json_before_model_access(
    tmp_path: Path,
) -> None:
    evidence = json.loads(DEFAULT_EVIDENCE.read_text())
    prompt_id = str(evidence["split"]["train_prompt_ids"][0])
    path = tmp_path / "llama" / "ifeval" / "references" / f"{prompt_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{malformed")
    with pytest.raises(
        ReplayProtocolError, match="invalid allowed reference"
    ):
        read_allowed_reference_m3(
            tmp_path, "llama", "ifeval", prompt_id, {prompt_id}
        )


def test_m3_parent_paths_are_distinct() -> None:
    assert DEFAULT_EVIDENCE != DEFAULT_LOCK


def test_direct_replay_lock_validation_requires_model_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m3_lock = tmp_path / "m3-lock.json"
    m3_lock.write_text("{}")
    calls: list[Path] = []

    def reject_incomplete_contract(path: Path) -> dict[str, object]:
        calls.append(path)
        raise ValueError("incomplete candidate contract")

    monkeypatch.setattr(
        magnitude_v3,
        "validate_m3_protocol_lock",
        reject_incomplete_contract,
    )
    with pytest.raises(ReplayProtocolError, match="validate M3 lock"):
        validate_m3_lock(
            m3_lock,
            DEFAULT_EVIDENCE,
            DEFAULT_LOCK,
            DEFAULT_EVIDENCE.with_name("magnitude_v2_sensor_lock.json"),
            DEFAULT_EVIDENCE.with_name("magnitude_v2_m2_result_freeze.json"),
        )
    assert calls == [m3_lock]


@pytest.mark.parametrize("value", [True, 1.5, float("nan"), "16"])
def test_m3_resume_rejects_non_exact_integer_boundaries(
    value: object,
) -> None:
    with pytest.raises(
        ReplayProtocolError, match="exact integer|not an integer"
    ):
        _exact_position({"s": value})
