"""End-to-end contracts for label-free M2 sensor replay."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from herald.magnitude_v2 import DEFAULT_EVIDENCE, DEFAULT_LOCK, sha256_file
from herald.press_sensors import sensor_feature_names
from herald.sensor_replay import (
    PROTOCOL_VERSION,
    ReplayConfig,
    ReplayProtocolError,
    ReplayReference,
    locked_prompt_ids,
    read_allowed_reference,
    replay_locked,
    replay_prompt,
    validate_sensor_lock,
)
from herald.storage import (
    append_sensor_record,
    read_sensor_records,
    sensor_sidecar_path,
    validate_sensor_records,
    write_sensor_manifest,
)


def _record(
    *,
    compressor: str = "knorm",
    ratio: float = 0.5,
    prefix_hash: str = "0" * 64,
) -> dict[str, object]:
    return {
        "prompt_id": "p-1",
        "compressor": compressor,
        "ratio": ratio,
        "s": 0,
        "sensors": {"sensor": 0.5},
        "prefix_hash": prefix_hash,
        "protocol_version": "protocol",
        "feature_names": ["sensor"],
        "evidence_sha256": "evidence",
        "lock_sha256": "lock",
        "sensor_lock_sha256": "sensor-lock",
        "model_key": "llama",
        "task": "ifeval",
    }


def test_locked_allowlist_rejects_quarantine_before_reference_access(
    tmp_path: Path,
) -> None:
    allowed, _, _ = locked_prompt_ids(DEFAULT_EVIDENCE, DEFAULT_LOCK)
    evidence = json.loads(DEFAULT_EVIDENCE.read_text())
    quarantined = str(evidence["split"]["test_prompt_ids"][0])
    sensor_lock = DEFAULT_LOCK.with_name("magnitude_v2_sensor_lock.json")
    assert (
        len(validate_sensor_lock(sensor_lock, DEFAULT_EVIDENCE, DEFAULT_LOCK))
        == 64
    )
    assert len(allowed) == 154
    with pytest.raises(ReplayProtocolError, match="quarantined"):
        read_allowed_reference(
            tmp_path,
            "llama",
            "ifeval",
            quarantined,
            allowed,
        )


def test_allowed_reference_requires_generated_tokens(tmp_path: Path) -> None:
    allowed, _, _ = locked_prompt_ids(DEFAULT_EVIDENCE, DEFAULT_LOCK)
    prompt_id = sorted(allowed)[0]
    path = tmp_path / "llama" / "ifeval" / "references" / f"{prompt_id}.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "prompt_input_ids": [1],
                "gen_ids": [],
            }
        )
    )
    with pytest.raises(ReplayProtocolError, match="invalid token IDs"):
        read_allowed_reference(
            tmp_path,
            "llama",
            "ifeval",
            prompt_id,
            allowed,
        )


def test_sidecar_repairs_only_torn_final_append(tmp_path: Path) -> None:
    append_sensor_record(tmp_path, "llama", "ifeval", _record())
    path = sensor_sidecar_path(tmp_path, "llama", "ifeval")
    with path.open("ab") as stream:
        stream.write(b'{"torn"')
    records = read_sensor_records(tmp_path, "llama", "ifeval")
    assert len(records) == 1
    assert path.read_bytes().endswith(b"\n")

    with path.open("ab") as stream:
        stream.write(b"{not-json}\n")
    with pytest.raises(ValueError, match="invalid complete sensor JSON"):
        read_sensor_records(tmp_path, "llama", "ifeval")


def test_sensor_validation_rejects_prefix_or_provenance_disagreement() -> (
    None
):
    first = _record(compressor="knorm")
    second = _record(
        compressor="expected_attention",
        prefix_hash="1" * 64,
    )
    with pytest.raises(ValueError, match="disagree on prefix"):
        validate_sensor_records(
            [first, second],
            model="llama",
            task="ifeval",
            evidence_sha256="evidence",
            lock_sha256="lock",
            sensor_lock_sha256="sensor-lock",
            protocol_version="protocol",
            feature_names=["sensor"],
        )
    first["lock_sha256"] = "wrong"
    with pytest.raises(ValueError, match="provenance mismatch"):
        validate_sensor_records(
            [first],
            model="llama",
            task="ifeval",
            evidence_sha256="evidence",
            lock_sha256="lock",
            sensor_lock_sha256="sensor-lock",
            protocol_version="protocol",
            feature_names=["sensor"],
        )


def test_resume_recomputes_prefix_hash_before_loading_model(
    tmp_path: Path,
) -> None:
    allowed, evidence_hash, lock_hash = locked_prompt_ids(
        DEFAULT_EVIDENCE,
        DEFAULT_LOCK,
    )
    prompt_id = sorted(allowed)[0]
    results_root = tmp_path / "results"
    reference_path = (
        results_root / "llama" / "ifeval" / "references" / f"{prompt_id}.json"
    )
    reference_path.parent.mkdir(parents=True)
    reference_path.write_text(
        json.dumps(
            {
                "prompt_id": prompt_id,
                "prompt_input_ids": [1, 2, 3, 4, 5],
                "gen_ids": [6],
            }
        )
    )
    sensor_lock_path = DEFAULT_LOCK.with_name("magnitude_v2_sensor_lock.json")
    feature_names = sensor_feature_names()
    output_root = tmp_path / "sensors"
    stale = _record(prefix_hash="0" * 64)
    stale.update(
        {
            "prompt_id": prompt_id,
            "sensors": dict.fromkeys(feature_names, 0.0),
            "feature_names": feature_names,
            "protocol_version": PROTOCOL_VERSION,
            "evidence_sha256": evidence_hash,
            "lock_sha256": lock_hash,
            "sensor_lock_sha256": sha256_file(sensor_lock_path),
        }
    )
    append_sensor_record(output_root, "llama", "ifeval", stale)

    def fail_model_load(*args: object, **kwargs: object) -> Any:
        raise AssertionError("prefix validation must precede model loading")

    config = ReplayConfig(
        results_root=results_root,
        evidence_path=DEFAULT_EVIDENCE,
        lock_path=DEFAULT_LOCK,
        sensor_lock_path=sensor_lock_path,
        output_root=output_root,
        device="cpu",
    )
    with pytest.raises(ReplayProtocolError, match="prefix hash mismatch"):
        replay_locked(
            config,
            prompt_ids=[prompt_id],
            model_loader=fail_model_load,
        )


def test_sensor_manifest_binds_sidecar_content(tmp_path: Path) -> None:
    record = _record()
    append_sensor_record(tmp_path, "llama", "ifeval", record)
    key = ("p-1", "knorm", 0.5, 0)
    manifest_path = write_sensor_manifest(
        tmp_path,
        "llama",
        "ifeval",
        expected_keys={key},
        evidence_sha256="evidence",
        lock_sha256="lock",
        sensor_lock_sha256="sensor-lock",
        protocol_version="protocol",
        feature_names=["sensor"],
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["record_count"] == 1
    assert len(manifest["sidecar_sha256"]) == 64
    assert manifest["feature_names"] == ["sensor"]


def test_tiny_replay_parity_and_dual_state_semantics() -> None:
    config = cast(
        Any,
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            eos_token_id=None,
            pad_token_id=0,
        ),
    )
    model = LlamaForCausalLM(config).eval()
    prompt = tuple(range(1, 13))
    input_ids = torch.tensor(prompt).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=17,
            return_dict_in_generate=True,
            do_sample=False,
        ).sequences[0, len(prompt) :]
    reference = ReplayReference(
        "p-1",
        prompt,
        tuple(int(token) for token in generated.tolist()),
        Path("unused"),
    )
    boundaries = 0
    reprefills = 0

    def before_boundary() -> None:
        nonlocal boundaries
        boundaries += 1

    def before_reprefill() -> None:
        nonlocal reprefills
        reprefills += 1

    records = replay_prompt(
        cast(Any, SimpleNamespace(model=model)),
        reference,
        before_boundary=before_boundary,
        before_reprefill=before_reprefill,
    )
    assert boundaries == 2
    assert reprefills == 2
    assert len(records) == 24
    assert {record["s"] for record in records} == {0, 16}
    assert all(
        set(cast(dict[str, float], record["sensors"]))
        == set(sensor_feature_names())
        for record in records
    )
    streaming_quarter = next(
        record
        for record in records
        if record["compressor"] == "streaming_llm"
        and record["ratio"] == 0.25
        and record["s"] == 0
    )
    sensors = cast(dict[str, float], streaming_quarter["sensors"])
    assert sensors["cutoff_margin_lmean"] == pytest.approx(1.0)
