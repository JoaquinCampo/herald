"""Pure CPU tests for the label-blind AttentionTap pilot."""

import argparse
import hashlib
import importlib.util
import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from herald import generate as generate_module
from herald.attention_features import tap_feature_names
from herald.attention_replay import (
    PILOT_ROSTER,
    TAP_LAYER_INDICES,
    AttentionReplayError,
    audit_arrays,
    classify_decision,
    hash_prompt_ids,
    load_prefix_hash_sidecar,
    prefix_hash,
    read_legacy_identity_prefix,
    sha256_file,
    validate_protocol_lock,
    validate_report,
    write_report,
)
from herald.config import MODELS
from herald.features import FEATURE_NAMES
from herald.ifeval import load_ifeval_exact
from herald.tasks import PromptRecord

_HASH = "0" * 64
_INPUT_HASH_NAMES = (
    "source_oof",
    "legacy_config",
    "legacy_manifest",
    "prefix_sidecar",
    "prefix_manifest",
    "attention_source",
    "generate_source",
    "ifeval_source",
    "attention_replay_source",
    "features_source",
    "script",
)


def _load_pilot_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_attention_tap_pilot.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_attention_tap_pilot", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PILOT = _load_pilot_module()


class _Column:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def __iter__(self) -> Iterator[object]:
        return iter(self.values)


class _Data:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def column(self, name: str) -> _Column:
        assert name == "key"
        return _Column(self.values)


class _Dataset:
    def __init__(
        self,
        rows: list[dict[str, object]],
        audit: list[tuple[str, object]] | None = None,
    ) -> None:
        self.rows = rows
        self.audit = audit if audit is not None else []
        self.data = _Data([row["key"] for row in rows])

    def select(self, indices: list[int]) -> "_Dataset":
        self.audit.append(("select", list(indices)))
        assert all(set(row) <= {"key", "prompt"} for row in self.rows)
        return _Dataset([self.rows[index] for index in indices], self.audit)

    def select_columns(self, columns: list[str]) -> "_Dataset":
        self.audit.append(("columns", list(columns)))
        return _Dataset(
            [{name: row[name] for name in columns} for row in self.rows],
            self.audit,
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)


class _FallbackDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows

    def select_columns(self, columns: list[str]) -> "_FallbackDataset":
        return _FallbackDataset(
            [{name: row[name] for name in columns} for row in self.rows]
        )

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.rows)


def _module_path(module: ModuleType) -> Path:
    if module.__file__ is None:
        raise AssertionError("test module has no source path")
    return Path(module.__file__).resolve()


def _pilot_args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        attention_source=_module_path(PILOT.attention_features_module),
        generate_source=_module_path(PILOT.generate_module),
        ifeval_source=_module_path(PILOT.ifeval_module),
        protocol_lock=tmp_path / "lock.json",
        legacy_reference_dir=tmp_path / "references",
        legacy_manifest=tmp_path / "manifest.json",
        legacy_prefix_sidecar=tmp_path / "prefix.jsonl",
        legacy_prefix_manifest=tmp_path / "prefix.manifest.json",
        source_oof=tmp_path / "oof.parquet",
        legacy_config=tmp_path / "config.json",
        output_root=tmp_path / "output",
        model_id=MODELS["llama"],
        device="cuda",
    )


def _protocol_lock(args: argparse.Namespace) -> dict[str, object]:
    return {
        "schema_version": "herald.attention_tap_pilot_protocol_lock.v1",
        "status": "locked_before_h5_r0_gpu",
        "pilot_roster": [
            {"fold": fold, "prompt_id": prompt_id}
            for fold, prompt_id in PILOT_ROSTER
        ],
        "pilot_prompt_ids_sha256": hash_prompt_ids(
            [prompt_id for _, prompt_id in PILOT_ROSTER]
        ),
        "generation": {
            "model_key": "llama",
            "model_id": MODELS["llama"],
            "dtype": "bfloat16",
            "device": "cuda",
            "attn_implementation": "sdpa",
            "batch_size": 1,
            "max_new_tokens": 1024,
            "tap_layer_indices": list(TAP_LAYER_INDICES),
        },
        "paths": {
            "attention_source": str(args.attention_source.resolve()),
            "generate_source": str(args.generate_source.resolve()),
            "ifeval_source": str(args.ifeval_source.resolve()),
            "attention_replay_source": str(
                _module_path(PILOT.attention_replay_module)
            ),
            "features_source": str(_module_path(PILOT.features_module)),
            "script_source": str(_module_path(PILOT)),
            "protocol_lock": str(args.protocol_lock.resolve()),
            "source_oof": str(args.source_oof.resolve()),
            "legacy_config": str(args.legacy_config.resolve()),
            "legacy_reference_dir": str(args.legacy_reference_dir.resolve()),
            "legacy_manifest": str(args.legacy_manifest.resolve()),
            "legacy_prefix_sidecar": str(
                args.legacy_prefix_sidecar.resolve()
            ),
            "legacy_prefix_manifest": str(
                args.legacy_prefix_manifest.resolve()
            ),
            "output_root": str(args.output_root.resolve()),
        },
        "input_hashes": {
            f"{name}_sha256": _HASH for name in _INPUT_HASH_NAMES
        },
        "access_policy": {
            "decode_text": False,
            "load_gold": False,
            "scorer": False,
            "outcomes": False,
            "protected_prompts": False,
            "resume": False,
        },
        "hard_gates": {
            "max_tapped_control_ratio": 2.0,
            "require_bitwise_identity": True,
            "require_tap_finite": True,
            "require_tap_nonconstant_within_prompt": True,
        },
    }


def _good_gate_audit() -> dict[str, object]:
    rng = np.random.default_rng(0)
    control = rng.standard_normal((3, len(FEATURE_NAMES))).astype(np.float16)
    taps = np.arange(3 * len(tap_feature_names()), dtype=np.float16).reshape(
        3, -1
    )
    tapped = np.concatenate([control, taps], axis=1)
    audit = audit_arrays(
        control,
        tapped,
        control.copy(),
        expected_gen_ids=[1, 2, 3],
        control_gen_ids=[1, 2, 3],
        tapped_gen_ids=[1, 2, 3],
        expected_prompt_ids=[8, 9],
        control_prompt_ids=[8, 9],
        tapped_prompt_ids=[8, 9],
        feature_names=[*FEATURE_NAMES, *tap_feature_names()],
    )
    audit.update(prefix_hashes_match=True, no_oom=True, cost_ok=True)
    return audit


def _report(*, oom: bool = False) -> dict[str, object]:
    names_hash = hashlib.sha256(
        json.dumps(
            [*FEATURE_NAMES, *tap_feature_names()], separators=(",", ":")
        ).encode()
    ).hexdigest()
    audits: dict[str, object] = {}
    for fold, _ in PILOT_ROSTER:
        gates = _good_gate_audit()
        gates["no_oom"] = not oom
        gates["cost_ok"] = not oom
        audits[f"fold_{fold}"] = {
            "fold": fold,
            "control_rows": 3,
            "tapped_rows": 3,
            "control_time_seconds": 1.0,
            "tapped_time_seconds": 1.5,
            "control_peak_allocated_bytes": 10,
            "tapped_peak_allocated_bytes": 20,
            **gates,
            "legacy_prompt_ids_sha256": _HASH,
            "legacy_gen_ids_sha256": _HASH,
            "control_prompt_ids_sha256": _HASH,
            "control_gen_ids_sha256": _HASH,
            "tapped_prompt_ids_sha256": _HASH,
            "tapped_gen_ids_sha256": _HASH,
            "control_array_sha256": _HASH,
            "tapped_array_sha256": _HASH,
        }
    return {
        "schema_version": "herald.attention_tap_pilot_report.v1",
        "decision": (
            "retire_attention_tap"
            if oom
            else "license_full_label_blind_replay"
        ),
        "pilot_count": len(PILOT_ROSTER),
        "pilot_prompt_ids_sha256": hash_prompt_ids(
            [prompt_id for _, prompt_id in PILOT_ROSTER]
        ),
        "feature_names_sha256": names_hash,
        "tap_layers": list(TAP_LAYER_INDICES),
        "tap_feature_count": len(tap_feature_names()),
        "control_time_seconds": 5.0,
        "tapped_time_seconds": 7.5,
        "tapped_control_time_ratio": None if oom else 1.5,
        "aggregate_cost_ok": not oom,
        "prompt_audits": audits,
        "no_oom": not oom,
        "input_hashes": {
            f"{name}_sha256": _HASH for name in _INPUT_HASH_NAMES
        },
        "protocol_lock_sha256": _HASH,
    }


def _sidecar_record(
    prompt_id: str, position: int, digest: str, **extra: object
) -> dict[str, object]:
    record: dict[str, object] = {
        "compressor": "knorm",
        "evidence_sha256": _HASH,
        "feature_names": ["sensor"],
        "lock_sha256": _HASH,
        "model_key": "llama",
        "prefix_hash": digest,
        "prompt_id": prompt_id,
        "protocol_version": "herald.m3.layer_band_replay.v1",
        "ratio": 0.5,
        "s": position,
        "sensor_lock_sha256": _HASH,
        "sensors": {"sensor": 1.0},
        "state_semantics": "herald.cache_native_pending_v1",
        "task": "ifeval",
    }
    record.update(extra)
    return record


def test_exact_loader_selects_columns_before_requested_rows() -> None:
    dataset = _Dataset(
        [
            {"key": 2, "prompt": "two", "secret": "hidden"},
            {"key": 1, "prompt": "one", "secret": "hidden"},
        ]
    )
    records = load_ifeval_exact(["ifeval-1", "ifeval-2"], dataset=dataset)
    assert dataset.audit == [
        ("columns", ["key", "prompt"]),
        ("select", [1, 0]),
        ("columns", ["key", "prompt"]),
    ]
    assert [record.prompt_id for record in records] == [
        "ifeval-1",
        "ifeval-2",
    ]
    assert [record.messages[0]["content"] for record in records] == [
        "one",
        "two",
    ]
    assert all(record.gold == {} for record in records)


@pytest.mark.parametrize(
    ("requested", "message"),
    [
        (["ifeval-1", "ifeval-1"], "duplicates"),
        (["ifeval-x"], "malformed"),
        (["ifeval-01"], "malformed"),
        (["ifeval-9"], "missing"),
    ],
)
def test_exact_loader_rejects_bad_ids(
    requested: list[str], message: str
) -> None:
    dataset = _Dataset([{"key": 1, "prompt": "one"}])
    with pytest.raises(ValueError, match=message):
        load_ifeval_exact(requested, dataset=dataset)


def test_exact_loader_rejects_nonintegral_source_keys() -> None:
    dataset = _FallbackDataset([{"key": "1", "prompt": "one"}])
    with pytest.raises(ValueError, match="integral"):
        load_ifeval_exact(["ifeval-1"], dataset=dataset)


def test_identity_reader_stops_before_secret_tail(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_bytes(
        b'{"prompt_id":"p","prompt_input_ids":[1,2],"gen_ids":[3,4],'
        b'"text": { deliberately invalid secret tail'
    )
    identity = read_legacy_identity_prefix(path, "p")
    assert identity.prompt_input_ids == (1, 2)
    assert identity.gen_ids == (3, 4)


def test_sparse_sidecar_boundaries_match_new_control_ids(
    tmp_path: Path,
) -> None:
    prompt_ids = (10,)
    gen_ids = (20, 30, 40)
    path = tmp_path / "prefix.jsonl"
    path.write_text(
        '{"prompt_id":"other","q":"malformed secret"\n'
        + json.dumps(
            _sidecar_record("p", 0, prefix_hash(prompt_ids, gen_ids, 0))
        )
        + "\n"
        + json.dumps(
            _sidecar_record("p", 2, prefix_hash(prompt_ids, gen_ids, 2))
        )
        + "\n"
    )
    sidecar = load_prefix_hash_sidecar(path, ["p"])
    control = PILOT.ArmRun(
        gen_ids,
        prompt_ids,
        np.empty((3, len(FEATURE_NAMES)), dtype=np.float16),
        None,
        1.0,
        0,
    )
    assert PILOT._prefix_hashes_match(sidecar, "p", control)
    assert set(sidecar) == {("p", 0), ("p", 2)}


def test_sidecar_rejects_selected_malformed_or_conflicting_lines(
    tmp_path: Path,
) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_text('{"prompt_id":"p","s":0\n')
    with pytest.raises(AttentionReplayError, match="malformed selected"):
        load_prefix_hash_sidecar(malformed, ["p"])

    forbidden = tmp_path / "forbidden.jsonl"
    forbidden.write_text(
        json.dumps(_sidecar_record("p", 0, "a" * 64, q="secret")) + "\n"
    )
    with pytest.raises(AttentionReplayError, match="forbidden field"):
        load_prefix_hash_sidecar(forbidden, ["p"])

    escaped = tmp_path / "escaped-forbidden.jsonl"
    escaped.write_text(
        json.dumps(_sidecar_record("p", 0, "a" * 64))[:-1]
        + ',"te\\u0078t":"secret"}\n'
    )
    with pytest.raises(AttentionReplayError, match="forbidden field"):
        load_prefix_hash_sidecar(escaped, ["p"])

    malformed_tail = tmp_path / "malformed-tail.jsonl"
    malformed_tail.write_text(
        json.dumps(_sidecar_record("p", 0, "a" * 64))[:-1] + ',"junk":}\n'
    )
    with pytest.raises(AttentionReplayError, match="malformed selected"):
        load_prefix_hash_sidecar(malformed_tail, ["p"])

    conflicting = tmp_path / "conflicting.jsonl"
    conflicting.write_text(
        json.dumps(_sidecar_record("p", 0, "a" * 64))
        + "\n"
        + json.dumps(_sidecar_record("p", 0, "b" * 64))
        + "\n"
    )
    with pytest.raises(AttentionReplayError, match="conflicting"):
        load_prefix_hash_sidecar(conflicting, ["p"])


def test_array_gate_distinguishes_control_and_tap_failures() -> None:
    good = _good_gate_audit()
    assert (
        classify_decision([good], aggregate_cost_ok=True)
        == "license_full_label_blind_replay"
    )

    tap_ids = dict(good, tap_gen_ids_match=False)
    assert (
        classify_decision([tap_ids], aggregate_cost_ok=True)
        == "retire_attention_tap"
    )
    feature_names = dict(good, shapes_names_match=False)
    assert (
        classify_decision([feature_names], aggregate_cost_ok=True)
        == "retire_attention_tap"
    )
    legacy = dict(good, control_legacy_match=False)
    assert (
        classify_decision([legacy], aggregate_cost_ok=True)
        == "inconclusive_environment_mismatch"
    )
    oom = dict(good, no_oom=False, cost_ok=False)
    assert (
        classify_decision([oom], aggregate_cost_ok=False)
        == "retire_attention_tap"
    )
    missing = dict(good)
    del missing["no_oom"]
    with pytest.raises(AttentionReplayError, match="missing fields"):
        classify_decision([missing], aggregate_cost_ok=True)


def test_array_audit_rejects_legacy_dtype_and_runtime_names() -> None:
    rng = np.random.default_rng(1)
    control = rng.standard_normal((3, len(FEATURE_NAMES))).astype(np.float16)
    taps = rng.standard_normal((3, len(tap_feature_names()))).astype(
        np.float16
    )
    tapped = np.concatenate([control, taps], axis=1)
    common: dict[str, Any] = {
        "expected_gen_ids": [1],
        "control_gen_ids": [1],
        "tapped_gen_ids": [1],
        "expected_prompt_ids": [2],
        "control_prompt_ids": [2],
        "tapped_prompt_ids": [2],
    }
    bad_dtype = audit_arrays(
        control,
        tapped,
        control.astype(np.float32),
        feature_names=[*FEATURE_NAMES, *tap_feature_names()],
        **common,
    )
    assert not bad_dtype["control_legacy_match"]

    bad_names = audit_arrays(
        control,
        tapped,
        control.copy(),
        feature_names=[*FEATURE_NAMES, *tap_feature_names()][::-1],
        **common,
    )
    assert bad_names["control_legacy_match"]
    assert not bad_names["shapes_names_match"]
    assert not bad_names["tapped_control_match"]


def test_exact_protocol_config_paths_and_hashes(tmp_path: Path) -> None:
    args = _pilot_args(tmp_path)
    lock = _protocol_lock(args)
    validate_protocol_lock(lock)
    PILOT._validate_generation_config(lock, args)
    PILOT._validate_bound_paths(lock, args)

    for path in (
        args.source_oof,
        args.legacy_config,
        args.legacy_manifest,
        args.legacy_prefix_sidecar,
        args.legacy_prefix_manifest,
    ):
        path.write_text(path.name)
    paths = {
        "source_oof": args.source_oof,
        "legacy_config": args.legacy_config,
        "legacy_manifest": args.legacy_manifest,
        "prefix_sidecar": args.legacy_prefix_sidecar,
        "prefix_manifest": args.legacy_prefix_manifest,
        "attention_source": args.attention_source,
        "generate_source": args.generate_source,
        "ifeval_source": args.ifeval_source,
        "attention_replay_source": _module_path(
            PILOT.attention_replay_module
        ),
        "features_source": _module_path(PILOT.features_module),
        "script": _module_path(PILOT),
    }
    lock["input_hashes"] = {
        f"{name}_sha256": sha256_file(path) for name, path in paths.items()
    }
    assert PILOT._validate_input_hashes(lock, args) == {
        name: sha256_file(path) for name, path in paths.items()
    }

    bad = _protocol_lock(args)
    generation = bad["generation"]
    assert isinstance(generation, dict)
    bad["generation"] = {
        **generation,
        "tap_layer_indices": [8, 16],
    }
    with pytest.raises(AttentionReplayError, match="generation config"):
        PILOT._validate_generation_config(bad, args)


def test_parser_exposes_every_bound_runtime_path(tmp_path: Path) -> None:
    values = [
        "--protocol-lock",
        str(tmp_path / "lock"),
        "--legacy-reference-dir",
        str(tmp_path / "refs"),
        "--legacy-manifest",
        str(tmp_path / "manifest"),
        "--legacy-prefix-sidecar",
        str(tmp_path / "sidecar"),
        "--legacy-prefix-manifest",
        str(tmp_path / "sidecar-manifest"),
        "--source-oof",
        str(tmp_path / "oof"),
        "--legacy-config",
        str(tmp_path / "config"),
        "--output-root",
        str(tmp_path / "output"),
    ]
    args = PILOT._parser().parse_args(values)
    for name in (
        "legacy_prefix_manifest",
        "source_oof",
        "legacy_config",
        "output_root",
        "model_id",
        "device",
    ):
        assert getattr(args, name) is not None


def test_selected_legacy_manifest_verifies_size_and_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "references"
    root.mkdir()
    identity = root / "ifeval-168.json"
    identity.write_text(
        '{"prompt_id":"ifeval-168","prompt_input_ids":[1],"gen_ids":[2]}'
    )
    array = root / "ifeval-168.npy"
    np.save(array, np.zeros((1, len(FEATURE_NAMES)), dtype=np.float16))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "artifact_files": [
                    {
                        "path": identity.name,
                        "sha256": sha256_file(identity),
                        "size": identity.stat().st_size,
                    },
                    {
                        "path": array.name,
                        "sha256": sha256_file(array),
                        "size": array.stat().st_size,
                    },
                ]
            }
        )
    )
    assert PILOT._validate_legacy_manifest(
        manifest, root, ("ifeval-168",)
    ) == {"ifeval-168": (identity.resolve(), array.resolve())}
    array.write_bytes(b"tampered")
    with pytest.raises(AttentionReplayError, match="size mismatch"):
        PILOT._validate_legacy_manifest(manifest, root, ("ifeval-168",))


def test_report_is_exact_and_oom_ratio_remains_json_finite(
    tmp_path: Path,
) -> None:
    report = _report()
    assert validate_report(report) == report
    output = tmp_path / "report.json"
    write_report(output, report)
    assert json.loads(output.read_text()) == report

    invalid_oom = _report(oom=True)
    invalid_oom["tapped_control_time_ratio"] = 0.0
    with pytest.raises(AttentionReplayError, match="ratio must be null"):
        validate_report(invalid_oom)
    forged_ratio = _report()
    forged_ratio["tapped_control_time_ratio"] = 1.0
    with pytest.raises(AttentionReplayError, match="ratio is inconsistent"):
        validate_report(forged_ratio)
    forged_times = _report()
    forged_times["control_time_seconds"] = 6.0
    forged_times["tapped_control_time_ratio"] = 1.25
    with pytest.raises(AttentionReplayError, match="aggregate timings"):
        validate_report(forged_times)
    oom_report = _report(oom=True)
    assert validate_report(oom_report)["tapped_control_time_ratio"] is None
    write_report(tmp_path / "oom.json", oom_report)

    forbidden = dict(report, text="secret")
    with pytest.raises(AttentionReplayError, match="report keys"):
        write_report(tmp_path / "forbidden.json", forbidden)
    assert not (tmp_path / "forbidden.json").exists()
    missing = dict(report)
    del missing["protocol_lock_sha256"]
    with pytest.raises(AttentionReplayError, match="report keys"):
        validate_report(missing)


def test_generate_reference_can_skip_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def decode(self, *_args: object, **_kwargs: object) -> str:
            raise AssertionError("decode must not be called")

    class Collector:
        def stacked(self) -> np.ndarray:
            return np.zeros((2, 1, len(FEATURE_NAMES)), dtype=np.float32)

        def argmax_tokens(self) -> np.ndarray:
            return np.asarray([[3], [4]], dtype=np.int64)

    monkeypatch.setattr(
        generate_module,
        "build_input_ids",
        lambda _lm, _record: torch.tensor([7, 8]),
    )
    monkeypatch.setattr(generate_module, "FeatureCollector", Collector)
    monkeypatch.setattr(
        generate_module,
        "_generate",
        lambda *_args, **_kwargs: ([[3, 4]], 2),
    )
    record = PromptRecord(
        task="ifeval",
        prompt_id="ifeval-1",
        messages=[{"role": "user", "content": "x"}],
        gold={},
    )
    lm = type("LM", (), {"tokenizer": Tokenizer()})()
    [result] = generate_module.generate_reference(
        lm, [record], 2, decode_text=False
    )
    assert result.text == ""


def test_pilot_arm_quantizes_live_float32_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_names = [*FEATURE_NAMES, *tap_feature_names()]

    class Tap:
        def __init__(self, _model: object, _layers: list[int]) -> None:
            self.removed = False

        def remove(self) -> None:
            self.removed = True

    def fake_generate(
        _lm: object,
        _records: list[PromptRecord],
        _max_new_tokens: int,
        *,
        tap: object,
        decode_text: bool,
    ) -> list[SimpleNamespace]:
        assert tap is not None
        assert not decode_text
        values = np.arange(2 * len(expected_names), dtype=np.float32).reshape(
            2, -1
        )
        return [
            SimpleNamespace(
                features=values,
                feature_names=expected_names,
                gen_ids=[3, 4],
                prompt_input_ids=[1, 2],
            )
        ]

    monkeypatch.setattr(PILOT, "AttentionTap", Tap)
    monkeypatch.setattr(PILOT, "generate_reference", fake_generate)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    record = PromptRecord(
        task="ifeval",
        prompt_id="ifeval-1",
        messages=[{"role": "user", "content": "x"}],
        gold={},
    )
    lm = SimpleNamespace(model=object())
    run = PILOT._generate_arm(lm, record, tapped=True)
    assert run.values.dtype == np.float16
    assert run.values.shape == (2, len(expected_names))
    assert run.feature_names == tuple(expected_names)


def test_oom_never_reports_a_timing_speedup() -> None:
    times = {"control": 2.0, "tapped": 0.0}
    assert PILOT._tapped_control_ratio(times, oom=True) is None
    assert PILOT._tapped_control_ratio(times, oom=False) == 0.0
    assert (
        PILOT._tapped_control_ratio(
            {"control": 0.0, "tapped": 1.0}, oom=False
        )
        is None
    )


def test_warmup_oom_is_explicit_retirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        PILOT,
        "generate_reference",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("CUDA out of memory")
        ),
    )
    lm = SimpleNamespace(model=object())
    with pytest.raises(
        AttentionReplayError, match="control warmup CUDA OOM retires"
    ):
        PILOT._warmup(lm, tapped=False)
