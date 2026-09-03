"""Focused model-free contracts for the faithful intervention sweep."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from herald.features import FEATURE_NAMES
from herald.intervention_sweep import (
    CACHE_NATIVE_SEMANTICS,
    EXPECTED_ATTENTION_SEMANTICS,
    GpuContentionError,
    GpuMonitoringError,
    InterventionSweepConfig,
    _existing_cells,
    active_gpu_compute_pids,
    gpu_contention_event,
    initialize_intervention_config,
    run_intervention_sweep,
)
from herald.live_controller import split_switch_boundary
from herald.storage import append_hybrid, reference_done, save_reference
from herald.switch_dataset import build_switch_rows
from herald.tasks import PromptRecord


def test_pending_boundary_is_zero_based_and_requires_token() -> None:
    assert split_switch_boundary([10, 11, 12], 0) == ([], 10)
    assert split_switch_boundary([10, 11, 12], 2) == ([10, 11], 12)
    with pytest.raises(ValueError, match="pending token"):
        split_switch_boundary([10], 1)


def test_gpu_contention_excludes_self_and_allowed_keepalive() -> None:
    assert (
        gpu_contention_event([2106, 99], allowed_pids=[2106], own_pid=99)
        is None
    )
    event = gpu_contention_event([2106, 100], allowed_pids=[2106], own_pid=99)
    assert event is not None
    assert event.unknown_pids == (100,)


def test_config_stride_one_is_valid() -> None:
    config = InterventionSweepConfig(stride=1)
    assert config.stride == 1


def test_config_rejects_out_of_scope_compressor_or_ratio() -> None:
    with pytest.raises(ValueError, match="compressors"):
        InterventionSweepConfig(compressors=("snapkv",))
    with pytest.raises(ValueError, match="approved scope"):
        InterventionSweepConfig(ratios=(0.1,))
    with pytest.raises(ValueError, match="at least one ratio"):
        InterventionSweepConfig(ratios=())


def test_intervention_config_binds_model_override(tmp_path: Path) -> None:
    original = InterventionSweepConfig(
        results_dir=tmp_path,
        model_id="model-a",
    )
    initialize_intervention_config(tmp_path, original)
    changed = InterventionSweepConfig(
        results_dir=tmp_path,
        model_id="model-b",
    )
    with pytest.raises(ValueError, match="does not match"):
        initialize_intervention_config(tmp_path, changed)


def test_gpu_monitoring_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "herald.intervention_sweep.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr="driver error",
        ),
    )
    with pytest.raises(GpuMonitoringError, match="driver error"):
        active_gpu_compute_pids()


def test_torn_final_hybrid_row_is_removed(tmp_path: Path) -> None:
    task_dir = tmp_path / "llama" / "ifeval"
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "streaming_llm",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[1],
        text="ok",
        q=1.0,
        dq=0.0,
    )
    shard = task_dir / "hybrids" / "streaming_llm__0.5000.jsonl"
    with shard.open("ab") as stream:
        stream.write(b'{"prompt_id":"torn"')

    cells = _existing_cells(task_dir, "streaming_llm", 0.5)

    assert set(cells) == {("p0", 0)}
    assert shard.read_bytes().endswith(b"\n")


def test_switch_rows_emit_matched_loose_and_strict_fields(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "ifeval",
        prompt_id="p0",
        prompt_input_ids=[1],
        gen_ids=[2, 3],
        text="ref",
        q=0.75,
        q_strict=0.5,
        features=np.ones((2, len(FEATURE_NAMES)), dtype=np.float32),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "streaming_llm",
        0.5,
        prompt_id="p0",
        s=0,
        new_ids=[2],
        text="hybrid",
        q=0.25,
        dq=0.5,
        q_control=0.75,
        q_control_strict=0.5,
        q_hybrid_strict=0.0,
        dq_strict=0.5,
        intervention_semantics=CACHE_NATIVE_SEMANTICS,
        generation_provenance={
            "control": "live_uncompressed_cache",
            "prefix_length": 0,
            "pending_token": 2,
        },
    )
    rows, _ = build_switch_rows(
        tmp_path / "llama" / "ifeval", model="llama", task="ifeval"
    )
    row = rows[0]
    assert row["q_control"] == 0.75
    assert row["q_control_strict"] == 0.5
    assert row["q_hybrid_strict"] == 0.0
    assert row["dq_strict"] == 0.5
    assert row["intervention_semantics"] == CACHE_NATIVE_SEMANTICS


def test_switch_rows_reject_strict_arithmetic_mismatch(
    tmp_path: Path,
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "ifeval",
        prompt_id="p0",
        prompt_input_ids=[],
        gen_ids=[2],
        text="ref",
        q=1.0,
        q_strict=1.0,
        features=np.ones((1, len(FEATURE_NAMES)), dtype=np.float32),
    )
    append_hybrid(
        tmp_path,
        "llama",
        "ifeval",
        "expected_attention",
        0.25,
        prompt_id="p0",
        s=0,
        new_ids=[3],
        text="hybrid",
        q=0.0,
        dq=1.0,
        q_control=1.0,
        q_control_strict=1.0,
        q_hybrid_strict=0.0,
        dq_strict=0.25,
        intervention_semantics=EXPECTED_ATTENTION_SEMANTICS,
    )
    with pytest.raises(ValueError, match="strict delta"):
        build_switch_rows(
            tmp_path / "llama" / "ifeval", model="llama", task="ifeval"
        )


def test_contention_stops_before_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("reference should not launch under contention")

    monkeypatch.setattr(
        "herald.intervention_sweep._reference_incremental", forbidden
    )
    config = InterventionSweepConfig(results_dir=tmp_path, prompt_count=1)
    record = PromptRecord(
        task="ifeval", prompt_id="p0", messages=[], gold={"prompt": "x"}
    )
    with pytest.raises(GpuContentionError):
        run_intervention_sweep(
            object(), [record], config, gpu_pid_provider=lambda: [12345]
        )
    assert not called


def test_complete_prompt_resume_does_not_regenerate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    save_reference(
        tmp_path,
        "llama",
        "ifeval",
        prompt_id="p0",
        prompt_input_ids=[],
        gen_ids=[1],
        text="ref",
        q=1.0,
        q_strict=1.0,
        features=np.ones((1, len(FEATURE_NAMES)), dtype=np.float32),
    )
    for compressor in ("streaming_llm", "knorm", "expected_attention"):
        semantics = (
            EXPECTED_ATTENTION_SEMANTICS
            if compressor == "expected_attention"
            else CACHE_NATIVE_SEMANTICS
        )
        for ratio in (0.25, 0.5, 0.75, 0.875):
            append_hybrid(
                tmp_path,
                "llama",
                "ifeval",
                compressor,
                ratio,
                prompt_id="p0",
                s=0,
                new_ids=[1],
                text="hybrid",
                q=0.0,
                dq=1.0,
                q_control=1.0,
                q_control_strict=1.0,
                q_hybrid_strict=0.0,
                dq_strict=1.0,
                intervention_semantics=semantics,
            )
    (tmp_path / "llama" / "ifeval" / "references" / "_done.jsonl").unlink()

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("complete prompt must be skipped")

    monkeypatch.setattr(
        "herald.intervention_sweep._reference_incremental", forbidden
    )
    config = InterventionSweepConfig(
        results_dir=tmp_path,
        max_new_tokens=1,
        prompt_count=1,
    )
    result = run_intervention_sweep(
        object(),
        [
            PromptRecord(
                task="ifeval",
                prompt_id="p0",
                messages=[],
                gold={"prompt": "x"},
            )
        ],
        config,
        gpu_pid_provider=lambda: [],
    )
    assert result["skipped_cells"] == 12
    assert reference_done(tmp_path, "llama", "ifeval") == {"p0"}
    feature_path = tmp_path / "llama" / "ifeval" / "references" / "p0.npy"
    feature_path.unlink()
    with pytest.raises(
        AssertionError, match="complete prompt must be skipped"
    ):
        run_intervention_sweep(
            object(),
            [
                PromptRecord(
                    task="ifeval",
                    prompt_id="p0",
                    messages=[],
                    gold={"prompt": "x"},
                )
            ],
            config,
            gpu_pid_provider=lambda: [],
        )
