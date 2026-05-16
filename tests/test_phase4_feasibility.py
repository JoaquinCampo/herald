"""CPU-only tests for the Phase 4 decode-time press feasibility gate."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from herald.phase4_feasibility import (
    DEFAULT_COMPRESSION_INTERVAL,
    EVENTS_COLUMNS,
    MIN_KVPRESS_VERSION,
    RUNS_COLUMNS,
    SEGMENT_K,
    SEGMENTS_COLUMNS,
    CompressionEvent,
    FeasibilityConfig,
    PlannedRun,
    TokenLog,
    aggregate_segments,
    classify_decode_events,
    instrument_press,
    plan_runs,
    render_dry_run,
    require_kvpress_version,
    summarize_events,
    write_parquet,
    write_report,
)
from scripts.run_phase4_decoding_press_feasibility import _parse_args


def _cfg(**overrides) -> FeasibilityConfig:
    base = dict(
        press="decoding_knorm",
        task="gsm8k",
        target_sizes=(256, 512),
        num_prompts=2,
    )
    base.update(overrides)
    return FeasibilityConfig(**base)


# --- config / planning -----------------------------------------------


def test_config_rejects_unknown_press() -> None:
    with pytest.raises(ValueError, match="Unsupported press"):
        FeasibilityConfig(press="bogus")


def test_config_rejects_non_gsm8k_task() -> None:
    with pytest.raises(ValueError, match="task=gsm8k"):
        FeasibilityConfig(press="decoding_knorm", task="humaneval")


def test_config_rejects_zero_prompts() -> None:
    with pytest.raises(ValueError, match="num_prompts"):
        FeasibilityConfig(press="decoding_knorm", num_prompts=0)


def test_config_rejects_empty_target_sizes() -> None:
    with pytest.raises(ValueError, match="target_sizes"):
        FeasibilityConfig(press="decoding_knorm", target_sizes=())


def test_plan_runs_one_per_target_size() -> None:
    cfg = _cfg(target_sizes=(128, 256, 512))
    runs = plan_runs(cfg)
    assert len(runs) == 3
    assert {r.target_size for r in runs} == {128, 256, 512}
    for r in runs:
        assert isinstance(r, PlannedRun)
        assert r.compression_interval == DEFAULT_COMPRESSION_INTERVAL
        assert r.press == cfg.press


def test_render_dry_run_lists_planned_runs() -> None:
    cfg = _cfg(target_sizes=(256, 1024))
    text = render_dry_run(cfg)
    assert "decoding_knorm" in text
    assert "target_size=256" in text
    assert "target_size=1024" in text
    assert "No model load" in text


# --- CLI / dry-run ---------------------------------------------------


def test_cli_parses_target_sizes() -> None:
    ns = _parse_args(
        [
            "--press",
            "decoding_knorm",
            "--target-sizes",
            "256",
            "512",
            "--dry-run",
        ]
    )
    assert ns.target_sizes == [256, 512]
    assert ns.dry_run is True


def test_cli_dry_run_main_exits_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from scripts.run_phase4_decoding_press_feasibility import main

    rc = main(
        [
            "--press",
            "decoding_knorm",
            "--target-sizes",
            "256",
            "512",
            "--num-prompts",
            "2",
            "--max-new-tokens",
            "32",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "Phase 4" in captured
    assert "target_size=256" in captured


# --- kvpress version gate --------------------------------------------


def test_require_kvpress_version_accepts_min() -> None:
    fake = SimpleNamespace(__version__="0.5.3")
    assert require_kvpress_version(_module=fake) == "0.5.3"


def test_require_kvpress_version_accepts_newer() -> None:
    fake = SimpleNamespace(__version__="0.6.0")
    assert require_kvpress_version(_module=fake) == "0.6.0"


def test_require_kvpress_version_refuses_older() -> None:
    fake = SimpleNamespace(__version__="0.5.1")
    with pytest.raises(RuntimeError, match="0.5.3"):
        require_kvpress_version(_module=fake)


def test_require_kvpress_version_falls_back_to_metadata() -> None:
    # No __version__ on the module: resolver lookup is used instead.
    fake = SimpleNamespace()
    resolver = lambda _name: "0.5.4"  # noqa: E731
    assert (
        require_kvpress_version(_module=fake, _resolver=resolver) == "0.5.4"
    )


def test_require_kvpress_version_refuses_when_unresolvable() -> None:
    fake = SimpleNamespace()

    def resolver(_name: str) -> str:
        raise LookupError("missing")

    with pytest.raises(RuntimeError, match="unresolvable"):
        require_kvpress_version(_module=fake, _resolver=resolver)


def test_min_kvpress_version_constant() -> None:
    assert MIN_KVPRESS_VERSION == (0, 5, 3)


# --- segment / event aggregation -------------------------------------


def _tok(i: int, e: float = 1.0) -> TokenLog:
    return TokenLog(
        step_idx=i,
        entropy=e,
        top1_prob=0.5,
        eff_vocab_size=2.0,
        top1_top2_ratio=1.5,
        tail_mass=0.1,
        wall_clock_seconds=0.001,
    )


def _evt(step: int, layer: int = 0, kind: str = "decode") -> CompressionEvent:
    return CompressionEvent(
        step_idx=step,
        layer_idx=layer,
        event_type=kind,
        retained_cache_len_before=100,
        retained_cache_len_after=64,
        target_size=64,
        threshold=None,
        wall_clock_seconds=0.0001,
    )


def test_aggregate_segments_sized_by_k() -> None:
    tokens = [_tok(i) for i in range(35)]
    events = [_evt(0), _evt(20), _evt(33)]
    rows = aggregate_segments(
        "rid", tokens, events, target_size=128, threshold=None, k=SEGMENT_K
    )
    # 35 tokens, K=16 -> 3 segments (16, 16, 3).
    assert len(rows) == 3
    assert rows[0]["segment_start"] == 0
    assert rows[0]["segment_end"] == 16
    assert rows[1]["segment_start"] == 16
    assert rows[1]["segment_end"] == 32
    assert rows[2]["segment_end"] == 35
    # Events binned by step_idx.
    assert rows[0]["compression_events_in_segment"] == 1
    assert rows[1]["compression_events_in_segment"] == 1
    assert rows[2]["compression_events_in_segment"] == 1
    # Schema columns present.
    for r in rows:
        assert set(SEGMENTS_COLUMNS).issubset(r.keys())


def test_aggregate_segments_empty_returns_empty() -> None:
    assert aggregate_segments("rid", [], [], 64, None) == []


def test_aggregate_segments_rejects_zero_k() -> None:
    with pytest.raises(ValueError, match="k must be > 0"):
        aggregate_segments("rid", [_tok(0)], [], 64, None, k=0)


def test_summarize_events_classifies_decode() -> None:
    events = [
        _evt(0, 0, "prefill"),
        _evt(1, 0, "decode"),
        _evt(2, 1, "decode"),
    ]
    s = summarize_events(events)
    assert s["decode_events"] == 2
    assert s["prefill_events"] == 1
    assert s["decode_compression_observed"] is True
    assert sorted(s["decode_layers_touched"]) == [0, 1]


def test_summarize_events_no_decode() -> None:
    events = [_evt(0, 0, "prefill")]
    s = summarize_events(events)
    assert s["decode_compression_observed"] is False


# --- press instrumentation ------------------------------------------


class _FakeKeys:
    """Stand-in for a key tensor; only `.shape[-2]` is used."""

    def __init__(self, seq_len: int) -> None:
        self.shape = (1, 4, seq_len, 64)


class _FakeModule:
    """Stand-in for a layer attention module; only layer_idx used."""

    def __init__(self, layer_idx: int) -> None:
        self.layer_idx = layer_idx


class _FakeBasePress:
    """Mimics kvpress's BasePress.compress signature.

    The real kvpress hook calls `self.compress(module, ...)` via
    bound-method lookup at fire time. We mirror that path here.
    """

    def __init__(self, target_after: int) -> None:
        self.target_after = target_after
        self.compress_calls = 0

    def compress(
        self,
        module,  # noqa: ANN001
        hidden_states,  # noqa: ANN001
        keys,  # noqa: ANN001
        values,  # noqa: ANN001
        attentions,  # noqa: ANN001
        kwargs,  # noqa: ANN001
    ):  # noqa: ANN201
        self.compress_calls += 1
        return (_FakeKeys(self.target_after), values)


def _fire_hook(
    press: _FakeBasePress, layer_idx: int, before_len: int
) -> None:
    """Simulate what a kvpress attention forward hook does.

    The hook resolves `press.compress` at fire time, so a patched
    attribute on the press instance is what gets invoked. This is
    the exact path `instrument_press` relies on.
    """
    press.compress(
        _FakeModule(layer_idx),
        hidden_states=None,
        keys=_FakeKeys(before_len),
        values=None,
        attentions=None,
        kwargs={},
    )


def test_instrument_press_records_events_via_attribute_lookup() -> None:
    press = _FakeBasePress(target_after=64)
    recorder = instrument_press(press, target_size=64, threshold=None)

    # Three "decode" hook fires across two layers.
    _fire_hook(press, layer_idx=0, before_len=128)
    _fire_hook(press, layer_idx=1, before_len=128)
    _fire_hook(press, layer_idx=0, before_len=96)

    assert recorder.fire_count == 3
    assert press.compress_calls == 3  # original was forwarded each time
    assert len(recorder.events) == 3

    ev0, ev1, ev2 = recorder.events
    assert ev0.layer_idx == 0
    assert ev0.retained_cache_len_before == 128
    assert ev0.retained_cache_len_after == 64
    assert ev0.target_size == 64
    assert ev1.layer_idx == 1
    assert ev2.retained_cache_len_before == 96


def test_instrument_press_classify_decode_events_in_place() -> None:
    press = _FakeBasePress(target_after=64)
    recorder = instrument_press(press, target_size=64, threshold=None)

    _fire_hook(press, layer_idx=0, before_len=128)  # prefill
    _fire_hook(press, layer_idx=1, before_len=128)  # prefill
    _fire_hook(press, layer_idx=0, before_len=96)  # decode
    _fire_hook(press, layer_idx=1, before_len=96)  # decode

    classify_decode_events(recorder.events)
    types = [e.event_type for e in recorder.events]
    assert types == ["prefill", "prefill", "decode", "decode"]
    assert summarize_events(recorder.events)["decode_compression_observed"]


# --- artifact writers ------------------------------------------------


def test_write_report_writes_json(tmp_path: Path) -> None:
    payload = {"press": "decoding_knorm", "pass_gate": True}
    out = write_report(tmp_path, payload)
    assert out.name == "decoding_press_report.json"
    parsed = json.loads(out.read_text())
    assert parsed["press"] == "decoding_knorm"


def test_write_parquet_round_trip(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        {
            "run_id": "r1",
            "prompt_id": "p1",
            "task": "gsm8k",
            "press": "decoding_knorm",
            "target_size": 256,
            "compression_interval": 16,
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "dtype": "float16",
            "seed": 42,
            "input_len": 64,
            "num_tokens_generated": 32,
            "stop_reason": "max_tokens",
            "wall_clock_seconds": 1.5,
            "wall_clock_per_token": 0.05,
            "peak_memory_mb": 1024.0,
            "compression_event_count": 4,
            "hook_fire_count_total": 32,
            "decode_compression_observed": True,
            "task_score": 1.0,
            "correct": True,
            "predicted_answer": "42",
            "ground_truth": "42",
            "generated_text": "the answer is 42",
            "kvpress_version": "0.5.3",
            "created_at": "2026-05-05T00:00:00Z",
        }
    ]
    out = write_parquet(tmp_path, "runs", rows, RUNS_COLUMNS)
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 1
    assert set(table.column_names) == set(RUNS_COLUMNS)


def test_write_parquet_empty_writes_empty_table(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    out = write_parquet(tmp_path, "events", [], EVENTS_COLUMNS)
    assert out.exists()
    table = pq.read_table(out)
    assert table.num_rows == 0
    assert set(table.column_names) == set(EVENTS_COLUMNS)
