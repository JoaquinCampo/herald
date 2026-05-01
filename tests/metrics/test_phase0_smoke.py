"""End-to-end Phase 0 smoke (Orion-only).

Pin `JS_NOISE_FLOOR` from `scripts/measure_replay_noise_floor.py` on
Orion before treating gates 3/4 as locked. The default below is a
plausibility floor for MPS/CUDA fp16; the real Orion gate may be
tighter.
"""

from pathlib import Path

import polars as pl
import pytest

JS_NOISE_FLOOR = 1e-3


@pytest.mark.gpu
@pytest.mark.e2e
def test_phase0_smoke_2prompts_1press_1ratio(tmp_path: Path) -> None:
    """Cheap end-to-end smoke (4 runs total). Runs on Orion."""
    from herald.metrics.io import finalize_dataset
    from herald.phase0 import run_phase0_sweep

    out = tmp_path / "phase0"
    run_phase0_sweep(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        manifest_path=Path("gold/phase0-random-manifest.json"),
        presses=("streaming_llm",),
        ratios=(0.5,),
        max_new_tokens=16,
        output_root=out,
        prompt_timeout_seconds=120.0,
        num_prompts=2,
    )
    finalize_dataset(out)

    # Gate 1: completion
    runs = pl.read_parquet(out / "final" / "runs.parquet")
    assert (runs["replay_status"] == "ok").all()

    # Gate 2: schema integrity
    assert runs["baseline_run_id"].is_not_null().all()
    for rid in runs["run_id"]:
        assert (out / "raw" / "tokens" / f"{rid}.parquet").exists()
        assert (out / "raw" / "replay" / f"{rid}.parquet").exists()

    # Gate 4: press=none replay JS ~ 0 along trajectory
    base_ids = runs.filter(pl.col("press") == "none")["run_id"]
    for rid in base_ids:
        rep = pl.read_parquet(out / "raw" / "replay" / f"{rid}.parquet")
        assert rep["js_full"].max() < JS_NOISE_FLOOR
