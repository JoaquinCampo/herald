"""End-to-end Phase 0 smoke (Orion-only).

`JS_NOISE_FLOOR` is the per-token JS divergence we are willing to
attribute to fp16 / CUDA non-determinism on the no-press baseline.
Calibrated 2026-05-02 from the Phase 0 sweep on Qwen2.5-7B-Instruct
(20 baseline runs, RTX 5090, fp16): cell_js_max=0.092,
cell_js_median=0.079. We pick 1e-1 as a clean ceiling above the
observed maximum; see gold/phase-0-results.md.
"""

from pathlib import Path

import polars as pl
import pytest

JS_NOISE_FLOOR = 1e-1


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
