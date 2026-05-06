"""Block 1 equivalence smoke (Orion-only).

Regenerates one Phase 0 baseline run via the new `FixedRatioPolicy`
code path and asserts the new RunResult matches the saved Phase 0
RunResult byte-for-byte on the load-bearing fields.

Why this exists: `poe check` exercises the new code path against
mocked models, but the only proof the refactor preserves Phase 0
numbers comes from regenerating against the saved gold artifacts.
See `gold/phase-0-results.md`.

Run:
    .venv/bin/python scripts/phase1_equivalence_check.py
"""

import sys
from pathlib import Path

import polars as pl


def main() -> int:
    from herald.config import ExperimentConfig
    from herald.experiment import load_model, run_single_with_replay
    from herald.metrics.io import PerRunPaths
    from herald.tasks import DEFAULT_TASK

    phase0_root = Path("results/phase0")
    runs_dir = phase0_root / "raw" / "runs"
    if not runs_dir.exists():
        print(f"FAIL: missing {runs_dir}")
        return 1

    # Pick a baseline (press=none) run to regenerate. Baselines are
    # the cleanest equivalence target — no compression nondeterminism.
    candidates = sorted(runs_dir.glob("*.parquet"))
    baseline_path = None
    for p in candidates:
        df = pl.read_parquet(p)
        if df["press"][0] == "none":
            baseline_path = p
            break
    if baseline_path is None:
        print("FAIL: no baseline run found in Phase 0 results")
        return 1

    gold = pl.read_parquet(baseline_path).to_dicts()[0]
    prompt_id = gold["prompt_id"]
    print(f"equivalence target: run_id={gold['run_id']}  prompt={prompt_id}")
    print(f"  gold: tokens={gold['num_tokens_generated']}  "
          f"stop={gold['stop_reason']}")

    cfg = ExperimentConfig(
        model_name=gold["model"],
        press_name="none",
        compression_ratio=0.0,
        num_prompts=200,  # only used for output paths
        seed=int(gold["seed"]),
        output_dir=Path("results/phase1_equiv"),
        max_new_tokens=int(gold["max_new_tokens"]),
        prompt_timeout_seconds=300.0,
    )
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Re-derive the original prompt by id. Phase 0 used the same
    # GSM8K loader and the same seed, so the same id → same question.
    prompts = DEFAULT_TASK.load(num_prompts=200, seed=int(gold["seed"]))
    target = next((p for p in prompts if p["id"] == prompt_id), None)
    if target is None:
        print(f"FAIL: prompt_id {prompt_id} not in current GSM8K loader")
        return 1

    model, tok, device = load_model(cfg)
    rr, rm = run_single_with_replay(
        model=model,
        tokenizer=tok,
        device=device,
        prompt_data=target,
        config=cfg,
        baseline_run_id=None,
        output_root=cfg.output_dir,
    )
    paths = PerRunPaths(root=cfg.output_dir, run_id=rr.run_id)
    new_replay = pl.read_parquet(paths.replay)
    new_tokens = pl.read_parquet(paths.tokens)
    js_max_new = float(new_replay["js_full"].max() or 0.0)

    # Equivalence checks (load-bearing).
    failures: list[str] = []

    if rr.run_id != gold["run_id"]:
        failures.append(
            f"run_id drift: new={rr.run_id} gold={gold['run_id']}"
        )
    if rr.generated_text != gold["generated_text"]:
        failures.append("generated_text differs")
    if list(rr.generated_token_ids) != list(gold["generated_token_ids"]):
        failures.append("generated_token_ids differ")
    if rr.num_tokens_generated != gold["num_tokens_generated"]:
        gold_n = gold["num_tokens_generated"]
        failures.append(
            f"num_tokens_generated drift: "
            f"new={rr.num_tokens_generated} gold={gold_n}"
        )
    if rr.stop_reason != gold["stop_reason"]:
        gold_sr = gold["stop_reason"]
        failures.append(
            f"stop_reason drift: new={rr.stop_reason} gold={gold_sr}"
        )
    if rr.replay_status != "ok":
        failures.append(f"replay_status={rr.replay_status}")

    # New cost-telemetry fields populate.
    if not (rr.wall_clock_per_token > 0):
        failures.append(
            f"wall_clock_per_token not positive: {rr.wall_clock_per_token}"
        )
    if not (rr.peak_memory_mb > 0):
        failures.append(
            f"peak_memory_mb not positive: {rr.peak_memory_mb}"
        )
    if rr.policy_name != "FixedRatioPolicy":
        failures.append(f"policy_name={rr.policy_name}")

    # Phase 0 baseline noise-floor gate.
    if js_max_new >= 0.1:
        failures.append(
            f"baseline replay JS_max={js_max_new:.4f} >= 0.1 noise floor"
        )

    print()
    print(f"  new : tokens={rr.num_tokens_generated}  "
          f"stop={rr.stop_reason}  wpt={rr.wall_clock_per_token:.4f}s  "
          f"peak={rr.peak_memory_mb:.0f}MiB  js_max={js_max_new:.4f}")
    print(f"  replay rows = {new_tokens.height}")

    if failures:
        print()
        print("EQUIVALENCE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 2
    print()
    print("EQUIVALENCE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
