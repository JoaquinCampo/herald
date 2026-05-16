"""Phase 4 single-threshold Pareto pilot (Orion only).

Goal:
  Run a small Pareto pilot on GSM8K to test whether HERALD-guided
  dynamic compression improves the quality / compute tradeoff
  versus fixed compression and random matched-budget gating.
  This is a pilot, not the final publishable controller experiment.

Per prompt, this script runs six policies:

  1. no_compression          : FixedBudgetPolicy(budget=large)
                               (high-quality, low-compression anchor;
                                the largest target acts as
                                "no_compression" because the press
                                will not need to evict.)
  2. fixed_64                : FixedBudgetPolicy(budget=64)
  3. fixed_128               : FixedBudgetPolicy(budget=128)
  4. fixed_256               : FixedBudgetPolicy(budget=256)
  5. herald_threshold_0.0639 : RiskBudgetStepPolicy on {64,128,256}
                               at threshold 0.0639 (online p75 from
                               results/phase4/controller_calibration_smoke
                               /online_distribution.json).
  6. random_matched_budget   : RandomMatchedBudgetPolicy that replays
                               HERALD's per-prompt action multiset
                               with a seeded shuffle.

Outputs (under --output-dir):
  controller_runs.parquet      one row per (prompt, policy) run
  controller_segments.parquet  one row per (run, segment)
  controller_events.parquet    one row per compress-fire
  pilot_manifest.json          predictor sha, args, env, kvpress ver

The downstream `scripts/analyze_phase4_pareto_pilot.py` script (Mac
side) computes ROUGE-L drop, paired deltas, and writes
`pareto_summary.json`.

Run with:
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
      .venv/bin/python scripts/run_phase4_pareto_pilot.py \
        --num-prompts 50 --max-new-tokens 512 \
        --output-dir results/phase4/pareto_pilot

Smoke first:
    ... --num-prompts 2 \
        --output-dir results/phase4/pareto_pilot_smoke
"""

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from herald.phase4_controller import (
    DEFAULT_BUDGETS,
    DEFAULT_K,
    RUN_COLUMNS,
    SEGMENT_COLUMNS,
    ControllerRun,
    FixedBudgetPolicy,
    RandomMatchedBudgetPolicy,
    RiskBudgetStepPolicy,
    attach_cache_size_per_segment,
    run_controller_once,
)
from herald.phase4_feasibility import (
    EVENTS_COLUMNS,
    write_parquet,
)
from herald.phase4_predictor import ExportedPredictor

# Canonical p75 from
# results/phase4/controller_calibration_smoke/online_distribution.json
# (240 segments from fixed_64 calibration smoke, 2026-05-06).
DEFAULT_THRESHOLD = 0.0639

# Anchor budget for the "no_compression" policy. Chosen far above
# any plausible peak cache length on Qwen2.5-7B-Instruct with
# max_new_tokens<=512 and GSM8K chat prompts (~200 token prefix).
NO_COMPRESSION_BUDGET = 4096

# Extended schema: existing run columns + threshold (for HERALD/
# random) + predictor metadata + cache headroom summary.
RUN_COLUMNS_EXT: tuple[str, ...] = (
    "threshold",
    "predictor_path",
    "predictor_sha256",
    "mean_retained_cache_size",
    "max_retained_cache_size",
) + RUN_COLUMNS

# Per-segment rows include the prompt_id for easy joins to runs.
SEGMENT_COLUMNS_EXT: tuple[str, ...] = ("prompt_id",) + SEGMENT_COLUMNS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictor",
        type=Path,
        default=Path("models/phase4_lr_all_cheap.json"),
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--num-prompts", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGETS),
        help=(
            "Ordered HERALD budget grid. Also defines fixed_{b} baselines."
        ),
    )
    ap.add_argument(
        "--no-compression-budget",
        type=int,
        default=NO_COMPRESSION_BUDGET,
        help=(
            "Target cache size for the no_compression anchor. "
            "Set well above plausible peak cache length so the "
            "press does not evict. Verify in smoke."
        ),
    )
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--random-seed", type=int, default=7)
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/phase4/pareto_pilot"),
    )
    return ap.parse_args(argv)


def _build_press_fn(_model: Any) -> Any:
    """Returns a closure (target_size, interval) -> DecodingPress."""
    from kvpress import (  # noqa: PLC0415
        DecodingPress,
        KnormPress,
    )

    def build(target_size: int, interval: int) -> Any:
        return DecodingPress(
            base_press=KnormPress(),
            target_size=int(target_size),
            compression_interval=int(interval),
        )

    return build


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _t_tag(t: float) -> str:
    return f"{t:.4f}".rstrip("0").rstrip(".") or "0"


def _segment_rows(run: ControllerRun, k: int) -> list[dict[str, Any]]:
    """Project ControllerRun.segments to extended row dicts."""
    base = attach_cache_size_per_segment(run, k=k)
    out = []
    for r in base:
        out.append(
            {
                "prompt_id": run.prompt_id,
                "run_id": run.run_id,
                "policy": run.policy_name,
                **r,
            }
        )
    return out


def _retained_cache_summary(
    seg_rows: list[dict[str, Any]],
) -> tuple[float | None, int | None]:
    """Mean and max of cache_size_observed >= 0 across the run."""
    sizes = [
        int(r["cache_size_observed"])
        for r in seg_rows
        if int(r["cache_size_observed"]) >= 0
    ]
    if not sizes:
        return None, None
    return sum(sizes) / len(sizes), max(sizes)


def _run_row(
    run: ControllerRun,
    threshold: float,
    predictor_path: Path,
    predictor_sha: str,
    mean_retained: float | None,
    max_retained: int | None,
) -> dict[str, Any]:
    n_relax = sum(1 for s in run.segments if s.decision == "relax")
    n_tighten = sum(1 for s in run.segments if s.decision == "tighten")
    n_keep = sum(1 for s in run.segments if s.decision == "keep")
    return {
        "threshold": float(threshold),
        "predictor_path": str(predictor_path),
        "predictor_sha256": predictor_sha,
        "mean_retained_cache_size": mean_retained,
        "max_retained_cache_size": max_retained,
        "run_id": run.run_id,
        "prompt_id": run.prompt_id,
        "policy": run.policy_name,
        "initial_budget": run.initial_budget,
        "num_tokens_generated": run.num_tokens_generated,
        "stop_reason": run.stop_reason,
        "wall_clock_seconds": run.wall_clock_seconds,
        "wall_clock_per_token": run.wall_clock_per_token,
        "peak_memory_mb": run.peak_memory_mb,
        "compression_event_count": run.compression_event_count,
        "decode_event_count": run.decode_event_count,
        "total_evicted_tokens": run.total_evicted_tokens,
        "n_segments": len(run.segments),
        "n_relax": n_relax,
        "n_tighten": n_tighten,
        "n_keep": n_keep,
        "task_score": (
            float(run.correct) if run.correct is not None else None
        ),
        "correct": run.correct,
        "predicted_answer": run.predicted_answer,
        "ground_truth": run.ground_truth,
        "generated_text": run.generated_text,
    }


def _event_rows(run: ControllerRun) -> list[dict[str, Any]]:
    out = []
    for ev in run.events:
        out.append(
            {
                "run_id": run.run_id,
                "step_idx": ev.step_idx,
                "layer_idx": ev.layer_idx,
                "event_type": ev.event_type,
                "retained_cache_len_before": ev.retained_cache_len_before,
                "retained_cache_len_after": ev.retained_cache_len_after,
                "target_size": ev.target_size,
                "threshold": ev.threshold,
                "wall_clock_seconds": ev.wall_clock_seconds,
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    args = _parse_args(argv)
    if not args.predictor.exists():
        print(f"predictor not found: {args.predictor}", file=sys.stderr)
        return 2
    predictor_sha = _file_sha256(args.predictor)
    print(f"loading predictor {args.predictor} sha={predictor_sha[:12]}")
    predictor = ExportedPredictor.from_path(args.predictor)

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
    )

    from herald.phase4_feasibility import (  # noqa: PLC0415
        require_kvpress_version,
    )
    from herald.prompts import load_gsm8k  # noqa: PLC0415

    if not torch.cuda.is_available():
        print("pareto pilot requires CUDA", file=sys.stderr)
        return 2
    device = "cuda"

    kvpress_version = require_kvpress_version()
    print(f"kvpress version: {kvpress_version}")

    prompts = load_gsm8k(args.num_prompts, seed=args.seed)
    print(f"loaded {len(prompts)} prompts")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"loading model {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device)  # type: ignore[arg-type]
    model.eval()

    build_press = _build_press_fn(model)
    budgets = tuple(int(b) for b in sorted(args.budgets))
    threshold = float(args.threshold)
    no_comp_budget = int(args.no_compression_budget)
    print(
        f"budgets={list(budgets)} threshold={threshold} "
        f"no_compression_budget={no_comp_budget} "
        f"K={args.k} max_new_tokens={args.max_new_tokens}"
    )

    runs: list[tuple[ControllerRun, float]] = []
    started_at = time.time()

    def _count(run: ControllerRun, d: str) -> int:
        return sum(1 for s in run.segments if s.decision == d)

    for prompt in prompts:
        # 1. no_compression anchor.
        no_comp_policy = FixedBudgetPolicy(
            budget=no_comp_budget, name="no_compression"
        )
        run = run_controller_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            predictor=predictor,
            policy=no_comp_policy,
            build_press_fn=build_press,
            initial_budget=no_comp_budget,
            compression_interval=args.k,
            max_new_tokens=args.max_new_tokens,
            device=device,
            k=args.k,
            threshold=float("nan"),
        )
        runs.append((run, float("nan")))
        print(
            f"  no_compression {prompt['id']}: "
            f"tokens={run.num_tokens_generated} "
            f"wct/tok={run.wall_clock_per_token:.4f}s "
            f"correct={run.correct} "
            f"evicted={run.total_evicted_tokens}"
        )

        # 2-4. Fixed budget baselines.
        for b in budgets:
            policy = FixedBudgetPolicy(budget=b, name=f"fixed_{b}")
            run = run_controller_once(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                predictor=predictor,
                policy=policy,
                build_press_fn=build_press,
                initial_budget=b,
                compression_interval=args.k,
                max_new_tokens=args.max_new_tokens,
                device=device,
                k=args.k,
                threshold=float("nan"),
            )
            runs.append((run, float("nan")))
            print(
                f"  fixed_{b} {prompt['id']}: "
                f"tokens={run.num_tokens_generated} "
                f"wct/tok={run.wall_clock_per_token:.4f}s "
                f"correct={run.correct}"
            )

        # 5. HERALD.
        herald_tag = _t_tag(threshold)
        herald_policy = RiskBudgetStepPolicy(
            budgets=budgets,
            threshold=threshold,
            start_index=0,
            name=f"herald_t{herald_tag}",
        )
        herald_run = run_controller_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            predictor=predictor,
            policy=herald_policy,
            build_press_fn=build_press,
            initial_budget=herald_policy.initial_budget(),
            compression_interval=args.k,
            max_new_tokens=args.max_new_tokens,
            device=device,
            k=args.k,
            threshold=threshold,
        )
        runs.append((herald_run, threshold))
        print(
            f"  herald_t{herald_tag} {prompt['id']}: "
            f"tokens={herald_run.num_tokens_generated} "
            f"wct/tok={herald_run.wall_clock_per_token:.4f}s "
            f"correct={herald_run.correct} "
            f"relax/tighten/keep="
            f"{_count(herald_run, 'relax')}/"
            f"{_count(herald_run, 'tighten')}/"
            f"{_count(herald_run, 'keep')}"
        )

        # 6. Random matched-budget: replay HERALD's per-prompt
        #    action sequence in seeded-shuffled order.
        herald_actions = [herald_run.initial_budget] + [
            s.next_budget for s in herald_run.segments
        ]
        random_policy = RandomMatchedBudgetPolicy(
            actions=tuple(herald_actions),
            seed=args.random_seed,
            name=f"random_t{herald_tag}",
        )
        random_run = run_controller_once(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            predictor=predictor,
            policy=random_policy,
            build_press_fn=build_press,
            initial_budget=random_policy.initial_budget(),
            compression_interval=args.k,
            max_new_tokens=args.max_new_tokens,
            device=device,
            k=args.k,
            threshold=threshold,
        )
        runs.append((random_run, threshold))
        print(
            f"  random_t{herald_tag} {prompt['id']}: "
            f"tokens={random_run.num_tokens_generated} "
            f"wct/tok={random_run.wall_clock_per_token:.4f}s "
            f"correct={random_run.correct}"
        )

    # Emit artifacts.
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    seg_rows: list[dict[str, Any]] = []
    ev_rows: list[dict[str, Any]] = []
    runs_rows: list[dict[str, Any]] = []
    for r, t in runs:
        rsegs = _segment_rows(r, k=args.k)
        seg_rows.extend(rsegs)
        ev_rows.extend(_event_rows(r))
        mean_ret, max_ret = _retained_cache_summary(rsegs)
        runs_rows.append(
            _run_row(
                r,
                t,
                args.predictor,
                predictor_sha,
                mean_ret,
                max_ret,
            )
        )

    write_parquet(out_dir, "controller_runs", runs_rows, RUN_COLUMNS_EXT)
    write_parquet(
        out_dir, "controller_segments", seg_rows, SEGMENT_COLUMNS_EXT
    )
    write_parquet(out_dir, "controller_events", ev_rows, EVENTS_COLUMNS)

    manifest = {
        "host": socket.gethostname(),
        "predictor_path": str(args.predictor),
        "predictor_sha256": predictor_sha,
        "model": args.model,
        "kvpress_version": kvpress_version,
        "torch_version": torch.__version__,
        "num_prompts": len(prompts),
        "budgets": list(budgets),
        "no_compression_budget": no_comp_budget,
        "k": args.k,
        "threshold": threshold,
        "max_new_tokens": args.max_new_tokens,
        "policies": sorted({r.policy_name for (r, _t) in runs}),
        "n_runs": len(runs),
        "n_segments": len(seg_rows),
        "n_events": len(ev_rows),
        "wall_clock_seconds_total": time.time() - started_at,
        "seed": args.seed,
        "random_seed": args.random_seed,
        "env": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
    }
    (out_dir / "pilot_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    # Smoke health report (printed every run; trivially short so it
    # doesn't hurt full pilots).
    _print_health_report(runs_rows, seg_rows, threshold, out_dir)
    return 0


def _print_health_report(
    runs_rows: list[dict[str, Any]],
    seg_rows: list[dict[str, Any]],
    threshold: float,
    out_dir: Path,
) -> None:
    print("\n--- health report ---")
    by_policy: dict[str, list[dict[str, Any]]] = {}
    for r in runs_rows:
        by_policy.setdefault(r["policy"], []).append(r)
    for policy in sorted(by_policy.keys()):
        rs = by_policy[policy]
        n = len(rs)
        n_correct = sum(1 for r in rs if r["correct"] is True)
        n_unscored = sum(1 for r in rs if r["correct"] is None)
        n_short = sum(1 for r in rs if int(r["num_tokens_generated"]) < 16)
        n_truncated = sum(1 for r in rs if r["stop_reason"] == "max_tokens")
        decode_evts = sum(int(r["decode_event_count"]) for r in rs)
        evicted = sum(int(r["total_evicted_tokens"]) for r in rs)
        n_relax = sum(int(r["n_relax"]) for r in rs)
        n_tighten = sum(int(r["n_tighten"]) for r in rs)
        n_keep = sum(int(r["n_keep"]) for r in rs)
        n_actions = n_relax + n_tighten + n_keep
        relax_rate = (
            f"{(n_relax / n_actions * 100):.1f}%" if n_actions else "n/a"
        )
        print(
            f"  {policy:24s} n={n} "
            f"correct={n_correct} unscored={n_unscored} "
            f"short(<16tok)={n_short} truncated={n_truncated} "
            f"decode_evts={decode_evts} evicted={evicted} "
            f"relax/tighten/keep={n_relax}/{n_tighten}/{n_keep} "
            f"relax_rate={relax_rate}"
        )
    print(f"  threshold={threshold} segments_total={len(seg_rows)}")
    print(f"  artifacts: {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
