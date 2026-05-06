"""Phase 1 Block 3 fixed-ratio sweep with watchdog.

Lifts the per-cell loop from `phase1_profile.py` and wires in the five
watchdog rules from `gold/phase-1-block3-launch-packet.md`:

    1. Cost drift per cell (CostWatchdog).
    2. Failure rate per cell > max-failure-rate -> abort sweep.
    3. Last K non-ok runs share the same replay_error class -> abort.
    4. Peak memory > max-peak-memory-mb -> abort.
    5. LongBench truncation metadata missing or loop_exhausted -> abort.

Plus:
    - dry-run mode: enumerate planned cells/runs, exit 0 (no model load).
    - smoke mode:   1 prompt × 1 baseline + 1 compressed cell.
    - skip-existing: per-cell + per-prompt resumability.
    - per-task baseline cells run before any compressed cell in the same
      task so baseline_run_id linkage is valid.

The cost budget is consulted via `CostBudget.resolve(...)` so a
ratio not present in the budget falls back to the nearest profiled
ratio for that (task, press); a press not present falls back to the
explicitly recorded `task_fallbacks[task]`; a missing task hard-fails.
"""

import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger

from herald.config import ExperimentConfig, make_run_id
from herald.cost_budget import (
    BudgetMatch,
    CostBudget,
    load_cost_budget,
)
from herald.experiment import (
    CostWatchdog,
    load_model,
    run_single_with_replay,
)
from herald.metrics.io import PerRunPaths
from herald.tasks import PHASE1_TASKS, Task

DEFAULT_MAX_PEAK_MEMORY_MB = 28.0 * 1024.0
DEFAULT_MAX_FAILURE_RATE = 0.02
DEFAULT_REPLAY_ERROR_STREAK_ABORT = 3


class SweepAbort(Exception):
    """Raised when a sweep-wide watchdog rule trips. The message is the
    user-facing reason; the exception is caught at the top of `main()`
    and turned into exit code 3."""


@dataclass
class CellPlan:
    task: Task
    press: str
    compression_ratio: float
    is_baseline: bool


@dataclass
class CellOutcome:
    task: str
    press: str
    compression_ratio: float
    n_attempted: int = 0
    n_failed: int = 0
    n_ok: int = 0
    n_skipped: int = 0
    aborted_by_watchdog: bool = False
    abort_reason: str | None = None
    budget_match_kind: str = "exact"
    wall_clock_per_token_median: float | None = None
    peak_memory_mb_median: float | None = None
    cell_wall_clock_seconds: float = 0.0


@dataclass
class ReplayErrorStreak:
    """Tracks the most recent non-ok runs (sweep-wide) so rule 3 can
    fire when the same error class repeats."""

    streak_window: int = DEFAULT_REPLAY_ERROR_STREAK_ABORT
    recent: list[str] = field(default_factory=list)

    def push(self, error_class: str) -> bool:
        self.recent.append(error_class)
        if len(self.recent) > self.streak_window:
            self.recent = self.recent[-self.streak_window :]
        if len(self.recent) < self.streak_window:
            return False
        return all(e == self.recent[0] for e in self.recent)


def _task_by_name(name: str) -> Task:
    for t in PHASE1_TASKS:
        if t.name == name:
            return t
    raise SystemExit(
        f"Unknown task {name!r}. Available: "
        f"{[t.name for t in PHASE1_TASKS]}"
    )


def _build_plan(
    tasks: list[Task],
    presses: list[str],
    ratios: list[float],
    include_baseline: bool,
) -> list[CellPlan]:
    plan: list[CellPlan] = []
    for task in tasks:
        if include_baseline:
            plan.append(
                CellPlan(
                    task=task,
                    press="none",
                    compression_ratio=0.0,
                    is_baseline=True,
                )
            )
        for press in presses:
            for ratio in ratios:
                plan.append(
                    CellPlan(
                        task=task,
                        press=press,
                        compression_ratio=ratio,
                        is_baseline=False,
                    )
                )
    return plan


def _cell_root(
    output_root: Path, task: str, press: str, ratio: float
) -> Path:
    return output_root / task / press / f"ratio={ratio:.4f}"


def _maybe_skip(paths: PerRunPaths) -> bool:
    if not paths.all_exist():
        return False
    df = pl.read_parquet(paths.run)
    return df.height == 1 and df["replay_status"][0] == "ok"


def _classify_replay_error(err: str | None) -> str:
    if not err:
        return "unknown"
    head = err.strip().splitlines()[0].strip()
    return head[:80] or "unknown"


def _load_truncation_records(
    cell_root: Path, run_id: str
) -> list[dict[str, Any]]:
    sidecar = cell_root / "raw" / "truncation.jsonl"
    if not sidecar.exists():
        return []
    out = []
    with sidecar.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("run_id") == run_id:
                out.append(rec)
    return out


def _check_truncation_rule(
    *,
    cell_root: Path,
    run_id: str,
    task_name: str,
    rr_df: pl.DataFrame,
) -> str | None:
    """Returns an abort reason (rule 5) if the truncation sidecar is
    inconsistent with the run record, else None.

    Applies only to LongBench-family tasks (where format_prompt may
    truncate). Other tasks short-circuit to None.
    """
    if not task_name.startswith("longbench"):
        return None
    records = _load_truncation_records(cell_root, run_id)
    if not records:
        return (
            f"truncation sidecar missing record for run_id={run_id} "
            f"(task={task_name})"
        )
    latest = records[-1]
    if latest.get("loop_exhausted") is True:
        return (
            f"truncation loop_exhausted=True for run_id={run_id} "
            f"(task={task_name})"
        )
    return None


def _resolve_budget(
    budget: CostBudget,
    plan: list[CellPlan],
    raise_on_missing: bool,
) -> tuple[
    dict[tuple[str, str, float], BudgetMatch],
    list[tuple[CellPlan, str]],
]:
    """Resolve every planned cell up-front. Returns (matches, errors).

    If `raise_on_missing` is True (production path), a hard-fail at any
    cell raises immediately. If False (dry-run), errors are accumulated
    so the operator can see the full picture before fixing the budget.
    """
    matches: dict[tuple[str, str, float], BudgetMatch] = {}
    errors: list[tuple[CellPlan, str]] = []
    for cell in plan:
        if cell.is_baseline:
            continue
        try:
            match = budget.resolve(
                cell.task.name, cell.press, cell.compression_ratio
            )
        except Exception as exc:  # BudgetTaskNotFound or similar
            if raise_on_missing:
                raise
            errors.append((cell, str(exc)))
            continue
        matches[(cell.task.name, cell.press, cell.compression_ratio)] = (
            match
        )
    return matches, errors


def _print_plan(
    plan: list[CellPlan],
    num_prompts: int,
    matches: dict[tuple[str, str, float], BudgetMatch] | None,
    longbench_subtask: str | None,
) -> None:
    n_cells = len(plan)
    n_runs = n_cells * num_prompts
    print()
    print(f"Phase 1 sweep plan: {n_cells} cells, {n_runs} runs")
    if longbench_subtask:
        print(f"  LongBench subtask = {longbench_subtask}")
    by_task: dict[str, int] = {}
    for c in plan:
        by_task[c.task.name] = by_task.get(c.task.name, 0) + 1
    for task_name, n in sorted(by_task.items()):
        print(f"  {task_name}: {n} cells")
    print()
    print(
        f"{'task':<18} {'press':<22} {'ratio':>7} {'baseline':>9} "
        f"{'budget_match':>13}"
    )
    print("-" * 74)
    for c in plan:
        key = (c.task.name, c.press, c.compression_ratio)
        if c.is_baseline:
            mk = "baseline"
        elif matches and key in matches:
            mk = matches[key].match_kind
        else:
            mk = "?"
        print(
            f"{c.task.name:<18} {c.press:<22} "
            f"{c.compression_ratio:>7.4f} {str(c.is_baseline):>9} "
            f"{mk:>13}"
        )
    print()


def _profile_cell(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    plan: CellPlan,
    prompts: list[dict[str, Any]],
    output_root: Path,
    model_name: str,
    max_new_tokens: int,
    seed: int,
    prompt_timeout_seconds: float,
    top_k: int,
    cost_match: BudgetMatch | None,
    skip_existing: bool,
    sweep_state: "SweepState",
    args: argparse.Namespace,
) -> CellOutcome:
    cell_root = _cell_root(
        output_root, plan.task.name, plan.press, plan.compression_ratio
    )
    cell_root.mkdir(parents=True, exist_ok=True)
    cfg = ExperimentConfig(
        model_name=model_name,
        press_name=plan.press,
        compression_ratio=plan.compression_ratio,
        num_prompts=len(prompts),
        seed=seed,
        output_dir=cell_root,
        max_new_tokens=max_new_tokens,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )

    outcome = CellOutcome(
        task=plan.task.name,
        press=plan.press,
        compression_ratio=plan.compression_ratio,
        budget_match_kind=cost_match.match_kind if cost_match else "baseline",
    )

    watchdog: CostWatchdog | None = None
    if cost_match is not None:
        watchdog = CostWatchdog(
            predicted_s_per_token=cost_match.cell.predicted_s_per_token,
            tolerance=args.watchdog_tolerance,
            consecutive_breaches=args.watchdog_consecutive_breaches,
            min_observations_before_trip=args.watchdog_min_observations,
        )

    wall_per_token: list[float] = []
    peak_mem: list[float] = []
    cell_t0 = time.perf_counter()

    for p in prompts:
        run_id = make_run_id(
            p["id"], plan.press, plan.compression_ratio, seed
        )
        baseline_id = (
            None
            if plan.is_baseline
            else make_run_id(p["id"], "none", 0.0, seed)
        )
        paths = PerRunPaths(root=cell_root, run_id=run_id)
        if skip_existing and _maybe_skip(paths):
            outcome.n_skipped += 1
            continue

        outcome.n_attempted += 1
        try:
            run_single_with_replay(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt_data=p,
                config=cfg,
                baseline_run_id=baseline_id,
                output_root=cell_root,
                task=plan.task,
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                f"  FAILED {plan.task.name}/{plan.press}@"
                f"{plan.compression_ratio} {p['id']}: {exc!r}"
            )
            outcome.n_failed += 1
            err_class = _classify_replay_error(repr(exc))
            if sweep_state.replay_error_streak.push(err_class):
                raise SweepAbort(
                    f"rule 3: last "
                    f"{sweep_state.replay_error_streak.streak_window} "
                    f"non-ok runs share error class {err_class!r}"
                )
            sweep_state.maybe_check_failure_rate(
                outcome, args.max_failure_rate
            )
            continue

        if not paths.all_exist():
            outcome.n_failed += 1
            err_class = _classify_replay_error("missing parquet")
            if sweep_state.replay_error_streak.push(err_class):
                raise SweepAbort(
                    "rule 3: repeated 'missing parquet' failures"
                )
            sweep_state.maybe_check_failure_rate(
                outcome, args.max_failure_rate
            )
            continue

        rr_df = pl.read_parquet(paths.run)
        replay_status = str(rr_df["replay_status"][0])
        replay_error = (
            None
            if rr_df["replay_error"][0] is None
            else str(rr_df["replay_error"][0])
        )
        if replay_status != "ok":
            outcome.n_failed += 1
            err_class = _classify_replay_error(
                replay_error or replay_status
            )
            if sweep_state.replay_error_streak.push(err_class):
                raise SweepAbort(
                    f"rule 3: last "
                    f"{sweep_state.replay_error_streak.streak_window} "
                    f"non-ok runs share error class {err_class!r}"
                )
            sweep_state.maybe_check_failure_rate(
                outcome, args.max_failure_rate
            )
            continue

        outcome.n_ok += 1
        # Streak resets on a successful run.
        sweep_state.replay_error_streak.recent.clear()

        wpt = float(rr_df["wall_clock_per_token"][0])
        pmem = float(rr_df["peak_memory_mb"][0])
        if wpt == wpt and wpt > 0:
            wall_per_token.append(wpt)
        if pmem == pmem and pmem > 0:
            peak_mem.append(pmem)
            if pmem > args.max_peak_memory_mb:
                raise SweepAbort(
                    f"rule 4: peak_memory_mb={pmem:.0f} > "
                    f"max_peak_memory_mb={args.max_peak_memory_mb:.0f} "
                    f"(run {run_id} in {plan.task.name}/{plan.press}@"
                    f"{plan.compression_ratio})"
                )

        trunc_reason = _check_truncation_rule(
            cell_root=cell_root,
            run_id=run_id,
            task_name=plan.task.name,
            rr_df=rr_df,
        )
        if trunc_reason is not None:
            raise SweepAbort(f"rule 5: {trunc_reason}")

        if watchdog is not None:
            wpt_obs = float(rr_df["wall_clock_per_token"][0])
            if watchdog.observe_wpt(wpt_obs):
                outcome.aborted_by_watchdog = True
                outcome.abort_reason = (
                    f"rule 1: wall_clock_per_token drifted above "
                    f"predicted_s_per_token "
                    f"({watchdog.predicted_s_per_token:.5f}) * tolerance "
                    f"({watchdog.tolerance}) for "
                    f"{watchdog.consecutive_breaches} consecutive prompts"
                )
                logger.warning(
                    f"  watchdog tripped: {outcome.abort_reason}"
                )
                break

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    outcome.cell_wall_clock_seconds = time.perf_counter() - cell_t0
    if wall_per_token:
        outcome.wall_clock_per_token_median = statistics.median(
            wall_per_token
        )
    if peak_mem:
        outcome.peak_memory_mb_median = statistics.median(peak_mem)

    summary_path = cell_root / "cell_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task": outcome.task,
                "press": outcome.press,
                "compression_ratio": outcome.compression_ratio,
                "n_attempted": outcome.n_attempted,
                "n_ok": outcome.n_ok,
                "n_failed": outcome.n_failed,
                "n_skipped": outcome.n_skipped,
                "aborted_by_watchdog": outcome.aborted_by_watchdog,
                "abort_reason": outcome.abort_reason,
                "budget_match_kind": outcome.budget_match_kind,
                "wall_clock_per_token_median": (
                    outcome.wall_clock_per_token_median
                ),
                "peak_memory_mb_median": outcome.peak_memory_mb_median,
                "cell_wall_clock_seconds": (
                    outcome.cell_wall_clock_seconds
                ),
            },
            indent=2,
        )
    )
    return outcome


@dataclass
class SweepState:
    replay_error_streak: ReplayErrorStreak = field(
        default_factory=ReplayErrorStreak
    )

    def maybe_check_failure_rate(
        self, outcome: CellOutcome, max_failure_rate: float
    ) -> None:
        if outcome.n_attempted == 0:
            return
        rate = outcome.n_failed / outcome.n_attempted
        if rate > max_failure_rate and outcome.n_attempted >= 5:
            raise SweepAbort(
                f"rule 2: cell failure rate {rate:.1%} > "
                f"{max_failure_rate:.1%} "
                f"(task={outcome.task} press={outcome.press} "
                f"ratio={outcome.compression_ratio})"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 Block 3 fixed-ratio sweep with watchdog."
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="gsm8k,humaneval,ifeval,longbench_single",
    )
    parser.add_argument(
        "--presses",
        type=str,
        default="streaming_llm,snapkv,knorm,expected_attention,tova,random",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        default="0.25,0.375,0.5,0.75,0.875,0.9375,0.96875",
    )
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-7B-Instruct"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/phase1")
    )
    parser.add_argument("--cost-budget", type=Path, default=None)
    parser.add_argument("--watchdog-tolerance", type=float, default=1.25)
    parser.add_argument(
        "--watchdog-consecutive-breaches", type=int, default=3
    )
    parser.add_argument(
        "--watchdog-min-observations", type=int, default=3
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATE,
    )
    parser.add_argument(
        "--max-peak-memory-mb",
        type=float,
        default=DEFAULT_MAX_PEAK_MEMORY_MB,
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Enumerate planned cells/runs and exit. No model load.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 prompt × (baseline + first compressed cell) per "
        "task. Forces --num-prompts=1, --include-baseline.",
    )
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument(
        "--prompt-timeout-seconds", type=float, default=300.0
    )
    args = parser.parse_args()

    task_names = [s.strip() for s in args.tasks.split(",") if s.strip()]
    press_names = [s.strip() for s in args.presses.split(",") if s.strip()]
    ratios = [float(s) for s in args.ratios.split(",") if s.strip()]
    tasks = [_task_by_name(n) for n in task_names]

    if args.smoke:
        args.num_prompts = 1
        args.include_baseline = True
        ratios = ratios[:1]
        press_names = press_names[:1]

    plan = _build_plan(
        tasks=tasks,
        presses=press_names,
        ratios=ratios,
        include_baseline=args.include_baseline,
    )

    budget: CostBudget | None = None
    longbench_subtask: str | None = None
    matches: dict[tuple[str, str, float], BudgetMatch] | None = None
    resolve_errors: list[tuple[CellPlan, str]] = []
    if args.cost_budget is not None and args.cost_budget.exists():
        budget = load_cost_budget(args.cost_budget)
        raw = json.loads(args.cost_budget.read_text())
        longbench_subtask = raw.get("longbench_subtask")
        matches, resolve_errors = _resolve_budget(
            budget, plan, raise_on_missing=not args.dry_run
        )

    if args.dry_run:
        _print_plan(plan, args.num_prompts, matches, longbench_subtask)
        if resolve_errors:
            print()
            print(
                f"BudgetTaskNotFound for {len(resolve_errors)} cells:"
            )
            for cell, err in resolve_errors[:10]:
                print(
                    f"  {cell.task.name}/{cell.press}@"
                    f"{cell.compression_ratio}: {err}"
                )
            if len(resolve_errors) > 10:
                print(f"  ... and {len(resolve_errors) - 10} more")
            print(
                "dry-run: no model load. Returning non-zero because "
                "the budget does not cover all planned cells."
            )
            return 2
        print("dry-run: no model load, no GPU work.")
        return 0

    if budget is None:
        raise SystemExit(
            "--cost-budget is required (use --dry-run to inspect a plan "
            "without a budget)."
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    _print_plan(plan, args.num_prompts, matches, longbench_subtask)

    cfg_template = ExperimentConfig(
        model_name=args.model,
        press_name="none",
        compression_ratio=0.0,
        num_prompts=args.num_prompts,
        seed=args.seed,
        output_dir=args.output_root,
        max_new_tokens=args.max_new_tokens,
        prompt_timeout_seconds=args.prompt_timeout_seconds,
    )
    model, tok, device = load_model(cfg_template)

    sweep_state = SweepState()
    outcomes: list[CellOutcome] = []
    overall_t0 = time.perf_counter()
    abort_reason: str | None = None

    for cell in plan:
        prompts = cell.task.load(args.num_prompts, args.seed)
        cost_match = (
            matches.get(
                (cell.task.name, cell.press, cell.compression_ratio)
            )
            if (matches and not cell.is_baseline)
            else None
        )
        if cell.is_baseline:
            cell_kind = "baseline"
        elif cost_match is not None:
            cell_kind = cost_match.match_kind
        else:
            cell_kind = "?"
        logger.info(
            f"=== {cell.task.name} / {cell.press} @ "
            f"{cell.compression_ratio} ({cell_kind}) ==="
        )
        try:
            outcome = _profile_cell(
                model=model,
                tokenizer=tok,
                device=device,
                plan=cell,
                prompts=prompts,
                output_root=args.output_root,
                model_name=args.model,
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                prompt_timeout_seconds=args.prompt_timeout_seconds,
                top_k=args.top_k,
                cost_match=cost_match,
                skip_existing=args.skip_existing,
                sweep_state=sweep_state,
                args=args,
            )
            outcomes.append(outcome)
        except SweepAbort as exc:
            logger.error(f"SWEEP ABORT: {exc}")
            abort_reason = str(exc)
            break

    total_wall = time.perf_counter() - overall_t0
    summary_path = args.output_root / "phase1_sweep_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model": args.model,
                "max_new_tokens": args.max_new_tokens,
                "num_prompts": args.num_prompts,
                "seed": args.seed,
                "longbench_subtask": longbench_subtask,
                "total_wall_clock_seconds": total_wall,
                "n_cells_completed": len(outcomes),
                "n_cells_planned": len(plan),
                "abort_reason": abort_reason,
                "outcomes": [
                    {
                        "task": o.task,
                        "press": o.press,
                        "compression_ratio": o.compression_ratio,
                        "n_attempted": o.n_attempted,
                        "n_ok": o.n_ok,
                        "n_failed": o.n_failed,
                        "n_skipped": o.n_skipped,
                        "aborted_by_watchdog": o.aborted_by_watchdog,
                        "abort_reason": o.abort_reason,
                        "budget_match_kind": o.budget_match_kind,
                        "wall_clock_per_token_median": (
                            o.wall_clock_per_token_median
                        ),
                        "peak_memory_mb_median": o.peak_memory_mb_median,
                        "cell_wall_clock_seconds": (
                            o.cell_wall_clock_seconds
                        ),
                    }
                    for o in outcomes
                ],
            },
            indent=2,
        )
    )
    logger.info(f"Sweep summary -> {summary_path}")

    print()
    print(f"Total wall-clock: {total_wall:.1f} s")
    return 3 if abort_reason else 0


if __name__ == "__main__":
    sys.exit(main())
