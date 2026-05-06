"""Phase 1 Block 3 sweep status snapshot.

One terminal command, one report. Reads `results/phase1/` (per-cell
summaries, run parquets, truncation sidecars) and
`results/phase1_sweep.log` (live activity, watchdog substitutions,
abort markers). CPU-only, dependency-light: stdlib + polars (already a
project dep, used to peek at run-record parquets in the active cell).

Run on Orion (or wherever the sweep is writing):

    uv run python scripts/phase1_status.py
    uv run python scripts/phase1_status.py \
        --results-dir results/phase1_sweep_smoke \
        --log results/phase1_sweep_smoke.log \
        --n-cells 8 --n-prompts-per-cell 1

Health verdict at the end is one of:
    OK        sweep is alive and has not tripped any rule
    WARN     finished partially, watchdog aborted a cell, or stalled
    ABORTED   sweep summary or log reports a sweep-wide abort
    UNKNOWN   no log, no summary, no completed cells (nothing to say)

Designed for Block 3 pre-flight + in-flight monitoring. No dashboard,
no daemon: re-run when curious.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

DEFAULT_N_CELLS = 172
DEFAULT_N_PROMPTS_PER_CELL = 200

_OK_C = "\033[32m"
_WARN_C = "\033[33m"
_ERR_C = "\033[31m"
_DIM_C = "\033[2m"
_RESET = "\033[0m"


def _color(s: str, code: str, use_color: bool) -> str:
    return f"{code}{s}{_RESET}" if use_color else s


def _find_sweep_process() -> tuple[int | None, str | None]:
    """Return (pid, full cmdline) of a phase1_sweep.py process, if any."""
    try:
        out = subprocess.run(
            ["pgrep", "-fa", "phase1_sweep.py"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line or "phase1_status.py" in line:
            continue
        parts = line.split(None, 1)
        try:
            pid = int(parts[0])
        except (ValueError, IndexError):
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        return pid, cmd
    return None, None


def _disk_usage(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        out = subprocess.run(
            ["du", "-sh", str(path)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    line = out.stdout.strip()
    return line.split(None, 1)[0] if line else None


def _scan_cell_summaries(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(root.glob("*/*/ratio=*/cell_summary.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _scan_active_cell(root: Path) -> dict[str, Any] | None:
    """Most-recently-active cell with run parquets but no cell_summary."""
    if not root.exists():
        return None
    candidates: list[dict[str, Any]] = []
    for ratio_dir in root.glob("*/*/ratio=*"):
        if not ratio_dir.is_dir():
            continue
        if (ratio_dir / "cell_summary.json").exists():
            continue
        runs_dir = ratio_dir / "raw" / "runs"
        if not runs_dir.exists():
            continue
        parquets = sorted(runs_dir.glob("*.parquet"))
        if not parquets:
            continue
        ratio_str = ratio_dir.name.split("=", 1)[1]
        try:
            ratio = float(ratio_str)
        except ValueError:
            continue
        latest = max(p.stat().st_mtime for p in parquets)
        candidates.append(
            {
                "task": ratio_dir.parent.parent.name,
                "press": ratio_dir.parent.name,
                "compression_ratio": ratio,
                "n_parquets": len(parquets),
                "latest_mtime": latest,
                "runs_dir": runs_dir,
            }
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: c["latest_mtime"], reverse=True)
    chosen = candidates[0]
    n_ok = 0
    n_failed = 0
    for p in sorted(chosen["runs_dir"].glob("*.parquet")):
        try:
            df = pl.read_parquet(p)
        except Exception:  # noqa: BLE001
            n_failed += 1
            continue
        if df.height == 0 or "replay_status" not in df.columns:
            n_failed += 1
            continue
        status = str(df["replay_status"][0])
        if status == "ok":
            n_ok += 1
        else:
            n_failed += 1
    chosen["n_ok"] = n_ok
    chosen["n_failed"] = n_failed
    return chosen


def _count_truncation_records(root: Path) -> int:
    if not root.exists():
        return 0
    n = 0
    for p in root.glob("longbench*/**/raw/truncation.jsonl"):
        try:
            with p.open() as f:
                for line in f:
                    if line.strip():
                        n += 1
        except OSError:
            continue
    return n


def _latest_artifact_mtime(root: Path) -> float | None:
    if not root.exists():
        return None
    latest = 0.0
    for pattern in ("**/*.parquet", "**/cell_summary.json"):
        for p in root.glob(pattern):
            try:
                m = p.stat().st_mtime
                if m > latest:
                    latest = m
            except OSError:
                continue
    return latest if latest > 0 else None


def _tail_log(log_path: Path, n: int = 20) -> list[str]:
    if not log_path.exists():
        return []
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = min(size, n * 400 + 4096)
            f.seek(size - block)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


_CELL_HEADER_RE = re.compile(
    r"=== ([\w_]+) / ([\w_]+) @ ([\d.]+) \(([\w_]+)\) ==="
)
_BUDGET_RESOLVE_RE = re.compile(r"budget\.resolve: \w+ substitution")
_SWEEP_ABORT_RE = re.compile(r"SWEEP ABORT: (.+)")


def _parse_log_signals(log_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "current_cell": None,
        "current_match_kind": None,
        "latest_substitution_line": None,
        "abort": None,
    }
    if not log_path.exists():
        return out
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        m = _CELL_HEADER_RE.search(line)
        if m:
            out["current_cell"] = (
                m.group(1),
                m.group(2),
                float(m.group(3)),
            )
            out["current_match_kind"] = m.group(4)
            continue
        if _BUDGET_RESOLVE_RE.search(line):
            out["latest_substitution_line"] = line.strip()
            continue
        m = _SWEEP_ABORT_RE.search(line)
        if m:
            out["abort"] = m.group(1).strip()
    return out


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.2f}h"
    return f"{seconds / 86400:.2f}d"


def _format_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def _nvidia_smi_brief() -> list[str] | None:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def _earliest_cell_summary_mtime(
    root: Path, summaries: list[dict[str, Any]]
) -> float | None:
    earliest = None
    for c in summaries:
        path = (
            root
            / c["task"]
            / c["press"]
            / f"ratio={c['compression_ratio']:.4f}"
            / "cell_summary.json"
        )
        if not path.exists():
            continue
        m = path.stat().st_mtime
        if earliest is None or m < earliest:
            earliest = m
    return earliest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 Block 3 sweep status snapshot."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/phase1")
    )
    parser.add_argument(
        "--log", type=Path, default=Path("results/phase1_sweep.log")
    )
    parser.add_argument("--n-cells", type=int, default=DEFAULT_N_CELLS)
    parser.add_argument(
        "--n-prompts-per-cell",
        type=int,
        default=DEFAULT_N_PROMPTS_PER_CELL,
    )
    parser.add_argument("--no-nvidia-smi", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--tail", type=int, default=20)
    args = parser.parse_args()

    use_color = sys.stdout.isatty() and not args.no_color
    now = time.time()

    pid, cmdline = _find_sweep_process()
    completed = _scan_cell_summaries(args.results_dir)
    active = _scan_active_cell(args.results_dir)

    n_attempted = sum(c.get("n_attempted", 0) for c in completed)
    n_ok = sum(c.get("n_ok", 0) for c in completed)
    n_failed = sum(c.get("n_failed", 0) for c in completed)
    n_skipped = sum(c.get("n_skipped", 0) for c in completed)
    if active:
        n_attempted += active["n_parquets"]
        n_ok += active["n_ok"]
        n_failed += active["n_failed"]
    n_total_planned = args.n_cells * args.n_prompts_per_cell

    watchdog_aborted = [c for c in completed if c.get("aborted_by_watchdog")]

    log_sig = _parse_log_signals(args.log)
    sweep_summary_path = args.results_dir / "phase1_sweep_summary.json"
    sweep_summary: dict[str, Any] | None = None
    if sweep_summary_path.exists():
        try:
            sweep_summary = json.loads(sweep_summary_path.read_text())
        except json.JSONDecodeError:
            sweep_summary = None

    latest_artifact = _latest_artifact_mtime(args.results_dir)
    log_mtime = args.log.stat().st_mtime if args.log.exists() else None
    log_ctime = args.log.stat().st_ctime if args.log.exists() else None

    disk = _disk_usage(args.results_dir)
    n_trunc = _count_truncation_records(args.results_dir)

    earliest_summary_mtime = _earliest_cell_summary_mtime(
        args.results_dir, completed
    )
    elapsed: float | None = None
    start_candidates = [
        t for t in (earliest_summary_mtime, log_ctime) if t is not None
    ]
    if start_candidates:
        elapsed = now - min(start_candidates)
    eta: float | None = None
    cells_done = len(completed)
    if elapsed is not None and cells_done > 0 and cells_done < args.n_cells:
        eta = elapsed * (args.n_cells / cells_done - 1)

    gpu_brief = None if args.no_nvidia_smi else _nvidia_smi_brief()

    health = "UNKNOWN"
    health_reasons: list[str] = []
    sweep_aborted = bool(
        (sweep_summary and sweep_summary.get("abort_reason"))
        or log_sig["abort"]
    )
    if sweep_aborted:
        health = "ABORTED"
        if sweep_summary and sweep_summary.get("abort_reason"):
            health_reasons.append(
                f"sweep_summary.abort_reason={sweep_summary['abort_reason']!r}"
            )
        if log_sig["abort"]:
            health_reasons.append(f"log SWEEP ABORT: {log_sig['abort']}")
    elif watchdog_aborted:
        health = "WARN"
        health_reasons.append(
            f"{len(watchdog_aborted)} cell(s) aborted by watchdog"
        )

    if health == "UNKNOWN":
        if pid is not None:
            health = "OK"
            health_reasons.append(f"sweep process alive (pid {pid})")
        elif sweep_summary and sweep_summary.get(
            "n_cells_completed"
        ) == sweep_summary.get("n_cells_planned"):
            health = "OK"
            health_reasons.append(
                "sweep summary reports all planned cells complete"
            )
        elif completed:
            health = "WARN"
            health_reasons.append(
                "no sweep process running, partial results present"
            )
        else:
            health = "UNKNOWN"
            health_reasons.append(
                "no sweep process and no completed cells found"
            )

    if n_failed > 0 and health == "OK":
        health = "WARN"
        health_reasons.append(
            f"{n_failed} failed run(s) recorded in completed cells"
        )

    if (
        pid is not None
        and latest_artifact is not None
        and (now - latest_artifact) > 600
    ):
        if health == "OK":
            health = "WARN"
        health_reasons.append(
            f"no artifact written for {_format_age(now - latest_artifact)} "
            f"(possible stall)"
        )

    print()
    print(
        f"Phase 1 sweep status @ "
        f"{datetime.now().isoformat(timespec='seconds')}"
    )
    print(
        f"results-dir: {args.results_dir}   log: {args.log}   "
        f"n_cells={args.n_cells}, n_prompts/cell={args.n_prompts_per_cell}"
    )
    print()

    print("Process")
    if pid is not None:
        print(f"  pid: {pid}")
        if cmdline:
            print(f"  cmd: {cmdline[:240]}")
    else:
        print("  (no phase1_sweep.py process found via pgrep)")
    print()

    print("Progress")
    print(
        f"  cells completed: {cells_done} / {args.n_cells} "
        f"({100.0 * cells_done / max(args.n_cells, 1):.1f}%)"
    )
    print(
        f"  runs (over completed + active cell): "
        f"attempted={n_attempted} / planned={n_total_planned}"
    )
    print(
        f"  ok={n_ok}  failed={n_failed}  skipped={n_skipped}  "
        f"watchdog_aborted_cells={len(watchdog_aborted)}"
    )
    if active:
        print(
            f"  active cell: {active['task']} / {active['press']} @ "
            f"{active['compression_ratio']:.4f}"
        )
        print(
            f"    parquets={active['n_parquets']}  "
            f"ok={active['n_ok']}  failed={active['n_failed']}  "
            f"last_write={_format_ts(active['latest_mtime'])} "
            f"({_format_age(now - active['latest_mtime'])} ago)"
        )
    elif log_sig["current_cell"]:
        t, p, r = log_sig["current_cell"]
        print(
            f"  last cell header in log: {t} / {p} @ {r}  "
            f"({log_sig['current_match_kind']})"
        )
    print()

    print("Timing")
    if elapsed is not None:
        print(
            f"  elapsed (since first cell or log start): "
            f"{_format_age(elapsed)}"
        )
    else:
        print("  elapsed: (no start signal)")
    if eta is not None:
        print(f"  ETA (linear from cell rate): ~{_format_age(eta)}")
    if latest_artifact:
        print(
            f"  latest artifact write: {_format_ts(latest_artifact)} "
            f"({_format_age(now - latest_artifact)} ago)"
        )
    if log_mtime:
        print(
            f"  log mtime: {_format_ts(log_mtime)} "
            f"({_format_age(now - log_mtime)} ago)"
        )
    print()

    print("Storage / artifacts")
    if disk:
        print(f"  disk used under {args.results_dir}: {disk}")
    print(f"  longbench truncation sidecar records: {n_trunc}")
    print()

    print("Watchdog / budget")
    if log_sig["latest_substitution_line"]:
        print("  latest substitution log line:")
        print(f"    {log_sig['latest_substitution_line'][:240]}")
    else:
        print("  no nearest_ratio / task_fallback substitutions in log")
    if completed:
        kinds: dict[str, int] = {}
        for c in completed:
            k = c.get("budget_match_kind", "?")
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  budget_match_kind distribution: {kinds}")
    if watchdog_aborted:
        print(f"  cells aborted by watchdog: {len(watchdog_aborted)}")
        for c in watchdog_aborted[:5]:
            print(
                f"    - {c['task']}/{c['press']}@{c['compression_ratio']}: "
                f"{c.get('abort_reason')}"
            )
    print()

    if gpu_brief:
        print("GPU (nvidia-smi: index, util%, mem_used MiB, mem_total MiB)")
        for line in gpu_brief:
            print(f"  {line}")
        print()

    if sweep_summary:
        print("Final sweep summary")
        print(
            f"  n_cells_completed={sweep_summary.get('n_cells_completed')}"
            f" / n_cells_planned={sweep_summary.get('n_cells_planned')}"
        )
        wall = sweep_summary.get("total_wall_clock_seconds")
        if isinstance(wall, (int, float)):
            print(f"  total_wall_clock_seconds={wall:.1f}")
        print(f"  abort_reason={sweep_summary.get('abort_reason')}")
        print()

    print(f"Last {args.tail} log lines")
    tail = _tail_log(args.log, args.tail)
    if not tail:
        print("  (log not present)")
    for line in tail:
        print(f"  {line}")
    print()

    color = (
        _OK_C
        if health == "OK"
        else _WARN_C
        if health == "WARN"
        else _ERR_C
        if health == "ABORTED"
        else _DIM_C
    )
    print(f"Health: {_color(health, color, use_color)}")
    for r in health_reasons:
        print(f"  - {r}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
