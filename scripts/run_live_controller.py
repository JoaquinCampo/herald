"""Run the live grace-window controller on the canonical test prompts.

For every (compressor, ratio, test prompt) episode this executes the
frozen-alarm grace-window policy during real generation
(herald.live_controller) and appends one JSONL record with everything
the fidelity evaluation needs: attempted grid points with live alarm
scores, the commit point, live output quality, wall-clock ledgers,
and a token-exact comparison of the live reference prefix against the
recorded sweep reference.

Also produces baseline.jsonl: one full uncompressed reference run per
prompt (timing baseline + determinism check against the recorded
reference).

Resumable: existing keys in the output files are skipped. Designed
for single-run Orion use (batch=1, one GPU).

Usage (Orion):
  HF_HUB_OFFLINE=1 uv run python scripts/run_live_controller.py \
      --limit-prompts 2   # smoke
  HF_HUB_OFFLINE=1 uv run python scripts/run_live_controller.py
"""

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from herald.config import TASKS  # noqa: E402
from herald.generate import generate_baseline, load_model  # noqa: E402
from herald.grace_window import AlarmBundle, GateBundle  # noqa: E402
from herald.live_controller import run_episode  # noqa: E402
from herald.presses import get_press  # noqa: E402
from herald.scoring import score  # noqa: E402
from herald.storage import safe_id  # noqa: E402
from herald.tasks import PromptRecord, load_prompts  # noqa: E402

DEFAULT_COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle-dir", default="results/predictor/alarm_bundle")
    p.add_argument("--gate-dir", default=None)
    p.add_argument("--out-dir", default="results/live_controller")
    p.add_argument(
        "--references-dir",
        default="results/sweep/llama/ifeval/references",
    )
    p.add_argument(
        "--compressors",
        default=",".join(DEFAULT_COMPRESSORS),
    )
    p.add_argument("--ratios", default="0.25,0.5,0.75,0.875")
    p.add_argument("--limit-prompts", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--model-id", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--sustain-interval", type=int, default=None)
    p.add_argument("--prompts-per-task", type=int, default=200)
    return p.parse_args()


def load_recorded_reference(
    ref_dir: Path, prompt_id: str
) -> dict[str, Any] | None:
    path = ref_dir / f"{safe_id(prompt_id)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())  # type: ignore[no-any-return]


def existing_keys(path: Path, key_field: str) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                keys.add(str(json.loads(line)[key_field]))
    return keys


def records_by_key(path: Path, key_field: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records[str(record[key_field])] = record
    return records


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")


def prefix_comparison(
    live_ids: list[int], recorded_ids: list[int]
) -> dict[str, Any]:
    n = min(len(live_ids), len(recorded_ids))
    div = next((i for i in range(n) if live_ids[i] != recorded_ids[i]), None)
    return {
        "live_len": len(live_ids),
        "recorded_len": len(recorded_ids),
        "divergence_index": div,
        "prefix_matches": div is None,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    baseline_path = out_dir / "baseline.jsonl"
    ref_dir = Path(args.references_dir)
    compressors = args.compressors.split(",")
    ratios = [float(r) for r in args.ratios.split(",")]
    max_new_tokens = TASKS["ifeval"].max_new_tokens

    bundles = {
        c: AlarmBundle.load(Path(args.bundle_dir) / c) for c in compressors
    }
    gates = (
        {c: GateBundle.load(Path(args.gate_dir) / c) for c in compressors}
        if args.gate_dir is not None
        else None
    )
    targets = json.loads(
        (Path(args.bundle_dir) / "fidelity_targets.json").read_text()
    )
    test_pids = {
        c: set(targets["compressors"][c]["test_prompt_ids"])
        for c in compressors
    }
    all_pids = sorted(set().union(*test_pids.values()))
    if args.limit_prompts is not None:
        all_pids = all_pids[: args.limit_prompts]
    print(
        f"live controller: {len(all_pids)} prompts x "
        f"{len(compressors)} compressors x {len(ratios)} ratios",
        flush=True,
    )

    records = {
        r.prompt_id: r
        for r in load_prompts(
            "ifeval", args.prompts_per_task, TASKS["ifeval"]
        )
        if r.prompt_id in set(all_pids)
    }
    missing = [p for p in all_pids if p not in records]
    if missing:
        raise RuntimeError(f"prompts not found in loader: {missing[:5]}")

    lm = load_model(
        "llama",
        dtype=args.dtype,
        device=args.device,
        model_id=args.model_id,
    )

    baselines = records_by_key(baseline_path, "prompt_id")
    done_baseline = set(baselines)
    done_episodes = existing_keys(episodes_path, "key")
    n_done = 0

    for pid in all_pids:
        record: PromptRecord = records[pid]
        recorded = load_recorded_reference(ref_dir, pid)
        if pid not in done_baseline:
            t0 = time.perf_counter()
            ref = generate_baseline(lm, record, max_new_tokens)
            wall = time.perf_counter() - t0
            q_ref_live = score("ifeval", ref.text, record.gold)
            base: dict[str, Any] = {
                "prompt_id": pid,
                "wall_s": wall,
                "ref_len": len(ref.gen_ids),
                "q_ref_live": q_ref_live,
                "peak_kv_cache_bytes": ref.peak_kv_cache_bytes,
            }
            if recorded is not None:
                base["q_ref_recorded"] = recorded.get("q")
                base["vs_recorded"] = prefix_comparison(
                    ref.gen_ids, list(recorded.get("gen_ids", []))
                )
            append_jsonl(baseline_path, base)
            baselines[pid] = base
            print(
                f"BASELINE {pid} len={len(ref.gen_ids)} "
                f"q={q_ref_live:.3f} wall={wall:.1f}s "
                f"match={base.get('vs_recorded', {}).get('prefix_matches')}",
                flush=True,
            )

        for compressor in compressors:
            if pid not in test_pids[compressor]:
                continue
            for ratio in ratios:
                key = f"{compressor}|{ratio:.4f}|{pid}"
                if key in done_episodes:
                    continue
                press_factory = partial(get_press, compressor, ratio)
                ep = run_episode(
                    lm,
                    record,
                    press_factory,
                    bundles[compressor],
                    compressor=compressor,
                    ratio=ratio,
                    max_new_tokens=max_new_tokens,
                    stride=args.stride,
                    gate=None if gates is None else gates[compressor],
                    sustain_interval=args.sustain_interval,
                )
                q_live = score("ifeval", ep.text, record.gold)
                q_ref_rec = (
                    recorded.get("q") if recorded is not None else None
                )
                q_ref_live = float(baselines[pid]["q_ref_live"])
                rec_len = (
                    len(recorded.get("gen_ids", []))
                    if recorded is not None
                    else None
                )
                savings = None
                if ep.commit_s is not None and rec_len:
                    savings = max(0.0, 1.0 - ep.commit_s / rec_len)
                elif ep.commit_s is None:
                    savings = 0.0
                obj: dict[str, Any] = {
                    "key": key,
                    "prompt_id": pid,
                    "compressor": compressor,
                    "ratio": ratio,
                    "sustain_interval": args.sustain_interval,
                    "commit_s": ep.commit_s,
                    "attempts": [
                        {
                            "s": a.s,
                            "score": a.score,
                            "committed": a.committed,
                            "n_new_tokens": a.n_new_tokens,
                            "wall_s": a.wall_s,
                            "peak_kv_cache_bytes": a.peak_kv_cache_bytes,
                            "recomputed_prefill_tokens": (
                                a.recomputed_prefill_tokens
                            ),
                            "gate_score": a.gate_score,
                        }
                        for a in ep.attempts
                    ],
                    "skips": [
                        {
                            "s": skip.s,
                            "gate_score": skip.gate_score,
                            "wall_s": skip.wall_s,
                        }
                        for skip in ep.skips
                    ],
                    "ref_len_live": len(ep.ref_ids),
                    "ref_done": ep.ref_done,
                    "n_new_ids": len(ep.new_ids),
                    "q_live": q_live,
                    "dq_live": q_ref_live - float(q_live),
                    "q_ref_recorded": q_ref_rec,
                    "dq_recorded": (
                        None
                        if q_ref_rec is None
                        else float(q_ref_rec) - float(q_live)
                    ),
                    "savings": savings,
                    "ref_wall_s": ep.ref_wall_s,
                    "total_wall_s": ep.total_wall_s,
                    "peak_mem_bytes": ep.peak_mem_bytes,
                    "peak_kv_cache_bytes": ep.peak_kv_cache_bytes,
                    "text": ep.text,
                }
                if recorded is not None:
                    obj["ref_vs_recorded"] = prefix_comparison(
                        ep.ref_ids, list(recorded.get("gen_ids", []))
                    )
                append_jsonl(episodes_path, obj)
                n_done += 1
                print(
                    f"EPISODE {key} commit_s={ep.commit_s} "
                    f"attempts={len(ep.attempts)} q={q_live:.3f} "
                    f"skips={len(ep.skips)} dq={obj['dq_live']} "
                    f"sav={savings} "
                    f"wall={ep.total_wall_s:.1f}s",
                    flush=True,
                )
    print(f"done: {n_done} new episodes", flush=True)


if __name__ == "__main__":
    main()
