# pyright: reportMissingImports=false

"""Run paired live evidence for the dependency-free int8 KV cache."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, "src")

from herald.config import MODELS, TASKS  # noqa: E402
from herald.deployment_evidence import (  # noqa: E402
    END_TO_END_RETAINED_KV_CACHE,
    candidate_id,
    initialize_live_run,
    verify_live_run,
)
from herald.expected_attention_stats import StatisticsArtifact  # noqa: E402
from herald.generate import (  # noqa: E402
    generate_always_on_press,
    generate_always_on_streaming,
    generate_always_on_sustained,
    generate_baseline,
    generate_int8_cache,
    load_model,
)
from herald.presses import get_press  # noqa: E402
from herald.scoring import score  # noqa: E402
from herald.tasks import PromptRecord, load_prompts  # noqa: E402

COMPRESSOR = "int8_cache"
RATIO = 0.5
RESIDUAL_LENGTH = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "results/expected_stats_alarm_bundle_full_s0_v2/fidelity_targets.json"
        ),
    )
    parser.add_argument(
        "--mechanism",
        choices=(
            "int8",
            "streaming_low_ratio",
            "snapkv",
            "expected_stats_sustained",
        ),
        default="int8",
    )
    parser.add_argument(
        "--expected-attention-stats",
        type=Path,
        default=Path(
            "results/calibration/expected_attention_stats_ifeval_s0_v2"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument("--prompts-per-task", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a") as file:
        file.write(json.dumps(value) + "\n")


def records_by_key(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open() as file:
        for line in file:
            if not line.strip():
                continue
            value = json.loads(line)
            record_key = str(value[key])
            if record_key in records:
                raise RuntimeError(f"duplicate {key} in {path}: {record_key}")
            records[record_key] = value
    return records


def main() -> None:
    args = parse_args()
    if args.mechanism == "int8":
        compressor = COMPRESSOR
        ratio = RATIO
        sustain_interval = None
    elif args.mechanism == "streaming_low_ratio":
        compressor = "streaming_llm_always_on"
        ratio = 0.05
        sustain_interval = 32
    elif args.mechanism == "snapkv":
        compressor = "snapkv_always_on"
        ratio = 0.25
        sustain_interval = None
    else:
        compressor = "expected_attention_stats_always_on"
        ratio = 0.25
        sustain_interval = 32
    statistics = (
        StatisticsArtifact.load(args.expected_attention_stats)
        if args.mechanism == "expected_stats_sustained"
        else None
    )
    targets = json.loads(args.targets.read_text())
    prompt_ids = sorted(
        targets["compressors"]["expected_attention_stats"]["test_prompt_ids"]
    )
    if args.limit_prompts is not None:
        prompt_ids = prompt_ids[: args.limit_prompts]
    current_candidate_id = candidate_id(compressor, ratio, sustain_interval)
    manifest = initialize_live_run(
        args.out_dir,
        {
            "task": "ifeval",
            "prompt_ids": prompt_ids,
            "candidate_ids": [current_candidate_id],
            "candidate_prompt_ids": {current_candidate_id: prompt_ids},
            "model_name_or_path": args.model_id or MODELS["llama"],
            "dtype": args.dtype,
            "device": args.device,
            "compressor": compressor,
            "ratio": ratio,
            "sustain_interval": sustain_interval,
            "mechanism": args.mechanism,
            "nbits": 8 if args.mechanism == "int8" else None,
            "residual_length": (
                RESIDUAL_LENGTH if args.mechanism == "int8" else None
            ),
            "targets": str(args.targets),
            "expected_attention_stats": (
                str(args.expected_attention_stats)
                if statistics is not None
                else None
            ),
            "expected_attention_stats_sha256": (
                statistics.digest if statistics is not None else None
            ),
        },
        resume=args.resume,
    )
    run_id = str(manifest["run_id"])
    baseline_path = args.out_dir / "baseline.jsonl"
    episodes_path = args.out_dir / "episodes.jsonl"
    baselines = records_by_key(baseline_path, "prompt_id")
    episodes = records_by_key(episodes_path, "key")
    errors = verify_live_run(
        manifest,
        list(baselines.values()),
        list(episodes.values()),
    )
    if errors:
        raise RuntimeError(
            "invalid resumed live evidence: " + "; ".join(errors)
        )

    records = {
        record.prompt_id: record
        for record in load_prompts(
            "ifeval", args.prompts_per_task, TASKS["ifeval"]
        )
        if record.prompt_id in set(prompt_ids)
    }
    missing = [
        prompt_id for prompt_id in prompt_ids if prompt_id not in records
    ]
    if missing:
        raise RuntimeError(f"prompts not found in loader: {missing[:5]}")
    model = load_model(
        "llama",
        dtype=args.dtype,
        device=args.device,
        model_id=args.model_id,
    )
    max_new_tokens = TASKS["ifeval"].max_new_tokens
    print(f"{args.mechanism}: {len(prompt_ids)} paired prompts", flush=True)

    completed = 0
    for prompt_id in prompt_ids:
        record: PromptRecord = records[prompt_id]
        if prompt_id not in baselines:
            started = time.perf_counter()
            baseline = generate_baseline(model, record, max_new_tokens)
            baseline_wall = time.perf_counter() - started
            baseline_record: dict[str, Any] = {
                "prompt_id": prompt_id,
                "wall_s": baseline_wall,
                "ref_len": len(baseline.gen_ids),
                "q_ref_live": score("ifeval", baseline.text, record.gold),
                "run_id": run_id,
                "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
                "peak_kv_cache_bytes": baseline.peak_kv_cache_bytes,
            }
            append_jsonl(baseline_path, baseline_record)
            baselines[prompt_id] = baseline_record
            print(
                f"BASELINE {prompt_id} len={len(baseline.gen_ids)} "
                f"q={baseline_record['q_ref_live']:.3f} "
                f"wall={baseline_wall:.1f}s",
                flush=True,
            )

        key = f"{current_candidate_id}|{prompt_id}"
        if key in episodes:
            continue
        started = time.perf_counter()
        if args.mechanism == "int8":
            candidate, cache = generate_int8_cache(
                model,
                record,
                max_new_tokens,
                residual_length=RESIDUAL_LENGTH,
            )
            final_kv_cache_bytes = cache.retained_peak_nbytes()
        elif args.mechanism == "streaming_low_ratio":
            candidate = generate_always_on_streaming(
                model,
                record,
                max_new_tokens,
                ratio=ratio,
                sustain_interval=32,
            )
            final_kv_cache_bytes = candidate.peak_kv_cache_bytes
        elif args.mechanism == "snapkv":
            candidate = generate_always_on_press(
                model,
                record,
                max_new_tokens,
                press=get_press("snapkv", ratio),
            )
            final_kv_cache_bytes = candidate.peak_kv_cache_bytes
        else:
            if statistics is None:
                raise RuntimeError("expected statistics were not loaded")
            candidate = generate_always_on_sustained(
                model,
                record,
                max_new_tokens,
                press=get_press(
                    "expected_attention_stats",
                    ratio,
                    model=model.model,
                    statistics=statistics,
                ),
                ratio=ratio,
                sustain_interval=32,
            )
            final_kv_cache_bytes = candidate.peak_kv_cache_bytes
        total_wall = time.perf_counter() - started
        q_live = score("ifeval", candidate.text, record.gold)
        q_ref = float(baselines[prompt_id]["q_ref_live"])
        episode: dict[str, Any] = {
            "key": key,
            "prompt_id": prompt_id,
            "compressor": compressor,
            "ratio": ratio,
            "sustain_interval": sustain_interval,
            "candidate_id": current_candidate_id,
            "run_id": run_id,
            "kv_measurement_scope": END_TO_END_RETAINED_KV_CACHE,
            "commit_s": 0,
            "attempts": [],
            "skips": [],
            "ref_len_live": int(baselines[prompt_id]["ref_len"]),
            "ref_done": True,
            "n_new_ids": len(candidate.gen_ids),
            "q_live": q_live,
            "dq_live": q_ref - float(q_live),
            "q_ref_recorded": None,
            "dq_recorded": None,
            "savings": None,
            "ref_wall_s": 0.0,
            "total_wall_s": total_wall,
            "peak_mem_bytes": 0,
            "peak_kv_cache_bytes": candidate.peak_kv_cache_bytes,
            "final_kv_cache_bytes": final_kv_cache_bytes,
            "text": candidate.text,
        }
        append_jsonl(episodes_path, episode)
        episodes[key] = episode
        completed += 1
        print(
            f"EPISODE {key} q={q_live:.3f} dq={episode['dq_live']} "
            f"wall={total_wall:.1f}s kv={candidate.peak_kv_cache_bytes}",
            flush=True,
        )

    errors = verify_live_run(
        manifest,
        list(baselines.values()),
        list(episodes.values()),
        require_complete=True,
    )
    if errors:
        raise RuntimeError("incomplete live evidence: " + "; ".join(errors))
    print(f"done: {completed} new episodes", flush=True)


if __name__ == "__main__":
    main()
