import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

COMPRESSORS = (
    "expected_attention",
    "knorm",
    "streaming_llm",
)
EXPECTED_EPISODE_COUNT = 552
EXPECTED_PROMPT_COUNT = 46
EXPECTED_RATIOS = (0.25, 0.5, 0.75, 0.875)
EXPECTED_EPISODES_PER_COMPRESSOR = 184
REFERENCE_SOURCE = "Money made = 9 (eggs left) * 2 (price per egg) = $18"
COMPRESSED_SOURCE = (
    "Number of boxes = Eggs left / Eggs per box = 9 / 3 = 3 boxes"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line
    ]


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric")
    return sum(values) / len(values)


def compressor_metrics(
    episodes: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    savings: list[float] = []
    costs: list[float] = []
    revert_wall: list[float] = []
    token_overhead: list[float] = []
    for episode in episodes:
        baseline = baselines[episode["prompt_id"]]
        committed = episode["commit_s"] is not None
        savings.append(
            max(0.0, 1.0 - episode["commit_s"] / baseline["ref_len"])
            if committed
            else 0.0
        )
        costs.append(
            float(baseline["q_ref_live"]) - float(episode["q_live"])
            if committed
            else 0.0
        )
        reverted = [
            attempt
            for attempt in episode["attempts"]
            if not attempt["committed"]
        ]
        revert_wall.append(
            sum(float(attempt["wall_s"]) for attempt in reverted)
            / float(baseline["wall_s"])
        )
        recorded_len = episode["ref_vs_recorded"]["recorded_len"]
        token_overhead.append(2.0 * len(reverted) / recorded_len)

    return {
        "episodes": len(episodes),
        "compressed_generation_fraction": mean(savings),
        "quality_cost": mean(costs),
        "revert_wall_overhead": mean(revert_wall),
        "token_overhead": mean(token_overhead),
    }


def validate_campaign(
    episodes: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    prompt_ids = [row["prompt_id"] for row in baseline_rows]
    duplicate_prompt_ids = sorted(
        prompt_id
        for prompt_id, count in {
            prompt_id: prompt_ids.count(prompt_id) for prompt_id in prompt_ids
        }.items()
        if count > 1
    )
    if duplicate_prompt_ids:
        raise ValueError(
            f"duplicate baseline prompt_id values: {duplicate_prompt_ids}"
        )
    if len(prompt_ids) != EXPECTED_PROMPT_COUNT:
        raise ValueError(
            "campaign requires exactly "
            f"{EXPECTED_PROMPT_COUNT} baseline prompts, "
            f"found {len(prompt_ids)}"
        )

    actual_compressors = {episode["compressor"] for episode in episodes}
    expected_compressors = set(COMPRESSORS)
    if actual_compressors != expected_compressors:
        raise ValueError(
            "campaign compressors differ: expected "
            f"{sorted(expected_compressors)}, "
            f"found {sorted(actual_compressors)}"
        )

    actual_ratios = {float(episode["ratio"]) for episode in episodes}
    expected_ratios = set(EXPECTED_RATIOS)
    if actual_ratios != expected_ratios:
        raise ValueError(
            "campaign ratios differ: expected "
            f"{sorted(expected_ratios)}, found {sorted(actual_ratios)}"
        )

    for compressor in COMPRESSORS:
        count = sum(
            episode["compressor"] == compressor for episode in episodes
        )
        if count != EXPECTED_EPISODES_PER_COMPRESSOR:
            raise ValueError(
                f"compressor {compressor} requires "
                f"{EXPECTED_EPISODES_PER_COMPRESSOR} episodes, found {count}"
            )

    if len(episodes) != EXPECTED_EPISODE_COUNT:
        raise ValueError(
            f"campaign requires {EXPECTED_EPISODE_COUNT} episodes, "
            f"found {len(episodes)}"
        )

    expected_rows = set(
        itertools.product(prompt_ids, COMPRESSORS, EXPECTED_RATIOS)
    )
    actual_rows = [
        (
            episode["prompt_id"],
            episode["compressor"],
            float(episode["ratio"]),
        )
        for episode in episodes
    ]
    if len(set(actual_rows)) != len(actual_rows):
        raise ValueError(
            "campaign contains duplicate prompt/compressor/ratio rows"
        )
    if set(actual_rows) != expected_rows:
        missing = sorted(expected_rows - set(actual_rows))[:3]
        unexpected = sorted(set(actual_rows) - expected_rows)[:3]
        raise ValueError(
            "campaign prompt/compressor/ratio matrix is incomplete: "
            f"missing {missing}, unexpected {unexpected}"
        )

    return {row["prompt_id"]: row for row in baseline_rows}


def extract_reference_answer(text: str) -> str:
    answer_lines = [
        line for line in text.splitlines() if line.startswith("####")
    ]
    if len(answer_lines) != 1:
        raise ValueError(
            "reference must contain exactly one GSM8K #### answer delimiter"
        )
    answer_match = re.fullmatch(r"####\s+(.+)", answer_lines[0])
    if answer_match is None:
        raise ValueError("GSM8K #### answer delimiter must precede an answer")
    answer = answer_match.group(1).strip()
    if re.fullmatch(r"[+-]?\d[\d,]*(?:\.\d+)?", answer) is None:
        raise ValueError(f"GSM8K answer must be numeric, found {answer!r}")
    return answer


def build_evidence(repo_root: Path) -> dict[str, Any]:
    live_dir = repo_root / "results/live_controller_v3"
    all_episodes = load_jsonl(live_dir / "episodes.jsonl")
    baseline_rows = load_jsonl(live_dir / "baseline.jsonl")
    baselines = validate_campaign(all_episodes, baseline_rows)
    by_compressor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in all_episodes:
        by_compressor[episode["compressor"]].append(episode)

    reference = json.loads(
        (
            repo_root / "results/sweep/llama/gsm8k/references/gsm8k-0.json"
        ).read_text()
    )
    hybrid_rows = load_jsonl(
        repo_root
        / "results/sweep/llama/gsm8k/hybrids/streaming_llm__0.7500.jsonl"
    )
    matching_hybrids = [
        row
        for row in hybrid_rows
        if row["prompt_id"] == "gsm8k-0" and row["s"] == 128
    ]
    if len(matching_hybrids) != 1:
        raise ValueError(
            "expected exactly one hybrid row for prompt_id gsm8k-0 and "
            f"s 128, found {len(matching_hybrids)}"
        )
    hybrid = matching_hybrids[0]

    boxed = re.search(r"\\boxed\{([^}]+)\}", hybrid["text"])
    if boxed is None:
        raise ValueError(
            "missing boxed answer in selected compressed artifact"
        )
    if REFERENCE_SOURCE not in reference["text"].splitlines():
        raise ValueError(
            "selected reference excerpt is not an exact artifact line"
        )
    if COMPRESSED_SOURCE not in hybrid["text"].splitlines():
        raise ValueError(
            "selected compressed excerpt is not an exact artifact line"
        )

    reference_answer = extract_reference_answer(reference["text"])

    return {
        "campaign": {
            "episode_count": len(all_episodes),
            "prompt_count": len(baselines),
            "ratio_count": len(
                {float(episode["ratio"]) for episode in all_episodes}
            ),
            "compressor_count": len(COMPRESSORS),
            "compressors": {
                name: compressor_metrics(by_compressor[name], baselines)
                for name in COMPRESSORS
            },
        },
        "gsm8k_example": {
            "prompt_id": "gsm8k-0",
            "ratio": 0.75,
            "switch_position": 128,
            "reference_answer": reference_answer,
            "compressed_answer": boxed.group(1),
            "reference_quality": reference["q"],
            "compressed_quality": hybrid["q"],
            "reference_excerpt": REFERENCE_SOURCE,
            "compressed_excerpt": COMPRESSED_SOURCE,
        },
    }


def check_generated_evidence(repo_root: Path, output: Path) -> None:
    checked_in = json.loads(output.read_text())
    generated = build_evidence(repo_root)
    if checked_in != generated:
        raise ValueError(
            f"generated evidence is out of date: run {Path(__file__).name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when checked-in evidence differs from source artifacts",
    )
    args = parser.parse_args()
    script = Path(__file__).resolve()
    repo_root = script.parents[3]
    output = (
        repo_root / "showcase/herald-video/src/data/evidence.generated.json"
    )
    if args.check:
        check_generated_evidence(repo_root, output)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_evidence(repo_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
