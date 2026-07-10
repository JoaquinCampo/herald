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


def build_evidence(repo_root: Path) -> dict[str, Any]:
    live_dir = repo_root / "results/live_controller_v3"
    all_episodes = load_jsonl(live_dir / "episodes.jsonl")
    baselines = {
        row["prompt_id"]: row
        for row in load_jsonl(live_dir / "baseline.jsonl")
    }
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
    hybrid = next(
        row
        for row in hybrid_rows
        if row["prompt_id"] == "gsm8k-0" and row["s"] == 128
    )

    boxed = re.search(r"\\boxed\{([^}]+)\}", hybrid["text"])
    if boxed is None:
        raise ValueError(
            "missing boxed answer in selected compressed artifact"
        )
    if REFERENCE_SOURCE not in reference["text"]:
        raise ValueError(
            "selected reference excerpt is absent from the artifact"
        )
    if COMPRESSED_SOURCE not in hybrid["text"]:
        raise ValueError(
            "selected compressed excerpt is absent from the artifact"
        )

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
            "reference_answer": reference["text"]
            .rsplit("####", 1)[-1]
            .strip(),
            "compressed_answer": boxed.group(1),
            "reference_quality": reference["q"],
            "compressed_quality": hybrid["q"],
            "reference_excerpt": REFERENCE_SOURCE,
            "compressed_excerpt": COMPRESSED_SOURCE,
        },
    }


def main() -> None:
    script = Path(__file__).resolve()
    repo_root = script.parents[3]
    output = (
        repo_root / "showcase/herald-video/src/data/evidence.generated.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_evidence(repo_root), indent=2) + "\n")


if __name__ == "__main__":
    main()
