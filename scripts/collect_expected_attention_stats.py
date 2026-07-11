# pyright: reportMissingImports=false

"""Collect a train-only ExpectedAttentionStatsPress artifact.

The held-out prompt IDs use the same deterministic grouping rule as frozen
alarm export. Only the complementary train side contributes query vectors.
The output directory contains ``metadata.json`` and ``query_statistics.npz``.

Example:
    HF_HUB_OFFLINE=1 uv run python \
        scripts/collect_expected_attention_stats.py \
        --out-dir results/calibration/expected_attention_stats_ifeval_s0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from herald.config import TASKS  # noqa: E402
from herald.expected_attention_stats import (  # noqa: E402
    StatisticsArtifact,
    StatisticsMetadata,
    collect_query_moments,
    fingerprint_calibration_inputs,
)
from herald.generate import build_input_ids, load_model  # noqa: E402
from herald.switch_baselines import split_prompt_ids  # noqa: E402
from herald.tasks import load_prompts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect frozen train-only query statistics"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="llama")
    parser.add_argument("--task", default="ifeval")
    parser.add_argument("--prompts-per-task", type=int, default=200)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--test-group-fraction", type=float, default=0.25)
    parser.add_argument("--max-calibration-prompts", type=int, default=100)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--n-future-positions", type=int, default=512)
    parser.add_argument("--n-sink", type=int, default=4)
    parser.add_argument("--no-covariance", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task != "ifeval":
        raise ValueError(
            "only the frozen IFEval deployment path is supported"
        )
    if args.max_calibration_prompts < 1:
        raise ValueError("max_calibration_prompts must be positive")
    if args.max_prompt_tokens < 1:
        raise ValueError("max_prompt_tokens must be positive")
    if args.n_future_positions < 1:
        raise ValueError("n_future_positions must be positive")
    if args.n_sink < 0:
        raise ValueError("n_sink must be non-negative")

    records = load_prompts(
        args.task,
        args.prompts_per_task,
        TASKS[args.task],
    )
    train_ids, test_ids = split_prompt_ids(
        [record.prompt_id for record in records],
        model=args.model_key,
        task=args.task,
        seed=args.split_seed,
        test_group_fraction=args.test_group_fraction,
    )
    calibration_ids = train_ids[: args.max_calibration_prompts]
    records_by_id = {record.prompt_id: record for record in records}

    lm = load_model(args.model_key, dtype=args.dtype, device=args.device)
    inputs = []
    for prompt_id in calibration_ids:
        token_ids = build_input_ids(lm, records_by_id[prompt_id])[
            : args.max_prompt_tokens
        ]
        if token_ids.numel() <= args.n_sink:
            raise ValueError(
                f"prompt {prompt_id!r} has no non-sink calibration tokens"
            )
        inputs.append(token_ids.cpu())

    mu, cov, query_token_count = collect_query_moments(
        lm.model,
        inputs,
        n_sink=args.n_sink,
    )
    config = lm.model.config
    metadata = StatisticsMetadata(
        model_id=str(config.name_or_path),
        model_type=str(config.model_type),
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        n_future_positions=args.n_future_positions,
        n_sink=args.n_sink,
        use_covariance=not args.no_covariance,
        calibration_task=args.task,
        calibration_prompt_ids=calibration_ids,
        excluded_test_prompt_ids=test_ids,
        max_prompt_tokens=args.max_prompt_tokens,
        query_token_count=query_token_count,
        calibration_input_sha256=fingerprint_calibration_inputs(
            calibration_ids,
            inputs,
        ),
    )
    artifact = StatisticsArtifact(metadata=metadata, mu=mu, cov=cov)
    digest = artifact.save(args.out_dir)
    print(
        json.dumps(
            {
                "event": "expected_attention_stats_saved",
                "out_dir": str(args.out_dir),
                "artifact_sha256": digest,
                "n_calibration_prompts": len(calibration_ids),
                "n_excluded_test_prompts": len(test_ids),
                "query_token_count": query_token_count,
                "calibration_input_sha256": (
                    metadata.calibration_input_sha256
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
