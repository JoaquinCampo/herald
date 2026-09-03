# pyright: reportMissingImports=false

"""Run bounded HERALD cache-intervention parity checks.

The default is one task prompt, one ratio, one switch, and all three approved
compressors.  A non-passing or incomplete case is emitted as JSON and causes a
non-zero exit status.
"""

import argparse
import json
import math
from pathlib import Path

from herald.config import MODELS, TASKS
from herald.generate import load_model
from herald.parity import APPROVED_COMPRESSORS, run_parity_case
from herald.scoring import score
from herald.tasks import load_prompts


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _ratios(value: str) -> list[float]:
    try:
        ratios = [float(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "ratios must be comma-separated numbers"
        ) from error
    if not ratios or any(
        not math.isfinite(ratio) or ratio <= 0 or ratio >= 1
        for ratio in ratios
    ):
        raise argparse.ArgumentTypeError("ratios must satisfy 0 < ratio < 1")
    return ratios


def _ints(value: str) -> list[int]:
    try:
        positions = [int(item) for item in _csv(value)]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "switch positions must be comma-separated integers"
        ) from error
    if not positions or any(position < 0 for position in positions):
        raise argparse.ArgumentTypeError(
            "switch positions must be non-negative"
        )
    return positions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HERALD parity pilot")
    parser.add_argument("--model", choices=sorted(MODELS), default="llama")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--task", choices=sorted(TASKS), default="ifeval")
    parser.add_argument(
        "--prompt-ids",
        type=_csv,
        default=None,
        help="comma-separated IDs; otherwise use the first --prompts records",
    )
    parser.add_argument("--prompts", type=int, default=1)
    parser.add_argument(
        "--compressors",
        type=_csv,
        default=list(APPROVED_COMPRESSORS),
    )
    parser.add_argument("--ratios", type=_ratios, default=[0.5])
    parser.add_argument("--switch-positions", type=_ints, default=[0, 16])
    parser.add_argument("--continuation-budget", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-score", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _prompt_count(args: argparse.Namespace) -> int:
    if not args.prompt_ids:
        return int(args.prompts)
    numeric = []
    for prompt_id in args.prompt_ids:
        suffix = prompt_id.rsplit("-", 1)[-1]
        if suffix.isdigit():
            numeric.append(int(suffix) + 1)
    return max([int(args.prompts), *numeric])


def _write_payload(
    args: argparse.Namespace, payload: dict[str, object]
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.write_text(rendered + "\n", encoding="utf-8")


def run_cli(args: argparse.Namespace) -> int:
    if args.prompts <= 0 or args.continuation_budget <= 0:
        raise ValueError("prompts and continuation budget must be positive")
    unknown = sorted(set(args.compressors) - set(APPROVED_COMPRESSORS))
    if unknown:
        raise ValueError(f"unsupported compressors: {unknown}")
    if not args.ratios or any(
        not math.isfinite(ratio) or ratio <= 0 or ratio >= 1
        for ratio in args.ratios
    ):
        raise ValueError("ratios must satisfy 0 < ratio < 1")
    if not args.switch_positions or any(
        switch < 0 or switch >= args.continuation_budget
        for switch in args.switch_positions
    ):
        raise ValueError(
            "switch positions must satisfy 0 <= s < continuation budget"
        )
    records = load_prompts(args.task, _prompt_count(args))
    if args.prompt_ids:
        wanted = set(args.prompt_ids)
        records = [record for record in records if record.prompt_id in wanted]
        if len(records) != len(wanted):
            found = {record.prompt_id for record in records}
            missing = sorted(wanted - found)
            raise ValueError(
                f"requested prompt IDs were not loaded: {missing}"
            )
    if not records:
        raise ValueError("prompt selection is empty")
    selected_model_id = args.model_id or MODELS[args.model]
    lm = load_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
        attn_implementation=args.attn,
        model_id=args.model_id,
    )
    scorer = (
        None
        if args.no_score
        else lambda task, text, gold: score(task, text, gold)
    )
    reports = []
    for record in records:
        for compressor in args.compressors:
            for ratio in args.ratios:
                for switch in args.switch_positions:
                    report = run_parity_case(
                        lm,
                        record,
                        compressor=compressor,
                        ratio=ratio,
                        s=switch,
                        continuation_budget=args.continuation_budget,
                        scorer=scorer,
                        seed=args.seed,
                        model_id=selected_model_id,
                    )
                    reports.append(report.to_dict())
    payload = {
        "schema_version": "herald.parity-pilot.v1",
        "cases": reports,
        "passed": bool(reports)
        and all(
            report["complete"] and report["passed"] for report in reports
        ),
    }
    _write_payload(args, payload)
    return 0 if payload["passed"] else 2


def main() -> None:
    args = build_parser().parse_args()
    try:
        status = run_cli(args)
    except Exception as error:
        _write_payload(
            args,
            {
                "schema_version": "herald.parity-pilot.v1",
                "cases": [],
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        status = 2
    raise SystemExit(status)


if __name__ == "__main__":
    main()
