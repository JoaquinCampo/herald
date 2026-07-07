"""Quantify oracle headroom from per-prompt compression ratio choice.

This is a zero-GPU study over recorded one-shot switch rows. The episode
currencies are:

* legacy savings: ``1 - s / L``.
* end-state bytes saved fraction: ``r * (P + s) / (P + L)``.
* peak-cache reduction:
  ``1 - max(P + s, (1 - r) * (P + s) + (L - s)) / (P + L)``.
* time-averaged cache reduction:
  ``1 - mean(compressed_cache_t) / mean(uncompressed_cache_t)`` over
  decode steps ``t = 1..L``. Without compression the cache after step
  ``t`` is ``P + t``. With a switch at ``s``, steps ``t <= s`` use
  ``P + t`` and steps ``t > s`` use
  ``(1 - r) * (P + s) + (t - s)``. Therefore the total uncompressed
  cache is ``L * P + L * (L + 1) / 2`` and the compressed total is the
  uncompressed total minus ``r * (P + s) * (L - s)``.

Never-commit episodes save 0 in every currency.
"""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from herald.switch_baselines import leave_one_compressor_splits

TASK = "ifeval"
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
RATIOS = (0.25, 0.5, 0.75, 0.875)
CONSTRAINTS = {"strict": 0.0, "lenient": 0.03}
CURRENCIES = (
    "legacy_savings",
    "end_state_bytes_saved",
    "peak_cache_reduction",
    "time_avg_cache_reduction",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("results/predictor/switch_dataset_attn.parquet"),
    )
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("results/sweep/llama/ifeval/references"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/ratio_headroom"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-group-fraction", type=float, default=0.25)
    args = parser.parse_args()

    rows = _load_rows(args.dataset)
    prompt_lengths = _load_prompt_lengths(args.reference_dir)
    filtered = _prepare_rows(rows, prompt_lengths)
    test_prompt_ids = _test_prompt_ids(
        filtered,
        seed=args.seed,
        test_group_fraction=args.test_group_fraction,
    )
    test_rows = [
        row for row in filtered if str(row["prompt_id"]) in test_prompt_ids
    ]

    summary = {
        "config": {
            "dataset": str(args.dataset),
            "reference_dir": str(args.reference_dir),
            "task": TASK,
            "compressors": list(COMPRESSORS),
            "ratios": list(RATIOS),
            "constraints": CONSTRAINTS,
            "currencies": list(CURRENCIES),
            "seed": args.seed,
            "test_group_fraction": args.test_group_fraction,
        },
        "prompt_length_source": {
            "source": "reference_json_prompt_input_ids",
            "path": str(args.reference_dir),
            "field": "prompt_input_ids",
            "value": "len(prompt_input_ids)",
        },
        "inventory": _inventory(filtered, test_rows, test_prompt_ids),
        "all_prompts": evaluate_scope(filtered),
        "split0_test_subset": evaluate_scope(test_rows),
    }
    summary["sanity_reproduction"] = {
        compressor: summary["split0_test_subset"]["strict"]["legacy_savings"][
            compressor
        ]["mixed"]["value"]
        for compressor in COMPRESSORS
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "summary.json"
    tables_path = args.out_dir / "tables.csv"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_tables(tables_path, summary)
    print(f"summary -> {summary_path}")
    print(f"tables -> {tables_path}")


def evaluate_scope(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for constraint_name, max_dq in CONSTRAINTS.items():
        out[constraint_name] = {}
        for currency in CURRENCIES:
            out[constraint_name][currency] = {}
            for compressor in COMPRESSORS:
                compressor_rows = [
                    row for row in rows if row["compressor"] == compressor
                ]
                out[constraint_name][currency][compressor] = (
                    evaluate_compressor(
                        compressor_rows,
                        currency=currency,
                        max_dq=max_dq,
                    )
                )
    return out


def evaluate_compressor(
    rows: Sequence[dict[str, Any]], *, currency: str, max_dq: float
) -> dict[str, Any]:
    fixed = {
        str(ratio): fixed_ratio_oracle(
            rows,
            ratio=ratio,
            currency=currency,
            max_dq=max_dq,
        )
        for ratio in RATIOS
    }
    best_fixed_ratio, best_fixed_value = max(
        fixed.items(), key=lambda item: (item[1], float(item[0]))
    )
    mixed = mixed_ratio_oracle(rows, currency=currency, max_dq=max_dq)
    joint_value, joint_counts = joint_oracle(
        rows, currency=currency, max_dq=max_dq
    )
    n_prompts = len(_prompt_ids(rows))
    return {
        "fixed_by_ratio": fixed,
        "best_fixed": {
            "ratio": float(best_fixed_ratio),
            "value": best_fixed_value,
        },
        "mixed": {"value": mixed},
        "joint": {"value": joint_value},
        "headroom": joint_value - best_fixed_value,
        "joint_ratio_distribution": _ratio_distribution(
            joint_counts, n_prompts
        ),
    }


def fixed_ratio_oracle(
    rows: Sequence[dict[str, Any]],
    *,
    ratio: float,
    currency: str,
    max_dq: float,
) -> float:
    groups = _group_by_prompt(
        row for row in rows if math.isclose(float(row["ratio"]), ratio)
    )
    values = [_best_group_value(group, currency, max_dq) for group in groups]
    return _mean(values)


def mixed_ratio_oracle(
    rows: Sequence[dict[str, Any]], *, currency: str, max_dq: float
) -> float:
    groups = _group_by_prompt_ratio(rows)
    values = [_best_group_value(group, currency, max_dq) for group in groups]
    return _mean(values)


def joint_oracle(
    rows: Sequence[dict[str, Any]], *, currency: str, max_dq: float
) -> tuple[float, Counter[float | None]]:
    values: list[float] = []
    ratios: Counter[float | None] = Counter()
    for group in _group_by_prompt(rows):
        feasible = [row for row in group if float(row["dq"]) <= max_dq]
        if not feasible:
            values.append(0.0)
            ratios[None] += 1
            continue
        best = max(
            feasible,
            key=lambda row: (
                float(row[currency]),
                _ratio_rank(float(row["ratio"])),
                -float(row["s"]),
            ),
        )
        values.append(float(best[currency]))
        ratios[float(best["ratio"])] += 1
    return _mean(values), ratios


def _best_group_value(
    rows: Sequence[dict[str, Any]], currency: str, max_dq: float
) -> float:
    feasible = [float(row[currency]) for row in rows if row["dq"] <= max_dq]
    if not feasible:
        return 0.0
    return max(feasible)


def _prepare_rows(
    rows: Sequence[dict[str, Any]], prompt_lengths: Mapping[str, int]
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if row.get("task") != TASK:
            continue
        if row.get("compressor") not in COMPRESSORS:
            continue
        ratio = float(row["ratio"])
        if ratio not in RATIOS:
            continue
        prompt_id = str(row["prompt_id"])
        if prompt_id not in prompt_lengths:
            raise ValueError(f"missing prompt length for {prompt_id}")
        out = dict(row)
        p = prompt_lengths[prompt_id]
        ref_len = _positive_int(out["ref_len"], "ref_len")
        s = _nonnegative_int(out["s"], "s")
        if s > ref_len:
            raise ValueError(
                f"switch position exceeds ref_len for {prompt_id}"
            )
        out["prompt_len"] = p
        out.update(_currencies(p=p, ref_len=ref_len, s=s, ratio=ratio))
        filtered.append(out)
    _validate_inventory(filtered)
    return filtered


def _currencies(
    p: int, ref_len: int, s: int, ratio: float
) -> dict[str, float]:
    denominator = p + ref_len
    legacy = 1.0 - (s / ref_len)
    end_state = ratio * (p + s) / denominator
    peak_after = (1.0 - ratio) * (p + s) + (ref_len - s)
    peak = 1.0 - max(p + s, peak_after) / denominator
    uncomp_sum = ref_len * p + ref_len * (ref_len + 1) / 2.0
    saved_sum = ratio * (p + s) * (ref_len - s)
    time_avg = saved_sum / uncomp_sum
    return {
        "legacy_savings": legacy,
        "end_state_bytes_saved": end_state,
        "peak_cache_reduction": peak,
        "time_avg_cache_reduction": time_avg,
    }


def _load_rows(path: Path) -> list[dict[str, Any]]:
    parquet = import_module("pyarrow.parquet")
    read_table = cast(Any, parquet.read_table)
    rows = cast(Any, read_table(path)).to_pylist()
    if not isinstance(rows, list):
        raise TypeError("expected parquet rows to be a list")
    return cast(list[dict[str, Any]], rows)


def _load_prompt_lengths(reference_dir: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    for path in sorted(reference_dir.glob("*.json")):
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError(f"reference must be an object: {path}")
        prompt_id = str(value.get("prompt_id"))
        prompt_ids = value.get("prompt_input_ids")
        if not isinstance(prompt_ids, list):
            raise ValueError(f"missing prompt_input_ids list in {path}")
        lengths[prompt_id] = len(prompt_ids)
    if not lengths:
        raise ValueError(f"no reference JSON files found in {reference_dir}")
    return lengths


def _test_prompt_ids(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    test_group_fraction: float,
) -> set[str]:
    splits = leave_one_compressor_splits(
        list(rows),
        compressors=COMPRESSORS,
        seed=seed,
        test_group_fraction=test_group_fraction,
    )
    prompt_ids = {
        str(row["prompt_id"]) for split in splits for row in split.test
    }
    if len(prompt_ids) != 46:
        raise ValueError(
            f"expected 46 split-0 test prompts, got {len(prompt_ids)}"
        )
    return prompt_ids


def _inventory(
    rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    test_prompt_ids: set[str],
) -> dict[str, Any]:
    return {
        "all_rows": len(rows),
        "all_prompts": len(_prompt_ids(rows)),
        "split0_test_rows": len(test_rows),
        "split0_test_prompts": len(test_prompt_ids),
        "rows_by_compressor": dict(
            sorted(Counter(str(row["compressor"]) for row in rows).items())
        ),
    }


def _validate_inventory(rows: Sequence[dict[str, Any]]) -> None:
    compressors = {str(row["compressor"]) for row in rows}
    ratios = {float(row["ratio"]) for row in rows}
    if compressors != set(COMPRESSORS):
        raise ValueError(f"unexpected compressors: {sorted(compressors)}")
    if ratios != set(RATIOS):
        raise ValueError(f"unexpected ratios: {sorted(ratios)}")


def _group_by_prompt(
    rows: Iterable[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["prompt_id"])].append(row)
    return [groups[key] for key in sorted(groups)]


def _group_by_prompt_ratio(
    rows: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["prompt_id"]), float(row["ratio"]))].append(row)
    return [groups[key] for key in sorted(groups)]


def _prompt_ids(rows: Sequence[dict[str, Any]]) -> set[str]:
    return {str(row["prompt_id"]) for row in rows}


def _ratio_distribution(
    counts: Counter[float | None], n_prompts: int
) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for ratio in [*RATIOS, None]:
        label = "never" if ratio is None else str(ratio)
        count = counts.get(ratio, 0)
        out[label] = {
            "count": count,
            "share": 0.0 if n_prompts == 0 else count / n_prompts,
        }
    return out


def _ratio_rank(ratio: float) -> int:
    return RATIOS.index(ratio)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return sum(values) / len(values)


def _positive_int(value: object, name: str) -> int:
    out = _nonnegative_int(value, name)
    if out <= 0:
        raise ValueError(f"{name} must be positive")
    return out


def _nonnegative_int(value: object, name: str) -> int:
    out = int(value)  # type: ignore[arg-type]
    if out < 0:
        raise ValueError(f"{name} must be nonnegative")
    return out


def _write_tables(path: Path, summary: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for scope in ("all_prompts", "split0_test_subset"):
        scope_summary = cast(Mapping[str, Any], summary[scope])
        for constraint in CONSTRAINTS:
            constraint_summary = cast(
                Mapping[str, Any], scope_summary[constraint]
            )
            for currency in CURRENCIES:
                currency_summary = cast(
                    Mapping[str, Any], constraint_summary[currency]
                )
                for compressor in COMPRESSORS:
                    metrics = cast(
                        Mapping[str, Any], currency_summary[compressor]
                    )
                    best_fixed = cast(
                        Mapping[str, Any], metrics["best_fixed"]
                    )
                    joint = cast(Mapping[str, Any], metrics["joint"])
                    rows.append(
                        {
                            "scope": scope,
                            "constraint": constraint,
                            "currency": currency,
                            "compressor": compressor,
                            "best_fixed_ratio": best_fixed["ratio"],
                            "best_fixed_value": best_fixed["value"],
                            "joint_value": joint["value"],
                            "headroom": metrics["headroom"],
                            "mixed_value": cast(
                                Mapping[str, Any], metrics["mixed"]
                            )["value"],
                        }
                    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
