"""Dev-only prompt-semantics and control-outcome audit for E1."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from datasets import load_dataset  # type: ignore[import-untyped]

from herald.magnitude import KNOWN_COMPRESSORS
from herald.magnitude_v2 import hash_prompt_ids, sha256_file

RATIOS = (0.25, 0.5, 0.75, 0.875)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sign_counts(values: pd.Series) -> dict[str, int]:
    array = values.to_numpy(dtype=np.float64)
    return {
        "negative": int((array < -1e-12).sum()),
        "zero": int((np.abs(array) <= 1e-12).sum()),
        "positive": int((array > 1e-12).sum()),
        "major": int((array >= 0.5 - 1e-12).sum()),
    }


def eta_squared(frame: pd.DataFrame, group: str, target: str) -> float:
    values = frame[target].to_numpy(dtype=np.float64)
    total = float(np.sum((values - np.mean(values)) ** 2))
    if total == 0.0:
        return 0.0
    means = frame.groupby(group, observed=True)[target].transform("mean")
    between = float(
        np.sum((means.to_numpy(dtype=np.float64) - np.mean(values)) ** 2)
    )
    return between / total


def group_mse(frame: pd.DataFrame, columns: list[str]) -> float:
    prediction = frame.groupby(columns, observed=True)["dq"].transform("mean")
    residual = frame["dq"].to_numpy(dtype=np.float64) - prediction.to_numpy(
        dtype=np.float64
    )
    return float(np.mean(residual**2))


def load_prompt_metadata(development: tuple[str, ...]) -> pd.DataFrame:
    dataset = load_dataset("google/IFEval", split="train")
    keys = [int(value.as_py()) for value in dataset.data.column("key")]
    index_by_key = {key: index for index, key in enumerate(keys)}
    if len(index_by_key) != len(keys):
        raise ValueError("IFEval dataset keys are not unique")
    dev_keys = [
        int(prompt_id.removeprefix("ifeval-")) for prompt_id in development
    ]
    missing = sorted(set(dev_keys) - set(index_by_key))
    if missing:
        raise ValueError(f"development keys absent from IFEval: {missing}")
    subset = dataset.select([index_by_key[key] for key in dev_keys])
    records: list[dict[str, Any]] = []
    for example in subset:
        key = int(example["key"])
        prompt_id = f"ifeval-{key}"
        if prompt_id not in development:
            raise ValueError("non-development prompt entered metadata subset")
        prompt = str(example["prompt"])
        instruction_ids = tuple(
            str(value) for value in example["instruction_id_list"]
        )
        kwargs = list(example["kwargs"])
        categories = tuple(
            sorted({value.split(":", 1)[0] for value in instruction_ids})
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "prompt_chars": len(prompt),
                "prompt_words": len(prompt.split()),
                "prompt_lines": len(prompt.splitlines()),
                "instruction_count": len(instruction_ids),
                "instruction_ids": instruction_ids,
                "instruction_categories": categories,
                "instruction_kwargs_nonnull": sum(
                    value is not None
                    for item in kwargs
                    for value in dict(item).values()
                ),
            }
        )
    frame = pd.DataFrame(records)
    if set(frame["prompt_id"]) != set(development) or len(frame) != 154:
        raise ValueError("prompt metadata does not match development roster")
    return frame


def load_dev_outcomes(
    base: Path, development: tuple[str, ...]
) -> pd.DataFrame:
    columns = [
        "prompt_id",
        "compressor",
        "ratio",
        "s",
        "q_ref",
        "q_control",
        "q_hybrid",
        "dq",
    ]
    task_field = ds.field("task")  # type: ignore[attr-defined, no-untyped-call]
    prompt_field = ds.field(  # type: ignore[attr-defined, no-untyped-call]
        "prompt_id"
    )
    compressor_field = ds.field(  # type: ignore[attr-defined, no-untyped-call]
        "compressor"
    )
    predicate = (task_field == "ifeval") & prompt_field.isin(
        list(development)
    )
    predicate = predicate & compressor_field.isin(list(KNOWN_COMPRESSORS))
    dataset = ds.dataset(  # type: ignore[no-untyped-call]
        base, format="parquet"
    )
    frame = dataset.to_table(
        columns=columns,
        filter=predicate,
    ).to_pandas()
    if set(frame["prompt_id"]) != set(development):
        raise ValueError(
            "outcome scanner did not recover exact development roster"
        )
    return frame


def prompt_macro(outcomes: pd.DataFrame) -> pd.DataFrame:
    trajectory = outcomes.groupby(
        ["compressor", "prompt_id", "ratio"], observed=True, as_index=False
    ).agg(
        dq=("dq", "mean"),
        positive_rate=("dq", lambda values: float((values > 1e-12).mean())),
        major_rate=(
            "dq",
            lambda values: float((values >= 0.5 - 1e-12).mean()),
        ),
        q_control=("q_control", "mean"),
        q_ref=("q_ref", "mean"),
    )
    return trajectory.groupby(
        ["compressor", "prompt_id"], observed=True, as_index=False
    ).agg(
        macro_dq=("dq", "mean"),
        macro_positive_rate=("positive_rate", "mean"),
        macro_major_rate=("major_rate", "mean"),
        mean_q_control=("q_control", "mean"),
        mean_q_ref=("q_ref", "mean"),
    )


def instruction_summary(joined: pd.DataFrame) -> dict[str, Any]:
    exploded = joined.explode("instruction_ids")
    output: dict[str, Any] = {}
    for compressor, group in exploded.groupby("compressor", sort=True):
        rows: list[dict[str, Any]] = []
        for instruction_id, values in group.groupby(
            "instruction_ids", sort=True
        ):
            rows.append(
                {
                    "instruction_id": str(instruction_id),
                    "prompts": int(values["prompt_id"].nunique()),
                    "mean_macro_dq": float(values["macro_dq"].mean()),
                    "mean_positive_rate": float(
                        values["macro_positive_rate"].mean()
                    ),
                    "mean_major_rate": float(
                        values["macro_major_rate"].mean()
                    ),
                    "mean_control_quality": float(
                        values["mean_q_control"].mean()
                    ),
                }
            )
        output[str(compressor)] = sorted(
            rows,
            key=lambda row: (
                -int(row["prompts"]),
                str(row["instruction_id"]),
            ),
        )
    return output


def category_summary(joined: pd.DataFrame) -> dict[str, Any]:
    exploded = joined.explode("instruction_categories")
    output: dict[str, Any] = {}
    for compressor, group in exploded.groupby("compressor", sort=True):
        rows: list[dict[str, Any]] = []
        for category, values in group.groupby(
            "instruction_categories", sort=True
        ):
            rows.append(
                {
                    "category": str(category),
                    "prompts": int(values["prompt_id"].nunique()),
                    "mean_macro_dq": float(values["macro_dq"].mean()),
                    "mean_positive_rate": float(
                        values["macro_positive_rate"].mean()
                    ),
                    "mean_major_rate": float(
                        values["macro_major_rate"].mean()
                    ),
                    "mean_control_quality": float(
                        values["mean_q_control"].mean()
                    ),
                }
            )
        output[str(compressor)] = rows
    return output


def control_geometry(outcomes: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for compressor, group in outcomes.groupby("compressor", sort=True):
        rows: list[dict[str, Any]] = []
        for q_control, values in group.groupby("q_control", sort=True):
            rows.append(
                {
                    "q_control": float(q_control),
                    "rows": len(values),
                    "prompts": int(values["prompt_id"].nunique()),
                    "mean_dq": float(values["dq"].mean()),
                    **sign_counts(values["dq"]),
                }
            )
        total_mse = float(np.mean((group["dq"] - group["dq"].mean()) ** 2))
        output[str(compressor)] = {
            "q_control_support": rows,
            "dq_q_control_pearson": float(
                group[["dq", "q_control"]].corr().iloc[0, 1]
            ),
            "total_mse_about_mean": total_mse,
            "forbidden_oracle_mse_q_control": group_mse(group, ["q_control"]),
            "forbidden_oracle_mse_q_control_ratio": group_mse(
                group, ["q_control", "ratio"]
            ),
            "forbidden_oracle_mse_q_control_ratio_s": group_mse(
                group, ["q_control", "ratio", "s"]
            ),
            "warning": (
                "q_control/q_ref are post-generation outcomes and forbidden "
                "inference inputs; these are estimand diagnostics only."
            ),
        }
    return output


def prompt_feature_summary(joined: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for compressor, group in joined.groupby("compressor", sort=True):
        correlations = {
            column: float(
                group[[column, "macro_dq"]].corr(method="spearman").iloc[0, 1]
            )
            for column in (
                "prompt_chars",
                "prompt_words",
                "prompt_lines",
                "instruction_count",
                "instruction_kwargs_nonnull",
            )
        }
        by_instruction_count = [
            {
                "instruction_count": int(count),
                "prompts": len(values),
                "mean_macro_dq": float(values["macro_dq"].mean()),
                "std_macro_dq": float(values["macro_dq"].std(ddof=0)),
                "mean_control_quality": float(
                    values["mean_q_control"].mean()
                ),
            }
            for count, values in group.groupby("instruction_count", sort=True)
        ]
        output[str(compressor)] = {
            "spearman_with_macro_dq": correlations,
            "instruction_count_groups": by_instruction_count,
            "eta_squared_instruction_count": eta_squared(
                group, "instruction_count", "macro_dq"
            ),
        }
    return output


def main() -> None:
    args = parse_args()
    evidence = load_json(args.evidence)
    development = tuple(
        str(value) for value in evidence["split"]["train_prompt_ids"]
    )
    quarantine = tuple(
        str(value) for value in evidence["split"]["test_prompt_ids"]
    )
    if len(development) != 154 or len(quarantine) != 46:
        raise ValueError("unexpected prompt roster")
    if set(development) & set(quarantine):
        raise ValueError("development and quarantine overlap")

    metadata = load_prompt_metadata(development)
    outcomes = load_dev_outcomes(args.base, development)
    macro = prompt_macro(outcomes)
    joined = macro.merge(metadata, on="prompt_id", validate="many_to_one")
    if set(joined["prompt_id"]) & set(quarantine):
        raise ValueError("protected prompt entered semantic audit")

    instruction_counts = Counter(
        instruction_id
        for values in metadata["instruction_ids"]
        for instruction_id in values
    )
    category_counts = Counter(
        category
        for values in metadata["instruction_categories"]
        for category in values
    )
    output = {
        "schema_version": "herald.magnitude_e1_semantics_audit.v1",
        "status": "descriptive_development_only",
        "safety": {
            "development_prompts": len(development),
            "development_prompt_ids_sha256": hash_prompt_ids(development),
            "quarantined_prompts": len(quarantine),
            "protected_prompt_rows_materialized": False,
            "key_column_selected_before_development_row_subset": True,
        },
        "input_sha256": {
            "base": sha256_file(args.base),
            "evidence": sha256_file(args.evidence),
        },
        "prompt_metadata": {
            "rows": len(metadata),
            "instruction_count_distribution": {
                str(key): int(value)
                for key, value in metadata["instruction_count"]
                .value_counts()
                .sort_index()
                .items()
            },
            "unique_instruction_ids": len(instruction_counts),
            "instruction_id_prompt_counts": dict(
                sorted(instruction_counts.items())
            ),
            "category_prompt_counts": dict(sorted(category_counts.items())),
            "prompt_chars": {
                "min": int(metadata["prompt_chars"].min()),
                "median": float(metadata["prompt_chars"].median()),
                "max": int(metadata["prompt_chars"].max()),
            },
            "representative_metadata": [
                {
                    "prompt_id": str(row.prompt_id),
                    "instruction_count": int(row.instruction_count),
                    "instruction_ids": list(row.instruction_ids),
                    "prompt_chars": int(row.prompt_chars),
                    "prompt_words": int(row.prompt_words),
                    "prompt_lines": int(row.prompt_lines),
                }
                for row in metadata.sort_values("prompt_id")
                .head(10)
                .itertuples()
            ],
        },
        "control_geometry": control_geometry(outcomes),
        "prompt_feature_relationships": prompt_feature_summary(joined),
        "instruction_relationships": instruction_summary(joined),
        "category_relationships": category_summary(joined),
        "interpretation_limits": [
            "IFEval gold instruction IDs and kwargs are benchmark "
            "annotations, not ordinary deployment inputs.",
            "Prompt text and lexical summaries are available before "
            "generation; annotation-derived features are "
            "deployment-privileged unless supplied by the caller.",
            "All relationships are adaptive, descriptive, and computed on "
            "154 reused development prompts.",
            "q_control and q_ref are post-generation outcomes and never "
            "candidate inputs.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "prompts": len(metadata),
                "joined_rows": len(joined),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
