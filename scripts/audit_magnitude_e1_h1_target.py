"""Audit reliability and fold geometry of the H1 prompt-macro target."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

KEY_COLUMNS = ("prompt_id", "compressor", "ratio", "s")
SOURCE_COLUMNS = (*KEY_COLUMNS, "dq", "fold")
COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
RATIOS = (0.25, 0.5, 0.75, 0.875)
TOTAL_ROWS = 45180
PROMPTS = 154
SCHEMA_VERSION = "herald.magnitude_e1_h1_target_audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        str(quantile): float(np.quantile(values, quantile))
        for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def prompt_eta_squared(frame: pd.DataFrame) -> float:
    values = frame["dq"].to_numpy(dtype=np.float64)
    grand_mean = float(np.mean(values))
    total = float(np.sum((values - grand_mean) ** 2))
    means = frame.groupby("prompt_id", sort=True)["dq"].transform("mean")
    between = float(
        np.sum((means.to_numpy(dtype=np.float64) - grand_mean) ** 2)
    )
    return between / total if total else 0.0


def fold_mean_baseline(
    prompt_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, Any]]:
    predictions = np.empty(len(prompt_frame), dtype=np.float64)
    fold_details = {}
    folds = prompt_frame["fold"].to_numpy(dtype=np.int64)
    targets = prompt_frame["macro_dq"].to_numpy(dtype=np.float64)
    for fold in sorted(int(value) for value in np.unique(folds)):
        training = folds != fold
        validation = folds == fold
        training_mean = float(np.mean(targets[training]))
        predictions[validation] = training_mean
        fold_targets = targets[validation]
        fold_details[str(fold)] = {
            "prompts": int(np.sum(validation)),
            "training_mean": training_mean,
            "validation_mean": float(np.mean(fold_targets)),
            "validation_std": float(np.std(fold_targets, ddof=1)),
        }
    residual = targets - predictions
    return predictions, {
        "mse": float(np.mean(residual**2)),
        "mae": float(np.mean(np.abs(residual))),
        "prediction_mean": float(np.mean(predictions)),
        "folds": fold_details,
    }


def ratio_correlation(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    table = (
        frame.groupby(["prompt_id", "ratio"], sort=True)["dq"]
        .mean()
        .unstack("ratio")
        .reindex(columns=list(RATIOS))
    )
    if table.shape != (PROMPTS, len(RATIOS)):
        raise ValueError("prompt-ratio target table is incomplete")
    return {
        str(left): {
            str(right): correlation(
                table[left].to_numpy(dtype=np.float64),
                table[right].to_numpy(dtype=np.float64),
            )
            for right in RATIOS
        }
        for left in RATIOS
    }


def state_split_reliability(frame: pd.DataFrame) -> dict[str, float | int]:
    positions = {
        int(value): index
        for index, value in enumerate(sorted(frame["s"].unique()))
    }
    state_rank = frame["s"].map(positions)
    if state_rank.isna().any():
        raise ValueError("state rank mapping is incomplete")
    even = (
        frame[state_rank % 2 == 0]
        .groupby("prompt_id", sort=True)["dq"]
        .mean()
    )
    odd = (
        frame[state_rank % 2 == 1]
        .groupby("prompt_id", sort=True)["dq"]
        .mean()
    )
    joined = pd.concat(
        [even.rename("even"), odd.rename("odd")], axis=1
    ).dropna()
    if len(joined) < 2:
        raise ValueError("alternating-state reliability split is too small")
    return {
        "prompts_with_both_state_halves": len(joined),
        "single_state_prompts_excluded": PROMPTS - len(joined),
        "pearson": correlation(
            joined["even"].to_numpy(dtype=np.float64),
            joined["odd"].to_numpy(dtype=np.float64),
        ),
        "mean_absolute_difference": float(
            np.mean(np.abs(joined["even"] - joined["odd"]))
        ),
    }


def compressor_audit(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fold_counts = frame.groupby("prompt_id")["fold"].nunique()
    if int((fold_counts != 1).sum()):
        raise ValueError("a prompt appears in multiple folds")
    macro = (
        frame.groupby("prompt_id", sort=True)
        .agg(macro_dq=("dq", "mean"), fold=("fold", "first"))
        .reset_index()
    )
    if len(macro) != PROMPTS:
        raise ValueError("prompt-macro target table is incomplete")
    values = macro["macro_dq"].to_numpy(dtype=np.float64)
    _predictions, baseline = fold_mean_baseline(macro)
    result = {
        "prompt_macro": {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "quantiles": quantiles(values),
            "exact_zero_prompts": int(np.sum(np.abs(values) <= 1e-15)),
        },
        "prompt_eta_squared_on_rows": prompt_eta_squared(frame),
        "fold_mean_baseline": baseline,
        "state_even_odd_reliability": state_split_reliability(frame),
        "ratio_prompt_correlations": ratio_correlation(frame),
    }
    return macro, result


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite H1 target audit: {args.output}"
        )
    table = pq.read_table(  # type: ignore[no-untyped-call]
        args.source_oof,
        columns=list(SOURCE_COLUMNS),
    )
    if tuple(table.column_names) != SOURCE_COLUMNS:
        raise ValueError("target audit loaded unexpected source columns")
    frame = table.to_pandas()
    if (
        len(frame) != TOTAL_ROWS
        or frame[list(KEY_COLUMNS)].duplicated().any()
        or set(frame["compressor"]) != set(COMPRESSORS)
        or {float(value) for value in frame["ratio"]} != set(RATIOS)
        or frame["prompt_id"].nunique() != PROMPTS
        or not np.isfinite(frame["dq"].to_numpy(dtype=np.float64)).all()
    ):
        raise ValueError("target audit source differs from the H1 contract")

    macro_tables = {}
    compressors = {}
    for compressor in COMPRESSORS:
        macro, values = compressor_audit(
            frame[frame["compressor"] == compressor]
        )
        macro_tables[compressor] = macro.set_index("prompt_id")["macro_dq"]
        compressors[compressor] = values
    macro_matrix = pd.DataFrame(macro_tables).sort_index()
    cross_compressor = {
        left: {
            right: correlation(
                macro_matrix[left].to_numpy(dtype=np.float64),
                macro_matrix[right].to_numpy(dtype=np.float64),
            )
            for right in COMPRESSORS
        }
        for left in COMPRESSORS
    }
    audit = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_target_reliability_audit",
        "provenance": {
            "source_oof_sha256": sha256_file(args.source_oof),
            "source_columns": list(SOURCE_COLUMNS),
            "development_rows": len(frame),
            "development_prompts": int(frame["prompt_id"].nunique()),
            "confirmation_status": "sealed_not_run",
        },
        "compressors": compressors,
        "cross_compressor_prompt_macro_correlations": cross_compressor,
        "interpretation": {
            "allowed": (
                "Descriptive reliability and fold geometry of the already "
                "selected prompt-macro dq probe target."
            ),
            "forbidden": [
                "text representation selection",
                "hyperparameter tuning",
                "cross-compressor candidate inputs",
                "future-prompt or confirmatory claims",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "h1_target_audit_complete",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "baseline_mse": {
                    compressor: values["fold_mean_baseline"]["mse"]
                    for compressor, values in compressors.items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
