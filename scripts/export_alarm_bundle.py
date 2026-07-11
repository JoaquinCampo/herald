# pyright: reportAttributeAccessIssue=false, reportMissingImports=false

"""Freeze grace-window alarm bundles and replay-fidelity targets.

The command is parameterized so a new compression family can only produce a
live bundle from its own parquet rows, aligned hybrid streams, and, for
ExpectedAttentionStatsPress, the exact frozen train-only statistics artifact.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "src")

from herald.expected_attention_stats import (  # noqa: E402
    StatisticsArtifact,
    validate_statistics_provenance,
)
from herald.grace_replay import (  # noqa: E402
    bootstrap_group_ci,
    calibrate_theta,
    oracle_replay,
    replay,
    replay_matrices,
)
from herald.grace_window import AlarmBundle, hyb_feature_names  # noqa: E402
from herald.hybrid_streams import validate_stream_alignment  # noqa: E402
from herald.switch_baselines import split_prompt_ids  # noqa: E402
from herald.switch_risk import featurize  # noqa: E402

DEFAULT_COMPRESSORS = ("expected_attention", "knorm", "streaming_llm")
SPLIT_SEED = 0
N_FOLDS = 5
SEEDS = (0, 1, 2)
K = 2
EPSILON = 0.03
BASE = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "nthread": -1,
}


def _csv(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("compressors cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze prompt-disjoint live-controller alarm bundles"
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("results/predictor/switch_dataset_attn.parquet"),
    )
    parser.add_argument(
        "--streams-npz",
        type=Path,
        default=Path("results/predictor/hybrid_streams_ifeval.npz"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/predictor/alarm_bundle"),
    )
    parser.add_argument(
        "--compressors",
        type=_csv,
        default=DEFAULT_COMPRESSORS,
    )
    parser.add_argument(
        "--expected-attention-stats",
        type=Path,
        default=None,
        help="required when exporting expected_attention_stats",
    )
    return parser.parse_args()


def _required_float(value: object, *, source: str) -> float:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected numeric {source}, got {value!r}")
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(
            f"expected numeric {source}, got {value!r}"
        ) from error
    if not np.isfinite(result):
        raise ValueError(f"expected finite {source}, got {value!r}")
    return result


def _required_int(value: object, *, source: str) -> int:
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, str | int | float):
        raise ValueError(f"expected integer {source}, got {value!r}")
    try:
        return int(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            f"expected integer {source}, got {value!r}"
        ) from error


def _load_streams(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load numeric stream arrays only, never pickle-backed object arrays."""
    try:
        with np.load(path, allow_pickle=False) as data:
            blocks = np.array(data["blocks"], copy=True)
            lengths = np.array(data["lengths"], copy=True)
            trailing = np.array(data["trailing"], copy=True)
            keys = np.array(data["keys"], copy=True)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(
            f"could not load hybrid streams from {path}"
        ) from error
    if (
        blocks.ndim != 3
        or lengths.ndim != 1
        or trailing.ndim != 2
        or keys.ndim != 1
        or keys.dtype.kind not in {"U", "S"}
    ):
        raise ValueError("hybrid streams have invalid dimensions")
    if (
        blocks.shape[0] != lengths.shape[0]
        or blocks.shape[0] != trailing.shape[0]
        or blocks.shape[0] != keys.shape[0]
    ):
        raise ValueError("hybrid stream arrays disagree on row count")
    return blocks, lengths, trailing, keys


def _statistics_digest(
    compressors: tuple[str, ...],
    stats_path: Path | None,
    test_prompt_ids: list[str],
) -> str | None:
    """Validate the no-leakage calibration split and return its digest."""
    needs_statistics = "expected_attention_stats" in compressors
    if not needs_statistics:
        if stats_path is not None:
            raise ValueError(
                "--expected-attention-stats requires expected_attention_stats"
            )
        return None
    if stats_path is None:
        raise ValueError(
            "expected_attention_stats requires --expected-attention-stats"
        )
    artifact = StatisticsArtifact.load(stats_path)
    validate_statistics_provenance(
        artifact,
        task="ifeval",
        test_prompt_ids=test_prompt_ids,
    )
    return artifact.digest


def hyb_summaries(
    blocks: np.ndarray,
    lengths: np.ndarray,
    trailing: np.ndarray,
    k: int,
) -> tuple[np.ndarray, list[str]]:
    """Build the vectorized training-side grace-window feature summaries."""
    b = blocks[:, :k].astype(np.float32)
    m = np.minimum(lengths, k)
    step = np.arange(k)[None, :, None]
    b = np.where(step < m[:, None, None], b, np.nan)
    with np.errstate(all="ignore"):
        mean = np.nanmean(b, axis=1)
        mn = np.nanmin(b, axis=1)
        mx = np.nanmax(b, axis=1)
    step0 = b[:, 0]
    last = b[np.arange(len(b)), np.maximum(m - 1, 0)]
    slope = (last - step0) / np.maximum(m - 1, 1).astype(np.float32)[:, None]
    dtrail = mean - trailing
    parts = [step0, mean, mn, mx, slope, dtrail]
    for part in parts:
        part[m == 0] = np.nan
    return np.concatenate(parts, axis=1), hyb_feature_names(k)


def main() -> None:
    args = parse_args()
    compressors = cast(tuple[str, ...], args.compressors)
    try:
        df = pd.read_parquet(args.parquet)
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"could not read switch parquet {args.parquet}"
        ) from error
    df = df[df["task"] == "ifeval"].reset_index(drop=True)
    blocks, lengths, trailing, stream_keys = _load_streams(args.streams_npz)
    if len(df) != len(blocks):
        raise ValueError(
            "parquet and stream row counts differ, regenerate aligned streams"
        )
    all_rows_value = cast(Any, df).to_dict(orient="records")
    if not isinstance(all_rows_value, list):
        raise RuntimeError("pandas did not produce record rows")
    all_rows = cast(list[dict[str, Any]], all_rows_value)
    validate_stream_alignment(all_rows, stream_keys)
    mask = df["compressor"].isin(compressors).to_numpy()
    df_selected = df[mask].reset_index(drop=True)
    if df_selected.empty:
        raise ValueError("no IFEval rows found for requested compressors")
    rows_value = cast(Any, df_selected).to_dict(orient="records")
    if not isinstance(rows_value, list):
        raise RuntimeError("pandas did not produce record rows")
    rows = cast(list[dict[str, Any]], rows_value)
    selected_blocks = blocks[mask]
    selected_lengths = lengths[mask]
    selected_trailing = trailing[mask]
    available = {str(row["compressor"]) for row in rows}
    missing = sorted(set(compressors) - available)
    if missing:
        raise ValueError(f"requested compressors have no rows: {missing}")
    model_names = {str(row["model"]) for row in rows}
    if len(model_names) != 1:
        raise ValueError("bundle export requires exactly one model")
    model_name = model_names.pop()
    all_prompt_ids = sorted({str(row["prompt_id"]) for row in rows})
    train_prompt_ids, test_prompt_ids = split_prompt_ids(
        all_prompt_ids,
        model=model_name,
        task="ifeval",
        seed=SPLIT_SEED,
    )
    statistics_digest = _statistics_digest(
        compressors,
        args.expected_attention_stats,
        test_prompt_ids,
    )
    mat, summary_names = hyb_summaries(
        selected_blocks,
        selected_lengths,
        selected_trailing,
        K,
    )
    feature_columns = sorted(
        column
        for column in df_selected.columns
        if column.startswith("feat__")
        and not column.startswith("feat__attn_")
    )
    columns = feature_columns + summary_names
    targets: dict[str, Any] = {
        "split_seed": SPLIT_SEED,
        "k": K,
        "epsilon": EPSILON,
        "theta_method": "point",
        "variant": "feat_plus_alarm",
        "parquet": str(args.parquet),
        "streams_npz": str(args.streams_npz),
        "compressors": {},
    }

    for compressor in compressors:
        started = time.time()
        comp_indices = [
            index
            for index, row in enumerate(rows)
            if row["compressor"] == compressor
        ]
        train_indices = [
            index
            for index in comp_indices
            if rows[index]["prompt_id"] in train_prompt_ids
        ]
        test_indices = [
            index
            for index in comp_indices
            if rows[index]["prompt_id"] in test_prompt_ids
        ]
        train = [rows[index] for index in train_indices]
        test = [rows[index] for index in test_indices]
        if not train or not test:
            raise ValueError(
                f"{compressor} has no prompt-disjoint train or test rows"
            )
        train_ids = sorted({str(row["prompt_id"]) for row in train})
        shuffled_ids = train_ids[:]
        random.Random(SPLIT_SEED).shuffle(shuffled_ids)
        folds = [
            set(shuffled_ids[index::N_FOLDS]) for index in range(N_FOLDS)
        ]
        prompt_id_array = np.asarray([row["prompt_id"] for row in train])
        labels = np.asarray(
            [_required_float(row["dq"], source="dq") > 0 for row in train],
            dtype=np.float32,
        )
        work_train = [dict(row) for row in train]
        work_test = [dict(row) for row in test]
        for work, indices in (
            (work_train, train_indices),
            (work_test, test_indices),
        ):
            for row, row_index in zip(work, indices, strict=True):
                for feature_index, name in enumerate(summary_names):
                    value = _required_float(
                        mat[row_index, feature_index], source=name
                    )
                    row[name] = None if np.isnan(value) else value
        train_features = featurize(work_train, columns)
        test_features = featurize(work_test, columns)
        oof_sum = np.zeros(len(train))
        test_sum = np.zeros(len(test))
        boosters = []
        for seed in SEEDS:
            params = {**BASE, "seed": seed}
            oof = np.full(len(train), np.nan)
            for fold_ids in folds:
                test_mask = np.isin(prompt_id_array, list(fold_ids))
                train_mask = ~test_mask
                classifier = xgb.train(
                    params,
                    xgb.DMatrix(
                        train_features[train_mask], label=labels[train_mask]
                    ),
                    num_boost_round=300,
                )
                oof[test_mask] = classifier.predict(
                    xgb.DMatrix(train_features[test_mask])
                )
            final = xgb.train(
                params,
                xgb.DMatrix(train_features, label=labels),
                num_boost_round=300,
            )
            boosters.append(final)
            oof_sum += oof
            test_sum += final.predict(xgb.DMatrix(test_features))
        oof_scores = oof_sum / len(SEEDS)
        test_scores = test_sum / len(SEEDS)
        train_matrices = replay_matrices(train, list(oof_scores))
        test_matrices = replay_matrices(test, list(test_scores))
        theta = calibrate_theta(
            train_matrices, list(oof_scores), epsilon=EPSILON, k=K
        )
        replay_result = replay(test_matrices, theta=theta, k=K)
        oracle_savings, oracle_overhead = oracle_replay(test_matrices, k=K)
        ci_savings = bootstrap_group_ci(
            replay_result.savings, test_matrices.prompt_ids
        )
        ci_cost = bootstrap_group_ci(
            replay_result.cost, test_matrices.prompt_ids
        )
        ci_overhead = bootstrap_group_ci(
            replay_result.overhead, test_matrices.prompt_ids
        )
        bundle_meta: dict[str, Any] = {
            "split_seed": SPLIT_SEED,
            "variant": "feat_plus_alarm",
            "theta_method": "point",
            "n_train_rows": len(train),
            "n_train_prompts": len(train_ids),
            "xgb_params": BASE,
            "num_boost_round": 300,
            "oof_mean_savings_at_theta": _required_float(
                replay(train_matrices, theta=theta, k=K).savings.mean(),
                source="OOF savings",
            ),
        }
        if compressor == "expected_attention_stats":
            bundle_meta["expected_attention_stats_sha256"] = statistics_digest
        bundle = AlarmBundle(
            compressor=compressor,
            k=K,
            epsilon=EPSILON,
            theta=_required_float(theta, source="theta"),
            feature_cols=columns,
            boosters=boosters,
            meta=bundle_meta,
        )
        bundle.save(args.out_dir / compressor)
        groups: list[dict[str, Any]] = []
        for index, (prompt_id, ratio) in enumerate(test_matrices.keys):
            valid_length = _required_int(
                test_matrices.valid[index], source="valid attempt count"
            )
            commit_index = _required_int(
                replay_result.commit_index[index], source="commit index"
            )
            groups.append(
                {
                    "prompt_id": prompt_id,
                    "ratio": ratio,
                    "ref_len": _required_float(
                        test_matrices.ref_len[index],
                        source="reference length",
                    ),
                    "attempt_s": test_matrices.s[
                        index, :valid_length
                    ].tolist(),
                    "scores": test_matrices.pred[
                        index, :valid_length
                    ].tolist(),
                    "commit_s": (
                        _required_int(
                            test_matrices.s[index, commit_index],
                            source="commit position",
                        )
                        if commit_index >= 0
                        else None
                    ),
                    "savings": _required_float(
                        replay_result.savings[index], source="savings"
                    ),
                    "cost": _required_float(
                        replay_result.cost[index], source="cost"
                    ),
                    "overhead": _required_float(
                        replay_result.overhead[index], source="overhead"
                    ),
                }
            )
        compressor_target: dict[str, Any] = {
            "theta": _required_float(theta, source="theta"),
            "test_prompt_ids": test_prompt_ids,
            "replay_test": {
                "n_groups": len(test_matrices.keys),
                "mean_savings": _required_float(
                    replay_result.savings.mean(), source="mean savings"
                ),
                "ci95_savings": list(ci_savings),
                "mean_cost": _required_float(
                    replay_result.cost.mean(), source="mean cost"
                ),
                "ci95_cost": list(ci_cost),
                "mean_overhead": _required_float(
                    replay_result.overhead.mean(), source="mean overhead"
                ),
                "ci95_overhead": list(ci_overhead),
                "oracle_savings": _required_float(
                    oracle_savings.mean(), source="oracle savings"
                ),
                "oracle_overhead": _required_float(
                    oracle_overhead.mean(), source="oracle overhead"
                ),
            },
            "groups": groups,
        }
        if compressor == "expected_attention_stats":
            compressor_target["expected_attention_stats_sha256"] = (
                statistics_digest
            )
        targets["compressors"][compressor] = compressor_target
        print(
            f"BUNDLE {compressor:20s} k={K} eps={EPSILON:.2f} "
            f"theta={theta:.6f} sav={replay_result.savings.mean():.4f} "
            f"cost={replay_result.cost.mean():.4f} "
            f"overhead={replay_result.overhead.mean():.3f} "
            f"oracle={oracle_savings.mean():.4f} "
            f"[{time.time() - started:.0f}s]",
            flush=True,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_path = args.out_dir / "fidelity_targets.json"
    target_path.write_text(json.dumps(targets, indent=1) + "\n")
    print(f"wrote {target_path}")


if __name__ == "__main__":
    main()
