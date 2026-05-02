"""Tests for src/herald/analysis/cross_press_transfer.py.

CPU-only, fast. Synthetic two-press fixture where both presses share
the same predictive structure (catastrophic-run entropy ramps before
onset), so within-press AUROC and cross-press AUROC should both clear
chance. We do NOT assert that cross-press transfer is identical to
within-press; the headline test is structural (matrix shape, counts,
insufficient handling), not a Phase 0 transfer claim.
"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from herald.analysis.cross_press_transfer import (
    CROSS_PRESS_METADATA_FEATURES,
    CrossPressTransferConfig,
    compute_cell,
    compute_transfer_matrix,
    resolve_feature_sets,
    run_cross_press_transfer,
)
from herald.analysis.information_ceiling import (
    POSITION_FEATURES,
    InformationCeilingConfig,
    build_token_dataset,
)
from herald.metrics.io import RUNS_SCHEMA, TOKENS_SCHEMA

# ----- fixture helpers -----------------------------------------------


def _looping_token_ids(window: int = 20, repeats: int = 4) -> list[int]:
    prefix = list(range(100, 150))
    pattern = list(range(window))
    suffix = list(range(50, 70))
    return prefix + pattern * repeats + suffix


def _runs_record(
    run_id: str,
    press: str,
    ratio: float,
    catastrophes: list[str],
    token_ids: list[int],
    max_new_tokens: int = 256,
    correct: bool | None = None,
    task: str = "gsm8k",
) -> dict:
    return {
        "run_id": run_id,
        "prompt_id": run_id + "_p",
        "prompt_text": "x",
        "prompt_hash": "h",
        "model": "fake",
        "model_revision": None,
        "tokenizer_revision": None,
        "dtype": "float16",
        "device_class": "cpu",
        "task": task,
        "press": press,
        "compression_ratio": ratio,
        "max_new_tokens": max_new_tokens,
        "decoding_config": [],
        "seed": 0,
        "baseline_run_id": "",
        "generated_text": "",
        "generated_token_ids": token_ids,
        "num_tokens_generated": len(token_ids),
        "stop_reason": "eos",
        "predicted_answer": None,
        "ground_truth": "",
        "correct": correct,
        "catastrophes": catastrophes,
        "replay_status": "ok",
        "replay_error": None,
        "created_at": "2026-05-02T00:00:00Z",
        "herald_git_sha": "test",
        "wall_clock_per_token": 0.0,
        "peak_memory_mb": 0.0,
        "kv_size_at_end": 0.0,
        "policy_name": "fixed_ratio",
        "replay_wall_clock_seconds": 0.0,
    }


def _tokens_rows(
    run_id: str,
    n: int,
    entropy_fn=lambda t: 1.0,
) -> list[dict]:
    rows: list[dict] = []
    for t in range(n):
        rows.append(
            {
                "run_id": run_id,
                "token_pos": t,
                "token_id": t,
                "token_str": "x",
                "entropy": float(entropy_fn(t)),
                "top1_prob": 0.9,
                "top5_prob": 0.95,
                "top5_logprobs": [-0.1, -1.0, -2.0, -3.0, -4.0],
                "h_alts": 0.5,
                "avg_logp": -1.0,
                "delta_h": 0.0,
                "delta_h_valid": True,
                "kl_div": 0.05,
                "top10_jaccard": 0.8,
                "eff_vocab_size": 5.0,
                "tail_mass": 0.05,
                "logit_range": 10.0,
                "lookback_ratio": float("nan"),
            }
        )
    return rows


def _write_runs(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(records, schema=RUNS_SCHEMA)
    pq.write_table(table, path)


def _write_tokens(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA)
    pq.write_table(table, path)


@pytest.fixture
def synthetic_two_press_dataset(tmp_path: Path) -> Path:
    """Two-press dataset (snapkv, streaming_llm). Each press has 12
    catastrophic and 12 healthy runs. Catastrophic runs share the
    same entropy-ramp structure across both presses so the predictor
    has a chance to transfer.
    """
    rng = np.random.default_rng(13)
    root = tmp_path / "phase0"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"

    runs: list[dict] = []
    onset_target = 70  # detect_looping_onset returns 50 + 20 = 70.

    presses = [("snapkv", 0.875), ("streaming_llm", 0.875)]

    n_per_class = 12
    for press, ratio in presses:
        for i in range(n_per_class):
            rid = f"{press}_cat_{i:02d}"
            ids = _looping_token_ids(window=20, repeats=4)
            runs.append(_runs_record(rid, press, ratio, ["looping"], ids))

            def cat_entropy(t: int) -> float:
                base = 0.6 + 0.05 * rng.standard_normal()
                ramp = max(0.0, (t - (onset_target - 30))) * 0.05
                return base + ramp

            _write_tokens(
                tokens_root
                / f"press={press}"
                / f"ratio={ratio:.4f}"
                / f"{rid}.parquet",
                _tokens_rows(rid, len(ids), entropy_fn=cat_entropy),
            )

        for i in range(n_per_class):
            rid = f"{press}_ctrl_{i:02d}"
            ids = list(range(150))
            runs.append(_runs_record(rid, press, ratio, [], ids))

            def ctrl_entropy(t: int) -> float:
                return 0.6 + 0.05 * rng.standard_normal()

            _write_tokens(
                tokens_root
                / f"press={press}"
                / f"ratio={ratio:.4f}"
                / f"{rid}.parquet",
                _tokens_rows(rid, len(ids), entropy_fn=ctrl_entropy),
            )

    _write_runs(runs_path, runs)
    return root


# ----- unit tests ----------------------------------------------------


def test_resolve_feature_sets_drops_press_code():
    """press_code must never appear in any cross-press feature set."""
    used_online = ["entropy", "top1_prob"]
    sets = resolve_feature_sets(
        used_online,
        ("position_metadata", "online", "all", "entropy_only"),
    )
    for name, feats in sets.items():
        assert "press_code" not in feats, (
            f"press_code leaked into feature set {name}"
        )
    # Sanity checks on contents.
    assert set(sets["online"]) == set(used_online)
    assert "compression_ratio" in sets["position_metadata"]
    assert "task_code" in sets["position_metadata"]
    assert sets["entropy_only"] == ["entropy"]


def test_resolve_feature_sets_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_feature_sets([], ("not_a_real_set",))


def test_resolve_feature_sets_entropy_only_empty_when_missing():
    sets = resolve_feature_sets(["top1_prob"], ("entropy_only",))
    assert sets["entropy_only"] == []


def test_compute_cell_marks_insufficient_when_no_positives(
    synthetic_two_press_dataset: Path,
):
    """If a horizon yields no positives on either side, the cell
    must be marked insufficient with a clear reason.
    """
    runs = pl.read_parquet(
        synthetic_two_press_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_two_press_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    ic_cfg = InformationCeilingConfig(horizons=(5,))
    df, _, _ = build_token_dataset(runs, tokens, ic_cfg)
    cfg = CrossPressTransferConfig(
        horizons=(5,),
        n_bootstrap=0,
        min_pos=10_000,  # impossible threshold
        min_neg=10_000,
    )
    cell = compute_cell(
        df,
        train_press="snapkv",
        test_press="streaming_llm",
        features=["entropy"],
        horizon=5,
        cfg=cfg,
    )
    assert cell["insufficient"] is True
    assert cell["insufficient_reason"]
    assert cell["auroc"] is None


def test_compute_cell_diagonal_runs_via_groupkfold(
    synthetic_two_press_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_two_press_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_two_press_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    ic_cfg = InformationCeilingConfig(horizons=(5,))
    df, used, _ = build_token_dataset(runs, tokens, ic_cfg)
    assert "entropy" in used
    cfg = CrossPressTransferConfig(horizons=(5,), n_bootstrap=0, n_splits=3)
    cell = compute_cell(
        df,
        train_press="snapkv",
        test_press="snapkv",
        features=["entropy"],
        horizon=5,
        cfg=cfg,
    )
    assert cell["insufficient"] is False
    assert cell["auroc"] is not None
    # Entropy ramp is genuinely predictive of upcoming onset, so the
    # within-press CV should clear chance.
    assert cell["auroc"] > 0.55
    assert cell["n_train_runs"] > 0
    assert cell["n_test_runs"] > 0


def test_compute_transfer_matrix_has_expected_shape(
    synthetic_two_press_dataset: Path,
):
    runs = pl.read_parquet(
        synthetic_two_press_dataset / "final" / "runs.parquet"
    )
    from herald.analysis.event_study import load_tokens_for_runs

    tokens = load_tokens_for_runs(
        synthetic_two_press_dataset / "final" / "tokens",
        runs["run_id"].to_list(),
    )
    ic_cfg = InformationCeilingConfig(horizons=(5, 10))
    df, used, _ = build_token_dataset(runs, tokens, ic_cfg)
    cfg = CrossPressTransferConfig(
        horizons=(5, 10), n_bootstrap=0, n_splits=3
    )
    fs = {"online": ["entropy"], "position": list(POSITION_FEATURES)}
    matrix = compute_transfer_matrix(
        df,
        presses=["snapkv", "streaming_llm"],
        feature_sets=fs,
        horizons=(5, 10),
        cfg=cfg,
    )
    # 2 horizons * 2 feature sets * 2 train * 2 test = 16 rows.
    assert matrix.height == 16
    assert set(matrix["train_press"].unique().to_list()) == {
        "snapkv",
        "streaming_llm",
    }
    assert set(matrix["test_press"].unique().to_list()) == {
        "snapkv",
        "streaming_llm",
    }
    # Diagonal cells should not be empty in either horizon.
    diag = matrix.filter(pl.col("train_press") == pl.col("test_press"))
    assert diag.height == 8


def test_run_cross_press_transfer_smoke(
    synthetic_two_press_dataset: Path, tmp_path: Path
):
    out = tmp_path / "cpt"
    cfg = CrossPressTransferConfig(
        horizons=(5, 10),
        feature_sets=("online", "position_metadata", "all"),
        n_bootstrap=10,
        n_splits=3,
    )
    result = run_cross_press_transfer(synthetic_two_press_dataset, out, cfg)
    summary = result.summary
    assert "blockers" not in summary
    assert summary["n_compressed_runs"] == 48
    assert summary["n_presses"] == 2
    assert summary["presses_found"] == ["snapkv", "streaming_llm"]
    assert "phase0_caveat" in summary
    # Aggregates exist for each (horizon, feature_set).
    agg = summary["aggregate_by_horizon_and_feature_set"]
    assert "5" in agg and "10" in agg
    assert "online" in agg["5"]
    # Headline diagonal mean is computable on this fixture.
    assert agg["5"]["online"]["diagonal_mean"] is not None
    # Files written.
    assert (out / "cross_press_transfer.parquet").exists()
    assert (out / "cross_press_transfer_summary.json").exists()
    assert (out / "cross_press_transfer_h5.png").exists()
    # press_code should not appear in resolved feature sets.
    for feats in summary["feature_sets_resolved"].values():
        assert "press_code" not in feats


def test_run_cross_press_transfer_blockers_when_input_missing(
    tmp_path: Path,
):
    cfg = CrossPressTransferConfig()
    result = run_cross_press_transfer(
        tmp_path / "nope", tmp_path / "out", cfg
    )
    assert "blockers" in result.summary
    assert (tmp_path / "out" / "cross_press_transfer_summary.json").exists()


def test_run_cross_press_transfer_handles_single_press(
    tmp_path: Path,
):
    """Single compressed press should fail cleanly (no off-diagonal)."""
    root = tmp_path / "single"
    runs_path = root / "final" / "runs.parquet"
    tokens_root = root / "final" / "tokens"
    runs: list[dict] = []
    for i in range(4):
        rid = f"only_{i}"
        ids = _looping_token_ids()
        runs.append(_runs_record(rid, "snapkv", 0.875, ["looping"], ids))
        _write_tokens(
            tokens_root / "press=snapkv" / f"{rid}.parquet",
            _tokens_rows(rid, len(ids)),
        )
    _write_runs(runs_path, runs)
    cfg = CrossPressTransferConfig(horizons=(5,), n_bootstrap=0)
    result = run_cross_press_transfer(root, tmp_path / "out", cfg)
    assert result.summary["status"] == "insufficient_presses"
    assert "phase0_caveat" in result.summary


def test_cli_main_synthetic(
    synthetic_two_press_dataset: Path, tmp_path: Path
):
    from scripts.build_cross_press_transfer import main

    out = tmp_path / "cli_out"
    rc = main(
        [
            "--input",
            str(synthetic_two_press_dataset),
            "--output",
            str(out),
            "--horizons",
            "5,10",
            "--feature-sets",
            "online,position_metadata,all",
            "--n-bootstrap",
            "5",
            "--n-splits",
            "3",
        ]
    )
    assert rc == 0
    summary = json.loads(
        (out / "cross_press_transfer_summary.json").read_text()
    )
    assert summary["n_presses"] == 2
    assert "aggregate_by_horizon_and_feature_set" in summary


def test_cross_press_metadata_excludes_press_code():
    """Belt-and-suspenders: the cross-press metadata constant itself
    must not contain press_code, so the matrix is never confounded.
    """
    assert "press_code" not in CROSS_PRESS_METADATA_FEATURES
    assert "compression_ratio" in CROSS_PRESS_METADATA_FEATURES
    assert "task_code" in CROSS_PRESS_METADATA_FEATURES
