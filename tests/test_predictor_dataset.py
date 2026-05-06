"""Tests for src/herald/predictor_dataset.py.

Behaviors under test:
- future-window aggregates use the open-closed interval (t, t+H]
- aggregates respect run boundaries (no leakage from run A to run B)
- right-censored rows produce null labels per horizon
- rolling features are causal (depend only on tokens at or before t)
- nll_ratio validator is sign-flipped
- end-to-end build over a tiny in-memory fixture matches expected counts
"""

import polars as pl

from herald.predictor_dataset import (
    add_future_labels,
    add_rolling_features,
    flip_nll_ratio_sign,
)

# ---------------------------------------------------------------
# Future-window aggregates
# ---------------------------------------------------------------


def test_future_sum_uses_open_close_interval() -> None:
    """At token t, future_sum_*_H = sum over (t, t+H], excluding t."""
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 5,
            "token_pos": [0, 1, 2, 3, 4],
            "js_full": [10.0, 1.0, 2.0, 3.0, 4.0],
            "kl_unc_comp_full": [10.0, 1.0, 2.0, 3.0, 4.0],
        }
    )
    out = add_future_labels(df, horizons=[2])
    # H=2: at t=0 sum is js[1]+js[2] = 1+2 = 3 (excluding js[0]=10)
    assert out["future_sum_js_2"].to_list()[:3] == [3.0, 5.0, 7.0]
    assert out["future_sum_kl_2"].to_list()[:3] == [3.0, 5.0, 7.0]


def test_future_max_uses_open_close_interval() -> None:
    """At token t, future_max_js_H = max over (t, t+H]."""
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 5,
            "token_pos": [0, 1, 2, 3, 4],
            "js_full": [1.0, 2.0, 3.0, 4.0, 5.0],
            "kl_unc_comp_full": [0.0] * 5,
        }
    )
    out = add_future_labels(df, horizons=[2])
    # H=2 windows over realized tokens, open-close (t, t+H]:
    # t=0: max(js[1], js[2]) = max(2,3) = 3
    # t=1: max(js[2], js[3]) = max(3,4) = 4
    # t=2: max(js[3], js[4]) = max(4,5) = 5
    # t=3: max(js[4]) = 5     (only 1 future row, partial window)
    # t=4: no future rows -> null
    vals = out["future_max_js_2"].to_list()
    assert vals[:3] == [3.0, 4.0, 5.0]


def test_future_window_right_censors_last_H_rows() -> None:
    """Last H rows of a run get null future_sum labels for horizon H."""
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 5,
            "token_pos": [0, 1, 2, 3, 4],
            "js_full": [1.0, 1.0, 1.0, 1.0, 1.0],
            "kl_unc_comp_full": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    out = add_future_labels(df, horizons=[2])
    sums = out["future_sum_js_2"].to_list()
    # Last 2 rows are right-censored for H=2.
    assert sums[-2:] == [None, None]
    # First 3 rows have a full H=2 window.
    assert sums[:3] == [2.0, 2.0, 2.0]


def test_future_window_does_not_cross_run_boundaries() -> None:
    """Concatenated runs: last row of run A must not pull from run B."""
    df = pl.DataFrame(
        {
            "run_id": ["A", "A", "A", "B", "B", "B"],
            "token_pos": [0, 1, 2, 0, 1, 2],
            "js_full": [1.0, 1.0, 1.0, 100.0, 100.0, 100.0],
            "kl_unc_comp_full": [1.0, 1.0, 1.0, 100.0, 100.0, 100.0],
        }
    )
    out = add_future_labels(df, horizons=[2])
    sums = out["future_sum_js_2"].to_list()
    # Run A: t=0 -> js_A[1]+js_A[2] = 2; t=1 right-censored -> null;
    # t=2 right-censored -> null. Crucially, t=1 and t=2 must NOT see B.
    assert sums[0] == 2.0
    assert sums[1] is None
    assert sums[2] is None
    # Run B labels start fresh.
    assert sums[3] == 200.0
    assert sums[4] is None
    assert sums[5] is None


# ---------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------


def test_rolling_features_are_causal_within_run() -> None:
    """rolling_mean_W at token t uses only tokens at positions <= t."""
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 4,
            "token_pos": [0, 1, 2, 3],
            "entropy": [1.0, 2.0, 3.0, 4.0],
            # Fill the rest of ROLLING_TARGETS so add_rolling_features
            # doesn't error out on missing columns.
            "top1_prob": [0.0] * 4,
            "h_alts": [0.0] * 4,
            "delta_h": [0.0] * 4,
            "kl_div": [0.0] * 4,
            "top10_jaccard": [0.0] * 4,
        }
    )
    out = add_rolling_features(df)
    means = out["entropy_mean_2"].to_list()
    # Window of 2 with min_periods=1, causal:
    # t=0: mean([1]) = 1
    # t=1: mean([1,2]) = 1.5
    # t=2: mean([2,3]) = 2.5
    # t=3: mean([3,4]) = 3.5
    assert means == [1.0, 1.5, 2.5, 3.5]


def test_rolling_features_do_not_cross_run_boundaries() -> None:
    """Rolling at first token of run B must not include run A's tail."""
    df = pl.DataFrame(
        {
            "run_id": ["A", "A", "B", "B"],
            "token_pos": [0, 1, 0, 1],
            "entropy": [10.0, 10.0, 1.0, 1.0],
            "top1_prob": [0.0] * 4,
            "h_alts": [0.0] * 4,
            "delta_h": [0.0] * 4,
            "kl_div": [0.0] * 4,
            "top10_jaccard": [0.0] * 4,
        }
    )
    out = add_rolling_features(df)
    means = out["entropy_mean_2"].to_list()
    # Run A: [10, 10] -> [10, 10]. Run B: [1, 1] -> [1, 1].
    # If boundaries leak, B's first row would average 10+1=5.5 etc.
    assert means == [10.0, 10.0, 1.0, 1.0]


# ---------------------------------------------------------------
# nll_ratio sign flip
# ---------------------------------------------------------------


def test_flip_nll_ratio_sign_negates_column() -> None:
    df = pl.DataFrame({"nll_ratio": [-1.0, 0.0, 2.0, None]})
    out = flip_nll_ratio_sign(df)
    assert out["nll_ratio_flipped"].to_list() == [1.0, 0.0, -2.0, None]
    # Original column preserved.
    assert out["nll_ratio"].to_list() == [-1.0, 0.0, 2.0, None]


def test_flip_nll_ratio_sign_no_op_when_column_missing() -> None:
    df = pl.DataFrame({"sum_kl": [1.0, 2.0]})
    out = flip_nll_ratio_sign(df)
    assert "nll_ratio_flipped" not in out.columns


# ---------------------------------------------------------------
# Configurable horizons
# ---------------------------------------------------------------


def test_add_future_labels_emits_one_column_set_per_horizon() -> None:
    df = pl.DataFrame(
        {
            "run_id": ["r"] * 3,
            "token_pos": [0, 1, 2],
            "js_full": [1.0, 1.0, 1.0],
            "kl_unc_comp_full": [1.0, 1.0, 1.0],
        }
    )
    out = add_future_labels(df, horizons=[1, 2])
    for col in (
        "future_sum_js_1",
        "future_sum_js_2",
        "future_sum_kl_1",
        "future_sum_kl_2",
        "future_max_js_1",
        "future_max_js_2",
    ):
        assert col in out.columns, f"missing column: {col}"


# ---------------------------------------------------------------
# End-to-end: tiny synthetic fixture
# ---------------------------------------------------------------


def test_build_token_dataset_smoke(tmp_path) -> None:
    """End-to-end build over a 2-run synthetic fixture.

    Verifies the builder produces one row per token, joins replay,
    computes labels, and respects run boundaries.
    """
    from herald.predictor_dataset import build_token_dataset

    final_dir = tmp_path / "phase1" / "final"
    tokens_dir = final_dir / "tokens" / "press=streaming_llm" / "ratio=0.5000"
    replay_dir = final_dir / "replay" / "press=streaming_llm" / "ratio=0.5000"
    tokens_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)

    metrics_dir = tmp_path / "phase1" / "metrics"
    metrics_dir.mkdir(parents=True)

    # Two runs, both press=streaming_llm ratio=0.5
    for rid in ("aaa", "bbb"):
        pl.DataFrame(
            {
                "run_id": [rid] * 4,
                "token_pos": [0, 1, 2, 3],
                "token_id": [1, 2, 3, 4],
                "token_str": ["a"] * 4,
                "entropy": [0.5, 0.6, 0.7, 0.8],
                "top1_prob": [0.9, 0.8, 0.7, 0.6],
                "top5_prob": [0.99] * 4,
                "top5_logprobs": [[0.0] * 5] * 4,
                "h_alts": [0.1] * 4,
                "avg_logp": [-0.1] * 4,
                "delta_h": [0.0] * 4,
                "delta_h_valid": [False, True, True, True],
                "kl_div": [0.0, 0.1, 0.1, 0.1],
                "top10_jaccard": [1.0, 0.9, 0.8, 0.7],
                "eff_vocab_size": [2.0] * 4,
                "tail_mass": [0.01] * 4,
                "logit_range": [10.0] * 4,
                "lookback_ratio": [0.0] * 4,
            }
        ).write_parquet(tokens_dir / f"{rid}.parquet")

        pl.DataFrame(
            {
                "run_id": [rid] * 4,
                "token_pos": [0, 1, 2, 3],
                "realized_token_id": [1, 2, 3, 4],
                "union_top_k_token_ids": [[1, 2]] * 4,
                "logprobs_compressed": [[0.0, -1.0]] * 4,
                "logprobs_uncompressed": [[0.0, -1.0]] * 4,
                "tail_mass_compressed": [0.0] * 4,
                "tail_mass_uncompressed": [0.0] * 4,
                "realized_logprob_compressed": [0.0] * 4,
                "realized_logprob_uncompressed": [0.0] * 4,
                "js_full": [0.0, 0.1, 0.2, 0.3],
                "kl_unc_comp_full": [0.0, 0.1, 0.2, 0.3],
                "kl_comp_unc_full": [0.0, 0.1, 0.2, 0.3],
                "top1_match": [True] * 4,
                "top1_rank_comp_under_unc": [0] * 4,
                "top1_rank_unc_under_comp": [0] * 4,
            }
        ).write_parquet(replay_dir / f"{rid}.parquet")

    pl.DataFrame(
        {
            "run_id": ["aaa", "bbb"],
            "prompt_id": ["p1", "p2"],
            "prompt_text": ["x", "y"],
            "prompt_hash": ["h1", "h2"],
            "model": ["m"] * 2,
            "model_revision": ["r"] * 2,
            "tokenizer_revision": ["r"] * 2,
            "dtype": ["fp16"] * 2,
            "device_class": ["gpu"] * 2,
            "task": ["gsm8k", "humaneval"],
            "press": ["streaming_llm"] * 2,
            "compression_ratio": [0.5, 0.5],
            "max_new_tokens": [512, 512],
            "decoding_config": ["{}"] * 2,
            "seed": [42, 42],
            "baseline_run_id": ["bA", "bB"],
            "generated_text": ["", ""],
            "generated_token_ids": [[1, 2, 3, 4]] * 2,
            "num_tokens_generated": [4, 4],
            "stop_reason": ["eos"] * 2,
            "predicted_answer": ["", ""],
            "ground_truth": ["", ""],
            "correct": [True, False],
            "catastrophes": [[], []],
            "replay_status": ["ok", "ok"],
            "replay_error": [None, None],
            "created_at": ["2026-05-05T00:00:00"] * 2,
            "herald_git_sha": ["sha"] * 2,
        }
    ).write_parquet(final_dir / "runs.parquet")

    pl.DataFrame(
        {
            "run_id": ["aaa", "bbb"],
            "task": ["gsm8k", "humaneval"],
            "press": ["streaming_llm"] * 2,
            "compression_ratio": [0.5, 0.5],
            "sum_kl": [0.5, 0.6],
            "sum_js": [0.4, 0.5],
            "nll_ratio": [-1.0, -2.0],
            "first_divergence_point": [1, 2],
            "rouge_l_drop": [0.1, 0.2],
            "char_edit_ratio": [0.1, 0.2],
            "length_diff_ratio": [0.0, 0.0],
            "embedding_cosine_drop": [None, None],
            "has_looping": [False, False],
            "has_non_termination": [False, False],
            "has_format_break": [False, False],
            "has_drift": [False, False],
            "gross_harm_final": [True, False],
            "gross_help_final": [False, False],
            "quality_delta": [0.5, 0.0],
            "quality_label_source": ["gsm8k_exact", "humaneval_pass"],
            "baseline_correct_final": [True, True],
            "compressed_correct_final": [False, True],
            "baseline_quality_score": [1.0, 1.0],
            "compressed_quality_score": [0.0, 1.0],
        }
    ).write_parquet(metrics_dir / "run_damage.parquet")

    output_path = tmp_path / "out.parquet"
    summary = build_token_dataset(
        final_dir=final_dir,
        run_damage_path=metrics_dir / "run_damage.parquet",
        output_path=output_path,
        horizons=[2],
    )
    df = pl.read_parquet(output_path)

    # 2 runs * 4 tokens = 8 rows.
    assert df.height == 8
    assert summary["n_rows"] == 8
    assert summary["n_runs"] == 2

    # Required label columns present.
    for col in ("future_sum_js_2", "future_sum_kl_2", "future_max_js_2"):
        assert col in df.columns

    # Required validator columns joined from run_damage.
    for col in (
        "gross_harm_final",
        "quality_delta",
        "rouge_l_drop",
        "has_looping",
        "nll_ratio_flipped",
    ):
        assert col in df.columns

    # press is in features (only dropped in the held-out-press split).
    for col in ("task", "press", "compression_ratio", "token_pos"):
        assert col in df.columns

    # nll_ratio sign flipped in the validator join.
    assert df.filter(pl.col("run_id") == "aaa")["nll_ratio_flipped"][0] == 1.0
