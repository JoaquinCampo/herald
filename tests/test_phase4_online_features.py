"""Online feature parity tests against the Phase 2 offline schema."""

import math
from pathlib import Path

import polars as pl
import pytest

from herald.config import TokenSignals
from herald.phase4_online_features import (
    CHEAP_ALL_FEATURE_ORDER,
    EWMA_HALF_LIVES,
    ROLLING_TARGETS,
    ROLLING_WINDOWS,
    OnlineFeatureState,
    _ewma_alpha,
)

FIXTURE = Path("tests/fixtures/phase4_online_parity_run.parquet")

# Tolerance budget per the implementation plan:
#   < 1e-6 deterministic, < 1e-4 floating reductions.
TOL_DETERMINISTIC = 1e-6
TOL_FLOATING = 1e-4

# Offline NaN-handling: polars treats NaN as a numeric value, so
# NaN propagates through both rolling-window sums and the EWMA
# recursion. The online state mirrors that: any window touching a
# NaN yields NaN, EWMA state stays NaN once a NaN enters. Both
# implementations now agree on every cell, NaN or otherwise.
EWMA_PARITY_TARGETS = ("entropy", "top1_prob")  # always non-NaN in fixture


# --- unit primitives -------------------------------------------------


def test_ewma_alpha_matches_definition() -> None:
    assert _ewma_alpha(8) == pytest.approx(1.0 - 0.5 ** (1.0 / 8))
    assert _ewma_alpha(32) == pytest.approx(1.0 - 0.5 ** (1.0 / 32))


def test_feature_order_is_24_columns() -> None:
    assert len(CHEAP_ALL_FEATURE_ORDER) == 24
    assert len(set(CHEAP_ALL_FEATURE_ORDER)) == 24


def test_feature_order_locked() -> None:
    # Pin: this is the order lr_all_cheap coefficients map to.
    # Any reorder silently invalidates the exported predictor.
    assert CHEAP_ALL_FEATURE_ORDER[0] == "entropy"
    assert CHEAP_ALL_FEATURE_ORDER[8] == "eff_vocab_size"
    assert CHEAP_ALL_FEATURE_ORDER[-1] == "compression_ratio"
    assert CHEAP_ALL_FEATURE_ORDER[-2] == "relative_progress"
    assert CHEAP_ALL_FEATURE_ORDER[-3] == "token_pos"


def test_constants_match_offline_dataset() -> None:
    from herald.predictor_dataset import (
        EWMA_HALF_LIVES as OFF_EWMA,
    )
    from herald.predictor_dataset import (
        ROLLING_TARGETS as OFF_TARGETS,
    )
    from herald.predictor_dataset import (
        ROLLING_WINDOWS as OFF_WINDOWS,
    )

    assert ROLLING_TARGETS == OFF_TARGETS
    assert ROLLING_WINDOWS == OFF_WINDOWS
    assert EWMA_HALF_LIVES == OFF_EWMA


# --- synthetic edge cases --------------------------------------------


def _sig(
    entropy=1.0,
    top1=0.5,
    top5=0.7,
    h_alts=0.5,
    delta_h=None,
    delta_h_valid=False,
    kl_div=None,
    top10_jaccard=None,
    eff_vocab=2.7,
    tail_mass=0.01,
    logit_range=10.0,
):
    return TokenSignals(
        entropy=entropy,
        top1_prob=top1,
        top5_prob=top5,
        top5_logprobs=[],
        h_alts=h_alts,
        avg_logp=-1.0,
        delta_h=float("nan") if delta_h is None else delta_h,
        delta_h_valid=delta_h_valid,
        kl_div=float("nan") if kl_div is None else kl_div,
        top10_jaccard=float("nan")
        if top10_jaccard is None
        else top10_jaccard,
        eff_vocab_size=eff_vocab,
        tail_mass=tail_mass,
        logit_range=logit_range,
    )


def test_first_token_nans_propagate() -> None:
    # delta_h, kl_div, top10_jaccard are NaN at t=0.
    state = OnlineFeatureState(
        press="streaming_llm", compression_ratio=0.875, max_new_tokens=512
    )
    row = state.update(_sig())
    assert math.isnan(row["entropy_std_8"])  # n=1, min_samples=2
    assert math.isnan(row["kl_div_mean_8"])  # NaN propagates from t=0
    assert row["entropy_mean_8"] == pytest.approx(1.0)
    assert row["entropy_ewma_hl8"] == pytest.approx(1.0)
    assert row["token_pos"] == 0
    assert row["relative_progress"] == pytest.approx(0.0)
    assert row["compression_ratio"] == 0.875


def test_kl_div_mean_8_stays_nan_until_window_clears() -> None:
    # Once NaN enters at t=0, kl_div_mean_8 is NaN for t in [0..7];
    # at t=8 the window is [t-7..t]=[1..8], NaN dropped, value computed.
    state = OnlineFeatureState(
        press="streaming_llm", compression_ratio=0.5, max_new_tokens=100
    )
    state.update(_sig())  # t=0, kl_div NaN
    for t in range(1, 8):
        row = state.update(
            _sig(
                kl_div=float(t),
                delta_h=0.1,
                delta_h_valid=True,
                top10_jaccard=0.5,
            )
        )
        assert math.isnan(row["kl_div_mean_8"]), (
            f"NaN should still propagate at t={t}"
        )
    row8 = state.update(
        _sig(kl_div=8.0, delta_h=0.1, delta_h_valid=True, top10_jaccard=0.5)
    )
    # window at t=8 is positions [1..8] -> mean(1..8) = 4.5
    assert row8["kl_div_mean_8"] == pytest.approx(4.5)


def test_rolling_std_seeds_at_t1() -> None:
    state = OnlineFeatureState(
        press="streaming_llm", compression_ratio=0.5, max_new_tokens=100
    )
    state.update(_sig(entropy=1.0))
    row = state.update(_sig(entropy=3.0))
    # mean=2; sample std (ddof=1) of (1,3) = sqrt(2).
    assert row["entropy_mean_8"] == pytest.approx(2.0)
    assert row["entropy_std_8"] == pytest.approx(math.sqrt(2.0))


def test_ewma_recursion_matches_formula() -> None:
    state = OnlineFeatureState(
        press="streaming_llm", compression_ratio=0.5, max_new_tokens=100
    )
    alpha8 = _ewma_alpha(8)
    state.update(_sig(entropy=1.0))
    state.update(_sig(entropy=2.0))
    row = state.update(_sig(entropy=3.0))
    expected = 1.0
    for x in (2.0, 3.0):
        expected = (1.0 - alpha8) * expected + alpha8 * x
    assert row["entropy_ewma_hl8"] == pytest.approx(expected)


def test_feature_vector_fills_nulls_with_zero() -> None:
    state = OnlineFeatureState(
        press="streaming_llm", compression_ratio=0.5, max_new_tokens=100
    )
    state.update(_sig())  # t=0 -> entropy_std_8 is null
    vec = state.feature_vector()
    assert len(vec) == 24
    idx = CHEAP_ALL_FEATURE_ORDER.index("entropy_std_8")
    assert vec[idx] == 0.0


# --- parity vs offline phase 2 dataset -------------------------------


def _row_to_signal(row: dict) -> TokenSignals:
    """Reconstruct a TokenSignals from one offline parquet row."""
    return TokenSignals(
        entropy=float(row["entropy"]),
        top1_prob=float(row["top1_prob"]),
        top5_prob=float(row["top5_prob"]),
        top5_logprobs=[],
        h_alts=float(row["h_alts"]),
        avg_logp=float(row.get("avg_logp") or 0.0),
        delta_h=(
            float(row["delta_h"])
            if row.get("delta_h") is not None
            else float("nan")
        ),
        delta_h_valid=bool(row.get("delta_h_valid", False)),
        kl_div=(
            float(row["kl_div"])
            if row.get("kl_div") is not None
            else float("nan")
        ),
        top10_jaccard=(
            float(row["top10_jaccard"])
            if row.get("top10_jaccard") is not None
            else float("nan")
        ),
        eff_vocab_size=float(row["eff_vocab_size"]),
        tail_mass=float(row["tail_mass"]),
        logit_range=float(row["logit_range"]),
    )


@pytest.fixture(scope="module")
def offline_run() -> pl.DataFrame:
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    df = pl.read_parquet(FIXTURE).sort("token_pos")
    return df


def test_parity_with_offline_phase2_run(offline_run: pl.DataFrame) -> None:
    df = offline_run
    press = df["press"][0]
    ratio = float(df["compression_ratio"][0])
    # max_new_tokens isn't stored in the per-token dataset, but
    # relative_progress = token_pos / max_new. Recover max_new from
    # the offline relative_progress at the last token.
    last_pos = int(df["token_pos"][-1])
    last_rel = float(df["relative_progress"][-1])
    if last_rel > 0:
        max_new = round(last_pos / last_rel)
    else:
        max_new = max(last_pos, 1)

    state = OnlineFeatureState(
        press=str(press), compression_ratio=ratio, max_new_tokens=max_new
    )

    # Columns to compare. Tier 0 columns come straight from the
    # input, so they're trivially equal; we still check them as a
    # sanity probe.
    deterministic_cols = (
        "entropy",
        "top1_prob",
        "top5_prob",
        "h_alts",
        "kl_div",
        "top10_jaccard",
        "tail_mass",
        "logit_range",
        "eff_vocab_size",
        "token_pos",
        "relative_progress",
        "compression_ratio",
    )
    floating_cols = (
        "entropy_mean_8",
        "entropy_std_8",
        "entropy_mean_32",
        "top1_prob_mean_8",
        "top1_prob_mean_32",
        "kl_div_mean_8",
        "h_alts_mean_8",
        "top10_jaccard_mean_8",
        "top10_jaccard_mean_32",
    )
    ewma_cols = (
        "entropy_ewma_hl8",
        "entropy_ewma_hl32",
        "top1_prob_ewma_hl8",
    )

    diffs: dict[str, float] = {
        c: 0.0 for c in (*deterministic_cols, *floating_cols, *ewma_cols)
    }

    rows = df.to_dicts()
    for offline_row in rows:
        sig = _row_to_signal(offline_row)
        live = state.update(sig)
        for col in (*deterministic_cols, *floating_cols, *ewma_cols):
            offline_v = offline_row.get(col)
            live_v = live.get(col)
            o_missing = offline_v is None or (
                isinstance(offline_v, float) and math.isnan(offline_v)
            )
            l_missing = live_v is None or (
                isinstance(live_v, float) and math.isnan(live_v)
            )
            if o_missing and l_missing:
                continue
            if o_missing or l_missing:
                pytest.fail(
                    f"null mismatch at token_pos={offline_row['token_pos']} "
                    f"col={col}: offline={offline_v} live={live_v}"
                )
            d = abs(float(offline_v) - float(live_v))
            if d > diffs[col]:
                diffs[col] = d

    # Enforce per-column tolerance.
    for col in deterministic_cols:
        assert diffs[col] <= TOL_DETERMINISTIC, (
            f"{col} diverged: max_abs_diff={diffs[col]}"
        )
    for col in floating_cols:
        assert diffs[col] <= TOL_FLOATING, (
            f"{col} diverged: max_abs_diff={diffs[col]}"
        )
    # EWMA parity restricted to always-non-null targets in this fixture.
    for col in ewma_cols:
        target = col.replace("_ewma_hl8", "").replace("_ewma_hl32", "")
        if target not in EWMA_PARITY_TARGETS:
            continue
        assert diffs[col] <= TOL_FLOATING, (
            f"{col} diverged: max_abs_diff={diffs[col]}"
        )
