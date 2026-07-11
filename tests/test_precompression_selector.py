import pytest

from herald.precompression_selector import (
    deterministic_selector_split,
    fit_selector_bundle,
    select_safe_threshold,
)


def _row(
    prompt_id: str,
    s: int,
    dq: float,
    *,
    major_damage: bool = False,
) -> dict[str, object]:
    return {
        "prompt_id": prompt_id,
        "s": s,
        "ref_len": 100,
        "dq": dq,
        "major_damage": major_damage,
        "feat__entropy": float(s) / 100,
        "ratio": 0.25,
    }


def test_selector_split_is_prompt_disjoint_and_deterministic() -> None:
    prompt_ids = [f"p{i:02d}" for i in range(20)]
    split = deterministic_selector_split(prompt_ids, ["p18", "p19"])

    assert split.calibration_prompt_ids == ["p00", "p05", "p10", "p15"]
    assert set(split.train_prompt_ids).isdisjoint(
        split.calibration_prompt_ids
    )
    assert set(split.train_prompt_ids).isdisjoint(split.target_prompt_ids)
    assert set(split.calibration_prompt_ids).isdisjoint(
        split.target_prompt_ids
    )
    assert (
        sorted(
            split.train_prompt_ids
            + split.calibration_prompt_ids
            + split.target_prompt_ids
        )
        == prompt_ids
    )


def test_threshold_rejects_unsafe_early_commits() -> None:
    rows = [
        _row("p0", 0, 0.5, major_damage=True),
        _row("p0", 16, 0.0),
        _row("p1", 0, 0.0),
        _row("p1", 16, 0.0),
    ]
    scores = [0.75, 0.80, 0.85, 0.70]

    result = select_safe_threshold(rows, scores)

    assert result.threshold == 0.80
    assert result.n_commits == 2
    assert result.damage_rate == 0.0
    assert result.major_damage_rate == 0.0
    assert result.mean_opportunity == pytest.approx(0.92)


def test_fit_bundle_uses_only_precompression_features() -> None:
    train = [
        _row(f"train-{index}", 0, 0.0 if index % 2 else 0.5)
        for index in range(24)
    ]
    calibration = [
        _row(f"cal-{index}", 0, 0.0 if index % 2 else 0.5)
        for index in range(8)
    ]

    bundle = fit_selector_bundle(
        train,
        calibration,
        compressor="expected_attention_stats",
        meta={"source": "test"},
        num_boost_round=2,
    )

    assert bundle.feature_cols == ["feat__entropy", "ratio"]
    assert "dq" not in bundle.feature_cols
    assert "major_damage" not in bundle.feature_cols
    assert len(bundle.boosters) == 3
    assert bundle.meta["n_train_prompts"] == 24
    assert bundle.meta["n_calibration_prompts"] == 8
