from __future__ import annotations

from pathlib import Path

import pytest

from herald.magnitude import (
    KNOWN_COMPRESSORS,
    Baseline,
    CompressorFit,
    _select_baselines,
    bootstrap_prompt_indices,
    causal_feature_names,
    fit_evidence,
    load_model_bundle,
    make_outer_split,
    trajectory_weights,
    validate_rows,
    write_model_bundle,
)


def _row(
    prompt: str,
    *,
    compressor: str = "streaming_llm",
    s: int = 0,
    dq: float = 0.2,
) -> dict[str, object]:
    return {
        "model": "llama",
        "task": "ifeval",
        "prompt_id": prompt,
        "compressor": compressor,
        "intervention_semantics": (
            "herald.matched_reprefill_v1"
            if compressor == "expected_attention"
            else "herald.cache_native_pending_v1"
        ),
        "feature_timing": "herald.pre_switch_pending_logit_v1",
        "ratio": 0.5,
        "s": s,
        "q_ref": 1.0,
        "q_control": 1.0,
        "q_hybrid": 1.0 - dq,
        "dq": dq,
        "damaged": int(dq > 0),
        "major_damage": int(dq >= 0.5),
        "feat__entropy": 1.0,
        "feat__attn_entropy_lmean": 99.0,
        "ref_len": 500,
    }


def test_weights_equalize_trajectories_not_rows() -> None:
    rows = [
        _row("short", s=0),
        _row("long", s=0),
        _row("long", s=16),
        _row("long", s=32),
    ]
    weights = trajectory_weights(rows)
    assert weights[0] == pytest.approx(1.0)
    assert weights[1:].sum() == pytest.approx(1.0)


def test_features_use_exact_base_logit_schema() -> None:
    row = _row("p")
    row["feat__oracle_future_quality"] = 7.0
    features = causal_feature_names([row])
    assert features == ("feat__entropy",)


def test_validation_rejects_inconsistent_label() -> None:
    row = _row("p")
    row["dq"] = 0.3
    with pytest.raises(ValueError, match="dq mismatch"):
        validate_rows([row], compressors=["streaming_llm"])


def test_validation_uses_matched_control_not_global_reference() -> None:
    row = _row("p", compressor="expected_attention", dq=0.25)
    row["q_ref"] = 1.0
    row["q_control"] = 0.5
    row["q_hybrid"] = 0.25
    validated = validate_rows([row], compressors=["expected_attention"])
    assert validated[0]["dq"] == pytest.approx(0.25)


def test_provenance_must_be_explicit() -> None:
    with pytest.raises(ValueError, match="provenance"):
        validate_rows(
            [_row("p")],
            compressors=["streaming_llm"],
            require_provenance=True,
        )


def test_known_default_scope_and_arithmetic_flags() -> None:
    assert KNOWN_COMPRESSORS == (
        "streaming_llm",
        "knorm",
        "expected_attention",
    )
    row = _row("p")
    row["damaged"] = 0
    with pytest.raises(ValueError, match="damaged flag"):
        validate_rows([row], compressors=["streaming_llm"])


def test_source_manifest_is_required() -> None:
    metadata = {
        b"herald.validation_status": b"validated",
        b"herald.sweep_config_sha256": b"0" * 64,
    }
    with pytest.raises(ValueError, match="source manifest"):
        validate_rows(
            [_row("p")],
            compressors=["streaming_llm"],
            metadata=metadata,
            require_provenance=True,
        )


def test_conflicting_provenance_status_is_rejected() -> None:
    metadata = {
        b"herald.validation_status": b"validated",
        b"herald.dataset_status": b"failed",
        b"herald.sweep_config_sha256": b"0" * 64,
        b"herald.source_manifest_sha256": b"1" * 64,
    }
    with pytest.raises(ValueError, match="conflicting status"):
        validate_rows(
            [_row("p")],
            compressors=["streaming_llm"],
            metadata=metadata,
            require_provenance=True,
        )


def test_bootstrap_preserves_duplicate_prompt_draws() -> None:
    rows = [_row("p0"), _row("p1"), _row("p2")]
    samples = bootstrap_prompt_indices(rows, seed=0, resamples=20)
    assert any(len(sample) > len(set(sample.tolist())) for sample in samples)


def test_split_is_deterministic_and_prompt_disjoint() -> None:
    rows = [_row(f"p{i}") for i in range(24)]
    first = make_outer_split(rows, seed=9)
    second = make_outer_split(list(reversed(rows)), seed=9)
    assert first == second
    assert set(first.train_prompt_ids).isdisjoint(first.test_prompt_ids)


def test_compressors_must_share_the_same_switch_grid() -> None:
    rows = [
        _row("p0", compressor="streaming_llm", s=0),
        _row("p0", compressor="knorm", s=16),
    ]
    with pytest.raises(ValueError, match="switch grids differ"):
        validate_rows(rows, compressors=["streaming_llm", "knorm"])


def test_baseline_selection_uses_validation_rows() -> None:
    rows = [_row("p0", dq=0.9), _row("p1", dq=0.9)]
    baselines = {
        "global_mean": Baseline("global_mean", "mean", (), 0.1, {}),
        "ratio_mean": Baseline(
            "ratio_mean", "mean", ("ratio",), 0.1, {(0.5,): 0.9}
        ),
        "ratio_position_mean": Baseline(
            "ratio_position_mean",
            "mean",
            ("ratio", "position_bucket"),
            0.1,
            {(0.5, 0): 0.4},
        ),
        "global_median": Baseline("global_median", "median", (), 0.1, {}),
        "ratio_median": Baseline(
            "ratio_median", "median", ("ratio",), 0.1, {(0.5,): 0.9}
        ),
        "ratio_position_median": Baseline(
            "ratio_position_median",
            "median",
            ("ratio", "position_bucket"),
            0.1,
            {(0.5, 0): 0.4},
        ),
    }
    fit = CompressorFit("streaming_llm", (), baselines, None, (), {}, (), ())
    assert _select_baselines(fit, rows) == ("ratio_mean", "ratio_median")


def test_small_synthetic_fit_beats_coarse_baseline(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("xgboost")
    rows: list[dict[str, object]] = []
    for compressor in KNOWN_COMPRESSORS:
        for prompt_number in range(32):
            for ratio in (0.25, 0.5, 0.75):
                for position in (0, 16):
                    feature = float(prompt_number % 4)
                    dq = 0.05 + 0.08 * feature + 0.02 * ratio
                    rows.append(
                        {
                            **_row(
                                f"p{prompt_number}",
                                compressor=compressor,
                                s=position,
                                dq=dq,
                            ),
                            "ratio": ratio,
                            "feat__entropy": feature,
                        }
                    )
    evidence = fit_evidence(
        rows,
        compressors=KNOWN_COMPRESSORS,
        bootstrap_resamples=8,
    )
    for content in evidence.report["compressors"].values():
        evaluation = content["evaluation"]
        model = evaluation["models"]["xgboost"]["overall"]["mse"]
        coarse = evaluation["baselines"][evaluation["strongest_mean"]][
            "overall"
        ]["mse"]
        assert model < coarse
    bundle = tmp_path / "models"
    write_model_bundle(evidence, bundle)
    for compressor in KNOWN_COMPRESSORS:
        restored = load_model_bundle(bundle, compressor)
        assert restored.features == evidence.fits[compressor].features
        assert len(restored.xgb_models) == 3
