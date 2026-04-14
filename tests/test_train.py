"""Tests for herald.train — XGBoost hazard predictor training."""

import json
from pathlib import Path

from herald.train import (
    _baseline_cv,
    _cv_metrics,
    _train_single_horizon,
    train_predictor,
)


def _synthetic_dataset(
    n_sequences: int = 20,
    tokens_per_seq: int = 50,
    n_features: int = 30,
    event_rate: float = 0.3,
) -> tuple[
    list[list[float]],
    list[int],
    list[str],
    list[str],
    list[bool],
]:
    """Create a small synthetic dataset for testing."""
    import random

    random.seed(42)
    feat_names = [f"feat_{i}" for i in range(n_features)]
    # Use a real feature name so baseline can find it
    feat_names[1] = "entropy_mean_8"
    X: list[list[float]] = []
    y: list[int] = []
    run_ids: list[str] = []
    pre_onset: list[bool] = []

    for seq_idx in range(n_sequences):
        has_event = random.random() < event_rate
        onset = (
            random.randint(tokens_per_seq // 2, tokens_per_seq - 1)
            if has_event
            else None
        )
        run_id = f"seq_{seq_idx}"

        for t in range(tokens_per_seq):
            features = [random.gauss(0, 1) for _ in range(n_features)]
            # Add some signal: features correlate with proximity
            # to event
            if onset is not None and t > onset - 10:
                features[0] += 2.0  # make feature 0 predictive
                features[1] += 1.5  # entropy_mean_8 baseline signal
            X.append(features)
            label = 1 if (onset is not None and t >= onset - 5) else 0
            y.append(label)
            run_ids.append(run_id)
            is_pre = onset is None or t < onset
            pre_onset.append(is_pre)

    return X, y, run_ids, feat_names, pre_onset


class TestBaselineCV:
    def test_returns_metrics(self):
        X, y, run_ids, feat_names, pre_onset = _synthetic_dataset()
        result = _baseline_cv(
            X, y, run_ids, feat_names, pre_onset
        )

        assert result["feature"] == "entropy_mean_8"
        assert "auroc_mean" in result
        assert "auprc_mean" in result
        assert len(result["fold_aurocs"]) == 5  # type: ignore[arg-type]
        # Pre-onset track present
        assert "pre_onset" in result

    def test_missing_feature_returns_empty(self):
        X, y, run_ids, _, pre_onset = _synthetic_dataset()
        fake_names = [f"other_{i}" for i in range(30)]
        result = _baseline_cv(
            X, y, run_ids, fake_names, pre_onset
        )
        assert result == {}


class TestCVMetrics:
    def test_returns_fold_metrics(self):
        X, y, run_ids, feat_names, pre_onset = _synthetic_dataset()
        cv = _cv_metrics(X, y, run_ids, feat_names, pre_onset)

        assert "fold_aurocs" in cv
        assert "fold_auprcs" in cv
        assert len(cv["fold_aurocs"]) == 5  # type: ignore[arg-type]
        assert len(cv["fold_auprcs"]) == 5  # type: ignore[arg-type]
        assert "auroc_mean" in cv
        assert "auroc_std" in cv
        assert "auprc_mean" in cv
        assert "auprc_std" in cv
        assert "best_iteration" in cv
        # Pre-onset evaluation track
        assert "pre_onset" in cv
        pre = cv["pre_onset"]
        assert "auroc_mean" in pre  # type: ignore[operator]
        assert "auprc_mean" in pre  # type: ignore[operator]


class TestTrainSingleHorizon:
    def test_produces_model_and_metrics(self, tmp_path: Path):
        X, y, run_ids, feat_names, pre_onset = _synthetic_dataset()
        metrics = _train_single_horizon(
            X,
            y,
            run_ids,
            feat_names,
            pre_onset,
            horizon=10,
            output_dir=tmp_path,
        )

        # Model file created
        model_path = tmp_path / "hazard_H10.json"
        assert model_path.exists()

        # 5-fold CV metrics populated
        assert metrics["horizon"] == 10
        assert "n_samples" in metrics
        assert "auroc_mean" in metrics
        assert "auprc_mean" in metrics
        assert "fold_aurocs" in metrics
        assert len(metrics["fold_aurocs"]) == 5  # type: ignore[arg-type]

    def test_metrics_are_reasonable(self, tmp_path: Path):
        X, y, run_ids, feat_names, pre_onset = _synthetic_dataset()
        metrics = _train_single_horizon(
            X,
            y,
            run_ids,
            feat_names,
            pre_onset,
            horizon=10,
            output_dir=tmp_path,
        )
        # With synthetic signal, should do better than random
        if metrics["auroc_mean"] is not None:
            assert metrics["auroc_mean"] > 0.5


class TestTrainPredictor:
    def test_no_data_returns_empty(self, tmp_path: Path):
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output_dir = tmp_path / "models"

        metrics = train_predictor(results_dir, output_dir, horizons=[10])
        assert metrics == {}

    def test_with_synthetic_results(self, tmp_path: Path):
        """End-to-end test with multiple presses (exercises LOCO CV)."""
        from herald.config import RunResult, TokenSignals

        results_dir = tmp_path / "results"

        sigs = []
        for t in range(30):
            sigs.append(
                TokenSignals(
                    entropy=1.0 + t * 0.05,
                    top1_prob=0.5,
                    top5_prob=0.9,
                    top5_logprobs=[-0.5, -1.0, -1.5, -2.0, -2.5],
                    h_alts=0.8,
                    avg_logp=-5.0,
                    delta_h=0.0 if t == 0 else 0.05,
                    delta_h_valid=t > 0,
                    kl_div=0.0 if t == 0 else 0.01,
                    top10_jaccard=0.0 if t == 0 else 0.7,
                    eff_vocab_size=2.72,
                    tail_mass=0.05,
                    logit_range=15.0,
                )
            )

        # Create results across 3 presses to exercise LOCO CV
        presses = ["streaming_llm", "snapkv", "knorm"]
        for press in presses:
            press_dir = results_dir / press
            press_dir.mkdir(parents=True)

            results = []
            for i in range(6):
                has_cat = i % 3 == 0
                result = RunResult(
                    prompt_id=f"gsm8k_{i}",
                    prompt_text="test",
                    model="test-model",
                    press=press,
                    compression_ratio=0.5,
                    seed=42,
                    generated_text="#### 42",
                    ground_truth="42",
                    predicted_answer="42",
                    correct=not has_cat,
                    stop_reason=("max_tokens" if has_cat else "eos"),
                    catastrophes=(["looping"] if has_cat else []),
                    num_tokens_generated=30,
                    catastrophe_onsets=({"looping": 20} if has_cat else {}),
                    signals=sigs,
                )
                results.append(result)

            data = {
                "config": {},
                "summary": {},
                "results": [r.model_dump(mode="json") for r in results],
            }
            (press_dir / "test_results.json").write_text(json.dumps(data))

        output_dir = tmp_path / "models"
        metrics = train_predictor(results_dir, output_dir, horizons=[5])

        assert "H5" in metrics
        assert (output_dir / "hazard_H5.json").exists()
        assert (output_dir / "metrics.json").exists()

        # 5-fold CV metrics present
        h5 = metrics["H5"]
        assert "auroc_mean" in h5  # type: ignore[operator]
        assert "fold_aurocs" in h5  # type: ignore[operator]

        # LOCO CV should have run with 3 presses
        assert "loco_cv" in h5  # type: ignore[operator]
