"""Integration tests: full pipeline from synthetic results → trained models.

These tests exercise the complete data flow:
  JSON results → features.build_dataset → labeling → train → LOCO CV
"""

import json
import math
import random
from pathlib import Path

import pytest

from herald.config import RunResult, TokenSignals
from herald.features import (
    add_rolling_features,
    build_dataset,
    flatten_signals,
)
from herald.labeling import create_horizon_labels, earliest_onset
from herald.train import train_predictor

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _make_signal(t: int, catastrophe_near: bool) -> TokenSignals:
    """Create a single TokenSignals with realistic-ish values."""
    noise = random.gauss(0, 0.1)
    entropy = 2.0 + noise
    if catastrophe_near:
        entropy += 1.5  # spike before failure

    return TokenSignals(
        entropy=round(entropy, 4),
        top1_prob=round(max(0.01, 0.5 - entropy * 0.05), 4),
        top5_prob=round(min(0.99, 0.85 + noise * 0.1), 4),
        top5_logprobs=[round(-0.5 - k * 0.5 + noise, 3) for k in range(5)],
        h_alts=round(max(0, 0.8 + noise), 4),
        avg_logp=round(-5.0 + noise, 4),
        delta_h=0.0 if t == 0 else round(random.gauss(0, 0.2), 4),
        delta_h_valid=t > 0,
        kl_div=(
            0.0 if t == 0 else round(max(0, random.gauss(0.05, 0.05)), 4)
        ),
        top10_jaccard=(
            0.0 if t == 0 else round(max(0, min(1, 0.7 + noise)), 4)
        ),
        eff_vocab_size=round(math.exp(entropy), 2),
        tail_mass=round(max(0, 0.05 + noise * 0.02), 4),
        logit_range=round(max(0, 15.0 + noise * 2), 4),
    )


def _create_sweep_results(
    results_dir: Path,
    presses: list[str],
    ratios: list[float],
    n_prompts: int = 8,
    tokens_range: tuple[int, int] = (60, 120),
) -> None:
    """Write synthetic sweep result JSON files."""
    random.seed(42)

    for press in presses:
        for ratio in ratios:
            press_dir = results_dir / press
            press_dir.mkdir(parents=True, exist_ok=True)

            results = []
            for i in range(n_prompts):
                cat_prob = ratio * 0.5
                has_cat = random.random() < cat_prob
                n_tok = random.randint(*tokens_range)

                onset_pos = int(n_tok * 0.6) if has_cat else None
                sigs = [
                    _make_signal(
                        t,
                        catastrophe_near=(
                            has_cat
                            and onset_pos is not None
                            and t > onset_pos - 15
                        ),
                    )
                    for t in range(n_tok)
                ]

                cats = ["looping"] if has_cat else []
                onsets = {"looping": onset_pos} if onset_pos else {}

                results.append(
                    RunResult(
                        prompt_id=f"gsm8k_{i}",
                        prompt_text="Solve: 2+2",
                        model="test-model",
                        press=press,
                        compression_ratio=ratio,
                        seed=42,
                        generated_text="#### 4",
                        ground_truth="4",
                        predicted_answer="4",
                        correct=not has_cat,
                        stop_reason=("max_tokens" if has_cat else "eos"),
                        catastrophes=cats,
                        num_tokens_generated=n_tok,
                        catastrophe_onsets=onsets,
                        signals=sigs,
                    )
                )

            fname = f"test-model_{ratio:.3f}_{n_prompts}p.json"
            data = {
                "config": {
                    "press_name": press,
                    "compression_ratio": ratio,
                },
                "summary": {},
                "results": [r.model_dump(mode="json") for r in results],
            }
            (press_dir / fname).write_text(json.dumps(data))


# -------------------------------------------------------------------
# Integration: features pipeline
# -------------------------------------------------------------------


class TestFeaturesPipeline:
    def test_flatten_then_rolling_produces_42_features(self):
        random.seed(0)
        sigs = [_make_signal(t, False) for t in range(50)]
        rows = flatten_signals(sigs, max_new_tokens=512)
        rows = add_rolling_features(rows)
        assert len(rows) == 50
        assert len(rows[0]) == 42
        # All values are finite floats
        for row in rows:
            for k, v in row.items():
                assert isinstance(v, float), f"{k} is {type(v)}"
                assert math.isfinite(v), f"{k} = {v}"

    def test_token_position_stays_bounded(self):
        random.seed(0)
        sigs = [_make_signal(t, False) for t in range(512)]
        rows = flatten_signals(sigs, max_new_tokens=512)
        # Position should be in [0, 1)
        assert rows[0]["token_position"] == 0.0
        assert rows[-1]["token_position"] < 1.0
        assert rows[-1]["token_position"] == 511.0 / 512.0


# -------------------------------------------------------------------
# Integration: labeling with NT proxy
# -------------------------------------------------------------------


class TestLabelingIntegration:
    def test_nt_proxy_onset_used(self):
        """Non-termination should use proxy, not last token."""
        onset = earliest_onset(
            catastrophe_onsets={},
            catastrophes=["non_termination"],
            max_new_tokens=512,
            nt_onset_frac=0.75,
            n_tokens=512,
        )
        assert onset == 384  # 0.75 * 512

        labels = create_horizon_labels(512, onset, horizon=10)
        # Token 373 (384-10-1) should be 0, token 374 should be 1
        assert labels[373] == 0
        assert labels[374] == 1
        assert labels[384] == 1
        assert labels[511] == 1

    def test_looping_onset_preferred_over_nt_proxy(self):
        """When both present, earliest wins."""
        onset = earliest_onset(
            catastrophe_onsets={"looping": 100},
            catastrophes=["looping", "non_termination"],
            max_new_tokens=512,
        )
        assert onset == 100  # looping at 100 < NT proxy at 384


# -------------------------------------------------------------------
# Integration: build_dataset
# -------------------------------------------------------------------


class TestBuildDataset:
    @pytest.fixture()
    def sweep_dir(self, tmp_path: Path) -> Path:
        results_dir = tmp_path / "results"
        _create_sweep_results(
            results_dir,
            presses=["streaming_llm", "snapkv", "knorm"],
            ratios=[0.5, 0.75],
            n_prompts=6,
        )
        return results_dir

    def test_returns_correct_structure(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        assert (
            len(ds.X)
            == len(ds.y)
            == len(ds.run_ids)
            == len(ds.press_ids)
            == len(ds.pre_onset)
            == len(ds.seq_ids)
        )
        assert len(ds.X) > 0
        assert len(ds.feat_names) == 42
        assert all(len(row) == 42 for row in ds.X)

    def test_labels_are_binary(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        assert set(ds.y).issubset({0, 1})

    def test_has_both_classes(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        assert 0 in ds.y and 1 in ds.y

    def test_press_ids_match_dirs(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        assert set(ds.press_ids) == {
            "streaming_llm",
            "snapkv",
            "knorm",
        }

    def test_all_features_finite(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        for row in ds.X:
            for v in row:
                assert math.isfinite(v)

    def test_pre_onset_mask(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        # Censored sequences: all tokens pre-onset
        for sid, onset in ds.seq_onsets.items():
            idxs = [i for i, s in enumerate(ds.seq_ids) if s == sid]
            if onset is None:
                assert all(ds.pre_onset[i] for i in idxs)
            else:
                # Tokens before onset are pre-onset
                for i in idxs:
                    t = i - idxs[0]
                    if t < onset:
                        assert ds.pre_onset[i]
                    else:
                        assert not ds.pre_onset[i]

    def test_seq_onsets_populated(self, sweep_dir: Path):
        ds = build_dataset(sweep_dir, horizon=10)
        assert len(ds.seq_onsets) > 0
        # Some sequences should have catastrophes
        has_cat = any(v is not None for v in ds.seq_onsets.values())
        assert has_cat


# -------------------------------------------------------------------
# Integration: full training pipeline
# -------------------------------------------------------------------


class TestFullPipeline:
    def test_train_with_loco_cv(self, tmp_path: Path):
        """End-to-end: create results → train → LOCO CV."""
        results_dir = tmp_path / "results"
        _create_sweep_results(
            results_dir,
            presses=["streaming_llm", "snapkv", "knorm"],
            ratios=[0.5, 0.75, 0.875],
            n_prompts=8,
        )

        output_dir = tmp_path / "models"
        metrics = train_predictor(results_dir, output_dir, horizons=[5, 10])

        # Both horizons trained
        assert "H5" in metrics
        assert "H10" in metrics

        # Model files exist
        assert (output_dir / "hazard_H5.json").exists()
        assert (output_dir / "hazard_H10.json").exists()
        assert (output_dir / "metrics.json").exists()

        # Metrics are populated (5-fold CV)
        for key in ("H5", "H10"):
            m = metrics[key]
            assert m["auroc_mean"] is not None  # type: ignore[index]
            assert m["auprc_mean"] is not None  # type: ignore[index]
            assert m["auroc_mean"] > 0.5  # type: ignore[operator]
            assert "fold_aurocs" in m  # type: ignore[operator]
            assert len(m["fold_aurocs"]) == 5  # type: ignore[arg-type]

        # LOCO CV ran (3 presses → 3 folds)
        h5 = metrics["H5"]
        assert "loco_cv" in h5  # type: ignore[operator]
        loco = h5["loco_cv"]  # type: ignore[index]
        assert len(loco["folds"]) == 3  # type: ignore[arg-type]
        assert loco["mean_auroc"] is not None  # type: ignore[index]

        # metrics.json is valid
        saved = json.loads((output_dir / "metrics.json").read_text())
        assert "H5" in saved
        assert "H10" in saved
