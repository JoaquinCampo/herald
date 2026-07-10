from pathlib import Path

from build_evidence import build_evidence

ROOT = Path(__file__).resolve().parents[3]


def test_live_campaign_metrics_match_current_artifacts() -> None:
    evidence = build_evidence(ROOT)
    assert evidence["campaign"]["episode_count"] == 552
    assert evidence["campaign"]["prompt_count"] == 46
    assert evidence["campaign"]["ratio_count"] == 4
    expected = {
        "expected_attention": (0.8092, 0.0027, 0.020),
        "knorm": (0.1271, 0.0118, 0.084),
        "streaming_llm": (0.3648, 0.0208, 0.051),
    }
    for compressor, values in expected.items():
        row = evidence["campaign"]["compressors"][compressor]
        assert round(row["compressed_generation_fraction"], 4) == values[0]
        assert round(row["quality_cost"], 4) == values[1]
        assert round(row["revert_wall_overhead"], 3) == values[2]


def test_gsm8k_example_is_the_real_s128_failure() -> None:
    example = build_evidence(ROOT)["gsm8k_example"]
    assert example["prompt_id"] == "gsm8k-0"
    assert example["switch_position"] == 128
    assert example["ratio"] == 0.75
    assert example["reference_answer"] == "18"
    assert example["compressed_answer"] == "3"
    assert example["compressed_quality"] == 0.0
    assert example["reference_excerpt"] == (
        "Money made = 9 (eggs left) * 2 (price per egg) = $18"
    )
    assert example["compressed_excerpt"] == (
        "Number of boxes = Eggs left / Eggs per box = 9 / 3 = 3 boxes"
    )
