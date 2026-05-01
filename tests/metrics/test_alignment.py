from pathlib import Path

import polars as pl

from herald.metrics.alignment import build


def test_alignment_matrix_includes_pairs(tmp_path: Path):
    out = tmp_path / "metrics"
    out.mkdir()
    pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "sum_kl": [0.1, 0.4, 0.2, 0.6],
            "first_divergence_point": [10, 5, 8, 3],
            "nll_ratio": [0.0, 0.5, 0.1, 0.7],
        }
    ).write_parquet(out / "trajectory_metrics.parquet")
    pl.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "rouge_l": [0.9, 0.5, 0.8, 0.3],
            "edit_distance_ratio": [0.05, 0.4, 0.1, 0.6],
            "embedding_cosine": [0.99, 0.7, 0.95, 0.6],
        }
    ).write_parquet(out / "sequence_metrics.parquet")

    build(out, out)

    align = pl.read_parquet(out / "alignment.parquet")
    pairs = set(zip(align["metric_a"], align["metric_b"]))
    assert ("sum_kl", "rouge_l") in pairs
    assert (
        align.filter(
            (pl.col("metric_a") == "sum_kl")
            & (pl.col("metric_b") == "rouge_l")
        )["spearman"][0]
        < 0
    )
