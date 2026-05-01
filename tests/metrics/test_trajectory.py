from pathlib import Path

import polars as pl

from herald.metrics.trajectory import build


def _make_replay(
    final: Path, run_id: str, press: str, ratio: float, n: int
) -> None:
    d = final / "replay" / f"press={press}" / f"ratio={ratio:.4f}"
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        {
            "run_id": [run_id] * n,
            "token_pos": list(range(n)),
            "realized_token_id": [1] * n,
            "union_top_k_token_ids": [[1, 2]] * n,
            "logprobs_compressed": [[-0.1, -2.0]] * n,
            "logprobs_uncompressed": [[-0.1, -2.0]] * n,
            "tail_mass_compressed": [0.0] * n,
            "tail_mass_uncompressed": [0.0] * n,
            "realized_logprob_compressed": [-0.5] * n,
            "realized_logprob_uncompressed": [-0.3] * n,
            "js_full": [0.05] * n,
            "kl_unc_comp_full": [0.04] * n,
            "kl_comp_unc_full": [0.04] * n,
            "top1_match": [True, True, False, True, True][:n],
            "top1_rank_comp_under_unc": [0, 0, 3, 0, 0][:n],
            "top1_rank_unc_under_comp": [0, 0, 3, 0, 0][:n],
        }
    )
    df.write_parquet(d / f"{run_id}.parquet")


def test_build_trajectory_metrics(tmp_path: Path):
    final = tmp_path / "final"
    _make_replay(final, "r1", "snapkv", 0.875, 5)

    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)

    df = pl.read_parquet(out / "trajectory_metrics.parquet")
    assert df.height == 1
    row = df.row(0, named=True)
    assert row["run_id"] == "r1"
    assert row["sum_kl"] > 0
    assert row["nll_ratio"] > 0
    assert row["first_divergence_point"] == 2
