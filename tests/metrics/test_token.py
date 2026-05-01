import math
from pathlib import Path

import polars as pl

from herald.metrics.token import build, recompute_kl_from_top_k


def test_recompute_kl_from_top_k_zero_when_dists_equal():
    lp_c = [-0.1, -2.0, -3.0]
    lp_u = [-0.1, -2.0, -3.0]
    kl, js = recompute_kl_from_top_k(lp_c, lp_u, 0.0, 0.0)
    assert kl < 1e-9
    assert js < 1e-9


def test_recompute_kl_from_top_k_handles_tail():
    lp_c = [math.log(0.6), math.log(0.3)]
    lp_u = [math.log(0.5), math.log(0.4)]
    kl, js = recompute_kl_from_top_k(lp_c, lp_u, tail_c=0.1, tail_u=0.1)
    assert kl > 0
    assert js > 0


def test_build_writes_token_metrics(tmp_path: Path):
    final = tmp_path / "final"
    (final / "replay" / "press=none" / "ratio=0.0000").mkdir(parents=True)
    df = pl.DataFrame(
        {
            "run_id": ["r1"],
            "token_pos": [0],
            "realized_token_id": [5],
            "union_top_k_token_ids": [[5, 7]],
            "logprobs_compressed": [[-0.1, -2.0]],
            "logprobs_uncompressed": [[-0.1, -2.0]],
            "tail_mass_compressed": [0.0],
            "tail_mass_uncompressed": [0.0],
            "realized_logprob_compressed": [-0.1],
            "realized_logprob_uncompressed": [-0.1],
            "js_full": [0.0],
            "kl_unc_comp_full": [0.0],
            "kl_comp_unc_full": [0.0],
            "top1_match": [True],
            "top1_rank_comp_under_unc": [0],
            "top1_rank_unc_under_comp": [0],
        }
    )
    df.write_parquet(
        final / "replay" / "press=none" / "ratio=0.0000" / "r1.parquet"
    )
    out = tmp_path / "metrics"
    out.mkdir()
    build(final, out)

    written = pl.read_parquet(out / "token_metrics.parquet")
    assert written.height == 1
    assert "trunc_bias_kl" in written.columns
    assert written["trunc_bias_kl"][0] < 1e-6
