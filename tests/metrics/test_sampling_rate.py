import json
from pathlib import Path

import polars as pl

from herald.metrics.sampling_rate import run_study


def test_run_study_emits_report(tmp_path: Path):
    final = tmp_path / "final"
    rep = final / "replay" / "press=snapkv" / "ratio=0.8750"
    rep.mkdir(parents=True)
    n = 40
    pl.DataFrame(
        {
            "run_id": ["r1"] * n,
            "token_pos": list(range(n)),
            "js_full": [0.01 + 0.001 * i for i in range(n)],
            "top1_match": [True] * n,
        }
    ).write_parquet(rep / "r1.parquet")
    out = tmp_path / "metrics"
    out.mkdir()
    run_study(final, out, horizons=(5, 10), rates=(1, 4, 8))
    report = json.loads((out / "sampling_rate_report.json").read_text())
    assert "spearman_future_max_js" in report
    assert "trajectory_rank_corr" in report
