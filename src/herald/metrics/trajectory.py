"""Trajectory-level metrics: NLL ratio, first-divergence-point, sum_KL."""

from pathlib import Path

import polars as pl


def build(final: Path, out: Path) -> None:
    parts = list((final / "replay").glob("press=*/ratio=*/*.parquet"))
    if not parts:
        return
    df = pl.concat([pl.read_parquet(p) for p in parts])

    grouped = df.group_by("run_id").agg(
        [
            pl.col("kl_unc_comp_full").sum().alias("sum_kl"),
            pl.col("js_full").sum().alias("sum_js"),
            (
                pl.col("realized_logprob_uncompressed").sum()
                - pl.col("realized_logprob_compressed").sum()
            ).alias("nll_ratio"),
        ]
    )

    fdp_rows = []
    for run_id, sub in df.sort("token_pos").group_by("run_id"):
        rid = run_id[0] if isinstance(run_id, tuple) else run_id
        mismatches = sub.filter(~pl.col("top1_match"))
        if mismatches.height > 0:
            fdp = int(mismatches["token_pos"][0])
        else:
            max_pos = sub["token_pos"].max()
            if max_pos is None:
                fdp = 0
            else:
                fdp = int(max_pos) + 1  # type: ignore[arg-type]
        fdp_rows.append({"run_id": rid, "first_divergence_point": fdp})
    fdp_df = pl.DataFrame(fdp_rows)

    grouped.join(fdp_df, on="run_id").write_parquet(
        out / "trajectory_metrics.parquet"
    )
