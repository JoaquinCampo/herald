"""Offline token metrics + truncation-bias audit."""

import math
from pathlib import Path

import polars as pl


def recompute_kl_from_top_k(
    log_p_c: list[float],
    log_p_u: list[float],
    tail_c: float,
    tail_u: float,
) -> tuple[float, float]:
    """KL(p_u || p_c) and JS from union top-K + tail bucket.

    Tail mass is treated as a single bucket; its contribution is
    `t * log(t/t')` for KL and the symmetric equivalent for JS.
    """
    p_c = [math.exp(lp) for lp in log_p_c]
    p_u = [math.exp(lp) for lp in log_p_u]
    kl_uc = 0.0
    js = 0.0
    for pc, pu, lpc, lpu in zip(p_c, p_u, log_p_c, log_p_u):
        if pu > 0:
            kl_uc += pu * (lpu - lpc)
        m = 0.5 * (pc + pu)
        if m > 0:
            if pc > 0:
                js += 0.5 * pc * (lpc - math.log(m))
            if pu > 0:
                js += 0.5 * pu * (lpu - math.log(m))
    if tail_c > 0 and tail_u > 0:
        kl_uc += tail_u * (math.log(tail_u) - math.log(tail_c))
        m = 0.5 * (tail_c + tail_u)
        js += 0.5 * tail_c * (math.log(tail_c) - math.log(m))
        js += 0.5 * tail_u * (math.log(tail_u) - math.log(m))
    return max(kl_uc, 0.0), max(js, 0.0)


def build(final: Path, out: Path) -> None:
    """Read replay parquet partitions, audit truncation bias."""
    replay_root = final / "replay"
    parts = list(replay_root.glob("press=*/ratio=*/*.parquet"))
    if not parts:
        return
    df = pl.concat([pl.read_parquet(p) for p in parts])

    rows = []
    for r in df.iter_rows(named=True):
        kl_topk, js_topk = recompute_kl_from_top_k(
            r["logprobs_uncompressed"],
            r["logprobs_compressed"],
            r["tail_mass_uncompressed"],
            r["tail_mass_compressed"],
        )
        rows.append(
            {
                "run_id": r["run_id"],
                "token_pos": r["token_pos"],
                "js_full": r["js_full"],
                "kl_unc_comp_full": r["kl_unc_comp_full"],
                "kl_comp_unc_full": r["kl_comp_unc_full"],
                "top1_match": r["top1_match"],
                "kl_topk": kl_topk,
                "js_topk": js_topk,
                "trunc_bias_kl": abs(r["kl_unc_comp_full"] - kl_topk),
                "trunc_bias_js": abs(r["js_full"] - js_topk),
            }
        )
    pl.DataFrame(rows).write_parquet(out / "token_metrics.parquet")
