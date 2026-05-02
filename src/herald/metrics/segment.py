"""Segment-level aggregates for K in {8, 16, 32}.

Offline, CPU-only. Operates on `tokens.parquet` (per-run, written by
`herald.metrics.io.write_tokens_rows`). Produces a single
`segments.parquet` with a `K` column so downstream code joins on
`(run_id, K, window_idx)` without juggling three files.

What this module does NOT compute:
- H-horizon partial sums over `[t+1, t+H]`. Those are predictor
  targets (Phase 2) and live closer to the training pipeline; they
  are constructed from `tokens.parquet` + `replay.parquet` directly.
- Repetition / progress markers that need decoded token strings or
  ground-truth comparison. Those land when the segment-level features
  feed into Phase 2 features.
"""

from pathlib import Path
from typing import Final

import polars as pl
import pyarrow as pa

# Tier-0 numeric columns from `TOKENS_SCHEMA` we aggregate. Keep this
# list explicit; polars' implicit-column behavior surprises later.
TIER0_FEATURES: Final[tuple[str, ...]] = (
    "entropy",
    "top1_prob",
    "top5_prob",
    "h_alts",
    "avg_logp",
    "delta_h",
    "kl_div",
    "top10_jaccard",
    "eff_vocab_size",
    "tail_mass",
    "logit_range",
)

DEFAULT_KS: Final[tuple[int, ...]] = (8, 16, 32)

# EWMA decay used for the per-window "ewma_last" aggregate. Matches
# the alpha used in the Phase 0 cheap-predictor sanity (Analysis #8).
EWMA_ALPHA: Final[float] = 0.1


def _segments_pa_schema() -> pa.Schema:
    fields: list[pa.Field] = [
        pa.field("run_id", pa.string()),
        pa.field("K", pa.int32()),
        pa.field("window_idx", pa.int32()),
        pa.field("window_start", pa.int32()),
        pa.field("window_end", pa.int32()),
        pa.field("n_tokens", pa.int32()),
    ]
    for col in TIER0_FEATURES:
        for stat in ("mean", "std", "min", "max", "p95", "ewma_last"):
            fields.append(pa.field(f"{col}_{stat}", pa.float32()))
    return pa.schema(fields)


SEGMENTS_SCHEMA: Final[pa.Schema] = _segments_pa_schema()


def _ewma_last(values: pl.Series, alpha: float) -> float:
    """Last value of an alpha-EWMA over `values`. NaN if empty."""
    if values.len() == 0:
        return float("nan")
    out = 0.0
    seen = False
    for v in values.to_list():
        if v is None:
            continue
            # NaN-coerced values are floats and pass through below.
        fv = float(v)
        out = fv if not seen else (alpha * fv + (1.0 - alpha) * out)
        seen = True
    return out if seen else float("nan")


def aggregate_run_segments(tokens: pl.DataFrame, K: int) -> pl.DataFrame:
    """Aggregate Tier-0 features into K-token windows for one run.

    `tokens` must be sorted by `token_pos` (ascending) and contain a
    single `run_id`. Returns one row per window.
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if tokens.is_empty():
        return pl.DataFrame(schema=_polars_segments_schema())
    run_ids = tokens["run_id"].unique()
    if len(run_ids) != 1:
        raise ValueError(
            f"aggregate_run_segments expects a single run_id, "
            f"got {len(run_ids)}"
        )
    run_id = run_ids[0]

    sorted_tokens = tokens.sort("token_pos")
    n = sorted_tokens.height
    rows: list[dict[str, object]] = []
    for w_start in range(0, n, K):
        w_end = min(w_start + K, n)
        window = sorted_tokens.slice(w_start, w_end - w_start)
        row: dict[str, object] = {
            "run_id": str(run_id),
            "K": int(K),
            "window_idx": int(w_start // K),
            "window_start": int(w_start),
            "window_end": int(w_end),
            "n_tokens": int(window.height),
        }
        for col in TIER0_FEATURES:
            series = window[col].cast(pl.Float64)
            row[f"{col}_mean"] = _safe_float(series.mean())
            row[f"{col}_std"] = _safe_float(series.std(ddof=0))
            row[f"{col}_min"] = _safe_float(series.min())
            row[f"{col}_max"] = _safe_float(series.max())
            row[f"{col}_p95"] = _safe_float(
                series.quantile(0.95, interpolation="linear")
            )
            row[f"{col}_ewma_last"] = _ewma_last(series, EWMA_ALPHA)
        rows.append(row)
    return pl.DataFrame(rows, schema=_polars_segments_schema())


def aggregate_all_segments(
    tokens_dir: Path,
    Ks: tuple[int, ...] = DEFAULT_KS,
) -> pl.DataFrame:
    """Aggregate every per-run tokens file under `tokens_dir`.

    `tokens_dir` is the directory containing `*.parquet` files (one
    per run); typically `<output_root>/raw/tokens/` written by
    `herald.metrics.io.write_tokens_rows`.
    """
    files = sorted(tokens_dir.glob("*.parquet"))
    if not files:
        return pl.DataFrame(schema=_polars_segments_schema())
    parts: list[pl.DataFrame] = []
    for f in files:
        tokens = pl.read_parquet(f)
        if tokens.is_empty():
            continue
        for K in Ks:
            parts.append(aggregate_run_segments(tokens, K=K))
    if not parts:
        return pl.DataFrame(schema=_polars_segments_schema())
    return pl.concat(parts, how="vertical")


def write_segments(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _safe_float(value: object) -> float:
    if value is None:
        return float("nan")
    return float(value)  # type: ignore[arg-type]


def _polars_segments_schema() -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "run_id": pl.Utf8(),
        "K": pl.Int32(),
        "window_idx": pl.Int32(),
        "window_start": pl.Int32(),
        "window_end": pl.Int32(),
        "n_tokens": pl.Int32(),
    }
    for col in TIER0_FEATURES:
        for stat in ("mean", "std", "min", "max", "p95", "ewma_last"):
            schema[f"{col}_{stat}"] = pl.Float32()
    return schema
