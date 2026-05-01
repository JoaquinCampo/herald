"""Parquet schemas + per-run writers + finalized-dataset readers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

RUNS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("prompt_id", pa.string()),
        pa.field("prompt_text", pa.string()),
        pa.field("prompt_hash", pa.string()),
        pa.field("model", pa.string()),
        pa.field("model_revision", pa.string()),
        pa.field("tokenizer_revision", pa.string()),
        pa.field("dtype", pa.string()),
        pa.field("device_class", pa.string()),
        pa.field("task", pa.string()),
        pa.field("press", pa.string()),
        pa.field("compression_ratio", pa.float64()),
        pa.field("max_new_tokens", pa.int32()),
        pa.field(
            "decoding_config",
            pa.map_(pa.string(), pa.string()),
        ),
        pa.field("seed", pa.int32()),
        pa.field("baseline_run_id", pa.string()),
        pa.field("generated_text", pa.string()),
        pa.field("generated_token_ids", pa.list_(pa.int32())),
        pa.field("num_tokens_generated", pa.int32()),
        pa.field("stop_reason", pa.string()),
        pa.field("predicted_answer", pa.string()),
        pa.field("ground_truth", pa.string()),
        pa.field("correct", pa.bool_()),
        pa.field("catastrophes", pa.list_(pa.string())),
        pa.field("replay_status", pa.string()),
        pa.field("replay_error", pa.string()),
        pa.field("created_at", pa.string()),
        pa.field("herald_git_sha", pa.string()),
    ]
)


TOKENS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("token_pos", pa.int32()),
        pa.field("token_id", pa.int32()),
        pa.field("token_str", pa.string()),
        pa.field("entropy", pa.float32()),
        pa.field("top1_prob", pa.float32()),
        pa.field("top5_prob", pa.float32()),
        pa.field("top5_logprobs", pa.list_(pa.float32())),
        pa.field("h_alts", pa.float32()),
        pa.field("avg_logp", pa.float32()),
        pa.field("delta_h", pa.float32()),
        pa.field("delta_h_valid", pa.bool_()),
        pa.field("kl_div", pa.float32()),
        pa.field("top10_jaccard", pa.float32()),
        pa.field("eff_vocab_size", pa.float32()),
        pa.field("tail_mass", pa.float32()),
        pa.field("logit_range", pa.float32()),
        pa.field("lookback_ratio", pa.float32()),
    ]
)


REPLAY_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string()),
        pa.field("token_pos", pa.int32()),
        pa.field("realized_token_id", pa.int32()),
        pa.field("union_top_k_token_ids", pa.list_(pa.int32())),
        pa.field("logprobs_compressed", pa.list_(pa.float32())),
        pa.field("logprobs_uncompressed", pa.list_(pa.float32())),
        pa.field("tail_mass_compressed", pa.float32()),
        pa.field("tail_mass_uncompressed", pa.float32()),
        pa.field("realized_logprob_compressed", pa.float32()),
        pa.field("realized_logprob_uncompressed", pa.float32()),
        pa.field("js_full", pa.float32()),
        pa.field("kl_unc_comp_full", pa.float32()),
        pa.field("kl_comp_unc_full", pa.float32()),
        pa.field("top1_match", pa.bool_()),
        pa.field("top1_rank_comp_under_unc", pa.int32()),
        pa.field("top1_rank_unc_under_comp", pa.int32()),
    ]
)


@dataclass(frozen=True)
class PerRunPaths:
    """Layout under results/phase0/raw/<kind>/<run_id>.parquet."""

    root: Path
    run_id: str

    @property
    def run(self) -> Path:
        return self.root / "raw" / "runs" / f"{self.run_id}.parquet"

    @property
    def tokens(self) -> Path:
        return self.root / "raw" / "tokens" / f"{self.run_id}.parquet"

    @property
    def replay(self) -> Path:
        return self.root / "raw" / "replay" / f"{self.run_id}.parquet"

    def all_exist(self) -> bool:
        return all(p.exists() for p in (self.run, self.tokens, self.replay))


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _coerce_decoding_config(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Stringify decoding_config values for the map<str,str> column."""
    record = dict(record)
    dc = record.get("decoding_config", {}) or {}
    record["decoding_config"] = [(str(k), str(v)) for k, v in dc.items()]
    return record


def write_run_record(record: dict[str, Any], path: Path) -> None:
    _ensure_parent(path)
    record = _coerce_decoding_config(record)
    table = pa.Table.from_pylist([record], schema=RUNS_SCHEMA)
    pq.write_table(table, path)


def write_tokens_rows(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    table = pa.Table.from_pylist(rows, schema=TOKENS_SCHEMA)
    pq.write_table(table, path)


def write_replay_rows(rows: list[dict[str, Any]], path: Path) -> None:
    _ensure_parent(path)
    table = pa.Table.from_pylist(rows, schema=REPLAY_SCHEMA)
    pq.write_table(table, path)


def read_runs(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def finalize_dataset(root: Path) -> None:
    """Concatenate raw/<kind>/*.parquet into final/."""
    raw = root / "raw"
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)

    runs_files = sorted((raw / "runs").glob("*.parquet"))
    if runs_files:
        runs_df = pl.concat(
            [pl.read_parquet(p) for p in runs_files],
            how="vertical_relaxed",
        )
        runs_df.write_parquet(final / "runs.parquet")

    for kind in ("tokens", "replay"):
        out_dir = final / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted((raw / kind).glob("*.parquet"))
        if not files or not runs_files:
            continue
        runs_meta = pl.read_parquet(final / "runs.parquet").select(
            ["run_id", "press", "compression_ratio"]
        )
        for f in files:
            df = pl.read_parquet(f).join(runs_meta, on="run_id", how="left")
            for keys, part in df.group_by(["press", "compression_ratio"]):
                press, ratio = keys
                pdir = out_dir / f"press={press}" / f"ratio={ratio:.4f}"
                pdir.mkdir(parents=True, exist_ok=True)
                part.drop(["press", "compression_ratio"]).write_parquet(
                    pdir / f.name
                )
