"""Sequence-level metrics: edit distance, embedding cosine, ROUGE-L.

Requires the `metrics` optional install group. BERTScore is deferred
to a future flag once Phase 0 gates pass.
"""

from pathlib import Path
from typing import Any

import polars as pl


def _import_extras() -> tuple[Any, Any, Any, Any]:
    try:
        import editdistance
        from rouge_score import rouge_scorer
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise ImportError(
            "Sequence metrics require the [metrics] extra: "
            "uv sync --extra metrics"
        ) from exc
    return editdistance, rouge_scorer, SentenceTransformer, util


def build(final: Path, out: Path) -> None:
    ed_mod, rouge_mod, st_cls, util = _import_extras()

    runs = pl.read_parquet(final / "runs.parquet").select(
        [
            "run_id",
            "prompt_id",
            "press",
            "baseline_run_id",
            "generated_text",
        ]
    )
    base = (
        runs.filter(pl.col("press") == "none")
        .rename(
            {
                "run_id": "baseline_id_check",
                "generated_text": "baseline_text",
            }
        )
        .select(["baseline_id_check", "prompt_id", "baseline_text"])
    )
    paired = (
        runs.filter(pl.col("press") != "none")
        .join(base, on="prompt_id", how="left")
        .filter(pl.col("baseline_id_check") == pl.col("baseline_run_id"))
    )

    scorer = rouge_mod.RougeScorer(["rougeL"], use_stemmer=False)
    embedder = st_cls("all-MiniLM-L6-v2")

    rows = []
    for r in paired.iter_rows(named=True):
        comp_text = r["generated_text"]
        base_text = r["baseline_text"]
        rouge_l = scorer.score(base_text, comp_text)["rougeL"].fmeasure
        emb = embedder.encode([base_text, comp_text], convert_to_tensor=True)
        cos = float(util.cos_sim(emb[0], emb[1]).item())
        denom = max(len(base_text), len(comp_text), 1)
        ed_ratio = float(ed_mod.eval(base_text, comp_text)) / denom
        rows.append(
            {
                "run_id": r["run_id"],
                "prompt_id": r["prompt_id"],
                "rouge_l": rouge_l,
                "embedding_cosine": cos,
                "edit_distance_ratio": ed_ratio,
            }
        )
    pl.DataFrame(rows).write_parquet(out / "sequence_metrics.parquet")
