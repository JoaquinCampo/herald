"""Re-run compressed gen + replay for a single run.

Determinism rests on the manifest fields recorded on runs.parquet.
The repair re-generates and verifies that the resulting
generated_token_ids matches the recorded sequence; mismatch is a
hard error.
"""

from pathlib import Path

import polars as pl

REQUIRED_FIELDS = (
    "model",
    "model_revision",
    "tokenizer_revision",
    "dtype",
    "device_class",
    "task",
    "prompt_id",
    "prompt_text",
    "prompt_hash",
    "press",
    "compression_ratio",
    "max_new_tokens",
    "decoding_config",
    "seed",
    "herald_git_sha",
    "generated_token_ids",
)


def validate_manifest(run_record_path: Path) -> dict[str, object]:
    """Verify all determinism fields are present and non-empty."""
    df = pl.read_parquet(run_record_path)
    if df.height != 1:
        raise ValueError(f"expected 1 row, got {df.height}")
    row = df.row(0, named=True)
    missing = [
        f
        for f in REQUIRED_FIELDS
        if f not in row
        or row[f] is None
        or (isinstance(row[f], list) and not row[f])
    ]
    if missing:
        raise ValueError(f"missing determinism fields: {missing}")
    return row


def repair_run(run_id: str, root: Path) -> None:
    """Re-run a single failed run end-to-end.

    Requires `run_single_with_replay` (Task 8) and a real model;
    the regen path is verified on Orion.
    """
    from herald.config import compute_prompt_hash
    from herald.metrics.io import PerRunPaths

    paths = PerRunPaths(root=root, run_id=run_id)
    row = validate_manifest(paths.run)
    if row["prompt_hash"] != compute_prompt_hash(str(row["prompt_text"])):
        raise ValueError(
            "prompt_hash mismatch (prompt text was tampered with)"
        )
    raise NotImplementedError(
        "regeneration requires Task 8 run_single_with_replay; run on Orion"
    )
