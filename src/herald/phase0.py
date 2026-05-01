"""Phase 0 orchestration: prompt manifest + run sweeper.

Sweep orchestration lives here too (added in Task 9).
"""

import random
from pathlib import Path

from pydantic import BaseModel

from herald.config import compute_prompt_hash
from herald.prompts import format_chat
from herald.tasks import DEFAULT_TASK


class ManifestEntry(BaseModel):
    prompt_id: str
    prompt_hash: str


class Manifest(BaseModel):
    name: str
    seed: int
    num_prompts: int
    entries: list[ManifestEntry]


def build_random_manifest(
    num_prompts: int, seed: int, out_path: Path
) -> Manifest:
    """Sample N prompts from the GSM8K test split with a fixed seed."""
    rng = random.Random(seed)
    full = DEFAULT_TASK.load(num_prompts=10_000, seed=seed)
    chosen = rng.sample(full, k=num_prompts)
    entries = []
    for p in chosen:
        chat_text = "".join(m["content"] for m in format_chat(p["question"]))
        entries.append(
            ManifestEntry(
                prompt_id=p["id"],
                prompt_hash=compute_prompt_hash(chat_text),
            )
        )
    manifest = Manifest(
        name="phase0-random",
        seed=seed,
        num_prompts=num_prompts,
        entries=entries,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(manifest.model_dump_json(indent=2))
    return manifest


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate_json(path.read_text())
