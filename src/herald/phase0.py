"""Phase 0 orchestration: prompt manifest + run sweeper.

Sweep orchestration lives here too (added in Task 9).
"""

import gc
import random
from pathlib import Path
from typing import Any

import polars as pl
import torch
from loguru import logger
from pydantic import BaseModel

from herald.config import (
    ExperimentConfig,
    compute_prompt_hash,
    make_run_id,
)
from herald.experiment import (
    get_press,
    load_model,
    run_single_with_replay,
)
from herald.metrics.io import PerRunPaths
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


def _resolve_prompts(
    manifest_path: Path, num_prompts: int | None
) -> list[dict[str, Any]]:
    manifest = load_manifest(manifest_path)
    order = [e.prompt_id for e in manifest.entries]
    wanted = set(order)
    full = DEFAULT_TASK.load(num_prompts=10_000, seed=manifest.seed)
    prompts = [p for p in full if p["id"] in wanted]
    prompts.sort(key=lambda p: order.index(p["id"]))
    if num_prompts is not None:
        prompts = prompts[:num_prompts]
    return prompts


def _maybe_skip(paths: PerRunPaths) -> bool:
    if not paths.all_exist():
        return False
    df = pl.read_parquet(paths.run)
    return df.height == 1 and df["replay_status"][0] == "ok"


def run_phase0_sweep(
    model_name: str,
    manifest_path: Path,
    presses: tuple[str, ...] = ("streaming_llm", "snapkv"),
    ratios: tuple[float, ...] = (0.5, 0.875, 0.9375),
    max_new_tokens: int = 512,
    output_root: Path = Path("results/phase0"),
    prompt_timeout_seconds: float = 300.0,
    num_prompts: int | None = None,
    top_k: int = 128,
    seed: int = 42,
) -> None:
    """Phase 0 orchestration: baselines first, then (press, ratio) cells."""
    prompts = _resolve_prompts(manifest_path, num_prompts)
    output_root.mkdir(parents=True, exist_ok=True)

    cfg_template = ExperimentConfig(
        model_name=model_name,
        press_name="none",
        compression_ratio=0.0,
        num_prompts=len(prompts),
        seed=seed,
        output_dir=output_root,
        max_new_tokens=max_new_tokens,
        prompt_timeout_seconds=prompt_timeout_seconds,
    )
    model, tok, device = load_model(cfg_template)

    logger.info("Phase 0: baselines first (press=none).")
    for p in prompts:
        run_id = make_run_id(p["id"], "none", 0.0, seed)
        paths = PerRunPaths(root=output_root, run_id=run_id)
        if _maybe_skip(paths):
            logger.info(f"  skip baseline {p['id']} (existing ok)")
            continue
        run_single_with_replay(
            model=model,
            tokenizer=tok,
            device=device,
            prompt_data=p,
            config=cfg_template,
            press=None,
            baseline_run_id=None,
            output_root=output_root,
            top_k=top_k,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for press_name in presses:
        for ratio in ratios:
            cfg = cfg_template.model_copy(
                update={
                    "press_name": press_name,
                    "compression_ratio": float(ratio),
                }
            )
            press = get_press(press_name, ratio)
            logger.info(f"Phase 0 cell: {press_name}@{ratio}")
            for p in prompts:
                run_id = make_run_id(p["id"], press_name, ratio, seed)
                baseline_id = make_run_id(p["id"], "none", 0.0, seed)
                paths = PerRunPaths(root=output_root, run_id=run_id)
                if _maybe_skip(paths):
                    logger.info(f"  skip {press_name}@{ratio} {p['id']}")
                    continue
                run_single_with_replay(
                    model=model,
                    tokenizer=tok,
                    device=device,
                    prompt_data=p,
                    config=cfg,
                    press=press,
                    baseline_run_id=baseline_id,
                    output_root=output_root,
                    top_k=top_k,
                )
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
