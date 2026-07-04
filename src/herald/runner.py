"""Sweep orchestration: generate references and hybrids, score, store.

Resumable and idempotent. References are generated once per (model,
prompt) and reused; hybrids are generated per (compressor, ratio, switch
position s) and batched across prompts at a fixed cell. Work already on
disk (per the storage manifests) is skipped, so a killed run resumes
where it stopped.

Damage is the task-grounded quality delta dq(s) = q(reference) -
q(hybrid at s). Quality q is the deterministic task scorer; the judge
layer is downstream and out of scope here.
"""

import json
import sys
import time
from collections.abc import Iterator, Sequence
from typing import cast

import numpy as np

from herald import storage
from herald.config import TASKS, Config
from herald.generate import (
    LoadedModel,
    ReferenceRun,
    generate_hybrids,
    generate_reference,
    load_model,
    switch_positions,
)
from herald.presses import get_press
from herald.scoring import score
from herald.tasks import PromptRecord, load_prompts


def _log(event: str, **fields: object) -> None:
    """Emit one structured JSON log line to stdout."""
    rec = {"t": round(time.time(), 3), "event": event, **fields}
    sys.stdout.write(json.dumps(rec) + "\n")
    sys.stdout.flush()


def _chunks[T](items: Sequence[T], size: int) -> Iterator[list[T]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def run_sweep(config: Config, *, device: str = "cuda") -> None:
    """Run the full sweep described by `config`, resumably."""
    for model_key in config.models:
        _log("model_load", model=model_key)
        lm = load_model(
            model_key,
            dtype=config.dtype,
            device=device,
            attn_implementation=config.attn_implementation,
        )
        for task in config.tasks:
            records = load_prompts(task, config.prompts_per_task, TASKS[task])
            _run_references(lm, task, records, config)
            _run_hybrids(lm, task, records, config)
            _log("task_done", model=model_key, task=task)
    _log("sweep_done")


def _run_references(
    lm: LoadedModel,
    task: str,
    records: list[PromptRecord],
    config: Config,
) -> None:
    m = TASKS[task].max_new_tokens
    done = storage.reference_done(config.results_dir, lm.key, task)
    todo = [r for r in records if r.prompt_id not in done]
    _log(
        "references_start",
        model=lm.key,
        task=task,
        total=len(records),
        todo=len(todo),
    )
    for batch in _chunks(todo, config.ref_batch_size):
        # Isolate failures per batch: a poison item (OOM, a degenerate
        # prompt) must not abort the unattended sweep. Unwritten items
        # stay un-done and are retried on the next resume.
        try:
            refs = generate_reference(lm, batch, m)
            for ref, rec in zip(refs, batch, strict=True):
                q = score(task, ref.text, rec.gold)
                storage.save_reference(
                    config.results_dir,
                    lm.key,
                    task,
                    prompt_id=ref.prompt_id,
                    prompt_input_ids=ref.prompt_input_ids,
                    gen_ids=ref.gen_ids,
                    text=ref.text,
                    q=q,
                    features=ref.features,
                )
                _log(
                    "reference_done",
                    model=lm.key,
                    task=task,
                    prompt_id=ref.prompt_id,
                    run_len=len(ref.gen_ids),
                    q=q,
                )
        except Exception as exc:  # noqa: BLE001
            _log(
                "batch_error",
                phase="reference",
                model=lm.key,
                task=task,
                prompt_ids=[r.prompt_id for r in batch],
                error=f"{type(exc).__name__}: {exc}",
            )


def _run_hybrids(
    lm: LoadedModel,
    task: str,
    records: list[PromptRecord],
    config: Config,
) -> None:
    m = TASKS[task].max_new_tokens
    done_refs = storage.reference_done(config.results_dir, lm.key, task)
    gold = {r.prompt_id: r.gold for r in records}
    q_ref: dict[str, float] = {}
    refs: dict[str, ReferenceRun] = {}
    for r in records:
        if r.prompt_id not in done_refs:
            continue
        d = storage.load_reference(
            config.results_dir, lm.key, task, r.prompt_id
        )
        # Features are not needed to generate hybrids, only the prompt
        # and reference tokens, so the reloaded run carries none.
        refs[r.prompt_id] = ReferenceRun(
            prompt_id=r.prompt_id,
            prompt_input_ids=cast(list[int], d["prompt_input_ids"]),
            gen_ids=cast(list[int], d["gen_ids"]),
            text=str(d["text"]),
            features=np.empty((0, 0), dtype=np.float32),
        )
        q_ref[r.prompt_id] = cast(float, d["q"])

    for compressor in config.compressors:
        for ratio in config.ratios:
            done = storage.hybrid_done(
                config.results_dir,
                lm.key,
                task,
                compressor,
                ratio,
                require_features=True,
            )
            by_s: dict[int, list[tuple[ReferenceRun, int]]] = {}
            for pid, ref in refs.items():
                length = len(ref.gen_ids)
                for s in switch_positions(length, config.switch_stride):
                    if s >= length:
                        continue  # s == run length is the reference
                    if (pid, s) in done:
                        continue
                    by_s.setdefault(s, []).append((ref, s))

            n_cells = sum(len(v) for v in by_s.values())
            _log(
                "hybrids_start",
                model=lm.key,
                task=task,
                compressor=compressor,
                ratio=ratio,
                todo=n_cells,
            )
            for s in sorted(by_s):
                for batch in _chunks(by_s[s], config.hybrid_batch_size):
                    try:
                        press = get_press(compressor, ratio)
                        hybrids = generate_hybrids(
                            lm,
                            batch,
                            compressor,
                            ratio,
                            press,
                            m,
                            seed=config.seed,
                        )
                        for (ref, _), hyb in zip(batch, hybrids, strict=True):
                            q_hyb = score(task, hyb.text, gold[ref.prompt_id])
                            dq = q_ref[ref.prompt_id] - q_hyb
                            storage.append_hybrid(
                                config.results_dir,
                                lm.key,
                                task,
                                compressor,
                                ratio,
                                prompt_id=ref.prompt_id,
                                s=s,
                                new_ids=hyb.new_ids,
                                text=hyb.text,
                                q=q_hyb,
                                dq=dq,
                                features=hyb.features,
                            )
                            _log(
                                "hybrid_done",
                                model=lm.key,
                                task=task,
                                compressor=compressor,
                                ratio=ratio,
                                prompt_id=ref.prompt_id,
                                s=s,
                                dq=dq,
                            )
                    except Exception as exc:  # noqa: BLE001
                        _log(
                            "batch_error",
                            phase="hybrid",
                            model=lm.key,
                            task=task,
                            compressor=compressor,
                            ratio=ratio,
                            s=s,
                            prompt_ids=[r.prompt_id for r, _ in batch],
                            error=f"{type(exc).__name__}: {exc}",
                        )
