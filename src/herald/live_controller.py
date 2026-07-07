"""Live grace-window controller (rung 2 mission).

Executes the grace-window policy during actual generation, with the
exact semantics the zero-GPU replay assumed (grace_replay):

- The reference decodes on an uncompressed cache, extended segment by
  segment with `past_key_values` chaining and one persistent
  FeatureCollector, so its logit stream is identical to a single
  `generate` call (kl_prev and all rolling dynamics stay continuous).
- At each grid point s (after reference token s exists, because the
  pre-switch features are the derived stream at row s), an attempt
  prefills `[prompt + ref[:s]]` inside the press context, the same
  code path that produced the recorded hybrid streams, and decodes.
- An AlarmStoppingCriteria evaluates the frozen alarm once, at the
  k-th post-switch token: on alarm the call stops (revert; the k
  observed tokens are the overhead), otherwise the same call simply
  keeps decoding to completion (commit costs nothing extra).
- On revert the run continues from the held uncompressed cache; on
  commit the episode output is `ref[:s] + attempt tokens` and the
  held cache is dropped.

Episodes record raw facts (attempted s, alarm scores, tokens,
timings, peak memory); savings/cost/fidelity are computed downstream
against the frozen fidelity targets.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from kvpress.presses.base_press import BasePress
from transformers import StoppingCriteria, StoppingCriteriaList

from herald.features import FeatureCollector, IncrementalDerived
from herald.generate import LoadedModel, build_input_ids
from herald.grace_window import (
    assemble_alarm_row_from_state,
    assemble_gate_row_from_state,
)
from herald.tasks import PromptRecord


class AlarmLike(Protocol):
    """What the controller needs from a frozen alarm (AlarmBundle)."""

    k: int

    def score(self, row: dict[str, Any]) -> float: ...

    def commits(self, score: float) -> bool: ...


class GateLike(Protocol):
    """What the controller needs from a frozen scorer gate."""

    def score(self, row: dict[str, Any]) -> float: ...

    def attempts(self, score: float) -> bool: ...


@dataclass
class AttemptRecord:
    s: int
    score: float
    committed: bool
    n_new_tokens: int
    block_len: int
    wall_s: float
    gate_score: float | None = None


@dataclass
class SkipRecord:
    s: int
    gate_score: float
    wall_s: float


@dataclass
class LiveEpisode:
    prompt_id: str
    compressor: str
    ratio: float
    commit_s: int | None
    ref_ids: list[int]
    ref_done: bool
    new_ids: list[int]
    text: str
    attempts: list[AttemptRecord] = field(default_factory=list)
    ref_wall_s: float = 0.0
    total_wall_s: float = 0.0
    peak_mem_bytes: int = 0
    skips: list[SkipRecord] = field(default_factory=list)


class AlarmStoppingCriteria(StoppingCriteria):
    """Stop an attempt at the k-th generated token iff the alarm fires.

    `decide` returns True to revert; it is called exactly once, at the
    step where k tokens exist. If generation ends earlier (EOS), the
    caller must invoke `decide_now` post-hoc on the partial block, the
    same rule the replay applied to short recorded streams.
    """

    def __init__(self, k: int, decide: Callable[[], bool]) -> None:
        self.k = k
        self._decide = decide
        self._steps = 0
        self.fired = False
        self.revert = False

    def decide_now(self) -> None:
        if not self.fired:
            self.fired = True
            self.revert = self._decide()

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        self._steps += 1
        if self._steps >= self.k:
            self.decide_now()
        return torch.full(
            (input_ids.shape[0],),
            self.fired and self.revert,
            dtype=torch.bool,
            device=input_ids.device,
        )


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _extend_reference(
    lm: LoadedModel,
    prompt_ids: torch.Tensor,
    ref_ids: list[int],
    cache: Any,
    collector: FeatureCollector,
    max_new: int,
) -> tuple[list[int], Any]:
    """Decode up to `max_new` more reference tokens on the held cache."""
    device = next(lm.model.parameters()).device
    full = torch.cat(
        [
            prompt_ids.to(device),
            torch.tensor(ref_ids, dtype=torch.long, device=device),
        ]
    ).unsqueeze(0)
    with torch.no_grad():
        out = lm.model.generate(  # type: ignore[operator]
            input_ids=full,
            attention_mask=torch.ones_like(full),
            generation_config=lm.gen_config,
            max_new_tokens=max_new,
            logits_processor=[collector],
            past_key_values=cache,
            return_dict_in_generate=True,
        )
    new_ids = out.sequences[0, full.shape[1] :].tolist()
    return [int(t) for t in new_ids], out.past_key_values


def _attempt(
    lm: LoadedModel,
    prompt_ids: torch.Tensor,
    prefix_ids: list[int],
    press: BasePress,
    alarm: AlarmLike,
    ref_state: IncrementalDerived,
    s: int,
    ratio: float,
    max_new: int,
    *,
    seed: int = 0,
) -> tuple[list[int], float, bool]:
    """One compression attempt at switch position s.

    Returns (new_ids, alarm_score, committed). The prefill fires the
    press once on the full-attention cache of [prompt + ref[:s]],
    exactly like the recorded hybrids; the alarm is evaluated at the
    k-th generated token inside the same generate call.
    """
    device = next(lm.model.parameters()).device
    seq = torch.cat(
        [
            prompt_ids.to(device),
            torch.tensor(prefix_ids, dtype=torch.long, device=device),
        ]
    ).unsqueeze(0)
    collector = FeatureCollector()
    holder: dict[str, float] = {}

    def decide() -> bool:
        block = collector.stacked()[: alarm.k, 0, :]
        row = assemble_alarm_row_from_state(
            state=ref_state, block=block, ratio=ratio, k=alarm.k
        )
        score = alarm.score(row)
        holder["score"] = score
        return not alarm.commits(score)

    criteria = AlarmStoppingCriteria(alarm.k, decide)
    torch.manual_seed(seed)
    with torch.no_grad(), press(lm.model):
        out = lm.model.generate(  # type: ignore[operator]
            input_ids=seq,
            attention_mask=torch.ones_like(seq),
            generation_config=lm.gen_config,
            max_new_tokens=max_new,
            logits_processor=[collector],
            stopping_criteria=StoppingCriteriaList([criteria]),
            return_dict_in_generate=True,
        )
    criteria.decide_now()
    new_ids = [int(t) for t in out.sequences[0, seq.shape[1] :].tolist()]
    return new_ids, holder["score"], not criteria.revert


def run_episode(
    lm: LoadedModel,
    record: PromptRecord,
    press_factory: Callable[[], BasePress],
    alarm: AlarmLike,
    *,
    compressor: str,
    ratio: float,
    max_new_tokens: int,
    stride: int = 16,
    gate: GateLike | None = None,
) -> LiveEpisode:
    """Run one live grace-window episode for one (prompt, ratio)."""
    device = next(lm.model.parameters()).device.type
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    prompt_ids = build_input_ids(lm, record)
    ref_collector = FeatureCollector()
    ref_state = IncrementalDerived()
    n_state_rows = 0
    ref_ids: list[int] = []
    cache: Any = None
    ref_done = False
    episode = LiveEpisode(
        prompt_id=record.prompt_id,
        compressor=compressor,
        ratio=ratio,
        commit_s=None,
        ref_ids=ref_ids,
        ref_done=False,
        new_ids=[],
        text="",
    )
    _sync(device)
    t_start = time.perf_counter()

    s = 0
    while True:
        target = s + 1
        while not ref_done and len(ref_ids) < target:
            budget = min(target - len(ref_ids), max_new_tokens - len(ref_ids))
            if budget <= 0:
                ref_done = True
                break
            _sync(device)
            t0 = time.perf_counter()
            new_ref, cache = _extend_reference(
                lm, prompt_ids, ref_ids, cache, ref_collector, budget
            )
            _sync(device)
            episode.ref_wall_s += time.perf_counter() - t0
            ref_ids.extend(new_ref)
            ref_raw = ref_collector.stacked()[:, 0, :]
            while n_state_rows < ref_raw.shape[0]:
                ref_state.update(ref_raw[n_state_rows])
                n_state_rows += 1
            if (
                len(new_ref) < budget
                or (new_ref and new_ref[-1] in lm.eos_ids)
                or len(ref_ids) >= max_new_tokens
            ):
                ref_done = True
        if len(ref_ids) < target:
            break  # reference ended before this grid point

        gate_score = None
        if gate is not None:
            t_gate = time.perf_counter()
            gate_row = assemble_gate_row_from_state(
                state=ref_state,
                ratio=ratio,
            )
            gate_score = gate.score(gate_row)
            gate_wall = time.perf_counter() - t_gate
            if not gate.attempts(gate_score):
                episode.skips.append(
                    SkipRecord(
                        s=s,
                        gate_score=gate_score,
                        wall_s=gate_wall,
                    )
                )
                s += stride
                continue

        _sync(device)
        t0 = time.perf_counter()
        new_ids, score, committed = _attempt(
            lm,
            prompt_ids,
            ref_ids[:s],
            press_factory(),
            alarm,
            ref_state,
            s,
            ratio,
            max_new_tokens - s,
        )
        _sync(device)
        wall = time.perf_counter() - t0
        episode.attempts.append(
            AttemptRecord(
                s=s,
                score=score,
                committed=committed,
                n_new_tokens=len(new_ids),
                block_len=min(alarm.k, len(new_ids)),
                wall_s=wall,
                gate_score=gate_score,
            )
        )
        if committed:
            episode.commit_s = s
            episode.new_ids = new_ids
            break
        s += stride

    _sync(device)
    episode.total_wall_s = time.perf_counter() - t_start
    episode.ref_done = ref_done
    if episode.commit_s is None:
        full_ids = list(ref_ids)
    else:
        full_ids = ref_ids[: episode.commit_s] + episode.new_ids
    episode.text = lm.tokenizer.decode(full_ids, skip_special_tokens=True)
    if device == "cuda":
        episode.peak_mem_bytes = int(torch.cuda.max_memory_allocated())
    return episode
