"""Phase 4 minimal controller smoke.

Implements the smallest credible HERALD-as-controller loop:

  - per-token features via OnlineFeatureState (parity-tested vs Phase 2),
  - per-token risk via the exported lr_all_cheap predictor,
  - per-segment (K=16) mean risk -> RiskBudgetStep policy,
  - mutate `decoding_press.target_size` for the NEXT segment.

Companion baselines:
  - FixedBudgetPolicy(budget): no-op control, used for {64, 128, 256}.
  - RandomMatchedBudgetPolicy(actions, seed): replays HERALD's own
    per-segment action multiset with a seeded shuffle. Compute-matched
    by construction: same multiset of budgets, only the order differs.

Off-by-one we intentionally accept (and log explicitly): a budget
mutation at the end of segment S takes effect on the next press fire,
which is the FIRST compress of segment S+1. The `next_budget`
column in the segment log is the budget that segment S+1 will run
under, NOT the budget segment S used.

K MUST equal `compression_interval` so segment boundaries align with
press fire boundaries; otherwise a mutation lands one fire late.
"""

import math
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from herald.phase4_online_features import OnlineFeatureState
from herald.phase4_predictor import ExportedPredictor

if TYPE_CHECKING:
    import torch

DEFAULT_K = 16
DEFAULT_BUDGETS: tuple[int, ...] = (64, 128, 256)
DEFAULT_THRESHOLD = 0.5

# What the segment log records, in this exact order.
SEGMENT_COLUMNS: tuple[str, ...] = (
    "run_id",
    "policy",
    "segment_idx",
    "tokens_completed",
    "start_token_pos",
    "end_token_pos",
    "segment_score_mean",
    "segment_score_max",
    "threshold",
    "current_budget",
    "next_budget",
    "decision",
    "cache_size_observed",
    "evicted_tokens_in_segment",
)

RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "prompt_id",
    "policy",
    "initial_budget",
    "num_tokens_generated",
    "stop_reason",
    "wall_clock_seconds",
    "wall_clock_per_token",
    "peak_memory_mb",
    "compression_event_count",
    "decode_event_count",
    "total_evicted_tokens",
    "n_segments",
    "n_relax",
    "n_tighten",
    "n_keep",
    "task_score",
    "correct",
    "predicted_answer",
    "ground_truth",
    "generated_text",
)


# --- Policies --------------------------------------------------------


@dataclass(slots=True)
class FixedBudgetPolicy:
    """Always returns the same budget; control baseline."""

    budget: int
    name: str = "fixed"

    def initial_budget(self) -> int:
        return self.budget

    def choose(
        self,
        segment_idx: int,
        segment_score: float,
        last_budget: int,
    ) -> int:
        return self.budget


@dataclass(slots=True)
class RiskBudgetStepPolicy:
    """One-step-up if risk > threshold, one-step-down otherwise.

    Symmetric: at every segment boundary the policy moves exactly one
    step in the budget grid. Bounded by the grid endpoints.
    """

    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    threshold: float = DEFAULT_THRESHOLD
    start_index: int = 0  # 0 = most aggressive
    name: str = "herald_risk_budget_step"
    _idx: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.budgets:
            raise ValueError("budgets must be non-empty")
        if not (0 <= self.start_index < len(self.budgets)):
            raise ValueError("start_index out of range")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1]")
        self._idx = self.start_index

    def initial_budget(self) -> int:
        return self.budgets[self.start_index]

    def choose(
        self,
        segment_idx: int,
        segment_score: float,
        last_budget: int,
    ) -> int:
        if segment_score > self.threshold:
            self._idx = min(self._idx + 1, len(self.budgets) - 1)
        else:
            self._idx = max(self._idx - 1, 0)
        return self.budgets[self._idx]


@dataclass(slots=True)
class RandomMatchedBudgetPolicy:
    """Replay a HERALD action sequence with a seeded shuffle.

    `actions[i]` is the budget HERALD chose for segment i (i.e. the
    `next_budget` it set at the end of segment i-1; `actions[0]` is
    HERALD's initial budget). The random baseline shuffles this list
    with a seed and emits the shuffled sequence.

    Compute-matched: same multiset of budget assignments, different
    order. This isolates the "*which* segments to spend budget on"
    decision, the only thing the predictor influences.
    """

    actions: tuple[int, ...]
    seed: int = 42
    name: str = "random_matched"
    _shuffled: tuple[int, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        if not self.actions:
            raise ValueError("actions must be non-empty")
        rng = random.Random(self.seed)
        a = list(self.actions)
        rng.shuffle(a)
        self._shuffled = tuple(a)

    def initial_budget(self) -> int:
        return self._shuffled[0]

    def choose(
        self,
        segment_idx: int,
        segment_score: float,
        last_budget: int,
    ) -> int:
        nxt = segment_idx + 1
        if nxt >= len(self._shuffled):
            return self._shuffled[-1]
        return self._shuffled[nxt]


# --- LogitsProcessor -------------------------------------------------


@dataclass(slots=True)
class SegmentEntry:
    """One per-segment log entry, the unit `__call__` emits."""

    segment_idx: int
    tokens_completed: int
    segment_score_mean: float
    threshold: float
    current_budget: int
    next_budget: int
    decision: str  # relax | tighten | keep
    segment_score_max: float = 0.0


class HeraldOnlineProcessor:
    """transformers LogitsProcessor that drives the controller loop.

    On every step:
      1. extract per-token TokenSignals from the logits,
      2. update OnlineFeatureState,
      3. score with the exported predictor,
      4. buffer the score.

    Every K tokens:
      5. compute mean risk over the segment,
      6. ask the policy for next_budget,
      7. mutate `press.target_size` for the next press fire,
      8. append a SegmentEntry.

    The class deliberately does NOT inherit from
    `transformers.LogitsProcessor` — it duck-types `__call__(input_ids,
    scores) -> scores` so CPU tests don't need transformers installed.
    """

    def __init__(
        self,
        predictor: ExportedPredictor,
        online_state: OnlineFeatureState,
        policy: Any,
        press: Any,
        k: int = DEFAULT_K,
        threshold: float = DEFAULT_THRESHOLD,
        signal_extractor: Any | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be > 0")
        self._predictor = predictor
        self._online = online_state
        self._policy = policy
        self._press = press
        self._k = k
        self._threshold = threshold
        self._extract = signal_extractor
        self.token_idx = 0
        self.segment_idx = 0
        self.score_buffer: list[float] = []
        self.segments: list[SegmentEntry] = []
        self._prev_state: Any | None = None

    def __call__(
        self, input_ids: "torch.Tensor", scores: "torch.Tensor"
    ) -> "torch.Tensor":
        if self._extract is None:
            from herald.signals import extract_signals  # noqa: PLC0415

            self._extract = extract_signals
        sig, self._prev_state = self._extract(
            scores[0], prev=self._prev_state
        )
        row = self._online.update(sig)
        risk = self._predictor.score_one(row)
        self.score_buffer.append(float(risk))
        self.token_idx += 1
        if len(self.score_buffer) >= self._k:
            self._close_segment()
        return scores

    def _close_segment(self) -> None:
        mean_score = sum(self.score_buffer) / len(self.score_buffer)
        max_score = max(self.score_buffer)
        old_budget = int(getattr(self._press, "target_size", 0))
        new_budget = int(
            self._policy.choose(self.segment_idx, mean_score, old_budget)
        )
        if new_budget > old_budget:
            decision = "relax"
        elif new_budget < old_budget:
            decision = "tighten"
        else:
            decision = "keep"
        self.segments.append(
            SegmentEntry(
                segment_idx=self.segment_idx,
                tokens_completed=self.token_idx,
                segment_score_mean=mean_score,
                segment_score_max=max_score,
                threshold=self._threshold,
                current_budget=old_budget,
                next_budget=new_budget,
                decision=decision,
            )
        )
        self._press.target_size = new_budget
        self.score_buffer.clear()
        self.segment_idx += 1


# --- Run helper (Orion only) ----------------------------------------


@dataclass(slots=True)
class ControllerRun:
    run_id: str
    prompt_id: str
    policy_name: str
    initial_budget: int
    generated_text: str
    num_tokens_generated: int
    stop_reason: str
    wall_clock_seconds: float
    wall_clock_per_token: float
    peak_memory_mb: float
    segments: list[SegmentEntry]
    events: list[Any]  # CompressionEvent
    compression_event_count: int
    decode_event_count: int
    total_evicted_tokens: int
    predicted_answer: str | None
    correct: bool | None
    ground_truth: str


def run_controller_once(
    model: Any,
    tokenizer: Any,
    prompt: dict[str, Any],
    predictor: ExportedPredictor,
    policy: Any,
    build_press_fn: Any,
    initial_budget: int,
    compression_interval: int,
    max_new_tokens: int,
    device: str,
    k: int = DEFAULT_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> ControllerRun:
    """Drive one controller-or-baseline generation on Orion.

    `build_press_fn(target_size, compression_interval)` returns a
    fresh DecodingPress(KnormPress) instance.
    """
    if k != compression_interval:
        raise ValueError(
            "k must equal compression_interval (segment boundaries "
            "must align with press fire boundaries)"
        )
    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        LogitsProcessorList,
    )

    from herald.detectors import parse_gsm8k_answer  # noqa: PLC0415
    from herald.phase4_feasibility import (  # noqa: PLC0415
        classify_decode_events,
        instrument_press,
    )
    from herald.prompts import format_chat  # noqa: PLC0415

    press = build_press_fn(initial_budget, compression_interval)
    recorder = instrument_press(
        press, target_size=initial_budget, threshold=None
    )
    online = OnlineFeatureState(
        press="decoding_knorm",
        compression_ratio=0.0,
        max_new_tokens=max_new_tokens,
    )
    proc = HeraldOnlineProcessor(
        predictor=predictor,
        online_state=online,
        policy=policy,
        press=press,
        k=k,
        threshold=threshold,
    )

    messages = format_chat(prompt["question"])
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad(), press(model):
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([proc]),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    generated_ids = outputs.sequences[0, input_len:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    n_gen = len(generated_ids)
    eos_id = tokenizer.eos_token_id
    eos_ids = set(eos_id) if isinstance(eos_id, list) else {eos_id}
    if generated_ids and generated_ids[-1] in eos_ids:
        stop_reason = "eos"
    elif n_gen >= max_new_tokens:
        stop_reason = "max_tokens"
    else:
        stop_reason = "other"

    classify_decode_events(recorder.events)
    decode_events = [e for e in recorder.events if e.event_type == "decode"]
    total_evicted = sum(
        max(0, e.retained_cache_len_before - e.retained_cache_len_after)
        for e in decode_events
    )

    try:
        predicted = parse_gsm8k_answer(str(generated_text))
    except Exception:  # noqa: BLE001
        predicted = None
    gt = str(prompt["ground_truth"]).strip()
    correct = None if predicted is None else (str(predicted).strip() == gt)

    run_id = f"phase4-ctrl-{policy.name}-{prompt['id']}-init{initial_budget}"
    return ControllerRun(
        run_id=run_id,
        prompt_id=str(prompt["id"]),
        policy_name=str(policy.name),
        initial_budget=int(initial_budget),
        generated_text=generated_text,
        num_tokens_generated=n_gen,
        stop_reason=stop_reason,
        wall_clock_seconds=float(elapsed),
        wall_clock_per_token=(float(elapsed / n_gen) if n_gen else math.nan),
        peak_memory_mb=float(peak_mem_mb),
        segments=proc.segments,
        events=recorder.events,
        compression_event_count=len(recorder.events),
        decode_event_count=len(decode_events),
        total_evicted_tokens=int(total_evicted),
        predicted_answer=predicted,
        correct=correct,
        ground_truth=gt,
    )


# --- Cache-size attribution ------------------------------------------


def attach_cache_size_per_segment(
    run: ControllerRun, k: int = DEFAULT_K
) -> list[dict[str, Any]]:
    """Project segment entries to row dicts with cache_size_observed.

    For each segment [s*k, (s+1)*k), pick the max retained_cache_len
    _after across decode events whose attribution lands in that
    window. We approximate event placement by sequential index across
    all decode events: kvpress fires once per layer per fire, so
    `n_layers` events form one logical fire group. Without a known
    n_layers we group by per-layer occurrence count.

    This is a smoke-grade attribution; the publishable run should
    record per-event step indices from a model-side hook.
    """
    rows: list[dict[str, Any]] = []
    decode = [e for e in run.events if e.event_type == "decode"]
    # Group events by per-layer fire ordinal: events sharing a
    # layer_idx in order are successive fires for that layer.
    by_layer: dict[int, list[Any]] = {}
    for e in decode:
        by_layer.setdefault(int(e.layer_idx), []).append(e)
    # For each segment idx s, attribute the s-th fire of every layer.
    n_segments_logged = len(run.segments)
    for seg in run.segments:
        s = seg.segment_idx
        cache_sizes: list[int] = []
        evicted_in_seg = 0
        for layer_events in by_layer.values():
            if s < len(layer_events):
                ev = layer_events[s]
                cache_sizes.append(int(ev.retained_cache_len_after))
                evicted_in_seg += max(
                    0,
                    int(ev.retained_cache_len_before)
                    - int(ev.retained_cache_len_after),
                )
        cache_size = max(cache_sizes) if cache_sizes else -1
        start_pos = seg.segment_idx * k
        end_pos = seg.tokens_completed
        rows.append(
            {
                "segment_idx": seg.segment_idx,
                "tokens_completed": seg.tokens_completed,
                "start_token_pos": start_pos,
                "end_token_pos": end_pos,
                "segment_score_mean": seg.segment_score_mean,
                "segment_score_max": seg.segment_score_max,
                "threshold": seg.threshold,
                "current_budget": seg.current_budget,
                "next_budget": seg.next_budget,
                "decision": seg.decision,
                "cache_size_observed": cache_size,
                "evicted_tokens_in_segment": int(evicted_in_seg),
            }
        )
    if not rows and n_segments_logged == 0:
        # Nothing to report; generation produced < K tokens.
        pass
    return rows
