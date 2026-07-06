# Controller design

What the HERALD controller is, which parts exist and are validated,
which parts are future engineering, and the dynamic-policy
extensions under consideration. Companion to
`online_forecasting.md` (the experimental arc that settled the
design) and `controller_metrics.md` (the locked metric).

## Runtime semantics (the policy)

For one request:

1. Generation starts with the full uncompressed KV cache. Each
   decode step yields ~20 cheap logit statistics (entropy, margin,
   max-prob, rolling aggregates, ...) at no extra forward-pass cost.
2. A per-compressor risk model (the scorer) maps the accumulated
   statistics at position s to a risk score: "if we compress now,
   does the rest of this answer survive?"
3. Earliest position where score <= tau (a threshold frozen at
   calibration time): compress the cache and continue. Savings for
   the request = max(0, 1 - s / ref_len).
4. Optional post-switch safety net (forecasting variant): watch the
   model's own per-token statistics for the first k=16 tokens after
   the switch; a second model can flag a bad switch for recovery.

## Validated components (as of 2026-07-06, ifeval)

- Scorers, one per compressor (cross-compressor transfer is
  falsified for every feature family tried; see
  `online_forecasting.md`): XGBoost 3-seed ensembles for
  expected_attention and streaming_llm, TabFM 1.0.0 (GPU) for
  knorm. Trained on ~15k recorded switch experiments per
  compressor.
- Calibration: 5-fold cross-fitted out-of-fold scoring of all
  training prompts, tau = most aggressive threshold whose OOF
  group-mean cost <= epsilon (0.01). Frozen before test.
- Evaluation: policy replay against recorded reality (the switch
  dataset holds true damage at every candidate switch point), test
  prompts disjoint from training. Deployable results: ea 0.484,
  sllm 0.227 (XGB), knorm 0.128 (TabFM), all within budget;
  worst-case 0.128.

## Not yet built (engineering, not science)

- The runtime wrapper: a generate() loop that computes features,
  calls the scorer, and triggers the kvpress compression hook
  mid-generation. All pieces exist separately in the sweep harness;
  they are not wired into one live system.
- Rollback mechanics for the safety net (hold the old cache k
  tokens, revert on flag) and its token-cost accounting in the
  savings metric.
- The budget-aware scorer/variant selection rule as code (select by
  OOF bootstrap bound, not raw OOF savings).

## Dynamic policies (future direction, proposed 2026-07-06)

The current policy compresses exactly once, irreversibly. The
obvious generalization is compress/decompress on demand. Physics
constraint: lossy cache compression cannot be undone in place; the
evicted information is gone. "Decompress" therefore means one of:

1. Grace window (cheapest): keep the uncompressed cache alive for
   the first k post-switch tokens; commit (free the memory) only if
   the online forecaster stays quiet, revert otherwise. Memory
   savings start at commit. This is the planned safety net; it is
   evaluable with a modest extension of existing data.
2. Alarm-triggered re-prefill (real recovery): run compressed; if
   the forecaster raises an alarm mid-generation, recompute the
   full cache with one prefill over prompt + generated tokens and
   continue uncompressed (optionally re-compress later at a safer
   point). Converts the online forecasting signal from a veto into
   a recovery mechanism. Costs one prefill per alarm; memory
   returns to full until the next compression. NOT evaluable from
   current data: recorded hybrid runs never re-prefill, so this
   needs a new sweep dimension (compress at s, alarm at s+j,
   re-prefill, continue) on Orion.
3. Progressive deepening (one-way ratchet): compressing MORE is
   always possible (evict further from an already-compressed
   cache). A policy that starts at a light ratio and deepens
   whenever the forecaster reads safe could dominate the
   single-shot policy without any decompression at all. Also needs
   new sweep data (multi-step compression grid).

4. TCP-style congestion control (proposed 2026-07-06, unifies 2+3):
   treat compression ratio like a congestion window. Additive
   increase (deepen compression) while the online forecaster reads
   safe; multiplicative decrease via re-prefill on alarm. The
   forecaster plays ECN: an early signal that fires before damage
   is visible in text (loss-based control would be too late by
   construction). Slow-start analogue: deepen aggressively until
   the first alarm, then probe linearly. Detection latency k is the
   control loop's RTT. Design trade: false alarms cost prefills
   (compute), misses cost quality (the epsilon budget is the SLA).
   Requires trajectory-executing sweeps (deepen/alarm/re-prefill
   sequences); not replayable from single-switch records.

Priority (agreed 2026-07-06): sweeps are the scarce resource, so
TCP is prototyped BEFORE scaling to avoid re-sweeping tasks for
trajectory data. Order: (1) harden the single-switch 0.128 (split
robustness, locked-evaluator confirmation, selection rule as code);
(2) AIMD feasibility study from existing ifeval streams, zero GPU
(alarm operating points, detection-latency sweep, idealized
free-revert upper bound - kills or funds the idea cheaply);
(3) trajectory harness + TCP prototype on ifeval; (4) one combined
gsm8k/humaneval campaign capturing single-switch streams AND
trajectories in the same pass.
