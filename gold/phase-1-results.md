# Phase 1 Results

# Open question (load-bearing — needs user decision before this doc is final)

The pre-registered Phase 1 success criterion failed under a strict
reading (2 of 4 tasks pass). But the failure mechanism is not "intrinsic
trajectory aggregates fail to predict damage." It is **instrumentation
saturation**: ifeval and longbench_single's `correct` field is `True`
on every run because `is_wrong` is a presence check, not a quality
verifier (`src/herald/tasks.py:235` and the LongBench/qasper analogue).
Where outcome harm is measurable (gsm8k, humaneval), intrinsic →
extrinsic prediction is strong: task-pooled AUROC 0.73 / 0.88,
per (task, press) AUROC up to 1.0 / 0.90.

The pre-registered rule says "if it fails, switch to the
negative-methodology fallback and write that up; do not run the
probe." The hard rule "pre-registered rules are not tuned post-hoc"
cuts toward strict failure. **But the rule was written assuming the
extrinsic label was always defined; it does not anticipate the
instrumentation-saturation case.** Two readings:

- **Strict.** Instrumentation saturation ⇒ criterion not satisfied
  ⇒ write the negative-methodology paper. Probe stays unrun.
- **Methodological.** The failure is in the evaluator wiring, not in
  the predictor. Re-run the criterion against a continuous severity
  target (the research plan's pre-anticipated substitution for the
  saturated-correctness case) before deciding the narrative.

This document does not pick. It surfaces the question, lays out the
strict-reading data, and lists the actions either reading needs.

---

Sweep run: 2026-05-04 → 2026-05-05, Orion (RTX 5090),
Qwen2.5-7B-Instruct, fp16, 4 tasks (gsm8k / humaneval / ifeval /
longbench_single=qasper) × {none, streaming_llm, snapkv, knorm,
expected_attention, tova, random} × {0.25, 0.375, 0.5, 0.75, 0.875,
0.9375, 0.96875}, 200 prompts/task (HumanEval capped at 164 by dataset
size), greedy decoding, max_new_tokens=512, every-8 matched-prefix
replay (decision recorded in `gold/phase-0-results.md`).

## Closeout

Two closeouts before block 4:

1. Cron `5a8017e7` (the rerun-watcher created in session
   `13770ceb-0a14-436a-a8a0-e8e7f0a7067e` on 2026-05-05 13:21 UTC) is
   session-scoped in-memory state. From this session `CronList` is
   empty and `CronDelete 5a8017e7` errors with "no scheduled job".
   Decision: leave it; it auto-expires 2026-05-12, the on-Orion sweep
   is `nohup`'d so the cron's hourly check has no side effect, and
   killing the session round-trip costs more than it saves.
2. The longbench rerun overwrote `phase1_sweep_summary.json` with the
   43 longbench cells. The original was preserved as
   `results/phase1/phase1_sweep_summary.longbench-rerun.json` and the
   unified summary rebuilt from the 172 surviving per-cell
   `cell_summary.json` files via
   `scripts/rebuild_phase1_summary.py`. Per-task ok+skipped totals
   match the announced sweep totals exactly:

   | task | cells | n_ok | n_skipped | n_failed |
   | --- | ---: | ---: | ---: | ---: |
   | gsm8k | 43 | 8 600 | 0 | 0 |
   | humaneval | 43 | 7 052 | 0 | 0 |
   | ifeval | 43 | 8 600 | 0 | 0 |
   | longbench_single | 43 | 4 659 | 3 941 | 0 |
   | **total** | **172** | **28 911** | **3 941** | **0** |

   Effective ok runs (ok + already-skipped-from-Block-3): 32 852.

## Completion and finalize coverage

`finalize_dataset` (extended in `src/herald/metrics/io.py` to walk the
nested `<task>/<press>/ratio=*/raw/*/` layout) produced
`results/phase1/final/{runs.parquet, tokens/, replay/}` covering all
**32 852** ok runs. Sanity check (the user's pre-flight #3) passes:

```
runs.parquet  unique run_id: 32 852  (replay_status='ok' for all)
tokens/       unique run_id: 32 852  (set diff vs runs.parquet: 0)
replay/       unique run_id: 32 852  (set diff vs runs.parquet: 0)
final/ size on disk: 11 GiB
```

Denominators are aligned across `runs.parquet`, `tokens.parquet`,
`replay.parquet` — no silent drop or duplicate during the cell-by-cell
finalize.

`metrics/cli.py` was extended to wire `segment.py` in as the last step
of `metrics build` (with a `--skip-segments` flag for cheap-family
re-runs). The trajectory + outcome + tags families ran to completion;
the segment build aborted partway through cell 13/172
(`gsm8k/knorm/ratio=0.9375`) — likely the per-window EWMA Python loop
hitting the heaviest compression ratio on the longest looped outputs.
Segments are flagged as **carry-forward**; nothing in the headline
alignment or success-criterion application below depends on them.

Sequence-level metrics (`rouge_l`, `embedding_cosine`,
`edit_distance_ratio`) require the `[metrics]` extra (`rouge-score`,
`editdistance`, `sentence-transformers`) which is not installed in the
Orion sandbox and cannot be installed without the SSH reverse-tunnel
proxy. A pure-Python continuous-severity proxy
(`scripts/phase1_severity.py`: ROUGE-L drop, char-level edit ratio,
length-diff ratio) is computing in the background.

## Alignment matrix (intrinsic ↔ intrinsic, pooled)

Pairwise Spearman + bootstrap 95% CI on 32 088 compressed runs (drops
the 764 baseline runs):

| pair | ρ | 95% CI |
|--|--|--|
| `sum_kl` ↔ `sum_js` | 0.97 | [0.97, 0.97] |
| `sum_kl` ↔ `nll_ratio` | -0.86 | [-0.87, -0.86] |
| `sum_js` ↔ `nll_ratio` | -0.83 | [-0.84, -0.82] |
| `sum_kl` ↔ `first_divergence_point` | -0.50 | [-0.51, -0.49] |
| `sum_js` ↔ `first_divergence_point` | -0.48 | [-0.49, -0.47] |
| `nll_ratio` ↔ `first_divergence_point` | 0.43 | [0.42, 0.44] |

Phase-0 parity check: `sum_kl ↔ sum_js` was 0.91 [0.86, 0.94] on
N=134 in Phase 0; at N=32 088 it tightens to 0.97 with negligible CI,
consistent with the same underlying agreement between the two
divergence aggregates. The Phase-0 row entries on `rouge_l` /
`edit_distance_ratio` / `embedding_cosine` are not reproduced here
because the sequence-metrics pipeline didn't run (see above); they
will be added once `phase1_severity.py` finishes or `[metrics]` is
installed.

The full per-stratum matrix lives at
`results/phase1/metrics/alignment_phase1.parquet` (1 038 rows: 1
global pool + 4 task pools + 168 (task, press, ratio) cells, six
pairs each).

## Per-task intrinsic-to-outcome-harm AUROC

Per-task pooled across all (press, ratio) compressed cells. Outcome
harm = `baseline_correct AND NOT compressed_correct` per the research
plan.

| task | n | n_pos | metric | AUROC | 95% CI | Spearman |
|--|--:|--:|--|--:|--|--:|
| gsm8k | 8 400 | 5 216 | `sum_kl` / `sum_js` | 0.73 | [0.72, 0.74] | +0.39 |
| humaneval | 6 888 | 1 467 | `sum_js` | 0.88 | [0.87, 0.89] | +0.54 |
| ifeval | 8 400 | **0** | undefined | — | — | — |
| longbench_single | 8 400 | **0** | undefined | — | — | — |

**Per (task, press) headline AUROCs** (best-cell across ratios):

| task | press | best AUROC |
|--|--|--:|
| gsm8k | streaming_llm | 0.90 |
| gsm8k | expected_attention | 0.85 |
| gsm8k | snapkv | 0.84 |
| gsm8k | tova | 0.71 |
| gsm8k | random | 0.68 |
| gsm8k | knorm | 0.66 |
| humaneval | knorm | 1.00 |
| humaneval | streaming_llm | 0.95 |
| humaneval | tova | 0.89 |
| humaneval | expected_attention | 0.83 |
| humaneval | snapkv | 0.80 |
| humaneval | random | 0.75 |

Where outcome harm is well-defined, intrinsic trajectory aggregates
predict it strongly (per-(task,press) AUROC mostly ≥ 0.7, several
≥ 0.85). Direction is consistent (positive ρ, higher trajectory KL/JS
means more harm).

## The saturation finding (ifeval / longbench_single)

The `correct` field for ifeval and longbench_single is `True` on
**all 8 600 / 8 600** runs (compressed and baseline). Reading
`src/herald/tasks.py:235` confirms this is by design: the IFEval
`is_wrong` rule is a presence check (`return not generated_text.strip()`),
not a constraint-by-constraint verifier. LongBench/qasper has the same
shape (no F1/EM evaluator wired in for Phase 1).

Consequence: the binary outcome-harm label is degenerate on those two
tasks, so the success criterion as stated cannot be applied
end-to-end. This is consistent with the research plan, which
anticipates the case explicitly:

> "If task correctness is saturated, Phase 2 should use continuous
>  sequence/trajectory severity as the primary target rather than
>  falling back to diagnostic tags." — `gold/research-plan.md`,
>  Phase 2 Targets.

The `phase1_severity.py` proxy (ROUGE-L drop, char-edit ratio,
length-diff ratio) is the substitute target the plan calls for; it
runs in the background and will be folded into a follow-up alignment
pass.

## Phase 1 success criterion (strict, pre-registered)

Operationalisation in `scripts/phase1_success_criterion.py` (locked
before observing outcomes):

- "Significantly predict" = AUROC ≥ 0.65 with paired bootstrap CI
  lower bound > 0.5 on the per-task pooled sample.
- "Directionally stable" = Spearman ρ has the same sign across all
  qualifying tasks (positive expected).
- "Pass" requires ≥ 3 of 4 tasks meeting both bars.

Verdict: **strict pre-registered criterion FAILS.**

| task | passes | reason |
|--|--|--|
| gsm8k | yes | AUROC 0.73, CI [0.72, 0.74], ρ +0.39 |
| humaneval | yes | AUROC 0.88, CI [0.87, 0.89], ρ +0.54 |
| ifeval | no | task-pooled AUROC undefined (`n_pos = 0`) |
| longbench_single | no | task-pooled AUROC undefined (`n_pos = 0`) |

2 of 4 < the 3-of-4 bar.

The pre-registered rule says: "If it fails, switch to the
negative-methodology fallback and write that up; do not run the
probe." The probe is not run.

## Actions per reading

### If the user reads the failure strictly

1. Move the contribution narrative to negative-methodology
   (`gold/contribution-validation.md`, "Five plausible measures of
   compression damage; how they fail to align"). Phase 1's evidence
   for that paper is rich: alignment among intrinsic aggregates
   (sum_kl/sum_js/nll_ratio) is essentially saturated (ρ > 0.83 in
   absolute value), but their alignment with paired outcome harm
   degrades from AUROC 0.88 (humaneval) to 0.73 (gsm8k) to "the
   extrinsic label is not even defined" (ifeval, longbench), which
   *is* the negative-methodology thesis: compression damage is harder
   to measure than the field assumes.
2. Do not run the probe.
3. The intrinsic AUROC numbers above stay in the appendix as
   evidence for the predictability claim, conditional on a
   well-specified extrinsic label.

### If the user reads it as instrumentation-only failure

1. **Severity rerun (in flight as of this writeup).**
   `scripts/phase1_severity.py` is computing pure-Python ROUGE-L
   drop, char-edit ratio, and length-diff ratio against the matched
   uncompressed baseline. Output:
   `results/phase1/metrics/severity_phase1.parquet`. When it
   finishes, the follow-up is one command:

   ```sh
   # On Orion or after rsync:
   .venv/bin/python scripts/build_phase1_alignment.py --root results/phase1
   .venv/bin/python scripts/phase1_success_criterion.py --root results/phase1
   ```

   `build_phase1_alignment.py` already auto-includes
   `severity_phase1.parquet` columns in the pairwise matrix when the
   file exists (its `_load` function reads `sequence_metrics.parquet`
   in that role; rename the severity file or extend the loader). The
   success-criterion script needs a small extension to substitute a
   severity column for `outcome_harm` on tasks where `n_pos = 0`.

   Pre-registration constraint: the bar (AUROC ≥ 0.65, CI > 0.5,
   ≥ 3 of 4 tasks) does NOT change; only the extrinsic label
   substitutes per the research plan's anticipated rule
   (`gold/research-plan.md`, Phase 2 Targets).

2. **Real evaluators are the durable fix.** Wire the official IFEval
   `instructions` library against `instruction_id_list`/`kwargs` per
   `src/herald/tasks.py:194` and qasper F1/EM into
   `src/herald/tasks.py`. Neither needs a re-sweep — they re-score
   `final/runs.parquet` in place. The severity proxy above is a
   stop-gap; real evaluators are what supports a Phase 1 paper that
   doesn't have the saturation caveat.

3. **Segment build root cause.** The segment build silently died
   between cells 12 and 13 (`gsm8k/knorm/ratio=0.9375`). No
   traceback in the log, nothing in dmesg the operator user can
   read. Possibilities (untriaged): NaN in one of the Tier-0 columns
   tripping `_ewma_last`, polars memory bloat from accumulating per-
   cell DataFrames, or a signal kill we can't see. Before the next
   run, wrap `aggregate_all_segments` in a per-cell try/except that
   prints the offending file and consider vectorising the per-window
   EWMA (the current Python `to_list()` + loop is the obvious
   suspect for repeated GC pressure).

## Ratio-grid carry-forwards

Per-cell gross_harm grid lives at
`results/phase1/metrics/per_cell_gross_harm.csv`. Headline shape
(gsm8k, baseline correctness 77.5 %):

| press | 0.25 | 0.375 | 0.5 | 0.75 | 0.875 | 0.9375 | 0.96875 |
|--|--:|--:|--:|--:|--:|--:|--:|
| streaming_llm | 0.04 | 0.19 | 0.61 | 0.76 | 0.76 | 0.78 | 0.78 |
| expected_attention | 0.12 | 0.22 | 0.45 | 0.72 | 0.77 | 0.78 | 0.78 |
| snapkv | 0.17 | 0.33 | 0.56 | 0.78 | 0.78 | 0.78 | 0.78 |
| tova | 0.55 | 0.72 | 0.75 | 0.76 | 0.77 | 0.76 | 0.76 |
| random | 0.21 | 0.36 | 0.42 | 0.72 | 0.76 | 0.78 | 0.78 |
| knorm | 0.64 | 0.73 | 0.75 | 0.75 | 0.77 | 0.78 | 0.77 |

Carries forward Phase 0's directional finding ("the cliff is at or
below 0.5 on this slice") but with much sharper resolution. The new
{0.25, 0.375} regime separates the presses cleanly into two groups:

- **Gentle-damage at low ratios**: `streaming_llm` (0.04 / 0.19),
  `expected_attention` (0.12 / 0.22), `snapkv` (0.17 / 0.33), `random`
  (0.21 / 0.36). These four presses validate the Phase-0
  recommendation that extending the grid below 0.5 was needed; their
  cliffs sit between 0.375 and 0.75.
- **Hard cliff at 0.25**: `knorm` (0.64) and `tova` (0.55) are already
  at most-of-the-way-to-saturation gross_harm at the gentlest measured
  ratio. They effectively skip the gentle regime on gsm8k. This is
  worth a Phase-2 sanity probe: either the press's compression
  semantics genuinely have no gentle regime on this model/task pair,
  or there is a wiring issue. Either way, neither is a useful gentle-
  regime baseline as currently configured.

## Open items for Phase 2

- The saturation finding is the load-bearing one. Either real
  evaluators land or the Phase 1 → Phase 2 baton hands over the
  severity-target substitution.
- Segment metrics need a vectorised re-implementation before the Phase
  2 feature pipeline can rely on them.
- `nll_ratio`'s sign on outcome harm is *negative* on the per-task
  pooled rows (e.g. gsm8k −0.37) where `sum_kl`/`sum_js` are positive.
  This is consistent (more compression → more negative `nll_ratio`)
  but the sign convention should be flipped for downstream Phase-2
  feature engineering or it will silently confuse the predictor.
- Phase 0's open carry-forwards
  (streaming_llm sum_kl non-monotone hump at 0.875, snapkv 0.875 ↔
  0.9375 indistinguishability, every-8 per-position fidelity) have not
  been re-checked against Phase 1's full N; that's a stand-alone
  follow-up using `results/phase1/final/`.

# Probe results — pending

The Phase 1 intervention probe (`gold/phase-1-intervention-probe.md`)
is **not run** in this session. Reasons:

1. The pre-registered rule says "do not run the probe" if the
   headline criterion fails, and under the strict reading it does (2
   of 4 tasks). The severity rerun above may pass, but the rule is
   "not tunable post-hoc" and the strict reading is what binds.
2. `SwitchAtOffsetPolicy` is currently a stub
   (`src/herald/policy.py:105-106`); the press-specific intervention
   vocabulary (mask-relax, stop-eviction, post-prefill ratio
   reduction, reprefill oracle) has not been implemented; the probe
   sweep launcher does not exist. None of those are 5-minute changes.

When the probe lands, this section is appended in place. The probe's
stratum-availability rule (`gold/phase-1-intervention-probe.md`,
"Stratum Availability Contingency") will need real numbers from the
Phase 1 outcome distribution; a quick eyeballing of the gsm8k cliff
shape suggests boundary-unstable strata are well-populated for
`streaming_llm`, `expected_attention`, and `random` but thin for
`knorm` (gentle regime collapses early) and possibly degenerate for
`ifeval`/`longbench_single` until the saturation gap is closed. That
will dictate the per-(task, press) anticipatory-only flagging.

# Post-hoc rescore status — added 2026-05-05

The strict pre-registered result above used **placeholder graders**
for ifeval and longbench_single (`is_wrong = not text.strip()`); the
predictability table on those two tasks is therefore informationless
(`n_pos = 0`), not a real failure of the intrinsic-vs-outcome bar.
The fix is post-hoc deterministic rescoring against the saved
generations and dataset metadata; **no re-sweep, no LLM judge, no GPU
launch**. It produces the canonical Phase 1 run-level table at
`results/phase1/metrics/run_damage.parquet` (one row per compressed
run, joined to its uncompressed baseline). Schema and operational
details: `gold/run-damage-table.md`.

Pipeline (run on local Mac after rsync from Orion, or on Orion):

```sh
uv run python scripts/build_run_damage.py --root results/phase1
uv run python scripts/build_phase1_alignment_final.py --root results/phase1
uv run python scripts/phase1_success_criterion_final.py --root results/phase1
```

Outputs:

- `metrics/run_damage.parquet` — canonical per-run table.
- `metrics/alignment_phase1.final.parquet` — pairwise Spearman over
  the corrected labels.
- `metrics/intrinsic_to_outcome_phase1.final.parquet` — Spearman +
  AUROC of intrinsic metrics against `gross_harm_final` and
  `quality_delta` per stratum.
- `metrics/phase1_success_criterion.final.json` — re-application of
  the unchanged pre-registered bar against the corrected labels.

The original strict artifacts (`alignment_phase1.parquet`,
`intrinsic_to_outcome_phase1.parquet`, `phase1_success_criterion.json`)
are preserved unchanged so the placeholder-grader reading remains
reproducible. The corrected verdict lives next to it under the
`*.final.*` filenames.

What the rescore changes:

- LongBench/qasper: SQuAD-style normalized exact match + token-level
  F1 over all gold answers, max over multiple golds. Default
  threshold for binary correctness is `f1 >= 0.5` (the standard Qasper-
  paper threshold; configurable via `--qasper-threshold`).
- IFEval: deterministic partial scorer over the supported instruction
  families (see `task_scoring.SUPPORTED_INSTRUCTIONS`). Constraints
  outside that set are explicitly reported as **unsupported**, never
  silently True. A row whose instruction list contains any unsupported
  type has `ifeval_correct_final = null` and contributes zero to
  `gross_harm_final` (it cannot prove harm or absence of harm). The
  durable upgrade path is installing
  `instruction_following_eval`; the builder will use it when present.
- GSM8K and HumanEval correctness fields are unchanged.

What the rescore preserves:

- Same pre-registered bar (AUROC ≥ 0.65, CI > 0.5, ≥ 3 of 4 tasks).
- Same intrinsic metrics, same trajectory aggregates, same prompts.
- Both readings of the open question above stay reportable: strict
  with placeholder graders, methodological with corrected labels.

Until `final/runs.parquet` is rsynced from Orion (it is not in this
checkout), the build pipeline above will exit early with
`missing results/phase1/final/runs.parquet`. The code path is
schema-validated against `results/phase0/final/runs.parquet` and unit-
tested against synthetic fixtures (`tests/metrics/test_task_scoring.py`,
`tests/metrics/test_run_damage.py`).
