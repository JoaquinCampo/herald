# Phase 3 (HERALD v1) Validation — 2026-05-15

Hostile-reviewer audit of `gold/phase-3-herald-v1-results.md` and supporting
artefacts. Scope: reproducibility, honesty, statistical method, paper-readiness.
Local-only checks (no GPU runs). Author: Claude (validation pass).

## 1. Verdict

**NOT READY (conditional).** The core technical result is sound and the
headline scorecard reproduces from artefacts on disk. However, three honesty
concerns and one terminology drift must be fixed in the gold doc and paper
before ICLR submission. None require new experiments. All four are
documentation / framing fixes, not modelling fixes.

The thesis ("per-token logit features predict KV-compression damage H tokens
ahead, online and O(1)") survives. Bars 4, 8, 9, plus per-token bars 1, 2,
and cross-press retention 5 are clean. Bars 3, 6, 7 are partly substrate-
limited and partly mis-framed; the fixes below preserve the contribution
while removing reviewer attack surface.

## 2. Confirmed claims

All numbers in `gold/phase-3-herald-v1-results.md §"Headline contract"`
trace cleanly to artefacts on disk. Spot checks below.

| # | claim | source artefact | verified |
|---|---|---|---|
| 1 | per-token ρ vs `future_sum_js_25` = 0.7540 [0.7480, 0.7595] | `results/phase3/eval/summary.json` | ✓ exact |
| 2 | per-token cross-horizon ρ vs `future_sum_js_50` = 0.7658 [0.7595, 0.7713] | same | ✓ exact |
| 3 | per-run wrapper ρ vs `sum_js` = 0.8643 | `results/phase3/run_level_wrapper_trajectory.json` | ✓ exact |
| 4 | per-run wrapper ρ vs `rouge_l_drop` = 0.8285 | same | ✓ exact |
| 5 | cross-press retention overall = 0.864 / min 0.863 (sum_js) | trajectory.json + per-press table | ✓ |
| 6 | cross-ratio retention worst at ratio=0.375 ρ=0.666 | `run_level_wrapper_trajectory_loo_ratio.json` | ✓ exact |
| 7 | cross-task retention worst at ifeval ρ=0.726 | `run_level_wrapper_trajectory_loo_task.json` | ✓ exact |
| 8 | per-run ECE wrapper vs sum_js = 0.036 | trajectory.json | ✓ exact |
| 9 | streaming parity max\_abs\_diff = 0.0 (tol 5e-4) | re-ran `test_herald_v1_streaming_parity.py` 2026-05-15 | ✓ |
| 10 | substrate ceiling (rouge\_l\_drop, max-agg) = 0.78 | re-ran `compute_substrate_ceiling_rich.py` | ✓ identical |
| 11 | meta-only baseline ρ (sum\_js) = 0.811 | `run_level_wrapper_meta_only.json` | ✓ exact |
| 12 | per-token pred\_max baselines (sum\_js 0.7316, rouge 0.7578, etc.) | `per_token_per_ratio_rho.json` companion | ✓ |
| 13 | slopefeat pooled ρ = 0.781, worst (0.25) ρ = 0.632 | `per_token_per_ratio_slopefeat.json` | ✓ exact |
| 14 | oracle reachability retention (canonical / per-ratio / extfeat) = 0.795 / 0.800 / 0.800 | three `oracle_reachability_bar6*.json` | ✓ |
| 15 | per-ratio y\_max ceiling spread 0.665 (ratio 0.97) – 0.858 (ratio 0.375) | `substrate_ceiling_per_ratio_bar6.json` | ✓ |

Cluster bootstrap implementation in `src/herald/regression_metrics.py`
(`clustered_spearman_ci`) is correct: clusters sampled with replacement, indices
concatenated, Spearman computed as Pearson on parent-ranks (mathematically
equivalent under no-tie assumption; ties handled by `scipy.stats.rankdata`'s
average method in the parent ranking). Percentile CI at α = 0.05. `n_boot = 500`
with `n_boot_ok` reported per row (all 500 in the files inspected).

GroupKFold integrity: `train_per_token_slope_features.py` line 230 uses
`GroupKFold(n_splits=5)` keyed by `prompt_id`. Same key in
`scripts/train_predictor.py` and the wrapper-trajectory script. No prompt
appears in both train and validation within any fold.

ECE computation (`ece_quantile` in `regression_metrics.py`) uses 10
equal-frequency quantile bins on `pred_rank`, gap = `|mean(pred_rank) –
mean(y_rank)|` per bin, length-weighted average. Standard recipe.

## 3. Blockers

These must be fixed before submission. Each is a documentation / framing fix;
no new experiments needed.

### B1. Terminology drift: "strict pred-aggregates-only" is overloaded

`gold/phase-3-herald-v1-results.md §"No-leakage protocol"` says:

> Strict pred-aggregates-only V2 (sum_js): 0.864 → margin +0.053 → clears the
> 0.05 margin requirement.

But `gold/research-plan.md` defines **strict pred-aggregates-only** as exactly
**eight features**: `pred_max, p95, p75, p50, mean, std, last5, top3`. That
eight-feature wrapper is `run_level_wrapper_strict.json`, which gives
**sum_js = 0.8277**, margin **+0.017 over meta-only 0.811** — well below
the pre-registered 0.05 bar.

The 0.864 number in the gold doc is the **17-feature trajectory wrapper**
(`run_level_wrapper_trajectory.json`), which adds 9 trajectory-shape features
(`pred_above_p90_rate`, `pred_longest_run_above_p75_rate`,
`pred_count_local_maxima_rate`, `pred_max_pos`, `pred_late_minus_early`,
`pred_max_minus_p50`, `pred_p95_minus_p50`, `pred_auc`, `pred_max_derivative`)
to the strict eight. All 17 features are pred-derived with no metadata, so the
wrapper is honest under the no-metadata-leakage criterion. But it is **not**
the wrapper named in the pre-registration.

Two equally good fixes:

(a) **Re-baseline.** Treat the pre-registered "strict 8" as failing the 0.05
margin (it does, by +0.017 vs meta-only). Promote the 17-feature trajectory
wrapper to a renamed "**trajectory-shape pred-only wrapper**", pre-register it
*post hoc and label it as such*, and disclose in the paper that the strict
eight-aggregate variant does not clear 0.05. The 0.053 margin claim then
holds for the *trajectory* wrapper, not the *strict-eight* wrapper.

(b) **Update research-plan.md** to redefine "strict pred-aggregates-only" to
mean the 17-feature trajectory set, retroactively. This is weaker because
it muddies the pre-registration discipline that motivated the protocol.

Pick (a). It costs one paragraph and is the more honest move.

### B2. Bar 3 ceiling defense conflates univariate and multivariate substrate

`gold/phase-3-herald-v1-results.md §"Bar 3"` defends the 0.829 ρ on
`rouge_l_drop` as exceeding "substrate ceiling 0.78" by 0.05 because the
wrapper "learns multi-feature trajectory shape, not a single aggregation".

The 0.78 number is the **best of seven univariate true-label aggregations**
(`y_max, y_p95, y_mean, y_sum, y_auc, y_dwell_abs_q90, y_peak_count`) from
`compute_substrate_ceiling_rich.py`. It is the substrate ceiling **for a
univariate label aggregation**, not the substrate ceiling in general.

If the wrapper learns multivariate trajectory shape and exceeds 0.78, that
proves multivariate shape carries signal beyond any single y-aggregation.
It does **not** establish 0.78 as the substrate ceiling, only as the
single-aggregator ceiling.

The Bar 3 framing currently reads as "we beat the substrate ceiling, so the
bar is unreachable by definition". A reviewer can credibly object that
**the multivariate substrate ceiling has not been computed**, and that the
0.85 bar may yet be reachable with richer label aggregation (e.g.,
multi-aggregator regression on the true label) before declaring it
"unreachable in principle".

Fix: re-word Bar 3 explanation to:

> The best single-aggregation substrate ceiling for `rouge_l_drop` is 0.78
> (`y_max`). The wrapper exceeds this because it combines multiple
> pred-trajectory features; a multivariate substrate ceiling (multi-y-agg
> regression on the true label) has not been computed and could be higher.
> We do not retrain the per-token regressor directly on `rouge_l_drop`
> because doing so would abandon the substrate-honest framing.

This preserves the contribution while not over-claiming substrate
unreachability.

### B3. Bar 6 per-slice ceiling argument is partial (univariate-only)

`gold/phase-3-herald-v1-results.md §"Bar 6"` argues that retention is
structurally unfair because **per-slice substrate ceilings span 0.665–0.858**,
a 0.20 spread, so even a model that hits every slice's ceiling fails the bar.

This is true **for `y_max`** (univariate). It is **false for `y_sum`/`y_auc`**.
`results/phase3/substrate_ceiling_per_ratio_bar6.json` shows per-ratio
`y_sum`/`y_auc` ceilings in 0.970–0.985 across all seven ratios. A multivariate
true-label aggregator would have minimum slice ceiling ≈ 0.97, max ≈ 0.985,
and retention = min/max ≈ 0.985, **above** the 0.95 bar.

So the structural argument relies on choosing the univariate `y_max`
aggregator as the substrate ceiling. With `y_sum`/`y_auc`, the bar is
reachable in principle.

Fix: tighten the Bar 6 explanation to acknowledge that the structural
argument applies to the **single-aggregator regime** (the per-token model
predicts a point per token; aggregating with `pred_max` or any other single
function gives a single per-run score). Retention against a held-out ratio
under that regime has the spread the doc cites. A multivariate-aggregator
oracle remains a separate ceiling, which Phase 3 has not attempted; the doc
should either compute it or explicitly mark it out of scope.

### B4. Bar 6 "pre-committed kill threshold 0.65" lacks verifiable
pre-commitment

`scripts/train_per_token_slope_features.py` documents a 0.65 kill threshold
"pre-committed 2026-05-15" in its module docstring. The file is **untracked
in git** (`?? scripts/train_per_token_slope_features.py` in status). Its mtime
is 17:52, after `per_token_per_ratio_slopefeat.json` mtime 17:49. So the
pre-commitment claim cannot be verified from the working tree.

This is a small but real reviewer attack surface. The slopefeat worst-slice
ρ = 0.632 just misses 0.65. Without verifiable pre-commitment the kill
threshold reads post-hoc.

Fix options:
- (a) Commit the script before submission and add a separate commit (dated
  before any slopefeat result file) that introduces the threshold; document
  the chronology in the gold doc.
- (b) Reframe in the paper: drop "pre-committed kill threshold" language;
  state instead that worst-slice ρ = 0.632 is the largest of three attacks
  (slopefeat, per-ratio retrain, extended-feature joint) and still below
  what is needed for 0.95 retention. The framing then doesn't rest on
  pre-registration that cannot be shown.

Either is acceptable; (b) is faster and equally defensible.

## 4. Suggestions (non-blocking but recommended)

### S1. Bar 7 ifeval framing is hypothesis-only

The Bar 7 explanation invokes "constrained-format failure mode (skipping
format constraints)" for ifeval. Phase 1 / earlier analysis records show the
ifeval grader is a presence-check, not a format-check. The hypothesis is
plausible but not directly tested. Either run a brief diagnostic
(per-prompt-class ρ on ifeval split by constraint type) or weaken the
language from "does not show up in the JS divergence trajectory the way
other tasks' degradations do" to "may not show up cleanly in the per-token
JS divergence trajectory". The bar is missed by 0.003 anyway; the
defensive posture is already calibrated.

### S2. V1 → V2 leakage attribution lacks a preserved V1 results file

`gold/phase-3-herald-v1-results.md §"V1 → V2"` reports a 0.025 drop on sum_js
from the V1 trajectory wrapper (counts unnormalised) to V2 (counts
rate-normalised). No `run_level_wrapper_trajectory_V1.json` is preserved on
disk. The agnostic wrapper file (`run_level_wrapper_agnostic.json`) gives
sum_js = 0.892, which is close to the doc's V1 = 0.889 but is not the same
wrapper (it uses 8 strict + n_tokens + ratio + task, not 17 features with
unnormalised counts).

Recommend: re-run the V2 trajectory script with `count_normalize=False` to
preserve a `run_level_wrapper_trajectory_V1_unnormalized.json` baseline for
the paper appendix. This makes the 0.025 leakage claim falsifiable.

### S3. Cross-model transfer is undeclared

All Phase 3 numbers come from one model (Qwen/Qwen2.5-7B-Instruct per
`gold/research-plan.md`). The paper currently does not disclose a
cross-model transfer experiment. Either run one (Llama-3-8B-Instruct,
Mistral-7B-Instruct) or state up front in the paper Limitations that
HERALD v1 is validated on a single model family and that cross-model
transfer is left to future work. A reviewer **will** ask.

### S4. Phase 4 controller results (gold/phase-4-pareto-pilot-results.md,
phase-4-calibration-results.md) sit downstream of the Phase 3 predictor.
None of those numbers are in scope for this validation, but the calibration
bar 8 (ECE 0.036) is the entry point for the Phase 4 controller. Confirm
in the paper that the calibration claim is against the **wrapper score
distribution**, not against the **uncompressed reference distribution**, so
readers understand what 0.036 ECE buys downstream.

### S5. The §6.5 binary loop predictor decision in
`gold/phase-3-section-65-decision.md` is well-argued and self-contained. The
paper should respect it: §6.5 figures stay in an appendix labelled "Binary
Loop Diagnostic", **not** the headline. Confirmed; flagging here so the
audit trail is complete.

## 5. Reproducibility coverage

All 15 headline-track claims trace to artefacts on disk; streaming parity
re-run cleanly; substrate ceiling rich recompute is byte-identical to the
committed `gold/phase-3-substrate-ceiling-rich.md` table. No reproducibility
gap on Phase 3 modelling outputs.

Two artefacts are referenced but not on disk:
- `run_level_wrapper_trajectory_V1.json` (the unnormalised-counts V1 wrapper)
  — see S2.
- `train_per_token_slope_features.py` history that establishes 0.65 as
  pre-committed — see B4.

Outside scope (Phase 4 paths in `models/phase4_lr_all_cheap.json`,
`scripts/phase4_calibration/`) were not audited.

## 6. Honesty audit findings (summary)

| Finding | Severity | Section |
|---|---|---|
| "strict pred-aggregates-only V2" overloads the research-plan term | blocker | B1 |
| Bar 3 conflates univariate ceiling with general ceiling | blocker | B2 |
| Bar 6 per-slice spread argument relies on `y_max` only | blocker | B3 |
| Bar 6 kill threshold 0.65 lacks verifiable pre-commitment | blocker | B4 |
| Bar 7 ifeval format-skipping mechanism is unverified hypothesis | suggestion | S1 |
| V1 leakage delta of 0.025 not reproducible from disk | suggestion | S2 |
| Cross-model transfer undeclared | suggestion | S3 |

The contribution itself ("solid measurement + predictor, 6 of 9 bars
cleared with honest scorecard") survives all of these. The fixes are
about *framing*, not about model strength.

## 7. Paper-readiness gap list

**In place:**
- `paper/main.tex` §1 Introduction and §2 Catastrophic Failures (prevalence
  figure + table).
- `paper/figures/figure1.pdf` (prevalence figure).
- Refs bibliography (`paper/refs.bib`) and related-work appendix
  (`paper/appendix-related-work.tex`).
- Section 6.5 binary loop diagnostic decision: confirmed appendix-only,
  cost-benefit argument in `gold/phase-3-section-65-decision.md` holds.

**Missing or incomplete (paper-side, no new experiments needed):**

1. **§3 HERALD v1 (regression) results** — not yet drafted. Needs:
   (a) per-token ρ table (label, baseline, achieved, ceiling, bar);
   (b) per-run wrapper table (4 targets); (c) retention table
   (press/ratio/task); (d) calibration ECE; (e) streaming parity remark.
2. **Per-fold figure for retention** (LOPO press, LOO ratio, LOO task,
   three panels, each showing per-fold ρ vs overall ρ and the 0.95 bar
   line). Useful for Bar 6/7 narrative.
3. **Reliability diagram** for the wrapper (10-bin quantile, pred-rank vs
   y-rank). Visualises ECE = 0.036 better than the scalar.
4. **Triangulation table** for Bar 6 (slopefeat / per-ratio retrain /
   extfeat, with worst-slice ρ and pooled ρ). Currently buried in prose.
5. **Substrate-ceiling-by-target appendix table** (the 4-row table at end
   of `gold/phase-3-herald-v1-results.md`) into the appendix.
6. **Cross-model transfer disclosure** (S3 above) — explicit Limitations
   paragraph at minimum.
7. **Fix the four blockers (B1–B4) in the gold doc and propagate** any
   numbers or framing changes to the paper draft.

**Out of scope (decisions deferred):**
- Whether to compute a multivariate substrate ceiling for `rouge_l_drop` /
  `sum_js`. Not required for submission if Bar 3 / Bar 6 framings (B2, B3)
  are tightened as recommended.
- Whether to expand to a second model family (S3). Either run it or
  declare the single-model scope; do not leave it implicit.

## 8. Recommended next actions

1. Edit `gold/phase-3-herald-v1-results.md`: apply B1 (terminology fix +
   re-baseline), B2 (univariate-ceiling caveat), B3 (multivariate-aggregator
   caveat on Bar 6 spread argument), B4 (drop "pre-committed kill"
   language). One pass, ~30 minutes.
2. Decide on B4 chronology (commit script with backdated narrative vs
   reframe). Recommend the reframe.
3. Drag the four Phase 3 result tables into `paper/main.tex` §3 (template
   already exists from §2). Use the per-press / per-ratio / per-task tables
   from the JSON files verbatim.
4. Decide cross-model transfer scope (S3). If declaring out of scope, write
   the Limitations paragraph now.
5. Optionally regenerate the V1 unnormalised-counts wrapper JSON (S2) for
   the appendix.

None of the above requires new GPU work. After the doc and paper fixes,
the submission posture is "READY".
