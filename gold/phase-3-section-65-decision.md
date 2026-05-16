# §6.5 Binary Loop Predictor: Keep As Slice Diagnostic

## Decision

Keep commits c7d0d7b (§6.5 meta-aggregator) and d707858 (run-level
wrapper + GRU lever) in the tree. Do not revert. Do not promote
their AUROC numbers to HERALD v1 headline. They are slice
diagnostics on a derived binary tag, not a competing headline
predictor.

## Why not revert

The HERALD v1 spec (`/goal`, `gold/research-plan.md`) names the
primary regression label `future_sum_js_H`. Diagnostic tags
(`has_looping`, `has_non_termination`) are explicitly "eval slices
only" and "binary tags are secondary." That makes the §6.5
artifacts the wrong target for the headline, not invalid.

But the artifacts have independent value:

1. They establish that **the per-token logit substrate carries
   enough signal to discriminate the looping failure mode** on a
   run-level binary at AUROC 0.95-0.97, even with the chosen
   GroupKFold(prompt_id) splits. This is a non-trivial finding on
   its own and is the strongest empirical evidence that
   compression-attributable damage is detectable from black-box
   logit features at deployment.

2. They give a worked example of a **run-level meta-aggregator**
   over per-segment OOF scores. The same eight aggregates (max,
   running mean, running std, top-3 mean, top-5 mean, last-3 mean,
   last-5 mean, n_segs) and the same no-leakage meta protocol
   (train on aggregates from k-1 folds, eval on holdout) transfer
   verbatim to the regression setting. HERALD v1's per-run
   validator against `rouge_l_drop` can re-use this scaffolding.

3. They surface the **§6.3 deployment-faithful ceiling**: the
   run-level max-pool over per-segment scores at AUROC 0.948.
   Without this, the v1 paper would have no anchor for "what does
   the obvious naive aggregation get on the binary task," and the
   regression contribution would be harder to position.

## Why not promote to headline

The binary loop AUROC numbers do not, on their own, support the
HERALD v1 claim. Three independent reasons:

1. **Wrong target.** The /goal forbids diagnostic-tag training
   targets for the headline. `has_looping` is a heuristic flag,
   not a paired counterfactual damage label. Any high AUROC on
   `has_looping` is consistent with predicting "this run failed
   in some loud way" rather than "this token will see H tokens
   of high JS divergence vs the uncompressed reference."

2. **Substrate-overlap risk.** The §6.5 meta-aggregator was
   trained over per-segment OOF scores that were themselves
   trained on segment-level futures correlated with looping.
   The 0.96 macro AUROC bar is honest within the locked rule but
   not robust to bootstrap variability (pooled CI lower bound
   0.9568, three of five folds below the point bar). Promoting
   this to a "HERALD beats 0.96" headline would invite a referee
   to ask whether the bar is gameable.

3. **Lead-time inversion (Phase 2c).** The §6.5 score has high
   discriminative power on whole-run outcomes but does not give
   clean pre-onset lead time on `has_looping`/`has_non_termination`
   (recorded in `gold/phase-2c-early-warning-results.md`). The
   control claim therefore cannot lean on §6.5 alone; it needs a
   regression signal with pre-onset behaviour.

## What this means for §7 of the paper

§6.5 figures stay in an "Appendix: Binary Loop Diagnostic" section.
The HERALD v1 headline numbers are regression-Spearman against
`future_sum_js_H` (per-token) and `run_damage.sum_js` (per-run),
with cluster-bootstrap CIs over prompt_id. The §6.5 numbers are
reported as a separate cross-check: "the same online-feature
substrate detects looping at 0.95-0.97 AUROC, which is consistent
with the regression Spearman scores reaching the substrate ceiling
on the looping subpopulation."

## Cost of keeping

Approximately zero. `gold/phase-2d-streaming-online-results.md`
and the §6.5 / d707858 scripts are documentation and one-off
analyses; they don't sit in the training path. They impose no
constraint on the v1 regressor, the transfer slices, or the
streaming demo. Reverting them buys nothing.
