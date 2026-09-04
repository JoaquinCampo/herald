# Paper structure

Per-section job statements and current status. Framing: HERALD predicts
continuous compression damage (`future_sum_js_H`) from zero-cost logit
features, online and O(1). Earlier binary-classification (AUROC, has_looping)
framing is deprecated; §6.5 binary loop diagnostic stays as an appendix only.

| # | Section | Job | Status |
|---|---------|-----|--------|
| - | Abstract | One paragraph. Damage trajectory, paired counterfactual replay, calibrated run-level score, cross-compressor transfer. | written |
| - | Figure 1 (teaser) | One image showing the predicted damage rising before failure on a real run. Caption refreshed; underlying PNGs predate Phase 3 and should be regenerated against the H=25 regressor before submission. | partial |
| 1 | Introduction | Convince a reader who never opens any other section. Damage-as-regression framing, headline ρ, the regimes where the signal degrades, runtime-intervention so-what. | written |
| 2 | Catastrophic failures under KV-cache compression | Define looping and non-termination detectors, their prevalence across compressors and ratios. Used to characterize the failure regime; HERALD trains on the continuous damage target defined in §3, not on these binary labels. | written |
| 3 | HERALD | Define `future_sum_js_H`; describe 24 zero-cost features; HGB per-token regressor; trajectory-shape per-run wrapper (17 pred-only features, no metadata); streaming O(1) implementation. | skeleton |
| 4 | Experimental setup | Model, tasks, compressors, ratios, sweep design, paired counterfactual replay protocol, splits (prompt_group / LOPO press / LOO ratio / LOO task), baselines (pred_max, meta-only), cluster-bootstrap CI. | skeleton |
| 5 | Results | Per-token ρ across horizons, per-run wrapper across 4 targets with substrate ceilings, leave-one-compressor / -ratio / -task generalization, calibration, streaming parity, and explicit reporting of the regimes where the signal degrades. Apply B1–B4 audit framing fixes. | skeleton |
| 6 | Analysis | Per-press, per-ratio, calibration reliability diagram, feature importance, inference cost, error modes. | not started |
| 7 | Discussion / Limitations | Single-model scope (Qwen2.5-7B-Instruct only) called out explicitly. Substrate ceilings as a structural limit. Phase 4 controller as future work (pilot exists, publishable run pending corrected grid). | not started |
| 8 | Conclusion | Short. Restate the damage-as-regression framing and the runtime-substrate angle. | not started |
| - | References / Appendix | Related-work appendix already drafted (`appendix-related-work.tex`). Bibliography in `refs.bib`. Binary-loop §6.5 diagnostic belongs here as a separate appendix. | partial |

## Decisions

- **Source of truth for numbers**: `results/phase3/eval/summary.json`,
  `results/phase3/run_level_wrapper_trajectory.json`, and the bar-6
  triangulation JSONs. Do not invent numbers; pull from these.
- **B1–B4 framing fixes from `notes/phase-3-validation-2026-05-15.md`
  bake into §5 directly**: rename "strict pred-aggregates-only" to
  "trajectory-shape pred-only wrapper" and disclose the +0.017 margin
  of the literal 8-feature strict variant; refer to the
  `rouge_l_drop` substrate ceiling as the best single-aggregator
  ceiling, not the substrate ceiling; scope the per-slice retention
  argument to the single-aggregator regime; drop "pre-committed kill
  threshold 0.65" language.
- **No standalone Related Work section**: the intro does the framing,
  and a related-work appendix exists; can promote to body if reviewers
  ask.
- **Figures to render before submission**: (a) teaser regenerated
  against the H=25 model; (b) per-fold retention figure (LOPO press,
  LOO ratio, LOO task, three panels); (c) wrapper reliability diagram
  (10-bin quantile, pred-rank vs y-rank); (d) feature-class ablation
  bar chart (already in `results/phase3/feature_ablation.json`).
- **Workflow per section**: outline as LaTeX comments inside `main.tex`,
  draft prose in place, commit incrementally. Stop using separate
  `paper/sectionN_*.md` planning files (the May 1 drafts were on the
  old framing and have been removed).
