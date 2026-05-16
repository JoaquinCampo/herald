# Paper structure

The plan we are working from. Each section has a one-line job statement so we can sanity-check whether prose actually serves the argument.

| # | Section | Job | Status |
|---|---------|-----|--------|
| - | Abstract | One paragraph. State the problem, the claim, the headline number. | ✅ |
| - | Figure 1 (teaser) | Show, in one image, that the warning fires before the failure. | ✅ |
| 1 | Introduction | Convince a reader who never opens any other section. Hook, gap, contribution, headline, contributions list. | ✅ |
| 2 | Catastrophic failures under KV-cache compression | Define the three failure modes, how we detect them, how often they occur across compressors and ratios. Earn the "we characterize…" contribution. | ⬜ |
| 3 | HERALD | Features (zero-cost logit signals), labels (token-level hazard with horizon $H$), model (XGBoost), training procedure. | ⬜ |
| 4 | Experimental setup | Model, dataset, sweep design, train/test split, LOCO protocol, baselines (rolling entropy). | ⬜ |
| 5 | Results | Main AUROC across horizons, LOCO generalization, pre-onset performance, baseline comparison. | ⬜ |
| 6 | Analysis | Per-press behavior, per-ratio scaling, calibration, error modes, feature importance, inference cost. | ⬜ |
| 7 | Discussion / limitations | What the signal is and isn't, caveats around the GSM8K / Qwen choice, generality claims. | ⬜ |
| 8 | Conclusion | Short. Restate contribution and the runtime-intervention angle. | ⬜ |
| - | References / Appendix | Citations, supplementary tables, extra qualitative examples. | ⬜ |

## Decisions

- **No standalone Related Work section for now.** The intro already does the framing work. Many recent ML papers fold positioning into the introduction; we can add a standalone section if reviewers ask.
- **Workflow per section.** Outline as markdown (`paper/sectionN_*.md`), agree on structure and arguments, draft prose, commit to LaTeX.
- **Numbers come from existing artifacts.** `models/metrics.json`, `models/analysis/per_press.json`, `models/analysis/per_ratio.json`, `models/analysis/calibration.json`, `models/analysis/errors.json`, `models/analysis/importances.json`, `models/analysis/qualitative.json`, `models/analysis/inference_cost.json`. Do not invent numbers; pull from these.
- **Figures planned beyond Figure 1.** TBD as we write each section. Likely: a per-press / per-ratio prevalence plot in §2, a feature-importance bar chart in §6, a calibration plot in §6.
