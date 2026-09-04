# Figure plan

| ID | What it shows | Section | Priority |
| --- | --- | --- | --- |
| F1 | Two-panel hero: left, a real compressed generation drifting into failure; right, HERALD's predicted damage rising tens of tokens before the failure is visible, against a flat reference signal. Concept carried from the prior draft's Figure 1, to be redrawn from the new forecaster. | Introduction | High |
| F2 | The target construction: a compressed run's realized prefix, the teacher-forced uncompressed evaluation on the same prefix, and the per-step divergences summed over the horizon window [t+1, t+k]. | Forecasting Target | High |
| F3 | Forecaster input/output contract: causal zero-cost logit statistics flowing in, multi-horizon damage forecasts flowing out; oracle fields marked as train-only. | Method | Medium |
| F4+ | Held-out results plots: per-horizon predicted-vs-realized, stratified degradation, quality-failure separation, transfer comparison. Exact selection follows the completed analysis. | Results | TBD |
