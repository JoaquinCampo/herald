# Prior evidence, read individual sources only when needed

Archive: /Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v3.
This organizational reset does not reset exposure. No predictor is validated.

| Evidence | Established | Limit |
| --- | --- | --- |
| [Pilot v1](../herald-v3/docs/research/pilot-v1-result.md) | On 120 Qwen2.5-7B IFEval prompts, token-32 Knorm assay, immediate JS addition had loose MSE 0.194360 versus baseline 0.182044. Independently audited negative result. | Scoped to fixed measurement/model/actions/boundary; not general impossibility. |
| [Recovered lookahead](../herald-v3/docs/research/lookahead-recovered-result.md) | Delayed JS MSE 0.189879 versus immediate-JS 0.185922; no supported gain. | Original operational-integrity failure persists. Recovered numbers are exploratory, not confirmation. |
| [Population audit](../herald-v3/docs/research/fresh-population-review.md) | All 216 singleton candidates accounted for: train 120, early EOS 20, test 76. | No untouched reserve in that population. Exposed data can support declared exploration only. |
| [Retrieval stop](../herald-v3/docs/research/retrieval-discovery-v1-stop.md) | Eager/SDPA exact token parity failed before head discovery. | Mechanism untested. Original failing sequences were not saved; cause/magnitude cannot be inferred. |
| [Shadow v2 review](../herald-v3/results/shadow-attention-independent-review-v2.json) | Independent GO for one fixed case, two label-free native-SDPA continuations with diagnostic QK rows; CPU chain verified. | No GPU result established here. QK shadows are not fused-kernel probabilities; no quality/predictor claim. |

Potential reusable components remain in v3 until an experiment needs them:
src/herald_v3/shadow_attention.py, scripts/check_shadow_attention.py and
tests/test_shadow_attention.py. Review their dependencies and sealed protocol
before any reuse. Do not assume copying the module preserves that protocol.

The research brief and autonomous scope are restated in RESEARCH.md. Detailed
historical owner state is deliberately excluded from routine reading.
