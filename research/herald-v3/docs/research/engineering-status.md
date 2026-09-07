# Engineering status

The eight-prompt full-budget run completed on Orion using Qwen2.5-7B BF16
SDPA, 1024 total output tokens and Knorm removal .25/.50. Seven prompts
reached the token32 decision boundary and passed all21 controls each,
147 checks total. All49 generated responses matched independent official
IFEval strict and loose instruction pass vectors. One reference ended at8
tokens, so it is an explicitly recorded ineligible decision.

Evidence: results/engineering/eight-01/{run.json,artifacts.json,
official-parity.json,owner-validation.json}. The original runner classified
ineligibility as overall failure. Its original artifacts remain untouched;
the real saved-row reproduction identified the aggregation defect, the
predicate was fixed, and the reproduction plus5runner tests pass. No
outputs, scores or engine gates changed. Source hashes match except this
subsequent documented aggregation-only fix in runner.py.

The engineering actions include5 positive,7 zero and2 negative signed loose
score differences. This verifies signed labels and meaningful variation in
this convenience slice, not predictive ability or a population estimate.

The separate no-reference/probe versus preserved-reference/probe diagnostic
passed its40-token full-model smoke with exact state/action/output parity.
It measured18,527,744 additional peak allocated bytes for the joint
preservation and probe mode on that prompt. The1024-token diagnostic also passed all six checks with exact full
continuation parity. Its joint-mode peak allocated difference was7,988,224
bytes (15,375,889,408 versus15,383,877,632). The pair took11.173 and11.183
seconds; this single ordered comparison is not a reliable latency-effect
estimate. Full engineering acceptance is complete. It is a joint
cost comparison, not an isolated probe cost or deployment result.

Earlier synthetic engine smoke passed21gates; the one-prompt scored smoke
also passed with7official response-vector comparisons. All evidence is
retained under results/engineering. No predictor has been trained.

Pilot collection is now running under results/pilot-v1 on Orion, following
the frozen160candidate manifest until120eligible cases. It passed the
10-second launch check. No predictive result is available yet. The fixed
evaluator is being implemented/tested independently of pilot outcomes.
