# One-case EA measurement result

The existing CWE002 development case passed on Orion Qwen2.5-7B-Instruct BF16,
kvpress 0.5.2, with the shared split-prefix protocol. This establishes measurement
transparency on one case, not predictive validity. No new outcome was sampled.

Frozen source: results/ea-v1-launch/frozen-measure.py,
SHA256 a0f41f7ce023c4534ad6fb179619d9719b6abc987903bcec062c2d32a0baf903.
Actual remote exit 0; measurement and owner-verification in results/ea-v1/.
Exact prompt and reference tokens also match the prior successful shared pilot.

Plain and instrumented cache/source tensors were exactly equal, independently
stored, and unchanged after independent no-op continuations. Tokens and termination
were identical. All 112 layer/KV-head summaries per action were finite. Masks used
native BF16 key norms, matching direct KnormPress selection; nonpositive or
nonfinite mass denominators fail explicitly. Owner recomputed excess from stored
mass and removal fraction across all 336 head/action rows.

Synchronized summed hook cost was 0.2383 seconds, including EA scoring, three action
summaries, assertions, and CPU transfers. Plain prefill 0.5489 seconds and instrumented
prefill 0.6521 seconds were sequential cold/warm observations, so their difference
is not a controlled deployment-overhead estimate. Instrumented peak increment
was 1,642,742,784 bytes relative to its own baseline, not attributable EA memory alone.

Mean non-sink EA excess was -0.1042/-0.2177/-0.2698 for removal .25/.5/.75.
These values are feature measurements only; their sign does not establish signed
quality-loss prediction. Proceed with locked exploratory design 007 and new data.
