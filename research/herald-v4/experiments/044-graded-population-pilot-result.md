# Graded population pilot result

The frozen 16-case population passed all study043 viability requirements.
Every uncompressed reference recovered all four values. Signed score losses
were 1.00 in nine cases, 0.75 in two, 0.50 in two, 0.25 in one, and zero in two.
There were five strictly partial-loss cases; sample standard deviation was
0.37046 and mean loss was 0.73438. No prediction model was fitted.

The population remains harsh: nine of sixteen cases lose every complete answer.
The partial-credit metric nevertheless exposes five loss levels, while perfect
reference scores separate this pilot from ordinary model retrieval failure.
Many incorrect outputs contain shortened numeric strings, so changing the
metric has not established a new failure mechanism.

The existing pair runner was unchanged. All 16 cases completed with equivalent
independent B0 state, exact reference/noop tokens and termination, immutable
source caches, and exact native Knorm .05 physical-effect checks. The owner
recomputed the pinned official score for all reference/noop/action outputs.
This is direct owner verification, not a new independent-agent audit.

Evidence: results/graded-v1-pilot, results/graded-v1-summary.json,
scripts/analyze_graded_pilot.py. Manifest SHA
2cda6bfa7fb492647dd9ab8f8e341b20dc90fe4ff7d32e609f2663e7cbbfce5f.
All 16 contexts are now exposed development data. No confirmation was opened.
Next choose an observation and learner prospectively; passing an outcome-
variation check is not evidence of useful loss prediction.
