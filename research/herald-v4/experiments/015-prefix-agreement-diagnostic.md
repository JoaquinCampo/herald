# Fixed short-prefix disagreement diagnostic

Use the existing20 RULER development prompts at the same shared prefill boundary
and actions .05/.10/.20. Do not return to the earlier token32 IFEval assay or
change both task and boundary before exhausting cheap evidence already available.

Hypotheses: (1) harmful retrieval changes already alter a short generated prefix;
(2) harmless counting variations also alter it, so disagreement is nonspecific;
(3) quality improvements create the same disagreement and require signed modeling;
(4) quality changes emerge later than a short prefix, limiting early observation.

Lock one observation before computing it: unequal first16 generated token-ID
sequences between reference and action, including unequal truncated lengths.
Do not use task names, answers, final scores or continuation tokens beyond16
to construct the observation. No horizon sweep, threshold choice or predictor fit.
Report the contingency table against negative/zero/positive signed loss, capture
rate for degradations, disagreement rate among unchanged scores and mean signed
loss in each group, overall and by task. These are descriptive, exposed-data results.

The observation is computationally available at decision time by simulating16
reference tokens and16 tokens under each candidate action from cloned boundary
state. Existing full continuations reproduce those prefixes deterministically.
This is not a zero-cost state feature. A deployed realization would cost at most
16 generated tokens per arm, with early EOS allowed and a shared reference branch.
Do not infer wall-time overhead from this offline extraction. Measure it separately
only if the diagnostic warrants a new prespecified predictor study.
