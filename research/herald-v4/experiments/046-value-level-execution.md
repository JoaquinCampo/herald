# Value-level study execution

Design045 and prospective amendment045a were frozen before any new outcomes.
Data contains128 discovery and64 locked evaluation prompts, with no protected
prompt/context collisions. Independent data and source preflight passed.
The final adapter locates four exact native seven-token spans in192/192 prompts.

The first CPU adapter used prompt length minus two for normalized position,
while045 specified full prompt length. This was corrected before GPU collection,
and the exact CPU prior replay passed again. Final collector SHA:
971fba0d6e771ef3397dd67045f2742a30b85068cdf06dcfd01f4f2f4e44121b.
Adapter SHA6b2e329f1ed51c7878c0ad58d3fbbcbf2797e3445c07b6c2f0f2a256612bfd43.
The old graded-v1-000 GPU replay reproduced reference, action, termination and
native mask hashes exactly, with four118-feature rows and valid signed scores.

All128 discovery cases completed. There were71 prompts with mixed per-value
signed losses, exceeding the required24. The candidate and four baselines fit
once on discovery using frozen settings. Model SHA:
d7192153c8c5b6fd2c1763c25fbd3f0e61e60fad28e34bb30ccdd870235b3f52.
Model script SHA43d0d8c8584247bc1909367f50000f18ccae20e07a7d9a19a13308f65c737f27.

Independent batch checks all passed, but the audit initially miscomputed its
final status flag. Correcting that audit-only logic and enforcing nonzero exit
on audit failure made the original full batch pass. The exact failed report is
preserved at results/value-level-discovery-audit-original-failure.json, verified
against its original SHA e3ce426e9fa463c029bf8df3def6edf52b0539fb0915bec796c91ae97a046ca6.
The independent fit audit also reproduced all Ridge/OLS coefficients.
No data, features, fitted model or prediction gates changed during that fix.

All64 evaluation cases completed, with42 mixed prompts and all references
perfect. Signed task losses were0:5,.25:12,.5:13,.75:17,1:17; no negative
contributions occurred in this population. Candidate MSE was0.1027406740.

Compared with mean, structural Ridge, aggregate Ridge and aggregate OLS,
MSE gains were7.729%,7.654%,5.683% and -2.865%; strict wins were33,33,36,31
of64. Every comparison failed the frozen10%gain and43wins requirements.
Candidate within-prompt concordance was0.710317 versus structural0.595238,
so that narrower ranking gate passed. Overall prediction gate FAILED.
The paired fixed-fit bootstrap gain interval versus mean was[3.36%,11.90%],
and versus aggregateOLS[-13.83%,8.10%]. These exclude fit uncertainty.

The full independent192-case audit reproduced scores, features, coefficients,
evaluation predictions and gate arithmetic:1378 checks passed. During its
first evaluation run, an audit-local scalar shadowed the concordance function;
the exact failing source is preserved at
results/value-level-audit-failures/pre_eval_fix.py. The local name and audit gate
reporting were corrected, then the exact original command passed. A correctly
reported failed scientific gate is valid audit evidence, not an integrity error.
Final audit SHA603a7ccff1a377fa0c0d07d85e0043de67d68011daae9e57ab497fade2285b18.
No model, outcome, data or scientific threshold changed during audit fixes.

Median observation wall time0.29685s includes compression validation;
median prediction0.000518s, prefill0.41718s separately. This is measurement,
not a deployment claim. See results/value-level-evaluation-cost.json.

Close the mask-retention prediction family operationally. Preserve the narrower
value-ranking finding, but do not promote the stronger OLS baseline post hoc,
search nonlinear models, tune penalties or select heads on these outcomes.
All192 prompts are now exposed development data. Confirmation remains unopened.
Evidence: results/value-level-evaluation-summary/summary.json and
results/value-level-independent-audit.json. Next return to mechanism under047.
