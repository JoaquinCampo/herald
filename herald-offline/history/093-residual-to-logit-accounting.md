# Reading the current-forward margin without mistaking accounting for causality

Understanding continuation after092, not a new prediction experiment. No model
inference or new outcome collection in this step. We inspected the installed
Qwen2RMSNorm, Qwen2MLP and bias-free lm_head and verified the algebra on CPU.

## What the output decision measures

Fix the two old case000 output tokens,0 and period. Let v be their lm_head weight
row difference, gamma the final RMSNorm weight, h the stored final residual state,
and s=sqrt(mean(h^2)+epsilon). In exact real arithmetic:

    margin = logit(0)-logit(period) = (v*gamma) dot h / s.

Since s is positive, denominator rescaling alone cannot reverse this margin's
sign. A sign change requires a changed projection onto c=v*gamma, or numerical
effects omitted by the ideal formula. This does not mean normalization is
irrelevant to magnitude or to upstream layer computations. Gamma is fixed
between branches; this argument concerns only the final normalization.

The actual implementation casts h to float32 for variance/rsqrt, casts the
normalized result back to the input dtype, then multiplies by gamma. The lm_head
projection also uses finite precision. We must measure their discrepancies,
not call the ideal equation an exact description of stored logits.

## An accounting identity that retains numerical effects

For each branch b, define h_b as the stored pre-finalnorm state, n_b as the actual
norm output, and m_b as the actual saved logit margin, subtracted in float64.
Choose float64 s_b as an analysis reference denominator, not the implementation's
fp32 denominator. Define:

    eN_b = n_b - gamma*h_b/s_b
    eL_b = m_b - v dot n_b

For reference R and history-replaced hybrid H from091, the identity is:

    m_R-m_H = (c/s_R) dot (h_R-h_H)
              + (c dot h_H)*(1/s_R-1/s_H)
              + v dot (eN_R-eN_H)
              + (eL_R-eL_H).

These terms separate stored-state displacement, denominator scaling, norm
implementation discrepancy, and readout implementation discrepancy. They are
not four independent causes. The chosen reference denominator makes the
accounting ordered; swapping the anchor reallocates terms. Record the actual
fp32 denominator too if a real trace is later selected.

For layer accounting, telescope stored state increments at each residual add:
attention increment = stored post-attention-add minus stored block input;
MLP increment = stored block output minus stored post-attention-add. Identical
pending tokens imply identical initial embeddings. Summing the differences of
these increments reconstructs h_R-h_H. Do not substitute raw attention/MLP module
outputs, because bfloat16 addition rounds before the next stored state.

## CPU validation and its limits

scripts/verify_residual_accounting.py uses width32 synthetic residual updates,
the actual installed Qwen2RMSNorm and a bias-free Linear readout on CPU. It is a
mathematical fixture, not Qwen inference or evidence about compression risk.
Observed margin difference .004638671875 reconstructs within3.82e-17 when all
terms are included. Naively summing emitted updates instead of stored increments
misses the final residual by maximum .0075073 and .0114136 in the two branches.
The small fixture's rounding terms are material relative to its margin; this
cannot establish the size of rounding effects in real case000.
Result: results/residual-accounting-cpu.json, with source hash and runtime.
An initial harmless autograd warning was removed by detaching analysis weights;
repeating the same CPU fixture preserved every numerical result.

Independent Luna review verified signs, telescoping and finite-precision scope.
It emphasized that the float64 denominator is an analysis reference, that eN/eL
must remain, and that a real trace should record the fp32 denominator as well.

## What this would and would not teach from a real trace

A complete trace can show which stored updates contribute to the final pairwise
margin and whether normalization/rounding materially alter the accounting.
A large projected update does not identify a causal head or necessary layer:
it may compensate for or propagate an upstream change. Later increments depend
on earlier states. Reading each intermediate state with the final lm_head is
not itself a valid assertion of what that intermediate layer knows or predicts.

Existing091 raw logits alone cannot supply the missing states. If a trace is
selected, capture all block input/post-attention-add/output vectors, final norm
input/output, fixed readout rows and gamma in one unchanged forward per arm.
Require exact saved091 logits, unchanged source caches, hook-free replay and
complete reconstruction, and report all terms without selecting winning layers.
This is an observation contract only, not a selected layer sweep or learned
feature. The next decision remains how to answer the specific current-forward
mechanism question with the least additional measurement.
