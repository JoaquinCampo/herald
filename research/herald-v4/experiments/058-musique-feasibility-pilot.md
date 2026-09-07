# MuSiQue fixed QA feasibility pilot

Owner selected2026-09-07 after054data review and057independent EA CPU acceptance. This is diagnostic exploration, with no predictor fit or confirmation outcomes.

## Competing hypotheses

1. Qwen2.5-7B can answer the complete two-hop context reliably, making paired compression loss interpretable.
2. Imperfect reference competence dominates. Poor reference F1 closes this fixed slice without prompt or horizon tuning on these outcomes.
3. EA preserves all answer quality at removal.10. No graded variation closes this fixed task/action slice without severity tuning.
4. Graded signed losses appear and differ from matched Knorm. This justifies designing a grouped prospective study, but does not establish admissible predictive information.

## Fixed design before outcomes

Acquire official MuSiQue-Ans dev through the sources in054. Record upstream revision, URL, archive/file SHA256, license, acquisition time and transformations. Preserve all original rows; do not inspect the test split. Select answerable two-hop rows by official ID ascending, first16, no outcome-based or length-based filtering. Validate20 complete paragraphs and decomposition/support integrity; retain original paragraph order. Audit normalized-question and component group overlap with protected prior data, without importing labels into prompts.

One user message, rendered with the pinned Qwen chat template and add_generation_prompt=True:

```text
Read the passages and answer the question. Return only the short answer, without an explanation.

[Paragraph {idx}] {title}
{paragraph_text}

(repeat all paragraphs in their original order)

Question: {question}
Answer:
```

Keep gold answers, aliases, support labels and decomposition out of model input. No assistant answer prefix. No truncation: if any selected prompt exceeds the pinned model context allowance including output, stop and report; do not substitute another row.

Freeze greedy decoding, seed0, max_new_tokens64, pinned Qwen2.5-7B revisiona09a35458c702b33eeacc393d103063234e8bc28. Before generation, report selected gold/alias token lengths and verify64 covers the longest answer plus16 tokens. If not, return to owner before any generation rather than silently changing horizon. Native EOS stopping is retained.

Reference is the ordinary shared split-prefill B0 continuation. Action is native EA removal.10 with512 future positions,4 sinks, covariance and vnorm enabled, epsilon0. Matched comparator is native Knorm removal.10. Independent instrumented no-op continuation audits score collection. Apply each action before the pending final prompt token at the same logical position. Plain and instrumented B0 caches must match exactly; no-op tokens and termination must match reference for every row. Preserve raw generations, failed outcomes, state isolation, mask/count/byte metadata and stage timings. No budget sweep.

Primary score is official answer token F1, maximum over answer and aliases, in[0,1]. Score raw decoded answer using the official normalizer, with no extra answer extraction or hand correction. Signed loss = referenceF1 minus actionF1. Report negative, zero and positive values separately. Knorm is a descriptive matched comparator, never a substitute to rescue an EA gate failure.

## Frozen feasibility gates

All16 rows must be scored, with all state/no-op gates passing. At least12 reference answers must have F1>=.80. EA action scores must have at least3 distinct F1 values, and at least4 rows must have positive signed loss. Report the complete signed-loss distribution and reference/action pairs. This is a small competence/variation gate, not prediction or deployment evidence.

Prepare and independently audit data/scoring first. Then implement the smallest experiment-local real-model runner. Verify Orion hardware and process ownership before GPU use; first run only the first selected row for technical acceptance, then the remaining15 under the identical frozen protocol if its technical checks pass. Do not interpret one-row task quality as a stopping or selection rule. On real technical failure preserve exact input/environment, test competing causes, rerun original reproduction before continuing.

A pass allows a new owner-selected discovery design with matched information baselines, grouped splits and sealed unseen evaluation. A failed fixed slice is closed without prompt, severity, feature or learner search on its outcomes. All16 pilot rows and related component groups remain exposed thereafter.
