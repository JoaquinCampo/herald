# Research restart: understand before collecting

2026-09-14. The user resumed research, rather than manuscript work, and explicitly
reaffirmed FRAMEWORK.md. This record defines an analysis-only restart from 065.
No new GPU collection, feature search, predictor fit, or confirmation is selected.

## What needs verification

The local paper branch predates the final research checkpoint. Research source
is recovered from commit 3884a1a in a separate local worktree. Orion's existing
v4 workspace is the data authority. Its RTX 5090 is reachable; the historical
v2 keepalive remains active. The old quality-risk v1 numerical evidence is not
present in the checked local branch or the searched remote recovery directory,
so its narrative is not treated as a newly reproduced result.

The strongest positive assumption result in v4 is study 052: same-prompt losses
at other compression rates cut an exposed-data prediction error by 50 percent.
This is privileged information and cannot serve as a prospective predictor.
The latest MuSiQue pilot instead reports no EA loss variation. Neither result
establishes impossibility of the research target.

## Competing hypotheses and cheapest discriminators

| Hypothesis | Expected observation | What would weaken it |
| --- | --- | --- |
| H1: prompt-specific susceptibility survives across rates | Recomputed raw losses match 052, and alignment beats independently permuted rate columns with the same marginal failures | The published effect fails reproduction or is typical under that null |
| H2: marginal failure rates explain the apparent stability | Breaking prompt identity across rates often preserves the reported gain | A small conditional permutation tail probability |
| H3: one prompt drives the signal | Deleting one prompt removes or reverses the gain | Positive gains throughout the leave-one-prompt deletion analysis |
| H4: poor task/action outcome variation caused some recent failures | Official rescoring reproduces low reference competence and zero EA loss in the fixed MuSiQue slice | Nonzero signed losses or a different canonical reference after a valid audit |
| H5: prior work offers a decision-time observable missing from our tests | A primary source links an available action-conditioned variable to completed task loss under matched baselines | The source predicts attention/NLL error, changes the model, or uses future outcomes instead |

## Selected small test, written before execution

Reconstruct all 20 prompts and 60 compressed outcomes from unchanged study 007/010
records. Verify source hashes, complete action sets, no-op parity, unique prompt
identities, scores and signed loss. NIAH's 12 prompts are the primary stratum;
CWE's eight remain descriptive. Recompute 052 without calling its diagnostic
function. Do not sample only successes or pool tasks.

For the binary NIAH matrix, hold the first rate column fixed and enumerate the
distinct arrangements of each other rate column, keeping its positive count.
The statistic is the original privileged proportional MSE reduction. Report
the fraction of arrangements at least as large as observed, including ties.
This is conditional on independent rate-column exchangeability under the null.
It is a post hoc diagnostic on selected, exposed data, not a confirmatory p-value,
and does not adjust for the preceding research search. Report single-prompt
deletion sensitivity as a stability check, not a confidence interval.

Also re-score the 16 canonical MuSiQue records with the preserved official
scorer and compare all 80 branch scores and hashes with the existing summary.
Keep its original feasibility criteria and sample grouping. Inspect two fixed
records selected by manifest position, not by score.

## Assumptions and next decisions

Source hashes prove byte continuity, not scientific validity. The preserved
scorers define each task's quality; their semantics and exact reference branch
must be checked. Prompt rows are the assumed units for the NIAH diagnostic;
small synthetic samples cannot establish population transfer. Data exposure
does not reset on a new branch or computer.

If reproduction fails, preserve the discrepancy and return to data understanding.
If H1 survives, ask what admissible measurement could identify susceptibility;
do not infer that any proposed feature works. If H2 or H3 survives, reduce the
weight assigned to 052. A new GPU experiment still requires a concrete mechanism,
matched information baselines, adequate outcome variation, an explicit observation
cost, and an untouched future evaluation population. The sequence is evidence
audit, mechanism review, observation definition, technical slice, development
study, then separately frozen evaluation if warranted.
