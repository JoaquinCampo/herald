# Population accounting after lookahead

All 216 singleton IFEval candidates are exhausted. The corrected partition is
120 accepted pilot prompts, 20 pilot early-EOS cases, and 76 completed test
candidates. The test comprises the 20 unused pilot-manifest entries plus 56
previously unselected entries. The pilot checkpoint and test are disjoint.
There is no remaining reserve. Naming a prompt in a manifest is not itself
outcome contamination; the processed ledger establishes this result.

Evidence: `results/pilot-v1/checkpoint.json`, `data/pilot-v1/prompts.json`,
`data/lookahead-v1/train-prompts.json`, `data/lookahead-v1/test-prompts.json`,
and `data/lookahead-v1/roster-derivation.json`. The initial source accounting
was 541 official rows, 321 previously exposed, then four excluded duplicates,
leaving these 216 singleton candidates.

Existing exposed cases may support explicitly exploratory mechanism diagnostics.
They cannot provide another untouched predictive evaluation. A later study
needs an independently sourced population with a pinned deterministic scorer,
provenance and split. No new population has been selected or downloaded.
