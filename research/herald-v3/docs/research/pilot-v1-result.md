# Pilot v1 result

The fixed one-step distribution-divergence specification did not improve
prediction of signed final instruction-compliance loss. This is a negative
result for this measurement, model and evaluation specification, not an
impossibility result for the broader research objective. Independent
numerical, implementation and data audits passed.

## What was tested

Qwen2.5-7B-Instruct, BF16 SDPA, greedy decoding, 1024 total new tokens.
At 32 committed output tokens, each candidate evicts .25 or .50 of the
live KV entries with Knorm, then decoding continues with normal cache growth.
The target is loose IFEval reference score minus action score, retaining
improvements as negative labels. Strict scoring is a sensitivity analysis.
The candidate feature is full-vocabulary Jensen-Shannon divergence between
matched one-token sandbox forwards, added to fixed scalar-probe features.

The frozen procedure uses five prompt folds, train-fold StandardScaler and
Ridge(alpha=1), no tuning, and prompt-equal MSE. The action-wise training mean
is an additional baseline. All action rows from a prompt share one fold.

## Observed evidence

The collection reached 120 eligible prompts after 140 candidates. Twenty
references ended before the decision boundary and remain in the evidence.
There are 240 scored candidate-action rows: 69 losses, 156 unchanged loose
scores, and 15 improvements. Fifty-four distinct prompts had a nonzero
loose effect, exceeding the prespecified information floor of 20.

| Predictor | Loose MSE | Strict MSE |
|---|---:|---:|
| Action-wise training mean | 0.183201 | 0.189468 |
| Action and size metadata | 0.182044 | 0.193092 |
| Metadata plus reference scalars | 0.186136 | 0.198104 |
| B1 plus scalar action probe | 0.191618 | 0.201129 |
| B2 plus full-vocabulary JS | 0.194360 | 0.203793 |

Lower MSE is better. B3 had 6.77% greater loose MSE than the best comparator,
B0. The paired 95% interval for B0 MSE minus B3 MSE was
[-0.021653, -0.002987]. Adding JS to B2 alone increased MSE by 1.43%; that
paired interval includes zero. Under strict scoring B3 was also worse,
with 7.56% greater MSE than the action-wise training mean. The fixed
criterion of at least 5% improvement against every comparator was not met.

These bootstrap intervals use 2000 seed-0 paired prompt resamples and
condition on the fitted fold models. They do not include training or
feature-selection uncertainty and are exploratory, not confirmatory.

## Verification and limits

All 2520 engine gate checks passed across eligible prompts. All 840 saved
response vectors matched the independently pinned official IFEval scorer,
and 420 indexed run/manifest/log files passed content-hash verification.
The evaluator rejects tampered outputs, unsupported early-EOS exclusions,
missing provenance and records beyond the 120th accepted case. It also
supports identical content relocated between Orion and the Mac. An independent
reviewer reproduced all MSE and paired bootstrap results from saved OOF
rows, checked all 480 signed targets against raw scores, and verified
120 raw run hashes, folds and exact collection stopping.

Prompt IDs were screened against 321 previously exposed official rows.
Four lexical overlap candidates were excluded before seeded selection.
This establishes the documented local exposure boundary, not absence of
model-training contamination or every possible semantic near duplicate.
The pilot uses the pinned official JSONL, including the documented key2785
prompt correction, rather than the older Arrow text for that row.

The result concerns finite-budget instruction compliance for one model,
one compression family and one decision point. It supplies no controller,
deployment memory saving, or general answer-quality guarantee. The separate
memory diagnostic measures the joint cost of preserving the reference and
acquiring a probe on one engineering prompt, not general deployment cost.

## Evidence

- Frozen design: data/pilot-v1/lock.json and owner-approval.json.
- Collection: results/pilot-v1/checkpoint.json and per-prompt artifacts.
- Official parity: results/pilot-v1-verification/summary.json.
- Fixed evaluation: results/pilot-v1-evaluation.json.
- Independent numerical audit: results/pilot-v1-independent-audit.json.
- Next step: assess whether one delayed-sensitivity measurement is justified
  before approving new collection. No model sweep or positive-result search
  on these now-exposed 120 prompts is approved.
