# Restart audit: NIAH repeatability survives, CWE is fragile

2026-09-14. Analysis-only result under 066. No GPU generation, predictor fit,
or new confirmation outcomes. Independent scientific review accepted both audits.

## What was reproduced

All 20 study007/010 prompts and 60 nonzero-action outcomes were re-scored with
the pinned RULER scorer. All 22 recorded input hashes, identities, complete action
sets, no-op comparisons, reference scores, signed losses and published statistics
matched. The new script does not call the original diagnostic function.

| Diagnostic | NIAH, 12 prompts | CWE, 8 prompts, descriptive |
| --- | --- | --- |
| Privileged MSE reduction | 50.00% | 53.57% |
| Reduction after deleting each prompt in turn | 37.10% to 63.51% | -2.27% to 79.10% |

NIAH baseline MSE is 0.2396694215, privileged MSE 0.1198347107. There are 7, 8
and 10 failures at removal fractions .05, .10 and .20, with fully nested failure
sets. Exactly 30 of 32,670 unique fixed-margin assignments achieved at least the
observed gain: 30/32670 = 1/1089 = 0.0009182736. Independently, the number of
nested assignments is choose(5,1) times choose(4,2) = 30. The null mean gain is
-50%; the observed 50% is maximal for these margins. Five separate mathematical
checks covered row-order invariance, a four-prompt exact 2/24 case, null mean,
and loss of association when alignment is broken.

This is post hoc association in a selected synthetic sample, conditional on
independent-column exchangeability. It is not a confirmatory p-value, correction
for the research search, proof of transfer, or proof of a unique latent cause.
The statistic uses other-rate outcomes unavailable to a prospective predictor.
The records share one generator seed and task family. Structure and semantics
have not been disentangled by this test.

For CWE, deleting `ruler-ea-dev-v1-cwe-004`, with loss .2 at all three rates,
changes the gain to -2.27%. Do not pool CWE with NIAH or claim robust cross-task
susceptibility. Signed improvements remain included.

The MuSiQue audit reproduced 80 official raw-answer F1 scores from 16 canonical
records, 11 component groups, with source hashes and the corrected reference.
Reference mean F1 is .4215277778; 5/16 meet .8. EA has 16 zero losses; Knorm
has one positive, one negative and 14 zero losses. Its feasibility failure stands.
The first two fixed manifest samples produce incorrect answers unchanged by EA:
`Douglas Mawson` versus `Chen Zheng`, and `Wenzhou` versus `Yongjia County`.
This fixed pilot has no useful EA loss variation; it does not establish general
unpredictability or invalidate QA as a task family.

## Reflection and next decision

H1 from066 survives within NIAH; H2 and single-prompt H3 receive less support
there. H3 matters in CWE. H4 reproduces. The source review in067 supplies no
validated decision-time measurement. Operationally closed studies remain closed,
but their negative results apply to their tested representations and designs.

The next bounded analysis is to reconcile the privileged advantage with the
already frozen structural baseline on the same prompts. Use preserved predictions
if available; otherwise define a separate exploratory comparison before fitting.
This distinguishes action-rate means, simple prompt structure and genuinely
additional susceptibility before choosing a new observation mechanism. No GPU
dataset, compressor, threshold, or learned representation is selected yet.

## Reproduction

For either `scripts/audit_restart_susceptibility.py` or `scripts/audit_restart_musique.py`,
pipe the source to this command on the Mac, changing the input script as needed:

    ssh orion '/clustergpu/home/jcampo/.local/bin/uv run --no-sync --project /clustergpu/home/jcampo/herald-v2 python - --data-root /clustergpu/home/jcampo/herald-v4' < scripts/audit_restart_susceptibility.py

Both real-data runs and the five math checks completed with exit0. Results and
input/scorer hashes are local in `results/restart-20260914/`; artifact hashes
are in that directory's `SHA256SUMS`. Original Orion outputs were unchanged.
The preserved runtime lacks Typer and Ruff; argparse avoids dependency changes,
and no lint pass is claimed. The discarded broad audit draft and execution logs
are retained in the ignored result directory. No GPU job or remote sync occurred.
