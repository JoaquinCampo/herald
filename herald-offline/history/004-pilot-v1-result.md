# Pilot v1: variation exists, partition audit fails on two cases

All 12 development cases were attempted on Orion with Qwen2.5-7B-Instruct,
BF16 SDPA, seed0 and the frozen last-prompt boundary. First-case launch exited0;
remaining11 launch exited1. Ten cases passed all checks. CWE002 and CWE003
failed exact full-prefill versus split-prefix uncompressed token parity.
Their clone/physical/source checks passed; compressed arms were not collected.

On the eight valid retrieval cases, references scored1. At removal.25, six
actions scored0 and two scored1. At.5 and.75, all eight scored0. On the two
valid aggregation cases, signed losses were [.2,0], [.2,-.1] and [.9,.4] for
the three severities. These are exploratory observations on passing cases,
not a clean 12-case assay or prediction evidence. Failed cases remain in scoring.

The first failed aggregation sequence differs at token118. Counting wording
changes without compression even though the stored split cache is not mutated.
The exact diagnostic subsequently reproduced both original sequences: full/full
and split/split repeats were stable, with cross-path divergence at118 and
unchanged source caches. This establishes a stable prefill-path difference in
this case, not a compression effect. See results/prefill-partition-v1/.

Independent review and the owner accept a separately defined v2 protocol:
uncompressed reference and no-op must both start from independent copies
of the actual split-prefix state.
The full-prefill audit remains separate, and this v1 failure remains unchanged.

Evidence: results/pilot-v1-first/, results/pilot-v1-rest/ and their scores.jsonl;
launch logs/exits and frozen-runner.py under results/pilot-v1-launch/.
Runner SHA5b28e27fc6cf13d072b49a1088c22870d8f49dd15398dc2aab32331d9e01dca0.
No predictor was fitted, no confirmation population was opened.
