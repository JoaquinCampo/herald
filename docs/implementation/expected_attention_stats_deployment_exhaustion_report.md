# ExpectedAttentionStats cache-fork deployment branch exhaustion

Date: 2026-07-11. Mission: `mission.md`. This report closes the prescribed
ratio-0.25 ExpectedAttentionStats direct-cache-fork branch. It does not reject
ExpectedAttentionStats as a general compression method.

## Scope and integrity

The branch followed `expected_attention_stats_live_protocol.md` with:

- Llama-3.1-8B-Instruct on IFEval;
- frozen ExpectedAttentionStats query moments collected outside the held-out
  split;
- compression ratio 0.25 and stride 16;
- a new alarm bundle trained from the complete, provenance-valid sweep;
- exact paired live baselines and retained KV tensor accounting; and
- the five frozen evaluation-only triage prompts.

The complete sweep contains 200 references and 4,923 exact switch cells. Its
root configuration SHA-256 is
`5b9a0938c1669d8b99ff0aef25573496029fddafc35286d3b084c663b1efcfa6`.
The frozen-statistics content digest is
`ca7450b7c388fb8612a32ed954ae72fbd7df810bcba7f7e353d6a13c10a21eaf`.
The alarm bundle binds that digest, the source parquet SHA-256
`6ad3b2f83f3a1272429c751ebb288f018dab47c36a5975e06a86816643bbe6a2`,
and the hybrid-stream SHA-256
`15d61009375c9c834274ce74c4abe0769651b02396d22761b06f1ec6f59ecc77`.

The practical controller modes were:

1. One-shot compression after a committed grace decision.
2. Sustained pruning every 32 decode tokens.
3. Sustained pruning every token, which bounds cache regrowth between pruning
   events.

No threshold, ratio, switch position, or sustain interval was selected from
triage outcomes. Interval 1 was the predefined mechanism bound after interval
32 failed, not a fitted value.

## Five-prompt results

Each candidate used the same prompt-disjoint five-prompt split and 2,000
prompt-cluster bootstrap resamples. The reports use the executable deployment
thresholds, with `--min-pairs 5` only for rejection triage.

| Candidate | Quality upper 95% | Major-damage upper 95% | Slowdown upper 95% | Peak-KV mean | Peak-KV lower 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-shot | 20.0% | 0.0% | 22.0% | -15.4% | -45.3% |
| Sustained every 32 tokens | 0.0% | 0.0% | 27.1% | -1.9% | -41.0% |
| Sustained every token | 0.0% | 0.0% | 49.4% | -5.5% | -42.6% |

One-shot failed quality, slowdown, and positive isolated KV savings. Both
sustained modes preserved quality on these five prompts but failed slowdown
and positive isolated KV savings. More frequent pruning increased latency and
did not make the peak-KV lower confidence bound positive.

The negative memory result is real end-to-end retained KV evidence. It
includes the concurrently held reference and forked grace cache and excludes
allocator peaks, exposure, replay savings, and analytical estimates.

## Verdict and limits

No candidate in this prescribed ratio-0.25 direct-cache-fork branch clears the
five-prompt triage. Therefore none may advance to N at least 30, broader
model/task testing, long-context validation, or a deployment claim. The
binding constraints are end-to-end slowdown and the lower confidence bound of
isolated peak-KV savings; one-shot also has a quality failure.

The raw results remain ignored under `results/`. Their compact manifests,
paired records, reports, frozen bundle, and hashes are listed in
`expected_attention_stats_deployment_evidence_manifest.json` and can be
checked with `scripts/verify_deployment_evidence.py --manifest`.

This closes only the documented ratio-0.25 cache-fork implementation. A new
ratio, controller information source, or model-native cache representation
would require a new prompt-disjoint design and fresh frozen artifacts before
any held-out evaluation.
