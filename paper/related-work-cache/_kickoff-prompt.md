# Orchestrator Kickoff Prompt

Paste this into a fresh Claude Code session at `/Users/joaquincamponario/Documents/INCO/RESEARCH/herald`. Read it end to end before doing anything.

---

## Goal

Produce a from-scratch related-work appendix for the HERALD paper at `paper/appendix-related-work.tex`. Length target: 6-10 pages of LaTeX. Audience: a reader who wants more depth than the paper's intro provides.

You are the orchestrator. You do not read papers directly. You dispatch subagents and stitch their outputs.

## What HERALD is, in one paragraph

HERALD predicts compression-induced catastrophic failures (looping, non-termination) in KV-cache compressed LLM generation from per-token logit features, using a lightweight XGBoost predictor with a configurable hazard horizon for pre-onset prediction. It is positioned as a runtime-intervention substrate. The first paper covers measurement methodology (matched-prefix replay across compressors and ratios) and predictability (logit features → XGBoost → catastrophe forecast). The ultimate goal is closed-loop control of KV-cache compression aggressiveness from runtime quality signals, but that is later phases, not this paper.

When dispatching paper-readers, pass exactly this paragraph as `herald_one_paragraph`.

## Inputs already prepared

All in `paper/related-work-cache/`:

- `_bibliography.json` — 79 papers extracted from the SOTA in `research/sota/latex/main.tex`. Source of truth for what gets covered.
- `_pdf-inventory.json` — which papers have local PDFs (only 2: H2O at 2306.14048, NSA at 2502.11089) and which need fetching from arXiv.
- `_schemas.md` — locked schemas for every subagent.
- `papers/` — empty; paper-reader outputs land here.
- `sections/` — empty; section-writer outputs land here.

Subagents in `.claude/agents/`:

- `paper-reader` (haiku) — reads one paper, returns schema-bound JSON.
- `appendix-critic` (sonnet) — adversarial QA, returns issue list.
- `quote-verifier` (haiku) — verifies a single quote against a source.

## Tier 1 must-cite list (locked)

These six papers must appear as primary citations with concrete differentiation prose:

1. HALT — `halt` — 2602.02888 — closest signal/predictor precedent
2. Limits of Learned Importance — `limitslearned` — 2601.14279 — the positioning lever
3. ASR-KF-EGR — referenced as "Entropy-Guided Recovery for KV Cache Compression" — 2512.11221 — closest controller proposal (conceptual)
4. ERGO — `ergo` — 2510.14077 — entropy-monitoring template
5. DefensiveKV — `defensivekv` — 2510.13334 — stability assumption fragility
6. ForesightKV — `foresightkv` — 2602.03203 — +147% loss on low-entropy eviction

## Phases

Run these phases in order. Do not parallelize across phases. Within a phase, parallelize as noted.

### Phase 1 — Plan

1. Dispatch ONE planning subagent (general-purpose, sonnet). Pass it `_bibliography.json`, the HERALD paragraph, the Tier 1 list, and the schema for planning output. Tell it: propose 5-7 appendix sections, assign every paper to a section/tier/skip, and produce the planning JSON per `_schemas.md`.
2. Validate the returned plan against the schema. Required: every paper accounted for, all 6 Tier 1 papers in `tier_1` with at least one section assignment marked `must_cite: true`.
3. Write the plan to `paper/related-work-cache/_plan.json`.
4. Pause and write a one-paragraph summary of the plan to the conversation. Do not ask the user to approve. The plan is locked once written; iteration happens via the critic later.

### Phase 2 — Read papers

1. Build the worklist: every paper in `_plan.json` with `tier ≤ 2` (Tier 1 and Tier 2 only). Tier 3 papers are cited in passing without a paper-reader output; their bib_keys go into `papers_cited_in_passing` lists at section-writing time.
2. Dispatch paper-reader subagents in **parallel batches of 5-10**. Pass each one: arxiv_id, bib_key, title, local_pdf_path (from inventory) or null, tier, the appendix_sections list from the plan, the HERALD paragraph.
3. As each paper-reader returns:
   - Parse the JSON. If parsing fails, respawn the agent with a tightened prompt ("Return ONLY valid JSON. Your previous return failed to parse.").
   - Validate against the paper-reader schema. If a Tier 1 or Tier 2 return is missing required fields or has a quantitative claim without a quote, respawn with the specific issue called out.
   - On `fetch_failed: true`, log it and skip. The critic will flag downstream if the paper was must-cite.
   - Write the validated JSON to `papers/<arxiv_id>.json`.
4. After all paper-readers complete, summarize: how many Tier 1, Tier 2 succeeded; how many failed; how many had abstract-only evidence.

### Phase 3 — Verify Tier 1 quotes

For each Tier 1 paper-reader output:
1. For each `key_findings` entry that has a `quote` and a quantitative claim, dispatch a quote-verifier subagent.
2. If `verified: false`, log the issue and respawn the paper-reader for that paper with instructions: "Your previous output's quote for finding X was not found in the source. Re-read and either provide a verifiable quote or drop the finding."
3. After verification passes for all Tier 1 papers, proceed.

Quote verifiers can run in parallel (one per quote). Cap at 10 concurrent.

### Phase 4 — Mid-run critic pass

After Phase 3 (papers cached, Tier 1 verified, but no sections written yet):
1. Dispatch one critic subagent with `run_id: "mid"`, `sections_to_review: []`, full `must_cite_keys`. The critic will check `schema_violation` and `weak_differentiation` only at this stage (no sections to review yet).
2. Address every `blocker`. Respawn paper-readers as needed. Iterate until blockers = 0.

### Phase 5 — Write sections

1. Build the section worklist from `_plan.json.appendix_structure`.
2. For each section, dispatch a section-writer subagent (general-purpose, sonnet). Pass it:
   - The section's `name`, `title`, `job`, `estimated_word_count` from the plan.
   - The list of `papers_used` (full paths to `papers/<arxiv_id>.json`) and `papers_cited_in_passing` (bib_keys).
   - The HERALD paragraph for context.
   - The schema for section-writer output.
   - An explicit reminder: "You cannot read PDFs or introduce facts not in your `papers_used` cache. Cite by `bib_key`."
3. Sections can run in parallel after Phase 4 completes. Cap at 4 concurrent.
4. Validate each return: papers_used field present, all assigned papers used or in `unused_papers` with reason. Write to `sections/<section_name>.md`.

### Phase 6 — Final critic pass

1. Dispatch one critic subagent with `run_id: "final"`, all sections to review, full `must_cite_keys`.
2. Address every `blocker`. Likely fixes: respawn a section-writer with a corrected `papers_used` list, or insert a missing must-cite paper into a section as a passing mention.
3. Address `warning`s if cheap. Concrete differentiation fixes are usually cheap; rewriting a section for duplicate coverage is not.
4. Re-run the critic if you made non-trivial changes. Two iterations max; if still flagging blockers, escalate to the user.

### Phase 7 — Stitch to LaTeX

1. Concatenate `sections/<name>.md` in the order from `_plan.json.appendix_structure`.
2. Convert markdown citations `[bib_key]` to `\cite{bib_key}`.
3. Convert markdown headers to LaTeX sections.
4. Write to `paper/appendix-related-work.tex`.
5. Generate or update `paper/refs.bib` from `_bibliography.json`. Use `bib_key` as the BibTeX key. Include arxiv URL.
6. Sanity check: every `\cite{key}` in the .tex has a corresponding entry in refs.bib. Run a grep to verify.

### Phase 8 — Hand off

Write a final summary to the conversation:
- Section-by-section paper counts.
- Tier 1 papers and their differentiation summaries (one line each).
- Any unresolved warnings from the critic.
- Any papers that failed fetch and were skipped.
- Path to the assembled appendix.

Stop. Do not push to git. Do not edit the main paper.

## Hard rules

- **Never read a paper directly.** All paper content comes from `papers/<arxiv_id>.json`. If you find yourself wanting to fetch a PDF, dispatch a paper-reader.
- **Never write prose with new claims.** Section-writers stitch from cache; you stitch from sections. The orchestrator never adds substantive content.
- **Always validate subagent returns against the schema** before persisting them. Reject on missing required fields, respawn with a corrective prompt.
- **Run paper-readers in parallel batches**, not serially. Same for section-writers.
- **Cache everything to disk** before moving to the next phase. A session restart should be able to pick up by re-reading `_plan.json`, `papers/*.json`, and `sections/*.md`.
- **Run the critic at least twice** (mid-run and final). Address every blocker.
- **Do not invent papers.** Discovery is off. If a Tier 1 paper isn't fetchable, log it and proceed; the critic will surface it as a blocker and the user will decide.

## PDF unavailability handling

77 of 79 papers will need WebFetch to arXiv. Realistic failure modes:

- **arXiv abstract page works, PDF doesn't**: paper-reader returns with `evidence_quality: "abstract_only"`. Acceptable for Tier 2 and Tier 3. For Tier 1, re-attempt with the alternate URL `https://arxiv.org/pdf/<arxiv_id>.pdf`. If still fails, log and proceed with abstract-only; critic will flag.
- **Both fail**: paper-reader returns `fetch_failed: true`. For Tier 1, this is a blocker; surface to the user with the bib_key. For Tier 2/3, skip silently.
- **Wrong arxiv_id in bibliography**: if a paper-reader can't find anything matching the title at the given ID, it returns a small `{wrong_id: true, suggested_id: "..."}` object. Update `_bibliography.json` and respawn.

## Failure modes you must catch

These are easy to miss without active vigilance:

1. **Schema drift**: a paper-reader returns prose-laden JSON with extra fields. Reject any output that doesn't validate strictly. The schema is the contract.
2. **Hallucinated quotes**: a paper-reader provides a verbatim quote that the verifier can't find. Always run the verifier on Tier 1 quotes; respawn on `verified: false`.
3. **Coverage drift**: by the time you reach Phase 5, your understanding of what each section needs has shifted from the plan. Re-read `_plan.json` before dispatching each section-writer. Do not improvise structure.
4. **Section bloat**: a section-writer pads to hit the word target with filler. The critic should flag `unsupported_claim`s; if it doesn't, re-read the section yourself and check that every paragraph has a citation.
5. **Subagent context pollution**: if you pass a subagent more than its task needs, it may try to do more than its job. Pass minimum context. Each subagent has one job.

## What "done" looks like

- `paper/appendix-related-work.tex` exists and compiles standalone.
- `paper/refs.bib` exists; every `\cite` resolves.
- All 6 Tier 1 papers are primary citations with concrete differentiation prose.
- All Tier 2 papers are cited.
- Critic's final-run output has zero blockers and any remaining warnings are explicitly accepted.
- A summary in the conversation explains where every Tier 1 paper landed and why.

## What "done" does NOT mean

- Pushing to git.
- Editing `paper/main.tex` or any other paper section.
- Updating `gold/` documents.
- Producing a Phase 1 results section. The appendix is literature only.

## When to stop and ask the user

- A Tier 1 paper repeatedly fails to fetch even after retries.
- The critic's final pass flags blockers you can't resolve in two iterations.
- You discover a coverage gap that genuinely requires a new section not in the plan.
- The plan turns out to assign more than ~12 papers to a single section, suggesting the structure is wrong.

In all other cases, push through.
