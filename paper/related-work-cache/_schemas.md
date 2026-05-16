# Schemas

Locked output schemas for all subagents. The orchestrator validates every subagent return against the relevant schema and rejects/respawns on missing required fields.

## paper-reader output

Returned as JSON, written to `paper/related-work-cache/papers/<arxiv_id>.json`.

```json
{
  "arxiv_id": "2602.02888",
  "bib_key": "halt",
  "title": "Hallucination Assessment via Log-Probs as Time Series",
  "year": 2026,
  "venue": "arXiv" | "NeurIPS" | "MLSys" | ...,
  "evidence_quality": "full_pdf" | "abstract_only" | "html_only",
  "source_url": "https://arxiv.org/abs/2602.02888",
  "tier": 1 | 2 | 3,

  "one_line_claim": "<single sentence: what the paper proves or proposes>",

  "method_summary": "<3 sentences max. Inputs, mechanism, outputs.>",

  "key_findings": [
    {
      "finding": "<sentence stating the result>",
      "quote": "<verbatim text from the paper>",
      "page": 7,
      "section": "5.2"
    }
  ],

  "limitations": [
    "<sentence stating a limitation, ideally one the authors acknowledge>"
  ],

  "relation_to_herald": {
    "shared": "<one sentence: what HERALD shares with this paper>",
    "differs_on": [
      "<one differentiation axis, concrete>",
      "<another>",
      "<at most three total>"
    ]
  },

  "use_in_appendix": {
    "section": "<which appendix section this paper belongs to>",
    "weight": "primary" | "secondary" | "footnote",
    "must_cite": true | false
  },

  "notes": "<optional: anything the orchestrator should know that doesn't fit above>"
}
```

### Required fields

`arxiv_id`, `bib_key`, `title`, `year`, `evidence_quality`, `source_url`, `tier`, `one_line_claim`, `method_summary`, `key_findings` (≥1), `relation_to_herald.shared`, `relation_to_herald.differs_on` (≥1), `use_in_appendix.section`, `use_in_appendix.weight`.

### Rules for the paper-reader

- Every quantitative claim in `key_findings` MUST include a verbatim `quote` and a `page` or `section` reference. Numbers without quotes are rejected.
- `method_summary` is hard-capped at 3 sentences. Reject if longer.
- `differs_on` must list concrete axes (e.g., "predicts hallucination on standard inference, not compression-induced failure"), not vague claims ("different goal").
- If only the abstract is accessible, set `evidence_quality: "abstract_only"` and limit `key_findings` to claims actually in the abstract; the orchestrator may downgrade tier or skip.

## section-writer output

Returned as a markdown chunk, written to `paper/related-work-cache/sections/<section_name>.md`.

```markdown
---
section: <name>
papers_used: [bib_key, bib_key, ...]
papers_cited_in_passing: [bib_key, ...]
---

<section markdown body, ~400-800 words>
```

### Rules for the section-writer

- Cannot introduce facts that don't trace back to a paper-reader output. Every numerical claim or specific finding must come from a cached `papers/<arxiv_id>.json`.
- Cite by `bib_key` in markdown, e.g. `[halt]`. The kickoff handles bib_key → \cite{halt} conversion at LaTeX assembly time.
- Cannot read PDFs directly. Reads only from `papers_used` cache files.
- Must use every paper in `papers_used` at least once. If a paper assigned to the section turns out not to fit, return it in a separate `unused_papers` list with one-line reason.

## critic output

Returned as JSON, written to `paper/related-work-cache/_critic-issues.json` (overwritten each run).

```json
{
  "run_id": "mid" | "final",
  "issues": [
    {
      "type": "unsupported_claim" | "missing_must_cite" | "weak_differentiation" | "duplicate_coverage" | "schema_violation",
      "severity": "blocker" | "warning",
      "section": "<section name or 'global'>",
      "location": "<line number, paragraph, or paper bib_key>",
      "description": "<one sentence>",
      "suggested_fix": "<one sentence, concrete>"
    }
  ],
  "summary": {
    "blockers": <int>,
    "warnings": <int>,
    "sections_reviewed": [<list>]
  }
}
```

### Rules for the critic

- Low-aggression mode: flag only concrete, fixable issues. Do NOT flag prose style, sentence-level wording preferences, or stylistic choices that don't affect correctness.
- `unsupported_claim`: a claim in section prose that doesn't appear in any paper-reader output for that section's `papers_used`.
- `missing_must_cite`: a paper marked `must_cite: true` in the bibliography that does not appear in any section.
- `weak_differentiation`: HERALD's `differs_on` for a Tier 1 paper is vague or missing a concrete axis.
- `duplicate_coverage`: the same paper is meaningfully discussed in two sections without justification.
- `schema_violation`: any cached file does not match its schema.

## quote-verifier output

Returned as JSON in the subagent reply. Not cached.

```json
{
  "verified": true | false,
  "found_at": {
    "page": 7,
    "section": "5.2",
    "context": "<surrounding text, ~50 words>"
  } | null,
  "confidence": "exact" | "fuzzy" | "not_found",
  "notes": "<optional>"
}
```

### Rules for the quote-verifier

- `confidence: "exact"` requires verbatim match.
- `confidence: "fuzzy"` requires the same factual claim with minor wording differences.
- `confidence: "not_found"` is the only acceptable answer when the quote does not appear in the source. Do not approximate.

## planning output

Returned as JSON, written to `paper/related-work-cache/_plan.json`.

```json
{
  "appendix_structure": [
    {
      "section_name": "<short name>",
      "title": "<full section title>",
      "job": "<one sentence: what this section does for the appendix>",
      "papers": [
        {"bib_key": "halt", "weight": "primary", "must_cite": true, "tier": 1}
      ],
      "estimated_word_count": 500
    }
  ],
  "tier_assignments": {
    "tier_1": [<bib_keys>],
    "tier_2": [<bib_keys>],
    "tier_3": [<bib_keys>]
  },
  "skipped": [
    {"bib_key": "...", "reason": "..."}
  ],
  "coverage_gaps": [
    {"topic": "<area>", "rationale": "<why this matters for HERALD>", "suggested_papers": [<arxiv_ids if known>]}
  ]
}
```

### Rules for the planning subagent

- Section count: target 5-7 sections, hard cap at 8.
- Every paper in `_bibliography.json` must appear in either `papers` (under some section), `tier_3` only, or `skipped` with a reason.
- Tier 1 list must include the locked must-cites: HALT, Limits of Learned Importance, ASR-KF-EGR, ERGO, DefensiveKV, ForesightKV.
- `coverage_gaps` is advisory only. The orchestrator does not act on it without user approval (discovery is off by default).
