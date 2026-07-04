# HERALD Subagent Profiles

These are repo-local prompt profiles for Codex subagents. They are not new
runtime roles. Use them by copying the `model`, `reasoning_effort`,
`agent_type`, and prompt skeleton into `spawn_agent`.

## Defaults

- Prefer `gpt-5.3-codex-spark` with `reasoning_effort: "medium"`.
- Use `gpt-5.4-mini` with `reasoning_effort: "medium"` when Spark needs more
  reliability.
- Use `gpt-5.5` with `reasoning_effort: "medium"` only for truly complex work.
- Use `reasoning_effort: "high"` only when there is a concrete reason.
- Never use `xhigh`.
- Do not assume skills or main-thread context are inherited. Every prompt must
  name required skills, project constraints, files to read first, and return
  shape.

## Spark-First Patterns

Use Spark before stronger models whenever the task can be specified tightly.
The main thread owns synthesis, final judgment, and any irreversible action.

### Parallel Explore

Use when the main thread needs context from several independent places.

Spawn 2 to 5 `herald-codebase-explorer` agents with:

```yaml
agent_type: explorer
model: gpt-5.3-codex-spark
reasoning_effort: medium
```

Each prompt should ask one narrow question and name exact files, directories,
or search terms. Return only evidence and a direct answer. The main thread
synthesizes across results.

### Spark Worker, Mini Checker

Use when an edit is bounded and testable.

1. Spawn `herald-bounded-worker` on Spark for the edit.
2. Spawn or run a `gpt-5.4-mini` checker only after the patch exists, unless
   review can run in parallel on a separate artifact.
3. Main thread integrates and verifies.

Use this shape for test additions, small implementation slices, doc updates,
and mechanical refactors with a disjoint write set.

### Spark Triage Before Escalation

Use when a bug, log, or failed run might be simple.

Ask Spark to classify the failure, extract the concrete evidence, and propose
the next local check. Escalate to `gpt-5.4-mini` only if Spark returns an
ambiguous verdict, conflicting evidence, or a fix that touches risky code.

### Spark Review Pass

Use Spark for cheap first-pass review:

- missing tests
- stale docs
- obvious data leakage
- unchecked assumptions
- style or convention drift
- unverified claims

Escalate to `gpt-5.4-mini` for higher-risk checks and to `gpt-5.5` only for
paper-critical methodology, architecture-setting decisions, or hard audits.

### Orion Read-Only Triage

Use `herald-orion-log-reader` on Spark for logs and process-output summaries.
Spark may recommend the next command, but the main thread owns launches,
process cleanup, smoke checks, and anything that changes hardware state.

## Spark Prompt Rules

- Make the task atomic.
- Include `AGENTS.md` and exact files to read first.
- Name required skills explicitly.
- State whether the task is read-only or has a disjoint write scope.
- Include the expected return shape.
- Include exact verification commands for workers.
- Ask for unknowns instead of guesses.
- Keep synthesis in the main thread.

## Escalation Ladder

1. Use Spark with a tight prompt.
2. If Spark is ambiguous, retry Spark once with narrower scope and clearer
   return shape.
3. Use `gpt-5.4-mini` when the task needs more reliable editing, checking, or
   synthesis.
4. Use `gpt-5.5` only for truly complex work.
5. Use `reasoning_effort: "high"` only when the prompt states why.

Do not use `xhigh`.

## Do Not Delegate

- credentials, tokens, or gated model license actions
- Orion GPU launches, process cleanup, or hardware-state changes
- commits, pushes, or outward-facing repo actions
- dependency changes
- final paper methodology decisions
- architecture-setting decisions
- anything where a wrong answer would change the research direction

## Spawn Cookbook

### One Spark Explorer

Use for one precise read-only question.

```json
{
  "agent_type": "explorer",
  "model": "gpt-5.3-codex-spark",
  "reasoning_effort": "medium",
  "message": "You are a HERALD codebase explorer. Use AGENTS.md. Required skills: none. Read first: AGENTS.md and <exact files>. Constraints: read-only, no em-dashes, no guesses, locate by content. Task: <one specific question>. Return: direct answer, file paths, evidence, unknowns."
}
```

### Three Spark Explorers In Parallel

Use when context can be split into independent questions. Spawn all ready
explorers in the same tool round.

```json
[
  {
    "agent_type": "explorer",
    "model": "gpt-5.3-codex-spark",
    "reasoning_effort": "medium",
    "message": "You are a HERALD codebase explorer. Use AGENTS.md. Required skills: none. Read first: AGENTS.md and docs/goal.md. Constraints: read-only, no em-dashes, no guesses. Task: What does the goal doc define as HERALD's core contribution? Return: answer, evidence, unknowns."
  },
  {
    "agent_type": "explorer",
    "model": "gpt-5.3-codex-spark",
    "reasoning_effort": "medium",
    "message": "You are a HERALD methodology explorer. Use AGENTS.md. Required skills: hazard-survival-modeling if relevant. Read first: AGENTS.md and docs/methodology.md. Constraints: read-only, no em-dashes, no guesses. Task: Identify the current labeling and evaluation assumptions. Return: answer, evidence, unknowns."
  },
  {
    "agent_type": "explorer",
    "model": "gpt-5.3-codex-spark",
    "reasoning_effort": "medium",
    "message": "You are a HERALD implementation explorer. Use AGENTS.md. Required skills: none unless the code imports a skill-relevant library. Read first: AGENTS.md, src/, and tests/. Constraints: read-only, no em-dashes, no guesses. Task: Summarize existing source and test coverage shape. Return: answer, file paths, evidence, unknowns."
  }
]
```

### Spark Worker

Use for a bounded edit with clear file ownership.

```json
{
  "agent_type": "worker",
  "model": "gpt-5.3-codex-spark",
  "reasoning_effort": "medium",
  "message": "You are a HERALD bounded worker. Use AGENTS.md. Required skills: <skills or none>. Read first: AGENTS.md and <owned files>. Write scope: only <paths>. Other agents may be editing the repo, do not revert unrelated changes. Constraints: no em-dashes, use uv run, no local GPU experiments, no from future annotations. Task: <specific edit>. Verify: <exact command>. Return: files changed, summary, verification result, anything not completed."
}
```

### Spark Worker, Mini Checker

Use after a Spark worker returns a patch.

```json
{
  "agent_type": "explorer",
  "model": "gpt-5.4-mini",
  "reasoning_effort": "medium",
  "message": "You are a HERALD checker. Use AGENTS.md. Required skills: <skills or none>. Read first: AGENTS.md and the changed files. Constraints: read-only, no em-dashes, no guesses, locate by content. Task: Review the patch for bugs, missing tests, convention drift, and incomplete verification. Return: findings ordered by severity, evidence, and minimal fixes."
}
```

### HF Or KV Explorer

Use for narrow Transformers, generation, logits, scores, KV-cache, or kvpress
questions.

```json
{
  "agent_type": "explorer",
  "model": "gpt-5.3-codex-spark",
  "reasoning_effort": "medium",
  "message": "You are a HERALD HF and KV-cache explorer. Use AGENTS.md. Required skills: hf-generate-internals, and kvpress if compression or KV-cache pruning is involved. Read first: AGENTS.md, the required skill files, and <exact project files>. Constraints: read-only, no em-dashes, distinguish raw logits from processed scores, no guesses. Task: <specific question>. Return: direct answer, evidence, practical HERALD implication, unknowns."
}
```

### Orion Log Reader

Use for read-only log triage. The main thread owns any action that changes
Orion state.

```json
{
  "agent_type": "explorer",
  "model": "gpt-5.3-codex-spark",
  "reasoning_effort": "medium",
  "message": "You are a HERALD Orion log reader. Use AGENTS.md. Required skills: hpc-python if interpreting GPU performance, CUDA behavior, multiprocessing, or dataloader behavior. Read first: AGENTS.md and <provided logs>. Constraints: read-only, do not launch GPU jobs, do not alter processes, no em-dashes, no guesses. Task: <specific log question>. Return: status summary, errors, evidence lines, recommended next command for the main thread if needed."
}
```

### Methodology Checker

Use only for truly complex or paper-critical judgment.

```json
{
  "agent_type": "explorer",
  "model": "gpt-5.5",
  "reasoning_effort": "medium",
  "message": "You are a HERALD methodology checker. Use AGENTS.md. Required skills: hazard-survival-modeling when reviewing labels, metrics, horizons, censoring, or survival framing; hf-generate-internals and kvpress when reviewing generation signals or compression mechanisms. Read first: AGENTS.md, docs/goal.md, docs/methodology.md, relevant docs/_why files, and <specific files>. Constraints: read-only, no em-dashes, separate facts from recommendations, flag paper-risk issues first. Task: <specific review>. Return: top risks, assumptions, concrete recommendation, evidence."
}
```

## Profiles

- `herald-codebase-explorer.md`: fast read-only repo questions.
- `herald-bounded-worker.md`: well-specified bounded edits.
- `herald-hf-kv-explorer.md`: Transformers, generation, and kvpress questions.
- `herald-validation-checker.md`: labels, metrics, leakage, and validation.
- `herald-methodology-checker.md`: thesis and experiment-design reviews.
- `herald-orion-log-reader.md`: read-only Orion logs and process inspection.
