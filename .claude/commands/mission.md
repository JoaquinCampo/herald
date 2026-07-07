---
description: Draft a Goal/Proof/Limits spec from a rough brief, confirm it, then hand it to /goal
argument-hint: <rough one-line description of where we want to get>
---

Draft an autonomous-work goal spec from this brief: $ARGUMENTS

Produce exactly three fields, nothing more:

# Goal
Where we want to get, one or two sentences describing the end state.
Not activities, an outcome.

## Proof
How we know we are there: the command(s) or artifact that demonstrates
it, and what a pass looks like. Point to locked spec files as the
source of truth; never paraphrase their numbers or definitions into
the goal. If no machine-checkable proof exists, say so and propose one
instead of arming a vague goal.

## Limits
Only mission-specific hard lines: what must not change, what must not
be done even if it would make the proof pass. Standing boundaries come
from CLAUDE.md; method comes from PRINCIPLES.md; do not restate them.

Then append this standing boilerplate verbatim:

> Method per PRINCIPLES.md, boundaries per CLAUDE.md. Choose your own
> path, experiments, and tools within the limits. If well-designed
> attempts show the goal is unreachable as stated, reporting that
> evidence with a written analysis IS completing the goal; do not
> grind and do not game the proof. Stop and ask for anything
> irreversible or outward-facing beyond the limits.

Before finishing: check the proof is actually runnable (the commands
exist, the spec files exist) and that the limits do not contradict the
proof. Show the complete spec to the user for approval, ask for any
corrections, and after approval tell them to arm it with:

/goal <the approved spec text>
