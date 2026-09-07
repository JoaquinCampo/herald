# HERALD v3 worker state

Milestone: research plan, reuse inventory, and minimal acceptance slice. Completed for owner review on 2026-09-05, with provenance verification and independent advisor review incorporated. This is planning completion, not experimental acceptance.

Owner: `01a072e7-6ae5-7e93-9c8e-7805f17986b1`. Worker: `01a0731c-2669-7db0-b082-8b089eeb8663`. All writes are in this task's `outputs/herald-v3/`; no edits to v2 or the owner's v3 checkout, remote actions, installs, training, GPU launches, or approval requests occurred.

## Deliverables

- `docs/research/research-plan.md`: proposed estimand, task/model/actions, probe comparison, exposure rules, and exploratory decision criteria.
- `docs/research/acceptance-slice.md`: exact boundary, independent-state and no-op parity, probe transparency, scorer validation, and overhead evidence required before prediction work.
- `docs/research/reuse-inventory.md`: reusable components and limits, prior-probe scope, recovered audit interpretation.
- `docs/research/reuse-manifest.json`: precise absolute paths, full SHA-256 values, revisions, source currency checks and all 1,920 verified artifact hash entries.

## Next owner action

Review the concrete proposal and choose whether to integrate it. If accepted, authorize bounded implementation of the eight-prompt engineering slice, pin the actual model/runtime and official scorer revision, and establish the exposure roster before a pilot lock. Owner runs the real GPU acceptance slice. No predictive or deployment claim is ready for acceptance.

The proposed full-distribution feature remains a hypothesis. A one-token probe was already described in v2. Direct source inspection, including the severe frozen collector, establishes that the recent collections compressed at prefill; token32 was an observation, not the intervention. Fresh prompts are not yet identified. If the new measurement fails its fixed comparison, preserve the negative result and require a justified change before another experiment.

## Review and verification

Independent advisor reviewed the concrete drafts and found four issues, now addressed: exact zero-based pending-token boundary, explicit direct live-cache eviction before the probe, the missing provenance manifest, and exclusion of benchmark annotation metadata from primary predictors. It also requested action-specific effect counts, now specified. The advisor judged the remaining proposed horizon, signed score, fixed regression comparison and exploratory uncertainty framing adequate for this milestone after those corrections. No GPU launch was approved or performed.

Artifact verification: 960 entries in each of two historical manifests, zero missing or mismatched. Moderate source provenance matches 13/13 current files; severe differs in 3/9, with matching frozen source copies recorded. Markdown and JSON structural checks are recorded in `docs/research/deliverable-verification.json`. These checks validate this handoff and historical artifact bytes, not scientific or runtime acceptance.

## Recovery

Recovered completed audits via `read_thread`: `01a0730c-933f-76f1-a318-4d4d430e031a` and `01a0730c-bda9-7ac0-91b9-6e51ec3aa70e`, host local. Do not restart these historical audits. The original parent worker remains outside this worker's ownership.
