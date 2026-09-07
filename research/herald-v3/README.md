# HERALD v3

Research toward adjusting KV-cache compression during generation to reduce
memory use while keeping task-quality loss within a chosen tolerance.

The agreed target is expected signed final task-quality loss from a
candidate compression action. A first engineering slice uses IFEval,
Qwen2.5-7B-Instruct, and direct Knorm eviction to validate paired runs.
The proposed full-distribution probe is a hypothesis, not a validated
predictor. See [the current plan](docs/research/research-plan.md) and
[acceptance slice](docs/research/acceptance-slice.md).

Start with [the research brief](docs/research-brief.md). Engineering rules
live in [AGENTS.md](AGENTS.md); optional prior evidence is indexed in
[prior-work.md](docs/prior-work.md), and portable skills in
[skills.md](docs/skills.md).

## Local setup

Use Python 3.12 or newer and `uv sync`. Run `uv run poe check` for
formatting verification, lint, strict typing, and tests. Run `uv run poe fmt`
to apply formatting. The initial checks verify only package importability;
they are not research evidence.

Engineering checks currently reuse a verified read-only v2 runtime with
explicit v3 import paths. The isolated Orion directory is
`/clustergpu/home/jcampo/herald-v3`; the verified runtime and model snapshot
are recorded in [engineering-environment.json](docs/research/engineering-environment.json).
There is no remote Git repository. GPU acceptance has not yet passed.
