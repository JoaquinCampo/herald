# HERALD

Predicts compression-induced output damage online, from cheap per-token
logit statistics, before the damage is visible in the generated text.

See `docs/goal.md` (thesis and contributions), `docs/methodology.md`
(design, provisional), `docs/_why/` (rationale per decision), and
`docs/implementation/` (build plans).

## Status

Greenfield. Design docs exist; no source code yet. The methodology is
provisional, and some scope is still open: compression ratios, prompts
per task, and the per-token feature set.

## Conventions

- Python 3.12+. No `from __future__ import annotations`.
- Modern type syntax: `list`, `dict`, `tuple`, `X | None`. Never
  `typing.List`, `typing.Dict`, `Optional[X]`.
- Pydantic models for structured config and data. Keep flat and simple.
- Functions over classes. Classes only when state is genuinely needed.
- Line length 78 (ruff enforces). Imports sorted by ruff.
- Package root under `src/herald/` stays flat. A subpackage is allowed
  only when multiple modules share helpers, form one cohesive concern,
  and flat placement would clutter the root. Default to flat; a single
  module never warrants a subpackage.
- Managed by `uv`; use `uv run` for all commands. Task runner is `poe`
  (poethepoet). `poe check` (format + lint + typecheck + test) is
  required before any commit.
- No em-dashes in any output.

## Working practice

- Test-driven: write the test, watch it fail, implement, watch it pass.
- A passing type checker does not mean the code works. Validate by
  running tests and exercising the real flow.
- For work that decomposes into delegable units, use subagents and keep
  the main thread's context lean (use the `orchestrating-subagents`
  skill early). Relevant skills here: `kvpress`,
  `hf-generate-internals`, `gsm8k-eval`, `hazard-survival-modeling`.
- Never present a guess as fact. If something cannot be observed
  (a server is unreachable, a value is unknown), say so.

## Subagents

- Use Codex model IDs, not Claude aliases such as `haiku`, `sonnet`, or
  `opus`.
- Use only `model: "gpt-5.3-codex-spark"`,
  `model: "gpt-5.4-mini"`, and `model: "gpt-5.5"` for subagents.
- Prefer Spark for speed. Explore subagents and well-specified bounded
  workers use `model: "gpt-5.3-codex-spark"` first.
- Use `model: "gpt-5.4-mini"` when Spark is too shallow or when a cheap
  subagent needs more reliability for edits, checks, or synthesis.
- Use `model: "gpt-5.5"` only for truly complex work, such as
  paper-critical methodology, architecture-setting decisions, or hard
  audits.
- Parent-model inheritance is an exception for genuinely ambiguous,
  paper-critical, or architecture-setting work.
- Set `reasoning_effort` explicitly on subagents. Use `medium` for Spark
  exploration and mechanical tasks, `medium` for `gpt-5.4-mini` edits or
  checks, `medium` for `gpt-5.5` by default, and `high` only when truly
  needed. Never use `xhigh`.
- Do not assume subagents inherit loaded skills or main-thread context.
  Every subagent prompt must explicitly name required skills, project
  constraints, files to read first, and expected return shape.
- Reusable prompt profiles live in `.agents/subagents/`. Use them as
  templates for consistent `agent_type`, `model`, `reasoning_effort`,
  required skills, and return shapes.
- Before spawning a subagent, confirm: task is atomic, model is explicit,
  `reasoning_effort` is explicit, required skills are named, files to read
  first are named, write scope or read-only status is clear, expected return
  shape is clear, and escalation criteria are clear.
- Do not delegate credentials, Orion hardware actions, commits, dependency
  changes, final methodology calls, or architecture-setting decisions.

## Experiment environment

- **GPU:** Orion, `ssh orion`, a single RTX 5090 (~32GB), runs
  directly (not via SLURM). Detailed server facts (driver hazard,
  proxy, downloads) are in this project's memory under `orion-server`.
- **Local Mac** (Apple Silicon, 16GB) is for code, tests, and analysis
  only. Do NOT run GPU experiments locally.
- **No internet on Orion:** installs need the SSH reverse tunnel + a
  CONNECT proxy from the Mac; for running already-cached models use
  `HF_HUB_OFFLINE=1` (no tunnel needed).
- **GPU idle hazard:** the open driver can fail to wake the 5090 from
  auto-suspend during idle gaps and the card "falls off the bus"
  (`nvidia-smi` reports no devices). Mitigate without sudo by keeping a
  trivial matmul alive every ~2s so the GPU never idles. Before any GPU
  job: check `nvidia-smi`, clear stale processes, confirm clean VRAM.
- **Code sync to Orion:** the Orion checkout is a flat `rsync` target,
  NOT a git repo. Use `rsync -av` and NEVER `--delete`. Sync only
  `src/ tests/ scripts/` (and `pyproject.toml` / `uv.lock` when
  changed). Never sync `.venv/ results/ models/ __pycache__/
  .mypy_cache/`. Confirm the Orion project path with the user before
  the first sync.
- **Every `nohup` launch on Orion** gets a 10-second smoke check
  (process alive past argument parsing, log producing output, no
  `Traceback` / proxy / `SYN-SENT`) before any away-check is
  scheduled. Do not report "launched" until it clears.

## Boundaries

**Always safe (no approval):** read any file in this project; edit
`src/` and `tests/`; run `poe check`, pytest, ruff, mypy, `uv run`;
build and test code; set up the tunnel/proxy/keepalive; run experiments
and interpret results.

**Ask first:** changing dependencies once `pyproject.toml` stabilizes;
pushing to any remote; anything touching the user's HF token, gated
model licenses, or other credentials.

**Never:** commit `results/` or `models/` data; add `from __future__
import annotations`; touch anything outside `/clustergpu/home/jcampo/`
on Orion; `sudo` or GPU reset (the user's call); read from, copy from,
or write into any sibling research directory on the Mac (this project
is self-contained).

## Autonomy

Act on reversible, low-risk, in-scope work without asking, and report
the decisions made. Reach out only for irreversible or outward-facing
actions, anything on the user's account / credentials / hardware,
pushing to repos that are not ours, or a genuine fork where a result
would change the paper's direction.
