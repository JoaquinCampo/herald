# Original collector replay

September15, before fresh collection. Orion RTX5090 had653MiB allocated only to
the preserved keepalive2106. All six reused collector source hashes matched the
local worktree and historical source. The unchanged study029 collector ran
niah_single_2-000 from the exposed EA development population, cached pinnedQwen,
bfloat16, same B16/.10/128 contract.

All state, no-op, instrumentation and physical-mask checks passed. Compared
directly with the old B16 record, z=.08965096899055425, signed loss1, reference
tokens, compressed tokens and native kept-index hash matched exactly.
Evidence on Orion: results/b16-restart-original-replay-20260915/run.json and
restart-comparison.json. No new prompt outcome was collected.

The first command failed before model loading because --engine-root omitted
the final src component. The correct root is
/clustergpu/home/jcampo/herald-v3/src. The corrected original command exited0.
No source, dependency, state or historical output changed to achieve parity.
This verifies the recovered runtime for one exposed case; the prospective
collector still requires its own replay, feature-persistence and baseline audit.
