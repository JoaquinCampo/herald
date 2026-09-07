# Portable engineering skills

This bootstrap copies four technical skills from the v2 project checkout into
the v3 project. Their supporting `references/` files are included so agents
can load detailed guidance only when a task needs it.

| Skill | Provenance | Rationale |
| --- | --- | --- |
| [HuggingFace generation internals](../.agents/skills/hf-generate-internals/SKILL.md) | v2 `.agents/skills/hf-generate-internals/` | Explains generation outputs, logits processing, stopping criteria, chat templates, and KV-cache flow. |
| [kvpress](../.agents/skills/kvpress/SKILL.md) | v2 `.agents/skills/kvpress/` | Describes KV-cache press classes, compression semantics, wrappers, and compatibility constraints. |
| [Pydantic](../.agents/skills/pydantic/SKILL.md) | v2 `.agents/skills/pydantic/` | Provides Pydantic v2 syntax, validation, serialization, and migration guidance. |
| [HPC Python](../.agents/skills/hpc-python/SKILL.md) | v2 `.agents/skills/hpc-python/` | Covers parallel Python, PyTorch DDP, data loading, caching, compilation, benchmarking, and latency hiding. |

These files are portable references. They do not commit v3 to a dependency,
model, Transformers or Pydantic version, KV compressor, compression ratio,
hardware target, or experimental methodology. Each entrypoint asks the agent
to check version-specific examples against the chosen v3 environment before
using them. Historical project-specific claims were removed from the copied
generation references so these skills provide engineering knowledge without
inheriting v2's research choices.

The source provenance is the local v2 checkout at
`/Users/joaquincamponario/Documents/INCO/RESEARCH/herald-v2/.agents/skills/`.
No package installation, model download, dataset, result, or implementation
file was copied.
