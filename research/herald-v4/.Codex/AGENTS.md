## Learned

- This task inherited a HERALD v3 working directory despite being authorized for HERALD v4, so all workers must use explicit HERALD v4 workdirs or absolute paths and must never mirror outputs into HERALD v3, discovered 2026-09-06
- With a physically shortened DynamicCache, Transformers 4.57 causal masking using logical cache_position can expose future tokens during multi-token teacher forcing; use explicit physical causal masks with logical RoPE positions and verify against sequential decoding, evidence: results/digit-copy-mask-hazard-reproduction.json, discovered 2026-09-06
- After reporting a source hash and completed validation, treat file ownership as released and coordinate further edits with the owner because the source may already be frozen, synced, or run, discovered 2026-09-06
- Official RULER generated records can separate answer_prefix from input, so preserve it as native assistant prefill before applying the official new-token horizon; omitting it can consume the full budget regenerating the prefix and create false task-competence failures, discovered 2026-09-07 in HERALD v4 VT049
