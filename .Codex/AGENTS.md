# Learnings

## Learned

- In HERALD, leverage subagents for decomposable work to keep the main thread focused and context lean — discovered 2026-06-29
- In HERALD, subagents should avoid inheriting the parent model by default because it uses too many tokens; prefer cheaper, faster models such as `gpt-5.3-codex-spark` for well-specified tasks, and use stronger models only when genuinely needed — discovered 2026-06-29
- In HERALD, subagent prompts must explicitly name required skills, project rules, files to read first, constraints, and expected return shape because subagents do not inherit loaded skills or main-thread context — discovered 2026-06-29
- In HERALD, subagents should use only `gpt-5.3-codex-spark`, `gpt-5.4-mini`, or `gpt-5.5`; prefer Spark for speed, use mini for cheap reliability, reserve 5.5 for truly complex work, and set explicit low reasoning unless justified — discovered 2026-06-29
- In HERALD, subagents should never use xhigh reasoning; use medium for Spark and gpt-5.5 by default, raising gpt-5.5 to high only when truly needed — discovered 2026-06-29
- HERALD's true online monitor requires per-token logit features from compressed hybrid generation, not only reference generation; without hybrid-stream features the dataset supports switch-risk prediction, not streaming damage forecasting — discovered 2026-06-29
