# EA boundary technical acceptance

Selected by owner 2026-09-07 after054/055. No QA collection or predictor fit.

## Competing explanations

1. Native EA scores can be collected without changing the paired B0 state. Exact plain/instrumented cache and no-op continuation parity should hold.
2. Hook timing or ratio-zero short-circuit prevents valid score collection. Direct comparison with the installed native score path distinguishes this from action failure.
3. Per-head gather or logical-position handling corrupts continuation. Native topk identity, physical-length checks and original logical positions distinguish adapter bugs from legitimate output changes.
4. EA integration is correct but costly. Separate prefill, scoring and gather timing identifies technical overhead, without predicting real-model deployment cost.

## Frozen acceptance slice

Use an existing tiny Qwen model on CPU, float32, with a deterministic prompt and seed recorded before execution. Plain split-prefill B0 excludes the pending final prompt token; act before processing that token. Use independent source/reference/action cache states.

Native ExpectedAttentionPress defaults: n_future_positions512, n_sink4, use_covariance=True, use_vnorm=True, epsilon0. Remove .10 of the boundary cache using native per-KV-head topk. Also execute ratio-zero no-op for parity. No alternative configuration search.

Require every technical check in055: exact plain/instrumented boundary equality, unchanged source, finite deterministic scores, native topk and gathered K/V equality, exact retained counts and physical byte reduction, sink retention, original logical positions, no-op token/termination parity, successful compressed continuation, distinct branch storage, and measured stages. Preserve actual fixtures and outputs under results/ea-boundary-cpu/.

A pass permits owner review of a16-row MuSiQue pilot and an independent adapter audit. It does not prove real-model compatibility, task competence, useful loss variation, or prediction. A failure triggers the smallest original reproduction, not downstream collection. After repeated integration failures, simplify or return to strategy.

Worker owns scripts/prove_ea_boundary.py and results/ea-boundary-cpu/. V3 stays read-only. All meaningful milestones must sync to Orion and the authorized GitHub branch before completion.
