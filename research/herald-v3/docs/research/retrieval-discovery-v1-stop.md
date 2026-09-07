# Retrieval discovery v1 stopped at instrumentation parity

The frozen run exited with an operational instrumentation failure before head
selection. At least one paired eager/SDPA sentinel produced different token
IDs. The original rule required exact identity and the run stopped. No head
roster, causal rescue result or new signed-quality prediction evidence follows.

The failure chain and source/model identity were independently verified in
`results/retrieval-discovery-stop-review.json`. Result SHA-256:
`20c328b9158d801e1e422ff3f848877f1d15abca8c4de22ea8223474d03cc2e6`.
The failure did not preserve the exact sentinel ID or divergent token sequences,
so its size and cause cannot be diagnosed from these artifacts. The mechanism
remains untested, not falsified. The failed assay will not be rerun with looser
parity, backend or head-selection rules.

One separately scoped instrumentation check is authorized: on the fixed case
`discovery_t1_l1024_d20`, compare native SDPA generation with the identical SDPA
call wrapped by a diagnostic calculation on its exact query/key/mask inputs.
The wrapper returns the original native output, never diagnostic output. This
is a new engineering feasibility protocol with separate source and execution
seal, not a recovered v1 scientific result.

Only two greedy continuations, at most50 tokens each under the existing stop
rule, are allowed. Require exact output identity, independent attention-row
and index validation, and synchronized time/peak-memory measurements under a
300-second total bound. Save both output sequences and the first divergence
on failure. No head discovery, IFEval outcomes, predictor fit or selection
follows automatically. Failure ends this instrumentation path in current scope.
