# EA056 technical result

Accepted2026-09-07 for the frozen tiny CPU scope. All10 maker gates and20 independent audit checks pass. Native EA scoring/topk/gather, source isolation, plain/instrumented boundary equality, pending-token timing and no-op continuation parity were independently exercised.

The31-position cache retains27 positions per layer/head at removal.10, with7936 bytes reduced to6912. All four sink positions remain. The action continues successfully. CPU timings are recorded but do not estimate real-model cost.

Maker: scripts/prove_ea_boundary.py SHA256155c0fb7826a8d60e0e922ed9a44eafb950637160e133c8612b0a4c2ff8d930b.
Evidence: results/ea-boundary-cpu/proof-run-6/0000-tiny-ea-boundary.json SHA25645c8473e5083de2738a6ab9b3405d067f555ea9f2462564a125716784d7e95aa.
Independent audit: results/ea-boundary-independent-audit.json SHA256e5bec489cc14726cc718eda3394fd88e1e531967e316a0cb5feee925be69a8eb.
Fixtures and restored rerun command are under results/ea-boundary-cpu/fixtures/ and preserved on Orion.

This does not establish Qwen7B compatibility, MuSiQue competence, task loss variation or a predictor. Next is058 fixed QA feasibility preparation. No closed NIAH feature family is reopened.
