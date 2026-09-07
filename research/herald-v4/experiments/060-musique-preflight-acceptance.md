# MuSiQue preflight acceptance

Owner independently inspected both passing audit artifacts before selecting the first real-model row.

Data: results/musique-data-corrected-independent-audit.json SHA2563d6da30de0c1cc5fbf69c3781a0ef5e056fc64eec49d17cd739b7e74a4495b97. All20 checks pass. Original grouping failure is preserved. Full2417-row dev graph has459 components; selected16 represent11. Every pilot manifest field except group_id is unchanged from the original frozen selection.

Runner: results/musique-runner-independent-audit.json SHA2569bad2f247a1c1a197244b969b7b19c8d3cf39af4d9d7b56ad8facbc3116e9e14.

All14 checks pass, including real tiny CPU first/all selection and explicit failed-row preservation. Prompt length32, sharedB0 length31, nativeEA length27: the earlier maker statement32->27 compared the complete prompt to the acted B0, not the actual cache reduction. Actual B0 cache bytes7936->6912. This is engineering acceptance only.

Orion preflight2026-09-07 verified host/user/path, RTX5090,32607MiB total,31459MiB free,0percent utilization. Only GPU process was jcampo's preserved v2keepalivePID2106 using644MiB. The pinned Qwen snapshot exists. Recheck capacity immediately before execution.

Authorized only the first selected058row in results/musique-pilot-v1-first/ using the fixed64-token protocol. Require every real-model state/no-op check and actual official per-branch answerF1. Owner reviews technical evidence before the remaining15. One-row task quality cannot select continuation. Preserve exact failures and diagnose before code changes. No predictor or confirmation claim.

First real row completed: all9 technical gates pass on pinned Qwen7B BF16,2518 prompt tokens and2517 cached B0 tokens. NativeEA retains2265 per layer,144334848->129884160 cache bytes. Every branch returned Douglas Mawson, official answerF1=0 against Chen Zheng. This is one ordinary reference failure, not a technical failure or selection criterion. Owner accepted technical evidence and dispatched the remaining15 unchanged. Evidence is preserved in results/musique-pilot-v1-first/ with exact command, environment and per-branch official scorer records.
