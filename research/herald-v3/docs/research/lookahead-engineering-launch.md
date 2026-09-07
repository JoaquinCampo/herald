# Lookahead engineering assay

The owner can run one exposed engineering prompt with the offline model
already cached on Orion:

```bash
uv run python scripts/measure_lookahead.py \
  --model /path/to/local/Qwen2.5-7B-Instruct \
  --prompts data/engineering-prompts.json \
  --prompt-id ifeval_3335 \
  --output results/lookahead-ifeval_3335.json
```

The command builds the exact token-32 boundary, applies one direct Knorm
action to an independent clone, and records at most eight distributions for
output indices 32 through 39. Each later input is the reference greedy token
from the preceding distribution, including when the action branch predicts
EOS. The JSON contains compact scalar probes, exact synthetic input positions,
synchronized timing, peak CUDA allocated bytes, source preservation checks,
explicit step-zero parity against the existing `engine.probe_action`, and
environment/source hashes. Its diagnostic block also compares a fixed
four-token Knorm continuation before and after probe acquisition by token IDs,
termination, and final-cache fingerprint. It contains no score or full answer.

The engineering result is usable only when top-level `passed` is true. Review
the `checks`, `diagnostics`, `timing_seconds`, `memory`, and realized output
indices before considering any separate predictor design. Probe and diagnostic
control timings are recorded separately. The timing totals include source
validation and cleanup. A CUDA peak is allocated memory and excludes reserved
memory, so it is an engineering measurement rather than a deployment claim.
