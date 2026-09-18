# Understanding after035failed

All96newstudy outcomes are now development-exposed. Independent discovery and
evaluation audits verified288official scores, raw features, stored predictions,
fit coefficients and failed gates. Do not reuse either split as confirmation.

First eight evaluation outputs reveal correct queried keys with numeric copying
errors. Across82positive losses,42have an emitted digit run that is a proper
prefix of the correct value;15additional have a proper subsequence;25other.
These categories are descriptive and do not establish a unique cause. Full
percase records: results/value-head-digit-error-diagnostic.json. This suggests
that which digits survive, or the decoder's ability to continue their sequence,
may matter more than mean missing fraction across the whole seven-token value.

The fixed headmean feature design lost digit identity. Conversely69varying head
columns and only7zero-loss training examples may leave insufficient information
to learn stable head roles. Neither diagnosis authorizes tuning the failed model.
A short functional measurement should be tested before another broad model fit.

Primary literature refresh2026-09-06:
[Compression-Aware Abstention](https://arxiv.org/html/2608.29934v1) trains answer/
abstention behavior using thresholded tight-span survival labels, initially on
prompt-style truncation, and separately studies actual compressed-cache decoding.
Its labels and trained behavioral objective differ from HERALD's paired signed
final-quality loss. Its reported deployment gap cautions against equating retained
text or a survival threshold with actual model performance. It does not establish
that a mask-only predictor estimates HERALD loss.

[Training Transformers for KV Cache Compressibility](https://arxiv.org/abs/2605.05971)
studies compressibility as a property of learned representations and trains the
model to improve compression tolerance. It motivates the distinction between
literal content survival and functional availability, but changing model training
is a different intervention and is not the next scoped prediction experiment.
