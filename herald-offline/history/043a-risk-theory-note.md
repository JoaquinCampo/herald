# Target boundary during the population retreat

[The risk of KV cache compression](https://arxiv.org/html/2607.01520v1),
Haverbeck et al., July 2026, defines risk as approximation error for a single
softmax attention head under future queries. Its analysis connects attainable
compression error to cache/query geometry and distinguishes query-aware from
query-agnostic compression. This is a primary-source reason to examine which
queries use a cache, but its risk is not expected signed final task-score loss.
The abstract's LongBench experiment does not establish a HERALD predictor.

Owner inference: changing the outcome population may improve measurement, but
cannot turn a bound on attention approximation into a calibrated task-quality
prediction. Keep the outcome, baseline and prospective evaluation requirements.
No new observable or experiment is selected from this paper here.
