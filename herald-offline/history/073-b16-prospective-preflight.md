# Prospective preflight and discovery launch

September15. Study071 was frozen before fresh generation. Its candidate is
explicitly seven inputs, six structural fields plus z. A draft fitting script
incorrectly used z alone; owner corrected it before any fit. Independent Luna
review then confirmed collector/fit interfaces, all three baseline gates,
source hashes, split guards and seven-feature scaler/coefficients.

The generator produced32 discovery and48 locked evaluation instances. All80
complete generated contexts are distinct; no exact prompt/context overlap with
536 historical manifest rows in17 files. Official length ranges3479..4096 and
3480..4095 respectively. Single-answer presence and two discovery examples were
inspected. The shared essay corpus is not novel: the claim is fresh randomized
instances conditional on this generator, not document or task transfer.
Data authority: Orion data/b16-replication-v1/{metadata,discovery,evaluation}.json.
Discovery SHA2ae26212a0d6e0d247185b5ba0298182e69b0b388d3f52d1ff7cbf129ffa079e.
Evaluation SHA55667658bdf83c49f44a340ba01c5c4a982062284c06c6fa656b96c2895614fe.

Recovered RULER generation required wonderwords3.0.1 and tenacity9.1.4, installed
from verified PyPI wheels into separate v4 .runtime-deps directories, not the
shared environment. Wheels are preserved under .runtime-wheels. Their SHAs are
4dd66deb6a76ca9e0b0422d1d3e111f9b910d7c16922d42de733ee8def98f8d0 and
6095a360c919085f28c6527de529e76a06ad89b23659fa881ae0649b867a9d55.
Generation uses those directories on PYTHONPATH. NLTK3.9.4 and its existing
punkt_tab resource are used; the old Mac inventory had NLTK3.10.3. This is a
fresh generation under recorded recovered runtime, not byte-identical old data.
PyTorch2.10.0+cu128, Transformers4.57.6, sklearn1.7.2, numpy2.4.6 and scipy1.16.3
remain unchanged. The model checkpoint and tokenizer are the pinned snapshots.

Original collector replay passed072. Prospective collector's first replay failed
before measurement because a timing wrapper's nested tuple was unpacked wrongly.
Failed source is retained locally at results/restart-20260915/failures/
collector-before-boundary-unpack.py, and remote failure record under
results/b16-restart-new-replay-20260915. Owner corrected only unpacking, then
replayed the original case successfully. Native-dtype structural norm semantics
were corrected before that run; a later byte-accounting correction was followed
by another complete replay in results/b16-restart-final-replay-20260915.
Latest replay retains exact z=.08965096899055425 and signed loss1 with all controls.

A separate32-case synthetic signed-target interface check compares serialized
model predictions against independently fitted sklearn Pipelines, including a
constant column. Maximum discrepancy1.11e-16; candidate dimension7. This is a
software check, not research evidence. Artifact: results/b16-replication-interface-audit.
The observed full feature wall time in the exposed replay is about .26s;
approximately388MB of cache features plus3MB of structural summaries transfer
to CPU. Batch costs, prediction cost and raw reconstruction remain to audit.

Fresh discovery runs under results/b16-replication-v1-discovery. Evaluation
outcomes remain ungenerated. Fit only after all32 complete, official scores
and feature reconstruction are checked, and the fixed feasibility gate passes.
