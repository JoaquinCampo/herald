# Fresh head-sensitive value-retention study

The verified033/034value-only rescue supports testing head-specific value
retention. Stop fitting or designing reductions on the exhausted twelve cases.
Competing explanations: specific head retention predicts damage; global amount
or position suffices; the schema adapter confounds task simplicity with damage;
head patterns are unstable and fail on new prompts. Public semantic-head work
motivates head heterogeneity but its Llama/Mistral head lists cannot be assumed
to apply to Qwen. Avoid new Q/K instrumentation for this first predictive test.

Scope is explicitly query-visible, known-schema numeric retrieval, Qwen2.5-7B,
original split-prefill shared B0, native Knorm .10, total128greedy tokens, official
RULER score and signed loss q_ref-q_action. No label clipping or removal of
improvements, reference failures, adapter misses or unscorable outcomes.

Generate48discovery and48locked-evaluation prompts with the pinned official
niah_multikey_1 generator (four distinct keys, one queried numeric value), length
4096, seeds2026090635 and2026090636 respectively. Use the existing bounded public
corpus and cached tokenizer. This is novel project prompt material, not a claim
of novel corpus or absence from model training. Record source/hash provenance and
check no exact prompt overlap with either old v4 manifest.

The grouping unit is the complete underlying generated context and its fact set,
not a queried key or row. Record a parent context group and a separate exact
context hash. Any future query/order/position variant must retain that parent
on one split and training fold. Check cross-split context duplication independent
of seed; keep all collisions in a reported blocked design, do not silently replace.
No variants are added in this first study.

Freeze a narrow prompt-only adapter before reading new outcomes. Split final
nonempty question line from context. Match context facts of the known form
'One of the special magic numbers for KEY is: DIGITS.'; require exactly one fact
whose literal key appears in the question with non-word boundaries. Derive value
character/token positions from that fact through exact Qwen chat offsets. No
answer field or generator metadata enters this function. Missing/ambiguous facts
produce an explicit miss flag and zero value-retention vector, never exclusion.
This adapter can solve the nominal schema task; report that limitation directly.
It is not a general semantic retriever or a generally useful compression policy.

Before paired continuations, compute112features, one fraction of candidate value
positions absent from each layer/KV head's native mask. All heads remain, with no
search, learned head subset, probing cue, timing change or custom intervention.
Structural features shared by every nonconstant comparator: adapter miss flag,
log prompt length, candidate value token count, normalized value midpoint (zero
on miss), and actual global removal fraction. Save native masks for audit and
record adapter/feature timing separately from ordinary compression and generation.

Fit after discovery only. Three baselines: discovery signed-loss mean; structural
Ridge; structural plus mean value-eviction Ridge. Predictor: structural plus all
112head fractions Ridge. For each Ridge use sklearn Pipeline(StandardScaler,
Ridge), GridSearchCV alpha in[1,10,100,1000], four GroupKFold folds and negative
MSE. Transforms are fit inside each fold; refit selected pipeline on discovery.
No feature selection, clipping, additional models or outcome-driven revisions.

Require discovery and evaluation each contain at least6positive and6nonpositive
signed losses. Any unscorable or failed model run blocks interpretation; preserve
it and diagnose the exact failure. Adapter misses remain ordinary model inputs.
Freeze trained model, parameters, code hashes and evaluation predictions before
joining evaluation labels. Prefer running evaluation only after model freeze.

Locked evaluation gates: at least10%lower MSE and32of48prompt wins against EACH
of the three baselines; AUC of predicted signed loss for positive-versus-nonpositive
outcome at least .80. Report raw signed bias, MAE, all outcomes and a paired
prompt bootstrap uncertainty interval. The gate is developmental only; even a
pass still requires a genuinely unseen frozen confirmation study and measured
complete observation cost. Do not generate or open confirmation in this study.

Use mature primitives and the existing state/scorer functions. Collector should
be a thin composition: no new generic runner framework or new attention hooks.
A real CPU fixture and one original exposed real-model replay validate feature
instrumentation without changing outcomes, then collect discovery once. No GPU
on the Mac; recheck Orion ownership/capacity before each remote launch.

Runtime note before collection: local v2 runtime has sklearn1.9.0, Orion has
sklearn1.7.2 (joblib1.5.3 on both). Fit and apply the serialized pipelines on
Orion CPU using its unchanged runtime, avoiding cross-version model loading.
Independent local arithmetic/metric audits may read plain JSON arrays and
exported coefficients. No shared dependency changes are needed.
