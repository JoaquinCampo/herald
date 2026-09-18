# Query-located span, fixed understanding audit

The B16 observation failed; return to content identity before another model fit.
Competing explanations are missing needed-content identity, suppression by
streaming-head aggregation, small-sample chance or unstable calibration, literal
NIAH template matching, and distributed layer/head redundancy. The oracle rescue
supports needed content causality, but not a deployable locator or useful scalar.

Run one CPU-only audit at the original B0 boundary, same twelve exposed NIAH
prompts and native .10 masks. No new generation, head discovery, timing changes
or predictor fit. The feature process receives a restricted JSON view containing
only ID, raw prompt, prompt length and native masks. No answers, scores, oracle
spans, continuations or generator metadata may enter the feature function.

Split the final nonempty newline-delimited question from its preceding context.
Split that context with the fixed regex [^.!?\n]+[.!?]*, trim surrounding
whitespace and preserve raw character offsets. Use sklearn TfidfVectorizer with
lowercase word unigrams, English stopwords and L2 normalization, fitted to this
prompt's context sentences only. Transform the question and choose highest
cosine similarity, earliest-position tie break. No template-specific key or
answer pattern. Map the chosen exact span through the cached native Qwen chat
tokenizer offsets, requiring exact tokenization and cached B0 positions.

The only action feature z is the mean fraction of selected token positions
missing from native masks across all layer/KV-head combinations. Keep all
positions and heads, no selection or weighting. Repeat with the next prompt's
question, cyclically in ID order, against the same own context and own masks.
Persist matched and shifted span, positions, similarity, z, per-head fractions,
timings, tokenizer identity and source hashes before joining any outcomes.

After feature freeze, independently audit against the earlier oracle spans and
B0 official signed losses. Require every matched span to cover the full oracle
sentence with at most 64 selected tokens. Report exact span differences. Measure
raw positive-direction AUC only, no fitted predictor: require at least .80 and
at least .15 greater than the shifted-question AUC to support the missing-identity
hypothesis. These are exposed-data understanding gates, not predictive accuracy.
Also report similarity-only and normalized-span-position AUC descriptively to
flag template/position confounding. Similar performance prevents attributing any
separation to action sensitivity and would require explicit matching in a new
population. Retain every failure or ambiguity, no drop, repair or replacement.

A failed locator or failed rank gate closes this fixed observation. A supportive
result only warrants designing one fresh query-visible cohort with distractors,
paraphrases and grouped context/query variants. It does not reopen confirmation,
validate a signed-loss predictor, or imply semantic/query-agnostic generality.
