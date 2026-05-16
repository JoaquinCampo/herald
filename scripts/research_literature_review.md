# HERALD Literature Research: Hazard/Survival Modeling for LLM Catastrophe Prediction

**Date**: 2026-03-23
**Purpose**: Comprehensive literature review for HERALD project — predicting catastrophic failures in KV-cache compressed LLM generation using survival/hazard modeling on per-token logit signals.

---

## 1. Catastrophe Detection in LLM Generation

### 1.1 Known Failure Modes

**Degenerative Repetition / Looping**
- The most studied failure mode. Models become trapped in self-reinforcing attractors, generating near-identical sequences cyclically.
- Causes: exposure bias, likelihood-driven decoding that over-amplifies frequent patterns, duplicated training data, high-inflow dynamics.
- In code generation, 20 distinct repetition patterns have been taxonomized across 19 state-of-the-art code LLMs (ACL 2025).
- Repetition becomes critically problematic in greedy generation where reproducibility must be guaranteed.

**Instruction Amnesia / Selective Forgetting**
- Under KV-cache compression, certain instructions degrade much faster than others, causing the LLM to completely ignore them.
- "Selective amnesia" introduces security vulnerabilities (e.g., system prompt leakage) — Chen et al., "The Pitfalls of KV Cache Compression" (arXiv:2510.00231).
- Instruction degradation is non-uniform: practitioners cannot predict which instructions will survive compression.

**Non-Termination**
- Lower KV cache budgets trigger longer reasoning traces that may never terminate.
- KNorm compression causes the greatest elongation; H2O also produces non-terminating outputs at low budgets.
- Hold Onto That Thought (arXiv:2512.12008) documents this on GSM8K and MATH500.

**Hallucination / Confabulation**
- Sharp "safety cliff" near 90% compression — strongly correlated with a spike in Global Eviction Ratio (GER).
- This is a phase transition in semantic reachability when answer-critical tokens are globally erased.
- Two distinct failure mechanisms: (1) token erasure (critical tokens removed), (2) representational rigidity (tokens survive but routing flexibility collapses).

### 1.2 Logit-Space Signals Predictive of Failure

**Entropy and Perplexity**
- Entropy-based inference scaling detects hallucinations via: elevated entropy across consecutive tokens during fabrication, entropy spikes at transition points where factual knowledge ends and confabulation begins, inconsistent entropy between related factual statements.
- Degenerated text exhibits surprisingly LOW standalone perplexity — a known trap. Perplexity alone is insufficient.
- Sentence-level perplexity statistics are diagnostic: LLM text has uniformly low perplexity (low CoV), human text varies.

**Surprisal and Its Dynamics**
- First-order differences (delta_S_t = S(x_t) - S(x_{t-1})) capture "stylistic volatility" — abrupt topic or tone changes.
- Mean and variance of delta_S_t quantify magnitude and variability of surprisal shifts.
- Distributional measures (skewness, kurtosis) of token-level unpredictability add further signal.

**Top-k Probability Mass**
- Top-k probability concentration indicates model confidence; collapse of probability mass into fewer tokens precedes repetition.
- Token frequency neurons boost/suppress logits proportionally to token frequency — the model defaults to unigram distribution under high uncertainty.

**Repetition-Specific Signals**
- Repetition neurons: small set of neurons that progressively activate more strongly as repetition continues.
- Two types: intermediate-layer neurons that detect repeating patterns, and uppermost-layer neurons that drive copying.
- Activation increases correlate with repetition onset — detectable signal before full loop engagement.
- Autocorrelation in token ID sequences (SpecRA method) robustly reveals periodicity via FFT.

### 1.3 Key Papers on Detection/Prediction

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| SpecRA (OpenReview) | 2025 | FFT-based autocorrelation for repetition detection in LLM agents; taxonomy from 1.13M agent traces |
| Repetition Neurons (NAACL 2025) | 2025 | Identified specific neurons responsible for repetition; progressive activation patterns |
| Perplexed (arXiv:2404.06634) | 2024 | Token-level perplexity analysis of where LLMs are confused |
| LOG-AID | 2024 | Zero-shot token-level statistics: mean surprisal, Jensen-Shannon divergence, entropy difference |
| EDT (arXiv:2403.14541) | 2024 | Entropy-based dynamic temperature sampling to prevent degeneration |
| Confidence Regulation Neurons (NeurIPS 2024) | 2024 | Token frequency neurons that modulate output distribution |
| Understanding the Repeat Curse (ACL 2025) | 2025 | Feature-level analysis of repetition from a representation perspective |

---

## 2. KV-Cache Compression and Failure Modes

### 2.1 Compression Methods

**Token Eviction (Attention-Based)**
- **H2O** (NeurIPS 2023): Heavy-Hitter Oracle retains tokens with highest accumulated attention scores. Formulated as dynamic submodular problem. 20% heavy hitters sufficient for most tasks.
- **ScissorHands** (arXiv:2305.17118): Persistence of Importance hypothesis — pivotal tokens that matter at one step will matter in future. Up to 5x memory reduction.
- **SnapKV** (2024): Observation-driven compression using attention patterns from end-of-prompt window. 380x compression on some tasks. Prefill-only by default.
- **StreamingLLM** (ICLR 2024): Retains 4 "attention sink" initial tokens + sliding window of recent tokens. Perplexity spikes 10-100x when sink tokens evicted. Up to 22.2x speedup.

**Quantization**
- **KIVI** (ICML 2024): Asymmetric 2-bit quantization. Keys per-channel, values per-token. 2.6x less peak memory. However, 2-bit often degrades accuracy on long-context reasoning vs. 4-bit which preserves it.

**Layer-Aware**
- **PyramidKV**: Dynamically adjusts KV cache per layer for optimal efficiency-utility balancing.

### 2.2 The PRESS/kvpress Library

- **NVIDIA/kvpress** (GitHub): Primary library for KV cache compression research.
- Implements multiple "presses": ObservedAttentionPress, PyramidKVPress, LagKVPress, KeyDiffPress, NonCausalAttnPress, LeverageScorePress, KVzipPress.
- Each press has a `compression_ratio` attribute measuring cache compression.
- Works with HuggingFace transformers; presses compress during prefilling phase.
- Context manager API: `with press(model): model.generate(...)`.
- Handling Llama 3.1-70B at 1M tokens in fp16 requires up to 330GB without compression.

### 2.3 Compression Ratio Thresholds and Catastrophic Failure

**The 90% Compression Safety Cliff**
- All evaluated architectures exhibit sharp hallucination spike near 90% compression (Ananthanarayanan et al., arXiv:2603.01426, Mar 2026).
- Strongly correlated with Global Eviction Ratio (GER) — measures proportion of answer-critical tokens evicted across all heads.
- This is a genuine phase transition, not gradual degradation.

**Sub-O(n) Methods at 75% (1/4) Compression**
- Sub-O(n) memory methods experience sharp performance drop at 1/4 compression rate.
- O(n) methods with sparse decoding (RetrievalAttention, KIVI) sustain higher performance.

**Reasoning Tasks Are More Sensitive**
- On GSM8K, performance deteriorates significantly below 20% cache budget.
- Accuracy drops from ~0.75 to below 0.5 at extreme compression.
- Low budgets paradoxically produce LONGER reasoning traces — a tradeoff between cache size and inference cost.

**Architecture-Dependent Resilience**
- LLaMA: early-layer consensus, late diversification → different compression profile.
- Qwen: funnel-like behavior with late-stage convergence.
- Different architectures require different compression strategies.

**Non-Uniform Instruction Degradation**
- Performance under compression does not degrade uniformly across instructions.
- Instruction order and KV eviction bias both affect which instructions survive.
- Simple changes to eviction policies can reduce these effects (Chen et al., 2025).

### 2.4 Key Papers

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Physics of KV Cache Compression (arXiv:2603.01426) | 2026 | Safety cliff at 90%, GER metric, two failure modes (erasure + rigidity) |
| Pitfalls of KV Cache Compression (arXiv:2510.00231) | 2025 | Instruction amnesia, selective degradation, system prompt leakage |
| Hold Onto That Thought (arXiv:2512.12008) | 2024 | Reasoning benchmark, H2O/SnapKV-D dominance, non-termination at low budgets |
| KV Cache Compression Benchmark (arXiv:2407.01527) | 2024 | 10+ methods across 7 task categories, comprehensive comparison |
| H2O (NeurIPS 2023) | 2023 | Heavy-hitter oracle, dynamic submodular eviction |
| StreamingLLM (ICLR 2024) | 2024 | Attention sinks, streaming with fixed memory |
| ScissorHands (arXiv:2305.17118) | 2023 | Persistence of importance hypothesis |
| SnapKV (2024) | 2024 | Observation-driven prefill compression |
| KIVI (ICML 2024) | 2024 | 2-bit asymmetric KV quantization |

---

## 3. Sequential Prediction / Early Warning Systems

### 3.1 Token-Level Early Warning

**Entropy-Based Signals**
- Entropy spikes at transition points between factual and hallucinated content are detectable.
- EDT (Entropy-based Dynamic Temperature) adjusts temperature per-token based on entropy: higher temp for uncertain tokens, lower for confident ones.
- Spike entropy (token-level entropy dynamics) can weight contributions for detection.

**Multi-Token Prediction Potential**
- LLMs internally "know" about future tokens — multi-token prediction (MTP) research shows models have latent knowledge about upcoming tokens that can be extracted (arXiv:2507.11851).
- RHO-1 introduces token-level scoring mechanisms for selective training on high-value tokens.

**SpecRA for Online Detection**
- Randomized FFT-based autocorrelation detection.
- Runs on streaming token sequences; peaks in autocorrelation reveal periodicity.
- Robust to minor variations (number increments, spelling changes) that confound exact matching.
- Practical thresholds established from 1.13M agent trace analysis.

### 3.2 Intervention Strategies

**Rollback and Resampling**
- CARE framework (arXiv:2509.06982): detect-rollback-intervene mechanism. When harmful/degenerate content detected in buffer, triggers introspection prompt.
- Practical for agent loops: detect, discard bad generation, re-prompt.

**Dynamic Temperature Adjustment**
- AdapT: dynamically adjusts temperature coefficient per-token.
- Hierarchical RL framework learns temperature policy from LLM internal states (arXiv:2602.13035).
- Simple temperature increases at high-entropy tokens allow exploration; decreases at confident tokens reduce tail noise.

**Decoding Strategy Switching**
- Nucleus sampling (top-p) dynamically adjusts candidate set size per step.
- Contrastive search introduces degeneration penalty during decoding.
- These can be triggered adaptively based on detected signals.

### 3.3 How Far Ahead Can Failures Be Predicted?

**Evidence from Repetition Neurons**
- Repetition neurons show progressively increasing activation BEFORE full loop engagement.
- The activation ramp-up provides a warning window of potentially tens of tokens before the model is fully locked into repetitive generation.

**Evidence from Entropy Dynamics**
- Entropy spikes at transition points precede sustained hallucination.
- First-order surprisal differences (delta_S_t) show detectable regime changes.

**Evidence from Attention Patterns**
- Attention loss (pre- vs post-eviction) correlates with eventual performance degradation.
- Can be tracked during generation as a leading indicator.

**Practical Horizon: H tokens**
- HERALD's target of predicting H tokens ahead is plausible based on:
  - Repetition neuron activation ramp-up (gradual, not sudden)
  - Entropy regime changes (detectable 10-50 tokens before sustained failure)
  - Autocorrelation buildup (requires at least one repetition period to detect)
  - For looping with period P, detectable after ~P tokens of repetition, predictable ~P tokens before full lock-in.

---

## 4. Time-to-Event Modeling for NLP/Sequences

### 4.1 Survival Analysis Fundamentals

**Core Concept**
- Models time to an event of interest. Labels are always positive (time until event). Labels may be censored (event not observed).
- Four censoring types: uncensored [a,a], right-censored [a,+inf), left-censored [0,b], interval-censored [a,b].
- HERALD parallel: sequences that complete successfully without catastrophe are RIGHT-CENSORED.

**Hazard Function**
- h(t) = probability of event at time t, given survival to t.
- In discrete time: h(t_j | x) = P(T = t_j | T >= t_j, x) — conditional probability of failure in interval j.
- Cumulative hazard H(t) = sum of h(t_j) up to t.
- Survival function S(t) = product of (1 - h(t_j)) up to t.

### 4.2 The Discrete-Time Binary Classification Trick

**THE KEY INSIGHT FOR HERALD**:

Discrete-time survival analysis can be reformulated as binary classification on a person-period dataset:

1. **Restructure data**: Each sequence contributes one row per token-step where the sequence is still "alive" (no catastrophe yet).
2. **Binary target**: y_{i,t} = 1 if catastrophe occurs at token t for sequence i, else 0.
3. **Features**: Per-token logit features (entropy, top-k mass, repetition scores) at each time step.
4. **Time indicator**: Include token position t as a feature (or discretized interval).
5. **Any classifier works**: Logistic regression, random forest, XGBoost — all valid.
6. **Prediction**: The classifier's P(y=1|x,t) directly estimates the discrete hazard h(t|x).
7. **Survival curve**: S(t) = product of (1 - h_hat(t_j)) for j=1..t.

**Advantages for HERALD**:
- No proportional hazards assumption needed.
- Naturally handles right-censored data (successful completions).
- Can use powerful ML classifiers (XGBoost) without survival-specific modifications.
- Time-varying features are naturally incorporated (features change each token step).
- Multiple failure types via competing risks (see below).

**Reference**: Berger et al., "Survival prediction models: an introduction to discrete-time modeling" (BMC Medical Research Methodology, 2022). PMC9316420.

### 4.3 Competing Risks for Multiple Failure Types

**HERALD has multiple catastrophe types**: looping, non-termination, instruction amnesia.

**Competing Risks Framework**:
- Cause-specific hazard: h_k(t|x) = P(T=t, event=k | T>=t, x) for each event type k.
- At each token step, estimate probability of EACH failure type separately.
- The cause-specific cumulative incidence function gives probability of specific event type by time t.

**Implementation Options**:
1. **Separate binary classifiers**: One XGBoost per failure type (simplest).
2. **Multi-class person-period**: Target is {0: no event, 1: looping, 2: non-termination, 3: amnesia}.
3. **DeepHit-style**: Joint distribution over (time, event type) — overkill for HERALD.

**DeepHit** (AAAI 2018, Lee et al.):
- Discrete-time deep learning survival with competing risks.
- Parameterizes PMF directly with neural network.
- Combines log-likelihood (right-censored, competing risks) with ranking loss.
- Dynamic-DeepHit extends to longitudinal (time-varying) features via RNN.
- Relevant architecture concept, but XGBoost is simpler and sufficient for HERALD's feature space.

### 4.4 Tree-Based Survival Models

**XGBoost AFT (Accelerated Failure Time)**:
- Native XGBoost support since v1.2.0.
- `objective='survival:aft'`, `eval_metric='aft-nloglik'`.
- Supports right-censored, left-censored, interval-censored, uncensored labels.
- Distribution options: normal, logistic, extreme (Gumbel).
- Implementation via `xgb.DMatrix` with `set_float_info('label_lower_bound', ...)` and `set_float_info('label_upper_bound', ...)`.
- **Limitation**: AFT models time-to-event directly, not the hazard at each step. Less natural for HERALD's per-token prediction.

**Random Survival Forest** (scikit-survival):
- Log-rank splitting criterion for right-censored data.
- Nelson-Aalen estimator for cumulative hazard in terminal nodes.
- Kaplan-Meier for survival functions.
- `predict_survival_function()` and `predict_cumulative_hazard_function()`.
- **Limitation**: Predicts at population level, not per-token conditional hazard.

**RECOMMENDED FOR HERALD: Discrete-time XGBoost binary classification**
- Standard XGBoost binary classifier on person-period dataset.
- Most natural fit: per-token features -> per-token hazard probability.
- No special survival objective needed — the person-period trick handles censoring.
- `objective='binary:logistic'` with person-period data structure.

### 4.5 Evaluation Metrics for Survival Models

- **Concordance index (C-index)**: Discrimination — how well model ranks sequences by failure time.
- **Time-dependent AUC**: AUC at specific time horizons (e.g., "predict failure within next 50 tokens").
- **Brier score**: Calibration — how close predicted probabilities are to actual outcomes.
- **Integrated Brier Score (IBS)**: Brier score integrated over time.
- **Kaplan-Meier calibration plots**: Visual check of predicted vs actual survival curves.

---

## 5. Practical Considerations for HERALD

### 5.1 Feature Engineering (Per-Token)

Based on the literature, the optimal feature set for predicting catastrophe includes:

**Entropy Features**:
- Shannon entropy of logit distribution: H(t) = -sum p_i log p_i
- Delta entropy: H(t) - H(t-1) — captures regime changes
- Rolling mean/std of entropy over window W
- Entropy quantiles over recent window

**Probability Concentration**:
- Top-1 probability (max softmax)
- Top-5, top-10, top-50 cumulative probability mass
- Ratio: top-1 / top-5 (concentration measure)
- Gini coefficient of probability distribution

**Repetition Signals**:
- N-gram repetition rate over recent window (bigrams, trigrams, 4-grams)
- Token-level autocorrelation at various lags
- Maximum autocorrelation peak (SpecRA-inspired)
- Period of maximum autocorrelation

**Surprisal Features**:
- Log probability of selected token: -log p(x_t)
- Delta surprisal: S(t) - S(t-1)
- Rolling mean/std of surprisal

**Positional Features**:
- Token position t (absolute)
- Token position relative to prompt length
- Discretized time interval (for discrete hazard baseline)

**Distribution Shape**:
- Kurtosis of logit distribution
- Skewness of logit distribution
- Number of tokens above various probability thresholds

### 5.2 Data Structure for Training

```
Person-Period Dataset Structure:
=====================================
| seq_id | token_pos | entropy | top1_prob | rep_3gram | ... | event | event_type |
|--------|-----------|---------|-----------|-----------|-----|-------|------------|
| s001   | 0         | 2.31    | 0.45      | 0.00      | ... | 0     | none       |
| s001   | 1         | 2.15    | 0.52      | 0.00      | ... | 0     | none       |
| ...    | ...       | ...     | ...       | ...       | ... | ...   | ...        |
| s001   | 147       | 0.82    | 0.91      | 0.67      | ... | 1     | looping    |
| s002   | 0         | 2.44    | 0.38      | 0.00      | ... | 0     | none       |
| ...    | ...       | ...     | ...       | ...       | ... | ...   | ...        |
| s002   | 511       | 1.95    | 0.48      | 0.01      | ... | 0     | none       |  <- right-censored (success)
```

For a sequence with catastrophe at token 147:
- Rows 0..146: event=0 (survived this step)
- Row 147: event=1, event_type=looping (catastrophe occurred)

For a successful sequence with 512 tokens:
- All rows: event=0 (right-censored — no catastrophe observed)

### 5.3 Training Pipeline

```
1. Run sweep: generate sequences under various compression ratios
2. Extract per-token signals (entropy, top-k, repetition) during generation
3. Detect catastrophes post-hoc (looping, non-termination, instruction amnesia)
4. Build person-period dataset: one row per (sequence, token) pair
5. Train XGBoost binary classifier: predict h(t|x) = P(catastrophe at t | survived to t)
6. For competing risks: either multi-class or separate classifiers per event type
7. Evaluate: C-index, time-dependent AUC, calibration
8. Deploy: at inference time, compute features per token, predict hazard, intervene if h(t) > threshold
```

### 5.4 Handling Variable-Length Sequences

- Person-period format naturally handles variable lengths — each sequence contributes as many rows as its length.
- No padding or truncation needed.
- Short sequences that fail early contribute few rows; long successful sequences contribute many.
- Class imbalance: most token-steps are event=0. Use `scale_pos_weight` in XGBoost or SMOTE.

### 5.5 Handling Multiple Failure Modes

**Option A: Binary (any catastrophe)**
- Simplest. Merge all failure types. event=1 for any catastrophe.
- Good for initial validation.

**Option B: Multi-class competing risks**
- Target: {0: survived, 1: looping, 2: non-termination, 3: instruction_amnesia}
- Use `objective='multi:softprob'` in XGBoost.
- Each class probability estimates cause-specific hazard.

**Option C: Separate binary models**
- One XGBoost per failure type.
- Each sequence is right-censored for failure types it doesn't experience.
- Most flexible, allows different feature importance per failure type.

### 5.6 Prediction Horizon (H tokens ahead)

To predict H tokens ahead:
- At token t, use features from tokens 0..t to predict P(catastrophe in [t+1, t+H]).
- This is a "dynamic prediction" problem.
- Simple approach: predict h(t+1), h(t+2), ..., h(t+H) and compute:
  P(failure within H) = 1 - product of (1 - h(t+j)) for j=1..H.
- More sophisticated: use features from window [t-W..t] as input, predict cumulative hazard over next H steps.

### 5.7 Intervention Decision

When predicted hazard exceeds threshold:
1. **Increase temperature** — allows exploration away from degenerate mode (EDT/AdapT approach).
2. **Rollback and resample** — discard last K tokens, regenerate with different sampling (CARE approach).
3. **Switch decoding strategy** — move from greedy to nucleus sampling.
4. **Reduce compression ratio** — if resources allow, allocate more KV cache.
5. **Early termination** — if failure is imminent and irrecoverable, stop generation and report.

### 5.8 Connection to Change Point Detection

- Catastrophe onset is essentially a change point in the token generation process.
- The hazard model captures this: h(t) should spike near the change point.
- CUSUM-like statistics on running hazard estimates could provide additional online detection.
- The duality between online prediction and anomaly detection (INTEL algorithm) is relevant.

---

## 6. Implementation Roadmap for HERALD

### Phase 1: Binary Hazard Model (Current)
- [x] Extract per-token logit signals during generation (signals.py)
- [x] Detect catastrophes post-hoc (detectors.py)
- [ ] Build person-period dataset from RunResults
- [ ] Train XGBoost binary classifier (h(t) = P(any catastrophe at t))
- [ ] Evaluate with C-index, time-dependent AUC

### Phase 2: Competing Risks
- [ ] Separate classifiers per failure type
- [ ] Multi-class person-period dataset
- [ ] Compare cause-specific hazard profiles across compression methods

### Phase 3: Dynamic Prediction
- [ ] Sliding window features for H-step-ahead prediction
- [ ] Cumulative hazard over prediction horizons
- [ ] Real-time hazard estimation during generation

### Phase 4: Intervention
- [ ] Threshold tuning (precision-recall tradeoff)
- [ ] Temperature adjustment intervention
- [ ] Rollback-resample intervention
- [ ] End-to-end evaluation: does intervention actually reduce catastrophes?

---

## 7. Complete Reference List

### Catastrophe Detection and LLM Failure Modes
1. SpecRA: Monitor Degenerative Repetition in LLM Agents (OpenReview, 2025)
2. Repetition Neurons: How Do Language Models Produce Repetitions? (NAACL 2025, arXiv:2410.13497)
3. Understanding the Repeat Curse in LLMs from a Feature Perspective (ACL 2025)
4. Rethinking Repetition Problems of LLMs in Code Generation (ACL 2025)
5. Confidence Regulation Neurons in Language Models (NeurIPS 2024)
6. Perplexed: Understanding When LLMs are Confused (arXiv:2404.06634)
7. LOG-AID: Logit-Based Statistical Features for AI Text Detection (2024)
8. The Curious Case of Neural Text Degeneration (ICLR 2020, arXiv:1904.09751)
9. Contrastive Search: Generating Human-level Text (HuggingFace Blog)

### KV-Cache Compression
10. Understanding the Physics of KV Cache Compression (arXiv:2603.01426, Mar 2026)
11. The Pitfalls of KV Cache Compression (arXiv:2510.00231, Sep 2025)
12. Hold Onto That Thought: KV Cache Compression on Reasoning (arXiv:2512.12008, Dec 2024)
13. KV Cache Compression Benchmark (arXiv:2407.01527, EMNLP 2024 Findings)
14. H2O: Heavy-Hitter Oracle (NeurIPS 2023, arXiv:2306.14048)
15. Efficient Streaming LLMs with Attention Sinks / StreamingLLM (ICLR 2024, arXiv:2309.17453)
16. ScissorHands: Persistence of Importance (arXiv:2305.17118, 2023)
17. SnapKV: LLM Knows What You are Looking for Before Generation (arXiv:2404.14469, 2024)
18. KIVI: Tuning-Free Asymmetric 2bit Quantization for KV Cache (ICML 2024, arXiv:2402.02750)
19. Can LLMs Maintain Fundamental Abilities under KV Cache Compression? (arXiv:2502.01941, 2025)
20. R-KV: Redundancy-aware KV Cache Compression for Reasoning Models (arXiv:2505.24133, 2025)
21. NVIDIA/kvpress GitHub (https://github.com/NVIDIA/kvpress)

### Survival/Hazard Modeling
22. Survival prediction models: intro to discrete-time modeling (BMC Med Res Methodol, 2022, PMC9316420)
23. Deep learning for survival analysis: a review (AI Review, 2023)
24. An Introduction to Deep Survival Analysis Models (arXiv:2410.01086, 2024)
25. DeepHit: Survival Analysis with Competing Risks (AAAI 2018)
26. Dynamic-DeepHit: Dynamic Survival with Competing Risks (2019)
27. Continuous and discrete-time survival prediction with NNs (Lifetime Data Analysis, 2021)
28. Survival Regression with AFT Model in XGBoost (JCGS, Vol 31 No 4, 2022; arXiv:2006.04920)
29. XGBoost AFT Documentation (xgboost.readthedocs.io)
30. scikit-survival: Time-to-Event Analysis (JMLR, Vol 21, 2020)
31. Random survival forests for competing risks (Biostatistics, 2014)
32. Discrete-Time Survival Analysis with Competing Risks - PyDTS (Tomer Meir)
33. discSurv: Discrete Time Survival Analysis (CRAN R package)
34. Empirical Comparison of Continuous and Discrete-time Representations (PMC, 2021)
35. Complete hazard ranking for right-censored data (PLOS Comp Biol, 2017)

### Intervention and Early Warning
36. CARE: Decoding Time Safety via Rollback and Introspection (arXiv:2509.06982, 2025)
37. EDT: Entropy-based Dynamic Temperature Sampling (arXiv:2403.14541, 2024)
38. Hot or Cold? Adaptive Temperature for Code Generation (arXiv:2309.02772, 2023)
39. Look Inward to Explore Outward: Learning Temperature Policy (arXiv:2602.13035, 2026)
40. Your LLM Knows the Future: Multi-Token Prediction Potential (arXiv:2507.11851, 2025)
41. Partnership on AI: Prioritizing Real-Time Failure Detection in AI Agents (2025)

### Change Point Detection
42. A Survey of Methods for Time Series Change Point Detection (PMC, 2017)
43. Sequential Online Prediction with Outliers and Change Points / INTEL (Neurocomputing, 2020)
44. OML-AD: Online Machine Learning for Anomaly Detection in Time Series (arXiv:2409.09742, 2024)
