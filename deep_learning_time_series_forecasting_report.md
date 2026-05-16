# Deep Learning and Foundation Models for Time Series Forecasting: State of the Art

**Research Report — March 2026**

---

## Table of Contents

1. [Neural Forecasting Architectures](#1-neural-forecasting-architectures)
2. [Transformer-Based Models](#2-transformer-based-models)
3. [The "Are Transformers Effective?" Debate](#3-the-are-transformers-effective-debate)
4. [Foundation Models for Time Series](#4-foundation-models-for-time-series)
5. [Foundation Models vs. Classical Methods](#5-foundation-models-vs-classical-methods)
6. [When to Use Deep Learning for Time Series](#6-when-to-use-deep-learning-for-time-series)
7. [Practical Considerations](#7-practical-considerations)
8. [Recent Developments (2025-2026)](#8-recent-developments-2025-2026)
9. [Summary and Recommendations](#9-summary-and-recommendations)

---

## 1. Neural Forecasting Architectures

### N-BEATS (Neural Basis Expansion Analysis for Time Series)

- **Paper**: Oreshkin et al., ICLR 2020
- **Architecture**: A deep residual model that decomposes time series using learned basis functions. It uses a stack of fully connected layers organized in a doubly residual architecture — residual connections exist both within and between blocks. Each block outputs a backcast (explaining the past) and a forecast (predicting the future). In the interpretable configuration, blocks are constrained to use trend and seasonality basis functions.
- **Key innovation**: Pure deep learning architecture with no time-series-specific components (no recurrence, no convolution), yet achieves interpretable decomposition into trend and seasonality.
- **Performance**: Strong on univariate forecasting; identified as one of the most robust algorithms with four wins across univariate benchmarks in recent 2024-2025 evaluations. Won the M4 competition in ensemble form.
- **When to use**: Univariate forecasting where interpretability of trend/seasonality decomposition matters. Works well with medium-to-large datasets. No exogenous variable support in the original formulation.

### N-HiTS (Neural Hierarchical Interpolation for Time Series)

- **Paper**: Challu et al., AAAI 2023
- **Architecture**: Builds on N-BEATS by adding multi-rate data processing and hierarchical interpolation. Different blocks operate at different temporal resolutions, allowing the model to efficiently capture patterns at multiple time scales. Uses MaxPool layers to downsample inputs at different rates for different blocks, then interpolates outputs back to the target resolution.
- **Key innovation**: Hierarchical interpolation reduces the number of parameters needed for long-horizon forecasting by orders of magnitude compared to N-BEATS, while improving accuracy.
- **Performance**: Consistently beats N-BEATS, especially for long-horizon forecasting. Runner-up in multivariate benchmarks (three wins), behind XGBoost (four wins) in recent evaluations.
- **When to use**: Long-horizon univariate or short multivariate forecasting. Preferred over N-BEATS when computational efficiency matters or when forecast horizons are long.

### TFT (Temporal Fusion Transformer)

- **Paper**: Lim et al., International Journal of Forecasting, 2021
- **Architecture**: A multi-horizon forecasting model that combines several specialized components:
  - **Variable Selection Networks**: Learn which input variables are most relevant at each time step.
  - **Gated Residual Networks (GRN)**: Provide flexible nonlinear processing with skip connections.
  - **Static Enrichment**: Incorporates static metadata (e.g., store ID, product category) into temporal processing.
  - **Temporal Self-Attention**: Multi-head attention with interpretable attention weights over time steps.
  - **Quantile Outputs**: Produces prediction intervals natively.
- **Key innovation**: Handles static covariates, known future inputs (e.g., holidays), and observed past inputs simultaneously — addressing the complexity of real-world forecasting scenarios. Provides interpretable attention weights and variable importance scores.
- **Performance**: Outperforms DeepAR by 36-69% in benchmarks. Strong on multi-horizon forecasting with mixed input types.
- **When to use**: Complex forecasting problems with multiple input types (static metadata, known future events, historical observations). When interpretability of variable importance and temporal attention is valuable. Retail demand forecasting, energy load forecasting, and similar applied domains.

### DeepAR

- **Paper**: Salinas et al., International Journal of Forecasting, 2020 (Amazon)
- **Architecture**: An autoregressive recurrent neural network (LSTM-based) that produces probabilistic forecasts. Instead of point predictions, it outputs parameters of a chosen probability distribution (e.g., Gaussian, negative binomial) at each time step. Training is done by maximizing the likelihood of observed data under the predicted distributions. At inference, samples are drawn autoregressively to produce forecast paths.
- **Key innovation**: First widely adopted deep learning model for probabilistic forecasting at scale. Trains a single model across many related time series (global model), sharing statistical strength.
- **Performance**: Strong for probabilistic forecasting, especially when many related time series are available. Outperformed by TFT on point accuracy benchmarks, but remains competitive for uncertainty quantification.
- **When to use**: When probabilistic forecasts (prediction intervals, quantiles) are the primary requirement. Works best with large collections of related time series (e.g., thousands of product demand series). Good for inventory optimization and supply chain applications where understanding uncertainty is critical.

### WaveNet (adapted for time series)

- **Paper**: Originally van den Oord et al. (DeepMind, 2016) for audio; adapted for time series by Borovykh et al. (2017)
- **Architecture**: Uses stacks of dilated causal convolutions — convolutional filters that skip inputs at exponentially increasing dilation rates (1, 2, 4, 8, ...). This creates an exponentially growing receptive field without an explosion in parameters. Residual connections and gated activations (similar to LSTMs) are used between layers. Conditioning on exogenous variables is done via parallel convolutional filters.
- **Key innovation**: Achieves very large receptive fields efficiently through dilated convolutions. Captures long-range dependencies without recurrence, enabling parallel computation during training.
- **Performance**: Outperforms recurrent models in learning long-term dependencies, with faster training. Competitive in Kaggle competitions (Web Traffic Time Series Forecasting).
- **When to use**: When long-range temporal dependencies are important and training speed matters (no sequential bottleneck like RNNs). Particularly effective for regularly sampled, high-frequency data. Has been largely superseded by transformer-based approaches for most applications.

---

## 2. Transformer-Based Models

### The Evolution of Time Series Transformers

The application of transformers to time series forecasting has gone through several generations, each addressing limitations of the previous:

### Informer (2021)

- **Paper**: Zhou et al., AAAI 2021 (Best Paper)
- **Architecture**: Encoder-decoder transformer with ProbSparse self-attention, which reduces the O(n^2) complexity of vanilla attention to O(n log n) by selecting only the most informative queries. Uses a self-attention distilling mechanism that progressively reduces the sequence length through the encoder. Generative-style decoder produces the entire forecast in one forward pass rather than autoregressively.
- **Key innovation**: ProbSparse attention makes long-sequence forecasting tractable. First to demonstrate transformers could handle very long input sequences for time series.
- **Performance**: Outperformed vanilla Transformers by 15-20% on ETT and Weather datasets.
- **When to use**: Long-sequence time series forecasting where computational efficiency of attention is a concern. Has been largely superseded by later models (Autoformer, PatchTST).

### Autoformer (2021)

- **Paper**: Wu et al., NeurIPS 2021
- **Architecture**: Replaces standard attention with an Auto-Correlation mechanism that operates in the frequency domain to discover period-based dependencies. Incorporates a series decomposition block as an inner operation — at each layer, the model progressively separates trend-cyclical and seasonal components. The encoder focuses on seasonal patterns while the decoder accumulates trend components.
- **Key innovation**: The Auto-Correlation mechanism discovers sub-series level similarities based on periodicity, which is more aligned with time series structure than point-wise attention. Progressive decomposition within the model itself.
- **Performance**: 10-12% improvement over Informer, particularly strong on structured, periodic data.
- **When to use**: Time series with strong periodic/seasonal patterns. The decomposition architecture makes it naturally suited for data with clear trend and seasonality.

### FEDformer (2022)

- **Paper**: Zhou et al., ICML 2022
- **Architecture**: Operates primarily in the frequency domain. Uses a Frequency Enhanced Block (FEB) and Frequency Enhanced Attention (FEA) that perform representation learning using either Fourier or Wavelet basis. A mixture-of-experts strategy combines trend components extracted by moving average kernels with various kernel sizes.
- **Key innovation**: Moving attention computation entirely to the frequency domain, which naturally captures the dominant periodic patterns in time series with linear complexity.
- **Performance**: 14.8% relative MSE reduction over Autoformer on six benchmark datasets; 22.6% improvement in univariate forecasting.
- **When to use**: When the time series has strong frequency-domain structure. Univariate long-term forecasting tasks.

### PatchTST (2023) — The Patching Revolution

- **Paper**: Nie et al., ICLR 2023 ("A Time Series is Worth 64 Words")
- **Architecture**: Two key innovations that fundamentally changed how transformers process time series:
  1. **Patching**: Instead of treating each time step as a token, PatchTST segments the time series into subseries-level patches (e.g., 16 consecutive time steps = 1 patch). These patches become tokens for the transformer. This dramatically reduces the sequence length while retaining local semantic information.
  2. **Channel Independence**: Each channel (variable) is processed independently through the same transformer, sharing weights across all channels. This avoids the curse of dimensionality in multivariate settings.
- **Key innovation**: Patching reduces the number of tokens by a factor equal to the patch length (e.g., 16x reduction), making attention O(L/P)^2 instead of O(L)^2. Each patch captures local temporal patterns, analogous to how word tokens capture local semantic meaning in NLP.
- **Performance**: Overall champion in comparative experiments. Identified as the most robust algorithm for univariate forecasting with six wins across benchmarks. Patching has become the de facto standard — virtually all subsequent models adopt it.
- **When to use**: General-purpose long-term time series forecasting. Strong default choice when channel independence is a reasonable assumption. Excellent when compute or memory is constrained due to the efficiency gains from patching.

### iTransformer (2024)

- **Paper**: Liu et al., ICLR 2024
- **Architecture**: "Inverts" the transformer by treating each entire time series (variable) as a single token, rather than treating time steps or patches as tokens. The attention mechanism then operates across variables rather than across time. This is equivalent to an extreme case of patching where the patch size equals the entire input length. Feed-forward layers capture temporal patterns within each variable.
- **Key innovation**: The inversion — attention across variables, FFN across time — naturally captures multivariate correlations that channel-independent models like PatchTST miss.
- **Performance**: Strong on multivariate datasets where cross-variable relationships matter. Slightly behind PatchTST on univariate tasks but better leverages multivariate information.
- **When to use**: Multivariate forecasting where relationships between variables are important (e.g., sensor networks, financial portfolios). When channel independence is too strong an assumption.

### TSMixer (2023)

- **Paper**: Chen et al., arXiv 2023 (Google)
- **Architecture**: An all-MLP architecture that alternates between time-mixing and feature-mixing MLP layers. Time-mixing layers capture temporal patterns, while feature-mixing layers capture cross-variate relationships. Uses residual connections and normalization throughout.
- **Key innovation**: Demonstrates that simple MLP-based mixing can be competitive with attention-based transformers for time series, at much lower computational cost.
- **Performance**: Slightly better than iTransformer overall in some comparative experiments, but behind PatchTST as the overall champion.
- **When to use**: When computational efficiency is paramount. Good default for multivariate forecasting when transformer overhead is not justified.

### Current Consensus on Transformer Architectures

Patching has been universally adopted as the standard tokenization strategy. PatchTST remains the strongest single-architecture choice for general forecasting. The field has moved toward simpler architectures (channel independence, MLPs) that often match or beat complex attention mechanisms. The most important design decision is often how to tokenize the time series, not which attention variant to use.

---

## 3. The "Are Transformers Effective?" Debate

### The Provocation: Zeng et al. (AAAI 2023)

**Paper**: "Are Transformers Effective for Time Series Forecasting?" — Zeng, Chen, Zhang, Xu

**Core Argument**: The paper made a provocative claim that the self-attention mechanism, while excellent for NLP where it captures semantic correlations between discrete tokens, is fundamentally misaligned with time series forecasting:

1. **Temporal information loss**: Self-attention is permutation-invariant — it treats input tokens as a set, not a sequence. While positional encodings partially address this, the mechanism does not inherently respect temporal ordering.
2. **Continuous vs. discrete**: Time series consist of continuous, ordered values where temporal proximity matters intrinsically. NLP tokens are discrete symbols where positional relationships are learned.

**The DLinear Model**: To demonstrate their point, the authors introduced DLinear — a shockingly simple model:
- A decomposition layer (borrowed from Autoformer) separates trend and remainder.
- Two single-layer linear models map the trend and remainder from input length to forecast length.
- The outputs are summed.

They also introduced NLinear (a single linear layer with a normalization trick) and a basic Linear model.

**Results**: These embarrassingly simple linear models outperformed Informer, Autoformer, FEDformer, and other transformer-based models on the majority of long-term forecasting benchmarks (ETT, Weather, Exchange, Traffic, Electricity, ILI).

### The Response

**Hugging Face / Kashif Rasul et al.**: Published "Yes, Transformers are Effective for Time Series Forecasting" arguing that:
- The DLinear comparison was not apples-to-apples: transformer models used in the paper were undertrained and poorly tuned.
- When properly sized and trained, transformer-based models (including Autoformer and the newer PatchTST) significantly outperform linear baselines.
- The paper compared models of vastly different sizes and training budgets.

### The Current Consensus (2025-2026)

The debate has largely been resolved with the following nuanced understanding:

1. **Simple linear models are a strong baseline**: The community now takes linear baselines seriously. Any new model must demonstrate improvement over DLinear/NLinear to be considered meaningful.

2. **Transformers are effective — when properly designed**: Models like PatchTST, which adopt patching and channel independence, significantly outperform linear models. The key was not adding more complex attention, but rather better tokenization (patching).

3. **Complexity must be justified**: The era of adding attention for attention's sake is over. Models need to demonstrate clear benefits over simpler alternatives. TSMixer showed that MLPs can match transformers.

4. **The real lesson was about tokenization**: PatchTST's success showed that the bottleneck was not the attention mechanism itself, but how time series were tokenized. Point-wise tokenization (one time step = one token) was the problem, not self-attention.

5. **Foundation models have changed the equation**: With the rise of pretrained foundation models (Chronos, TimesFM, etc.), the debate has shifted from "are transformers effective?" to "are pretrained transformers effective?", and the answer is increasingly yes.

---

## 4. Foundation Models for Time Series

### Overview

Time series foundation models (TSFMs) are large pretrained models that can forecast unseen time series without task-specific training (zero-shot) or with minimal adaptation (few-shot). They represent a paradigm shift from training a model per dataset to using a single pretrained model across domains.

### Chronos (Amazon, March 2024) and Chronos-2 (October 2025)

**Chronos (v1)**:
- **Architecture**: Based on the T5 (encoder-decoder) family, with vocabulary size reduced from 32,128 to 4,096. Models range from 8M to 710M parameters.
- **Tokenization**: A novel approach that converts continuous time series into discrete tokens:
  1. Scale the time series by its absolute mean (normalization).
  2. Quantize values into a fixed number of uniformly spaced bins.
  3. Add special tokens (PAD for missing values, EOS for end-of-sequence).
- **Training**: Cross-entropy loss on token prediction (like language modeling). Pretrained on publicly available datasets plus synthetic data generated via Gaussian processes.
- **Performance**: On a 42-dataset benchmark, significantly outperforms other methods on in-distribution data and achieves comparable zero-shot performance on unseen data relative to task-specific models.

**Chronos-Bolt** (mid-2024 update):
- Faster inference variant optimized for production deployment. Processes 300+ forecasts per second on a single GPU.

**Chronos-2** (October 2025):
- **Architecture**: 120M-parameter encoder-only model (inspired by T5 encoder). Major departure from the original encoder-decoder design.
- **Key innovation — Group Attention**: Alternates between time attention (across patches within one series) and group attention (across all series at each patch index). This enables in-context learning from related series.
- **Capabilities**: Supports univariate, multivariate, and covariate-informed forecasting within a single architecture. Native support for all covariate types (past, future, static).
- **Performance**: Over 90% win rate vs. Chronos-Bolt in head-to-head comparisons. Largest gains on covariate-informed tasks.
- **Practical guidance**: Currently considered the most mature option for teams adopting foundation model forecasting. Strong zero-shot performance, active development, good documentation.

### TimesFM (Google, ICML 2024)

- **Architecture**: Decoder-only transformer (GPT-style), 200M parameters. Much smaller than LLMs but still powerful.
  - Time series are divided into patches (groups of contiguous time points) that serve as tokens.
  - A residual MLP block converts each patch into a token embedding with positional encodings.
  - Stacked multi-head causal self-attention layers process the token sequence.
  - Causal masking ensures the model can only attend to past patches.
- **Pretraining**: Trained on 100 billion real-world time points, primarily from Google Trends search interest data and Wikipedia pageviews. Uses a patch masking strategy during training to prevent overfitting and allow flexible context lengths at inference.
- **Performance**: Near state-of-the-art zero-shot performance on unseen datasets. Battle-tested in Google's production environments.
- **Key development (2025)**: In-Context Fine-Tuning (ICF) presented at ICML 2025 transforms TimesFM into a few-shot learner through continued pre-training. This matches the performance of dataset-specific supervised fine-tuning without actually running fine-tuning per dataset.
- **Practical guidance**: Good balance of performance and efficiency. Available via Google Cloud BigQuery for production deployment. Strong choice when you want a well-supported, efficient model.

### Moirai (Salesforce, 2024) and Moirai 2.0 (2025)

**Moirai (v1)**:
- **Architecture**: Masked encoder-based transformer. Key innovations:
  1. **Multiple Patch Size Projection Layers**: Different projection layers for different data frequencies, allowing a single model to capture temporal patterns across frequencies.
  2. **Any-Variate Attention**: An attention mechanism that respects permutation variance between variates while capturing temporal dynamics. Handles any number of input variables without architectural changes.
  3. **Mixture Distribution Output**: Models flexible predictive distributions using a mixture of parametric distributions.
- **Training Data**: LOTSA (Large-scale Open Time Series Archive) — 27 billion observations spanning nine domains.
- **Performance**: Strong zero-shot multivariate forecasting across diverse domains and frequencies.

**Moirai 2.0** (2025):
- Faster, more accurate, fewer parameters than Moirai-Large.
- Ranks #1 by MASE on GIFT-Eval leaderboard among non-leaking models.
- Significant improvement in efficiency — "when less is more."

**Moirai-MoE**: A mixture-of-experts variant for further improved performance.

- **Practical guidance**: Best choice for any-variate (variable number of input channels) zero-shot forecasting. Strong when you need a single model to handle diverse datasets with different numbers of variables.

### Lag-Llama (February 2024)

- **Architecture**: Decoder-only transformer based on the LLaMA architecture:
  - Uses RMSNorm (pre-layer normalization) and RoPE (Rotary Positional Encoding).
  - **Tokenization via lags**: Instead of patching, constructs features using lagged values at multiple scales — daily lag (t-1), weekly lag (t-7), monthly lag (t-30), etc.
  - A linear projection layer maps lag features to the transformer's hidden dimension.
  - A distribution head outputs Student's t-distribution parameters for probabilistic forecasting.
- **Training**: Pretrained on a diverse corpus of time series from multiple domains.
- **Performance**: Strong zero-shot generalization. When fine-tuned on small fractions of new datasets, achieves state-of-the-art performance outperforming prior deep learning approaches.
- **Practical guidance**: Good choice for probabilistic univariate forecasting. The lag-based tokenization is intuitive and handles multiple frequencies naturally. Open-source and relatively lightweight. Strong few-shot performance with minimal fine-tuning data.

### Timer (Tsinghua, ICML 2024)

- **Architecture**: GPT-style decoder-only transformer for time series:
  - Converts forecasting, imputation, and anomaly detection into a unified next-token prediction task.
  - Uses the Single-Series Sequence (S3) format to unify heterogeneous time series.
  - 84M parameters in the base version, pretrained on 260B time points.
- **Key innovation**: Demonstrates that the generative pre-training paradigm (next-token prediction) from LLMs transfers directly to time series, enabling a single model to handle multiple tasks (forecasting, imputation, anomaly detection).
- **Subsequent work — Sundial** (ICML 2025 Oral): Scaled the Timer approach to 1 trillion time points using flow-matching loss instead of discrete tokenization. See Section 8.
- **Practical guidance**: Interesting for multi-task scenarios (need forecasting AND imputation AND anomaly detection from one model). The unified task formulation is elegant but still early in adoption.

### TimeGPT (Nixtla)

- **Architecture**: Encoder-decoder transformer with multiple layers, residual connections, and layer normalization. A final linear layer maps the decoder output to the forecast horizon. Independently trained on time series data (not derived from any LLM).
- **Training**: Over 100 billion data points from diverse domains (retail, electricity, finance, IoT).
- **Access model**: API-only (closed-source). Users upload data and receive forecasts or anomaly detection results via API calls. Minimal code required.
- **Recent development**: TimeGPT-2 announced in private preview (Mini, standard, and Pro tiers) with up to 60% accuracy improvement over TimeGPT-1.
- **Practical guidance**: Best for teams that want a "forecasting-as-a-service" approach without managing infrastructure. The API model means no GPU requirements, but also means data must leave your environment. Not suitable for sensitive data or offline deployment. Closed-source nature limits reproducibility.

### Comparison Table: Foundation Models

| Model | Params | Architecture | Tokenization | Univariate | Multivariate | Covariates | Probabilistic | Open Source |
|-------|--------|-------------|-------------|-----------|-------------|-----------|--------------|------------|
| Chronos-2 | 120M | Encoder-only (T5) | Quantization bins | Yes | Yes | Yes (all types) | Yes (quantile) | Yes |
| TimesFM | 200M | Decoder-only (GPT) | Patches | Yes | Limited | Limited | Yes | Yes |
| Moirai 2.0 | Variable | Masked encoder | Multi-scale patches | Yes | Yes (any-variate) | Yes | Yes (mixture) | Yes |
| Lag-Llama | ~50M | Decoder-only (LLaMA) | Lag features | Yes | No | No | Yes (Student-t) | Yes |
| Timer | 84M | Decoder-only (GPT) | S3 patches | Yes | Limited | Limited | Yes | Yes |
| TimeGPT-2 | Unknown | Encoder-decoder | Proprietary | Yes | Yes | Yes | Yes | No (API) |
| Sundial | 128M | Decoder-only | Flow-matching (continuous) | Yes | Limited | Limited | Yes (flow) | Yes |
| Time-MoE | 2.4B (1B active) | Decoder-only + MoE | Patches | Yes | Limited | Limited | Yes | Yes |

---

## 5. Foundation Models vs. Classical Methods

### Benchmark Evidence

**GIFT-Eval Benchmark** (Salesforce, 2024-2026):
- 23 datasets, 144,000+ time series, 177 million data points, 7 domains, 10 frequencies, 97 configurations.
- Current top performers (zero-shot): Moirai 2.0 (#1 by MASE), FlowState (#2), Chronos-2, Chronos-Bolt.
- Foundation models generally outperform both statistical and deep learning models across most domains.

**Frequency-Dependent Performance** (critical nuance):
- **High-frequency (seconds, sub-second)**: Statistical models lead. Foundation models struggle with noisy high-frequency patterns.
- **Medium-frequency (minutes, hours)**: Deep learning models dominate.
- **Low-frequency (daily to yearly)**: Foundation models consistently outperform all other approaches.

**Specific Comparisons**:
- Chronos vs. XGBoost/LightGBM in hospitality sales forecasting: Chronos nearly matches XGBoost and LightGBM in RMSE, with the massive advantage of zero-shot inference and minimal feature engineering.
- Chronos vs. tuned statistical models: Consistently beats tuned statistical models out of the box.
- In electricity load forecasting: Chronos-Bolt and TimesFM lead, with TimesFM 2.0 occasionally winning on specific datasets.

**Financial Markets** (cautionary note):
- Foundation models show limited advantage in financial time series. Fine-tuning TSFMs on financial data does not fully close the performance gap relative to ensemble models (CatBoost, XGBoost, LightGBM). Financial time series have fundamentally different statistical properties (non-stationarity, regime changes, low signal-to-noise ratio) that challenge the pretraining assumptions.

### M-Competition Historical Context

- **M4 (2018)**: Pure ML methods performed poorly. The top 17 methods were mostly statistical combinations. Winner was a hybrid (ES-RNN by Smyl/Uber).
- **M5 (2020)**: Complete reversal — all 50 top methods were ML-based. LightGBM won. However, 92.5% of teams could not beat a simple baseline.
- **Lesson**: ML/DL methods need sufficient data complexity and cross-series information to shine. On simple univariate forecasting with few series, statistical methods remain competitive.

### When Foundation Models Win

1. **Zero-shot scenarios**: No historical data for the specific series; need forecasts immediately.
2. **Many diverse series**: Heterogeneous time series across multiple domains.
3. **Low-to-medium frequency**: Daily, weekly, monthly forecasting.
4. **Limited ML expertise**: Teams without the resources to tune per-series models.
5. **Rapid prototyping**: Need a strong baseline quickly before investing in custom solutions.

### When Classical/ML Methods Win

1. **High-frequency data**: Sub-minute granularity with domain-specific noise patterns.
2. **Financial time series**: Low signal-to-noise, regime changes, non-stationarity.
3. **Single well-understood series**: When you have deep domain knowledge and sufficient history.
4. **Tabular features dominate**: When the forecasting problem is really a tabular regression with engineered features (LightGBM/XGBoost excel here).
5. **Extreme accuracy requirements**: When the last 1-2% of accuracy matters and you can afford per-series tuning.

---

## 6. When to Use Deep Learning for Time Series

### Decision Framework

```
                        Few series (<10)          Many series (100+)
                    ┌───────────────────────┬──────────────────────────┐
Short history       │ Statistical models     │ Foundation models        │
(<100 points/       │ (ETS, ARIMA, Theta)    │ (zero-shot Chronos,     │
 series)            │                        │  TimesFM, Moirai)       │
                    ├───────────────────────┼──────────────────────────┤
Medium history      │ Statistical or         │ Global DL models        │
(100-10K points/    │ LightGBM per-series    │ (N-HiTS, PatchTST, TFT) │
 series)            │                        │ or fine-tuned FMs       │
                    ├───────────────────────┼──────────────────────────┤
Long history        │ Per-series DL or       │ Global DL models        │
(>10K points/       │ LightGBM with          │ (PatchTST, TFT) or     │
 series)            │ feature engineering    │ foundation models        │
                    └───────────────────────┴──────────────────────────┘
```

### Data Requirements for Deep Learning

- **Minimum viable**: ~1,000 time points per series for single-series DL models. With global models (shared across series), can work with shorter per-series histories if you have many series.
- **Comfortable**: 10,000+ time points per series, or 100+ related series with 1,000+ points each.
- **Foundation models change this**: Zero-shot models need NO training data from the target series. Few-shot models need as few as 5-10 examples.

### When Simple Models Suffice

1. **Highly seasonal with clear patterns**: ETS or SARIMA will capture this efficiently.
2. **Short series, few series**: Not enough data to train DL. Use statistical models or foundation model zero-shot.
3. **Stationary processes**: ARIMA family handles these well.
4. **Interpretability is paramount**: Statistical models provide clear, well-understood confidence intervals.
5. **Deployment constraints**: No GPU, strict latency requirements, embedded systems.

### When Deep Learning Adds Value

1. **Complex nonlinear patterns**: Interactions between variables, regime changes, nonlinear trends.
2. **Multiple input types**: Static metadata + time-varying covariates + known future events.
3. **Cross-series learning**: Patterns shared across hundreds or thousands of related series.
4. **Long-range dependencies**: Dependencies spanning hundreds or thousands of time steps.
5. **High-dimensional multivariate**: Many interrelated variables (sensor networks, supply chains).

---

## 7. Practical Considerations

### Compute Costs

| Approach | Training | Inference | GPU Required |
|----------|----------|-----------|-------------|
| Statistical (ARIMA, ETS) | Seconds/series | Milliseconds | No |
| LightGBM/XGBoost | Minutes (all series) | Milliseconds | No |
| Per-series DL (N-BEATS) | Minutes-hours | Milliseconds | Recommended |
| Global DL (PatchTST, TFT) | Hours | Milliseconds | Yes |
| Foundation model (zero-shot) | None (pretrained) | Milliseconds-seconds | Recommended* |
| Foundation model (fine-tuning) | Minutes-hours | Milliseconds-seconds | Yes |

*Some foundation models (Chronos-Bolt) are efficient enough for CPU inference; larger models benefit from GPU.

### Zero-Shot vs. Fine-Tuning vs. Training from Scratch

**Zero-Shot** (use pretrained model directly):
- **When**: Quick prototyping, no domain data, many diverse series.
- **Expected performance**: 80-90% of tuned models for most domains. May struggle with highly specialized domains (finance, some scientific data).
- **Cost**: Inference only. No training compute needed.

**Few-Shot / In-Context Fine-Tuning** (emerging approach, 2025):
- **When**: Want to improve on zero-shot without full fine-tuning complexity.
- **How**: TimesFM's In-Context Fine-Tuning (ICF) provides 5-10 examples at inference time. Matches supervised fine-tuning performance.
- **Cost**: Same as inference (no gradient updates).

**Supervised Fine-Tuning**:
- **When**: Domain-specific data available, zero-shot performance insufficient.
- **How**: Fine-tune last layers or full model on target dataset.
- **Expected performance**: 5-15% improvement over zero-shot in most domains. Less improvement in finance.
- **Cost**: Minutes to hours of GPU training.

**Training from Scratch**:
- **When**: Massive proprietary dataset, unique domain, specific architectural requirements.
- **Expected performance**: Can exceed fine-tuned foundation models if sufficient data and compute.
- **Cost**: Hours to days of GPU training. Significant hyperparameter tuning effort.

**Recommendation**: Start with zero-shot evaluation of 2-3 foundation models (Chronos-2, TimesFM, Moirai 2.0). If insufficient, try few-shot/fine-tuning. Only train from scratch if you have a clear reason and sufficient resources.

### Probabilistic Forecasting

Deep learning approaches to uncertainty quantification:

1. **Parametric distributions** (DeepAR, Lag-Llama): Output parameters of chosen distributions (Gaussian, Student-t, negative binomial). Simple but assumes a distribution family.

2. **Quantile regression** (TFT, Chronos-2): Directly predict quantiles (e.g., 10th, 50th, 90th percentiles). Distribution-free but requires choosing which quantiles to predict.

3. **Mixture distributions** (Moirai): Output parameters of a mixture model. More flexible than single distributions.

4. **Flow-matching** (Sundial): Generate samples from a learned continuous distribution via flow matching. Most flexible but computationally more expensive.

5. **Conformal prediction** (post-hoc): Wrap any point forecaster with distribution-free prediction intervals. Guarantees marginal coverage. Recent advances include:
   - Ensemble Conformalized Quantile Regression (EnCQR) for heteroscedastic data.
   - Feature-fitted online conformal prediction (2025) for non-stationary data.
   - Relational conformal prediction for correlated time series using graph deep learning.

**Recommendation**: For most applications, quantile regression (built into Chronos-2 and TFT) provides a good balance of flexibility and simplicity. For strict coverage guarantees, add conformal prediction on top.

---

## 8. Recent Developments (2025-2026)

### Sundial (ICML 2025 Oral)

- **Authors**: Tsinghua (same group as Timer)
- **Key innovation**: TimeFlow Loss based on flow-matching — enables native pre-training on continuous-valued time series without discrete tokenization. Trained on TimeBench (1 trillion time points). Generates multiple probable forecast paths.
- **Performance**: State-of-the-art on both point and probabilistic benchmarks with millisecond inference.
- **Significance**: Eliminates the information loss from discretization (Chronos's quantization bins) by working directly with continuous values.

### FlowState (IBM, NeurIPS 2025)

- **Architecture**: State Space Model (SSM) encoder + functional basis decoder.
- **Key innovation**: Continuous-time modeling that generalizes across sampling rates without retraining. 9.1M parameters — smallest model in GIFT-Eval top 10, outperforming models 20x larger.
- **Significance**: Demonstrates that transformers are not the only viable architecture for TSFMs. SSMs offer better efficiency for certain time series tasks.

### Time-MoE (ICLR 2025 Spotlight)

- **Architecture**: Decoder-only transformer with Mixture-of-Experts. 2.4B total parameters, 1B activated per prediction.
- **Training**: Time-300B dataset (300 billion time points, 9 domains).
- **Key finding**: Validates scaling laws for time series — more data and larger models consistently improve performance. Sparse MoE keeps inference efficient despite massive model size.
- **Significance**: First billion-scale TSFM. Shows that scaling works for time series, though whether the gains justify the cost is debated.

### Reverso (February 2026)

- **Architecture**: Hybrid model using long convolution and linear RNN layers (DeltaNet).
- **Key innovation**: Models as small as 0.2M-2.6M parameters match or outperform transformer-based TSFMs that are 100x larger.
- **Performance**: Outperforms Sundial, Timer-XL, and others on LTSF benchmarks at a fraction of the parameter count.
- **Significance**: Challenges the assumption that large transformers are necessary. Pushes the efficiency Pareto frontier dramatically.

### Chronos-2 (October 2025)

- See Section 4. Major upgrade: encoder-only, group attention, native multivariate + covariate support. 90%+ win rate over predecessor.

### Moirai 2.0 (2025)

- See Section 4. #1 on GIFT-Eval by MASE. Smaller and faster than Moirai-Large.

### TimesFM In-Context Fine-Tuning (ICML 2025)

- Few-shot learning without gradient updates. Matches supervised fine-tuning quality.

### Emerging Research Directions (ICLR 2026 and beyond)

1. **Agentic forecasting**: Combining foundation models with LLM agents that can select models, preprocess data, and interpret results autonomously.
2. **Interpretability**: Understanding what foundation models learn about temporal patterns.
3. **Context-informed prediction**: Incorporating textual context (news, reports) alongside numerical time series.
4. **Efficiency revolution**: Reverso and FlowState show that sub-10M parameter models can compete with 100M+ models.
5. **Multi-task unification**: Single models for forecasting + imputation + anomaly detection + classification.

---

## 9. Summary and Recommendations

### The Landscape in March 2026

The time series forecasting field has undergone a transformation comparable to what happened in NLP with BERT and GPT. Foundation models have emerged as the default starting point for many forecasting tasks, but the picture is nuanced:

### Tier 1: Start Here (Foundation Models for Zero-Shot)

For most new forecasting problems, begin with zero-shot evaluation of:
1. **Chronos-2** — Most mature, best covariate support, strong community.
2. **Moirai 2.0** — Best for any-variate scenarios, #1 on GIFT-Eval.
3. **TimesFM** — Battle-tested at Google scale, good few-shot capabilities.

### Tier 2: Task-Specific Deep Learning

If foundation models underperform or you need specific capabilities:
- **PatchTST** — Best general-purpose transformer for long-term forecasting.
- **TFT** — Best for complex multi-input scenarios with interpretability needs.
- **N-HiTS** — Best for efficient long-horizon univariate forecasting.
- **DeepAR / Lag-Llama** — Best for probabilistic forecasting emphasis.

### Tier 3: Classical / ML Methods

Remain the right choice in specific scenarios:
- **LightGBM/XGBoost** — High-frequency data, tabular features, financial time series.
- **ETS/ARIMA** — Few series, short history, interpretability requirements.
- **Theta/Naive** — Baselines that must always be tested against.

### Key Takeaways

1. **Foundation models are real**: They provide strong zero-shot performance across domains, eliminating weeks of per-dataset tuning for many applications.

2. **But they are not universally superior**: High-frequency data, financial markets, and highly specialized domains still favor task-specific approaches.

3. **Efficiency is the new frontier**: Reverso (0.2M params) and FlowState (9.1M params) challenge the "bigger is better" assumption. Small, well-designed models can match giants.

4. **Patching was the real revolution**: More than any attention variant, the idea of patching time series (treating subsequences as tokens) unlocked transformer effectiveness for time series.

5. **The DLinear lesson endures**: Always compare against simple baselines. If a linear model beats your deep learning model, your model is not adding value.

6. **Probabilistic forecasting is maturing**: From parametric distributions (DeepAR) to quantile regression (Chronos-2) to flow-matching (Sundial), the field has multiple principled approaches to uncertainty quantification.

7. **Few-shot is the sweet spot**: In-context fine-tuning (TimesFM-ICF) and similar approaches provide the best of both worlds — foundation model generality with task-specific adaptation, without the cost of full fine-tuning.

---

## Sources

### Neural Forecasting Architectures
- [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)
- [A Review of ML Time Series Forecasting Models — Nima Sarang](https://nimasarang.com/blog/2025-02-28-time-series-forecasting/)
- [Time-series forecasting in smart manufacturing systems](https://www.sciencedirect.com/science/article/pii/S073658452500064X)
- [Nixtla NeuralForecast](https://github.com/Nixtla/neuralforecast)

### Transformer-Based Models
- [PatchTST: A Time Series is Worth 64 Words (ICLR 2023)](https://arxiv.org/abs/2211.14730)
- [iTransformer: The Latest Breakthrough in Time Series Forecasting](https://www.datasciencewithmarco.com/blog/itransformer-the-latest-breakthrough-in-time-series-forecasting)
- [PatchTST: A Breakthrough in Time Series Forecasting](https://www.datasciencewithmarco.com/blog/patchtst-a-breakthrough-in-time-series-forecasting)
- [Hugging Face PatchTST Blog](https://huggingface.co/blog/patchtst)
- [FEDformer (ICML 2022)](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf)
- [Autoformer (NeurIPS 2021)](https://github.com/thuml/Autoformer)

### The Transformer Effectiveness Debate
- [Are Transformers Effective for Time Series Forecasting? (AAAI 2023)](https://arxiv.org/abs/2205.13504)
- [Yes, Transformers are Effective for Time Series Forecasting (Hugging Face)](https://huggingface.co/blog/autoformer)
- [A systematic review for transformer-based long-term series forecasting](https://link.springer.com/article/10.1007/s10462-024-11044-2)

### Foundation Models
- [Chronos: Learning the Language of Time Series (Amazon)](https://arxiv.org/abs/2403.07815)
- [Introducing Chronos-2: From univariate to universal forecasting](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting)
- [Chronos-2 on Hugging Face](https://huggingface.co/amazon/chronos-2)
- [TimesFM: A decoder-only foundation model for time-series forecasting (Google)](https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/)
- [Time series foundation models can be few-shot learners (Google, ICML 2025)](https://research.google/blog/time-series-foundation-models-can-be-few-shot-learners/)
- [Moirai: A Time Series Foundation Model for Universal Forecasting (Salesforce)](https://www.salesforce.com/blog/moirai/)
- [Introducing Moirai 2.0 (Salesforce)](https://www.salesforce.com/blog/moirai-2-0/)
- [Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting](https://arxiv.org/abs/2310.08278)
- [Timer: Generative Pre-trained Transformers Are Large Time Series Models (ICML 2024)](https://arxiv.org/abs/2402.02368)
- [TimeGPT-1 (Nixtla)](https://www.nixtla.io/docs/introduction/introduction)
- [TimeGPT-2 Announcement (Nixtla)](https://www.nixtla.io/blog/timegpt-2-announcement)

### Benchmarks and Comparisons
- [GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation](https://arxiv.org/abs/2410.10393)
- [GIFT-Eval Leaderboard (Hugging Face)](https://huggingface.co/spaces/Salesforce/GIFT-Eval)
- [The 2026 Time Series Toolkit: 5 Foundation Models](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [Benchmarking Foundation Models for Time-Series Forecasting](https://www.mdpi.com/2813-0324/11/1/32)
- [Makridakis Competitions (Wikipedia)](https://en.wikipedia.org/wiki/Makridakis_Competitions)

### Recent Developments (2025-2026)
- [Sundial: A Family of Highly Capable Time Series Foundation Models (ICML 2025 Oral)](https://arxiv.org/abs/2502.00816)
- [FlowState: IBM's SSM-based TSFM (NeurIPS 2025)](https://research.ibm.com/blog/SSM-time-series-model)
- [Time-MoE: Billion-Scale TSFMs with Mixture of Experts (ICLR 2025 Spotlight)](https://arxiv.org/abs/2409.16040)
- [Reverso: Efficient Time Series Foundation Models (February 2026)](https://arxiv.org/abs/2602.17634)
- [Kronos: Foundation Model for Financial Markets](https://jonathankinlay.com/2026/02/time-series-foundation-models-for-financial-markets-kronos-and-the-rise-of-pre-trained-market-models/)
- [NeurIPS 2025 Workshop: Recent Advances in Time Series Foundation Models](https://neurips.cc/virtual/2025/workshop/109585)

### Practical Considerations
- [Deep Learning for Time Series Forecasting: A Survey (March 2025)](https://arxiv.org/abs/2503.10198)
- [A comprehensive survey of deep learning for time series forecasting (Springer, 2025)](https://link.springer.com/article/10.1007/s10462-025-11223-9)
- [Re(Visiting) Time Series Foundation Models in Finance](https://arxiv.org/html/2511.18578v1)
