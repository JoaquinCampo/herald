# Time Series Forecasting at Scale: Industry Practices Research Report

**Date**: 2026-03-23
**Scope**: Comprehensive survey of tools, methods, and best practices for time series forecasting in production environments.

---

## Table of Contents

1. [Meta/Facebook Prophet](#1-metafacebook-prophet)
2. [Amazon Forecasting Ecosystem](#2-amazon-forecasting-ecosystem)
3. [Uber Orbit](#3-uber-orbit)
4. [Nixtla Ecosystem](#4-nixtla-ecosystem)
5. [Hierarchical and Grouped Forecasting](#5-hierarchical-and-grouped-forecasting)
6. [Forecasting Pipelines and MLOps](#6-forecasting-pipelines-and-mlops)
7. [Automated Model Selection](#7-automated-model-selection)
8. [Domain-Specific Considerations](#8-domain-specific-considerations)
9. [Time Series Foundation Models](#9-time-series-foundation-models)
10. [Practical Decision Framework](#10-practical-decision-framework)

---

## 1. Meta/Facebook Prophet

### How It Works

Prophet is an additive regression model developed by Meta for business forecasting at scale. It decomposes time series into three main components:

- **Trend**: Piecewise linear or logistic growth curves with automatic changepoint detection
- **Seasonality**: Modeled via Fourier series (daily, weekly, yearly)
- **Holidays/Events**: User-specified irregular events with windowed effects

The model uses a Bayesian framework (Stan backend) to fit parameters and generate uncertainty intervals. It supports both additive and multiplicative seasonality modes.

**Python package**: `prophet`

### When It Is Appropriate

- **Daily data with strong seasonality** (e.g., web traffic, retail sales)
- Business time series with **holiday effects** (Black Friday, national holidays)
- Series with **missing data** or **outliers** (Prophet handles gaps gracefully)
- When the analyst needs **interpretable decomposition** of trend + seasonality
- Rapid prototyping where **speed matters more than maximum accuracy**
- Datasets with **multiple seasonalities** (weekly + yearly)

### Known Limitations and Criticisms

1. **Univariate only**: Cannot natively model multivariate dependencies or exogenous variables beyond simple regressors
2. **Poor with volatile data**: Struggles with abrupt trend shifts, regime changes, or high-frequency volatility
3. **Additive assumption may not hold**: The decomposition assumption (trend + seasonality + holidays) is too rigid for many real-world series
4. **Outperformed by alternatives**: In benchmarks, Prophet consistently underperforms tuned ARIMA, XGBoost, and neural methods. Its use is recommended only when simplicity and speed are the primary requirements
5. **No deep nonlinearity modeling**: Cannot capture complex nonlinear patterns that tree-based or neural models handle
6. **Sub-daily data issues**: Requires careful recalibration of Fourier order and periodicity for high-frequency data
7. **Overfitting risk with default settings**: Hyperparameters should be fixed globally rather than individually grid-searched to guard against overfitting

### Prophet vs. Alternatives

| Criterion | Prophet | ARIMA | XGBoost/LightGBM | LSTM/Transformers |
|-----------|---------|-------|-------------------|-------------------|
| Ease of use | Excellent | Moderate | Moderate | Low |
| Handling seasonality | Automatic | Manual (SARIMA) | Feature engineering | Learned |
| Multiple regressors | Limited | Limited (ARIMAX) | Native | Native |
| Accuracy (benchmarks) | Low-moderate | Moderate | High | High (with data) |
| Interpretability | High | High | Moderate | Low |
| Scalability | Moderate | Low | High | Moderate |
| Setup time | Minutes | Hours | Hours | Days |

### Practical Recommendations

- Use Prophet for **quick baselines** and stakeholder-facing decompositions, not as a production forecasting engine
- Aggregate input data to the lowest meaningful periodicity before fitting
- Always **backtest using expanding-window cross-validation** to estimate expected error distributions
- Communicate forecast error using MAPE, which is business-interpretable
- Add domain knowledge (known events, capacity constraints) through the holidays and regressor interfaces
- Consider **Nixtla's StatsForecast** or **AutoGluon-TimeSeries** for production workloads that need better accuracy

### Key Sources

- [Facebook Prophet Official Documentation](https://facebook.github.io/prophet/)
- [DataCamp Prophet Tutorial](https://www.datacamp.com/tutorial/facebook-prophet)
- [Prophet Interactive Guide - Brenndoerfer](https://mbrenndoerfer.com/writing/prophet-time-series-forecasting-trend-seasonality-holiday-effects)

---

## 2. Amazon Forecasting Ecosystem

Amazon has built a vertically integrated forecasting stack spanning statistical methods, deep learning, AutoML, and foundation models.

### DeepAR

**What it does**: DeepAR is an autoregressive recurrent neural network (RNN) that produces probabilistic forecasts. It trains a single global model across many related time series, learning shared patterns while generating per-series predictions.

**Architecture**: Uses an LSTM/GRU encoder-decoder that outputs parameters of a probability distribution (e.g., negative binomial for count data, Gaussian for continuous) at each time step.

**When to use**:
- Large collections of related time series (thousands of SKUs, hundreds of sensors)
- When probabilistic forecasts (prediction intervals) matter
- Cold-start problems where individual series are short but the collection is large
- Demand forecasting, inventory management, workforce planning

**Limitations**:
- Requires substantial training data across many series
- Training is computationally expensive
- Superseded by newer foundation models (Chronos-2) in many benchmarks

**Python package**: `gluonts` (via `gluonts.model.deepar`)

### AutoGluon-TimeSeries

**What it does**: AutoGluon-TimeSeries is an AutoML framework that ensembles diverse forecasting algorithms automatically. It combines:

- Statistical methods: ETS, ARIMA (via StatsForecast)
- Tree-based: LightGBM, XGBoost
- Deep learning: DeepAR, Temporal Fusion Transformer, PatchTST
- Foundation models: Chronos, Chronos-2

**Key insight**: "Limited training time is better spent training and ensembling many diverse models, rather than hyperparameter-tuning a restricted set of models."

**When to use**:
- When you want a strong baseline with minimal effort
- Production systems that need reliable forecasts across heterogeneous series
- Teams without deep forecasting expertise
- Benchmarking: use AutoGluon's ensemble as the bar to beat

**Practical recommendations**:
- Start with `presets="best_quality"` for benchmarks, `presets="fast_training"` for prototyping
- Use the built-in backtesting (`TimeSeriesPredictor.evaluate()`) before deployment
- AutoGluon handles feature engineering, model selection, and ensembling automatically

**Python package**: `autogluon.timeseries`

### Amazon Chronos / Chronos-2

**What it does**: Chronos is a family of pretrained transformer models (T5 architecture) that treat time series forecasting as a token generation task. Values are tokenized through scaling and quantization, then forecast as a language modeling problem.

**Chronos-2** (October 2025) expanded capabilities:
- **Univariate, multivariate, and covariate-informed** forecasting in a single architecture
- 120M-parameter encoder-only model inspired by T5 encoder
- Multi-step-ahead **quantile forecasts**
- Group attention mechanism for efficient in-context learning
- **300+ forecasts/second** on a single A10G GPU
- Five model sizes: 9M to 710M parameters
- Best performance on fev-bench, GIFT-Eval, and Chronos Bench II among pretrained models

**When to use**:
- Zero-shot forecasting where training data is limited or absent
- Rapid deployment without model training
- As a strong baseline that often beats tuned statistical models out of the box
- Production systems on AWS (native SageMaker and AutoGluon integration)

**Limitations**:
- Context window limits (may not capture very long-range seasonality)
- GPU recommended for optimal throughput (though CPU inference is supported)
- Like all foundation models, may underperform domain-tuned models on highly specialized data

**Python packages**: `chronos-forecasting`, `autogluon.timeseries`

### Key Sources

- [AWS Blog: Chronos-Bolt and AutoGluon](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [AutoGluon-TimeSeries Model Zoo](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-model-zoo.html)
- [Galileo AI: Amazon Chronos Guide](https://galileo.ai/blog/amazon-chronos-ai-time-series-forecasting-guide)

---

## 3. Uber Orbit

### What It Does

Orbit (Object-ORiented BayesIan Time Series) is a Python package for Bayesian time series forecasting and inference. Developed by Uber's Marketing Data Science team, it provides a familiar initialize-fit-predict interface while using probabilistic programming (Stan, Pyro) under the hood.

### Core Models

**Local Global Trend (LGT)**:
- Exponential smoothing with local and global trend components
- Handles both level and growth rate dynamics
- Best for series with complex trend behavior

**Damped Local Trend (DLT)**:
- Simpler variant with damped trend
- Decomposes into: trend + seasonality + regression + error
- More stable for longer-horizon forecasts

Both models support:
- **Regression components** for exogenous variables
- **Seasonality** modeling
- **Full Bayesian inference** with uncertainty quantification

### Causal Impact Estimation

Orbit connects naturally to **Bayesian structural time series (BSTS)** for causal impact estimation:

- Constructs a **counterfactual** prediction of what would have happened absent an intervention
- Infers the **temporal evolution of attributable impact**
- Incorporates **empirical priors** on parameters
- Accommodates **multiple sources of variation**

This makes Orbit particularly valuable for:
- **Marketing mix modeling** (MMM): measuring incrementality and efficiency of marketing channels
- **A/B testing with time series controls**: when randomized experiments are infeasible
- **Policy evaluation**: estimating the effect of a pricing change, feature launch, etc.

### When to Use Orbit

- You need **interpretable, probabilistic** forecasts with full posterior distributions
- **Marketing science** and **media mix modeling** use cases
- Estimating **causal impact** of interventions on business KPIs
- When you want a **Bayesian alternative** to Prophet with better extensibility
- Small-to-medium scale forecasting (not designed for millions of series)

### Limitations

- Slower than frequentist alternatives due to MCMC sampling
- Smaller community and ecosystem compared to Prophet or Nixtla
- Not designed for high-frequency or very large-scale forecasting
- Limited model zoo compared to AutoGluon or Nixtla

**Python package**: `orbit-ml`

### Key Sources

- [Uber Blog: Introducing Orbit](https://www.uber.com/blog/orbit/)
- [Orbit GitHub](https://github.com/uber/orbit)
- [Uber Blog: Orbit v1.1 Release](https://www.uber.com/blog/the-new-version-of-orbit-v1-1-is-released/)

---

## 4. Nixtla Ecosystem

Nixtla has emerged as the most comprehensive open-source forecasting ecosystem, offering specialized libraries for every forecasting paradigm.

### StatsForecast

**What it does**: Lightning-fast implementations of classical statistical forecasting methods, optimized for fitting millions of time series.

**Key models**:
- **AutoARIMA**: Fastest Python implementation, automatic parameter selection via information criteria
- **AutoETS**: Exponential smoothing with automatic model selection
- **AutoCES**: Complex exponential smoothing
- **MSTL**: Multiple seasonal-trend decomposition using LOESS
- **Theta method**: Simple but competitive benchmark

**When to use**:
- As the **first baseline** for any forecasting project
- Massive-scale forecasting (millions of series) where speed matters
- When statistical models are sufficient (many business series)
- Benchmarking more complex methods

**Advantage**: Orders of magnitude faster than `statsmodels` or R implementations. Can fit AutoARIMA on millions of series in minutes.

**Python package**: `statsforecast`

### NeuralForecast

**What it does**: Production-ready implementations of 30+ state-of-the-art neural forecasting models.

**Key models**:
- **N-BEATS**: Neural basis expansion analysis
- **N-HiTS**: Neural hierarchical interpolation
- **PatchTST**: Patch time series transformer
- **TFT**: Temporal Fusion Transformer
- **TimesNet**: Temporal 2D-variation modeling
- **iTransformer**: Inverted transformer

**When to use**:
- Complex patterns that statistical models cannot capture
- Large datasets where deep learning can leverage scale
- When you need attention-based models for interpretability (TFT)
- Multivariate forecasting with exogenous variables

**Python package**: `neuralforecast`

### MLForecast

**What it does**: Machine learning models adapted for time series through automated feature engineering (lags, rolling statistics, date features).

**Key models**: LightGBM, XGBoost, CatBoost, linear models, random forests

**When to use**:
- When tree-based models outperform neural and statistical methods (common in tabular/retail data)
- Large feature sets with exogenous variables
- When interpretability of feature importance matters

**Python package**: `mlforecast`

### HierarchicalForecast

**What it does**: Tools for hierarchical and grouped time series reconciliation.

**Key methods**: Bottom-up, top-down, MinT (minimum trace), ERM (empirical risk minimization), optimal reconciliation

**When to use**: Retail (store/region/national), supply chain, organizational hierarchies

**Python package**: `hierarchicalforecast`

### TimeGPT (Cloud API)

**What it does**: A commercial foundation model API for zero-shot forecasting and anomaly detection.

**Versions**:
- **TimeGPT-1**: Production-ready, trained on 100B+ data points
- **TimeGPT-2 Mini / TimeGPT-2 / TimeGPT-2 Pro**: Enterprise-grade, up to 60% accuracy improvement over v1
- **TimeGPT-2.1**: First multivariate model in the TimeGPT family

**When to use**:
- Rapid prototyping without model training
- When you want a cloud API rather than managing infrastructure
- Anomaly detection alongside forecasting

**Limitations**: Closed-source, requires API access, data leaves your infrastructure

**Python package**: `nixtla` (API client)

### Why This Ecosystem Is Gaining Traction

1. **Unified API**: All libraries share a consistent `fit/predict` interface with `pandas`/`polars` DataFrames
2. **Speed**: StatsForecast is the fastest statistical forecasting library in Python
3. **Breadth**: Covers statistical, ML, neural, hierarchical, and foundation model approaches
4. **Production-ready**: Designed for scale (millions of series)
5. **Active development**: Frequent releases, strong community, excellent documentation
6. **Composable**: Libraries work together (e.g., StatsForecast base forecasts + HierarchicalForecast reconciliation)

### Key Sources

- [Nixtla Homepage](https://www.nixtla.io/)
- [StatsForecast Documentation](https://nixtlaverse.nixtla.io/statsforecast/index.html)
- [NeuralForecast GitHub](https://github.com/Nixtla/neuralforecast)
- [Nixtla Suite Medium Article](https://medium.com/@kyle-t-jones/nixtla-suite-for-time-series-forecasting-with-python-b0f318365e9b)

---

## 5. Hierarchical and Grouped Forecasting

### The Problem

Many real-world forecasting problems have a natural hierarchy: a retailer forecasts at the SKU level, store level, region level, and national level. Independently generated forecasts at different levels will be **incoherent** -- they won't add up. Hierarchical reconciliation ensures coherence while improving accuracy.

### Approaches

#### Bottom-Up
- Forecast at the most granular level and aggregate upward
- **Pro**: Captures detail, no information loss
- **Con**: Noisy at the bottom level, errors compound upward
- **When to use**: When bottom-level data is rich and reliable

#### Top-Down
- Forecast at the aggregate level, then disaggregate using historical proportions
- **Pro**: Stable aggregate forecasts, less noise
- **Con**: Loses bottom-level patterns, proportions may change
- **When to use**: When top-level patterns are strong and bottom-level data is sparse

#### Middle-Out
- Forecast at an intermediate level, then reconcile both up and down
- **Pro**: Balances detail and stability
- **When to use**: When a natural "middle" level exists (e.g., category in retail)

#### Optimal Reconciliation (MinT)

MinT (Minimum Trace) finds the reconciliation matrix that minimizes the trace of the forecast error variance-covariance matrix while satisfying aggregation constraints. Proposed by Wickramasuriya et al., it uses in-sample residuals with:

- **SAM**: Sample covariance matrix estimation
- **SHR**: Shrinkage estimation (better for large hierarchies)

**Key insight**: Optimal reconciliation uses all information available within the hierarchy, generating more accurate coherent forecasts than traditional alternatives that use limited information.

#### Empirical Risk Minimization (ERM)

Ben Taieb & Koo relaxed the unbiasedness assumption through ERM, formulating the problem as L1-regularized empirical risk minimization. This can outperform MinT when the unbiasedness assumption is violated.

### Practical Recommendations

1. **Always reconcile**: Independent forecasts at different levels are almost always incoherent. Even simple bottom-up is better than nothing
2. **Use MinT with shrinkage** for most applications
3. **Cross-validate** to choose between reconciliation methods -- performance depends on data structure
4. **For large hierarchies**: Use sub-hierarchy decomposition for computational scalability
5. **Grouped time series**: When items belong to multiple overlapping groups (e.g., product x region), use grouped reconciliation rather than strict hierarchies

### When Hierarchical Methods Matter

- **Retail**: SKU / store / region / national forecasts must be coherent for inventory planning
- **Supply chain**: Component / assembly / product forecasts drive procurement
- **Finance**: Department / division / company budgets
- **Energy**: Household / neighborhood / city / grid forecasts

**Python packages**: `hierarchicalforecast` (Nixtla), `scikit-hts`

### Key Sources

- [Hyndman & Athanasopoulos: Forecasting Principles and Practice, Ch. 11.3](https://otexts.com/fpp3/reconciliation.html)
- [Wickramasuriya et al.: MinT Paper](https://robjhyndman.com/publications/mint/)
- [Nixtla HierarchicalForecast GitHub](https://github.com/Nixtla/hierarchicalforecast)

---

## 6. Forecasting Pipelines and MLOps

### Core MLOps Concepts for Time Series

Time series MLOps extends standard MLOps with critical temporal considerations: data cannot be randomly split, retraining must respect temporal order, and concept drift is the norm rather than the exception.

### Feature Stores for Time Series

**Purpose**: Centralize and version time-dependent features (lags, rolling statistics, calendar features, external regressors) to ensure consistency between training and inference.

**Key tools**:
- **Feast**: Open-source feature store with time-travel capabilities (point-in-time correct joins)
- **Hopsworks**: Feature store + model registry + model serving in a unified platform
- **Tecton**: Managed feature platform with real-time feature computation

**Best practices**:
- Use **feature views** to abstract underlying data sources
- Implement **point-in-time correct** feature retrieval to prevent data leakage
- Version feature transformations alongside model code

### Model Retraining Strategies

**Trigger types**:
1. **Scheduled**: Retrain on a fixed cadence (daily, weekly, monthly)
2. **Drift-triggered**: Retrain when monitoring detects performance degradation
3. **Event-triggered**: Retrain after known distribution shifts (new product launch, policy change)

**Retraining patterns**:
- **Full retrain**: Refit model on all available data. Simple but expensive
- **Incremental update**: Update model with new data only. Faster but may accumulate errors
- **Sliding window**: Train on a fixed-size recent window. Balances recency and stability

**Cooling periods**: After a drift-triggered retrain, implement a cooling period before allowing another retrain. Wait for a lookahead period to evaluate the new model's forecasts before declaring success.

### Monitoring Forecast Quality

**Metrics to track in production**:
- **MAPE / WMAPE**: Business-interpretable accuracy
- **Coverage**: What fraction of actuals fall within prediction intervals
- **Bias**: Systematic over/under-forecasting
- **Residual distributions**: Shifts indicate model degradation

**Tools**:
- **Evidently AI**: Open-source ML monitoring with drift detection
- **Prometheus + Grafana**: Infrastructure-level monitoring of forecast latency and throughput
- **Whylogs**: Lightweight data profiling and drift detection
- **Azure ML Diagnostics** / **SageMaker Model Monitor**: Cloud-native solutions

### Concept Drift Detection

**What is concept drift?** The statistical relationship between input features X and target Y changes over time. In time series, this manifests as:
- **Sudden drift**: Abrupt regime change (e.g., pandemic lockdown)
- **Gradual drift**: Slow shift in consumer behavior
- **Seasonal drift**: Recurring but evolving seasonal patterns
- **Recurring drift**: Patterns that reappear periodically

**Detection methods**:
- **DDM (Drift Detection Method)**: Monitors error rate; signals when error exceeds a threshold above the minimum observed. Classic and well-understood
- **KSWIN (Kolmogorov-Smirnov Windowing)**: Compares distributions of recent vs. historical predictions using the KS test. Particularly effective with interpretable hyperparameters
- **ADWIN (Adaptive Windowing)**: Maintains a variable-length window and detects drift by comparing sub-windows
- **Page-Hinkley test**: Sequential hypothesis test for detecting mean shifts

**Practical approach**:
1. Monitor prediction error (not just input distributions) -- this catches concept drift directly
2. Use **multiple detectors** in parallel for robustness
3. Set **alert thresholds** based on business impact, not just statistical significance
4. Maintain a **drift log** to correlate detected drifts with real-world events

### A/B Testing and Backtesting Forecasts

#### Backtesting (Time Series Cross-Validation)

Traditional cross-validation fails for time series because it violates temporal ordering. Instead, use:

**Walk-forward validation (expanding window)**:
- Train on data up to time t, forecast t+1 to t+h
- Expand training window by one period, repeat
- Preserves temporal order; gold standard for evaluation

**Sliding window validation**:
- Fixed-size training window slides forward
- Better when older data is less relevant

**Purging**: Introduce a gap between training and test sets to prevent data leakage from highly correlated adjacent observations.

**Refitting strategies**:
- Refit every iteration (most accurate, slowest)
- Refit every N iterations (practical compromise)
- Fit once, forecast with updated features (fastest)

#### A/B Testing Forecasts in Production

- Deploy competing models to serve different segments
- Measure downstream business impact (inventory costs, stockouts) rather than just statistical accuracy
- Use **shadow mode**: new model runs in parallel but doesn't serve decisions until validated
- Consider **multi-armed bandit** approaches for dynamic allocation between models

**Python packages**: `skforecast` (backtesting), `mlflow` (experiment tracking), `evidently` (monitoring)

### Key Sources

- [Neptune.ai: MLOps Pipeline for Time Series](https://neptune.ai/blog/mlops-pipeline-for-time-series-prediction-tutorial)
- [AWS Blog: Robust Time Series Forecasting with MLOps](https://aws.amazon.com/blogs/machine-learning/robust-time-series-forecasting-with-mlops-on-amazon-sagemaker/)
- [Deepchecks: Addressing Drifts in Time Series](https://www.deepchecks.com/addressing-drifts-in-time-series-forecasting/)
- [Skforecast: Backtesting Documentation](https://skforecast.org/0.14.0/user_guides/backtesting.html)

---

## 7. Automated Model Selection

### The Landscape

| Framework | Methods Covered | Key Strength | Limitation |
|-----------|----------------|--------------|------------|
| **AutoGluon-TimeSeries** | Statistical + ML + DL + Foundation | Best ensemble approach | Requires more compute |
| **StatsForecast auto_arima** | ARIMA family | Fastest auto-ARIMA in Python | Statistical models only |
| **Auto_TS** | ARIMA, SARIMAX, Prophet, XGBoost | Single-line interface | Less sophisticated ensembling |
| **FLAML** | LightGBM, XGBoost, etc. | Fast, resource-aware tuning | Not time-series-native |
| **Merlion** | Statistical + DL + Anomaly | Salesforce-backed, anomaly detection | Smaller community |

### When Automation Works Well

1. **Large portfolios of heterogeneous series**: Impossible to manually tune thousands of models
2. **Initial exploration**: Quickly identify which model families work for your data
3. **Baseline establishment**: Automated ensemble often beats any single hand-tuned model
4. **Resource-constrained teams**: No dedicated forecasting expertise available

### Risks of Fully Automated Forecasting

1. **Overfitting to validation**: Automated systems optimize validation metrics, which may not reflect real deployment conditions
2. **Loss of domain insight**: Automation can obscure business-critical patterns (e.g., known upcoming promotions)
3. **Computational cost**: Exhaustive search across many model families is expensive
4. **Black-box ensembles**: Hard to explain why the system chose a particular model
5. **Distribution shift blindness**: Models selected on historical data may not generalize to regime changes
6. **False confidence**: Strong backtesting results can create overconfidence if the evaluation protocol is flawed

### Practical Recommendations

1. **Use AutoGluon-TimeSeries as the default starting point** for automated forecasting
2. **Always backtest with walk-forward validation** -- never use random splits
3. **Combine automation with domain knowledge**: use automated model selection, but inject known events, constraints, and business logic
4. **Monitor automated systems aggressively**: drift detection is essential when humans aren't in the loop
5. **Set compute budgets**: AutoGluon's `time_limit` parameter prevents runaway training
6. **Evaluate on multiple metrics**: MAPE alone is insufficient; check coverage, bias, and tail behavior

### Key Sources

- [AutoGluon-TimeSeries Documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html)
- [AutoGluon-TimeSeries Paper](https://arxiv.org/pdf/2308.05566)
- [Auto_TS GitHub](https://github.com/AutoViML/Auto_TS)

---

## 8. Domain-Specific Considerations

### Demand Forecasting / Supply Chain

**Characteristics**: Intermittent demand (many zeros), promotional effects, new product launches, hierarchical structure (SKU/store/region).

**Recommended methods**:
- **Croston's method** or **IMAPA** for intermittent demand
- **LightGBM/XGBoost** with promotional features for retail
- **DeepAR** or **Temporal Fusion Transformer** for large SKU portfolios
- **Hierarchical reconciliation** for coherent multi-level forecasts

**Best practices**:
- Feature-engineer promotional calendars, price changes, competitor actions
- Use **MAPA** (Multiple Aggregation Prediction Algorithm) for seasonality -- outperforms Holt-Winters
- Measure accuracy with **WMAPE** (weighted by revenue/volume) rather than simple MAPE
- Combine quantitative forecasts with qualitative sales intelligence
- AI adoption for supply chain forecasting more than doubled for SMBs (23% to 48%) from 2024 to 2025

**Python packages**: `statsforecast` (Croston, IMAPA), `mlforecast` (tree-based), `gluonts` (DeepAR, TFT)

### Financial Time Series

**Characteristics**: Non-stationary, heavy-tailed distributions, volatility clustering, regime changes, high-frequency data, mean reversion vs. momentum.

**Recommended methods**:
- **GARCH family** (GARCH, EGARCH, GJR-GARCH) for volatility modeling
- **Hybrid GARCH-Neural Network** models: GARCH-GRU achieves the best accuracy-efficiency tradeoff (3x faster than GARCH-LSTM with comparable accuracy)
- **LSTM/GRU** for capturing nonlinear dependencies
- **Transformer-based models** for multi-asset forecasting

**Key findings (2025)**:
- Deep learning consistently outperforms GARCH at medium and long-term horizons
- GARCH-informed neural networks (embedding GARCH dynamics into RNN gating structures) represent the current state of the art
- Hybrid models combining GARCH with MLP-Mixer outperform benchmarks in VaR estimation

**Best practices**:
- Never forecast raw prices; forecast **returns** or **volatility**
- Use **rolling-window evaluation** to account for regime changes
- Implement **purging and embargo** in cross-validation to prevent lookahead bias
- Consider **ensemble of GARCH + neural** rather than replacing one with the other

**Python packages**: `arch` (GARCH), `pytorch-forecasting` (TFT), `gluonts`

### Energy Load Forecasting

**Characteristics**: Strong daily/weekly/annual seasonality, weather dependence, peak load criticality, high-frequency data (15-min or hourly).

**Recommended methods**:
- **SARIMAX** with weather covariates for short-term
- **Random Forest / XGBoost** with engineered features (temperature, humidity, day-of-week, holidays)
- **CNN-LSTM hybrids**: CNN extracts spatial features and reduces noise; LSTM captures temporal dependencies
- **Foundation models**: Chronos achieves lowest NRMSE for short-term load forecasting among contemporary models

**Best practices**:
- Weather is the single most important exogenous variable
- Model separately for **base load** vs. **peak load**
- Use ensemble methods (boosting, bagging) for robustness
- Monitor for structural changes (solar panel adoption, EV charging, heat pump rollout)

**Python packages**: `neuralforecast`, `statsforecast`, `skforecast`

### Healthcare / Epidemiological Forecasting

**Characteristics**: Compartmental dynamics (SIR/SEIRD), intervention effects, reporting delays, spatial spread, high uncertainty.

**Recommended methods**:
- **Compartmental models** (SIR, SEIRD) for mechanistic understanding
- **ARIMA / SARIMA** for empirical forecasting
- **Foundation models** (Chronos, TimeGPT, TabPFN-TS) show strong accuracy across diverse pathogens
- **PandemicLLM**: Multimodal LLM that combines text (policies), genomic data, spatial data, and time series
- **Ensemble approaches** combining mechanistic + statistical + ML models

**Best practices**:
- Combine **mechanistic** (compartmental) with **empirical** (statistical/ML) models
- Account for **reporting delays** and **data revisions**
- Use **probabilistic forecasts** with calibrated uncertainty -- point forecasts are dangerous for policy
- Validate against **multiple past outbreaks**, not just one
- Foundation models are increasingly competitive for multi-season forecasting

**Python packages**: `epyestim`, `epiforecasts`, `neuralforecast`, `chronos-forecasting`

### Weather and Climate

**Characteristics**: Massive spatial-temporal datasets, physics-based constraints, high dimensionality, extreme event prediction.

**Recommended methods**:
- **Numerical Weather Prediction (NWP)**: Physics-based, computationally expensive, the traditional gold standard
- **Deep learning foundation models** (2025 state of the art):
  - **ClimaX**: Self-supervised transformer on CMIP6 data, superior in weather forecasting and climate projections
  - **FourCastNet 3**: Geometric approach to probabilistic ML weather forecasting
  - **Pangu-Weather**, **GraphCast**, **GenCast**: Large-scale models competitive with operational NWP
- **NeuralForecast library**: 14 transformer and RNN models evaluated for weather patterns

**Key taxonomy of training paradigms**:
1. **Deterministic predictive learning**: Direct point forecasts
2. **Probabilistic generative learning**: Distribution estimation
3. **Pre-training and fine-tuning**: Foundation model approach

**Best practices**:
- Foundation models are faster and sometimes more accurate than traditional NWP
- Physics-informed constraints improve neural model reliability
- Ensemble of multiple models (both NWP and ML) provides best calibration
- Spatial-temporal attention mechanisms are critical for capturing geographic dependencies

**Python packages**: `neuralforecast`, `weatherbench2`, `climetlab`

---

## 9. Time Series Foundation Models

### The Paradigm Shift

Foundation models transform time series forecasting from a **model training problem** into a **model selection problem**. Pretrained on massive datasets, they can forecast new patterns without additional training (zero-shot), similar to how LLMs can write about topics never explicitly seen.

### The 2025-2026 Landscape

| Model | Developer | Architecture | Size | Key Strength |
|-------|-----------|-------------|------|--------------|
| **Chronos-2** | Amazon | T5 encoder | 9M-710M | Production maturity, multivariate, covariates |
| **MOIRAI-2** | Salesforce | Decoder-only transformer | Various | Any-Variate Attention, universal forecasting |
| **Lag-Llama** | Academic | Decoder-only (LLaMA-inspired) | Various | Probabilistic, fully open-source |
| **Time-LLM** | Academic | LLM adapter/reprogrammer | Uses frozen LLM | Leverages existing LLM infrastructure |
| **TimesFM** | Google | Patch-based decoder-only | Various | Enterprise-grade, 100B training points |
| **TimeGPT-2.1** | Nixtla | Proprietary | Proprietary | Multivariate, cloud API |

### Detailed Profiles

**Chronos-2** (Amazon): Most mature option. T5-based, tokenizes values through scaling and quantization. 300+ forecasts/sec on a single GPU. Best performance on major benchmarks. Five sizes for performance-compute tradeoff. Native AWS integration.

**MOIRAI-2** (Salesforce): "Any-Variate Attention" dynamically adapts to any number of variables without fixed input dimensions. Trained on LOTSA dataset (27B observations, 9 domains). Strong on both in-distribution and zero-shot tasks. Fully open-source.

**Lag-Llama** (Academic): Decoder-only transformer generating full probability distributions (not just point forecasts). Uses lagged features as covariates. Strong few-shot learning when fine-tuned on small datasets. Fully open-source with permissive licensing.

**Time-LLM** (Academic): Reprograms frozen LLMs (GPT-2, LLaMA, BERT) for forecasting by translating time series patches into text prototypes. "Prompt-as-Prefix" injects domain knowledge via natural language. No separate forecasting infrastructure needed.

**TimesFM** (Google): Patch-based decoder-only, pretrained on 100B real-world time points from Google's internal datasets. Battle-tested in Google production environments. Enterprise-grade reliability.

### Critical Assessment

**When foundation models excel**:
- Zero-shot scenarios with no training data available
- Rapid prototyping and baseline establishment
- Cross-domain transfer where patterns are universal
- Short-term forecasting at moderate frequencies

**When they struggle**:
- High-frequency data requiring long context windows (e.g., minute-level data with yearly seasonality needs 525,600 context tokens)
- Highly domain-specific patterns that deviate from pre-training distribution
- When domain-tuned models have been carefully optimized
- When traditional methods are already very strong for the specific dataset

**Honest benchmark assessment** (Superlinear, 2024): "Traditional methods still outperform [foundation models] in many scenarios." However, Chronos-2 (2025) has narrowed this gap significantly, and the trajectory is clearly toward foundation models becoming competitive or superior.

### Key Sources

- [MLMastery: 2026 Time Series Toolkit](https://machinelearningmastery.com/the-2026-time-series-toolkit-5-foundation-models-for-autonomous-forecasting/)
- [Superlinear: Foundation Models -- Future or Folly?](https://superlinear.eu/insights/articles/foundation-models-for-forecasting-the-future-or-folly)
- [Amazon Chronos-2 on Hugging Face](https://huggingface.co/amazon/chronos-2)
- [MOIRAI-2 GitHub](https://github.com/SalesforceAIResearch/uni2ts)

---

## 10. Practical Decision Framework

### Choosing a Method

```
START
  |
  v
How many time series? ---------> 1-10 series ---------> Manual modeling
  |                                                       (ARIMA, ETS, Prophet)
  v
  10-1000 series ---------> AutoGluon-TimeSeries ensemble
  |
  v
  1000+ series ---------> StatsForecast (baseline) +
  |                        DeepAR/TFT (if patterns are complex) +
  |                        Hierarchical reconciliation (if structure exists)
  v
Do you have training data? ---> No ---------> Foundation model (Chronos-2)
  |
  v
  Yes
  |
  v
Is the data tabular with     --> Yes ---------> LightGBM/XGBoost (MLForecast)
many exogenous features?
  |
  v
  No
  |
  v
Strong seasonality,          --> Yes ---------> StatsForecast (AutoARIMA, ETS)
simple patterns?
  |
  v
  No
  |
  v
Complex nonlinear patterns,  --> Yes ---------> NeuralForecast (TFT, N-HiTS)
large dataset?
  |
  v
Need causal impact /         --> Yes ---------> Orbit (Bayesian structural TS)
Bayesian inference?
  |
  v
Need probabilistic            --> Yes ---------> Lag-Llama, DeepAR, or
uncertainty quantification?                       conformal prediction on top
```

### The Recommended Stack for 2026

| Layer | Tool | Purpose |
|-------|------|---------|
| **Baseline** | StatsForecast | Fast statistical baselines (AutoARIMA, ETS) |
| **ML** | MLForecast | Tree-based models with feature engineering |
| **Neural** | NeuralForecast | Deep learning for complex patterns |
| **Foundation** | Chronos-2 / TimesFM | Zero-shot and transfer learning |
| **Ensemble** | AutoGluon-TimeSeries | Automated model selection + ensembling |
| **Hierarchy** | HierarchicalForecast | Coherent multi-level forecasts |
| **Causal** | Orbit | Bayesian inference + causal impact |
| **Monitoring** | Evidently + custom | Drift detection + forecast quality |
| **Orchestration** | Airflow / Prefect | Pipeline scheduling + retraining |
| **Feature Store** | Feast / Hopsworks | Feature management + serving |

### Universal Best Practices

1. **Always start with simple baselines** (seasonal naive, AutoARIMA, ETS). If you can't beat them, your complex model is adding noise, not signal
2. **Use walk-forward cross-validation** for all evaluations. Never use random splits for time series
3. **Monitor continuously in production**. Concept drift is the norm, not the exception
4. **Reconcile hierarchical forecasts**. Incoherent forecasts waste resources
5. **Combine models via ensembling**. Diversity of model families consistently outperforms any single model
6. **Measure what matters to the business**. MAPE is standard but not universal -- consider WMAPE, coverage, and downstream costs
7. **Inject domain knowledge**. The best forecasting system combines automated model selection with human expertise on events, constraints, and business context
8. **Version everything**. Data, features, models, and evaluation results must be reproducible
9. **Plan for retraining**. Define triggers (schedule, drift, events) and cooling periods before deployment
10. **Foundation models are not a silver bullet** (yet). Evaluate them alongside traditional methods on your specific data before committing

---

*Research compiled from web sources, academic papers, official documentation, and industry blog posts. All findings reflect the state of the art as of March 2026.*
