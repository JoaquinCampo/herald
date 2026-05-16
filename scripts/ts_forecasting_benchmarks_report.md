# Time Series Forecasting Benchmarks and Competitions: A Comprehensive Empirical Survey

**Date**: 2026-03-23
**Purpose**: Inform best practices for time series forecasting based on decades of competition evidence and recent foundation model benchmarks.

---

## Table of Contents

1. [M4 Competition (2018)](#1-m4-competition-2018)
2. [M5 Competition (2020)](#2-m5-competition-2020)
3. [M6 Competition (2022-2023)](#3-m6-competition-2022-2023)
4. [Monash Time Series Forecasting Archive](#4-monash-time-series-forecasting-archive)
5. [Key Meta-Findings Across Competitions](#5-key-meta-findings-across-competitions)
6. [Recent Large-Scale Benchmarks (2024-2026)](#6-recent-large-scale-benchmarks-2024-2026)
7. [Actionable Takeaways for Practitioners](#7-actionable-takeaways-for-practitioners)

---

## 1. M4 Competition (2018)

**Scale**: 100,000 real-life time series across multiple domains and frequencies (yearly, quarterly, monthly, weekly, daily, hourly). 49 valid submissions evaluated.

### Key Empirical Findings

1. **ES-RNN hybrid won decisively.** Slawek Smyl (Uber Technologies) submitted a hybrid Exponential Smoothing + Recurrent Neural Network (ES-RNN) model that was ~10% more accurate than the Combination (Comb) benchmark by sMAPE. For context, the M3 Competition's best method was only 4% better than the same benchmark — the ES-RNN represented a step-change in improvement.

2. **Combinations dominated the leaderboard.** Of the 17 most accurate methods, 12 were "combinations" of mostly statistical approaches. The second-place method combined seven statistical methods and one ML method, with combination weights learned by a ML algorithm (joint submission from University of A Coruna and Monash University).

3. **Pure ML methods performed poorly.** The six pure ML methods submitted all performed worse than the Combination benchmark, and only one beat Naive2 (seasonal naive). This was the single most surprising and discussed result of M4.

4. **Prediction intervals were finally calibrated.** The top two methods (ES-RNN and the stat+ML combination) correctly specified 95% prediction intervals — the first methods known to have done so without substantially underestimating uncertainty.

5. **The OWA (Overall Weighted Average) metric** combined sMAPE and MASE, normalized relative to Naive2, providing a single composite accuracy score. The ES-RNN achieved an OWA substantially below 1.0 (the Naive2 baseline).

### What Surprised the Community

- The complete failure of standalone deep learning / ML methods was unexpected given the hype around neural forecasting at the time.
- The winning approach was *not* end-to-end neural — it explicitly embedded statistical structure (exponential smoothing per-series) inside the neural architecture.
- Simple combinations of classical methods remained extremely competitive — just averaging ETS, ARIMA, and Theta variants got you into the top tier.

### Actionable Takeaways

- **Hybridize, don't replace.** Use ML to enhance statistical structures rather than ignoring decades of time series theory.
- **Always include a combination baseline** — average 3-5 well-chosen statistical methods before trying anything more complex.
- **The Comb benchmark** (simple average of SES, Holt, and Damped exponential smoothing) is a remarkably tough baseline.

---

## 2. M5 Competition (2020)

**Scale**: 42,840 hierarchical time series of Walmart unit sales across 10 stores, 3 states, and ~3,000 products. Two parallel tracks: Accuracy (point forecasts) and Uncertainty (prediction intervals). 5,558 teams from 101 countries participated in the Accuracy challenge.

### Key Empirical Findings

1. **LightGBM dominated.** The winning submission and virtually all top-50 solutions used LightGBM as their primary model. This was the first M-competition where pure ML approaches comprehensively beat all statistical benchmarks and their combinations. The M5 officially marked the inflection point where ML methods overtook classical stats for large-scale retail forecasting.

2. **Ensembles remained king.** Consistent with M4, the top three performers each employed ensembles of separately trained and tuned models with different training procedures and datasets.

3. **Top-50 all improved accuracy by >14%.** All top-50 submissions improved overall forecasting accuracy over the best-performing benchmark by more than 14%. The top-5 methods improved by >20%, and the winner achieved a 22.4% improvement.

4. **Hierarchical structure mattered.** The evaluation used WRMSSE (Weighted Root Mean Scaled Squared Error) across all hierarchical aggregation levels — from individual product-store to total national sales. Top solutions universally adopted a "global bottom-up" approach: train global models at the most granular level, then aggregate. This outperformed top-down and middle-out reconciliation.

5. **Feature engineering was decisive.** Winning features included: calendar effects (SNAP food stamp dates, holidays, sporting events), price features (rolling means, relative prices), lag features (7, 14, 28-day lags), rolling statistics, and encoding of hierarchical structure (store/department/category identifiers).

6. **Multiple direct models per horizon** (e.g., weekly LightGBM models for days 1-7, 8-14, 15-21, 22-28) outperformed single recursive models by better capitalizing on recent lag information.

### What Surprised the Community

- The complete dominance of gradient boosting over neural networks. Deep learning methods (LSTM, DeepAR, N-BEATS) did not appear in the top rankings despite being state-of-the-art on other benchmarks.
- Statistical methods — which had been competitive through M4 — were comprehensively outperformed. The key differentiator was the availability of rich exogenous variables (prices, calendars, promotions) that tree-based methods could exploit naturally.
- The Uncertainty track was far harder; few teams performed well on both tracks simultaneously.

### Actionable Takeaways

- **When rich covariates exist, use gradient boosting** (LightGBM, XGBoost, CatBoost). These models naturally handle mixed-type features, missing values, and irregular patterns.
- **Invest in feature engineering over model architecture.** Calendar encoding, price signals, and hierarchical identifiers drove most of the accuracy gains.
- **Use bottom-up hierarchical forecasting** with global models trained across all series simultaneously.
- **Consider multi-horizon direct strategies** (separate model per forecast window) rather than recursive multi-step forecasting.

---

## 3. M6 Competition (2022-2023)

**Scale**: 12-month live competition (February 2022 - February 2023). Participants forecasted monthly returns and made investment decisions for 100 assets (50 S&P 500 stocks + 50 international ETFs). 160 teams submitted, evaluated by Ranked Probability Score (forecasting) and Information Ratio (investment).

### Key Empirical Findings

1. **The vast majority of participants failed to beat the benchmark.** Over the full 48-week duration:
   - Only **23.3%** provided more precise forecasts than the benchmark
   - Only **28.8%** developed better-performing portfolios
   - Only **6.7%** achieved both better forecasts AND better portfolios
   - Only **3 participants** beat the benchmark in every single month

2. **Results support the Efficient Market Hypothesis.** These competition results mirror the professional investment industry: over the past 10-20 years, less than 7-10% of actively managed U.S. equity funds have outperformed their benchmarks. The M6 participants (23-29% beating benchmarks) actually performed *better* than professional fund managers, possibly due to self-selection of skilled quantitative participants.

3. **Limited connection between forecasting accuracy and investment performance.** Having accurate forecasts did not reliably translate into good investment returns. Risk management and portfolio construction mattered as much as prediction quality.

4. **The "wisdom of crowds" added value.** Aggregated predictions across participants performed well, consistent with ensemble/combination findings from earlier competitions.

5. **One winning approach was remarkably simple.** Marco Gorelli (2nd place Q1, 10th overall) used covariance estimation methods from the `precise` Python library — not deep learning, not complex ML. His approach involved cross-validated covariance estimator selection and simple portfolio optimization. He explicitly noted: "not with deep learning."

### What Surprised the Community

- The sheer difficulty of financial time series forecasting, even for a self-selected group of quantitative researchers and data scientists.
- That even among competition participants with access to sophisticated methods, most could not beat a simple benchmark.
- The weak link between forecasting accuracy and investment performance highlighted that prediction alone is insufficient for decision-making.

### Actionable Takeaways

- **Be humble about financial forecasting.** If 77% of competition participants and 93% of professional fund managers can't beat benchmarks, the bar is extremely high.
- **Forecasting accuracy != decision quality.** Good forecasts require good decision frameworks (risk management, portfolio construction) to translate into value.
- **Simple, well-calibrated methods can win** — covariance estimation and mean-variance optimization beat complex ML pipelines.
- **Aggregate forecasts when possible** — the wisdom of crowds effect is robust across all M-competitions.

---

## 4. Monash Time Series Forecasting Archive

**Scale**: 30 datasets (58 variations) across diverse domains, frequencies, and characteristics. Established as the standard benchmarking repository for global forecasting models. Available at forecastingdata.org.

### Datasets Included

The archive covers domains including:
- **Tourism**: Australian tourism visitor nights (monthly, quarterly, yearly)
- **Finance**: Stock prices, exchange rates, cryptocurrency
- **Energy**: Electricity demand, solar power, wind farms
- **Transport**: Traffic, rideshare
- **Nature**: Temperature, rainfall, sunspot activity
- **Retail**: M-competition subsets (M1, M3, M4)
- **Healthcare**: Hospital admissions, COVID-19
- **Web**: Wikipedia page views, KDD Cup

Datasets vary in frequency (minutely to yearly), series count (tens to hundreds of thousands), series length, and presence of missing values. The `.tsf` format was introduced for standardized storage.

### Benchmark Results

13 baseline methods were evaluated across all datasets using 8 error metrics (MASE, sMAPE, RMSE, etc.):

**Traditional univariate models** (6): ETS, ARIMA, Theta, TBATS, SES, Naive
**Global forecasting models** (7): CatBoost, FFNN, DeepAR, N-BEATS, WaveNet, Transformer, Pooled Regression

Key findings from the benchmark:
1. **No single method dominates across all datasets.** Performance varies substantially by dataset characteristics.
2. **Simple univariate benchmarks excel on data with strong trend and autocorrelation** (high ACF1). ETS and Theta methods are particularly robust here.
3. **ML and deep learning models perform better on high-entropy, uncertain datasets** where complex cross-series patterns can be learned.
4. **Global models (trained across series) show "huge potential"** compared to local univariate models, especially when many related series are available.
5. **CatBoost and N-BEATS** were among the strongest global models across the benchmark, though neither dominated universally.

### Actionable Takeaways

- **Use the Monash archive for benchmarking** any new forecasting method — it's the standard reference.
- **Match method to data characteristics**: high trend/seasonality favors statistical methods; high entropy/complexity favors ML/DL.
- **Always include simple baselines** (Naive, SES, ETS) — they win more often than you'd expect.
- **Global models shine when you have many related series** — don't default to fitting each series independently.

---

## 5. Key Meta-Findings Across Competitions

### 5.1 Simple Methods Are Often Competitive with Complex Ones

This is the single most replicated finding in forecasting research:

- **Armstrong & Green (2015)** reviewed 97 comparisons across 32 papers. *None* provided a balance of evidence that complexity improves accuracy. **Complexity increased forecast error by 27% on average** across 25 papers with quantitative comparisons.
- **M1 Competition (1982)**: The simplest suitable methods (deseasonalized random walk and single exponential smoothing) were at least as accurate as all 16 more complex methods. The two simplest methods reduced MAPE by **34%** compared to more complex methods.
- **M2 Competition (1993)**: Simple methods reduced MAPE by **27%** vs. complex methods.
- **M3 Competition (2000)**: Simple methods reduced MAPE by **~1%** vs. complex methods — the gap narrowed but the direction held.
- **Nassim Taleb** in *The Black Swan*: "Makridakis and Hibon reached the sad conclusion that 'statistically sophisticated and complex methods do not necessarily provide more accurate forecasts than simpler ones.'"
- **Key caveat from M5**: When rich exogenous variables are available, complex ML methods *do* substantially outperform simple ones. The "simple is competitive" finding holds strongest for univariate extrapolation.

### 5.2 Combination/Ensemble Methods Consistently Rank High

This finding is universal across every M-competition:

- **M1 (1982)**: "The accuracy when various methods are combined outperforms, on average, the individual methods being combined."
- **M3 (2000)**: Combinations among the top performers.
- **M4 (2018)**: 12 of the 17 most accurate methods were combinations. The Comb benchmark was remarkably hard to beat.
- **M5 (2020)**: Top-3 performers all used ensembles of separately-trained models with different training procedures.
- **Meta-learning research (2020-2024)**: All proposed meta-learner combination methods outperform simple Median and Mean combinations, and meta-models consistently produce more accurate predictions than base models.

**The theoretical explanation**: Combining reduces variance without increasing bias (analogous to bagging), and different methods capture different aspects of the time series structure.

### 5.3 Domain-Specific Features Matter More Than Model Complexity

Empirical evidence strongly supports prioritizing feature engineering:

- **M5 (2020)**: The decisive factor was not model architecture (everyone used LightGBM) but feature engineering — calendar events, price features, hierarchical encodings, lag structures.
- **Intermittent demand forecasting (2026)**: "Robust, statistically informed feature engineering can be more effective than increased model complexity." Two-stage hurdle models did *not* outperform single-stage feature-enhanced frameworks.
- **Gradient boosting success** across competitions is attributable to its ability to naturally incorporate domain-specific engineered features (even simple domain-specific lags).
- **General principle**: "Lagged values provide a universal foundation, but each domain's unique physics — market trends, human schedules, periodicity, sparsity patterns — dictate which features ultimately drive predictive power."

### 5.4 Statistical Methods vs. ML Methods: The Tradeoffs

| Dimension | Statistical Methods | ML/DL Methods |
|-----------|-------------------|---------------|
| **Univariate extrapolation** | Competitive or better (M1-M4) | Worse unless hybridized (M4) |
| **Rich covariates available** | Limited ability to incorporate | Substantially better (M5) |
| **Small sample / few series** | Better (less overfitting) | Need more data |
| **Many related series** | Cannot leverage cross-series info | Global models excel |
| **Interpretability** | High | Low-Medium |
| **Computational cost** | Very low | Medium-High |
| **Uncertainty quantification** | Well-calibrated (ETS, ARIMA) | Often overconfident |
| **Financial time series** | Nearly impossible to beat benchmarks (M6) | Also nearly impossible (M6) |

---

## 6. Recent Large-Scale Benchmarks (2024-2026)

### 6.1 The Rise of Time Series Foundation Models (TSFMs)

A new generation of pre-trained foundation models for time series has emerged:

| Model | Developer | Parameters | Key Feature |
|-------|-----------|------------|-------------|
| **Chronos-2** | Amazon | 120M | Encoder-only, supports covariates, 300+ forecasts/sec on A10G |
| **Chronos-Bolt** | Amazon | Various (Tiny to Large) | Fast inference variant of Chronos |
| **TimesFM** | Google | ~200M | Decoder-only, strong zero-shot performance |
| **Moirai 2.0** | Salesforce | Various | Quantile forecasting + multi-token prediction |
| **TTM** | IBM | ~9M | Tiny but competitive |
| **FlowState** | IBM | 9.1M | Smallest model in GIFT-Eval top-10 |
| **TimeGPT** | Nixtla | Undisclosed | First commercial TSFM, 100B+ training points |

### 6.2 GIFT-Eval Benchmark (2024-present)

The primary leaderboard for TSFMs, introduced by Salesforce at NeurIPS 2024:
- **23 datasets, 144,000+ time series, 177 million data points**
- Spans 7 domains, 10 frequencies, multivariate inputs
- Short to long-term prediction lengths

**Key findings**:
- Foundation models generally outperform both statistical and deep learning models in most domains.
- TSFMs struggle in high-entropy, low-trend domains (Web/CloudOps, Transport).
- IBM's FlowState (9.1M params) outperformed models 20x its size — model size is not everything.
- Chronos-2 achieved the highest skill score (0.473) on fev-bench, beating TiRex (0.426) and Toto-1.0 (0.407).

### 6.3 TSFM-Bench (2024)

A comprehensive benchmark covering zero-shot, few-shot, and full-shot evaluation:
- Covers TSFMs based on both large language models and time-series-native pre-training.
- **Critical finding on data contamination**: Some benchmark datasets were inadvertently included in the pretraining data of TimesFM, UniTS, and TTM, leading to an advantage of **47%-184% lower MSE** due to memorization. This means many reported zero-shot results are overly optimistic.

### 6.4 Key Results: Foundation Models vs. Classical Methods

**Where foundation models excel**:
- Zero-shot scenarios (no task-specific training data)
- Cross-domain generalization
- When minimal preprocessing is acceptable
- Covariate-informed tasks (Chronos-2 shows "substantially large" gap over prior models here)

**Where classical/statistical methods remain competitive**:
- According to DM statistical testing, **no TSFM (including Chronos-Bolt) obtains a statistically significant improvement over MSTL** (seasonal-trend decomposition) for electricity price forecasting with pronounced daily/weekly seasonality.
- Datasets with strong, regular seasonality and trend
- When computational budget is constrained
- When well-calibrated uncertainty estimates are required

**Where gradient boosting still excels**:
- Tasks with rich tabular covariates (prices, promotions, hierarchical features)
- Domain-specific forecasting with engineered features
- Production systems requiring interpretability and fast inference

### 6.5 The DLinear Controversy (2023-2025)

The AAAI 2023 paper "Are Transformers Effective for Time Series Forecasting?" showed a simple linear decomposition model (DLinear) outperforming most Transformer-based models:
- DLinear outperformed FEDformer by >40% on Exchange rate, ~30% on Traffic/Electricity/Weather, ~25% on ETTm1.
- This sparked intense debate. Subsequent work showed that with proper training protocols, Transformer variants (iTransformer, TimeXer, TimeMixer) recover and surpass DLinear.
- **The lesson**: Many published deep learning results in time series were inflated by suboptimal baselines and inconsistent experimental protocols.

---

## 7. Actionable Takeaways for Practitioners

### The Universal Rules (validated across all competitions)

1. **Always start with simple baselines.** Naive, seasonal naive, SES, ETS, and Theta methods. If you can't beat these, your complex model is doing something wrong.

2. **Combine forecasts.** A simple average of 3-5 diverse methods is one of the most reliable strategies in forecasting. This has been true since M1 (1982) and remains true in 2026.

3. **Feature engineering > model architecture** when covariates are available. Calendar features, price signals, lag structures, and hierarchical identifiers drive more accuracy than switching from XGBoost to a neural network.

4. **Match your method to your data regime:**
   - *Univariate, few series, strong seasonality*: ETS, Theta, ARIMA
   - *Many related series, rich covariates*: LightGBM, XGBoost
   - *Zero-shot, no training data*: Chronos-2, TimesFM
   - *Financial time series*: Accept that most methods won't beat benchmarks

5. **Use ensembles at every level.** Combine statistical + ML, combine multiple training strategies, combine multiple horizons.

6. **Evaluate on multiple metrics.** sMAPE, MASE, RMSE, and OWA can give different rankings. No single metric tells the whole story.

7. **Be skeptical of published benchmarks.** Data contamination in foundation model pretraining (47-184% MSE advantage) means many reported numbers are unreliable. Always run your own evaluation.

8. **Don't neglect uncertainty quantification.** Point forecast accuracy gets all the attention, but calibrated prediction intervals are equally important for decision-making (M4 and M6 both highlighted this).

9. **The gap between forecasting and decision-making is real.** M6 showed that accurate forecasts don't automatically lead to good decisions. Build decision frameworks, not just predictors.

10. **Complexity has a cost.** Armstrong & Green's meta-analysis found complexity increases error by 27% on average. Only add complexity when you have clear evidence it helps on *your* data.

---

## Sources

### Competition Papers
- [M4 Competition Results (Makridakis et al., 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785)
- [M5 Accuracy Competition (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S0169207021001874)
- [M6 Competition (Makridakis et al., 2023)](https://arxiv.org/abs/2310.13357)
- [Makridakis Competitions (Wikipedia)](https://en.wikipedia.org/wiki/Makridakis_Competitions)
- [M4 Methods Repository (GitHub)](https://github.com/Mcompetitions/M4-methods)
- [M6 Winner Approach (Gorelli)](https://m-e-gorelli.medium.com/how-i-won-6-000-in-the-m6-forecasting-competition-888df68bf132)

### Benchmarks and Archives
- [Monash Time Series Forecasting Archive](https://arxiv.org/abs/2105.06643)
- [Monash Forecasting Repository](https://forecastingdata.org/)
- [GIFT-Eval Benchmark](https://arxiv.org/abs/2410.10393)
- [TSFM-Bench](https://arxiv.org/abs/2410.11802)

### Foundation Models
- [Chronos-2 (Amazon)](https://arxiv.org/abs/2510.15821)
- [Moirai 2.0](https://arxiv.org/html/2511.11698v1)
- [IBM FlowState on GIFT-Eval](https://research.ibm.com/blog/SSM-time-series-model)

### Meta-Analyses
- [Simple vs. Complex Forecasting: The Evidence (Armstrong & Green, 2015)](https://www.sciencedirect.com/science/article/abs/pii/S014829631500140X)
- [Are Transformers Effective for Time Series Forecasting? (Zeng et al., AAAI 2023)](https://arxiv.org/abs/2205.13504)
- [Benchmarking Foundation Models for Time-Series Forecasting](https://www.mdpi.com/2813-0324/11/1/32)

### Practitioner Guides
- [M5 Lessons from Artefact](https://medium.com/artefact-engineering-and-data-science/sales-forecasting-in-retail-what-we-learned-from-the-m5-competition-445c5911e2f6)
- [Chronos-Bolt and AutoGluon (AWS)](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/)
- [Feature Engineering Primacy (Scientific Reports, 2026)](https://www.nature.com/articles/s41598-026-35197-y)
