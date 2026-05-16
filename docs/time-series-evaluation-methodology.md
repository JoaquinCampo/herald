# Time Series Forecasting Evaluation Methodology

## A Comprehensive Research Report on Best Practices and Common Pitfalls

*Compiled: 2026-03-23*

---

## Table of Contents

1. [Forecasting Metrics](#1-forecasting-metrics)
2. [Proper Train/Test Splitting](#2-proper-traintest-splitting)
3. [Backtesting](#3-backtesting)
4. [Baseline Models](#4-baseline-models)
5. [Data Leakage](#5-data-leakage-in-time-series)
6. [Uncertainty Quantification](#6-uncertainty-quantification)
7. [Forecast Combination](#7-forecast-combination)
8. [Common Mistakes Checklist](#8-common-mistakes-checklist)
9. [Decision Flowcharts](#9-decision-flowcharts)
10. [References](#10-references)

---

## 1. Forecasting Metrics

### 1.1 Scale-Dependent Metrics

These metrics are on the same scale as the data and **cannot be used to compare across series with different units or scales**.

| Metric | Formula | Notes |
|--------|---------|-------|
| **MAE** | `mean(\|e_t\|)` | Easy to interpret. Minimizing MAE produces median forecasts. |
| **RMSE** | `sqrt(mean(e_t^2))` | Penalizes large errors more heavily. Minimizing RMSE produces mean forecasts. |
| **MSE** | `mean(e_t^2)` | Sensitive to outliers; useful for optimization but hard to interpret in original units. |
| **MdAE** | `median(\|e_t\|)` | Robust to outliers. |

**When to use**: Comparing models on a *single* time series or across series measured in the same units and at similar scales. Prefer MAE for robustness; RMSE when large errors are disproportionately costly.

### 1.2 Percentage-Based Metrics

| Metric | Formula | Range | Problems |
|--------|---------|-------|----------|
| **MAPE** | `100 * mean(\|e_t / y_t\|)` | [0, inf) | Undefined when y_t = 0. Asymmetric: penalizes over-forecasts more than under-forecasts. Biased toward low forecasts. Near-zero actuals cause explosive values. Assumes meaningful zero point (fails for temperature in C/F). |
| **sMAPE** | `200 * mean(\|e_t\| / (\|y_t\| + \|y_hat_t\|))` | [0, 200] | Despite name, still asymmetric. Multiple inconsistent definitions in literature. Undefined when both actual and forecast are zero. Penalizes over-predictions more than under-predictions. Hyndman explicitly recommends avoiding it. |
| **WAPE** | `sum(\|e_t\|) / sum(\|y_t\|)` | [0, inf) | More stable than MAPE for datasets with near-zero values. Equivalent to volume-weighted MAPE. |

**Critical warning from Hyndman**: The sMAPE was used in the M3 and M4 competitions but has been widely criticized. There are at least four different definitions in the literature (Armstrong 1985, Makridakis 1993, Flores 1986, Makridakis & Hibon 2000), none of which are truly symmetric. Hyndman states: "I would not recommend using any of them."

**When to use percentage metrics**: Only when all series are strictly positive, far from zero, and stakeholders need easily interpretable "percentage accuracy." Even then, prefer WAPE over MAPE.

### 1.3 Scaled Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MASE** | `mean(\|e_t\|) / mean(\|y_t - y_{t-m}\|)` | < 1: better than seasonal naive. = 1: equivalent to seasonal naive. > 1: worse than seasonal naive. |

**Why Hyndman recommends MASE** (Hyndman & Koehler, 2006):

1. **Scale-free**: Numerator and denominator are both in original units, so they cancel out.
2. **Handles zeros**: No division by actual values, so zero actuals cause no problems.
3. **Symmetric**: Equal penalty for over- and under-forecasting.
4. **Meaningful zero**: Does not require the measurement scale to have a meaningful zero point.
5. **Interpretable benchmark**: Directly compares against the naive (or seasonal naive) method.
6. **Well-defined**: Unlike sMAPE, has a single unambiguous definition.
7. **Works for intermittent demand**: Handles sparse/zero-inflated series gracefully.

MASE has been proposed as the standard measure for comparing forecast accuracy across multiple time series, and was used as the primary metric in the M5 competition.

### 1.4 Relative Metrics

| Metric | Definition | Notes |
|--------|------------|-------|
| **Relative MAE (RelMAE)** | `MAE_method / MAE_benchmark` | Depends on choice of benchmark. |
| **Geometric Mean Relative MAE (GMRAE)** | `exp(mean(log(RelMAE_i)))` | Geometric mean handles the skewness of ratios better than arithmetic mean. |
| **MdRAE** | `median(RelMAE_i)` | Robust alternative to GMRAE. |

### 1.5 Probabilistic Metrics

Point forecasts alone are insufficient. The following metrics evaluate the full predictive distribution.

| Metric | What It Measures | Notes |
|--------|-----------------|-------|
| **CRPS** (Continuous Ranked Probability Score) | Overall quality of predictive distribution | Generalizes MAE to distributions. Lower is better. Equivalent to MAE for point forecasts. |
| **Quantile Loss / Pinball Loss** | Accuracy at specific quantiles | `rho_q(y, f) = 2(1-q)(f-y)` if `y < f`, else `2q(y-f)`. Used in M5 competition. |
| **Weighted Quantile Loss (WQL)** | Average quantile loss across quantile levels | Scale-dependent; used by AutoGluon as default. |
| **Scaled Quantile Loss (SQL)** | Quantile loss normalized by seasonal naive error | Scale-free analogue of WQL. Equivalent to MASE when only median quantile is used. |
| **Coverage** | Fraction of actuals falling within prediction intervals | Target: should match nominal level (e.g., 95% interval should cover 95% of actuals). |
| **Winkler Score** | Interval width + penalty for non-coverage | Rewards narrow intervals that still cover the actual. |

**When to use probabilistic metrics**: Always, if your application requires uncertainty estimates. CRPS is the gold standard for evaluating full predictive distributions. Use quantile loss when you care about specific risk levels.

### 1.6 Metric Selection Guidelines

| Scenario | Recommended Metric(s) |
|----------|----------------------|
| Single series, same scale | MAE, RMSE |
| Multiple series, different scales | MASE, GMRAE |
| Stakeholders need percentages | WAPE (not MAPE) |
| Intermittent / sparse demand | MASE (never MAPE/sMAPE) |
| Probabilistic forecasts | CRPS, WQL, SQL |
| Competition / benchmarking | MASE + sMAPE (M4), MASE + WRMSSE (M5) |
| Financial applications | Custom loss aligned with economic cost |

---

## 2. Proper Train/Test Splitting

### 2.1 Why Random Splitting Is WRONG

Random train/test splitting violates the fundamental temporal ordering of time series data. Consequences:

- **Data leakage**: Training on future data to predict the past inflates performance estimates.
- **Broken autocorrelation structure**: Random splits destroy temporal dependencies the model should learn.
- **Unrealistic evaluation**: Production models never have access to future data.
- **Overly optimistic results**: Models appear more accurate than they will be in deployment.

**One exception** (Bergmeir et al., 2018): For pure AR models where residuals are uncorrelated, randomized CV can be valid. However, this requires verifying residual independence via the Ljung-Box test on out-of-sample residuals. If autocorrelation remains, CV will underestimate the true generalization error.

### 2.2 Temporal Train/Test Split

The simplest valid approach: train on the first portion of the data, test on the last portion.

```
|--- Training Set (e.g., 80%) ---|--- Test Set (e.g., 20%) ---|
t=1                              t=T_split                    t=T
```

**Guidelines for the split point**:
- Test set should be **at least as long as the maximum forecast horizon**.
- Typical test set size: 20% of total data, but depends on series length.
- For seasonal data, test set should contain at **least one full seasonal cycle**.
- For short series (< 100 observations), hold out less data to preserve training signal.

### 2.3 Expanding Window (Growing Training Set)

At each step, the training set grows by including more historical data:

```
Fold 1: [=======TRAIN=======][TEST]
Fold 2: [========TRAIN========][TEST]
Fold 3: [=========TRAIN=========][TEST]
...
```

**Best for**: Small datasets / short series where you cannot afford to discard older data.

### 2.4 Sliding Window (Fixed-Size Training Set)

Training window stays fixed in size, sliding forward:

```
Fold 1: [===TRAIN===][TEST]
Fold 2:  [===TRAIN===][TEST]
Fold 3:   [===TRAIN===][TEST]
...
```

**Best for**: Long series where older data may be less relevant (concept drift), or ML models that have no built-in notion of time beyond the input window.

### 2.5 Walk-Forward Validation

The gold standard for time series model evaluation. Combines expanding or sliding windows with sequential forecasting:

1. Train on initial window.
2. Forecast the next h steps.
3. Record forecast errors.
4. Advance the origin by one (or more) steps.
5. Retrain (or update) the model.
6. Repeat until the end of the series.

**Key decisions**:
- **Retrain vs. update**: Full retraining at every step is expensive. For ML models, periodic retraining (e.g., monthly) with data updates in between is practical.
- **Step size**: Rolling forward by 1 gives the most evaluations but is expensive. Skip origins (analogous to k-fold) for computational savings.
- **Initial training size**: Must be large enough for the model to learn seasonal patterns (at least 2-3 full seasonal cycles).

### 2.6 Gap/Purge for Financial Data

Standard time series CV can still leak information when observations near the train/test boundary are correlated. Marcos Lopez de Prado (2018) introduced:

- **Purging**: Remove training samples that are too close (in time) to any test sample, preventing overlap of information sets.
- **Embargo**: After each test fold, exclude a fixed percentage of subsequent observations from training (e.g., 5% of total observations).
- **Combinatorial Purged CV (CPCV)**: Partition data into N ordered groups, test on k of them, generating C(N,k) paths. Recent research (2025) shows CPCV's marked superiority in mitigating overfitting risks via lower Probability of Backtest Overfitting and superior Deflated Sharpe Ratio.

**When to use**: Financial time series with overlapping labels, features derived from rolling windows, or any scenario where train/test boundary leakage is a concern.

---

## 3. Backtesting

### 3.1 Definition

Backtesting evaluates model performance by applying it retrospectively to historical data using a rolling or expanding origin scheme. It is the time series analogue of cross-validation.

### 3.2 Rolling Origin Evaluation

The forecast origin rolls forward through time, producing multiple sets of forecast errors:

```
Origin 1: Train[1..T1]  -> Forecast[T1+1..T1+h]
Origin 2: Train[1..T2]  -> Forecast[T2+1..T2+h]   (T2 = T1 + step)
Origin 3: Train[1..T3]  -> Forecast[T3+1..T3+h]
...
```

### 3.3 How Many Origins to Use

- **More origins = more reliable error estimates**, but at higher computational cost.
- **Minimum**: At least 5-10 origins to get stable error estimates.
- **Ideal**: As many as computationally feasible. Each origin provides one independent evaluation.
- **Skip origins**: If step = 1 is too expensive, skip by k steps (analogous to thinning). This is the difference between "rolling origin" and "time series cross-validation" in the literature.

### 3.4 Computational Tradeoffs

| Approach | Cost | Reliability |
|----------|------|------------|
| Single train/test split | 1 model fit | Low - one evaluation point |
| Walk-forward, retrain every step | O(n) model fits | High - many evaluations |
| Walk-forward, retrain periodically | O(n/k) model fits | Good - balance of cost and reliability |
| Walk-forward, update only (no retrain) | 1 model fit + O(n) updates | Moderate - tests model stability |

**Practical advice**: For expensive models (deep learning), retrain every k steps and update parameters between retrains. For cheap models (ARIMA, ETS), retrain every step.

---

## 4. Baseline Models

### 4.1 Why Baselines Are Mandatory

"You should always (always) have a naive model. It's the simplest, cleanest, most intuitive way to explain whether your system is at least treading water." -- Andrew Gelman (Columbia University)

Baselines answer critical questions:
- **Is the data even predictable?** If naive methods perform well, sophisticated models may be unnecessary.
- **Is my model adding value?** Without a baseline, you cannot quantify improvement.
- **Is there a bug?** If your model underperforms naive, something is wrong (data leakage, feature engineering error, etc.).
- **Is complexity justified?** A 2% improvement over naive may not justify 100x compute cost.

### 4.2 Standard Baseline Methods

| Method | Forecast | Best For |
|--------|----------|----------|
| **Naive** (Persistence) | y_hat_{T+h} = y_T | Random walk data (finance, exchange rates) |
| **Seasonal Naive** | y_hat_{T+h} = y_{T+h-m} (last same-season value) | Data with strong seasonality |
| **Mean** | y_hat_{T+h} = mean(y_1, ..., y_T) | Stationary data with no trend |
| **Drift** | y_hat_{T+h} = y_T + h * (y_T - y_1) / (T-1) | Data with linear trend |
| **STL + Naive** | Decompose, then naive on seasonally adjusted | Complex seasonality |

### 4.3 When "Simple" Methods Actually Win

Research consistently shows simple methods winning in many scenarios:

- **M3 Competition** (Makridakis & Hibon, 2000): Simple methods like Theta outperformed complex approaches.
- **M4 Competition** (2018): Hybrid statistical-ML (ES-RNN) won, but pure ML methods underperformed exponential smoothing baselines.
- **Financial series**: The random walk (naive) remains extremely difficult to beat for exchange rates and stock prices (Rossi, 2013 survey).
- **Short series**: With limited data, parameter-rich models overfit; simple methods generalize better.
- **High noise**: When signal-to-noise ratio is low, complex models fit noise rather than signal.

A 2025 paper ("Mind the naive forecast!") specifically documents how sophisticated ML models published in top venues fail to beat naive baselines when properly evaluated.

### 4.4 Actionable Guidelines

1. **Always report baseline performance alongside model performance.**
2. **Use MASE**, which directly encodes the naive benchmark (MASE < 1 means beating naive).
3. **Include at least two baselines**: naive and seasonal naive.
4. **If your model cannot beat seasonal naive, investigate before publishing claims.**
5. **Report the percentage improvement over baseline**, not just raw metric values.

---

## 5. Data Leakage in Time Series

### 5.1 Types of Leakage

#### 5.1.1 Feature Leakage (Using Future Information)

Using features derived from future data points to predict past/present values:

- **Rolling statistics computed on the full series**: A rolling mean calculated before the train/test split uses future values in training features.
- **Lag features crossing the boundary**: Creating lag features after splitting can accidentally use test-set values.
- **External variables with future information**: Using economic indicators published after the forecast date.

#### 5.1.2 Target Leakage

The target variable (or a proxy) is directly or indirectly included as a feature:

- **Derived features that encode the target**: e.g., error rates computed from actuals, cluster labels derived from the full dataset.
- **Post-hoc labels**: Features created from events that occurred after the forecast origin.

#### 5.1.3 Leakage Through Global Normalization

**This is one of the most common and insidious forms of leakage.**

**WRONG**:
```python
# Normalize the entire dataset, THEN split
scaler.fit(full_data)           # <-- leaks test statistics into training
full_data_scaled = scaler.transform(full_data)
train, test = split(full_data_scaled)
```

**CORRECT**:
```python
# Split FIRST, then normalize
train, test = split(full_data)
scaler.fit(train)               # <-- fit only on training data
train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)  # <-- use training statistics
```

This applies to ALL preprocessing that computes statistics: scaling, PCA, imputation, decomposition, smoothing, differencing statistics, etc.

#### 5.1.4 Leakage Through Decomposition

Performing STL decomposition, EMD, or wavelet transforms on the full series before splitting leaks future information into the trend and seasonal components. A 2024 Nature paper specifically documented this problem with EMD-based methods.

#### 5.1.5 Leakage in Cross-Validation

Using k-fold CV on time series without temporal ordering allows the model to train on future data. Even with temporal CV, look-ahead bias can enter through:

- Features engineered before the CV loop.
- Hyperparameter tuning using information from test folds.
- Global feature selection based on all data.

### 5.2 Prevention Checklist

- [ ] Split data BEFORE any preprocessing.
- [ ] Fit all transformations (scaling, PCA, imputers) on training data only.
- [ ] Decomposition (STL, etc.) must be done within each CV fold separately.
- [ ] Rolling features must be computed using only past data at each time step.
- [ ] External features must be realistically available at forecast time.
- [ ] Hyperparameter search must use only training + validation data (never test).
- [ ] Verify that no feature has unrealistically high correlation with the target (sign of leakage).

---

## 6. Uncertainty Quantification

### 6.1 Why Point Forecasts Alone Are Insufficient

- **Decision-making requires risk assessment**: "Demand will be 100 units" is less useful than "Demand will be between 80-120 units with 95% confidence."
- **Forecast uncertainty varies over time**: Some periods are more predictable than others.
- **Uncertainty grows with horizon**: 1-step-ahead is more certain than 30-step-ahead.
- **Asymmetric costs**: The cost of under-forecasting may differ from over-forecasting. Prediction intervals enable cost-optimal decisions.

### 6.2 Parametric Prediction Intervals

Derived from distributional assumptions of the model:

- **ARIMA**: Based on residual variance and MA(inf) representation. First-step interval: `y_hat +/- 1.96 * sigma`.
- **ETS**: Analytical formulas exist for additive models (Table 8.9 in Hyndman FPP3). For some multiplicative models, simulation is required.
- **Assumption**: Residuals are normally distributed and homoscedastic. **Often violated in practice**, leading to poorly calibrated intervals.

### 6.3 Bootstrap Prediction Intervals

Simulate future paths by resampling from the fitted residuals:

1. Fit model, extract residuals.
2. For each simulation: sample residuals with replacement, generate a future path.
3. Repeat B times (e.g., B = 1000).
4. Take percentiles of simulated paths as prediction intervals.

**Advantages**: Does not require normality assumption. Captures non-linear effects.
**Limitations**: Computationally expensive. Assumes residual distribution is stationary.

### 6.4 Conformal Prediction for Time Series

Distribution-free methods that provide finite-sample coverage guarantees under minimal assumptions.

#### 6.4.1 EnbPI (Ensemble Batch Prediction Interval) -- Xu & Xie, ICML 2021

- First conformal prediction algorithm designed for time series.
- Trains ensemble models on different bootstrap subsets of historical data.
- Derives prediction intervals from the ensemble's conformity scores.
- **Does not require data exchangeability** (key advantage over standard conformal prediction).
- Avoids data splitting and retraining.
- **Limitation**: Can fail to meet target coverage under strong distribution shifts.

#### 6.4.2 ACI (Adaptive Conformal Inference) -- Gibbs & Candes, 2021

- Adapts the significance level alpha_t online to handle distribution shifts.
- Updates alpha_t recursively based on whether the previous interval covered the actual value.
- **Better adaptation to non-stationarity** than EnbPI.
- Works with any base predictor.

#### 6.4.3 Recent Benchmarks (2025)

A 2025 benchmarking paper found that:
- ACI successfully meets or exceeds 90% target coverage.
- EnbPI can fail to provide required coverage under certain conditions.
- Global-CP and parametric PI methods also perform well.
- No single method dominates across all scenarios.

### 6.5 Quantile Regression

Directly estimates conditional quantiles rather than the conditional mean:

- Fit separate models for each quantile (e.g., 0.025, 0.5, 0.975 for 95% interval).
- Or use a single model with quantile loss function.
- **Advantage**: No distributional assumptions. Can capture asymmetric uncertainty.
- **Risk**: Quantile crossing (upper quantile below lower quantile). Use monotone quantile regression or post-hoc sorting to fix.

### 6.6 Calibration Assessment

A prediction interval is **well-calibrated** if a nominal 95% interval contains 95% of actual observations:

| Metric | Definition | Target |
|--------|-----------|--------|
| **Empirical coverage** | Fraction of actuals within the interval | Should match nominal level |
| **Winkler score** | Interval width + penalty for misses | Lower is better |
| **PIT histogram** | Probability Integral Transform | Should be uniform |
| **Calibration plot** | Nominal coverage vs. empirical coverage | Should lie on diagonal |

**Common problems**:
- Intervals too narrow (under-coverage): Model is overconfident.
- Intervals too wide (over-coverage): Model is underconfident; lacks sharpness.
- Correct average coverage but wrong conditional coverage: Intervals are well-calibrated on average but not for specific subgroups/periods.

---

## 7. Forecast Combination

### 7.1 Why Combinations Usually Beat Individual Models

Bates and Granger (1969) initiated over 50 years of research showing:

> "The results have been virtually unanimous: combining multiple forecasts leads to increased forecast accuracy. In many cases one can make dramatic performance improvements by simply averaging the forecasts." -- Clemen (1989)

**Theoretical reasons**:
- **Diversification**: Different models capture different patterns. Errors are partially uncorrelated.
- **Robustness**: Reduces risk of picking the single worst model.
- **Bias-variance tradeoff**: Averaging reduces variance without increasing bias (much).

### 7.2 Simple Averaging vs. Weighted Combinations

| Method | Description | Performance |
|--------|------------|-------------|
| **Simple average** | Equal weight to all models | Surprisingly hard to beat. Robust to estimation error. |
| **Inverse-error weighting** | Weight proportional to 1/error on validation set | Slightly better when model quality varies greatly. |
| **Optimal (regression) weights** | OLS regression of actuals on forecasts | Prone to overfitting; often worse than simple average. |
| **Trimmed average** | Drop worst k models, average the rest | Good when some models are clearly poor. |

**Key finding** (Wang et al., 2023, reviewing 50+ years of research): Simple averaging has proven hard to beat. Weighted combinations require estimating weights, which introduces estimation error that often offsets any theoretical gain.

### 7.3 Dynamic Combinations

- Update weights after each new forecast based on recent performance.
- Better for non-stationary series where relative model performance changes over time.
- Examples: exponentially weighted averaging, Bayesian model averaging, online learning approaches.

### 7.4 Stacking and Meta-Learning

- **Stacking**: Train a meta-learner (e.g., linear regression, gradient boosting) on the out-of-sample predictions of base models.
- **Critical requirement**: Base model predictions used for training the meta-learner MUST be out-of-sample (generated via walk-forward validation), or you will have severe data leakage.
- **Meta-learning**: Learn which model works best for which type of series based on time series features (e.g., FFORMA approach from the M4 competition).

### 7.5 Practical Guidelines

1. **Start with simple averaging** of 3-5 diverse models.
2. **Ensure diversity**: Combine models of different families (e.g., ETS + ARIMA + ML).
3. **Validate the combination**: The combined forecast should be evaluated on the same held-out data as individual models.
4. **Do not combine highly correlated models**: Two neural networks with different seeds add less diversity than ARIMA + neural network.
5. **Consider combinations of forecast distributions**, not just point forecasts, for better prediction intervals.

---

## 8. Common Mistakes Checklist

### 8.1 Data Handling Mistakes

- [ ] **Random train/test split**: Must preserve temporal order.
- [ ] **Global normalization**: Fit scalers on training data only.
- [ ] **Preprocessing before splitting**: Decomposition, smoothing, differencing must happen after the split (or within each CV fold).
- [ ] **Ignoring missing values**: Forward-fill, interpolation, or explicit handling needed. Method choice must not use future information.
- [ ] **Not checking for duplicates or timestamp errors**: Verify data quality before modeling.

### 8.2 Modeling Mistakes

- [ ] **Overfitting**: Too many parameters for the data size. Complex models (deep learning) on small datasets.
- [ ] **Ignoring seasonality**: Failure to identify or correctly specify seasonal periods. Multiple seasonalities (daily + weekly + yearly) need explicit handling.
- [ ] **Over-differencing**: Differencing a stationary series introduces artificial negative autocorrelation. Use ADF and KPSS tests together (test both null hypotheses).
- [ ] **Not checking residual autocorrelation**: Significant autocorrelation in residuals means the model has not captured all available signal. Use Ljung-Box test.
- [ ] **Confusing in-sample fit with out-of-sample accuracy**: A model with enough parameters can always fit training data perfectly. This says nothing about forecast quality.
- [ ] **Defaulting to complex models**: Neural networks are not always superior. Simple methods often win, especially with limited data.

### 8.3 Evaluation Mistakes

- [ ] **Using MAPE on data with near-zero values**: Causes infinite or misleading errors.
- [ ] **No baseline comparison**: Without naive/seasonal naive benchmarks, results are uninterpretable.
- [ ] **Single train/test split**: One evaluation point is unreliable. Use walk-forward validation.
- [ ] **Not testing statistical significance**: Better average performance may be due to chance. Use Diebold-Mariano test for pairwise comparison, or MCB (Multiple Comparisons with the Best) / Friedman test with Nemenyi post-hoc for multiple methods.
- [ ] **Reporting only point forecast metrics**: Include probabilistic evaluation (CRPS, coverage, Winkler score).
- [ ] **Cherry-picking forecast horizons**: Report accuracy across all horizons, not just the ones where your model looks best.

### 8.4 Deployment Mistakes

- [ ] **Not accounting for concept drift**: Data distributions change. Monitor forecast accuracy in production and retrain when performance degrades.
- [ ] **Using stale models**: Models trained on historical data degrade as the data-generating process evolves.
- [ ] **Ignoring external shocks**: Holidays, promotions, policy changes, pandemics -- these structural breaks invalidate historical patterns.
- [ ] **Not monitoring coverage of prediction intervals**: In production, track whether your 95% intervals actually cover 95% of actuals.

### 8.5 Comprehensive Pre-Publication Checklist

Before claiming "our method outperforms the state of the art":

1. [ ] Did you compare against naive and seasonal naive baselines?
2. [ ] Did you use a proper temporal train/test split (not random)?
3. [ ] Did you use walk-forward validation with multiple origins?
4. [ ] Did you fit all preprocessing (scaling, decomposition) on training data only?
5. [ ] Did you use appropriate metrics (MASE, not just MAPE)?
6. [ ] Did you test statistical significance of improvements?
7. [ ] Did you evaluate on multiple forecast horizons?
8. [ ] Did you report probabilistic forecast quality (not just point forecasts)?
9. [ ] Did you check residuals for autocorrelation?
10. [ ] Did you verify that no future information leaked into features?

---

## 9. Decision Flowcharts

### 9.1 Choosing a Metric

```
Is this a single series or multiple series of the same scale?
  YES -> MAE or RMSE
  NO  -> Are all values strictly positive and far from zero?
           YES -> WAPE (if stakeholders need percentages), else MASE
           NO  -> MASE (always safe)
                  Does your application need uncertainty estimates?
                    YES -> Add CRPS or WQL/SQL
                    NO  -> Point metrics sufficient, but consider adding them anyway
```

### 9.2 Choosing a Validation Strategy

```
Is computational budget very limited?
  YES -> Single temporal train/test split (minimum viable)
  NO  -> Is the series very short (< 100 observations)?
           YES -> Expanding window walk-forward validation
           NO  -> Is there likely concept drift?
                    YES -> Sliding window walk-forward validation
                    NO  -> Expanding window walk-forward validation
                          Is this financial data with overlapping features?
                            YES -> Add purging + embargo (de Prado CPCV)
                            NO  -> Standard walk-forward is sufficient
```

---

## 10. References

### Key Papers

- **Hyndman, R.J. & Koehler, A.B. (2006)**. "Another look at measures of forecast accuracy." *International Journal of Forecasting*, 22(4), 679-688. -- Proposes MASE; critiques MAPE and sMAPE.
- **Hewamalage, H., Bergmeir, C., & Bandara, K. (2023)**. "Forecast evaluation for data scientists: common pitfalls and best practices." *Data Mining and Knowledge Discovery*, 37, 788-832. -- Comprehensive tutorial on forecast evaluation for ML practitioners.
- **Bates, J.M. & Granger, C.W.J. (1969)**. "The Combination of Forecasts." *Operational Research Quarterly*, 20(4), 451-468. -- Foundational paper on forecast combination.
- **Wang, X., Hyndman, R.J., et al. (2023)**. "Forecast combinations: an over 50-year review." *International Journal of Forecasting*. -- Comprehensive review showing simple averaging is hard to beat.
- **Xu, C. & Xie, Y. (2021)**. "Conformal prediction interval for dynamic time-series." *ICML 2021*. -- EnbPI method.
- **Gibbs, I. & Candes, E. (2021)**. "Adaptive conformal inference under distribution shift." *NeurIPS 2021*. -- ACI method.
- **Lopez de Prado, M. (2018)**. *Advances in Financial Machine Learning*. Wiley. -- Purged k-fold CV.
- **Bergmeir, C., Hyndman, R.J., & Koo, B. (2018)**. "A note on the validity of cross-validation for evaluating autoregressive time series prediction." *Computational Statistics & Data Analysis*, 120, 70-83. -- When random CV is valid for time series.
- **Makridakis, S. & Hibon, M. (2000)**. "The M3-Competition." *International Journal of Forecasting*, 16(4), 451-476.
- **Tashman, L.J. (2000)**. "Out-of-sample tests of forecasting accuracy: an analysis and review." *International Journal of Forecasting*, 16(4), 437-450. -- Definitive guide to rolling origin evaluation.
- **Diebold, F.X. & Mariano, R.S. (2002)**. "Comparing predictive accuracy." *Journal of Business & Economic Statistics*, 20(1), 134-144. -- DM test for pairwise forecast comparison.

### Textbooks

- **Hyndman, R.J. & Athanasopoulos, G. (2021)**. *Forecasting: Principles and Practice*, 3rd ed. OTexts. Available free at https://otexts.com/fpp3/.

### Online Resources

- AutoGluon Time Series Metrics: https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html
- Skforecast Backtesting Guide: https://skforecast.org/0.14.0/user_guides/backtesting.html
- scikit-learn TimeSeriesSplit: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
