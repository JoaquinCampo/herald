# Gradient Boosting for Time Series Forecasting: Research Report

**Date**: 2026-03-23
**Purpose**: Comprehensive best-practices survey for ML-based time series forecasting with gradient boosting (XGBoost, LightGBM, CatBoost).

---

## 1. Feature Engineering for Time Series ML

### 1.1 Lag Features

Lag features capture temporal dependencies by shifting target values backward by k steps. They are the single most important feature family for tree-based time series models.

**Best practices:**
- Start with lags matching known periodicities (lag 7 for daily data with weekly seasonality, lag 24 for hourly data with daily cycles, lag 365 for yearly).
- Include a contiguous block of recent lags (1 through p) to let the model learn short-term dynamics.
- Add "seasonal lags" at multiples of the period (7, 14, 21, 28 for weekly patterns).
- Use autocorrelation (ACF) and partial autocorrelation (PACF) plots to select informative lag orders.

```python
# Manual lag creation
for lag in [1, 2, 3, 7, 14, 28]:
    df[f"lag_{lag}"] = df["target"].shift(lag)

# With MLforecast (automated)
from mlforecast import MLForecast
fcst = MLForecast(
    models=[LGBMRegressor()],
    lags=[1, 2, 3, 7, 14, 28],
)
```

**Critical rule**: When predicting at time t, only lags >= forecast_horizon h are valid. If h=7, lag_1 through lag_6 will NOT be available at inference time (unless you use recursive prediction). This is the #1 source of leakage in time series ML.

### 1.2 Rolling Window Statistics

Rolling (sliding window) statistics smooth noise and capture local trends and volatility.

**Common statistics:**
- Mean (trend proxy)
- Standard deviation (volatility proxy)
- Min / Max (range)
- Skewness, kurtosis (distribution shape)
- Quantiles (e.g., 10th, 90th percentile)

**Best practices:**
- Use multiple window sizes: short (3-7), medium (14-30), long (60-90+) to capture different temporal scales.
- Combine rolling stats with lags: e.g., rolling mean of lag_7 through lag_13 gives "last week's average" without leakage.
- Always shift rolling features by at least 1 step to avoid including the current target value.

```python
# Safe rolling features (shifted to avoid leakage)
df["rolling_mean_7"] = df["target"].shift(1).rolling(7).mean()
df["rolling_std_7"] = df["target"].shift(1).rolling(7).std()
df["rolling_mean_28"] = df["target"].shift(1).rolling(28).mean()

# With skforecast
from skforecast.preprocessing import RollingFeatures
window_features = RollingFeatures(
    stats=["mean", "std", "min", "max"],
    window_sizes=[7, 14, 28]
)
```

### 1.3 Calendar / Date Features

Encode temporal position to capture seasonality and calendar effects.

**Features to extract:**
- Day of week (0-6)
- Month (1-12)
- Day of month (1-31)
- Week of year (1-52)
- Quarter (1-4)
- Year (for trend, though trees cannot extrapolate -- see Section 2.3)
- Is weekend (binary)
- Is holiday (binary, use `holidays` library)
- Day of year (1-366)
- Hour, minute (for sub-daily data)

**Encoding strategies for tree models:**
- **Integer encoding**: Trees can split on integers directly. Day_of_week=0,1,...,6 works fine for tree models (no need for one-hot).
- **Cyclical (sine/cosine) encoding**: Maps periodic features onto a unit circle. More useful for neural networks; trees can learn periodicity from integer encoding but sine/cosine can help with features that have large cardinality.

```python
# Sine/cosine encoding for cyclical features
import numpy as np
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
```

### 1.4 Fourier Features for Seasonality

Fourier terms model complex seasonal patterns with sine/cosine pairs at different frequencies (harmonics). Each pair captures a different aspect: amplitude (strength), frequency (cycle length), and phase (shift).

**Best practices:**
- Use K Fourier pairs for a season of period m: sin(2*pi*k*t/m) and cos(2*pi*k*t/m) for k=1,...,K
- K controls smoothness: K=1 captures the dominant cycle, higher K captures sharper patterns
- K <= m/2 (Nyquist limit). Typical: K=3-5 for weekly seasonality, K=5-10 for yearly.
- Particularly useful for long-period seasonality where creating 365 dummy variables would be wasteful.
- Risk of overfitting with too many harmonics.

```python
def fourier_features(t, period, n_harmonics):
    features = {}
    for k in range(1, n_harmonics + 1):
        features[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features)

# Weekly seasonality with 3 harmonics
df_fourier = fourier_features(df.index.dayofyear, period=365.25, n_harmonics=5)
```

### 1.5 Target Encoding and Group Statistics

- **Target encoding**: Encode categorical groups (store_id, product_id) with historical target statistics. Must use only past data to avoid leakage.
- **Group aggregations**: Mean/median/std of target per group (store, category, region). Useful for global models.
- **Interaction features**: Price * holiday, temperature * hour_of_day.
- **Difference features**: target_t - target_{t-7} (week-over-week change), target_t - rolling_mean_28 (deviation from trend).

---

## 2. XGBoost / LightGBM / CatBoost for Time Series

### 2.1 Framing Forecasting as Supervised Learning

The core idea: transform the time series into tabular (X, y) pairs using a sliding window.

```
Time series: [v1, v2, v3, v4, v5, v6, v7, v8]

With lags=[1,2,3] and horizon=1:
  X = [v1, v2, v3] -> y = v4
  X = [v2, v3, v4] -> y = v5
  X = [v3, v4, v5] -> y = v6
  ...
```

**Key considerations:**
- The window length (number of lags + features) determines how much history each sample sees.
- Longer windows capture more context but reduce training set size.
- Include exogenous variables (weather, price, promotions) as additional columns.

### 2.2 Model Comparison: XGBoost vs LightGBM vs CatBoost

| Aspect | XGBoost | LightGBM | CatBoost |
|--------|---------|----------|----------|
| Tree growth | Level-wise (depth-first) | Leaf-wise (best-first) | Symmetric (balanced) |
| Speed | Fast | Fastest (often 2-5x faster) | Slower training, fast inference |
| Categorical features | Requires encoding | Native support (optimal split) | Best native support (ordered target stats) |
| Overfitting resistance | Good with regularization | Needs careful tuning (leaf-wise can overfit) | Best out-of-box (ordered boosting) |
| Missing values | Native handling | Native handling | Native handling |
| Memory | Moderate | Lower (histogram-based) | Higher |
| Kaggle popularity | High | Highest for time series | Growing |

**When to choose which:**
- **LightGBM**: Default choice for time series. Fastest training, excellent with large datasets, native categorical support. Dominant in Kaggle forecasting competitions (M5 winner used it).
- **CatBoost**: Best when you have many categorical features (store_id, product_category, day_of_week). Ordered boosting provides built-in protection against temporal leakage. Less hyperparameter tuning needed.
- **XGBoost**: Mature, well-documented, robust. Good baseline. Slightly slower than LightGBM.

### 2.3 The Trend Extrapolation Problem

**Critical limitation**: Tree-based models CANNOT extrapolate trends. They predict by averaging training samples in leaf nodes, so predictions are bounded by the range of training targets. If the test period has higher/lower values than training, trees will systematically under/over-predict.

**Workarounds:**
1. **Differencing**: Model diff(y_t) = y_t - y_{t-1} instead of y_t. Removes trend. Reverse the transform at prediction time.
2. **Detrending**: Fit a linear/polynomial trend, model residuals with the tree, add trend back at prediction.
3. **Target normalization per window**: Divide by rolling mean or use percent-change features.
4. **Include trend-aware features**: Time index, cumulative sums, days since event.
5. **Hybrid models**: Combine a linear model (for trend) with a tree model (for nonlinear patterns).

```python
# Differencing approach
df["target_diff"] = df["target"].diff()
# Train model on target_diff
# At prediction: y_hat_t = y_{t-1} + diff_hat_t

# Detrending approach
from sklearn.linear_model import LinearRegression
trend_model = LinearRegression()
trend_model.fit(df[["time_index"]], df["target"])
df["detrended"] = df["target"] - trend_model.predict(df[["time_index"]])
# Train tree on detrended, add trend back at prediction
```

### 2.4 Key Hyperparameters for Time Series

**LightGBM recommended starting point:**
```python
params = {
    "n_estimators": 1000,       # Use early stopping
    "learning_rate": 0.05,      # Lower = more trees needed but better
    "num_leaves": 31,           # Default; reduce for regularization
    "max_depth": -1,            # No limit (leaf-wise handles this)
    "min_child_samples": 20,    # Increase for noisy data
    "subsample": 0.8,           # Row subsampling
    "colsample_bytree": 0.8,    # Feature subsampling
    "reg_alpha": 0.1,           # L1 regularization
    "reg_lambda": 0.1,          # L2 regularization
    "random_state": 42,
    "verbose": -1,
}
```

---

## 3. Temporal Cross-Validation

### 3.1 Why Random CV is Wrong

Random k-fold CV shuffles data, destroying temporal order. This causes:
- **Future leakage**: Training folds contain data from after the test fold.
- **Overly optimistic metrics**: Model "sees" patterns that won't be available at inference.
- **Broken autocorrelation**: Nearby time points (which are correlated) end up in different folds.

**Rule**: Never shuffle time series data for cross-validation.

### 3.2 Expanding Window (Stretching Window)

The training set grows with each fold. The first fold uses minimal history; the last uses almost everything.

```
Fold 1: [=====Train=====][Test]
Fold 2: [========Train========][Test]
Fold 3: [===========Train===========][Test]
Fold 4: [==============Train==============][Test]
```

**Pros**: Uses all available history, mimics real deployment (you retrain on all data).
**Cons**: Early folds have little training data; model quality varies across folds. Favors stability over recency.

### 3.3 Sliding Window (Rolling Window)

Fixed-size training window slides forward.

```
Fold 1: [=====Train=====][Test]
Fold 2:    [=====Train=====][Test]
Fold 3:       [=====Train=====][Test]
Fold 4:          [=====Train=====][Test]
```

**Pros**: Each fold has the same amount of training data; more responsive to recent patterns; detects concept drift.
**Cons**: Discards older data that might be useful; must choose window size carefully.

**Guidance**: Use expanding window when the data-generating process is stable. Use sliding window when recent patterns matter more (e.g., fast-changing markets).

### 3.4 Gap Between Train and Validation

When your forecast horizon is h steps, insert a gap of h between training and validation sets. This ensures you never evaluate on data that would overlap with the last training observation's prediction horizon.

```
Fold: [=====Train=====]---gap=h---[Test]
```

### 3.5 Purging and Embargoing (Financial ML)

Developed by Marcos Lopez de Prado for financial time series where labels depend on future returns.

- **Purging**: Remove any training observation whose label horizon overlaps with the test period. Prevents leakage when labels are computed over a future window (e.g., 5-day forward return).
- **Embargoing**: After each test period, remove a buffer of observations from the training set. Prevents leakage from autocorrelated features and delayed market reactions.

```python
# Conceptual implementation
def purged_cv_split(X, y, n_splits, embargo_pct=0.01, label_horizon=5):
    splits = []
    for test_start, test_end in time_folds(X, n_splits):
        # Purge: remove training samples whose labels overlap test
        train_mask = (X.index + label_horizon < test_start) | \
                     (X.index > test_end)
        # Embargo: remove buffer after test
        embargo_end = test_end + int(len(X) * embargo_pct)
        train_mask &= ~((X.index > test_end) & (X.index <= embargo_end))
        splits.append((X.index[train_mask], X.index[test_start:test_end]))
    return splits
```

### 3.6 Practical Implementation with skforecast

```python
from skforecast.model_selection import backtesting_forecaster

# Expanding window backtesting
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=data["target"],
    initial_train_size=len(data_train),
    fixed_train_size=False,  # Expanding window
    steps=24,                # Forecast horizon
    metric="mean_absolute_error",
    refit=True,              # Retrain at each fold
)

# Sliding window: set fixed_train_size=True
```

---

## 4. Multi-Step Forecasting Strategies

### 4.1 Recursive (Iterated) Strategy

Train a single 1-step-ahead model. For multi-step, feed predictions back as inputs.

```
Step 1: y_hat_{t+1} = f(y_t, y_{t-1}, ...)
Step 2: y_hat_{t+2} = f(y_hat_{t+1}, y_t, ...)
Step 3: y_hat_{t+3} = f(y_hat_{t+2}, y_hat_{t+1}, ...)
```

**Pros:**
- Only one model to train and maintain.
- Preserves temporal dependencies between steps.
- Works well for short horizons.

**Cons:**
- Error accumulation: prediction errors compound at each step.
- Slow at inference (sequential).
- Model never trained on noisy inputs (train-test mismatch).

**Best for**: Short horizons (1-5 steps), stable series, when you want simplicity.

### 4.2 Direct Strategy

Train H separate models, one per forecast step.

```
Model 1: y_hat_{t+1} = f1(y_t, y_{t-1}, ...)
Model 2: y_hat_{t+2} = f2(y_t, y_{t-1}, ...)
Model H: y_hat_{t+H} = fH(y_t, y_{t-1}, ...)
```

**Pros:**
- No error propagation between steps.
- Each model can learn horizon-specific patterns (important: what predicts 1-step-ahead differs from what predicts 28-step-ahead).
- Can be parallelized.

**Cons:**
- H models to train, tune, and maintain.
- Ignores dependencies between forecast steps.
- More data needed (H times the computation).

**Best for**: Longer horizons, when accuracy per step matters, Kaggle competitions (M5 winner used this).

### 4.3 DirRec (Direct-Recursive) Strategy

Train H models, but each model also receives predictions from previous steps as features.

```
Model 1: y_hat_{t+1} = f1(y_t, y_{t-1}, ...)
Model 2: y_hat_{t+2} = f2(y_t, y_{t-1}, ..., y_hat_{t+1})
Model 3: y_hat_{t+3} = f3(y_t, y_{t-1}, ..., y_hat_{t+1}, y_hat_{t+2})
```

**Pros**: Captures inter-step dependencies + no error propagation in the direct component.
**Cons**: Most complex to implement; still requires H models; sequential at inference.

### 4.4 MIMO (Multi-Input Multi-Output) Strategy

Single model outputs all H steps simultaneously.

```
[y_hat_{t+1}, ..., y_hat_{t+H}] = f(y_t, y_{t-1}, ...)
```

For XGBoost/LightGBM, this is implemented with `MultiOutputRegressor` wrapping or by using native multi-output support.

```python
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor

model = MultiOutputRegressor(LGBMRegressor())
# X shape: (n_samples, n_features)
# y shape: (n_samples, H) -- H target columns
model.fit(X_train, y_train)
y_pred = model.predict(X_test)  # shape: (n_test, H)
```

**Pros**: Preserves dependencies between steps; single model; fast inference.
**Cons**: Some studies show MIMO produces lower RMSE than other strategies; however, the collective evidence is inconclusive across all domains.

### 4.5 Practical Recommendation

| Scenario | Recommended Strategy |
|----------|---------------------|
| Short horizon (1-5 steps) | Recursive |
| Medium horizon (5-30 steps) | Direct (one model per horizon) |
| Many series, scalability needed | Recursive with global model |
| Kaggle / maximum accuracy | Direct or DirRec with ensemble |
| Real-time / low latency | MIMO |

---

## 5. Global Models vs Local Models

### 5.1 Definitions

- **Local model**: One model per time series. Traditional approach (ARIMA, ETS per series).
- **Global model**: One model trained on ALL time series simultaneously. Series identity encoded as a feature.

### 5.2 When Global Models Win

- **Many short series**: If each series has <100 observations, local models overfit. A global model pools information across series. This is the most common real-world scenario (thousands of products, each with 1-2 years of weekly data).
- **Related series**: Series sharing similar dynamics (same product category, same region) benefit from cross-learning.
- **Cold-start problem**: New series with no history can still be forecasted using patterns from similar series.
- **Scalability**: One model for 10,000+ series vs. 10,000 individual models. Orders of magnitude faster.
- **Non-linear methods**: Tree-based global models handle heterogeneity well by learning to split on series-level features.

### 5.3 When Local Models Win

- **Highly heterogeneous series**: If series have fundamentally different dynamics and you have enough data per series.
- **Abundant data per series**: >500+ observations per series with unique patterns.
- **Interpretability needs**: Easier to explain one model for one series.

### 5.4 Implementation Pattern for Global Models

```python
import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast

# Data format: long format with unique_id, ds (date), y (target)
# unique_id identifies each individual time series

fcst = MLForecast(
    models=[LGBMRegressor(n_estimators=500, learning_rate=0.05)],
    freq="D",
    lags=[1, 7, 14, 28],
    lag_transforms={
        7: [RollingMean(window_size=7), RollingStd(window_size=7)],
        28: [RollingMean(window_size=28)],
    },
    date_features=["dayofweek", "month"],
    num_threads=4,
)

fcst.fit(df)
forecasts = fcst.predict(h=28)
```

**Key feature engineering for global models:**
- Series identifier (as categorical feature for LightGBM/CatBoost).
- Group-level aggregations (mean target per store, per category).
- Static features (store size, location, product category).

### 5.5 Hybrid Approach

The best practical approach is often a global model with per-series features:
- Train one global model on all series.
- Include series_id as a categorical feature.
- Add group-level statistics as features.
- Optionally: train global + local models and ensemble their predictions.

---

## 6. Kaggle Competition Insights

### 6.1 M5 Competition (Walmart Sales)

The M5 was the most influential recent time series competition, with 30,490 time series of daily Walmart sales.

**Winning solution characteristics:**
- **Ensemble of 6 models**: LightGBM + XGBoost + feedforward neural networks, equal-weighted average.
- **One model per forecast horizon**: Separate models for day 1, day 2, ..., day 28. This was a key innovation allowing each model to learn horizon-specific patterns.
- **Global modeling**: All 30,490 series in one model (cross-learning).
- **Feature engineering was king**:
  - Lag features: lag_7, lag_14, lag_21, lag_28 (weekly multiples)
  - Rolling statistics: rolling_mean_7, rolling_mean_28, rolling_std_7
  - Calendar features: day_of_week, month, event indicators
  - Price features: price, price_change, price relative to category mean
  - Encoding: department_id, category_id, store_id, state_id as categoricals
- **LightGBM dominated**: Used by the vast majority of top solutions.
- **Simple ensembles worked best**: Equal-weighted averaging outperformed complex stacking.

### 6.2 Recurring Kaggle Patterns

From analysis of multiple winning solutions across forecasting competitions:

1. **Feature engineering > model complexity**: Most winning solutions use 50-200 carefully crafted features with LightGBM, not complex architectures.
2. **Validation strategy is critical**: Winners spend significant time designing proper temporal validation before tuning models.
3. **Hierarchical reconciliation**: When data has hierarchy (store > department > product), ensure forecasts are coherent across levels.
4. **External data matters**: Weather, holidays, promotions, economic indicators consistently improve accuracy.
5. **Training data recency weighting**: Recent observations weighted more heavily (via sample_weight or time-based subsampling).
6. **Post-processing**: Clipping negative forecasts to zero (for demand), rounding to integers, applying business rules.

### 6.3 Typical Top Solution Architecture

```
Raw data
  -> Feature engineering (60% of effort)
     - Lags (7, 14, 21, 28 for daily)
     - Rolling stats (mean, std over 7, 14, 28, 60 windows)
     - Calendar (dow, month, holiday, event type)
     - Price features (current, lagged, relative)
     - Category encodings (target mean per group)
  -> Multiple models
     - LightGBM (primary)
     - XGBoost (diversity)
     - Neural network (captures different patterns)
  -> Per-horizon training (28 models for 28-day forecast)
  -> Simple ensemble (average or weighted average)
  -> Post-processing (clip, round, reconcile)
```

---

## 7. Common Pitfalls

### 7.1 Target Leakage Through Feature Engineering

**The problem**: Including information that would not be available at prediction time.

**Common mistakes:**
- Using `rolling_mean` that includes the current observation: `df["target"].rolling(7).mean()` at time t includes y_t.
- Using future values in group statistics: computing "average sales per store" using the entire dataset (including future).
- Using exogenous variables that won't be known at forecast time (e.g., actual weather for future dates).

**Fix**: Always shift features by at least 1 (or by the forecast horizon h).
```python
# WRONG: includes y_t in the rolling window
df["rolling_mean_7"] = df["target"].rolling(7).mean()

# CORRECT: excludes y_t
df["rolling_mean_7"] = df["target"].shift(1).rolling(7).mean()

# MOST CORRECT for horizon h=7: excludes all unavailable values
df["rolling_mean_7"] = df["target"].shift(7).rolling(7).mean()
```

### 7.2 Using Future Information in Rolling Features

**The problem**: Computing rolling statistics using the entire series before train/test split.

**Example**: Normalizing by the global mean/std of the entire series (including test period).

**Fix**: Compute all statistics using only training data. Recompute for each cross-validation fold.

```python
# WRONG: normalizing with full-series statistics
mean = df["target"].mean()  # includes test data!
std = df["target"].std()
df["normalized"] = (df["target"] - mean) / std

# CORRECT: normalize with training statistics only
train_mean = df.loc[:split_date, "target"].mean()
train_std = df.loc[:split_date, "target"].std()
df["normalized"] = (df["target"] - train_mean) / train_std
```

Similarly, **decomposition leakage**: applying STL decomposition, EMD, or wavelet transforms to the entire series before splitting leaks future structure into training features.

### 7.3 Not Accounting for Prediction Horizon in Validation

**The problem**: Validating on the next observation when you actually need to predict 28 days ahead.

**Fix**: Your validation setup must match your production forecast horizon. If you predict 28 days ahead, your validation test set must be 28 days, and training must stop 28 days before it.

### 7.4 Overfitting to Recent Patterns

**The problem**: Model learns recent regime perfectly but fails when patterns shift.

**Signs**: Great validation performance on the last fold but poor on earlier folds; performance degrades when deployed.

**Fixes:**
- Use multiple CV folds spanning different time periods.
- Regularize aggressively (lower num_leaves, higher min_child_samples).
- Use sample weighting that doesn't completely ignore older data.
- Monitor feature importance: if a feature only works in recent data, it may be spurious.

### 7.5 Improper Data Splitting

**The problem**: Using random train/test splits for time series.

**Fix**: Always split chronologically. Never shuffle. Use temporal CV (Section 3).

### 7.6 Ignoring the Stationarity Requirement

**The problem**: Tree models assume the relationship between features and target is stable. Strong trends violate this.

**Fix**: Detrend (Section 2.3), difference, or use ratio/percent-change features.

### 7.7 Not Handling Missing Values and Zeros

- Intermittent demand (many zeros) requires special treatment (Croston's method or zero-inflated models as preprocessing).
- Missing values in lag features at series boundaries need careful imputation or masking.

---

## 8. Libraries and Tools

### 8.1 Core Gradient Boosting Libraries

| Library | Best For | Key Feature |
|---------|----------|-------------|
| **XGBoost** | General purpose, robust baseline | Mature, GPU support, well-documented |
| **LightGBM** | Large datasets, speed-critical | Fastest training, native categoricals, leaf-wise growth |
| **CatBoost** | Many categoricals, minimal tuning | Ordered boosting (temporal leakage protection), best categorical handling |
| **HistGradientBoosting** (sklearn) | Simple pipelines, no extra deps | Built into scikit-learn, fast histogram-based |

### 8.2 Time Series ML Frameworks

| Library | Strengths | Weaknesses | Best For |
|---------|-----------|------------|----------|
| **MLforecast** (Nixtla) | Fastest for many series, automated feature engineering, scales to millions of series with Spark/Dask/Ray | Less statistical rigor | Production at scale, Kaggle |
| **skforecast** | Excellent docs, backtesting built-in, scikit-learn compatible | Slower than MLforecast for very large datasets | Learning, research, medium-scale |
| **sktime** | Academic gold standard, unified API for classification/forecasting/clustering, composable pipelines | Steeper learning curve | Research papers, complex pipelines |
| **Darts** | Bridges statistical + ML models, probabilistic forecasting, multivariate | Heavier dependency footprint | When mixing ARIMA + LightGBM |
| **tslearn** | Time series classification and clustering | Not focused on forecasting | DTW, time series similarity |

### 8.3 MLforecast Quick Start

```python
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd, ExpandingMean
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# Define forecaster with automated feature engineering
fcst = MLForecast(
    models={
        "lgbm": LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31),
        "xgb": XGBRegressor(n_estimators=500, learning_rate=0.05),
    },
    freq="D",
    lags=[1, 7, 14, 28],
    lag_transforms={
        1: [ExpandingMean()],
        7: [RollingMean(window_size=7), RollingStd(window_size=7)],
        28: [RollingMean(window_size=28)],
    },
    date_features=["dayofweek", "month", "year"],
    num_threads=6,
)

# Fit on all series at once (global model)
fcst.fit(df, id_col="unique_id", time_col="ds", target_col="y")

# Predict
forecasts = fcst.predict(h=28)

# Cross-validation
from mlforecast.utils import PredictionIntervals
cv_results = fcst.cross_validation(
    df,
    h=28,
    n_windows=3,       # Number of CV folds
    step_size=28,       # Gap between folds
)
```

### 8.4 skforecast Quick Start

```python
from skforecast.recursive import ForecasterRecursive
from skforecast.preprocessing import RollingFeatures
from skforecast.model_selection import backtesting_forecaster, grid_search_forecaster
from lightgbm import LGBMRegressor

# Define window features
window_features = RollingFeatures(
    stats=["mean", "std", "min", "max"],
    window_sizes=[24, 48, 168]  # 1 day, 2 days, 1 week (hourly)
)

# Create and train forecaster
forecaster = ForecasterRecursive(
    regressor=LGBMRegressor(random_state=42, verbose=-1),
    lags=48,
    window_features=window_features,
)
forecaster.fit(y=train_series)

# Backtest
metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=full_series,
    initial_train_size=len(train_series),
    fixed_train_size=False,  # Expanding window
    steps=24,
    metric="mean_absolute_error",
    refit=True,
)
```

---

## 9. Summary: Decision Flowchart

```
START
  |
  v
How many time series?
  |
  +-- 1-10 series --> Local models (one per series)
  |                     Use skforecast or manual XGBoost/LightGBM
  |
  +-- 10-1000 series --> Global model with series_id feature
  |                        Use MLforecast or skforecast multi-series
  |
  +-- 1000+ series --> Global model, MLforecast with Spark/Dask
  |
  v
What forecast horizon?
  |
  +-- 1-5 steps --> Recursive strategy
  +-- 5-30 steps --> Direct (one model per horizon) or MIMO
  +-- 30+ steps --> Direct + ensemble
  |
  v
Is there a trend?
  |
  +-- Yes --> Detrend or difference first
  +-- No --> Use raw target
  |
  v
Validation strategy:
  - Use temporal CV (expanding or sliding window)
  - Gap = forecast horizon between train and test
  - Minimum 3-5 folds
  |
  v
Feature engineering (60% of effort):
  1. Lags at key periodicities
  2. Rolling stats (mean, std) at multiple window sizes
  3. Calendar features (dow, month, holiday)
  4. Exogenous variables (if available)
  5. Group statistics (for global models)
  6. Fourier terms (for complex seasonality)
  |
  v
Model: LightGBM (default) or CatBoost (many categoricals)
  - Start with default hyperparameters
  - Use early stopping with temporal validation
  - Regularize: reduce num_leaves, increase min_child_samples
  |
  v
Ensemble (if accuracy is critical):
  - Train LightGBM + XGBoost + CatBoost
  - Simple average of predictions
  - Optionally add a neural network for diversity
```

---

## Sources

### Feature Engineering
- [Practical Guide for Feature Engineering of Time Series Data](https://dotdata.com/blog/practical-guide-for-feature-engineering-of-time-series-data/)
- [6 Powerful Feature Engineering Techniques for Time Series](https://www.analyticsvidhya.com/blog/2019/12/6-powerful-feature-engineering-techniques-time-series/)
- [Lag and Rolling Features: A Complete Guide](https://www.analyticsvidhya.com/blog/2026/02/lag-and-rolling-features/)
- [Feature Engineering for Time-Series Data - GeeksforGeeks](https://www.geeksforgeeks.org/data-analysis/feature-engineering-for-time-series-data-methods-and-applications/)

### Gradient Boosting for Time Series
- [Forecasting with Skforecast, XGBoost, LightGBM, CatBoost](https://cienciadedatos.net/documentos/py39-forecasting-time-series-with-skforecast-xgboost-lightgbm-catboost.html)
- [Multi-step Time Series Forecasting with XGBoost (TDS)](https://towardsdatascience.com/multi-step-time-series-forecasting-with-xgboost-65d6820bec39/)
- [XGBoost for Time Series - MachineLearningMastery](https://machinelearningmastery.com/xgboost-for-time-series-forecasting/)
- [Overcoming Limitations of Tree-Based Models in TS Forecasting](https://medium.com/@simon.peter.mueller/overcoming-the-limitations-of-tree-based-models-in-time-series-forecasting-c2c5bd71a8f1)
- [CatBoost for Accurate Time-Series Predictions](https://aicompetence.org/catboost-for-accurate-time-series-predictions/)

### Temporal Cross-Validation
- [Cross Validation in Finance: Purging, Embargoing, Combinatorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/)
- [Time Series Cross-Validation - Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html)
- [Validation Methods for Time Series (Kaggle)](https://www.kaggle.com/code/konradb/ts-10-validation-methods-for-time-series)

### Multi-Step Forecasting
- [A Review and Comparison of Strategies for Multi-Step Ahead TS Forecasting (Taieb et al.)](https://arxiv.org/pdf/1108.3259)
- [Recursive and Direct Multi-Step Forecasting (Hyndman)](https://robjhyndman.com/papers/rectify.pdf)
- [6 Methods for Multi-step Forecasting (TDS)](https://towardsdatascience.com/6-methods-for-multi-step-forecasting-823cbde4127a/)
- [Stratify: Unifying Multi-Step Forecasting Strategies](https://arxiv.org/html/2412.20510v1)

### Global vs Local Models
- [Local vs Global Forecasting: What You Need to Know (TDS)](https://medium.com/data-science/local-vs-global-forecasting-what-you-need-to-know-1cc29e66cae0)
- [Principles and Algorithms for Forecasting Groups of Time Series (Hyndman)](https://robjhyndman.com/publications/global-forecasting/)
- [Global Forecasting Models: Comparative Analysis](https://cienciadedatos.net/documentos/py53-global-forecasting-models)
- [When and How to Use Global Forecasting Methods on Heterogeneous Datasets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4629272)

### Kaggle Competitions
- [Learnings from Kaggle's Forecasting Competitions (paper)](https://arxiv.org/pdf/2009.07701)
- [Chapter 8: Winningest Methods in TS Forecasting (M5 Handbook)](https://phdinds-aim.github.io/time_series_handbook/08_WinningestMethods/lightgbm_m5_forecasting.html)
- [M5 Accuracy Competition Results (paper)](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf)

### Pitfalls and Leakage
- [Avoiding Data Leakage in Timeseries 101 (TDS)](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/)
- [Look-Ahead Bias in Rolling Window Features](https://www.mhtechin.com/support/look-ahead-bias-in-rolling-window-features/)
- [Common Mistakes in Time-Series Forecasting](https://medium.com/@gayatrishetti1/common-mistakes-in-time-series-forecasting-and-how-to-avoid-them-56e9a33a987d)
- [Data Leakage in Time-Dependent Feature Engineering](https://bitpeak.com/data-leakage-in-time-dependent-feature-engineering/)

### Libraries
- [MLforecast - Nixtla](https://nixtlaverse.nixtla.io/mlforecast/index.html)
- [skforecast Documentation](https://skforecast.org/)
- [sktime - Unified ML Framework for Time Series](https://github.com/sktime/sktime)
- [Darts vs Sktime vs MLforecast Comparison](https://piotrpomorski.substack.com/p/darts-vs-sktime-vs-mlforecast-a-no)

### Fourier Features
- [Encoding Cyclical Features in Time Series](https://mlpills.substack.com/p/issue-89-encoding-cyclical-features)
- [Understanding Fourier Terms for Seasonality](https://python.plainenglish.io/understanding-fourier-terms-for-seasonality-in-time-series-analysis-a43b4ddbfd9e)
- [Cyclical Features in Time Series - skforecast](https://skforecast.org/latest/faq/cyclical-features-time-series.html)
