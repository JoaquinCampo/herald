# Classical Statistical Methods for Time Series Forecasting: Best Practices

A comprehensive research report covering methods, parameters, pitfalls, and Python implementations.

---

## Table of Contents

1. [ARIMA / SARIMA](#1-arima--sarima)
2. [Exponential Smoothing (ETS)](#2-exponential-smoothing-ets)
3. [Theta Method](#3-theta-method)
4. [STL Decomposition](#4-stl-decomposition)
5. [Stationarity and Preprocessing](#5-stationarity-and-preprocessing)
6. [Key Principles from Hyndman's fpp3](#6-key-principles-from-hyndmans-fpp3)
7. [Classical vs Modern Methods: When to Use What](#7-classical-vs-modern-methods-when-to-use-what)

---

## 1. ARIMA / SARIMA

### 1.1 What It Is

ARIMA (AutoRegressive Integrated Moving Average) models the autocorrelation structure of a **stationary** time series. The model is parameterized by three integers:

- **p** (AR order): number of lagged observations used as predictors. An AR process shows gradual decay in the ACF and sharp cutoff in the PACF.
- **d** (differencing order): number of times the series is differenced to achieve stationarity.
- **q** (MA order): size of the moving-average "window" over past forecast errors. An MA process shows sharp cutoff in the ACF and gradual decay in the PACF.

SARIMA extends this with seasonal terms `(P, D, Q, m)` where `m` is the seasonal period.

### 1.2 When to Use

- Data exhibits autocorrelation structure (check ACF/PACF plots).
- Series is or can be made stationary through differencing.
- No strong nonlinear patterns.
- Works well for short-to-medium term forecasts.
- Complements ETS: ARIMA focuses on autocorrelation, ETS on trend/seasonality description.

### 1.3 Identifying (p, d, q) Orders

**Manual approach (Box-Jenkins method):**

1. **Determine d**: Use unit root tests (ADF, KPSS) or the `ndiffs()` function. Start with d=0; if non-stationary, try d=1. Rarely need d=2.
2. **Examine ACF/PACF of differenced series**:
   - ACF cuts off at lag q, PACF decays -> MA(q) model
   - PACF cuts off at lag p, ACF decays -> AR(p) model
   - Both decay -> ARMA(p,q) needed
3. **For seasonal terms**: Examine ACF/PACF at seasonal lags (12, 24, 36... for monthly data).

**Automatic approach (auto_arima):**

The Hyndman-Khandakar (2008) stepwise algorithm:

1. Starts with four candidate models including ARIMA(2,d,2), ARIMA(0,d,0), ARIMA(1,d,0), ARIMA(0,d,1).
2. Iteratively varies p, q, P, Q by +/-1 from current best.
3. Selects the model minimizing an information criterion (AICc by default).
4. Stepwise search is much faster than brute-force grid search and less likely to overfit.

### 1.4 Key Parameters and Settings

| Parameter | Guidance |
|-----------|----------|
| `d` | Use `ndiffs()` with KPSS test. Almost never > 2. |
| `D` | Use `nsdiffs()` for seasonal differencing. Apply seasonal diff first. |
| `p, q` | Typically 0-5. Let auto_arima search. |
| `P, Q` | Typically 0-2 for seasonal terms. |
| Information criterion | AICc preferred for small samples, BIC for parsimony. |
| `stepwise` | True (default) for speed; False for exhaustive search on critical applications. |

### 1.5 Common Mistakes

1. **Not testing stationarity first.** Always run ADF + KPSS before modeling.
2. **Over-differencing.** If lag-1 autocorrelation of differenced series < -0.5, you've over-differenced. Check that d is minimal.
3. **Ignoring seasonal differencing.** Apply seasonal differencing before regular differencing; the result is sometimes already stationary.
4. **Blindly trusting auto_arima.** Always check residual diagnostics (Ljung-Box test, ACF of residuals, normality).
5. **Using ARIMA on trended data without differencing.** The "I" in ARIMA is there for a reason.
6. **Applying to very short series** (< 50 observations) without caution -- parameter estimates become unreliable.

### 1.6 Python Libraries

| Library | Function | Notes |
|---------|----------|-------|
| `statsforecast` (Nixtla) | `ARIMA()`, `AutoARIMA()` | Fast, Hyndman-compatible, production-ready |
| `pmdarima` | `auto_arima()` | Python port of R's forecast::auto.arima |
| `statsmodels` | `SARIMAX()` | Full-featured, more manual control |

```python
# statsforecast (recommended for speed)
from statsforecast.models import AutoARIMA
from statsforecast import StatsForecast

sf = StatsForecast(models=[AutoARIMA(season_length=12)], freq="MS")
forecast = sf.forecast(df=df, h=12)

# pmdarima
import pmdarima as pm
model = pm.auto_arima(y, seasonal=True, m=12, stepwise=True,
                      information_criterion='aicc')

# statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,12))
results = model.fit()
```

---

## 2. Exponential Smoothing (ETS)

### 2.1 What It Is

Exponential smoothing produces forecasts as **weighted averages of past observations**, with weights decaying exponentially as observations get older. The ETS framework (Error, Trend, Seasonality) provides a unified taxonomy of 30 models (15 methods x 2 error types):

**Trend component options:**
- N (None)
- A (Additive / linear)
- Ad (Additive damped)

**Seasonal component options:**
- N (None)
- A (Additive)
- M (Multiplicative)

**Error type options:**
- A (Additive)
- M (Multiplicative)

This gives models like ETS(A,A,A) = additive errors, additive trend, additive seasonality.

### 2.2 The Three Main Methods

**Simple Exponential Smoothing (SES) -- ETS(*,N,N):**
- For data with no trend and no seasonality.
- One parameter: alpha (level smoothing, 0 < alpha < 1).
- Forecast equation: y_hat = l_t (flat forecast).
- Closer alpha is to 1, more weight on recent observations ("rougher"); closer to 0, smoother.

**Holt's Linear Method -- ETS(*,A,N):**
- For data with trend but no seasonality.
- Two parameters: alpha (level) and beta* (trend).
- Produces linear trend forecasts that extend indefinitely.

**Holt-Winters Method -- ETS(*,A,A) or ETS(*,A,M):**
- For data with both trend and seasonality.
- Three parameters: alpha (level), beta* (trend), gamma (seasonal).
- **Additive seasonality**: when seasonal variations are roughly constant in magnitude.
- **Multiplicative seasonality**: when seasonal variations scale proportionally with the level.

### 2.3 Damped Trends

The **damped trend** variant (Ad) adds a parameter phi (0 < phi < 1) that dampens the trend toward a flat line over the forecast horizon. This is one of the most important practical innovations:

- **Linear trends** extrapolate forever, often producing unrealistic long-range forecasts.
- **Damped trends** converge to a flat line, which is almost always more realistic.
- Gardner and McKenzie (1985) showed damped trends outperform linear trends across many series.
- phi near 1.0 = almost linear; phi near 0.8 = strong damping.
- **Default recommendation**: use damped trend unless you have strong reason for linear extrapolation.

### 2.4 Model Selection

ETS model selection uses information criteria (AICc, BIC) computed from the likelihood:

- AIC = -2 * log(L) + 2k
- AICc = AIC + 2k(k+1) / (T-k-1)
- BIC = AIC + k[log(T) - 2]

**Important constraints:**
- **Avoid ETS(A,*,M)** models: additive errors + multiplicative seasonality causes numerical instability (division by near-zero values). Prefer ETS(M,*,M).
- **Multiplicative error models** require strictly positive data.
- When data contain zeros or negatives, only the six fully additive models are considered.

### 2.5 When ETS Beats ARIMA

- Data with clear, describable trend and seasonal patterns.
- Shorter series where parsimonious models are preferable.
- When you want automatic, fast, reliable forecasts at scale.
- ETS is often the default "workhorse" in competition benchmarks.
- In practice: **try both ETS and ARIMA, pick by AICc or cross-validation**.

### 2.6 Common Mistakes

1. **Using additive seasonality when multiplicative is appropriate** (and vice versa). Check if seasonal amplitude grows with level.
2. **Not using damped trends.** Undamped linear trends are rarely justified for multi-step forecasts.
3. **Insufficient data for seasonal models.** Need at least 2 full seasonal cycles (e.g., 24 months for monthly data with annual seasonality).
4. **Not preprocessing.** Handle missing values (forward fill or interpolation) and outliers before fitting.
5. **Ignoring multiplicative error advantages** for positive data with heteroscedastic errors.

### 2.7 Python Libraries

| Library | Function | Notes |
|---------|----------|-------|
| `statsforecast` | `AutoETS()` | Fast, automatic model selection |
| `statsmodels` | `ETSModel()` | Full state-space implementation |

```python
# statsforecast
from statsforecast.models import AutoETS
sf = StatsForecast(models=[AutoETS(season_length=12)], freq="MS")

# statsmodels
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
model = ETSModel(y, error="add", trend="add", seasonal="add",
                 damped_trend=True, seasonal_periods=12)
results = model.fit()
```

---

## 3. Theta Method

### 3.1 What It Is

The Theta method was proposed by Assimakopoulos and Nikolopoulos (2000) and achieved remarkable success in the **M3 Competition** (2000), outperforming far more complex methods on 3,003 real time series.

The method works by decomposing a time series into **theta lines** using a coefficient theta applied to the second differences:

1. **Theta = 0 line**: A straight line (linear regression through the data) -- captures long-term behavior.
2. **Theta = 2 line**: Amplifies local curvatures -- captures short-term dynamics.
3. The theta=0 line is extrapolated via linear regression; the theta=2 line via **Simple Exponential Smoothing (SES)**.
4. Forecasts from both lines are **averaged** (simple combination).

### 3.2 The Key Insight

Hyndman and Billah (2003) proved in "Unmasking the Theta Method" that the Theta method, as implemented in M3, is **mathematically equivalent to Simple Exponential Smoothing (SES) with drift**:

- The drift equals **half the slope of a linear trend** fitted to the data.
- This dramatically simplifies understanding and implementation.

### 3.3 Why It Won M3

1. **Simplicity**: It combines only two components -- a trend line and SES.
2. **Combination advantage**: Even simple combinations of forecasts tend to outperform individual methods (the "wisdom of crowds" for forecasting).
3. **Robustness**: The method is resistant to overfitting because it has very few parameters.
4. **Implicit shrinkage**: Averaging a linear trend with SES naturally dampens extreme forecasts.

### 3.4 When to Use

- As a strong **baseline** method.
- For non-seasonal data (apply seasonal adjustment first for seasonal data).
- When you want a simple, robust method with minimal tuning.
- When forecasting many series at scale (minimal computation).
- Particularly good for annual and microeconomic data (M3 results).

### 3.5 Common Mistakes

1. **Applying to seasonal data without deseasonalizing first.** The standard Theta method assumes non-seasonal input; use multiplicative classical decomposition to deseasonalize.
2. **Over-complicating it.** The beauty of Theta is its simplicity. Resist adding complexity.
3. **Not using it as a benchmark.** Given its M3 performance, it should always be in your comparison set.

### 3.6 Python Libraries

| Library | Function | Notes |
|---------|----------|-------|
| `statsforecast` | `Theta()`, `OptimizedTheta()`, `DynamicTheta()`, `DynamicOptimizedTheta()` | Multiple variants |
| `sktime` | `ThetaForecaster()` | Scikit-learn compatible |
| `statsmodels` | `ThetaModel()` | Basic implementation |

```python
from statsforecast.models import Theta, OptimizedTheta
sf = StatsForecast(
    models=[Theta(season_length=12), OptimizedTheta(season_length=12)],
    freq="MS"
)
```

---

## 4. STL Decomposition

### 4.1 What It Is

**STL** = Seasonal and Trend decomposition using **LOESS** (Locally Estimated Scatterplot Smoothing). Developed by Cleveland et al. (1990), it decomposes a time series into three components:

```
Y_t = T_t + S_t + R_t
```

Where T_t is trend-cycle, S_t is seasonal, and R_t is remainder.

**MSTL** (Multiple Seasonal-Trend Decomposition using LOESS) extends STL to handle **multiple seasonal patterns** (e.g., daily + weekly + annual seasonality in hourly data). Developed by Bandara et al. (2021), it iteratively applies STL to extract each seasonal component.

### 4.2 Advantages Over Classical Decomposition

- Handles **any type of seasonality** (not just monthly/quarterly like X-11 and SEATS).
- Seasonal component **can change over time** (controlled by user).
- **Robust to outliers** when `robust=True` -- outliers only affect the remainder, not trend/seasonal estimates.
- User controls smoothness of both trend and seasonal components.

### 4.3 Disadvantages

- Only provides **additive** decompositions directly. For multiplicative: take logs first, decompose, then back-transform.
- Does not handle **trading day or calendar variation** automatically.

### 4.4 Key Parameters

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `season(window=?)` | Controls how rapidly seasonality can change | Larger = more stable seasonality. `"periodic"` = fixed seasonality |
| `trend(window=?)` | Controls smoothness of trend-cycle | Larger = smoother trend. Should be odd number, > 1.5 * period / (1 - 1.5/season_window) |
| `robust` | Use robust fitting (bisquare weights) | Set `True` when outliers present |

**Rules of thumb:**
- Seasonal window = 7 is a common default; "periodic" for fixed seasonality.
- Trend window should be at least 1.5x the seasonal period.
- Always set `robust=True` if you suspect outliers.

### 4.5 Forecasting with STL

STL itself is a **decomposition** method, not a forecaster. The forecasting workflow is:

1. Decompose using STL.
2. Forecast the **seasonally adjusted series** (trend + remainder) using any method (ARIMA, ETS, etc.).
3. Add back the seasonal component (using the last seasonal cycle, or forecasted seasonal).

This is implemented in `STLForecast` in statsmodels and `MSTL` with downstream models in statsforecast.

### 4.6 When to Decompose Before Forecasting

- When you want to forecast trend and seasonality separately.
- When seasonality is complex or evolving.
- When you need to remove seasonality to apply a non-seasonal model.
- **MSTL is essential** for high-frequency data with multiple seasonal patterns (hourly data with daily + weekly patterns).

### 4.7 Common Mistakes

1. **Not choosing appropriate window sizes.** Too small = overfitting noise; too large = missing real changes.
2. **Forgetting that STL is additive only.** For multiplicative patterns, log-transform first.
3. **Not using robust mode** when outliers are present.
4. **Treating the remainder as noise** -- check for remaining autocorrelation.

### 4.8 Python Libraries

| Library | Function | Notes |
|---------|----------|-------|
| `statsmodels` | `STL()`, `MSTL()`, `STLForecast()` | Full implementation |
| `statsforecast` | `MSTL()` | Multiple seasonalities, integrated forecasting |

```python
# statsmodels STL
from statsmodels.tsa.seasonal import STL, MSTL
stl = STL(y, period=12, seasonal=7, trend=15, robust=True)
result = stl.fit()

# statsmodels STLForecast
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA
stlf = STLForecast(y, ARIMA, model_kwargs={"order": (1,1,0)}, period=12)
result = stlf.fit()
forecast = result.forecast(12)

# statsmodels MSTL (multiple seasonalities)
mstl = MSTL(y, periods=[24, 168])  # daily + weekly for hourly data
result = mstl.fit()
```

---

## 5. Stationarity and Preprocessing

### 5.1 Unit Root Tests

Three primary tests, each with different null hypotheses:

| Test | Null Hypothesis | Reject Means | Python |
|------|----------------|---------------|--------|
| **ADF** (Augmented Dickey-Fuller) | Series has a unit root (non-stationary) | Series is stationary | `statsmodels.tsa.stattools.adfuller()` |
| **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) | Series is trend-stationary | Series is non-stationary | `statsmodels.tsa.stattools.kpss()` |
| **PP** (Phillips-Perron) | Series has a unit root (non-stationary) | Series is stationary | `arch.unitroot.PhillipsPerron()` |

**Critical best practice -- always use both ADF and KPSS together:**

| ADF Result | KPSS Result | Conclusion |
|------------|-------------|------------|
| Reject (stationary) | Fail to reject (stationary) | **Series is stationary** |
| Fail to reject (non-stationary) | Reject (non-stationary) | **Series is non-stationary** -- difference it |
| Reject (stationary) | Reject (non-stationary) | **Trend-stationary** -- detrend or difference |
| Fail to reject | Fail to reject | **Inconclusive** -- gather more data or try PP test |

**Automated approach:**
```python
# statsforecast provides ndiffs/nsdiffs
from statsforecast.arima import ndiffs, nsdiffs

d = ndiffs(y, test="kpss")     # regular differencing order
D = nsdiffs(y, m=12, test="ocsb")  # seasonal differencing order
```

### 5.2 Differencing Strategies

**Regular differencing (removes trend):**
- y'_t = y_t - y_{t-1}
- Start with d=1. Check stationarity. Only go to d=2 if absolutely necessary.
- **Never** go beyond d=2 in practice.

**Seasonal differencing (removes seasonal pattern):**
- y'_t = y_t - y_{t-m} (where m = seasonal period)
- **Apply seasonal differencing FIRST** -- the result is sometimes already stationary, avoiding the need for regular differencing.
- Combined: (y_t - y_{t-m}) - (y_{t-1} - y_{t-m-1})

**Detecting over-differencing:**
- If lag-1 autocorrelation of differenced series is more negative than **-0.5**, you have over-differenced.
- If all autocorrelations are small and patternless, stop differencing.
- Over-differencing introduces **false dynamics** and noise.

### 5.3 Box-Cox and Log Transforms

**Purpose:** Stabilize variance when it changes with the level of the series.

The Box-Cox transformation:
```
w_t = log(y_t)              if lambda = 0
w_t = (sign(y_t)|y_t|^lambda - 1) / lambda    otherwise
```

**Special cases:**
| lambda | Transformation |
|--------|---------------|
| 1 | No transformation |
| 0 | Log transform |
| 0.5 | Square root |
| -1 | Inverse |

**Choosing lambda:**
- **Guerrero's method (1993)**: Minimizes the coefficient of variation for subseries. Automated and reliable.
- **Log-likelihood method**: Maximizes profile log-likelihood of a linear model.
- Use `guerrero()` from feasts (R) or manual implementation in Python.

**Practical guidelines:**
- A good value of lambda is one that makes the seasonal variation roughly constant.
- Choose a simple lambda if possible (0, 0.5, 1) -- interpretability matters.
- **Always back-transform forecasts** and be aware that back-transformation of means gives medians, not means (bias adjustment needed).
- Log transforms cannot handle zeros or negatives; use shifted log or Box-Cox with handling.

```python
# scipy Box-Cox
from scipy.stats import boxcox
y_transformed, lambda_opt = boxcox(y)

# Manual Guerrero implementation or use statsforecast
# statsmodels power_transform
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')  # requires positive values
```

### 5.4 Handling Outliers

**Detection:**
- Visual inspection of time plot and decomposition remainder.
- Points beyond 3 standard deviations of residuals.
- STL with `robust=True` isolates outliers in the remainder component.
- IQR method on decomposition residuals.

**Treatment options:**
1. **Robust methods**: Use STL with robust=True, or robust regression.
2. **Replacement**: Replace with interpolated values or seasonal average.
3. **Indicator variables**: Include outlier dummies in regression models.
4. **Winsorization**: Cap extreme values at percentile thresholds.

**Key principle:** Investigate outliers before removing them. They may carry real information (e.g., COVID-19 effects).

### 5.5 Handling Missing Values

- **Short gaps** (1-2 observations): Forward fill or linear interpolation.
- **Longer gaps**: Seasonal interpolation (use same season from adjacent years).
- **Structural gaps**: Consider whether the missing period changes the data-generating process.
- For ETS/ARIMA: missing values must be handled before fitting (the recursive algorithms break on gaps).

```python
# pandas interpolation
y_filled = y.interpolate(method='linear')
y_filled = y.interpolate(method='spline', order=3)

# seasonal interpolation
y_filled = y.interpolate(method='time')  # time-weighted
```

### 5.6 Trend and Seasonality Identification

**Visual tools:**
1. **Time plot**: Overall patterns, trend direction, seasonal regularity.
2. **Seasonal plot**: Overlay each season (year, week) to see patterns.
3. **Seasonal subseries plot**: Separate plots for each season.
4. **ACF plot**: Slow decay = non-stationary; spikes at seasonal lags = seasonality.

**Statistical tools:**
1. **Strength of trend**: F_T = max(0, 1 - Var(R_t) / Var(T_t + R_t))
2. **Strength of seasonality**: F_S = max(0, 1 - Var(R_t) / Var(S_t + R_t))
3. **Unit root tests** for stationarity.
4. **Spectral analysis** / periodogram for detecting dominant frequencies.

---

## 6. Key Principles from Hyndman's fpp3

Rob Hyndman and George Athanasopoulos's "Forecasting: Principles and Practice" (3rd edition, freely available at otexts.com/fpp3) is the canonical reference for practical time series forecasting. Key principles:

### 6.1 Always Start with Visualization

- Plot the data. Look at it carefully.
- Use multiple plot types: time plots, seasonal plots, ACF plots.
- Identify the components: trend, seasonality, cycles, outliers.

### 6.2 Use Appropriate Benchmarks

Never report forecast accuracy without comparing to simple benchmarks:
- **Naive method**: y_hat = last observation.
- **Seasonal naive**: y_hat = observation from same season last year.
- **Drift method**: y_hat = last observation + average change.
- **Mean method**: y_hat = historical average.

If your fancy model cannot beat these, it is not useful.

### 6.3 Cross-Validation, Not Just Train/Test Split

- Use **time series cross-validation** (rolling origin evaluation).
- Evaluate over multiple forecast horizons.
- Report multiple accuracy metrics (RMSE, MAE, MASE, MAPE).

### 6.4 Combine Forecasts

- Simple averages of forecasts from different methods often outperform individual methods.
- This was consistently confirmed across M-competitions.

### 6.5 Transformations Before Modeling

- Apply **calendar adjustments**, **population adjustments**, **inflation adjustments** as appropriate.
- Use **Box-Cox** (especially log) to stabilize variance.
- Use **differencing** to achieve stationarity (for ARIMA).

### 6.6 Let the Data Choose the Model

- Use information criteria (AICc) for model selection within a family.
- Use cross-validation for comparing across families (ETS vs ARIMA vs Theta).
- The AutoETS and AutoARIMA functions automate within-family selection.

### 6.7 Check Residuals

Good residuals should be:
1. **Uncorrelated** (check ACF, Ljung-Box test).
2. **Zero mean**.
3. **Constant variance**.
4. **Normally distributed** (for valid prediction intervals).

If residuals show patterns, the model has missed something.

### 6.8 Prediction Intervals Matter

- Point forecasts alone are insufficient for decision-making.
- ETS state-space models and ARIMA produce proper prediction intervals.
- Intervals widen with forecast horizon (uncertainty grows).
- Report 80% and 95% intervals as standard practice.

### 6.9 Damped Trends Are Almost Always Better

- For multi-step forecasts, damped trends consistently outperform linear trends.
- Unless you have strong domain knowledge that a linear trend will continue, use damping.

### 6.10 Practical Workflow

1. **Tidy and visualize** the data.
2. **Transform** if needed (Box-Cox, calendar adjustments).
3. **Decompose** (STL) to understand components.
4. **Fit multiple models** (ETS, ARIMA, Theta + benchmarks).
5. **Check residuals** for each model.
6. **Evaluate accuracy** via cross-validation.
7. **Select or combine** the best models.
8. **Produce forecasts** with prediction intervals.

---

## 7. Classical vs Modern Methods: When to Use What

### 7.1 Key Research Findings

**The Makridakis et al. (2018) Study (PLoS ONE):**
- Evaluated classical and ML methods on 1,045 monthly time series from M3.
- **Classical methods (ETS, ARIMA) outperformed ML and DL methods** for both one-step and multi-step forecasting.
- The Theta method and ARIMA were top performers for multi-step.
- LSTM and MLP performed worse than simple exponential smoothing.

**The M4 Competition (2018/2020) -- 100,000 series, 61 methods:**
1. **12 of the top 17 methods were combinations** of mostly statistical approaches.
2. The **winning method (Smyl/Uber)** was a hybrid: statistical features + ML (ES-RNN). ~10% more accurate than the combination benchmark.
3. The second-best was a combination of 7 statistical + 1 ML method.
4. **All six pure ML methods performed poorly** -- none beat the combination benchmark; only one beat Naive2.
5. Conclusion: "The six pure ML methods that were submitted in the M4 all performed poorly."

**Size Matters (arXiv:1909.13316):**
- ML methods improve relative to statistical methods **as sample size grows**.
- For short series typical in business forecasting, statistical methods dominate.
- For very long series (thousands of observations), ML can catch up or surpass.

### 7.2 Decision Framework

| Data Characteristic | Recommended Approach |
|--------------------|---------------------|
| **Short series** (< 100 obs) | Classical (ETS, ARIMA, Theta) |
| **Many short series** (1000s) | Classical methods per-series, or global ML models |
| **Long single series** (> 1000 obs) | ML/DL become competitive; still benchmark against classical |
| **Strong seasonality** | ETS (Holt-Winters), SARIMA, STL + model |
| **Multiple seasonalities** | MSTL + model, TBATS, or specialized DL (N-BEATS, etc.) |
| **Exogenous variables** | ARIMAX, regression with ARIMA errors, or ML (XGBoost, etc.) |
| **Nonlinear patterns** | ML/DL methods; or use STL decomposition + nonlinear model for remainder |
| **Intermittent demand** (zeros) | Croston's method, SBA, IMAPA (classical, specialized) |
| **Need for interpretability** | Classical methods (clear components, prediction intervals) |
| **Need for uncertainty quantification** | ETS, ARIMA (proper state-space PI); conformal prediction for ML |
| **Production at scale** | statsforecast (fastest), or global ML models |

### 7.3 Hybrid Approaches (The Emerging Best Practice)

The M4 results strongly suggest the future is **hybrid**:

1. Use statistical methods to extract features (trend, seasonality, level).
2. Use ML to learn combination weights or residual patterns.
3. The ES-RNN (Smyl) approach: per-series exponential smoothing for level/seasonality + global RNN for residuals.
4. Simple combination (average of ETS, ARIMA, Theta) is a strong baseline that often beats individual ML methods.

### 7.4 Common Mistakes

1. **Jumping to deep learning without trying classical methods first.** Simple methods are hard to beat on univariate series.
2. **Not using classical methods as baselines.** You must demonstrate your ML model adds value beyond ETS/ARIMA.
3. **Ignoring the "combination" approach.** Averaging 3-5 methods is almost always better than picking one.
4. **Applying global ML models to a single short series.** ML needs data; for single short series, classical wins.
5. **Assuming more complexity = more accuracy.** In the M3 and M4 competitions, this was consistently disproved.

---

## Summary Comparison Table

| Method | Best For | Parameters | Automation | Python Library |
|--------|----------|-----------|------------|---------------|
| **ARIMA/SARIMA** | Autocorrelated data, medium-term forecasts | (p,d,q)(P,D,Q,m) | AutoARIMA | statsforecast, pmdarima, statsmodels |
| **ETS** | Trend + seasonality, many series at scale | Error, Trend, Seasonal type + alpha, beta, gamma, phi | AutoETS | statsforecast, statsmodels |
| **Theta** | Robust baseline, annual/micro data | Theta coefficient (usually 0 and 2) | Built-in | statsforecast, statsmodels, sktime |
| **STL/MSTL** | Decomposition, multiple seasonalities | seasonal_window, trend_window, robust | Semi-automatic | statsmodels, statsforecast |
| **Naive/SNaive** | Benchmarks | None | N/A | statsforecast |
| **Combinations** | General best practice | Weights (equal or optimized) | Manual or ML-optimized | Custom |

---

## Recommended Python Ecosystem

For modern Python-based time series work, the **Nixtla ecosystem** is the most comprehensive and performant:

- **`statsforecast`**: Lightning-fast statistical methods (ARIMA, ETS, Theta, CES, MSTL, and 30+ models). Up to 300x faster than statsmodels.
- **`mlforecast`**: ML methods (LightGBM, XGBoost) for time series with proper lag features.
- **`neuralforecast`**: Deep learning methods (N-BEATS, N-HiTS, TFT, PatchTST).
- **`hierarchicalforecast`**: Hierarchical/grouped time series reconciliation.
- **`utilsforecast`**: Evaluation metrics, plotting, cross-validation.

For more control and academic use:
- **`statsmodels`**: Full-featured, well-documented, state-space models.
- **`pmdarima`**: Faithful Python port of R's auto.arima.
- **`sktime`**: Scikit-learn-compatible time series framework.

---

## Sources

- [Hyndman & Athanasopoulos, Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/)
- [Hyndman & Athanasopoulos, fpp Pythonic Way -- ARIMA Chapter](https://otexts.com/fpppy/nbs/09-arima.html)
- [Hyndman & Athanasopoulos, fpp Pythonic Way -- ETS Chapter](https://otexts.com/fpppy/nbs/08-exponential-smoothing.html)
- [Hyndman & Billah, Unmasking the Theta Method (2003)](https://robjhyndman.com/papers/Theta.pdf)
- [Assimakopoulos & Nikolopoulos, The Theta Model (2000)](https://www.sciencedirect.com/science/article/abs/pii/S0169207000000662)
- [pmdarima Tips and Tricks](https://alkaline-ml.com/pmdarima/tips_and_tricks.html)
- [Hyndman fpp3 -- STL Decomposition](https://otexts.com/fpp3/stl.html)
- [Hyndman fpp3 -- Transformations and Adjustments](https://otexts.com/fpp3/transformations.html)
- [statsmodels -- Stationarity and Detrending (ADF/KPSS)](https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html)
- [statsmodels -- STL Decomposition](https://www.statsmodels.org/dev/examples/notebooks/generated/stl_decomposition.html)
- [statsmodels -- MSTL Decomposition](https://www.statsmodels.org/dev/examples/notebooks/generated/mstl_decomposition.html)
- [Nixtla statsforecast -- Theta Model](https://nixtlaverse.nixtla.io/statsforecast/docs/models/standardtheta.html)
- [Brenndoerfer -- ETS Complete Guide](https://mbrenndoerfer.com/writing/exponential-smoothing-ets-time-series-forecasting)
- [Makridakis et al., The M4 Competition: Results (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0169207018300785)
- [Makridakis et al., M4 Competition: 100,000 time series (2020)](https://www.sciencedirect.com/science/article/pii/S0169207019301128)
- [Makridakis Competitions -- Wikipedia](https://en.wikipedia.org/wiki/Makridakis_Competitions)
- [Makridakis et al., Statistical and ML forecasting methods: Concerns and ways forward (PLoS ONE, 2018)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0194889)
- [ML vs Statistical Methods: Size Matters (arXiv:1909.13316)](https://arxiv.org/abs/1909.13316)
- [Comparing Statistical and ML Methods for Time Series (MDPI, 2025)](https://www.mdpi.com/1099-4300/27/1/25)
- [ML Mastery -- Comparing Classical and ML Methods](https://machinelearningmastery.com/findings-comparing-classical-and-machine-learning-methods-for-time-series-forecasting/)
- [Bandara et al., MSTL Algorithm (arXiv:2107.13462)](https://arxiv.org/abs/2107.13462)
- [Guerrero (1993), Time-series analysis supported by power transformations](https://doi.org/10.1002/for.3980120104)
