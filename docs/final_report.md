[Română](final_report.ro.md)

# Electricity Load Forecasting - Final Project Deliverable

## 1. Project objective

The project forecasts electricity demand for Romania using hourly historical load data from the ENTSO-E Transparency Platform.

The forecasting problem is defined as:

- **Geography:** Romania (`RO`)
- **Target:** Actual Total Load (`load_mw`)
- **Granularity:** hourly
- **Forecast horizon:** 24 hours ahead
- **Forecast type:** point forecast, with an optional P10-P90 prediction interval

The main modeling comparison includes two simple baselines, LightGBM, and N-BEATS.

## 2. Data and ingestion

Historical electricity load is collected from ENTSO-E in two supported forms:

1. CSV exports from the Transparency Platform.
2. The ENTSO-E API through `entsoe-py`.

The ingestion layer normalizes timestamps to UTC and resamples the source data to an hourly `load_mw` series. Missing values are intentionally kept for preprocessing rather than being silently filled.

For the final experiment, the dataset covers **July 2025 through December 2025**.

## 3. Data quality and preprocessing

The preprocessing pipeline checks the main structural issues relevant to hourly load forecasting:

- timezone handling and UTC normalization;
- duplicate timestamps;
- chronological ordering;
- missing values and publication-delay NaNs at the end of the series;
- negative load values;
- regular hourly spacing;
- interior missing values versus trailing missing values.

Trailing NaNs are removed because they correspond to the incomplete publication tail. Interior NaNs are treated as a data-quality problem rather than being silently ignored.

The data is split chronologically into train, validation, and test windows. The final experiment reserves one complete 24-hour validation window and one complete 24-hour test window.

## 4. Exploratory data analysis and seasonality

The EDA examines:

- the load series over time;
- average load by local hour;
- average load by day of week;
- weekday versus weekend load;
- load distribution and boxplot;
- monthly average load;
- a day-of-week by hour load profile;
- autocorrelation at 24-hour and 168-hour lags.

A repeating hourly profile supports a daily seasonal component, while differences between weekdays and weekends support a weekly seasonal component.

The measured autocorrelation is **0.803 at 24 hours** and **0.855 at 168 hours**. These values provide numerical evidence of strong daily and weekly persistence in the series.

### Important limitation on yearly seasonality

The available final dataset covers only July to December 2025. Therefore, the monthly profile is useful as a descriptive within-year comparison, but it should not be presented as a reliable full yearly seasonal estimate. A stronger yearly seasonal conclusion would require multiple complete annual cycles.

The final EDA notebook contains the seasonality plots and autocorrelation checks.

## 5. Feature engineering

The LightGBM model uses features derived only from information available before the prediction timestamp:

- `hour_local`
- `day_of_week_local`
- `is_weekend_local`
- `month_local`
- `load_mw_lag1`
- `load_mw_lag24`
- `load_mw_lag168`
- `load_mw_rolling_mean_3h`
- `load_mw_rolling_mean_24h`

Rolling features use a one-step shift before calculating the mean, so the current target value is not included.

During the multi-step validation and test forecast, LightGBM features are constructed recursively. Predicted values replace unavailable future observations, which prevents future actual load values from entering the forecast.

## 6. Forecasting models

### Baseline 1 - Naive

The Naive forecast repeats the last observed load value for all 24 forecast hours.

### Baseline 2 - Seasonal Naive

The Seasonal Naive forecast repeats the most recent 24-hour load profile. This provides a simple reference for the daily seasonal pattern.

### LightGBM

LightGBM is used as the main feature-based machine-learning model. A point model is trained for point forecasts, and separate quantile models are trained for the 10th, 50th, and 90th quantiles for the optional prediction interval.

### N-BEATS

N-BEATS is used as a second model family that learns directly from the historical load series rather than the engineered LightGBM feature table. The training series is scaled using a scaler fitted only on the training data.

## 7. Evaluation methodology

Point forecasts are evaluated using:

- **MAE** - mean absolute error;
- **RMSE** - root mean squared error;
- **MAPE** - mean absolute percentage error, interpreted with caution for very small actual values.

Prediction intervals are evaluated using:

- P10 pinball loss;
- P90 pinball loss;
- empirical coverage;
- mean interval width.

The final test comparison is evaluated on the same 24-hour test index for every model.

## 8. Error analysis and interpretability

The project includes:

- actual versus predicted load;
- residuals over time;
- residual distribution;
- mean absolute residual by local hour;
- the largest absolute forecast errors;
- LightGBM feature importance.

The LightGBM feature importance plot shows that `hour_local` and `load_mw_lag1` contribute the most to the model. `load_mw_lag168` and the 3-hour rolling mean are also important, followed by the 24-hour rolling mean and `load_mw_lag24`. Calendar features such as `month_local` and `day_of_week_local` contribute less, while `is_weekend_local` receives no measured importance in this fitted model.

The residual plot shows larger negative residuals during the early morning, followed by values closer to zero and positive residuals around the afternoon period. The residuals become mixed again toward the evening. Because this analysis uses a single 24-hour test window, these patterns should be interpreted as observations for this particular forecast horizon rather than as general behavior across the full dataset.

## 9. Final results

The final quantitative comparison is based on the 24-hour test window from **2025-12-31 00:00:00 UTC** through the end of that day.

### Results table

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive | 773.24 | 975.43 | 11.2704 |
| Seasonal Naive | 199.375 | 256.484 | 3.02983 |
| LightGBM | 230.052 | 266.821 | 3.64566 |
| N-BEATS | 300.971 | 352.976 | 4.73213 |

On this test window, the Seasonal Naive baseline has the lowest MAE, RMSE, and MAPE among the evaluated models. LightGBM follows with relatively close errors, while N-BEATS has higher errors on this particular 24-hour horizon. The result shows that a simple daily seasonal reference can remain highly informative for this dataset and test period.

### P10-P90 prediction interval

The LightGBM quantile models produced the following interval metrics:

| Metric | Value |
|---|---:|
| P10 pinball loss | 39.7004 |
| P90 pinball loss | 59.5282 |
| Empirical coverage | 87.5% |
| Mean interval width | 922.193 MW |

The empirical coverage is based on the same 24-hour test window, so it should be interpreted cautiously because the sample is small.

### Visual comparison

The final comparison plot shows the actual load together with all four point forecasts over the 24-hour test horizon. The Naive baseline remains nearly flat, while the Seasonal Naive forecast follows the daily shape more closely. LightGBM and N-BEATS both reproduce the main intraday pattern, with different deviations around the morning and afternoon periods.

## 10. Project structure

The notebooks document the analysis and experimentation workflow, while the Python modules contain the reusable implementation extracted from that work.

Main notebooks:

- `00_explore_raw_csv.ipynb`
- `01_api_ingestion.ipynb`
- `02_preprocessing_exploration.ipynb`
- `03_eda.ipynb`
- `04_baselines.ipynb`
- `05_feature_engineering.ipynb`
- `06_lightgbm.ipynb`
- `07_nbeats.ipynb`
- `08_error_analysis.ipynb`
- `09_final_results.ipynb`

Main Python modules:

- `ingestion.py`
- `preprocessing.py`
- `features.py`
- `baselines.py`
- `lightgbm_model.py`
- `nbeats_model.py`
- `evaluation.py`
- `interval_evaluation.py`
- `error_analysis.py`
- `config.py`

Tests are under `tests/` and use imports through `energy_load_forecast.<module>` without a `conftest.py` path hack.

## 11. Reproducibility

The project uses `uv` for environment and dependency management. The repository includes `uv.lock` so the dependency versions are locked.

The final workflow should be run in this order:

1. obtain the ENTSO-E data or use the cached hourly CSV;
2. run the preprocessing and EDA notebooks as needed;
3. run the model notebooks or the final results notebook;
4. wait for N-BEATS training to finish;
5. inspect `docs/results.md` and the saved figures in `docs/`.

The final test horizon must contain exactly 24 hourly observations.

## 12. Requirement checklist

| Requirement | Status |
|---|---|
| Forecast problem clearly defined | Complete |
| Historical electricity load collected | Complete |
| Data cleaning and alignment | Complete |
| Daily/weekly seasonality EDA | Complete |
| Yearly seasonality discussion | Complete with dataset limitation noted |
| Feature engineering | Complete |
| Chronological train/validation/test split | Complete |
| Simple baselines | Complete |
| Multiple candidate models | Complete |
| MAE/RMSE/MAPE | Complete |
| Prediction interval evaluation | Complete |
| Feature importance and error analysis | Complete |
| Final visual deliverable | Complete |
