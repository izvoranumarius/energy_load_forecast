"""Tests for the LightGBM forecasting helpers.

These tests check pipeline mechanics such as fitting, prediction shape,
index alignment, and quantile-model handling rather than model accuracy.
Accuracy is evaluated in the forecasting notebooks against the baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_load_forecast.models.lightgbm_model import (
    feature_importance,
    make_lightgbm_forecaster,
    predict,
    predict_quantiles,
    recursive_forecast,
    train_lightgbm,
    train_quantile_models,
)


def _synthetic_data(n: int = 50, n_features: int = 3):
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {f"feature_{i}": rng.normal(size=n) for i in range(n_features)},
        index=index,
    )
    y = pd.Series(rng.normal(size=n), index=index, name="load_mw")
    return X, y


class TestTrainLightgbm:
    def test_returns_fitted_model(self):
        X, y = _synthetic_data()

        model = train_lightgbm(X, y, n_estimators=5)

        assert hasattr(model, "feature_importances_")

    def test_raises_on_mismatched_index(self):
        X, y = _synthetic_data()
        y_bad = y.copy()
        y_bad.index = y_bad.index + pd.Timedelta(hours=1000)

        with pytest.raises(ValueError, match="same index"):
            train_lightgbm(X, y_bad)

    def test_random_state_defaults_to_42_for_reproducibility(self):
        X, y = _synthetic_data()

        model_a = train_lightgbm(X, y, n_estimators=5)
        model_b = train_lightgbm(X, y, n_estimators=5)

        pred_a = predict(model_a, X)
        pred_b = predict(model_b, X)
        pd.testing.assert_series_equal(pred_a, pred_b)


class TestPredict:
    def test_output_length_matches_input(self):
        X, y = _synthetic_data(n=50)
        model = train_lightgbm(X, y, n_estimators=5)

        preds = predict(model, X)

        assert len(preds) == len(X)

    def test_output_index_matches_input(self):
        X, y = _synthetic_data(n=50)
        model = train_lightgbm(X, y, n_estimators=5)

        preds = predict(model, X)

        assert preds.index.equals(X.index)

    def test_output_is_named_forecast(self):
        X, y = _synthetic_data(n=50)
        model = train_lightgbm(X, y, n_estimators=5)

        preds = predict(model, X)

        assert preds.name == "forecast"


class TestFeatureImportance:
    def test_returns_one_value_per_feature_sorted_descending(self):
        X, y = _synthetic_data(n=50, n_features=4)
        model = train_lightgbm(X, y, n_estimators=10)

        importances = feature_importance(model, list(X.columns))

        assert set(importances.index) == set(X.columns)
        assert list(importances.values) == sorted(importances.values, reverse=True)

    def test_raises_on_mismatched_feature_count(self):
        X, y = _synthetic_data(n=50, n_features=3)
        model = train_lightgbm(X, y, n_estimators=5)

        with pytest.raises(ValueError, match="feature names"):
            feature_importance(model, ["only_one_name"])


class TestTrainQuantileModels:
    def test_returns_one_model_per_alpha(self):
        X, y = _synthetic_data()

        models = train_quantile_models(X, y, alphas=[0.1, 0.5, 0.9], n_estimators=5)

        assert set(models.keys()) == {0.1, 0.5, 0.9}

    def test_raises_on_mismatched_index(self):
        X, y = _synthetic_data()
        y_bad = y.copy()
        y_bad.index = y_bad.index + pd.Timedelta(hours=1000)

        with pytest.raises(ValueError, match="same index"):
            train_quantile_models(X, y_bad)

    def test_raises_if_objective_passed_explicitly(self):
        X, y = _synthetic_data()

        with pytest.raises(ValueError, match="objective"):
            train_quantile_models(X, y, objective="regression")

    def test_raises_if_alpha_passed_explicitly(self):
        X, y = _synthetic_data()

        with pytest.raises(ValueError, match="alpha"):
            train_quantile_models(X, y, alpha=0.5)


class TestPredictQuantiles:
    def test_returns_one_column_per_alpha(self):
        X, y = _synthetic_data()
        models = train_quantile_models(X, y, alphas=[0.1, 0.5, 0.9], n_estimators=5)

        preds = predict_quantiles(models, X)

        assert set(preds.columns) == {"q0.1", "q0.5", "q0.9"}
        assert preds.index.equals(X.index)


def _load_history(n_days: int = 10) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=n_days * 24, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    values = 1000.0 + 100.0 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values + rng.normal(scale=5.0, size=len(index)), index=index, name="load_mw")


class TestRecursiveForecast:
    def _fitted_model(self, history: pd.Series):
        X = pd.DataFrame(
            {
                "load_mw_lag1": history.shift(1),
                "load_mw_lag24": history.shift(24),
                "load_mw_rolling_mean_3h": history.shift(1).rolling(3).mean(),
            },
            index=history.index,
        ).dropna()
        model = train_lightgbm(X, history.loc[X.index], n_estimators=5, verbose=-1)
        return model, list(X.columns), (1, 24), (3,)

    def test_returns_one_forecast_per_future_hour(self):
        history = _load_history()
        model, columns, lags, windows = self._fitted_model(history)
        future_index = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=1), periods=24, freq="h"
        )

        forecast = recursive_forecast(
            model, history, future_index, columns, lags=lags, windows=windows
        )

        assert forecast.index.equals(future_index)
        assert forecast.name == "forecast"
        assert forecast.notna().all()

    def test_raises_if_future_index_does_not_follow_history(self):
        history = _load_history()
        model, columns, lags, windows = self._fitted_model(history)
        future_index = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=2), periods=24, freq="h"
        )

        with pytest.raises(ValueError, match="one hour after"):
            recursive_forecast(
                model, history, future_index, columns, lags=lags, windows=windows
            )

    def test_raises_on_unknown_feature_column(self):
        history = _load_history()
        model, columns, _, windows = self._fitted_model(history)
        future_index = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=1), periods=3, freq="h"
        )

        with pytest.raises(KeyError):
            recursive_forecast(
                model, history, future_index, columns, lags=(1,), windows=windows
            )

    def test_raises_if_history_is_shorter_than_largest_lag(self):
        history = _load_history(n_days=2)
        model, columns, _, windows = self._fitted_model(history)
        future_index = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=1), periods=3, freq="h"
        )

        with pytest.raises(ValueError, match="at least 168"):
            recursive_forecast(
                model, history, future_index, columns, lags=(1, 168), windows=windows
            )


class TestMakeLightgbmForecaster:
    def test_forecast_starts_right_after_history(self):
        history = _load_history()
        forecaster = make_lightgbm_forecaster(n_estimators=5, verbose=-1)

        forecast = forecaster(history, 24)

        assert len(forecast) == 24
        assert forecast.index[0] == history.index[-1] + pd.Timedelta(hours=1)

    def test_same_history_gives_same_forecast(self):
        history = _load_history()
        forecaster = make_lightgbm_forecaster(n_estimators=5, verbose=-1)

        forecast_a = forecaster(history, 24)
        forecast_b = forecaster(history, 24)

        pd.testing.assert_series_equal(forecast_a, forecast_b)

