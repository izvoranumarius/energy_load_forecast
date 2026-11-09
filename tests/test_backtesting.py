"""Tests for the rolling-origin backtesting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.backtesting import (
    backtest_metrics,
    compare_models,
    make_origins,
    rolling_origin_backtest,
    summarize_backtest,
)
from energy_load_forecast.models.baselines import (
    naive_forecast,
    seasonal_naive_forecast,
)


def _hourly_series(start: str, values: list[float]) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, name="load_mw")


def _daily_pattern(n_days: int) -> pd.Series:
    return _hourly_series("2026-01-01", [1000.0 + hour for hour in range(24)] * n_days)


class TestMakeOrigins:
    def test_last_window_ends_at_last_timestamp(self):
        series = _daily_pattern(5)

        origins = make_origins(series.index, n_origins=3, horizon=24, step=24)

        assert origins[-1] + pd.Timedelta(hours=23) == series.index[-1]

    def test_origins_are_step_hours_apart(self):
        series = _daily_pattern(5)

        origins = make_origins(series.index, n_origins=3, horizon=24, step=12)

        assert len(origins) == 3
        assert (origins.to_series().diff().dropna() == pd.Timedelta(hours=12)).all()

    def test_raises_if_no_history_before_first_origin(self):
        series = _daily_pattern(2)

        with pytest.raises(ValueError, match="Not enough data"):
            make_origins(series.index, n_origins=2, horizon=24, step=24)

    def test_raises_on_non_positive_n_origins(self):
        series = _daily_pattern(3)

        with pytest.raises(ValueError, match="n_origins"):
            make_origins(series.index, n_origins=0)


class TestRollingOriginBacktest:
    def test_returns_one_row_per_forecast_hour(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)

        results = rolling_origin_backtest(series, naive_forecast, origins)

        assert len(results) == 3 * 24
        assert list(results.columns) == ["origin", "actual", "forecast"]

    def test_history_ends_before_origin(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)
        seen_history_ends = []

        def recording_forecast(history: pd.Series, horizon: int) -> pd.Series:
            seen_history_ends.append(history.index[-1])
            return naive_forecast(history, horizon)

        rolling_origin_backtest(series, recording_forecast, origins)

        expected = [origin - pd.Timedelta(hours=1) for origin in origins]
        assert seen_history_ends == expected

    def test_seasonal_naive_is_perfect_on_repeating_pattern(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)

        results = rolling_origin_backtest(series, seasonal_naive_forecast, origins)

        assert (results["actual"] == results["forecast"]).all()

    def test_raises_if_forecast_index_is_misaligned(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=2)

        def shifted_forecast(history: pd.Series, horizon: int) -> pd.Series:
            forecast = naive_forecast(history, horizon)
            forecast.index = forecast.index + pd.Timedelta(hours=1)
            return forecast

        with pytest.raises(ValueError, match="does not cover"):
            rolling_origin_backtest(series, shifted_forecast, origins)

    def test_raises_if_window_ends_after_series(self):
        series = _daily_pattern(3)
        origins = pd.DatetimeIndex([series.index[-1]], name="origin")

        with pytest.raises(ValueError, match="ends after the series"):
            rolling_origin_backtest(series, naive_forecast, origins)


class TestBacktestMetrics:
    def test_one_row_per_origin(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)
        results = rolling_origin_backtest(series, naive_forecast, origins)

        metrics = backtest_metrics(results)

        assert list(metrics.index) == list(origins)
        assert list(metrics.columns) == ["MAE", "RMSE", "MAPE"]

    def test_raises_on_missing_columns(self):
        with pytest.raises(KeyError, match="Missing required columns"):
            backtest_metrics(pd.DataFrame({"actual": [1.0]}))


class TestSummarizeBacktest:
    def test_contains_mean_and_std_for_each_metric(self):
        metrics = pd.DataFrame({"MAE": [10.0, 30.0], "RMSE": [12.0, 36.0], "MAPE": [1.0, 3.0]})

        summary = summarize_backtest(metrics)

        assert summary.loc["MAE", "mean"] == pytest.approx(20.0)
        assert list(summary.columns) == ["mean", "std", "min", "max"]


class TestCompareModels:
    def test_ranks_seasonal_naive_ahead_of_naive_on_daily_pattern(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)

        comparison = compare_models(
            series,
            {"Naive": naive_forecast, "Seasonal Naive": seasonal_naive_forecast},
            origins,
        )

        assert comparison.loc["Seasonal Naive", "MAE"] == 0.0
        assert comparison.loc["Naive", "MAE"] > 0.0

    def test_raises_on_empty_models(self):
        series = _daily_pattern(5)
        origins = make_origins(series.index, n_origins=3)

        with pytest.raises(ValueError, match="must not be empty"):
            compare_models(series, {}, origins)
