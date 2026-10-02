"""Tests for the baseline forecasting helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.models.baselines import (
    naive_forecast,
    seasonal_naive_forecast,
)


def _hourly_series(start: str, values: list[float]) -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


class TestNaiveForecast:
    def test_repeats_last_value(self):
        history = _hourly_series("2026-01-01", [10.0, 20.0, 30.0])

        forecast = naive_forecast(history, horizon=5)

        assert (forecast == 30.0).all()

    def test_forecast_length_matches_horizon(self):
        history = _hourly_series("2026-01-01", [10.0, 20.0, 30.0])

        forecast = naive_forecast(history, horizon=24)

        assert len(forecast) == 24

    def test_forecast_starts_right_after_history(self):
        history = _hourly_series("2026-01-01", [10.0, 20.0, 30.0])

        forecast = naive_forecast(history, horizon=3)

        expected_start = history.index[-1] + pd.Timedelta(hours=1)
        assert forecast.index[0] == expected_start


class TestSeasonalNaiveForecast:
    def test_copies_values_from_24h_before(self):
        day1 = list(range(24))
        day2 = [100 + i for i in range(24)]
        history = _hourly_series("2026-01-01", day1 + day2)

        forecast = seasonal_naive_forecast(history, horizon=24, season_length=24)

        assert list(forecast.to_numpy()) == day2

    def test_raises_if_not_enough_history(self):
        history = _hourly_series("2026-01-01", [1.0, 2.0, 3.0])  

        with pytest.raises(ValueError, match="at least 24"):
            seasonal_naive_forecast(history, horizon=24, season_length=24)

    def test_tiles_when_horizon_exceeds_season_length(self):
        history = _hourly_series("2026-01-01", list(range(24)))

        forecast = seasonal_naive_forecast(history, horizon=30, season_length=24)

        assert len(forecast) == 30
        assert list(forecast.to_numpy()[:24]) == list(range(24))
        assert list(forecast.to_numpy()[24:30]) == list(range(6))
