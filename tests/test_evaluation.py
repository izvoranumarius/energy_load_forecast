"""Tests for the forecasting evaluation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.evaluation import evaluate, mae, mape, rmse


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


class TestMae:
    def test_perfect_forecast_is_zero(self):
        actual = _series([100.0, 200.0, 300.0])
        assert mae(actual, actual) == 0.0

    def test_known_value(self):
        actual = _series([100.0, 200.0])
        predicted = _series([110.0, 190.0])  
        assert mae(actual, predicted) == pytest.approx(10.0)


class TestRmse:
    def test_perfect_forecast_is_zero(self):
        actual = _series([100.0, 200.0, 300.0])
        assert rmse(actual, actual) == 0.0

    def test_penalizes_large_errors_more_than_mae(self):
        actual = _series([100.0, 100.0])
        predicted = _series([100.0, 200.0])  
        assert rmse(actual, predicted) > mae(actual, predicted)


class TestMape:
    def test_known_value(self):
        actual = _series([200.0, 400.0])
        predicted = _series([220.0, 380.0]) 
        assert mape(actual, predicted) == pytest.approx(7.5)

    def test_warns_on_small_actual_values(self, caplog):
        actual = _series([1.0, 200.0])  
        predicted = _series([2.0, 210.0])

        with caplog.at_level("WARNING"):
            mape(actual, predicted)

        assert "unreliable" in caplog.text

    def test_zero_actual_is_excluded_without_division_by_zero(self, caplog):
        actual = _series([0.0, 200.0])
        predicted = _series([10.0, 220.0])

        with caplog.at_level("WARNING"):
            result = mape(actual, predicted)

        assert result == pytest.approx(10.0)
        assert "exactly zero" in caplog.text

    def test_all_zero_actual_returns_nan(self):
        actual = _series([0.0, 0.0])
        predicted = _series([10.0, 20.0])

        result = mape(actual, predicted)

        assert pd.isna(result)


class TestEvaluate:
    def test_returns_all_three_metrics(self):
        actual = _series([100.0, 200.0])
        predicted = _series([110.0, 190.0])

        result = evaluate(actual, predicted)

        assert set(result.keys()) == {"MAE", "RMSE", "MAPE"}

    def test_raises_on_mismatched_index(self):
        actual = _series([100.0, 200.0])
        predicted = pd.Series(
            [100.0, 200.0],
            index=pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC"),
        )

        with pytest.raises(ValueError, match="same index"):
            evaluate(actual, predicted)
