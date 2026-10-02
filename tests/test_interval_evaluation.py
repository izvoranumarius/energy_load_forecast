"""Tests for probabilistic forecast evaluation helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.interval_evaluation import (
    coverage,
    interval_width,
    pinball_loss,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


class TestPinballLoss:
    def test_perfect_forecast_is_zero(self):
        actual = _series([100.0, 200.0, 300.0])
        assert pinball_loss(actual, actual, alpha=0.5) == pytest.approx(0.0)

    def test_known_value_for_median(self):
        actual = _series([100.0, 200.0])
        predicted = _series([110.0, 190.0]) 
        assert pinball_loss(actual, predicted, alpha=0.5) == pytest.approx(5.0)

    def test_underprediction_penalized_more_for_high_alpha(self):
        actual = _series([100.0])
        under = _series([90.0]) 
        over = _series([110.0])  

        loss_under = pinball_loss(actual, under, alpha=0.9)
        loss_over = pinball_loss(actual, over, alpha=0.9)

        assert loss_under > loss_over

    def test_raises_on_mismatched_index(self):
        actual = _series([100.0, 200.0])
        predicted = pd.Series(
            [100.0, 200.0],
            index=pd.date_range("2027-01-01", periods=2, freq="h", tz="UTC"),
        )
        with pytest.raises(ValueError, match="same index"):
            pinball_loss(actual, predicted, alpha=0.5)

    def test_raises_on_invalid_alpha(self):
        actual = _series([100.0])
        with pytest.raises(ValueError, match="alpha"):
            pinball_loss(actual, actual, alpha=1.5)


class TestCoverage:
    def test_all_values_inside_band_is_100_percent(self):
        actual = _series([100.0, 200.0, 300.0])
        lower = _series([50.0, 150.0, 250.0])
        upper = _series([150.0, 250.0, 350.0])

        assert coverage(actual, lower, upper) == pytest.approx(100.0)

    def test_value_outside_band_reduces_coverage(self):
        actual = _series([100.0, 999.0])  
        lower = _series([50.0, 150.0])
        upper = _series([150.0, 250.0])

        assert coverage(actual, lower, upper) == pytest.approx(50.0)

    def test_raises_on_mismatched_index(self):
        actual = _series([100.0])
        lower = _series([50.0])
        upper = pd.Series([150.0], index=pd.date_range("2027-01-01", periods=1, tz="UTC"))

        with pytest.raises(ValueError, match="same index"):
            coverage(actual, lower, upper)


class TestIntervalWidth:
    def test_known_value(self):
        lower = _series([50.0, 100.0])
        upper = _series([150.0, 300.0])
        assert interval_width(lower, upper) == pytest.approx(150.0)

    def test_raises_on_mismatched_index(self):
        lower = _series([50.0])
        upper = pd.Series([150.0], index=pd.date_range("2027-01-01", periods=1, tz="UTC"))

        with pytest.raises(ValueError, match="same index"):
            interval_width(lower, upper)
