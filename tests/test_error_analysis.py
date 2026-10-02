"""Tests for the forecasting error-analysis helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.error_analysis import (
    compute_residuals,
    residuals_by_group,
    top_n_errors,
)


def _series(values: list[float], start: str = "2026-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index)


class TestComputeResiduals:
    def test_known_values(self):
        actual = _series([100.0, 200.0, 300.0])
        predicted = _series([90.0, 210.0, 280.0])

        residuals = compute_residuals(actual, predicted)

        assert residuals.tolist() == pytest.approx([10.0, -10.0, 20.0])

    def test_result_is_named_residual(self):
        actual = _series([100.0])
        predicted = _series([90.0])

        residuals = compute_residuals(actual, predicted)

        assert residuals.name == "residual"

    def test_raises_on_mismatched_index(self):
        actual = _series([100.0, 200.0])
        predicted = _series([100.0, 200.0], start="2027-01-01")

        with pytest.raises(ValueError, match="same index"):
            compute_residuals(actual, predicted)


class TestResidualsByGroup:
    def test_groups_and_averages_correctly(self):
        residuals = _series([10.0, -10.0, 20.0, -20.0])
        group = pd.Series(["a", "a", "b", "b"], index=residuals.index)

        result = residuals_by_group(residuals, group)

        assert result.loc["a"] == pytest.approx(10.0)
        assert result.loc["b"] == pytest.approx(20.0)

    def test_result_is_named_mean_abs_residual(self):
        residuals = _series([10.0, -10.0])
        group = pd.Series(["a", "a"], index=residuals.index)

        result = residuals_by_group(residuals, group)

        assert result.name == "mean_abs_residual"

    def test_raises_on_mismatched_index(self):
        residuals = _series([10.0, -10.0])
        group = pd.Series(["a", "a"], index=_series([0.0, 0.0], start="2027-01-01").index)

        with pytest.raises(ValueError, match="same index"):
            residuals_by_group(residuals, group)


class TestTopNErrors:
    def test_returns_largest_absolute_errors_descending(self):
        actual = _series([100.0, 100.0, 100.0, 100.0])
        predicted = _series([95.0, 50.0, 90.0, 110.0])

        result = top_n_errors(actual, predicted, n=2)

        assert len(result) == 2
        assert result["abs_error"].iloc[0] == pytest.approx(50.0)
        assert list(result.columns) == ["actual", "predicted", "abs_error"]

    def test_n_larger_than_data_returns_all_rows(self):
        actual = _series([100.0, 200.0])
        predicted = _series([90.0, 190.0])

        result = top_n_errors(actual, predicted, n=10)

        assert len(result) == 2

    def test_raises_on_mismatched_index(self):
        actual = _series([100.0, 200.0])
        predicted = _series([100.0, 200.0], start="2027-01-01")

        with pytest.raises(ValueError, match="same index"):
            top_n_errors(actual, predicted)
