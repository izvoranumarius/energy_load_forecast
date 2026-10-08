"""Tests for nbeats_model.py.

These tests focus on validation and forecast-shape mechanics. The tests do not
assert model accuracy because that belongs in the forecasting notebooks.
Training is kept minimal so the suite remains practical for local development.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("darts")

from energy_load_forecast.models.nbeats_model import (
    predict,
    series_to_darts,
    train_nbeats,
)


def _hourly_series(
    periods: int = 60,
    start: str = "2026-01-01",
) -> pd.Series:
    """Build a clean hourly UTC load series for tests."""
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    return pd.Series(range(periods), index=index, dtype=float, name="load_mw")


class TestSeriesToDarts:
    def test_converts_utc_series_to_timezone_naive_darts_series(self):
        series = _hourly_series(periods=24)

        result = series_to_darts(series)

        assert len(result) == 24
        assert result.time_index.tz is None
        assert pd.Timestamp(result.time_index[0]) == pd.Timestamp("2026-01-01")

    def test_raises_on_naive_index(self):
        series = pd.Series(
            [1.0, 2.0, 3.0],
            index=pd.date_range("2026-01-01", periods=3, freq="h"),
        )

        with pytest.raises(ValueError, match="UTC"):
            series_to_darts(series)

    def test_raises_on_non_hourly_index(self):
        index = pd.date_range("2026-01-01", periods=4, freq="30min", tz="UTC")
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)

        with pytest.raises(ValueError, match="exact hours"):
            series_to_darts(series)

    def test_raises_on_missing_values(self):
        series = _hourly_series(periods=4)
        series.iloc[1] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            series_to_darts(series)


class TestTrainNbeats:
    def test_raises_when_training_series_is_too_short(self):
        series = _hourly_series(periods=40)

        with pytest.raises(ValueError, match="too short"):
            train_nbeats(
                series,
                input_chunk_length=24,
                output_chunk_length=24,
                n_epochs=1,
            )

    def test_raises_on_non_positive_hyperparameters(self):
        series = _hourly_series(periods=60)

        with pytest.raises(ValueError, match="input_chunk_length"):
            train_nbeats(series, input_chunk_length=0, n_epochs=1)

        with pytest.raises(ValueError, match="output_chunk_length"):
            train_nbeats(series, output_chunk_length=0, n_epochs=1)

        with pytest.raises(ValueError, match="n_epochs"):
            train_nbeats(series, n_epochs=0)


class TestPredict:
    def test_raises_when_horizon_does_not_match_target_index(self):
        target_index = pd.date_range(
            "2026-01-03", periods=24, freq="h", tz="UTC"
        )

        with pytest.raises(ValueError, match="horizon"):
            predict(None, None, None, horizon=23, target_index=target_index)

    def test_raises_on_non_hourly_target_index(self):
        target_index = pd.date_range(
            "2026-01-03", periods=4, freq="30min", tz="UTC"
        )

        with pytest.raises(ValueError, match="hourly"):
            predict(None, None, None, horizon=4, target_index=target_index)

    def test_raises_on_non_utc_target_index(self):
        target_index = pd.date_range(
            "2026-01-03", periods=4, freq="h", tz="Europe/Bucharest"
        )

        with pytest.raises(ValueError, match="UTC"):
            predict(None, None, None, horizon=4, target_index=target_index)

    def test_raises_on_unsorted_target_index(self):
        target_index = pd.date_range(
            "2026-01-03", periods=4, freq="h", tz="UTC"
        )[::-1]

        with pytest.raises(ValueError, match="sorted"):
            predict(None, None, None, horizon=4, target_index=target_index)
