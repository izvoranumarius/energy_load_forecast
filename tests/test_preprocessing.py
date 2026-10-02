"""Tests for preprocessing.py.

Each test builds a small synthetic Series or DataFrame directly in code.
This keeps the tests fast, readable, and in control of the exact scenario
being checked instead of depending on external data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.preprocessing import (
    add_calendar_features,
    chronological_split,
    clean_tail,
)


def _hourly_index(start: str, periods: int, tz: str = "UTC") -> pd.DatetimeIndex:
    """Helper: build a clean hourly UTC DatetimeIndex for test data."""
    return pd.date_range(start, periods=periods, freq="h", tz=tz)



class TestCleanTail:
    def test_trims_trailing_nan(self):
        """The classic ENTSO-E case: last few hours unpublished yet."""
        index = _hourly_index("2026-01-01", periods=10)
        values = [1.0] * 7 + [None, None, None]
        series = pd.Series(values, index=index)

        result = clean_tail(series)

        assert len(result) == 7
        assert result.isna().sum() == 0
        assert result.index.max() == index[6]

    def test_no_trailing_nan_returns_series_unchanged(self):
        index = _hourly_index("2026-01-01", periods=5)
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=index)

        result = clean_tail(series)

        pd.testing.assert_series_equal(result, series)

    def test_raises_if_nan_appears_outside_the_tail(self):
        """A gap in the MIDDLE of the series is a data quality problem,
        not a publication delay -- clean_tail must not hide it by
        trimming past it."""
        index = _hourly_index("2026-01-01", periods=6)
        values = [1.0, None, 3.0, 4.0, 5.0, None] 
        series = pd.Series(values, index=index)

        with pytest.raises(ValueError, match="not confined to the tail"):
            clean_tail(series)

    def test_raises_on_unsorted_index(self):
        index = _hourly_index("2026-01-01", periods=3)[::-1]
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with pytest.raises(ValueError, match="sorted"):
            clean_tail(series)

    def test_raises_on_all_nan_series(self):
        index = _hourly_index("2026-01-01", periods=3)
        series = pd.Series([None, None, None], index=index)

        with pytest.raises(ValueError, match="only NaN"):
            clean_tail(series)

    def test_raises_on_empty_series(self):
        series = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))

        with pytest.raises(ValueError, match="empty"):
            clean_tail(series)

    def test_raises_on_non_series_input(self):
        with pytest.raises(TypeError, match="pandas Series"):
            clean_tail([1, 2, 3]) 

    def test_raises_on_non_datetime_index(self):
        series = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])

        with pytest.raises(TypeError, match="DatetimeIndex"):
            clean_tail(series)



class TestAddCalendarFeatures:
    def test_index_stays_utc(self):
        """The base index must NEVER be converted to local time -- only
        the derived feature columns should reflect local time."""
        index = _hourly_index("2026-06-15", periods=24) 
        df = pd.DataFrame({"load_mw": range(24)}, index=index)

        result = add_calendar_features(df)

        assert result.index.equals(index)
        assert str(result.index.tz) == "UTC"

    def test_hour_local_reflects_dst_offset(self):
        """00:00 UTC in June (DST active, UTC+3) should be hour_local=3."""
        index = _hourly_index("2026-06-15", periods=1) 
        df = pd.DataFrame({"load_mw": [100.0]}, index=index)

        result = add_calendar_features(df)

        assert result["hour_local"].iloc[0] == 3

    def test_weekend_flag_correct(self):
        index = _hourly_index("2026-06-15", periods=24 * 7) 
        df = pd.DataFrame({"load_mw": range(24 * 7)}, index=index)

        result = add_calendar_features(df)

        saturdays_sundays = result["day_of_week_local"].isin([5, 6])
        assert (result["is_weekend_local"] == saturdays_sundays).all()

    def test_raises_on_tz_naive_index(self):
        index = pd.date_range("2026-01-01", periods=5, freq="h")  
        df = pd.DataFrame({"load_mw": range(5)}, index=index)

        with pytest.raises(TypeError, match="timezone-aware"):
            add_calendar_features(df)



class TestChronologicalSplit:
    def test_no_temporal_overlap_between_splits(self):
        """The core guarantee: train ends strictly before val starts, and
        val ends strictly before test starts. If this ever breaks, the
        model is silently trained on data from its own future."""
        index = _hourly_index("2026-01-01", periods=240)  
        df = pd.DataFrame({"load_mw": range(240)}, index=index)

        train, val, test = chronological_split(
            df, train_end="2026-01-06", val_end="2026-01-08"
        )

        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_split_sizes_sum_to_total(self):
        index = _hourly_index("2026-01-01", periods=240)
        df = pd.DataFrame({"load_mw": range(240)}, index=index)

        train, val, test = chronological_split(
            df, train_end="2026-01-06", val_end="2026-01-08"
        )

        assert len(train) + len(val) + len(test) == len(df)

    def test_no_row_appears_in_more_than_one_split(self):
        index = _hourly_index("2026-01-01", periods=240)
        df = pd.DataFrame({"load_mw": range(240)}, index=index)

        train, val, test = chronological_split(
            df, train_end="2026-01-06", val_end="2026-01-08"
        )

        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert train_idx.isdisjoint(val_idx)
        assert val_idx.isdisjoint(test_idx)
        assert train_idx.isdisjoint(test_idx)

    def test_raises_if_boundaries_out_of_range(self):
        """Guards against silently producing an empty val/test set."""
        index = _hourly_index("2026-01-01", periods=48) 
        df = pd.DataFrame({"load_mw": range(48)}, index=index)

        with pytest.raises(ValueError, match="Split boundaries"):
            chronological_split(df, train_end="2026-01-10", val_end="2026-01-12")

    def test_raises_if_train_end_after_val_end(self):
        index = _hourly_index("2026-01-01", periods=240)
        df = pd.DataFrame({"load_mw": range(240)}, index=index)

        with pytest.raises(ValueError, match="Split boundaries"):
            chronological_split(df, train_end="2026-01-08", val_end="2026-01-06")

    def test_raises_on_unsorted_index(self):
        index = _hourly_index("2026-01-01", periods=48)[::-1]
        df = pd.DataFrame({"load_mw": range(48)}, index=index)

        with pytest.raises(ValueError, match="sorted"):
            chronological_split(df, train_end="2026-01-02", val_end="2026-01-03")
