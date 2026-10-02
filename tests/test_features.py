"""Tests for the lag and rolling feature helpers.

The leakage tests are the whole point of this module: they prove that
changing a future value cannot change a feature computed for an earlier
row. Everything else is secondary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.features import (
    add_all_features,
    add_lag_features,
    add_rolling_features,
)


def _df(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.DataFrame({"load_mw": values}, index=index)


# ---------------------------------------------------------------------------
# add_lag_features
# ---------------------------------------------------------------------------


class TestAddLagFeatures:
    def test_lag1_shifts_by_one_hour(self):
        df = _df([10.0, 20.0, 30.0, 40.0])

        result = add_lag_features(df, lags=[1])

        assert result["load_mw_lag1"].iloc[0:1].isna().all()
        assert result["load_mw_lag1"].iloc[1:].tolist() == [10.0, 20.0, 30.0]

    def test_first_n_rows_are_nan_for_lag_n(self):
        df = _df(list(range(30)))

        result = add_lag_features(df, lags=[24])

        assert result["load_mw_lag24"].iloc[:24].isna().all()
        assert result["load_mw_lag24"].iloc[24:].isna().sum() == 0

    def test_lag_longer_than_series_is_entirely_nan(self):
        """Simulates the current 7-day dataset with a 168h (weekly) lag:
        there isn't a week of history yet, so the whole column is NaN --
        this must not raise, just produce an all-NaN column."""
        df = _df(list(range(48))) 

        result = add_lag_features(df, lags=[168])

        assert result["load_mw_lag168"].isna().all()

    def test_does_not_mutate_input_df(self):
        df = _df([10.0, 20.0, 30.0])
        original_columns = list(df.columns)

        add_lag_features(df, lags=[1])

        assert list(df.columns) == original_columns

    def test_raises_without_load_mw_column(self):
        df = pd.DataFrame({"other": [1, 2, 3]})

        with pytest.raises(KeyError, match="load_mw"):
            add_lag_features(df, lags=[1])

    def test_raises_on_invalid_lag(self):
        df = _df([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="positive integers"):
            add_lag_features(df, lags=[0])

    def test_raises_on_duplicate_lag(self):
        df = _df([10.0, 20.0, 30.0])

        with pytest.raises(ValueError, match="duplicates"):
            add_lag_features(df, lags=[1, 1])

    def test_no_leakage_from_future_values(self):
        """The core guarantee: changing a future value must not change
        any lag feature computed for an earlier row."""
        df = _df([10.0, 20.0, 30.0, 40.0, 50.0])
        result_before = add_lag_features(df.copy(), lags=[1])

        df_modified = df.copy()
        df_modified.loc[df_modified.index[-1], "load_mw"] = 9999.0
        result_after = add_lag_features(df_modified, lags=[1])

        pd.testing.assert_series_equal(
            result_before["load_mw_lag1"].iloc[:-1],
            result_after["load_mw_lag1"].iloc[:-1],
        )


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    def test_window_excludes_current_hour(self):
        """The classic bug this guards against: rolling(w).mean() without
        a prior shift(1) would include row t's own value."""
        df = _df([10.0, 20.0, 30.0, 40.0])

        result = add_rolling_features(df, windows=[2])

        assert result["load_mw_rolling_mean_2h"].iloc[2] == pytest.approx(15.0)

    def test_first_window_rows_are_nan(self):
        df = _df(list(range(10)))

        result = add_rolling_features(df, windows=[3])

        assert result["load_mw_rolling_mean_3h"].iloc[:3].isna().all()

    def test_no_leakage_from_future_values(self):
        df = _df([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        result_before = add_rolling_features(df.copy(), windows=[3])

        df_modified = df.copy()
        df_modified.loc[df_modified.index[-1], "load_mw"] = 9999.0
        result_after = add_rolling_features(df_modified, windows=[3])

        pd.testing.assert_series_equal(
            result_before["load_mw_rolling_mean_3h"].iloc[:-1],
            result_after["load_mw_rolling_mean_3h"].iloc[:-1],
        )

    def test_raises_without_load_mw_column(self):
        df = pd.DataFrame({"other": [1, 2, 3]})

        with pytest.raises(KeyError, match="load_mw"):
            add_rolling_features(df, windows=[3])


# ---------------------------------------------------------------------------
# add_all_features
# ---------------------------------------------------------------------------


class TestAddAllFeatures:
    def test_adds_both_lag_and_rolling_columns(self):
        df = _df(list(range(200)))

        result = add_all_features(df, lags=[1, 24], windows=[3])

        assert "load_mw_lag1" in result.columns
        assert "load_mw_lag24" in result.columns
        assert "load_mw_rolling_mean_3h" in result.columns
