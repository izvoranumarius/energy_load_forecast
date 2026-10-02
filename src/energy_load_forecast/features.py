"""Create lag and rolling features for hourly load forecasting.

This module:
    - adds lagged load features based on previous observations
    - adds rolling mean features based only on previous observations
    - combines lag and rolling features into a single DataFrame

All features use information strictly before the current timestamp to prevent
data leakage. NaN values caused by insufficient history are kept for downstream
processing. The expected time index is a continuous hourly UTC index.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _validate_hourly_frame(df: pd.DataFrame) -> None:
    """Validate the time index expected by lag and rolling features.
    
        Args:
            df: DataFrame whose index should be a continuous hourly UTC
                DatetimeIndex.
    
        Raises:
            TypeError: If ``df`` does not have a DatetimeIndex.
            ValueError: If the index is not UTC, sorted, unique, hourly, or
                aligned to exact hours.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("df index must be UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted")
    if ((df.index.minute != 0) | (df.index.second != 0) | (df.index.microsecond != 0)).any():
        raise ValueError("df index must be aligned to exact hours")
    if df.index.has_duplicates:
        raise ValueError("df index contains duplicate timestamps")
    deltas = df.index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("df index must have exactly hourly spacing")


def add_lag_features(
    df: pd.DataFrame, lags: tuple[int, ...] = (1, 24, 168)
) -> pd.DataFrame:
    """Add lagged load features to a DataFrame.
    
        Each lag uses the load value from a previous hour and therefore does not
        use the current or future load value.
    
        Args:
            df: DataFrame containing a ``load_mw`` column and a continuous hourly
                UTC index.
            lags: Lag lengths in hours. Defaults to 1, 24, and 168 hours.
    
        Returns:
            A copy of ``df`` with one ``load_mw_lag{N}`` column for each lag.
            Rows without enough previous history contain NaN.
    
        Raises:
            KeyError: If ``df`` does not contain a ``load_mw`` column.
            ValueError: If a lag is not a positive integer.
    """
    if "load_mw" not in df.columns:
        raise KeyError("Expected a 'load_mw' column in df")
    _validate_hourly_frame(df)
    if any(not isinstance(lag, int) or lag <= 0 for lag in lags):
        raise ValueError(f"lags must contain positive integers, got {lags!r}")
    if len(set(lags)) != len(lags):
        raise ValueError(f"lags must not contain duplicates, got {lags!r}")

    out = df.copy()
    for lag in lags:
        out[f"load_mw_lag{lag}"] = out["load_mw"].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame, windows: tuple[int, ...] = (3, 24)
) -> pd.DataFrame:
    """Add rolling mean features to a DataFrame.
    
        The rolling means are calculated using only previous load values so that
        the current load value is not included in the feature.
    
        Args:
            df: DataFrame containing a ``load_mw`` column and a continuous hourly
                UTC index.
            windows: Rolling window lengths in hours. Defaults to 3 and 24 hours.
    
        Returns:
            A copy of ``df`` with one ``load_mw_rolling_mean_{N}h`` column for each
            window. Rows without enough previous history contain NaN.
    
        Raises:
            KeyError: If ``df`` does not contain a ``load_mw`` column.
            ValueError: If a rolling window is not a positive integer.
    """
    if "load_mw" not in df.columns:
        raise KeyError("Expected a 'load_mw' column in df")
    _validate_hourly_frame(df)
    if any(not isinstance(window, int) or window <= 0 for window in windows):
        raise ValueError(f"windows must contain positive integers, got {windows!r}")
    if len(set(windows)) != len(windows):
        raise ValueError(f"windows must not contain duplicates, got {windows!r}")

    out = df.copy()
    shifted = out["load_mw"].shift(1)
    for window in windows:
        out[f"load_mw_rolling_mean_{window}h"] = shifted.rolling(window=window).mean()
    return out


def add_all_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 24, 168),
    windows: tuple[int, ...] = (3, 24),
) -> pd.DataFrame:
    """Add lag and rolling mean features to a DataFrame.
    
        Lag features are added first, followed by rolling mean features. All
        generated features use only information available before each timestamp.
    
        Args:
            df: DataFrame containing a ``load_mw`` column and a continuous hourly
                UTC index.
            lags: Lag lengths in hours passed to ``add_lag_features``.
            windows: Rolling window lengths in hours passed to
                ``add_rolling_features``.
    
        Returns:
            A copy of ``df`` with the requested lag and rolling mean features added.
            NaN values caused by insufficient previous history are preserved.
    
        Raises:
            KeyError: If ``df`` does not contain a ``load_mw`` column.
            ValueError: If any lag or rolling window is not a positive integer.
    """
    out = add_lag_features(df, lags=lags)
    out = add_rolling_features(out, windows=windows)

    n_nan_rows = int(out.isna().any(axis=1).sum())
    if n_nan_rows:
        logger.info(
            "%d/%d rows contain NaN values, mainly from insufficient feature history",
            n_nan_rows,
            len(out),
        )
    return out
