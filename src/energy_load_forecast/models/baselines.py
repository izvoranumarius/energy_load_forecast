"""Naive baseline forecasting methods for hourly energy load.

These methods provide simple reference forecasts that more complex models
should be compared against.

The module includes:
    - ``naive_forecast``: Repeats the last observed value for the entire
      forecast horizon.
    - ``seasonal_naive_forecast``: Repeats the values from the most recent
      seasonal cycle.

Both methods produce the full requested horizon once from the end of the
available history.
"""
from __future__ import annotations

import pandas as pd


def _validate_history(history: pd.Series) -> None:
    """Validate an hourly UTC history series.
    
        Args:
            history: Historical load series used as the forecast input.
    
        Raises:
            TypeError: If ``history`` is not a pandas Series or does not have a
                DatetimeIndex.
            ValueError: If the history is empty, contains NaN values, is not UTC,
                or is not a continuous hourly series.
    """
    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series")
    if history.empty:
        raise ValueError("history must not be empty")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise TypeError("history must have a DatetimeIndex")
    if history.index.tz is None or str(history.index.tz) != "UTC":
        raise ValueError("history index must be UTC")
    if not history.index.is_monotonic_increasing:
        raise ValueError("history index must be sorted")
    if ((history.index.minute != 0) | (history.index.second != 0) | (history.index.microsecond != 0)).any():
        raise ValueError("history index must be aligned to exact hours")
    if history.index.has_duplicates:
        raise ValueError("history index contains duplicate timestamps")
    deltas = history.index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("history index must have exactly hourly spacing")
    if history.isna().any():
        raise ValueError("history must not contain NaN values")


def _validate_horizon(horizon: int) -> None:
    """Validate a forecast horizon.
    
        Args:
            horizon: Number of hours to forecast.
    
        Raises:
            ValueError: If ``horizon`` is not a positive integer.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive integer, got {horizon!r}")


def naive_forecast(history: pd.Series, horizon: int = 24) -> pd.Series:
    """Forecast by repeating the last observed value.
    
        Args:
            history: Hourly load series up to the forecast origin. The index must
                contain sorted, unique UTC timestamps aligned to exact hours.
            horizon: Number of hours to forecast. Defaults to 24.
    
        Returns:
            A series containing ``horizon`` forecast values, indexed by the next
            hourly UTC timestamps. Every value equals the last observed value.
    
        Raises:
            TypeError: If ``history`` is not a pandas Series.
            ValueError: If history or horizon is invalid.
    """
    _validate_history(history)
    _validate_horizon(horizon)
    future_index = pd.date_range(
        history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
        tz="UTC",
    )
    return pd.Series(history.iloc[-1], index=future_index, name="forecast")


def seasonal_naive_forecast(
    history: pd.Series,
    horizon: int = 24,
    season_length: int = 24,
) -> pd.Series:
    """Forecast by repeating the most recent seasonal cycle.
    
        Args:
            history: Hourly load series up to the forecast origin. The index must
                contain sorted, unique UTC timestamps aligned to exact hours.
            horizon: Number of hours to forecast. Defaults to 24.
            season_length: Length of the seasonal cycle in hours. A value of 24
                represents a daily seasonal pattern.
    
        Returns:
            A series containing ``horizon`` forecast values, indexed by the next
            hourly UTC timestamps. The most recent seasonal block is repeated as
            needed when the horizon is longer than one cycle.
    
        Raises:
            TypeError: If ``history`` is not a pandas Series.
            ValueError: If the history, horizon, or season length is invalid, or
                if fewer than ``season_length`` historical observations exist.
    """
    _validate_history(history)
    _validate_horizon(horizon)
    if not isinstance(season_length, int) or season_length <= 0:
        raise ValueError(
            f"season_length must be a positive integer, got {season_length!r}"
        )
    if len(history) < season_length:
        raise ValueError(
            f"Need at least {season_length} hours of history, got {len(history)}"
        )

    future_index = pd.date_range(
        history.index[-1] + pd.Timedelta(hours=1),
        periods=horizon,
        freq="h",
        tz="UTC",
    )
    seasonal_block = history.iloc[-season_length:]
    values = list(seasonal_block.to_numpy())
    values *= -(-horizon // season_length)
    return pd.Series(values[:horizon], index=future_index, name="forecast")
