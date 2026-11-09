"""Rolling-origin backtesting for hourly load forecasts.

This module:
    - builds a sequence of forecast origins at the end of a series
    - runs a forecasting function once from each origin
    - computes MAE, RMSE, and MAPE for every forecast window
    - summarizes and compares models across all forecast windows

A single 24-hour test window is too small to compare models reliably, because
the result depends strongly on which day is selected. Rolling-origin
backtesting repeats the same forecast task from several origins and averages
the metrics, which gives a more stable estimate of forecast accuracy.

The forecasting function receives only the history strictly before each
origin, so actual values from the forecast window cannot enter the model
inputs. Any model retraining is the responsibility of the forecasting
function and happens once per origin.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from energy_load_forecast.evaluation import evaluate

logger = logging.getLogger(__name__)

ForecastFn = Callable[[pd.Series, int], pd.Series]


def _validate_hourly_series(series: pd.Series) -> None:
    """Validate a non-empty hourly UTC load series.

        Args:
            series: Load series expected to have a continuous hourly UTC index.

        Raises:
            TypeError: If the input is not a pandas Series or does not use a
                DatetimeIndex.
            ValueError: If the series is empty, contains duplicates, is not UTC,
                is not sorted, is not hourly, or contains NaN values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series must not be empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError("series index must be UTC")
    if not series.index.is_monotonic_increasing:
        raise ValueError("series index must be sorted")
    if ((series.index.minute != 0) | (series.index.second != 0) | (series.index.microsecond != 0)).any():
        raise ValueError("series index must be aligned to exact hours")
    if series.index.has_duplicates:
        raise ValueError("series index contains duplicate timestamps")
    deltas = series.index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("series index must have exactly hourly spacing")
    if series.isna().any():
        raise ValueError("series must not contain NaN values")


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer.

        Args:
            value: Value to validate.
            name: Parameter name used in the error message.

        Raises:
            ValueError: If ``value`` is not a positive integer.
    """
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def make_origins(
    index: pd.DatetimeIndex,
    n_origins: int,
    horizon: int = 24,
    step: int = 24,
) -> pd.DatetimeIndex:
    """Build forecast origins that end at the last timestamp of an index.

        An origin is the first timestamp of a forecast window. The last origin
        is chosen so that its window ends exactly at ``index[-1]``. Earlier
        origins are placed ``step`` hours apart, moving backwards in time.

        Args:
            index: Hourly UTC DatetimeIndex of the full series.
            n_origins: Number of forecast origins to create.
            horizon: Number of hours forecast from each origin. Defaults to 24.
            step: Number of hours between consecutive origins. Defaults to 24,
                which gives one forecast per day. A step smaller than the
                horizon produces overlapping forecast windows.

        Returns:
            A sorted UTC DatetimeIndex containing ``n_origins`` origins.

        Raises:
            TypeError: If ``index`` is not a DatetimeIndex.
            ValueError: If an argument is not a positive integer, the index is
                not UTC, or the series is too short to leave history before the
                earliest origin.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if index.tz is None or str(index.tz) != "UTC":
        raise ValueError("index must be UTC")
    _validate_positive_int(n_origins, "n_origins")
    _validate_positive_int(horizon, "horizon")
    _validate_positive_int(step, "step")

    last_origin = index[-1] - pd.Timedelta(hours=horizon - 1)
    first_origin = last_origin - pd.Timedelta(hours=step * (n_origins - 1))
    if first_origin <= index[0]:
        raise ValueError(
            f"Not enough data for {n_origins} origins with horizon={horizon} "
            f"and step={step}: the earliest origin {first_origin} leaves no "
            f"history before it (data starts at {index[0]})"
        )

    return pd.date_range(
        first_origin,
        periods=n_origins,
        freq=pd.Timedelta(hours=step),
        name="origin",
    )


def rolling_origin_backtest(
    series: pd.Series,
    forecast_fn: ForecastFn,
    origins: pd.DatetimeIndex,
    horizon: int = 24,
) -> pd.DataFrame:
    """Run a forecasting function from each origin and collect the forecasts.

        For every origin, the forecasting function receives the history strictly
        before the origin and must return ``horizon`` hourly forecasts starting
        at the origin.

        Args:
            series: Hourly, UTC-indexed ``load_mw`` series without NaN values.
            forecast_fn: Callable with signature ``forecast_fn(history, horizon)``
                that returns a Series indexed by the next ``horizon`` hourly UTC
                timestamps after ``history``.
            origins: Forecast origins, usually created by ``make_origins``.
            horizon: Number of hours forecast from each origin. Defaults to 24.

        Returns:
            A DataFrame indexed by ``timestamp_utc`` with the columns ``origin``,
            ``actual``, and ``forecast``. Rows from all origins are stacked in
            chronological order of their origins.

        Raises:
            TypeError: If the inputs have the wrong type.
            ValueError: If the series is invalid, an origin is outside the
                series or has no history, or a forecast does not match the
                expected forecast window.
    """
    _validate_hourly_series(series)
    _validate_positive_int(horizon, "horizon")
    if not callable(forecast_fn):
        raise TypeError("forecast_fn must be callable")
    if not isinstance(origins, pd.DatetimeIndex) or origins.empty:
        raise ValueError("origins must be a non-empty DatetimeIndex")

    windows: list[pd.DataFrame] = []
    for origin in origins:
        window_index = pd.date_range(origin, periods=horizon, freq="h")
        if window_index[0] <= series.index[0]:
            raise ValueError(f"Origin {origin} leaves no history before it")
        if window_index[-1] > series.index[-1]:
            raise ValueError(
                f"Forecast window from origin {origin} ends after the series "
                f"({window_index[-1]} > {series.index[-1]})"
            )

        history = series.loc[series.index < origin]
        actual = series.loc[window_index]
        forecast = forecast_fn(history, horizon)

        if not isinstance(forecast, pd.Series):
            raise TypeError(
                f"forecast_fn must return a pandas Series, got {type(forecast).__name__}"
            )
        if not forecast.index.equals(actual.index):
            raise ValueError(
                f"Forecast from origin {origin} does not cover the expected "
                f"window {window_index[0]} to {window_index[-1]}"
            )

        windows.append(
            pd.DataFrame(
                {
                    "origin": origin,
                    "actual": actual.to_numpy(dtype=float),
                    "forecast": forecast.to_numpy(dtype=float),
                },
                index=actual.index.rename("timestamp_utc"),
            )
        )

    logger.info("Backtested %d origins with horizon=%d", len(origins), horizon)
    return pd.concat(windows)


def backtest_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Compute MAE, RMSE, and MAPE separately for each forecast origin.

        Args:
            results: DataFrame returned by ``rolling_origin_backtest``.

        Returns:
            A DataFrame indexed by ``origin`` with ``MAE``, ``RMSE``, and
            ``MAPE`` columns.

        Raises:
            KeyError: If the expected backtest columns are missing.
            ValueError: If ``results`` is empty.
    """
    missing_cols = {"origin", "actual", "forecast"} - set(results.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    if results.empty:
        raise ValueError("results must not be empty")

    rows = {
        origin: evaluate(window["actual"], window["forecast"])
        for origin, window in results.groupby("origin", sort=True)
    }
    metrics = pd.DataFrame.from_dict(rows, orient="index")
    metrics.index.name = "origin"
    return metrics


def summarize_backtest(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize per-origin metrics with their mean and standard deviation.

        The standard deviation shows how much accuracy changes between forecast
        days. A large value means that a single test window would be an
        unreliable basis for comparing models.

        Args:
            metrics: DataFrame returned by ``backtest_metrics``.

        Returns:
            A DataFrame with one row per metric and the columns ``mean``,
            ``std``, ``min``, and ``max``.

        Raises:
            ValueError: If ``metrics`` is empty.
    """
    if metrics.empty:
        raise ValueError("metrics must not be empty")
    return metrics.agg(["mean", "std", "min", "max"]).T


def compare_models(
    series: pd.Series,
    forecast_fns: dict[str, ForecastFn],
    origins: pd.DatetimeIndex,
    horizon: int = 24,
) -> pd.DataFrame:
    """Backtest several models on the same origins and compare mean metrics.

        Every model is evaluated on exactly the same forecast windows so that
        the comparison is fair.

        Args:
            series: Hourly, UTC-indexed ``load_mw`` series without NaN values.
            forecast_fns: Dictionary mapping each model name to its forecasting
                function.
            origins: Forecast origins shared by all models.
            horizon: Number of hours forecast from each origin. Defaults to 24.

        Returns:
            A DataFrame indexed by model name with the mean ``MAE``, ``RMSE``,
            and ``MAPE`` across all origins.

        Raises:
            ValueError: If ``forecast_fns`` is empty or a backtest fails.
    """
    if not forecast_fns:
        raise ValueError("forecast_fns must not be empty")

    rows: dict[str, pd.Series] = {}
    for name, forecast_fn in forecast_fns.items():
        results = rolling_origin_backtest(series, forecast_fn, origins, horizon=horizon)
        rows[name] = backtest_metrics(results).mean()
        logger.info("%s: mean MAE %.1f MW", name, rows[name]["MAE"])

    comparison = pd.DataFrame(rows).T
    comparison.index.name = "model"
    return comparison
