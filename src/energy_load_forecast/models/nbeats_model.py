"""N-BEATS point-forecast model for hourly load using Darts.

This module:
    - converts pandas Series to Darts TimeSeries objects
    - scales the training series
    - trains an N-BEATS model
    - generates point forecasts and converts them back to pandas Series

The model uses the historical load series directly and does not use the
calendar, lag, or rolling features used by the LightGBM models.

Darts requires tz-naive timestamps, so timestamps are converted to naive
values before creating a TimeSeries. Forecast timestamps are restored to the
UTC-aware index expected by the rest of the pipeline.

The scaler is fitted only on the training data and reused for validation,
test, and prediction.
"""
from __future__ import annotations

import pandas as pd
import torch
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NBEATSModel

torch.set_float32_matmul_precision("medium")


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


def series_to_darts(series: pd.Series) -> TimeSeries:
    """Convert a pandas Series to a Darts TimeSeries.
    
        Darts requires timezone-naive timestamps, so the UTC timezone is removed
        before creating the TimeSeries.
    
        Args:
            series: Hourly load series indexed by continuous UTC timestamps.
    
        Returns:
            A Darts TimeSeries with a timezone-naive DatetimeIndex.
    
        Raises:
            TypeError: If ``series`` is not a pandas Series.
            ValueError: If ``series`` does not satisfy the hourly UTC requirements.
    """
    _validate_hourly_series(series)
    naive = series.copy()
    naive.index = naive.index.tz_localize(None)
    return TimeSeries.from_series(naive)


def train_nbeats(
    train_series: pd.Series,
    input_chunk_length: int = 24,
    output_chunk_length: int = 24,
    n_epochs: int = 50,
    random_state: int = 42,
    use_gpu: bool = False,
) -> tuple[NBEATSModel, Scaler, TimeSeries]:
    """Train an N-BEATS model on the training load series.
    
        The scaler is fitted only on ``train_series`` and returned together with
        the trained model for reuse during prediction. The requested input and
        output chunks must fit inside the available training history.
    
        Args:
            train_series: Hourly ``load_mw`` training series.
            input_chunk_length: Number of historical hours used for each
                prediction.
            output_chunk_length: Number of hours predicted by the model.
            n_epochs: Number of training epochs.
            random_state: Random seed used for reproducible training.
            use_gpu: Whether to train using a GPU. Defaults to False.
    
        Returns:
            A tuple containing the fitted model, the scaler fitted on the
            training data, and the scaled training TimeSeries.
    
        Raises:
            ValueError: If model window lengths, epoch count, or training data
                length are invalid.
            RuntimeError: If GPU training is requested but CUDA is unavailable.
    """
    _validate_hourly_series(train_series)
    if not isinstance(input_chunk_length, int) or input_chunk_length <= 0:
        raise ValueError("input_chunk_length must be a positive integer")
    if not isinstance(output_chunk_length, int) or output_chunk_length <= 0:
        raise ValueError("output_chunk_length must be a positive integer")
    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError("n_epochs must be a positive integer")
    if len(train_series) < input_chunk_length + output_chunk_length:
        raise ValueError(
            "Training series is too short for the requested input/output chunks: "
            f"need at least {input_chunk_length + output_chunk_length} points, "
            f"got {len(train_series)}"
        )
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA is not available")

    train_ts = series_to_darts(train_series)
    scaler = Scaler()
    train_scaled = scaler.fit_transform(train_ts)

    pl_trainer_kwargs = (
        {"accelerator": "gpu", "devices": [0]}
        if use_gpu
        else {"accelerator": "cpu"}
    )
    model = NBEATSModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=output_chunk_length,
        n_epochs=n_epochs,
        random_state=random_state,
        pl_trainer_kwargs=pl_trainer_kwargs,
    )
    model.fit(train_scaled, verbose=False)
    return model, scaler, train_scaled


def predict(
    model: NBEATSModel,
    scaler: Scaler,
    history_scaled: TimeSeries,
    horizon: int,
    target_index: pd.DatetimeIndex,
) -> pd.Series:
    """Generate an N-BEATS forecast for the requested horizon.
    
        The forecast is generated from the scaled history and then transformed
        back to the original load scale using the scaler fitted on the training
        data. The target index must describe the next contiguous hourly UTC period.
    
        Args:
            model: Fitted N-BEATS model.
            scaler: Scaler fitted on the training data.
            history_scaled: Scaled historical series used as model input.
            horizon: Number of future hours to forecast.
            target_index: UTC-aware DatetimeIndex for the forecast period.
    
        Returns:
            A pandas Series containing predicted load values, named ``forecast``
            and indexed by ``target_index``.
    
        Raises:
            TypeError: If ``target_index`` is not a DatetimeIndex.
            ValueError: If horizon or target index is invalid or not hourly UTC.
    """
    if not isinstance(target_index, pd.DatetimeIndex):
        raise TypeError("target_index must be a DatetimeIndex")
    if target_index.tz is None or str(target_index.tz) != "UTC":
        raise ValueError("target_index must be UTC")
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    if horizon != len(target_index):
        raise ValueError(
            f"horizon ({horizon}) must match len(target_index) ({len(target_index)})"
        )
    if target_index.has_duplicates or not target_index.is_monotonic_increasing:
        raise ValueError("target_index must be sorted and unique")
    deltas = target_index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("target_index must have exactly hourly spacing")

    forecast_scaled = model.predict(n=horizon, series=history_scaled)
    forecast_series = scaler.inverse_transform(forecast_scaled).to_series()
    forecast_series.index = target_index
    forecast_series.name = "forecast"
    return forecast_series
