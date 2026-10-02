"""Forecast evaluation metrics for hourly load predictions.

This module:
    - computes Mean Absolute Error (MAE)
    - computes Root Mean Squared Error (RMSE)
    - computes Mean Absolute Percentage Error (MAPE)
    - combines the metrics into a single evaluation result

MAE and RMSE are expressed in MW. MAPE is expressed on a 0-100 percentage
scale. MAPE is less reliable when actual values are close to zero, so a
warning is logged when such values occur in the evaluation window.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
_MAPE_MIN_SAFE_VALUE = 100.0


def _validate_metric_inputs(actual: pd.Series, predicted: pd.Series) -> None:
    """Validate two aligned, finite metric series.
    
        Args:
            actual: Series containing observed values.
            predicted: Series containing forecast values.
    
        Raises:
            TypeError: If either input is not a pandas Series.
            ValueError: If the series are empty, have different indices, or
                contain NaN or infinite values.
    """
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("actual and predicted must both be pandas Series")
    if not actual.index.equals(predicted.index):
        raise ValueError(
            "actual and predicted must share the same index -- got "
            f"{len(actual)} actual points and {len(predicted)} predicted points, "
            "or a mismatched index"
        )
    if actual.empty:
        raise ValueError("actual and predicted must not be empty")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("actual and predicted must not contain NaN values")
    if not np.isfinite(actual.to_numpy(dtype=float)).all():
        raise ValueError("actual must contain only finite values")
    if not np.isfinite(predicted.to_numpy(dtype=float)).all():
        raise ValueError("predicted must contain only finite values")


def mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute Mean Absolute Error (MAE).
    
        Args:
            actual: Series containing the observed load values.
            predicted: Series containing the predicted load values. Its index
                must match ``actual``.
    
        Returns:
            Mean absolute error in MW.
    
        Raises:
            TypeError: If inputs are not pandas Series.
            ValueError: If the series are empty, misaligned, or contain non-finite values.
    """
    _validate_metric_inputs(actual, predicted)
    return float(np.mean(np.abs(actual.to_numpy() - predicted.to_numpy())))


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute Root Mean Squared Error (RMSE).
    
        Args:
            actual: Series containing the observed load values.
            predicted: Series containing the predicted load values. Its index
                must match ``actual``.
    
        Returns:
            Root mean squared error in MW.
    
        Raises:
            TypeError: If inputs are not pandas Series.
            ValueError: If the series are empty, misaligned, or contain non-finite values.
    """
    _validate_metric_inputs(actual, predicted)
    return float(np.sqrt(np.mean((actual.to_numpy() - predicted.to_numpy()) ** 2)))


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute Mean Absolute Percentage Error (MAPE).
    
        A warning is logged when any actual value is below
        ``_MAPE_MIN_SAFE_VALUE`` because small actual values can make MAPE
        unstable. Actual values equal to zero are excluded from the percentage
        calculation; this avoids division by zero and is reported by a warning.
    
        Args:
            actual: Series containing the observed load values.
            predicted: Series containing the predicted load values. Its index
                must match ``actual``.
    
        Returns:
            Mean absolute percentage error on a 0-100 scale.
    
        Raises:
            TypeError: If inputs are not pandas Series.
            ValueError: If the series are empty, misaligned, or contain non-finite values,
                or if all actual values are zero.
    """
    _validate_metric_inputs(actual, predicted)

    actual_arr = actual.to_numpy(dtype=float)
    predicted_arr = predicted.to_numpy(dtype=float)

    n_small = int((np.abs(actual_arr) < _MAPE_MIN_SAFE_VALUE).sum())
    if n_small:
        logger.warning(
            "%d/%d actual values are below %.0f MW -- MAPE may be unreliable",
            n_small,
            len(actual_arr),
            _MAPE_MIN_SAFE_VALUE,
        )

    nonzero = actual_arr != 0
    n_zero = int((~nonzero).sum())
    if n_zero:
        logger.warning(
            "%d/%d actual values are exactly zero and are excluded from MAPE",
            n_zero,
            len(actual_arr),
        )
    if not nonzero.any():
        return float("nan")

    return float(
        np.mean(
            np.abs((actual_arr[nonzero] - predicted_arr[nonzero]) / actual_arr[nonzero])
        )
        * 100
    )


def evaluate(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a forecast.
    
        Args:
            actual: Series containing the observed values.
            predicted: Series containing the forecast values. Its index must
                match the index of ``actual``.
    
        Returns:
            A dictionary containing ``MAE``, ``RMSE``, and ``MAPE`` values.
    
        Raises:
            TypeError: If inputs are not pandas Series.
            ValueError: If the series are empty, misaligned, or contain non-finite values,
                or if MAPE cannot be computed because all actual values are zero.
    """
    _validate_metric_inputs(actual, predicted)
    return {
        "MAE": mae(actual, predicted),
        "RMSE": rmse(actual, predicted),
        "MAPE": mape(actual, predicted),
    }
