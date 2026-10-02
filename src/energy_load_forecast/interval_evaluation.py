"""Evaluation metrics for probabilistic load forecasts.

This module provides pinball loss for quantile forecasts and two complementary
interval diagnostics: empirical coverage and average interval width. Coverage
measures how often actual values fall inside the interval, while interval width
measures how wide the interval is on average.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_series_index(*series: pd.Series) -> None:
    """Validate two forecast series for interval evaluation.
    
        Args:
            actual: Observed load series.
            predicted: Forecast series to compare with ``actual``.
    
        Raises:
            ValueError: If the input indices do not match or contain invalid
                timestamps.
    """
    if not series:
        raise ValueError("At least one series is required")
    first = series[0]
    if not all(isinstance(item, pd.Series) for item in series):
        raise TypeError("All inputs must be pandas Series")
    if any(not first.index.equals(item.index) for item in series[1:]):
        raise ValueError("All input series must share the same index")
    if first.empty:
        raise ValueError("Input series must not be empty")
    if any(item.isna().any() for item in series):
        raise ValueError("Input series must not contain NaN values")
    if any(not np.isfinite(item.to_numpy(dtype=float)).all() for item in series):
        raise ValueError("Input series must contain only finite values")


def pinball_loss(actual: pd.Series, predicted: pd.Series, alpha: float) -> float:
    """Compute pinball loss for a quantile forecast.
    
        Args:
            actual: Observed values.
            predicted: Predicted quantile values.
            alpha: Quantile level between 0 and 1.
    
        Returns:
            Mean pinball loss for the requested quantile.
    
        Raises:
            ValueError: If ``alpha`` is not between 0 and 1 or the inputs are
                not aligned.
    """
    _validate_series_index(actual, predicted)
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    diff = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Compute empirical coverage of a prediction interval.
    
        Args:
            actual: Observed load values.
            lower: Lower interval bound.
            upper: Upper interval bound.
    
        Returns:
            Fraction of actual values falling between the lower and upper bounds.
    
        Raises:
            ValueError: If the series are not aligned or any lower bound exceeds
                the corresponding upper bound.
    """
    _validate_series_index(actual, lower, upper)
    if (lower > upper).any():
        raise ValueError("lower interval bound cannot exceed upper interval bound")

    within = (actual >= lower) & (actual <= upper)
    return float(within.mean() * 100)


def interval_width(lower: pd.Series, upper: pd.Series) -> float:
    """Compute the average width of a prediction interval.
    
        Args:
            lower: Lower interval bound.
            upper: Upper interval bound.
    
        Returns:
            Mean interval width in the same units as the load values.
    
        Raises:
            ValueError: If the interval bounds are not aligned or lower exceeds
                upper at any timestamp.
    """
    _validate_series_index(lower, upper)
    if (lower > upper).any():
        raise ValueError("lower interval bound cannot exceed upper interval bound")
    return float((upper - lower).mean())
