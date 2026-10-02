"""Utilities for analyzing forecasting errors and residuals.

The functions in this module keep error computation separate from model
training and evaluation. They support residual analysis, grouped error
summaries, and inspection of the largest absolute forecast errors.
"""
from __future__ import annotations

import pandas as pd


def _validate_aligned(actual: pd.Series, predicted: pd.Series) -> None:
    """Validate two aligned non-empty Series.
    
        Args:
            actual: Series containing observed values.
            predicted: Series containing forecast values.
    
        Raises:
            TypeError: If either input is not a pandas Series.
            ValueError: If the series are empty, have different indices, or
                contain NaN values.
    """
    if not isinstance(actual, pd.Series) or not isinstance(predicted, pd.Series):
        raise TypeError("actual and predicted must both be pandas Series")
    if not actual.index.equals(predicted.index):
        raise ValueError("actual and predicted must have the same index")
    if actual.empty:
        raise ValueError("actual and predicted must not be empty")
    if actual.isna().any() or predicted.isna().any():
        raise ValueError("actual and predicted must not contain NaN values")


def compute_residuals(actual: pd.Series, predicted: pd.Series) -> pd.Series:
    """Compute residuals as ``actual - predicted``.
    
        Args:
            actual: Series containing the observed values.
            predicted: Series containing the predicted values. Its index must
                match ``actual``.
    
        Returns:
            A Series containing ``actual - predicted``, named ``residual``.
    
        Raises:
            TypeError: If the inputs are not pandas Series.
            ValueError: If the inputs are empty, misaligned, or contain NaN values.
    """
    _validate_aligned(actual, predicted)
    return (actual - predicted).rename("residual")


def residuals_by_group(residuals: pd.Series, group: pd.Series) -> pd.Series:
    """Compute mean absolute residual for each group.
    
        Args:
            residuals: Series containing residual values.
            group: Series containing group labels aligned with ``residuals``.
    
        Returns:
            A Series containing the mean absolute residual for each group,
            named ``mean_abs_residual``.
    
        Raises:
            TypeError: If either input is not a pandas Series.
            ValueError: If the inputs are empty, misaligned, or contain NaN values.
    """
    if not isinstance(residuals, pd.Series) or not isinstance(group, pd.Series):
        raise TypeError("residuals and group must both be pandas Series")
    if not residuals.index.equals(group.index):
        raise ValueError("residuals and group must have the same index")
    if residuals.empty:
        raise ValueError("residuals and group must not be empty")
    if residuals.isna().any():
        raise ValueError("residuals must not contain NaN values")

    result = residuals.abs().groupby(group).mean()
    result.name = "mean_abs_residual"
    return result


def top_n_errors(
    actual: pd.Series,
    predicted: pd.Series,
    n: int = 5,
) -> pd.DataFrame:
    """Return the rows with the largest absolute prediction errors.
    
        Args:
            actual: Series containing the observed values.
            predicted: Series containing the predicted values.
            n: Number of largest errors to return. Defaults to 5.
    
        Returns:
            A DataFrame containing ``actual``, ``predicted``, and ``abs_error``
            columns for the ``n`` largest absolute errors.
    
        Raises:
            TypeError: If the inputs are not pandas Series.
            ValueError: If the inputs are empty, misaligned, contain NaN values,
                or ``n`` is not a positive integer.
    """
    _validate_aligned(actual, predicted)
    if not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be a positive integer, got {n!r}")

    abs_error = (actual - predicted).abs()
    top_n_idx = abs_error.nlargest(n).index
    return pd.DataFrame(
        {
            "actual": actual.loc[top_n_idx],
            "predicted": predicted.loc[top_n_idx],
            "abs_error": abs_error.loc[top_n_idx],
        }
    )
