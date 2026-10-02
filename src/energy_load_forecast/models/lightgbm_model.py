"""LightGBM models for hourly load forecasting.

This module:
    - trains a LightGBM model for point forecasts
    - trains separate LightGBM models for quantile forecasts
    - generates point and quantile predictions
    - reports feature importance

The model wrapper is kept small so that feature engineering and evaluation
remain separate from model training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, **lgbm_params) -> LGBMRegressor:
    """Train a LightGBM regression model.
    
        Args:
            X_train: Feature matrix used for training. May contain NaN values,
                which LightGBM can handle natively.
            y_train: Target load values aligned with ``X_train``.
            **lgbm_params: Parameters passed to ``LGBMRegressor``. Defaults to
                ``random_state=42`` if not provided.
    
        Returns:
            A fitted ``LGBMRegressor``.
    
        Raises:
            ValueError: If ``X_train`` and ``y_train`` do not have the same index.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must share the same index")
    if X_train.empty:
        raise ValueError("Training data must not be empty")
    if X_train.shape[1] == 0:
        raise ValueError("X_train must contain at least one feature")
    if y_train.isna().any():
        raise ValueError("y_train must not contain NaN values")
    if X_train.isna().all(axis=0).any():
        raise ValueError("X_train contains a feature that is entirely NaN")

    lgbm_params.setdefault("random_state", 42)
    model = LGBMRegressor(**lgbm_params)
    model.fit(X_train, y_train)
    return model


def predict(model: LGBMRegressor, X: pd.DataFrame) -> pd.Series:
    """Generate load predictions from a fitted LightGBM model.
    
        Args:
            model: Fitted ``LGBMRegressor``.
            X: Feature matrix used for prediction.
    
        Returns:
            A Series of predicted load values indexed like ``X`` and named
            ``forecast``.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    preds = np.asarray(model.predict(X)).reshape(-1)
    return pd.Series(preds, index=X.index, name="forecast")


def feature_importance(model: LGBMRegressor, feature_names: list[str]) -> pd.Series:
    """Return LightGBM feature importances sorted in descending order.
    
        Args:
            model: Fitted ``LGBMRegressor``.
            feature_names: Feature names in the same order as the columns used
                to train the model.
    
        Returns:
            A Series containing the feature importance for each feature, sorted
            from highest to lowest importance.
    
        Raises:
            ValueError: If the number of feature names does not match the number
                of feature importances reported by the model.
    """
    importances = model.feature_importances_
    if len(feature_names) != len(importances):
        raise ValueError(
            f"Got {len(feature_names)} feature names but the model has "
            f"{len(importances)} importances"
        )
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def train_quantile_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alphas: tuple[float, ...] = (0.1, 0.5, 0.9),
    **lgbm_params,
) -> dict[float, LGBMRegressor]:
    """Train separate LightGBM models for the requested quantiles.
    
        Each quantile is trained independently using LightGBM's quantile
        objective. The resulting predictions are not guaranteed to be ordered,
        so a lower quantile can theoretically be greater than a higher quantile.
    
        Args:
            X_train: Feature matrix used for training. May contain NaN values,
                which LightGBM can handle natively.
            y_train: Target load values aligned with ``X_train``.
            alphas: Quantiles to predict, expressed as values between 0 and 1.
                Defaults to 0.1, 0.5, and 0.9.
            **lgbm_params: Parameters passed to each ``LGBMRegressor``.
                ``objective`` and ``alpha`` must not be provided because they are
                set separately for each quantile. ``random_state`` defaults to 42.
    
        Returns:
            A dictionary mapping each quantile to its fitted ``LGBMRegressor``.
    
        Raises:
            ValueError: If the training index is misaligned, if a quantile is
                outside (0, 1), or if ``objective`` or ``alpha`` is supplied in
                ``lgbm_params``.
    """
    if not isinstance(X_train, pd.DataFrame) or not isinstance(y_train, pd.Series):
        raise TypeError("X_train must be a DataFrame and y_train must be a Series")
    if not X_train.index.equals(y_train.index):
        raise ValueError("X_train and y_train must share the same index")
    if X_train.empty:
        raise ValueError("Training data must not be empty")
    if X_train.shape[1] == 0:
        raise ValueError("X_train must contain at least one feature")
    if y_train.isna().any():
        raise ValueError("y_train must not contain NaN values")
    if "objective" in lgbm_params or "alpha" in lgbm_params:
        raise ValueError("'objective' and 'alpha' are set internally per quantile")
    if not alphas or any(not isinstance(alpha, (int, float)) or not 0 < alpha < 1 for alpha in alphas):
        raise ValueError(f"alphas must contain values strictly between 0 and 1, got {alphas!r}")
    if len(set(alphas)) != len(alphas):
        raise ValueError(f"alphas must not contain duplicates, got {alphas!r}")

    lgbm_params.setdefault("random_state", 42)
    models: dict[float, LGBMRegressor] = {}
    for alpha in alphas:
        model = LGBMRegressor(objective="quantile", alpha=alpha, **lgbm_params)
        model.fit(X_train, y_train)
        models[alpha] = model
    return models


def predict_quantiles(
    models: dict[float, LGBMRegressor],
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions from trained quantile models.
    
        Args:
            models: Dictionary mapping each quantile to its fitted LightGBM model.
            X: Feature matrix used for prediction.
    
        Returns:
            A DataFrame indexed like ``X``, with one column for each quantile.
            Columns are named using the format ``q{alpha}``, such as ``q0.1``,
            ``q0.5``, and ``q0.9``.
    
        Raises:
            ValueError: If ``models`` is empty.
            TypeError: If ``X`` is not a pandas DataFrame.
    """
    if not models:
        raise ValueError("models must not be empty")
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    predictions = {
        f"q{alpha}": np.asarray(model.predict(X)).reshape(-1)
        for alpha, model in models.items()
    }
    return pd.DataFrame(predictions, index=X.index)
