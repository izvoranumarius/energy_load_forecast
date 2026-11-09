"""LightGBM models for hourly load forecasting.

This module:
    - trains a LightGBM model for point forecasts
    - trains separate LightGBM models for quantile forecasts
    - generates point and quantile predictions
    - reports feature importance
    - produces recursive multi-step forecasts from a fitted model
    - builds a forecasting function that retrains at each backtest origin
    - optionally adds exogenous features, such as temperature, that are known
      for the forecast hours

The model wrapper is kept small so that feature engineering and evaluation
remain separate from model training.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from energy_load_forecast.features import add_all_features
from energy_load_forecast.preprocessing import (
    add_calendar_features,
    add_holiday_features,
)


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


def _select_exog(exog: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Select exogenous features for the requested timestamps.
    
        Args:
            exog: DataFrame of exogenous features indexed by hourly UTC
                timestamps.
            index: Timestamps for which features are required.
    
        Returns:
            The rows of ``exog`` for ``index``, in the same order.
    
        Raises:
            TypeError: If ``exog`` is not a pandas DataFrame.
            ValueError: If ``exog`` is missing timestamps or contains NaN values
                for the requested timestamps.
    """
    if not isinstance(exog, pd.DataFrame):
        raise TypeError("exog must be a pandas DataFrame")
    missing = index.difference(exog.index)
    if not missing.empty:
        raise ValueError(
            f"exog is missing {len(missing)} timestamps, first missing: {missing[0]}"
        )
    selected = exog.loc[index]
    if selected.isna().any().any():
        raise ValueError("exog contains NaN values for the requested timestamps")
    return selected


def recursive_forecast(
    model: LGBMRegressor,
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    feature_columns: list[str],
    lags: tuple[int, ...] = (1, 24, 168),
    windows: tuple[int, ...] = (3, 24),
    tz: str = "Europe/Bucharest",
    country: str = "RO",
    exog: pd.DataFrame | None = None,
) -> pd.Series:
    """Forecast each future hour using previous observations or predictions.
    
        Lag and rolling features for the first forecast hour are built from the
        observed history. Each prediction is then appended to the history and
        used for the following hours, so actual values from the forecast
        window never enter the features. Feature names follow
        ``features.add_all_features``, and calendar and holiday features follow
        ``preprocessing.add_calendar_features`` and
        ``preprocessing.add_holiday_features``.
    
        Args:
            model: Fitted LightGBM model.
            history: Hourly, UTC-indexed load series ending right before
                ``future_index``.
            future_index: Hourly UTC timestamps to forecast.
            feature_columns: Feature names in the order used to train ``model``.
            lags: Lag lengths in hours used when training ``model``.
            windows: Rolling window lengths in hours used when training ``model``.
            tz: IANA timezone used for calendar features. Defaults to
                ``"Europe/Bucharest"``.
            country: Country code used for holiday features. Defaults to
                ``"RO"``.
            exog: Optional DataFrame of exogenous features, such as
                ``temperature_c``, indexed by hourly UTC timestamps. It must
                cover every timestamp in ``future_index``. For an honest
                forecast these values must be known in advance, for example
                weather forecasts rather than observed weather.
    
        Returns:
            A Series of predicted load values indexed by ``future_index`` and
            named ``forecast``.
    
        Raises:
            TypeError: If ``history`` or ``future_index`` has the wrong type.
            ValueError: If the history is too short for the requested lags and
                windows, ``future_index`` does not start right after it, or
                ``exog`` does not cover ``future_index``.
            KeyError: If ``feature_columns`` contains a feature that cannot be
                built from ``lags``, ``windows``, the calendar and holiday
                features, and ``exog``.
    """
    if not isinstance(history, pd.Series):
        raise TypeError("history must be a pandas Series")
    if not isinstance(future_index, pd.DatetimeIndex) or future_index.empty:
        raise TypeError("future_index must be a non-empty DatetimeIndex")
    if history.isna().any():
        raise ValueError("history must not contain NaN values")
    required_history = max((*lags, *windows))
    if len(history) < required_history:
        raise ValueError(
            f"Need at least {required_history} hours of history, got {len(history)}"
        )
    if future_index[0] != history.index[-1] + pd.Timedelta(hours=1):
        raise ValueError("future_index must start one hour after the end of history")

    calendar = add_calendar_features(pd.DataFrame(index=future_index), tz=tz)
    calendar = add_holiday_features(calendar, country=country, tz=tz)
    if exog is not None:
        calendar = calendar.join(_select_exog(exog, future_index))
    values = list(history.to_numpy(dtype=float))
    predictions: list[float] = []

    for timestamp in future_index:
        row = calendar.loc[timestamp].to_dict()
        for lag in lags:
            row[f"load_mw_lag{lag}"] = values[-lag]
        for window in windows:
            row[f"load_mw_rolling_mean_{window}h"] = float(np.mean(values[-window:]))

        X_row = pd.DataFrame([row], index=[timestamp])[feature_columns]
        prediction = float(np.asarray(model.predict(X_row)).reshape(-1)[0])
        predictions.append(prediction)
        values.append(prediction)

    return pd.Series(predictions, index=future_index, name="forecast")


def make_lightgbm_forecaster(
    lags: tuple[int, ...] = (1, 24, 168),
    windows: tuple[int, ...] = (3, 24),
    tz: str = "Europe/Bucharest",
    country: str = "RO",
    exog: pd.DataFrame | None = None,
    **lgbm_params,
) -> Callable[[pd.Series, int], pd.Series]:
    """Build a forecasting function that retrains LightGBM on each history.
    
        The returned function matches the ``forecast_fn(history, horizon)``
        signature used by ``backtesting.rolling_origin_backtest``. Every call
        builds calendar, holiday, lag, and rolling features from ``history`` only,
        adds the optional exogenous features, removes rows without enough
        feature history, trains a new model, and forecasts the next ``horizon``
        hours recursively.
    
        Args:
            lags: Lag lengths in hours. Defaults to 1, 24, and 168 hours.
            windows: Rolling window lengths in hours. Defaults to 3 and 24 hours.
            tz: IANA timezone used for calendar features. Defaults to
                ``"Europe/Bucharest"``.
            country: Country code used for holiday features. Defaults to
                ``"RO"``.
            exog: Optional DataFrame of exogenous features indexed by hourly
                UTC timestamps. It must cover both the history and the forecast
                hours of every call.
            **lgbm_params: Parameters passed to ``train_lightgbm``.
    
        Returns:
            A callable that takes an hourly UTC history and a horizon and
            returns a Series of forecasts named ``forecast``.
    """

    def forecast(history: pd.Series, horizon: int) -> pd.Series:
        frame = add_calendar_features(history.to_frame(name="load_mw"), tz=tz)
        frame = add_holiday_features(frame, country=country, tz=tz)
        if exog is not None:
            frame = frame.join(_select_exog(exog, frame.index))
        features = add_all_features(frame, lags=lags, windows=windows).dropna()
        X_train = features.drop(columns=["load_mw"])
        model = train_lightgbm(X_train, features["load_mw"], **lgbm_params)

        future_index = pd.date_range(
            history.index[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )
        return recursive_forecast(
            model,
            history,
            future_index,
            list(X_train.columns),
            lags=lags,
            windows=windows,
            tz=tz,
            country=country,
            exog=exog,
        )

    return forecast
