"""Read hourly air temperature from Open-Meteo for load forecasting.

This module:
    - fetches hourly 2 m air temperature for several locations from Open-Meteo
    - combines the locations into one population-weighted temperature series
    - writes the canonical hourly temperature series to CSV

Temperature is one of the main drivers of electricity load, through heating
in winter and cooling in summer. A single city does not represent the whole
country, so the default locations are the largest Romanian cities weighted by
approximate population.

Two sources are supported:
    - ``"forecast"``: archived weather forecasts from the Open-Meteo Historical
      Forecast API. These match what would have been known when a day-ahead
      load forecast was made, so backtest results are not too optimistic.
    - ``"observed"``: ERA5 reanalysis from the Open-Meteo Archive API. These
      are the best estimate of the actual temperature, but using them in a
      backtest assumes a perfect weather forecast.

The Open-Meteo APIs used here do not require an API key.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


_SOURCE_URLS = {
    "forecast": "https://historical-forecast-api.open-meteo.com/v1/forecast",
    "observed": "https://archive-api.open-meteo.com/v1/archive",
}
_REQUEST_TIMEOUT_SECONDS = 60

# Latitude, longitude, and approximate population in thousands (2021 census).
ROMANIA_LOCATIONS: dict[str, tuple[float, float, float]] = {
    "bucharest": (44.43, 26.10, 1716.0),
    "cluj_napoca": (46.77, 23.60, 286.0),
    "iasi": (47.16, 27.59, 271.0),
    "constanta": (44.18, 28.63, 263.0),
    "timisoara": (45.75, 21.23, 250.0),
    "brasov": (45.65, 25.61, 237.0),
    "craiova": (44.32, 23.80, 234.0),
    "galati": (45.44, 28.05, 217.0),
}


def _as_utc(value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp-like value to a UTC-aware Timestamp.

        Args:
            value: Timestamp-like value. Naive values are interpreted as UTC.

        Returns:
            A timezone-aware UTC ``pd.Timestamp``.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _validate_locations(locations: dict[str, tuple[float, float, float]]) -> None:
    """Validate location coordinates and weights.

        Args:
            locations: Dictionary mapping a location name to latitude,
                longitude, and weight.

        Raises:
            ValueError: If no locations are given, coordinates are out of range,
                or a weight is not positive.
    """
    if not locations:
        raise ValueError("locations must not be empty")
    for name, (latitude, longitude, weight) in locations.items():
        if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
            raise ValueError(f"Invalid coordinates for {name!r}: ({latitude}, {longitude})")
        if weight <= 0:
            raise ValueError(f"Weight for {name!r} must be positive, got {weight}")


def read_raw_temperature(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    locations: dict[str, tuple[float, float, float]] = ROMANIA_LOCATIONS,
    source: str = "forecast",
) -> pd.DataFrame:
    """Fetch hourly temperature for several locations from Open-Meteo.

        Args:
            start: Start of the requested range, inclusive. Naive timestamps
                are treated as UTC.
            end: End of the requested range, exclusive. Naive timestamps are
                treated as UTC.
            locations: Dictionary mapping a location name to latitude,
                longitude, and weight. Defaults to ``ROMANIA_LOCATIONS``.
            source: ``"forecast"`` for archived forecasts or ``"observed"`` for
                ERA5 reanalysis. Defaults to ``"forecast"``.

        Returns:
            A DataFrame indexed by hourly UTC timestamps in ``[start, end)``,
            with one temperature column in degrees Celsius per location.
            Missing values are preserved as NaN.

        Raises:
            ValueError: If the range, source, or locations are invalid, or the
                API response has an unexpected shape.
            RuntimeError: If the request dependencies are unavailable or the
                API request fails.
    """
    if source not in _SOURCE_URLS:
        raise ValueError(f"source must be one of {sorted(_SOURCE_URLS)}, got {source!r}")
    _validate_locations(locations)
    start = _as_utc(start)
    end = _as_utc(end)
    if start >= end:
        raise ValueError(f"Invalid range: start ({start}) must be before end ({end})")

    try:
        import requests
        import truststore
    except ImportError as exc:
        raise RuntimeError("Open-Meteo request dependencies are not installed") from exc

    truststore.inject_into_ssl()
    names = list(locations)
    params = {
        "latitude": ",".join(str(locations[name][0]) for name in names),
        "longitude": ",".join(str(locations[name][1]) for name in names),
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": (end - pd.Timedelta(hours=1)).strftime("%Y-%m-%d"),
        "hourly": "temperature_2m",
        "timezone": "GMT",
    }

    try:
        response = requests.get(
            _SOURCE_URLS[source], params=params, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.warning("Open-Meteo request failed: %s", exc)
        raise RuntimeError("Open-Meteo request failed") from exc

    # A single location returns one object; several locations return a list.
    results = payload if isinstance(payload, list) else [payload]
    if len(results) != len(names):
        raise ValueError(
            f"Expected {len(names)} locations in the Open-Meteo response, got {len(results)}"
        )

    columns: dict[str, pd.Series] = {}
    for name, result in zip(names, results, strict=True):
        try:
            hourly = result["hourly"]
            index = pd.to_datetime(hourly["time"], utc=True)
            values = pd.to_numeric(pd.Series(hourly["temperature_2m"]), errors="coerce")
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Unexpected Open-Meteo response for {name!r}") from exc
        columns[name] = pd.Series(values.to_numpy(), index=index)

    raw = pd.DataFrame(columns)
    raw.index.name = "timestamp_utc"
    raw = raw.loc[(raw.index >= start) & (raw.index < end)]

    n_missing = int(raw.isna().sum().sum())
    if n_missing:
        logger.info("%d missing temperature values across %d locations", n_missing, len(names))
    return raw


def weighted_temperature(
    raw: pd.DataFrame,
    locations: dict[str, tuple[float, float, float]] = ROMANIA_LOCATIONS,
) -> pd.Series:
    """Combine temperatures from several locations into one weighted series.

        When some locations are missing at a timestamp, the weights of the
        available locations are rescaled so that they still sum to one. A
        timestamp with no available location remains NaN.

        Args:
            raw: DataFrame returned by ``read_raw_temperature``, with one column
                per location.
            locations: Dictionary mapping a location name to latitude,
                longitude, and weight. Defaults to ``ROMANIA_LOCATIONS``.

        Returns:
            A Series named ``temperature_c`` with the weighted temperature in
            degrees Celsius, indexed like ``raw``.

        Raises:
            KeyError: If ``raw`` does not contain a column for every location.
            ValueError: If the locations are invalid.
    """
    _validate_locations(locations)
    missing_cols = set(locations) - set(raw.columns)
    if missing_cols:
        raise KeyError(f"Missing temperature columns: {missing_cols}")

    names = list(locations)
    weights = pd.Series({name: locations[name][2] for name in names})
    values = raw[names]
    available_weight = values.notna().mul(weights, axis=1).sum(axis=1)
    weighted_sum = values.mul(weights, axis=1).sum(axis=1, min_count=1)

    result = (weighted_sum / available_weight.where(available_weight > 0)).rename(
        "temperature_c"
    )
    result.index.name = "timestamp_utc"
    return result


def load_temperature(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    locations: dict[str, tuple[float, float, float]] = ROMANIA_LOCATIONS,
    source: str = "forecast",
) -> pd.Series:
    """Fetch Open-Meteo temperatures and return one weighted hourly series.

        Args:
            start: Start of the requested range, inclusive.
            end: End of the requested range, exclusive.
            locations: Dictionary mapping a location name to latitude,
                longitude, and weight. Defaults to ``ROMANIA_LOCATIONS``.
            source: ``"forecast"`` or ``"observed"``. Defaults to ``"forecast"``.

        Returns:
            Hourly, UTC-indexed ``temperature_c`` series. May contain NaN when
            no location has data for an hour.
    """
    raw = read_raw_temperature(start, end, locations=locations, source=source)
    return weighted_temperature(raw, locations=locations)


def save_temperature(series: pd.Series, out_path: Path) -> Path:
    """Save an hourly temperature series to CSV.

        Args:
            series: Hourly, UTC-indexed temperature series to save.
            out_path: Destination CSV file. Parent directories are created if needed.

        Returns:
            The path of the written CSV file.

        Raises:
            TypeError: If ``series`` is not a pandas Series with a DatetimeIndex.
            ValueError: If the series is empty or its index is not UTC.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if series.empty:
        raise ValueError("series must not be empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError("series index must be UTC")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("temperature_c").to_csv(out_path, index_label="timestamp_utc")
    logger.info("Wrote %d rows to %s", len(series), out_path)
    return out_path


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Fetch population-weighted Romanian temperatures from Open-Meteo"
    )
    parser.add_argument("start")
    parser.add_argument("end")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--source", choices=sorted(_SOURCE_URLS), default="forecast")
    args = parser.parse_args()

    result = load_temperature(args.start, args.end, source=args.source)
    save_temperature(result, args.output_csv)
