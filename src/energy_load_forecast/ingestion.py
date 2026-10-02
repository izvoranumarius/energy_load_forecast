"""Read ENTSO-E load data and convert it to an hourly UTC time series.

This module:
    - reads load data from either a CSV export or the ENTSO-E API
    - normalizes timestamps to UTC
    - converts quarter-hourly observations to hourly values
    - preserves missing values for downstream preprocessing
    - writes the canonical hourly series to CSV

For quarter-hourly input, an hourly value is calculated only when all four
15-minute observations are available. Partial hours remain NaN so missing
data cannot be hidden by averaging an incomplete hour.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


_MTU_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"
_MTU_SEPARATOR = " - "

_ACTUAL_LOAD_COL = "Actual Total Load (MW)"
_MTU_COL = "MTU (UTC)"


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read an ENTSO-E export CSV and convert the actual load column to numeric.
    
        Args:
            path: Path to a CSV exported from the ENTSO-E Transparency Platform
                containing Actual Total Load data.
    
        Returns:
            The raw DataFrame with the actual load column converted to numeric.
            Missing values such as "-" are converted to NaN. No rows are removed
            or resampled.
    
        Raises:
            FileNotFoundError: If the CSV file does not exist.
            KeyError: If one or more expected columns are missing from the CSV.
                This helps catch changes to the ENTSO-E export format.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path, na_values=["-"])
    missing_cols = {_MTU_COL, _ACTUAL_LOAD_COL} - set(df.columns)
    if missing_cols:
        raise KeyError(
            f"Expected columns not found in {path}: {missing_cols}. "
            f"Actual columns: {list(df.columns)}"
        )

    df[_ACTUAL_LOAD_COL] = pd.to_numeric(df[_ACTUAL_LOAD_COL], errors="coerce")
    n_missing = int(df[_ACTUAL_LOAD_COL].isna().sum())
    if n_missing:
        logger.info(
            "%s: %d missing values in %r",
            path.name,
            n_missing,
            _ACTUAL_LOAD_COL,
        )

    return df


def read_raw_api(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
) -> pd.Series | pd.DataFrame:
    """Fetch raw load data from the ENTSO-E API.
    
        Args:
            country_code: ENTSO-E area code, such as "RO" or "DE_LU".
            start: Start of the requested range. Naive timestamps are treated as UTC.
            end: End of the requested range. Naive timestamps are treated as UTC.
            api_key: ENTSO-E API key.
    
        Returns:
            Raw load data returned by entsoe-py.
    
        Raises:
            ValueError: If the requested range is invalid or ENTSO-E has no data.
            RuntimeError: If the API dependencies are unavailable or the API request fails.
    """
    if not country_code:
        raise ValueError("country_code must not be empty")
    if not api_key:
        raise ValueError("api_key must not be empty")

    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    if start >= end:
        raise ValueError(f"Invalid range: start ({start}) must be before end ({end})")

    try:
        import truststore
        from entsoe.entsoe import EntsoePandasClient
        from entsoe.exceptions import NoMatchingDataError
    except ImportError as exc:
        raise RuntimeError("ENTSO-E API dependencies are not installed") from exc

    truststore.inject_into_ssl()
    client = EntsoePandasClient(api_key=api_key)

    try:
        return client.query_load(country_code, start=start, end=end)
    except NoMatchingDataError as exc:
        raise ValueError(
            f"ENTSO-E has no load data for {country_code} between {start} and {end}"
        ) from exc
    except Exception as exc:
        logger.warning("ENTSO-E API query failed: %s", exc)
        raise RuntimeError("ENTSO-E API query failed") from exc


def _resample_to_hourly(series: pd.Series) -> pd.Series:
    """Convert hourly or quarter-hourly observations to hourly UTC values.
    
        For quarter-hourly input, an hour is considered complete only when four
        observations are present. Incomplete hours remain NaN rather than being
        averaged from fewer observations.
    
        Args:
            series: UTC-indexed load series with observations on an hourly or
                quarter-hourly grid.
    
        Returns:
            A UTC-indexed hourly ``load_mw`` series. Complete hours contain the
            mean of their four quarter-hour observations; incomplete hours are NaN.
    
        Raises:
            TypeError: If ``series`` is not a pandas Series or does not use a
                DatetimeIndex.
            ValueError: If the series is empty, timestamps are duplicated, or
                load values are negative.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Load data must be a pandas Series")
    if series.empty:
        raise ValueError("Load series is empty")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Load series must have a DatetimeIndex")
    if series.index.tz is None:
        raise ValueError("Load series index must be timezone-aware")

    series = pd.to_numeric(series.copy(), errors="coerce")
    series.index = series.index.tz_convert("UTC")
    series.index.name = "timestamp_utc"
    series = series.sort_index().rename("load_mw")

    if series.index.has_duplicates:
        raise ValueError("Load series contains duplicate timestamps")

    has_subhourly_timestamp = bool(
        (
            (series.index.minute != 0)
            | (series.index.second != 0)
            | (series.index.microsecond != 0)
        ).any()
    )

    if has_subhourly_timestamp:
        if not all(
            timestamp.minute in {0, 15, 30, 45}
            and timestamp.second == 0
            and timestamp.microsecond == 0
            for timestamp in series.index
        ):
            raise ValueError("Sub-hourly timestamps must be aligned to 15-minute intervals")

        stats = series.resample("1h").agg(["mean", "count"])
        hourly = stats["mean"].where(stats["count"] == 4)
    else:
        hourly = series.resample("1h").mean()

    hourly = hourly.rename("load_mw")
    hourly.index.name = "timestamp_utc"

    if (hourly.dropna() < 0).any():
        raise ValueError("Negative load values present after resampling")

    n_missing = int(hourly.isna().sum())
    if n_missing:
        logger.info(
            "%d/%d hourly values are NaN after resampling; left for preprocessing",
            n_missing,
            len(hourly),
        )

    return hourly


def to_hourly(df: pd.DataFrame) -> pd.Series:
    """Parse MTU timestamps and resample actual load to hourly values.
    
        Quarter-hourly ENTSO-E rows are grouped into hourly UTC periods. An hour
        receives a numeric value only when all four expected 15-minute readings
        are available; otherwise its value is NaN for downstream preprocessing.
    
        Args:
            df: Raw DataFrame returned by ``read_raw_csv``. It must still contain
                the original ``_MTU_COL`` and ``_ACTUAL_LOAD_COL`` columns.
    
        Returns:
            A UTC-indexed hourly ``load_mw`` series. Complete quarter-hourly hours
            are averaged; incomplete hours are preserved as NaN.
    
        Raises:
            KeyError: If the expected ENTSO-E columns are missing.
            ValueError: If timestamps cannot be parsed or any hourly load value
                is negative after resampling.
    """
    missing_cols = {_MTU_COL, _ACTUAL_LOAD_COL} - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")

    interval_start = df[_MTU_COL].astype(str).str.split(_MTU_SEPARATOR).str[0]
    timestamps = pd.to_datetime(
        interval_start,
        format=_MTU_TIMESTAMP_FORMAT,
        errors="raise",
    ).dt.tz_localize("UTC")

    if timestamps.duplicated().any():
        raise ValueError("CSV contains duplicate timestamps")

    working = df[[_ACTUAL_LOAD_COL]].copy()
    working["timestamp_utc"] = timestamps
    series = working.set_index("timestamp_utc")[_ACTUAL_LOAD_COL]
    return _resample_to_hourly(series)


def to_hourly_api(raw: pd.Series | pd.DataFrame) -> pd.Series:
    """Convert ENTSO-E API load data to an hourly UTC time series.
    
        Quarter-hourly API observations are converted to hourly values. Complete
        hours use the mean of four observations, while incomplete hours remain
        NaN for downstream preprocessing.
    
        Args:
            raw: Raw Series or single-column DataFrame returned by the ENTSO-E API.
    
        Returns:
            A UTC-indexed hourly ``load_mw`` series. Missing or incomplete hours
            are preserved as NaN for downstream preprocessing.
    
        Raises:
            ValueError: If the API response is empty, has an unexpected shape,
                contains duplicate timestamps, or has negative load values.
    """
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            raise ValueError("ENTSO-E API returned no data")
        if raw.shape[1] != 1:
            raise ValueError(f"Unexpected API response columns: {list(raw.columns)}")
        raw = raw.iloc[:, 0]

    if not isinstance(raw, pd.Series):
        raise TypeError("ENTSO-E API load data must be a Series or single-column DataFrame")
    if raw.empty:
        raise ValueError("ENTSO-E API returned no data")

    index = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="raise"))
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    series = pd.to_numeric(raw, errors="coerce")
    series.index = index
    series.index.name = "timestamp_utc"
    return _resample_to_hourly(series.rename("load_mw"))


def load_and_resample(path: Path) -> pd.Series:
    """Read an ENTSO-E CSV export and return an hourly UTC series.
    
        Args:
            path: Path to an ENTSO-E export CSV.
    
        Returns:
            Hourly, UTC-indexed ``load_mw`` series. May contain NaN; incomplete
            or missing hours are left for preprocessing.
    """
    return to_hourly(read_raw_csv(path))


def load_and_resample_api(
    country_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: str,
) -> pd.Series:
    """Fetch ENTSO-E API data and return an hourly UTC series.
    
        Args:
            country_code: ENTSO-E area code, such as "RO" or "DE_LU".
            start: Start of the requested range.
            end: End of the requested range.
            api_key: ENTSO-E API key.
    
        Returns:
            Hourly, UTC-indexed ``load_mw`` series. May contain NaN; incomplete
            or missing hours are left for preprocessing.
    """
    return to_hourly_api(read_raw_api(country_code, start, end, api_key))


def save_raw(series: pd.Series, out_path: Path) -> Path:
    """Save an hourly load series to CSV.
    
        Args:
            series: Hourly, UTC-indexed load series to save.
            out_path: Destination CSV file. Parent directories are created if needed.
    
        Returns:
            The path of the written CSV file.
    
        Raises:
            ValueError: If the series is not in the expected hourly UTC format.
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
        raise ValueError("series must have exactly hourly spacing")
    if (pd.to_numeric(series, errors="coerce").dropna() < 0).any():
        raise ValueError("series contains negative load values")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("load_mw").to_csv(out_path, index_label="timestamp_utc")
    logger.info("Wrote %d rows to %s", len(series), out_path)
    return out_path


if __name__ == "__main__":
    import argparse
    import os

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Load ENTSO-E load data and convert it to hourly CSV"
    )
    subparsers = parser.add_subparsers(dest="source", required=True)

    csv_parser = subparsers.add_parser("csv", help="Parse an ENTSO-E CSV export")
    csv_parser.add_argument("input_csv", type=Path)
    csv_parser.add_argument("output_csv", type=Path)

    api_parser = subparsers.add_parser("api", help="Fetch ENTSO-E load data from the API")
    api_parser.add_argument("country_code")
    api_parser.add_argument("start")
    api_parser.add_argument("end")
    api_parser.add_argument("output_csv", type=Path)

    args = parser.parse_args()

    if args.source == "csv":
        result = load_and_resample(args.input_csv)
    else:
        api_key = os.environ.get("ENTSOE_API_KEY")
        if not api_key:
            raise RuntimeError("ENTSOE_API_KEY is not set")
        result = load_and_resample_api(
            args.country_code,
            pd.Timestamp(args.start),
            pd.Timestamp(args.end),
            api_key,
        )

    save_raw(result, args.output_csv)
