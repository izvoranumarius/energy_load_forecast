"""Clean an hourly load series and prepare it for modelling.

This module:
    - removes trailing NaN values caused by publication delays
    - verifies that the input is a continuous hourly UTC series
    - adds calendar features based on local time
    - splits the data chronologically into train, validation, and test sets
    - saves the resulting splits as CSV files

The input is expected to be an hourly, UTC-indexed ``load_mw`` series produced
by the ingestion module. Missing values inside the retained range are treated
as data-quality errors rather than silently imputed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def clean_tail(series: pd.Series) -> pd.Series:
    """Remove trailing NaN values from an hourly load series.
    
        Trailing NaN values can occur because ENTSO-E publishes data with a short
        delay. These values are removed, while NaN values inside the remaining
        time range are treated as data-quality errors. The input must be a
        continuous hourly UTC series.
    
        Args:
            series: Hourly load series indexed by UTC timestamps. The series may
                contain trailing NaN values.
    
        Returns:
            The series truncated at its last valid observation.
    
        Raises:
            TypeError: If ``series`` is not a pandas Series or does not have a
                DatetimeIndex.
            ValueError: If the series is empty, not UTC, unsorted, duplicated,
                non-hourly, all NaN, or contains an interior NaN.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"Expected a pandas Series, got {type(series).__name__}")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex, got {type(series.index).__name__}"
        )
    if series.empty:
        raise ValueError("Series is empty")
    if series.index.tz is None or str(series.index.tz) != "UTC":
        raise ValueError("Series index must be UTC")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Series index must be sorted")
    if ((series.index.minute != 0) | (series.index.second != 0) | (series.index.microsecond != 0)).any():
        raise ValueError("Series index must be aligned to exact hours")
    if series.index.has_duplicates:
        raise ValueError("Series index contains duplicate timestamps")

    deltas = series.index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("Series must have exactly hourly UTC spacing")

    last_valid = series.last_valid_index()
    if last_valid is None:
        raise ValueError("Series contains only NaN values")

    trimmed = series.loc[:last_valid]
    if trimmed.isna().any():
        raise ValueError("NaN values are not confined to the tail: interior gaps found")

    n_missing = int(series.isna().sum())
    if n_missing:
        logger.info(
            "Trimmed %d trailing NaN values, %d rows remain",
            n_missing,
            len(trimmed),
        )

    return trimmed


def add_calendar_features(
    df: pd.DataFrame, tz: str = "Europe/Bucharest"
) -> pd.DataFrame:
    """Add calendar features derived from local time.
    
        UTC timestamps are converted to the specified local timezone before the
        calendar features are calculated. The original UTC index is preserved.
    
        Args:
            df: DataFrame with a timezone-aware UTC DatetimeIndex aligned to exact hours.
            tz: IANA timezone used to derive local calendar features. Defaults to
                ``"Europe/Bucharest"``.
    
        Returns:
            A copy of ``df`` with the following additional columns:
                ``hour_local``: Hour of day from 0 to 23.
                ``day_of_week_local``: Day of week from Monday=0 to Sunday=6.
                ``is_weekend_local``: Whether the local day is Saturday or Sunday.
                ``month_local``: Month of year from 1 to 12.
    
        Raises:
            TypeError: If the DataFrame index is not a timezone-aware
                DatetimeIndex.
            ValueError: If the index is not UTC or is not aligned to exact hours.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex, got {type(df.index).__name__}"
        )
    if df.index.tz is None:
        raise TypeError("DatetimeIndex must be timezone-aware")
    if str(df.index.tz) != "UTC":
        raise ValueError("DatetimeIndex must be UTC")
    if ((df.index.minute != 0) | (df.index.second != 0) | (df.index.microsecond != 0)).any():
        raise ValueError("DatetimeIndex must be aligned to exact hours")

    local_index = df.index.tz_convert(tz)
    out = df.copy()
    out["hour_local"] = local_index.hour
    out["day_of_week_local"] = local_index.day_of_week
    out["is_weekend_local"] = local_index.day_of_week >= 5
    out["month_local"] = local_index.month
    return out


def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp-like value to a UTC-aware Timestamp.
    
        Args:
            value: Timestamp-like value. Naive values are interpreted as UTC;
                timezone-aware values are converted to UTC.
    
        Returns:
            A timezone-aware UTC ``pd.Timestamp`` aligned to an exact hour.
    
        Raises:
            ValueError: If the timestamp is not aligned to an exact hour.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts.minute != 0 or ts.second != 0 or ts.microsecond != 0:
        raise ValueError(f"Split timestamp must be aligned to an exact hour, got {ts}")
    return ts


def chronological_split(
    df: pd.DataFrame,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame into train, validation, and test sets.
    
        The split boundaries are chronological. Training data contains timestamps
        before ``train_end``, validation data contains timestamps from ``train_end``
        up to but not including ``val_end``, and test data contains timestamps from
        ``val_end`` onward. Split boundaries are interpreted in UTC and must be
        aligned to exact hours.
    
        Args:
            df: DataFrame with a sorted, unique, continuous hourly UTC DatetimeIndex.
            train_end: Timestamp marking the start of the validation period.
            val_end: Timestamp marking the start of the test period.
    
        Returns:
            A tuple containing the training, validation, and test DataFrames.
    
        Raises:
            TypeError: If ``df`` does not have a DatetimeIndex.
            ValueError: If the index is invalid, split timestamps are not on the
                hourly grid, or the boundaries do not satisfy
                ``data_start < train_end < val_end < data_end``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            f"Expected a DatetimeIndex, got {type(df.index).__name__}"
        )
    if df.empty:
        raise ValueError("DataFrame is empty")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise ValueError("DataFrame index must be UTC")
    if not df.index.is_monotonic_increasing:
        raise ValueError("DataFrame index must be sorted")
    if ((df.index.minute != 0) | (df.index.second != 0) | (df.index.microsecond != 0)).any():
        raise ValueError("DataFrame index must be aligned to exact hours")
    if df.index.has_duplicates:
        raise ValueError("DataFrame index contains duplicate timestamps")

    deltas = df.index.to_series().diff().dropna()
    if not deltas.empty and not deltas.eq(pd.Timedelta(hours=1)).all():
        raise ValueError("DataFrame index must have exactly hourly spacing")

    train_end_ts = _as_utc_timestamp(train_end)
    val_end_ts = _as_utc_timestamp(val_end)

    if not (df.index.min() < train_end_ts < val_end_ts < df.index.max()):
        raise ValueError(
            "Split boundaries must satisfy: data_start < train_end < val_end < data_end. "
            f"Got train_end={train_end_ts}, val_end={val_end_ts}, "
            f"data range=[{df.index.min()}, {df.index.max()}]"
        )

    train = df.loc[df.index < train_end_ts]
    val = df.loc[(df.index >= train_end_ts) & (df.index < val_end_ts)]
    test = df.loc[df.index >= val_end_ts]

    if train.empty or val.empty or test.empty:
        raise ValueError("Train, validation, and test sets must all be non-empty")

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train),
        len(val),
        len(test),
    )
    return train, val, test


def preprocess(
    series: pd.Series,
    train_end: str | pd.Timestamp,
    val_end: str | pd.Timestamp,
    tz: str = "Europe/Bucharest",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare an hourly load series for modelling.
    
        The series is first trimmed at its last valid observation. Calendar
        features are then added using the specified local timezone, and the
        resulting DataFrame is split chronologically into train, validation,
        and test sets.
    
        Args:
            series: Hourly, UTC-indexed ``load_mw`` series produced by ingestion.
            train_end: Timestamp marking the start of the validation period.
            val_end: Timestamp marking the start of the test period.
            tz: IANA timezone used to derive calendar features. Defaults to
                ``"Europe/Bucharest"``.
    
        Returns:
            A tuple containing the training, validation, and test DataFrames with
            calendar features added.
    
        Raises:
            TypeError: If ``series`` is not a valid pandas Series or the expected
                index type is missing.
            ValueError: If the input or split boundaries are invalid.
    """
    trimmed = clean_tail(series)
    frame = trimmed.to_frame(name=trimmed.name or "load_mw")
    frame = add_calendar_features(frame, tz=tz)
    return chronological_split(frame, train_end, val_end)


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> dict[str, Path]:
    """Save train, validation, and test sets to CSV files.
    
        The output directory is created if it does not already exist.
    
        Args:
            train: Training DataFrame.
            val: Validation DataFrame.
            test: Test DataFrame.
            out_dir: Directory where the split CSV files will be written.
    
        Returns:
            A dictionary mapping each split name to its output file path.
            The keys are ``train``, ``val``, and ``test``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index_label="timestamp_utc")
        logger.info("Wrote %d rows to %s", len(frame), path)
        paths[name] = path
    return paths


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Clean an hourly load CSV, split it chronologically, and save CSV files"
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--tz", default="Europe/Bucharest")
    args = parser.parse_args()

    cached = pd.read_csv(args.input_csv, index_col="timestamp_utc")
    cached.index = pd.to_datetime(cached.index, utc=True)
    train, val, test = preprocess(
        cached["load_mw"],
        train_end=args.train_end,
        val_end=args.val_end,
        tz=args.tz,
    )
    save_splits(train, val, test, args.output_dir)
