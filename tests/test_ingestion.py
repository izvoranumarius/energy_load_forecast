"""Tests for the ENTSO-E ingestion helpers.

Each test builds a small synthetic DataFrame or Series directly in code.
This keeps the tests fast, readable, and in control of the exact scenario
being checked instead of depending on external data files or a network call.
The ENTSO-E API client itself is not exercised here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from energy_load_forecast.ingestion import (
    load_and_resample,
    load_and_resample_api,
    read_raw_api,
    read_raw_csv,
    save_raw,
    to_hourly,
    to_hourly_api,
)

MTU_COL = "MTU (UTC)"
LOAD_COL = "Actual Total Load (MW)"


def _mtu(start: str) -> str:
    """Helper: build an ENTSO-E style 15-minute MTU interval string."""
    begin = pd.Timestamp(start)
    end = begin + pd.Timedelta(minutes=15)
    fmt = "%d/%m/%Y %H:%M"
    return f"{begin.strftime(fmt)} - {end.strftime(fmt)}"


def _raw_df(values: list[float]) -> pd.DataFrame:
    """Helper: build a raw ENTSO-E style DataFrame with 15-minute rows."""
    starts = pd.date_range("2026-01-01 00:00", periods=len(values), freq="15min")
    return pd.DataFrame({MTU_COL: [_mtu(str(s)) for s in starts], LOAD_COL: values})


def _api_series(values: list, tz: str | None = "UTC", start: str = "2026-01-01") -> pd.Series:
    """Helper: build a 15-minute series shaped like an entsoe-py response."""
    index = pd.date_range(start, periods=len(values), freq="15min", tz=tz)
    return pd.Series(values, index=index)


# ---------------------------------------------------------------------------
# read_raw_csv
# ---------------------------------------------------------------------------


class TestReadRawCsv:
    def test_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_raw_csv(tmp_path / "missing.csv")

    def test_raises_if_expected_columns_missing(self, tmp_path):
        """Guards against ENTSO-E silently changing its export format."""
        path = tmp_path / "bad.csv"
        path.write_text("a,b\n1,2\n")

        with pytest.raises(KeyError, match="Expected columns not found"):
            read_raw_csv(path)

    def test_dash_becomes_nan(self, tmp_path):
        """ENTSO-E marks unpublished values with '-'."""
        path = tmp_path / "ok.csv"
        path.write_text(
            f"{MTU_COL},{LOAD_COL}\n"
            f"{_mtu('2026-01-01 00:00')},100\n"
            f"{_mtu('2026-01-01 00:15')},-\n"
        )

        result = read_raw_csv(path)

        assert len(result) == 2
        assert result[LOAD_COL].iloc[0] == 100
        assert pd.isna(result[LOAD_COL].iloc[1])

    def test_non_numeric_becomes_nan(self, tmp_path):
        path = tmp_path / "ok.csv"
        path.write_text(
            f"{MTU_COL},{LOAD_COL}\n"
            f"{_mtu('2026-01-01 00:00')},abc\n"
            f"{_mtu('2026-01-01 00:15')},5\n"
        )

        result = read_raw_csv(path)

        assert pd.isna(result[LOAD_COL].iloc[0])
        assert result[LOAD_COL].iloc[1] == 5

    def test_no_rows_are_dropped(self, tmp_path):
        """read_raw_csv must not remove rows -- NaN handling is a
        downstream decision."""
        path = tmp_path / "ok.csv"
        path.write_text(
            f"{MTU_COL},{LOAD_COL}\n"
            f"{_mtu('2026-01-01 00:00')},-\n"
            f"{_mtu('2026-01-01 00:15')},-\n"
        )

        assert len(read_raw_csv(path)) == 2


# ---------------------------------------------------------------------------
# read_raw_api -- validation only, runs before any network call
# ---------------------------------------------------------------------------


class TestReadRawApiValidation:
    START = pd.Timestamp("2026-01-01")
    END = pd.Timestamp("2026-01-02")

    def test_raises_on_empty_country_code(self):
        with pytest.raises(ValueError, match="country_code"):
            read_raw_api("", self.START, self.END, "key")

    def test_raises_on_empty_api_key(self):
        with pytest.raises(ValueError, match="api_key"):
            read_raw_api("RO", self.START, self.END, "")

    def test_raises_if_start_after_end(self):
        with pytest.raises(ValueError, match="Invalid range"):
            read_raw_api("RO", self.END, self.START, "key")

    def test_raises_if_start_equals_end(self):
        with pytest.raises(ValueError, match="Invalid range"):
            read_raw_api("RO", self.START, self.START, "key")

    def test_range_is_checked_after_converting_to_utc(self):
        """02:00 in Bucharest (UTC+2) is 00:00 UTC, so this range is
        empty even though the wall-clock times look ordered."""
        start = pd.Timestamp("2026-01-01 02:00", tz="Europe/Bucharest")
        end = pd.Timestamp("2026-01-01 00:00", tz="UTC")

        with pytest.raises(ValueError, match="Invalid range"):
            read_raw_api("RO", start, end, "key")


# ---------------------------------------------------------------------------
# to_hourly (CSV path)
# ---------------------------------------------------------------------------


class TestToHourly:
    def test_averages_four_quarter_hours_per_hour(self):
        df = _raw_df([100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0])

        result = to_hourly(df)

        assert result.tolist() == [250.0, 25.0]

    def test_output_is_named_utc_hourly_series(self):
        df = _raw_df([1.0, 2.0, 3.0, 4.0])

        result = to_hourly(df)

        assert result.name == "load_mw"
        assert result.index.name == "timestamp_utc"
        assert str(result.index.tz) == "UTC"
        assert result.index[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")

    def test_partial_nan_makes_hour_nan(self):
        """An incomplete 15-minute hour must remain NaN for preprocessing."""
        df = _raw_df([float("nan"), 200.0, 300.0, 400.0])

        result = to_hourly(df)

        assert pd.isna(result.iloc[0])

    def test_fully_missing_hour_stays_nan(self):
        """Missing data is preserved for preprocessing, not filled in."""
        df = _raw_df([float("nan")] * 4 + [10.0, 20.0, 30.0, 40.0])

        result = to_hourly(df)

        assert pd.isna(result.iloc[0])
        assert result.iloc[1] == 25.0

    def test_gap_in_time_creates_nan_rows(self):
        """Skipped hours must appear as NaN rows, not be silently
        collapsed -- otherwise the series is no longer regular hourly."""
        df = pd.DataFrame(
            {
                MTU_COL: [_mtu("2026-01-01 00:00"), _mtu("2026-01-01 03:00")],
                LOAD_COL: [10.0, 20.0],
            }
        )

        result = to_hourly(df)

        assert len(result) == 4
        assert result.isna().tolist() == [False, True, True, False]

    def test_raises_on_negative_load(self):
        df = _raw_df([-1.0, -1.0, -1.0, -1.0])

        with pytest.raises(ValueError, match="Negative load"):
            to_hourly(df)

    def test_raises_on_unparseable_timestamp(self):
        df = pd.DataFrame({MTU_COL: ["garbage - garbage"], LOAD_COL: [1.0]})

        with pytest.raises(ValueError):
            to_hourly(df)


# ---------------------------------------------------------------------------
# to_hourly_api
# ---------------------------------------------------------------------------


class TestToHourlyApi:
    def test_averages_series_to_hourly(self):
        result = to_hourly_api(_api_series([100.0, 200.0, 300.0, 400.0]))

        assert result.name == "load_mw"
        assert result.index.name == "timestamp_utc"
        assert result.tolist() == [250.0]

    def test_accepts_single_column_dataframe(self):
        raw = _api_series([100.0, 200.0, 300.0, 400.0]).to_frame("load")

        assert to_hourly_api(raw).tolist() == [250.0]

    def test_naive_index_is_treated_as_utc(self):
        result = to_hourly_api(_api_series([1.0, 2.0, 3.0, 4.0], tz=None))

        assert str(result.index.tz) == "UTC"
        assert result.index[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")

    def test_other_timezone_is_converted_to_utc(self):
        """02:00 Bucharest (UTC+2) must land on 00:00 UTC. A bug here
        would silently shift the whole dataset by hours."""
        raw = _api_series(
            [1.0, 2.0, 3.0, 4.0], tz="Europe/Bucharest", start="2026-01-01 02:00"
        )

        result = to_hourly_api(raw)

        assert str(result.index.tz) == "UTC"
        assert result.index[0] == pd.Timestamp("2026-01-01 00:00", tz="UTC")

    def test_unsorted_input_is_sorted(self):
        raw = _api_series([100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0])

        result = to_hourly_api(raw.iloc[::-1])

        assert result.index.is_monotonic_increasing
        assert result.tolist() == [250.0, 25.0]

    def test_non_numeric_value_makes_quarter_hour_hour_incomplete(self):
        result = to_hourly_api(_api_series(["1", "x", "3", "4"]))

        assert pd.isna(result.iloc[0])

    def test_gap_in_time_creates_nan_hour(self):
        index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 02:00"], tz="UTC")
        raw = pd.Series([1.0, 3.0], index=index)

        result = to_hourly_api(raw)

        assert len(result) == 3
        assert pd.isna(result.iloc[1])

    def test_raises_on_empty_series(self):
        with pytest.raises(ValueError, match="no data"):
            to_hourly_api(pd.Series(dtype=float))

    def test_raises_on_empty_dataframe(self):
        with pytest.raises(ValueError, match="no data"):
            to_hourly_api(pd.DataFrame())

    def test_raises_on_multi_column_dataframe(self):
        index = pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")
        raw = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=index)

        with pytest.raises(ValueError, match="Unexpected API response columns"):
            to_hourly_api(raw)

    def test_raises_on_duplicate_timestamps(self):
        index = pd.DatetimeIndex(["2026-01-01 00:00", "2026-01-01 00:00"], tz="UTC")
        raw = pd.Series([1.0, 2.0], index=index)

        with pytest.raises(ValueError, match="duplicate timestamps"):
            to_hourly_api(raw)

    def test_raises_on_negative_load(self):
        with pytest.raises(ValueError, match="Negative load"):
            to_hourly_api(_api_series([-1.0, -1.0, -1.0, -1.0]))


# ---------------------------------------------------------------------------
# load_and_resample / load_and_resample_api
# ---------------------------------------------------------------------------


class TestLoadAndResample:
    def test_csv_end_to_end(self, tmp_path):
        path = tmp_path / "in.csv"
        _raw_df([100.0, 200.0, 300.0, 400.0, 10.0, 20.0, 30.0, 40.0]).to_csv(
            path, index=False
        )

        result = load_and_resample(path)

        assert result.tolist() == [250.0, 25.0]

    def test_csv_end_to_end_with_dash(self, tmp_path):
        """A missing quarter-hour makes the hourly value incomplete."""
        path = tmp_path / "in.csv"
        path.write_text(
            f"{MTU_COL},{LOAD_COL}\n"
            f"{_mtu('2026-01-01 00:00')},100\n"
            f"{_mtu('2026-01-01 00:15')},200\n"
            f"{_mtu('2026-01-01 00:30')},-\n"
            f"{_mtu('2026-01-01 00:45')},300\n"
        )

        result = load_and_resample(path)

        assert len(result) == 1
        assert pd.isna(result.iloc[0])

    def test_api_end_to_end(self, monkeypatch):
        raw = _api_series([10.0, 20.0, 30.0, 40.0])
        monkeypatch.setattr(
            "energy_load_forecast.ingestion.read_raw_api", lambda *a, **k: raw
        )

        result = load_and_resample_api(
            "RO", pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"), "key"
        )

        assert result.tolist() == [25.0]


# ---------------------------------------------------------------------------
# save_raw
# ---------------------------------------------------------------------------


class TestSaveRaw:
    def test_creates_parent_directories_and_returns_path(self, tmp_path):
        index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
        series = pd.Series([1.0, 2.0], index=index, name="load_mw")
        out_path = tmp_path / "a" / "b" / "out.csv"

        returned = save_raw(series, out_path)

        assert returned == out_path
        assert out_path.exists()

    def test_written_csv_has_expected_columns_and_nan(self, tmp_path):
        index = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
        series = pd.Series([1.0, float("nan")], index=index, name="load_mw")
        out_path = tmp_path / "out.csv"

        save_raw(series, out_path)
        saved = pd.read_csv(out_path)

        assert list(saved.columns) == ["timestamp_utc", "load_mw"]
        assert saved["load_mw"].iloc[0] == 1.0
        assert pd.isna(saved["load_mw"].iloc[1])