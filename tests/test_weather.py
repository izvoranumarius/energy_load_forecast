"""Tests for weather.py.

The Open-Meteo API is replaced by a fake response so the tests run offline
and check parsing, weighting, and validation rather than the remote service.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("requests")
pytest.importorskip("truststore")

import requests
import truststore

from energy_load_forecast.weather import (
    read_raw_temperature,
    save_temperature,
    weighted_temperature,
)

_LOCATIONS = {
    "big_city": (44.0, 26.0, 3.0),
    "small_city": (46.0, 23.0, 1.0),
}


def _api_result(start: str, values: list[float | None]) -> dict:
    """Helper: build one location in the Open-Meteo response format."""
    times = pd.date_range(start, periods=len(values), freq="h")
    return {
        "hourly": {
            "time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
            "temperature_2m": values,
        }
    }


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the HTTP call and record the request parameters."""
    calls: list[dict] = []

    def install(payload):
        def fake_get(url, params, timeout):
            calls.append({"url": url, **params})
            return _FakeResponse(payload)

        monkeypatch.setattr(requests, "get", fake_get)
        return calls

    monkeypatch.setattr(truststore, "inject_into_ssl", lambda: None)
    return install


class TestReadRawTemperature:
    def test_returns_one_column_per_location_in_utc(self, fake_api):
        fake_api(
            [
                _api_result("2026-01-01", [1.0] * 24),
                _api_result("2026-01-01", [2.0] * 24),
            ]
        )

        raw = read_raw_temperature("2026-01-01", "2026-01-02", locations=_LOCATIONS)

        assert list(raw.columns) == ["big_city", "small_city"]
        assert len(raw) == 24
        assert str(raw.index.tz) == "UTC"

    def test_slices_to_the_requested_range(self, fake_api):
        fake_api([_api_result("2026-01-01", list(range(24)))])

        raw = read_raw_temperature(
            "2026-01-01 06:00",
            "2026-01-01 09:00",
            locations={"only": (44.0, 26.0, 1.0)},
        )

        assert list(raw["only"]) == [6.0, 7.0, 8.0]

    def test_accepts_single_location_object_response(self, fake_api):
        fake_api(_api_result("2026-01-01", [5.0] * 24))

        raw = read_raw_temperature(
            "2026-01-01", "2026-01-02", locations={"only": (44.0, 26.0, 1.0)}
        )

        assert (raw["only"] == 5.0).all()

    def test_end_date_is_last_requested_hour(self, fake_api):
        calls = fake_api([_api_result("2026-01-01", [0.0] * 24)] * 2)

        read_raw_temperature("2026-01-01", "2026-01-02", locations=_LOCATIONS)

        assert calls[0]["start_date"] == "2026-01-01"
        assert calls[0]["end_date"] == "2026-01-01"

    def test_observed_source_uses_archive_api(self, fake_api):
        calls = fake_api([_api_result("2026-01-01", [0.0] * 24)] * 2)

        read_raw_temperature(
            "2026-01-01", "2026-01-02", locations=_LOCATIONS, source="observed"
        )

        assert "archive-api" in calls[0]["url"]

    def test_raises_on_location_count_mismatch(self, fake_api):
        fake_api([_api_result("2026-01-01", [0.0] * 24)])

        with pytest.raises(ValueError, match="Expected 2 locations"):
            read_raw_temperature("2026-01-01", "2026-01-02", locations=_LOCATIONS)

    def test_raises_on_unknown_source(self):
        with pytest.raises(ValueError, match="source"):
            read_raw_temperature("2026-01-01", "2026-01-02", source="satellite")

    def test_raises_on_invalid_range(self):
        with pytest.raises(ValueError, match="Invalid range"):
            read_raw_temperature("2026-01-02", "2026-01-01")


class TestWeightedTemperature:
    def _raw(self, big: list[float | None], small: list[float | None]) -> pd.DataFrame:
        index = pd.date_range("2026-01-01", periods=len(big), freq="h", tz="UTC")
        return pd.DataFrame({"big_city": big, "small_city": small}, index=index)

    def test_weights_by_population(self):
        raw = self._raw([10.0], [2.0])

        result = weighted_temperature(raw, locations=_LOCATIONS)

        assert result.iloc[0] == pytest.approx((3 * 10.0 + 1 * 2.0) / 4)
        assert result.name == "temperature_c"

    def test_rescales_weights_when_a_location_is_missing(self):
        raw = self._raw([None], [2.0])

        result = weighted_temperature(raw, locations=_LOCATIONS)

        assert result.iloc[0] == pytest.approx(2.0)

    def test_all_locations_missing_gives_nan(self):
        raw = self._raw([None], [None])

        result = weighted_temperature(raw, locations=_LOCATIONS)

        assert result.isna().all()

    def test_raises_on_missing_location_column(self):
        raw = self._raw([1.0], [2.0]).drop(columns=["small_city"])

        with pytest.raises(KeyError, match="small_city"):
            weighted_temperature(raw, locations=_LOCATIONS)


class TestSaveTemperature:
    def test_round_trip_keeps_negative_values(self, tmp_path):
        index = pd.date_range("2026-01-01", periods=3, freq="h", tz="UTC")
        series = pd.Series([-5.5, 0.0, 3.2], index=index, name="temperature_c")

        path = save_temperature(series, tmp_path / "weather" / "temperature.csv")
        loaded = pd.read_csv(path, index_col="timestamp_utc")

        assert list(loaded["temperature_c"]) == [-5.5, 0.0, 3.2]

    def test_raises_on_naive_index(self, tmp_path):
        index = pd.date_range("2026-01-01", periods=3, freq="h")
        series = pd.Series([1.0, 2.0, 3.0], index=index)

        with pytest.raises(ValueError, match="UTC"):
            save_temperature(series, tmp_path / "temperature.csv")
