from __future__ import annotations

import argparse

import pandas as pd

from market_forecasting_engine.cli import (
    _append_forecast_log,
    _filter_prices_as_of,
    _interval_to_minutes,
    _refresh_cache_for_request,
)


def test_interval_to_minutes_supports_hour_and_intraday_bars() -> None:
    assert _interval_to_minutes("5m") == 5.0
    assert _interval_to_minutes("1h") == 60.0
    assert _interval_to_minutes("1d") == 1440.0


def test_filter_prices_as_of_truncates_to_timestamp() -> None:
    index = pd.date_range("2026-05-29 09:30", periods=4, freq="1h")
    prices = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0]}, index=index)

    filtered = _filter_prices_as_of(prices, "2026-05-29T11:00:00", target_column="close")

    assert list(filtered["close"]) == [100.0, 101.0]


def test_live_yahoo_runs_refresh_cache_when_end_is_omitted() -> None:
    args = argparse.Namespace(refresh_data_cache=False, allow_live_cache=False, end=None, csv=None)

    assert _refresh_cache_for_request(args, "yahoo") is True
    assert _refresh_cache_for_request(args, "csv") is False


def test_append_forecast_log_deduplicates_same_as_of_and_horizon(tmp_path) -> None:
    path = tmp_path / "forecast_log.csv"
    report = {
        "generated_at_utc": "2026-05-29T12:00:00+00:00",
        "ticker": "TSLA",
        "as_of_timestamp": "2026-05-29T15:00:00",
        "forecast_interval": "1h",
        "current_price": 100.0,
        "suggested_action": "Hold",
        "risk_level": "Medium",
        "forecasts": [
            {
                "horizon_days": 1,
                "forecast_date": "2026-05-29T16:00:00",
                "predicted_price": 101.0,
                "lower_price": 99.0,
                "upper_price": 103.0,
                "expected_return": 0.01,
                "expected_direction": "Up",
                "directional_confidence": 0.60,
                "selected_model": "test",
            }
        ],
    }

    _append_forecast_log(report, path)
    _append_forecast_log(report, path)

    logged = pd.read_csv(path)
    assert len(logged) == 1
    assert logged.loc[0, "forecast_timestamp"] == "2026-05-29T16:00:00"
