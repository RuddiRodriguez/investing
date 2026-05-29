from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecasting_engine.daily_trade import (
    DailyTradeConfig,
    build_daily_trade_plan,
    infer_bar_interval_minutes,
)


def _intraday_prices(rows: int = 78) -> pd.DataFrame:
    index = pd.date_range("2026-05-28 09:30", periods=rows, freq="5min")
    close = 100 + np.linspace(0, 3.0, rows)
    volume = np.linspace(100_000, 180_000, rows)
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.10,
            "low": close - 0.15,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_infer_bar_interval_minutes_uses_intraday_gaps() -> None:
    prices = _intraday_prices()

    assert infer_bar_interval_minutes(prices.index) == 5.0


def test_daily_trade_plan_builds_long_setup_from_intraday_data() -> None:
    prices = _intraday_prices()
    report = build_daily_trade_plan(prices, DailyTradeConfig(ticker="TEST", minimum_score_to_trade=2.0))

    assert report["requires_intraday_data"] is True
    assert report["has_intraday_data"] is True
    assert report["interval_minutes"] == 5.0
    assert report["trade_plan"]["action"] == "long"
    assert report["trade_plan"]["stop"] < report["latest_price"]
    assert report["trade_plan"]["take_profit"] > report["latest_price"]


def test_daily_trade_plan_warns_when_input_is_daily() -> None:
    index = pd.bdate_range("2026-01-02", periods=30)
    close = 100 + np.arange(30)
    prices = pd.DataFrame({"close": close, "volume": 100_000}, index=index)

    report = build_daily_trade_plan(prices, DailyTradeConfig(ticker="TEST"))

    assert report["has_intraday_data"] is False
    assert "daily/end-of-day" in report["data_warning"]
