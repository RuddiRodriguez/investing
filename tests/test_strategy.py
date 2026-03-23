from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import StrategyConfig, backtest, build_weights


def _make_prices() -> pd.DataFrame:
    index = pd.bdate_range("2023-01-02", periods=320)
    return pd.DataFrame(
        {
            "AAA": np.linspace(100, 180, len(index)),
            "BBB": np.linspace(100, 140, len(index)),
            "CCC": np.linspace(100, 90, len(index)),
        },
        index=index,
    )


def test_build_weights_selects_top_assets_above_trend() -> None:
    prices = _make_prices()
    config = StrategyConfig(top_n=2, rebalance_frequency="Q", lookback_months=6)

    rebalance_weights, daily_weights = build_weights(prices, config)

    assert not rebalance_weights.empty
    latest = rebalance_weights.iloc[-1]
    assert latest["AAA"] == 0.5
    assert latest["BBB"] == 0.5
    assert latest["CCC"] == 0.0
    assert latest["CASH"] == 0.0
    assert daily_weights.index.equals(prices.index)


def test_backtest_uses_cash_when_no_asset_qualifies() -> None:
    index = pd.bdate_range("2023-01-02", periods=320)
    prices = pd.DataFrame(
        {
            "AAA": np.linspace(180, 100, len(index)),
            "BBB": np.linspace(160, 90, len(index)),
        },
        index=index,
    )
    config = StrategyConfig(top_n=2, rebalance_frequency="Q", lookback_months=6)

    results = backtest(prices, config)
    latest = results["latest_allocation"]

    assert latest.index.tolist() == ["CASH"]
    assert latest.iloc[0] == 1.0
    assert (results["equity_curve"] > 0).all()
