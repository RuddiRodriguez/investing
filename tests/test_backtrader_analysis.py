from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.backtrader_pipeline import build_backtrader_ohlcv_frame, run_advanced_backtrader_analysis


def _make_ohlcv_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2022-01-03", periods=260)
    log_returns = rng.normal(loc=0.0005, scale=0.012, size=len(index))
    close = 100.0 * np.exp(np.cumsum(log_returns))
    open_price = close * (1.0 + rng.normal(0.0, 0.002, size=len(index)))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.01, size=len(index)))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.01, size=len(index)))
    volume = rng.integers(1_000_000, 5_000_000, size=len(index))
    return pd.DataFrame(
        {
            "AAA": close,
            "OPEN_AAA": open_price,
            "HIGH_AAA": high,
            "LOW_AAA": low,
            "VOLUME_AAA": volume,
        },
        index=index,
    )


def test_build_backtrader_ohlcv_frame_returns_expected_columns() -> None:
    prices = _make_ohlcv_frame()

    frame = build_backtrader_ohlcv_frame(prices, "AAA")

    assert list(frame.columns) == ["open", "high", "low", "close", "volume", "openinterest"]
    assert len(frame) == len(prices)


def test_run_advanced_backtrader_analysis_returns_expected_sections() -> None:
    prices = _make_ohlcv_frame()

    result = run_advanced_backtrader_analysis(
        prices,
        ticker="AAA",
        lookback_days=220,
        train_ratio=0.7,
        optimization_mode="focused",
        initial_cash=50000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    assert {"optimization_results", "best_params", "train", "test", "full", "metrics_table", "split_date"}.issubset(result)
    assert not result["optimization_results"].empty
    assert not result["metrics_table"].empty
    assert "portfolio_value" in result["full"]["equity_curve"].columns
    assert "signal" in result["full"]["signals"].columns