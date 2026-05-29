from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.transformer_pipeline.analysis import (
    generate_transformer_analysis,
    train_transformer_artifacts,
)


def _make_price_frame() -> pd.DataFrame:
    rng = np.random.default_rng(19)
    index = pd.bdate_range("2022-01-03", periods=280)
    benchmark_log_returns = rng.normal(loc=0.0003, scale=0.009, size=len(index))
    stock_log_returns = benchmark_log_returns + rng.normal(loc=0.0002, scale=0.012, size=len(index))
    benchmark_close = 100.0 * np.exp(np.cumsum(benchmark_log_returns))
    close = 90.0 * np.exp(np.cumsum(stock_log_returns))
    open_price = close * (1.0 + rng.normal(0.0, 0.002, size=len(index)))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.01, size=len(index)))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.01, size=len(index)))
    volume = rng.integers(500_000, 2_000_000, size=len(index))
    return pd.DataFrame(
        {
            "AAA": close,
            "OPEN_AAA": open_price,
            "HIGH_AAA": high,
            "LOW_AAA": low,
            "VOLUME_AAA": volume,
            "SPY": benchmark_close,
        },
        index=index,
    )


def test_train_and_generate_transformer_analysis() -> None:
    prices = _make_price_frame()
    analysis_end_date = str(prices.index[-40].date())
    artifacts = train_transformer_artifacts(
        ticker="AAA",
        benchmark="SPY",
        prices=prices,
        analysis_end_date=analysis_end_date,
        forecast_horizon_days=5,
        sequence_length=24,
        epochs=2,
        batch_size=32,
        d_model=32,
        nhead=4,
        num_layers=1,
    )

    result = generate_transformer_analysis(
        ticker="AAA",
        benchmark="SPY",
        prices=prices,
        analysis_end_date=analysis_end_date,
        artifacts=artifacts,
    )

    assert "summary" in result
    assert "history" in result
    assert "historical_predictions" in result
    assert "validation_metrics" in result
    assert 0.0 <= result["summary"]["probability_of_outperformance"] <= 1.0
    assert not result["training_history"].empty