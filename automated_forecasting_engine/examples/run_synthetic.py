from __future__ import annotations

import numpy as np
import pandas as pd

from market_forecasting_engine import ForecastConfig, ForecastingEngine


def synthetic_prices(rows: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-02", periods=rows)
    returns = rng.normal(0.0004, 0.015, size=rows)
    close = 100 * np.exp(np.cumsum(returns))
    volume = rng.integers(1_000_000, 3_000_000, size=rows)
    return pd.DataFrame({"close": close, "volume": volume}, index=dates)


if __name__ == "__main__":
    config = ForecastConfig(ticker="SYNTH", horizons=(1, 5, 30), include_lightgbm=False)
    report = ForecastingEngine(config).run(synthetic_prices())
    for forecast in report["forecasts"]:
        print(forecast)
