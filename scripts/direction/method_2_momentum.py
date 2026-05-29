"""Method 2: volatility-adjusted momentum with a no-trade zone."""

from __future__ import annotations

import numpy as np

try:
    from scripts.direction.common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices
except ImportError:
    from common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices


LOOKBACK_DAYS = 60
VOLATILITY_DAYS = 20
SIGNAL_THRESHOLD = 0.5


def add_signal(prices):
    output = prices.copy()
    output["Momentum Return"] = output[PRICE_COLUMN] / output[PRICE_COLUMN].shift(LOOKBACK_DAYS) - 1.0
    daily_return = output[PRICE_COLUMN].pct_change()
    output["Volatility"] = daily_return.rolling(VOLATILITY_DAYS).std() * np.sqrt(252)
    output["Momentum Score"] = output["Momentum Return"] / output["Volatility"].replace(0.0, np.nan)
    output["Prediction"] = output["Momentum Score"].apply(
        lambda score: latest_signal_label(score, SIGNAL_THRESHOLD, -SIGNAL_THRESHOLD)
    )
    return output


def predict_direction(prices):
    signal = add_signal(prices)
    return str(signal["Prediction"].iloc[-1])


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
