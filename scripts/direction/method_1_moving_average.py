"""Method 1: long-term trend signal with a no-trade zone around the 200-day average."""

from __future__ import annotations

try:
    from scripts.direction.common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices
except ImportError:
    from common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices


MOVING_AVERAGE_DAYS = 200
NO_TRADE_BAND = 0.02


def add_signal(prices):
    output = prices.copy()
    output["Moving Average"] = output[PRICE_COLUMN].rolling(MOVING_AVERAGE_DAYS).mean()
    output["Trend Gap"] = output[PRICE_COLUMN] / output["Moving Average"] - 1.0
    output["Prediction"] = output["Trend Gap"].apply(
        lambda score: latest_signal_label(score, NO_TRADE_BAND, -NO_TRADE_BAND)
    )
    return output


def predict_direction(prices):
    signal = add_signal(prices)
    return str(signal["Prediction"].iloc[-1])


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
