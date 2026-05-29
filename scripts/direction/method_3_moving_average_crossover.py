"""Method 3: moving-average crossover with spread threshold and no-trade zone."""

from __future__ import annotations

try:
    from scripts.direction.common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices
except ImportError:
    from common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices


SHORT_AVERAGE_DAYS = 50
LONG_AVERAGE_DAYS = 200
CROSSOVER_THRESHOLD = 0.01


def add_signal(prices):
    output = prices.copy()
    output["Short Average"] = output[PRICE_COLUMN].rolling(SHORT_AVERAGE_DAYS).mean()
    output["Long Average"] = output[PRICE_COLUMN].rolling(LONG_AVERAGE_DAYS).mean()
    output["Crossover Spread"] = output["Short Average"] / output["Long Average"] - 1.0
    output["Prediction"] = output["Crossover Spread"].apply(
        lambda score: latest_signal_label(score, CROSSOVER_THRESHOLD, -CROSSOVER_THRESHOLD)
    )
    return output


def predict_direction(prices):
    signal = add_signal(prices)
    return str(signal["Prediction"].iloc[-1])


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
