"""Method 4: weighted indicator voting with no-trade band."""

from __future__ import annotations

try:
    from scripts.direction.common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices
except ImportError:
    from common import CSV_FILE, PRICE_COLUMN, latest_signal_label, read_prices


SHORT_AVERAGE_DAYS = 50
LONG_AVERAGE_DAYS = 200
RSI_DAYS = 14
VOTE_THRESHOLD = 0.5


def add_rsi(prices):
    output = prices.copy()
    price_change = output[PRICE_COLUMN].diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(RSI_DAYS).mean()
    average_loss = losses.rolling(RSI_DAYS).mean()

    output["RSI"] = 100 - (100 / (1 + average_gain / average_loss))
    return output


def add_signal(prices):
    output = prices.copy()
    output["Short Average"] = output[PRICE_COLUMN].rolling(SHORT_AVERAGE_DAYS).mean()
    output["Long Average"] = output[PRICE_COLUMN].rolling(LONG_AVERAGE_DAYS).mean()
    output = add_rsi(output)

    output["Trend Vote"] = (output[PRICE_COLUMN] > output["Long Average"]).astype(float) * 2 - 1
    output["Crossover Vote"] = (output["Short Average"] > output["Long Average"]).astype(float) * 2 - 1
    output["RSI Vote"] = ((output["RSI"] > 55).astype(float) - (output["RSI"] < 45).astype(float))
    output["Vote Score"] = 0.4 * output["Trend Vote"] + 0.4 * output["Crossover Vote"] + 0.2 * output["RSI Vote"]
    output["Prediction"] = output["Vote Score"].apply(
        lambda score: latest_signal_label(score, VOTE_THRESHOLD, -VOTE_THRESHOLD)
    )
    return output


def predict_direction(prices):
    signal = add_signal(prices)
    return str(signal["Prediction"].iloc[-1])


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
