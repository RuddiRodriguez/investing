"""Method 5: predict direction using improved logistic regression."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
FUTURE_DAYS = 30
MIN_FUTURE_RETURN = 0.02
UP_PROBABILITY = 0.60
DOWN_PROBABILITY = 0.45
FEATURE_COLUMNS = [
    "Return 1",
    "Return 5",
    "Return 10",
    "Return 20",
    "Distance 50",
    "Distance 200",
    "Volatility 20",
    "RSI",
    "Volume Change 20",
    "Distance High 20",
    "Distance Low 20",
    "Drawdown 60",
]


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def add_rsi(prices):
    price_change = prices[PRICE_COLUMN].diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)

    average_gain = gains.rolling(14).mean()
    average_loss = losses.rolling(14).mean()

    prices["RSI"] = 100 - (100 / (1 + average_gain / average_loss))
    return prices


def add_features(prices):
    prices["Return 1"] = prices[PRICE_COLUMN].pct_change()
    prices["Return 5"] = prices[PRICE_COLUMN].pct_change(5)
    prices["Return 10"] = prices[PRICE_COLUMN].pct_change(10)
    prices["Return 20"] = prices[PRICE_COLUMN].pct_change(20)

    prices["Moving Average 50"] = prices[PRICE_COLUMN].rolling(50).mean()
    prices["Moving Average 200"] = prices[PRICE_COLUMN].rolling(200).mean()
    prices["Distance 50"] = prices[PRICE_COLUMN] / prices["Moving Average 50"] - 1
    prices["Distance 200"] = prices[PRICE_COLUMN] / prices["Moving Average 200"] - 1

    prices["Volatility 20"] = prices["Return 1"].rolling(20).std()
    prices["Volume Change 20"] = prices["Volume"].pct_change(20)

    prices["High 20"] = prices[PRICE_COLUMN].rolling(20).max()
    prices["Low 20"] = prices[PRICE_COLUMN].rolling(20).min()
    prices["High 60"] = prices[PRICE_COLUMN].rolling(60).max()
    prices["Distance High 20"] = prices[PRICE_COLUMN] / prices["High 20"] - 1
    prices["Distance Low 20"] = prices[PRICE_COLUMN] / prices["Low 20"] - 1
    prices["Drawdown 60"] = prices[PRICE_COLUMN] / prices["High 60"] - 1

    prices = add_rsi(prices)
    return prices


def add_target(prices):
    prices["Future Close"] = prices[PRICE_COLUMN].shift(-FUTURE_DAYS)
    prices["Future Return"] = prices["Future Close"] / prices[PRICE_COLUMN] - 1
    prices["Target"] = prices["Future Return"] > MIN_FUTURE_RETURN
    return prices


def train_model(prices):
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    settings = {
        "logisticregression__C": [0.01, 0.1, 1, 10],
        "logisticregression__class_weight": [None, "balanced"],
    }

    split = TimeSeriesSplit(n_splits=10)
    search = GridSearchCV(model, settings, cv=split)
    search.fit(prices[FEATURE_COLUMNS], prices["Target"])

    print("Best settings:")
    print(search.best_params_)

    return search.best_estimator_


def get_signal(probability_up):
    if probability_up >= UP_PROBABILITY:
        return "UP"

    if probability_up <= DOWN_PROBABILITY:
        return "DOWN"

    return "HOLD"


def predict_direction(prices):
    prices = add_features(prices)

    training_prices = add_target(prices.copy())
    training_prices = training_prices.dropna()

    model = train_model(training_prices)

    latest_prices = prices.dropna().tail(1)
    probability_up = model.predict_proba(latest_prices[FEATURE_COLUMNS])[0][1]

    print(f"Probability of more than {MIN_FUTURE_RETURN:.0%} gain in {FUTURE_DAYS} days: {probability_up:.2%}")
    return get_signal(probability_up)


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    direction = predict_direction(prices)
    print(direction)
