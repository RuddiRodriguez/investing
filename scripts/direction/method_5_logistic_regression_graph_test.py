"""Graph test for method 5: improved logistic regression."""

import pandas as pd
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
FUTURE_DAYS = 10
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
    prices["Next Close"] = prices[PRICE_COLUMN].shift(-1)
    prices["Next Day Return"] = prices["Next Close"] / prices[PRICE_COLUMN] - 1
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

    split = TimeSeriesSplit(n_splits=5)
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


def check_predictions(prices):
    prices = add_features(prices)
    prices = add_target(prices)
    prices = prices.dropna()

    train_size = int(len(prices) * 0.7)
    train_prices = prices.iloc[:train_size]
    test_prices = prices.iloc[train_size:].copy()

    model = train_model(train_prices)
    test_prices["Probability Up"] = model.predict_proba(test_prices[FEATURE_COLUMNS])[:, 1]
    test_prices["Signal"] = test_prices["Probability Up"].apply(get_signal)
    test_prices["Prediction"] = test_prices["Signal"] == "UP"
    test_prices["Correct"] = test_prices["Prediction"] == test_prices["Target"]
    test_prices["Strategy Return"] = 0.0
    test_prices.loc[test_prices["Signal"] == "UP", "Strategy Return"] = test_prices["Next Day Return"]
    test_prices["Buy Hold Return"] = test_prices["Next Day Return"]
    test_prices["Strategy Equity"] = (1 + test_prices["Strategy Return"]).cumprod()
    test_prices["Buy Hold Equity"] = (1 + test_prices["Buy Hold Return"]).cumprod()
    return test_prices


def get_max_drawdown(equity):
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def print_results(prices):
    up_signals = prices[prices["Signal"] == "UP"]
    down_signals = prices[prices["Signal"] == "DOWN"]
    hold_signals = prices[prices["Signal"] == "HOLD"]
    strong_signals = prices[prices["Signal"] != "HOLD"]
    strategy_trades = prices[prices["Strategy Return"] != 0]
    winning_trades = strategy_trades[strategy_trades["Strategy Return"] > 0]
    losing_trades = strategy_trades[strategy_trades["Strategy Return"] < 0]

    accuracy = prices["Correct"].mean()
    strong_accuracy = strong_signals["Correct"].mean()
    average_return = prices["Future Return"].mean()
    up_average_return = up_signals["Future Return"].mean()
    down_average_return = down_signals["Future Return"].mean()
    strategy_total_return = prices["Strategy Equity"].iloc[-1] - 1
    buy_hold_total_return = prices["Buy Hold Equity"].iloc[-1] - 1
    strategy_max_drawdown = get_max_drawdown(prices["Strategy Equity"])
    buy_hold_max_drawdown = get_max_drawdown(prices["Buy Hold Equity"])
    win_rate = (strategy_trades["Strategy Return"] > 0).mean()
    average_win = winning_trades["Strategy Return"].mean()
    average_loss = losing_trades["Strategy Return"].mean()

    print(f"Accuracy: {accuracy:.2%}")
    print(f"Strong signal accuracy: {strong_accuracy:.2%}")
    print(f"Average 10-day return: {average_return:.2%}")
    print(f"UP signals: {len(up_signals)}")
    print(f"HOLD signals: {len(hold_signals)}")
    print(f"DOWN signals: {len(down_signals)}")
    print(f"Average return after UP signals: {up_average_return:.2%}")
    print(f"Average return after DOWN signals: {down_average_return:.2%}")
    print(f"Strategy total return: {strategy_total_return:.2%}")
    print(f"Buy and hold total return: {buy_hold_total_return:.2%}")
    print(f"Strategy max drawdown: {strategy_max_drawdown:.2%}")
    print(f"Buy and hold max drawdown: {buy_hold_max_drawdown:.2%}")
    print(f"Strategy win rate: {win_rate:.2%}")
    print(f"Average winning trade: {average_win:.2%}")
    print(f"Average losing trade: {average_loss:.2%}")


def plot_predictions(prices):
    fig = px.scatter(
        prices,
        x="Date",
        y=PRICE_COLUMN,
        color="Signal",
        color_discrete_map={"UP": "green", "HOLD": "gray", "DOWN": "red"},
        title="Method 5 Logistic Regression Prediction Test",
    )
    fig.show()


def plot_equity(prices):
    fig = px.line(
        prices,
        x="Date",
        y=["Strategy Equity", "Buy Hold Equity"],
        title="Method 5 Strategy vs Buy And Hold",
    )
    fig.show()


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    prices = check_predictions(prices)
    print_results(prices)
    plot_predictions(prices)
    plot_equity(prices)
