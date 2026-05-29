"""Walk-forward graph test for the LightGBM three-class model."""

import sys
from pathlib import Path

import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.direction.method_5_lgbm import (
    CSV_FILE,
    DOWN_SIGNAL,
    FEATURE_COLUMNS,
    FUTURE_DAYS,
    HOLD_SIGNAL,
    PRICE_COLUMN,
    UP_SIGNAL,
    add_features,
    add_target,
    fit_model,
    get_class_probabilities,
    get_signal,
    get_trade_return,
    read_prices,
    tune_model_settings,
)


MIN_TRAIN_SIZE = 252
TEST_RATIO = 0.25
TRANSACTION_COST = 0.001


def check_predictions(prices):
    prepared_prices = add_features(prices.copy())
    prepared_prices = add_target(prepared_prices)
    prepared_prices = prepared_prices.dropna().reset_index(drop=True)

    test_start = max(int(len(prepared_prices) * (1 - TEST_RATIO)), MIN_TRAIN_SIZE)
    if test_start >= len(prepared_prices):
        raise ValueError("Not enough data to reserve an untouched test period.")

    tuning_prices = prepared_prices.iloc[:test_start]
    best_settings = tune_model_settings(tuning_prices)

    results = []
    for current_index in range(test_start, len(prepared_prices), FUTURE_DAYS):
        train_prices = prepared_prices.iloc[:current_index]
        if train_prices["Target"].nunique() < 2:
            continue

        current_row = prepared_prices.iloc[[current_index]]
        model = fit_model(train_prices, best_settings)
        probabilities = get_class_probabilities(model, current_row[FEATURE_COLUMNS])
        signal = get_signal(probabilities[UP_SIGNAL], probabilities[DOWN_SIGNAL])
        future_return = current_row["Future Return"].iat[0]

        results.append(
            {
                "Date": current_row["Date"].iat[0],
                PRICE_COLUMN: current_row[PRICE_COLUMN].iat[0],
                "Probability Up": probabilities[UP_SIGNAL],
                "Probability Hold": probabilities[HOLD_SIGNAL],
                "Probability Down": probabilities[DOWN_SIGNAL],
                "Signal": signal,
                "Prediction": {UP_SIGNAL: 1, HOLD_SIGNAL: 0, DOWN_SIGNAL: -1}[signal],
                "Target": current_row["Target"].iat[0],
                "Correct": {UP_SIGNAL: 1, HOLD_SIGNAL: 0, DOWN_SIGNAL: -1}[signal]
                == current_row["Target"].iat[0],
                "Future Return": future_return,
                "Strategy Return": get_trade_return(signal, future_return, TRANSACTION_COST),
                "Buy Hold Return": future_return,
            }
        )

    test_prices = prices.__class__(results)
    test_prices["Strategy Equity"] = (1 + test_prices["Strategy Return"]).cumprod()
    test_prices["Buy Hold Equity"] = (1 + test_prices["Buy Hold Return"]).cumprod()
    return test_prices


def get_max_drawdown(equity):
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def print_results(prices):
    up_signals = prices[prices["Signal"] == UP_SIGNAL]
    down_signals = prices[prices["Signal"] == DOWN_SIGNAL]
    hold_signals = prices[prices["Signal"] == HOLD_SIGNAL]
    strong_signals = prices[prices["Signal"] != HOLD_SIGNAL]
    strategy_trades = prices[prices["Strategy Return"] != 0]
    winning_trades = strategy_trades[strategy_trades["Strategy Return"] > 0]
    losing_trades = strategy_trades[strategy_trades["Strategy Return"] < 0]

    accuracy = prices["Correct"].mean()
    strong_accuracy = strong_signals["Correct"].mean()
    average_return = prices["Future Return"].mean()
    up_average_return = up_signals["Future Return"].mean()
    down_average_return = down_signals["Future Return"].mean()
    up_total_return = (1 + up_signals["Strategy Return"]).prod() - 1
    up_win_rate = (up_signals["Strategy Return"] > 0).mean()
    up_average_trade = up_signals["Strategy Return"].mean()
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
    print(f"Transaction cost per trade: {TRANSACTION_COST:.2%}")
    print(f"Average return after UP signals: {up_average_return:.2%}")
    print(f"Average return after DOWN signals: {down_average_return:.2%}")
    print(f"Strategy total return: {strategy_total_return:.2%}")
    print(f"Buy and hold total return: {buy_hold_total_return:.2%}")
    print(f"Strategy max drawdown: {strategy_max_drawdown:.2%}")
    print(f"Buy and hold max drawdown: {buy_hold_max_drawdown:.2%}")
    print(f"Strategy win rate: {win_rate:.2%}")
    print(f"Average winning trade: {average_win:.2%}")
    print(f"Average losing trade: {average_loss:.2%}")
    print(f"UP-only total return: {up_total_return:.2%}")
    print(f"UP-only win rate: {up_win_rate:.2%}")
    print(f"UP-only average trade return: {up_average_trade:.2%}")


def plot_predictions(original_prices, predictions):
    fig = px.line(
        original_prices,
        x="Date",
        y=PRICE_COLUMN,
        title="Method 5 LightGBM Prediction Test",
    )
    fig.update_traces(line=dict(color="steelblue", width=2), name="Close")

    signal_fig = px.scatter(
        predictions,
        x="Date",
        y=PRICE_COLUMN,
        color="Signal",
        color_discrete_map={UP_SIGNAL: "green", HOLD_SIGNAL: "gray", DOWN_SIGNAL: "red"},
    )

    for trace in signal_fig.data:
        trace.update(marker=dict(size=10))
        fig.add_trace(trace)

    fig.show()


def plot_equity(prices):
    fig = px.line(
        prices,
        x="Date",
        y=["Strategy Equity", "Buy Hold Equity"],
        title="Method 5 LightGBM Strategy vs Buy And Hold",
    )
    fig.show()


if __name__ == "__main__":
    original_prices = read_prices(CSV_FILE)
    predictions = check_predictions(original_prices)
    print_results(predictions)
    plot_predictions(original_prices, predictions)
    plot_equity(predictions)
