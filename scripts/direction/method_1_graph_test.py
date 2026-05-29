"""Graph test for method 1: long-term trend signal with no-trade band."""

import plotly.express as px

try:
    from scripts.direction.common import CSV_FILE, PRICE_COLUMN, add_prediction_correctness
    from scripts.direction.method_1_moving_average import add_signal, read_prices
except ImportError:
    from common import CSV_FILE, PRICE_COLUMN, add_prediction_correctness
    from method_1_moving_average import add_signal, read_prices


def add_predictions(prices):
    output = add_signal(prices)
    output = add_prediction_correctness(output)
    output = output.dropna(subset=["Moving Average", "Actual"])
    return output


def print_accuracy(prices):
    traded = prices.loc[prices["Traded"]]
    accuracy = float(traded["Correct"].mean()) if not traded.empty else 0.0
    coverage = float(prices["Traded"].mean()) if len(prices) else 0.0
    print(f"Trade accuracy: {accuracy:.2%}")
    print(f"Signal coverage: {coverage:.2%}")


def plot_predictions(prices):
    prices["Result"] = prices["Prediction"].map({"UP": "Up signal", "DOWN": "Down signal", "NO_TRADE": "No trade"})

    fig = px.scatter(
        prices,
        x="Date",
        y=PRICE_COLUMN,
        color="Result",
        color_discrete_map={"Up signal": "green", "Down signal": "red", "No trade": "gray"},
        title="Method 1 Prediction Test",
    )
    fig.add_scatter(
        x=prices["Date"],
        y=prices["Moving Average"],
        mode="lines",
        name="Moving Average",
    )
    fig.show()


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    prices = add_predictions(prices)
    print_accuracy(prices)
    plot_predictions(prices)
