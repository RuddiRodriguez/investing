"""Risk method 2: drawdown."""

import pandas as pd


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def calculate_drawdown(prices):
    prices["Highest Price"] = prices[PRICE_COLUMN].cummax()
    prices["Drawdown"] = prices[PRICE_COLUMN] / prices["Highest Price"] - 1
    current_drawdown = prices["Drawdown"].iloc[-1]
    worst_drawdown = prices["Drawdown"].min()
    return current_drawdown, worst_drawdown


def classify_risk(current_drawdown):
    if current_drawdown > -0.10:
        return "LOW"

    if current_drawdown > -0.25:
        return "MEDIUM"

    return "HIGH"


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    current_drawdown, worst_drawdown = calculate_drawdown(prices)
    risk = classify_risk(current_drawdown)
    print(f"Current drawdown: {current_drawdown:.2%}")
    print(f"Worst drawdown: {worst_drawdown:.2%}")
    print(f"Risk: {risk}")
