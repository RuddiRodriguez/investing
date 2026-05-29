"""Risk method 3: historical Value at Risk."""

import pandas as pd


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
DAYS = 252
CONFIDENCE = 0.95


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def calculate_var(prices):
    prices["Return"] = prices[PRICE_COLUMN].pct_change()
    returns = prices["Return"].tail(DAYS)
    var = returns.quantile(1 - CONFIDENCE)
    return var


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    var = calculate_var(prices)
    print(f"Daily VaR {CONFIDENCE:.0%}: {var:.2%}")
    print("Meaning: on a bad normal day, loss may be worse than this.")
