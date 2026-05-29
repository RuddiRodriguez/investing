"""Risk method 4: expected shortfall."""

import pandas as pd


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
DAYS = 252
CONFIDENCE = 0.95


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def calculate_expected_shortfall(prices):
    prices["Return"] = prices[PRICE_COLUMN].pct_change()
    returns = prices["Return"].tail(DAYS)
    var = returns.quantile(1 - CONFIDENCE)
    expected_shortfall = returns[returns <= var].mean()
    return var, expected_shortfall


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    var, expected_shortfall = calculate_expected_shortfall(prices)
    print(f"Daily VaR {CONFIDENCE:.0%}: {var:.2%}")
    print(f"Expected shortfall: {expected_shortfall:.2%}")
    print("Meaning: when bad days are worse than VaR, this is the average bad loss.")
