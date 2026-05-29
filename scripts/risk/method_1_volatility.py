"""Risk method 1: volatility."""

import pandas as pd


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
DAYS = 20


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def calculate_volatility(prices):
    prices["Return"] = prices[PRICE_COLUMN].pct_change()
    volatility = prices["Return"].tail(DAYS).std() * (252 ** 0.5)
    return volatility


def classify_risk(volatility):
    if volatility < 0.20:
        return "LOW"

    if volatility < 0.40:
        return "MEDIUM"

    return "HIGH"


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    volatility = calculate_volatility(prices)
    risk = classify_risk(volatility)
    print(f"Volatility: {volatility:.2%}")
    print(f"Risk: {risk}")
