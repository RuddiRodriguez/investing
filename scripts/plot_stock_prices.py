"""Simple stock price plotter."""

import pandas as pd
import plotly.express as px


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"


def read_prices(csv_file):
    return pd.read_csv(csv_file)


def plot_prices(prices, price_column):
    fig = px.line(prices, x="Date", y=price_column, title=f"{price_column} price")
    fig.show()


if __name__ == "__main__":
    prices = read_prices(CSV_FILE)
    plot_prices(prices, PRICE_COLUMN)
