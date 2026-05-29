"""Simple Yahoo Finance historical price downloader."""

import yfinance as yf


TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = None
OUTPUT_FILE = "stock_prices.csv"


def download_stock_prices(ticker, start_date, end_date, output_file):
    company = yf.Ticker(ticker)
    prices = company.history(
        start=start_date,
        end=end_date,
        auto_adjust=True,
    )
    prices.to_csv(output_file)
    return prices


if __name__ == "__main__":
    download_stock_prices(TICKER, START_DATE, END_DATE, OUTPUT_FILE)
