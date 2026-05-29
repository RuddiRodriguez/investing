"""Shared helpers for direction methods."""

from __future__ import annotations

import pandas as pd


CSV_FILE = "stock_prices.csv"
PRICE_COLUMN = "Close"
DATE_COLUMN = "Date"


def read_prices(csv_file: str) -> pd.DataFrame:
    prices = pd.read_csv(csv_file)
    if DATE_COLUMN in prices.columns:
        parsed_dates = pd.to_datetime(prices[DATE_COLUMN], errors="coerce", utc=True)
        prices[DATE_COLUMN] = parsed_dates.dt.tz_convert(None)
    return prices


def latest_signal_label(score: float, positive_threshold: float, negative_threshold: float) -> str:
    if score >= positive_threshold:
        return "UP"
    if score <= negative_threshold:
        return "DOWN"
    return "NO_TRADE"


def add_actual_direction(prices: pd.DataFrame) -> pd.DataFrame:
    output = prices.copy()
    output["Tomorrow Close"] = output[PRICE_COLUMN].shift(-1)
    output["Actual"] = output["Tomorrow Close"] > output[PRICE_COLUMN]
    return output


def add_prediction_correctness(prices: pd.DataFrame) -> pd.DataFrame:
    output = add_actual_direction(prices)
    output["PredictionBool"] = output["Prediction"].map({"UP": True, "DOWN": False})
    output["Traded"] = output["Prediction"].isin(["UP", "DOWN"])
    output["Correct"] = output["PredictionBool"] == output["Actual"]
    output.loc[~output["Traded"], "Correct"] = pd.NA
    return output


def summarize_predictions(prices: pd.DataFrame) -> tuple[float, float]:
    traded = prices.loc[prices["Traded"]].copy()
    coverage = float(prices["Traded"].mean()) if len(prices) else 0.0
    accuracy = float(traded["Correct"].mean()) if not traded.empty else 0.0
    return accuracy, coverage
