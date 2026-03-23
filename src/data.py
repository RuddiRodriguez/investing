"""Data loading helpers for the Streamlit app."""

from __future__ import annotations

import json
from io import StringIO
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf


FRED_CPI_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
EUROSTAT_BASE_URL = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"


def download_prices(tickers: list[str], start_date: str) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance."""

    data = yf.download(
        tickers=tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if data.empty:
        raise ValueError("Price download returned no data.")

    if "Close" in data.columns:
        prices = data["Close"].copy()
    else:
        prices = data.copy()

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    prices.columns = [str(column).upper() for column in prices.columns]
    return prices.dropna(how="all")


def parse_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Parse a CSV where the first column is the date index."""

    text = file_bytes.decode("utf-8")
    frame = pd.read_csv(StringIO(text), index_col=0, parse_dates=True)
    frame.columns = [str(column).upper() for column in frame.columns]
    return frame.sort_index()


def _read_text(url: str) -> str:
    with urlopen(url) as response:
        return response.read().decode("utf-8")


def _standardize_inflation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["cpi_index"] = pd.to_numeric(frame["cpi_index"], errors="coerce")
    frame["inflation_yoy"] = pd.to_numeric(frame["inflation_yoy"], errors="coerce")
    frame = frame.dropna(subset=["date", "cpi_index"]).sort_values("date")
    return frame.reset_index(drop=True)


def load_us_cpi() -> pd.DataFrame:
    """Load U.S. CPI data and derive year-over-year inflation."""

    text = _read_text(FRED_CPI_URL)
    frame = pd.read_csv(StringIO(text))
    if frame.empty or len(frame.columns) < 2:
        raise ValueError("CPI download returned no usable data.")

    date_column = frame.columns[0]
    value_column = frame.columns[1]
    frame = frame.rename(columns={date_column: "date", value_column: "cpi_index"})
    frame["inflation_yoy"] = pd.to_numeric(frame["cpi_index"], errors="coerce").pct_change(12)
    return _standardize_inflation_frame(frame[["date", "cpi_index", "inflation_yoy"]])


def _build_eurostat_url(dataset: str, **params: str) -> str:
    return f"{EUROSTAT_BASE_URL}/{dataset}?{urlencode(params)}"


def _parse_eurostat_series(payload: dict) -> pd.Series:
    if "value" not in payload or "dimension" not in payload:
        raise ValueError("Eurostat response did not include dataset values.")

    time_dimension = payload["dimension"].get("time", {})
    time_index = time_dimension.get("category", {}).get("index", {})
    if not time_index:
        raise ValueError("Eurostat response did not include a time dimension.")

    ordered_times = [label for label, _ in sorted(time_index.items(), key=lambda item: item[1])]
    values = payload["value"]

    if isinstance(values, list):
        series_values = values[: len(ordered_times)]
    elif isinstance(values, dict):
        series_values = [values.get(str(position)) for position in range(len(ordered_times))]
    else:
        raise ValueError("Eurostat response contained an unsupported value format.")

    return pd.Series(series_values, index=ordered_times, dtype="float64")


def load_euro_area_inflation(country_code: str) -> pd.DataFrame:
    """Load HICP index and annual inflation for a euro-area country."""

    index_url = _build_eurostat_url(
        "prc_hicp_midx",
        geo=country_code,
        coicop="CP00",
        unit="I15",
        freq="M",
    )
    rate_url = _build_eurostat_url(
        "prc_hicp_manr",
        geo=country_code,
        coicop="CP00",
        unit="RCH_A",
        freq="M",
    )

    index_payload = json.loads(_read_text(index_url))
    rate_payload = json.loads(_read_text(rate_url))

    index_series = _parse_eurostat_series(index_payload)
    rate_series = _parse_eurostat_series(rate_payload) / 100.0

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(index_series.index, format="%Y-%m"),
            "cpi_index": index_series.values,
            "inflation_yoy": rate_series.reindex(index_series.index).values,
        }
    )
    return _standardize_inflation_frame(frame)


def load_inflation_series(region: str) -> pd.DataFrame:
    """Load inflation data for a supported region."""

    if region == "US":
        return load_us_cpi()
    if region in {"NL", "ES"}:
        return load_euro_area_inflation(region)
    raise ValueError(f"Unsupported inflation region: {region}")
