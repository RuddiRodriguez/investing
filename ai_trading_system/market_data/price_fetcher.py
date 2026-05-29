import math

import yfinance as yf

from market_data.schemas import PriceBar


def _row_get(row, field: str, ticker: str):
    # yfinance can return MultiIndex columns like ('Open', 'NVDA').
    if (field, ticker) in row.index:
        return row[(field, ticker)]
    if field in row.index:
        return row[field]
    # Fallback: find any tuple whose first element is the field name.
    for key in row.index:
        if isinstance(key, tuple) and len(key) >= 1 and key[0] == field:
            return row[key]
    return None


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None


def fetch_price_bars(
    sector: str,
    ticker: str,
    timeframe: str = "1d",
    period: str = "6mo",
) -> list[PriceBar]:
    if timeframe not in {"1d", "1h"}:
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use '1d' or '1h'.")

    data = yf.download(
        tickers=ticker,
        period=period,
        interval=timeframe,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        return []

    price_bars: list[PriceBar] = []
    for timestamp, row in data.iterrows():
        if timeframe == "1d":
            timestamp_str = timestamp.strftime("%Y-%m-%d")
        else:
            timestamp_str = timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        adjusted_close = _safe_float(_row_get(row, "Adj Close", ticker))

        price_bars.append(
            PriceBar(
                sector=sector,
                ticker=ticker,
                timestamp=timestamp_str,
                timeframe=timeframe,
                open=_safe_float(_row_get(row, "Open", ticker)),
                high=_safe_float(_row_get(row, "High", ticker)),
                low=_safe_float(_row_get(row, "Low", ticker)),
                close=_safe_float(_row_get(row, "Close", ticker)),
                adjusted_close=adjusted_close,
                volume=_safe_int(_row_get(row, "Volume", ticker)),
                source="yfinance",
            )
        )

    return price_bars


def fetch_daily_price_bars(
    sector: str,
    ticker: str,
    period: str = "6mo",
) -> list[PriceBar]:
    return fetch_price_bars(
        sector=sector,
        ticker=ticker,
        timeframe="1d",
        period=period,
    )
