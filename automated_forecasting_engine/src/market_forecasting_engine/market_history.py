from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd


MARKET_RANGE_CONFIG: dict[str, dict[str, Any]] = {
    "1d": {"days": 1, "interval": "5m"},
    "2d": {"days": 2, "interval": "5m"},
    "1m": {"days": 31, "interval": "1h"},
    "2m": {"days": 62, "interval": "1h"},
    "3m": {"days": 93, "interval": "1d"},
    "6m": {"days": 186, "interval": "1d"},
}


def deribit_currency_to_yahoo_symbol(currency: str) -> str:
    return f"{currency.strip().upper()}-USD"


def fetch_yahoo_market_history(*, symbol: str, range_key: str = "6m") -> dict[str, Any]:
    range_key = range_key.lower().strip()
    config = MARKET_RANGE_CONFIG.get(range_key, MARKET_RANGE_CONFIG["6m"])
    end = datetime.now(UTC)
    start = end - timedelta(days=int(config["days"]))
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required for dashboard market history.") from exc
    frame = yf.download(
        symbol,
        start=start,
        end=end,
        interval=str(config["interval"]),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    rows = _frame_to_points(frame)
    return {
        "symbol": symbol,
        "range": range_key,
        "interval": config["interval"],
        "source": "yahoo_finance",
        "fetched_at": end.isoformat(),
        "current_price": rows[-1]["close"] if rows else None,
        "current_time": rows[-1]["time"] if rows else None,
        "point_count": len(rows),
        "points": rows,
    }


def _frame_to_points(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.copy()
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = [str(col[0]).lower() for col in clean.columns]
    else:
        clean.columns = [str(col).lower() for col in clean.columns]
    if "close" not in clean.columns:
        return []
    close = pd.to_numeric(clean["close"], errors="coerce").dropna()
    if close.empty:
        return []
    if close.index.tz is None:
        index = close.index.tz_localize(UTC)
    else:
        index = close.index.tz_convert(UTC)
    return [
        {"time": timestamp.isoformat(), "close": float(value)}
        for timestamp, value in zip(index.to_pydatetime(), close.to_numpy(), strict=False)
    ]
