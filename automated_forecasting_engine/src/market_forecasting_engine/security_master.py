from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_security_master(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Security master `{path}` is empty.")
    frame.columns = [_safe_label(str(column)) for column in frame.columns]
    ticker_column = _first_present(frame.columns, ("ticker", "symbol", "asset", "security"))
    if ticker_column is None:
        raise ValueError("Security master must include a ticker or symbol column.")
    frame[ticker_column] = frame[ticker_column].astype(str).str.upper()
    return frame.set_index(ticker_column, drop=False)


def resolve_security_metadata(
    ticker: str,
    prices: pd.DataFrame,
    security_master: pd.DataFrame | None = None,
    provider_metadata: dict[str, Any] | None = None,
    calendar: str = "XNYS",
    adjustment_policy: str = "auto_adjust",
) -> dict[str, Any]:
    symbol = ticker.upper()
    metadata: dict[str, Any] = {
        "ticker": symbol,
        "calendar": calendar,
        "adjustment_policy": adjustment_policy,
        "provider_metadata": provider_metadata or {},
        "corporate_action_columns_present": {
            "dividends": "dividends" in prices.columns,
            "stock_splits": "stock_splits" in prices.columns,
        },
        "split_event_rows": int((pd.to_numeric(prices.get("stock_splits", pd.Series(dtype=float)), errors="coerce") > 0).sum()),
        "dividend_event_rows": int((pd.to_numeric(prices.get("dividends", pd.Series(dtype=float)), errors="coerce") > 0).sum()),
    }

    if security_master is not None and symbol in security_master.index:
        row = security_master.loc[symbol]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]
        for key in (
            "exchange",
            "currency",
            "sector",
            "industry",
            "asset_class",
            "country",
            "active",
            "delisted",
            "first_trade_date",
            "last_trade_date",
            "figi",
            "cusip",
            "isin",
        ):
            if key in row.index and pd.notna(row[key]):
                metadata[key] = _json_scalar(row[key])
    return metadata


def _first_present(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _safe_label(label: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in label).strip("_")


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
