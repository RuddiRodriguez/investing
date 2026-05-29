"""Live Yahoo Finance data access with persistent caching."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .cache import DataCache
from .config import PipelineConfig


FUNDAMENTAL_FIELDS = {
    "revenue_growth": "revenueGrowth",
    "earnings_growth": "earningsGrowth",
    "gross_margin": "grossMargins",
    "operating_margin": "operatingMargins",
    "profit_margin": "profitMargins",
    "debt_to_equity": "debtToEquity",
    "return_on_equity": "returnOnEquity",
    "forward_pe": "forwardPE",
    "trailing_pe": "trailingPE",
    "market_cap": "marketCap",
    "free_cashflow": "freeCashflow",
    "sector": "sector",
}
POSITIVE_NEWS_WORDS = {
    "beat",
    "beats",
    "raise",
    "raised",
    "upgrade",
    "growth",
    "record",
    "profit",
    "strong",
    "surge",
    "wins",
    "approval",
}
NEGATIVE_NEWS_WORDS = {
    "miss",
    "misses",
    "cut",
    "downgrade",
    "weak",
    "fall",
    "falls",
    "lawsuit",
    "probe",
    "risk",
    "warning",
    "loss",
}


class LiveDataClient:
    """Fetch live market, fundamental, and news data with cache reuse."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = DataCache(config.cache_dir, ttl_hours=config.cache_ttl_hours)

    def fetch_prices(self) -> pd.DataFrame:
        """Return adjusted close prices for configured tickers and benchmark."""

        symbols = tuple(sorted(self.config.all_market_symbols()))
        params = {
            "symbols": symbols,
            "start": self.config.start_date,
            "end": self._effective_end_date(),
            "interval": self.config.interval,
            "auto_adjust": True,
        }
        key = self.cache.key_for("prices", params)
        if not self.config.force_refresh:
            cached = self.cache.read_frame(key)
            if cached is not None:
                return cached

        raw = yf.download(
            tickers=list(symbols),
            start=self.config.start_date,
            end=self.config.end_date,
            interval=self.config.interval,
            auto_adjust=True,
            group_by="column",
            progress=False,
            threads=True,
        )
        prices = self._extract_close_prices(raw, symbols)
        self.cache.write_frame(key, prices, params)
        return prices

    def fetch_fundamentals(self) -> pd.DataFrame:
        """Return current fundamentals from Yahoo Finance, cached by ticker/day."""

        symbols = tuple(sorted(self.config.normalized_tickers()))
        params = {"symbols": symbols, "as_of": self._effective_end_date(), "fields": sorted(FUNDAMENTAL_FIELDS)}
        key = self.cache.key_for("fundamentals", params)
        if not self.config.force_refresh:
            cached = self.cache.read_frame(key)
            if cached is not None:
                return cached

        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            info = self._safe_ticker_info(symbol)
            row = {"ticker": symbol}
            for output_name, yahoo_name in FUNDAMENTAL_FIELDS.items():
                row[output_name] = info.get(yahoo_name)
            market_cap = _to_float(row.get("market_cap"))
            free_cashflow = _to_float(row.get("free_cashflow"))
            row["free_cashflow_yield"] = free_cashflow / market_cap if market_cap and market_cap > 0 else np.nan
            rows.append(row)

        frame = pd.DataFrame(rows).set_index("ticker")
        self.cache.write_frame(key, frame, params)
        return frame

    def fetch_news_signals(self) -> pd.DataFrame:
        """Return simple live news sentiment features from recent Yahoo headlines."""

        symbols = tuple(sorted(self.config.normalized_tickers()))
        params = {"symbols": symbols, "as_of": self._effective_end_date(), "engine": "keyword_v1"}
        key = self.cache.key_for("news", params)
        if not self.config.force_refresh:
            cached = self.cache.read_frame(key)
            if cached is not None:
                return cached

        rows = []
        for symbol in symbols:
            headlines = self._safe_news(symbol)
            positive = 0
            negative = 0
            for title in headlines:
                words = set(str(title).lower().replace("-", " ").split())
                positive += len(words & POSITIVE_NEWS_WORDS)
                negative += len(words & NEGATIVE_NEWS_WORDS)
            total = max(len(headlines), 1)
            rows.append(
                {
                    "ticker": symbol,
                    "news_items": len(headlines),
                    "news_sentiment": (positive - negative) / total,
                    "positive_news_intensity": positive / total,
                    "negative_news_intensity": negative / total,
                }
            )

        frame = pd.DataFrame(rows).set_index("ticker")
        self.cache.write_frame(key, frame, params)
        return frame

    def _effective_end_date(self) -> str:
        return self.config.end_date or date.today().isoformat()

    def _extract_close_prices(self, raw: pd.DataFrame, symbols: tuple[str, ...]) -> pd.DataFrame:
        if raw.empty:
            raise ValueError("Yahoo Finance returned no price data.")
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise ValueError("Yahoo Finance response did not include Close prices.")
            prices = raw["Close"].copy()
        else:
            close_column = "Close" if "Close" in raw.columns else raw.columns[-1]
            prices = raw[[close_column]].copy()
            prices.columns = [symbols[0]]

        prices.columns = [str(column).upper() for column in prices.columns]
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        prices = prices.sort_index().apply(pd.to_numeric, errors="coerce").ffill().dropna(how="all")
        return prices

    def _safe_ticker_info(self, symbol: str) -> dict[str, Any]:
        try:
            return dict(yf.Ticker(symbol).get_info())
        except Exception:
            return {}

    def _safe_news(self, symbol: str) -> list[str]:
        try:
            news_items = yf.Ticker(symbol).news or []
        except Exception:
            return []
        headlines = []
        for item in news_items:
            title = item.get("title") or item.get("content", {}).get("title")
            if title:
                headlines.append(str(title))
        return headlines


def split_assets_and_benchmark(prices: pd.DataFrame, config: PipelineConfig) -> tuple[pd.DataFrame, pd.Series | None]:
    tickers = list(config.normalized_tickers())
    asset_prices = prices[tickers].copy()
    benchmark = None
    if config.benchmark and config.benchmark.upper() in prices.columns:
        benchmark = prices[config.benchmark.upper()].copy()
        benchmark.name = config.benchmark.upper()
    return asset_prices, benchmark


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
