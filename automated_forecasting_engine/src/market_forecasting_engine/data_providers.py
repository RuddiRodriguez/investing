from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from market_forecasting_engine.data import load_price_csv, normalize_price_frame
from market_forecasting_engine.data_store import MarketDataStore, frame_sha256, request_key


@dataclass(frozen=True)
class DataRequest:
    ticker: str
    start: str | None = None
    end: str | None = None
    target_column: str = "close"
    interval: str = "1d"
    adjustment_policy: str = "auto_adjust"
    source_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker.upper(),
            "start": self.start,
            "end": self.end,
            "target_column": self.target_column.lower(),
            "interval": self.interval,
            "adjustment_policy": self.adjustment_policy,
            "source_path": self.source_path,
        }

    def cache_key(self, provider: str) -> str:
        return request_key({"provider": provider, **self.to_dict()})


@dataclass
class ProviderResult:
    frame: pd.DataFrame
    raw_frame: pd.DataFrame | None
    metadata: dict[str, Any]


class MarketDataProvider(Protocol):
    name: str

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        ...


class YahooFinanceProvider:
    name = "yahoo"

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for Yahoo Finance downloads.") from exc

        auto_adjust = request.adjustment_policy in {"auto_adjust", "adjusted", "adjusted_close"}
        raw = yf.download(
            request.ticker,
            start=request.start,
            end=request.end,
            interval=request.interval,
            auto_adjust=auto_adjust,
            progress=False,
            actions=True,
        )
        normalized = normalize_price_frame(raw, target_column=request.target_column)
        return ProviderResult(
            frame=normalized,
            raw_frame=raw,
            metadata={
                "provider": self.name,
                "ticker": request.ticker.upper(),
                "request": request.to_dict(),
                "auto_adjust": auto_adjust,
                "raw_data_hash": frame_sha256(raw) if raw is not None and not raw.empty else None,
                "normalized_data_hash": frame_sha256(normalized),
                "cache_hit": False,
            },
        )


class CsvPriceProvider:
    name = "csv"

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        if not request.source_path:
            raise ValueError("CsvPriceProvider requires source_path.")
        raw = pd.read_csv(request.source_path)
        normalized = load_price_csv(request.source_path, target_column=request.target_column)
        return ProviderResult(
            frame=normalized,
            raw_frame=raw,
            metadata={
                "provider": self.name,
                "ticker": request.ticker.upper(),
                "request": request.to_dict(),
                "source_path": str(Path(request.source_path)),
                "raw_data_hash": frame_sha256(_index_raw_csv(raw)),
                "normalized_data_hash": frame_sha256(normalized),
                "cache_hit": False,
            },
        )


class NotConfiguredProvider:
    def __init__(self, name: str, setup_hint: str) -> None:
        self.name = name
        self.setup_hint = setup_hint

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        raise RuntimeError(
            f"Provider `{self.name}` is scaffolded but not configured. {self.setup_hint}"
        )


def provider_for_name(name: str) -> MarketDataProvider:
    normalized = name.lower().replace("_", "-")
    if normalized == "yahoo":
        return YahooFinanceProvider()
    if normalized == "csv":
        return CsvPriceProvider()
    if normalized == "polygon":
        return NotConfiguredProvider("polygon", "Add a Polygon.io API-key-backed provider implementation.")
    if normalized in {"alpha-vantage", "alphavantage"}:
        return NotConfiguredProvider("alpha-vantage", "Add an Alpha Vantage API-key-backed provider implementation.")
    if normalized in {"nasdaq-data-link", "quandl"}:
        return NotConfiguredProvider("nasdaq-data-link", "Add a Nasdaq Data Link API-key-backed provider implementation.")
    raise ValueError(f"Unknown data provider `{name}`.")


def load_prices_with_provider(
    provider_name: str,
    request: DataRequest,
    store: MarketDataStore | None = None,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> ProviderResult:
    provider = provider_for_name(provider_name)
    key = request.cache_key(provider.name)
    if store is not None and use_cache and not refresh_cache:
        cached = store.find_frame("normalized", provider.name, request.ticker, key)
        if cached is not None:
            frame = store.read_frame(cached)
            metadata = {
                "provider": provider.name,
                "ticker": request.ticker.upper(),
                "request": request.to_dict(),
                "request_key": key,
                "cache_hit": True,
                "artifacts": {
                    "normalized": {
                        "path": str(cached),
                        "format": cached.suffix.lstrip("."),
                        "normalized_data_hash": frame_sha256(frame),
                    }
                },
                "normalized_data_hash": frame_sha256(frame),
            }
            return ProviderResult(frame=frame, raw_frame=None, metadata=metadata)

    result = provider.fetch_prices(request)
    result.metadata["request_key"] = key
    if store is not None:
        artifacts: dict[str, Any] = {}
        if result.raw_frame is not None:
            artifacts["raw"] = store.write_frame("raw", provider.name, request.ticker, key, _storable_raw_frame(result.raw_frame)).to_dict()
        artifacts["normalized"] = store.write_frame("normalized", provider.name, request.ticker, key, result.frame).to_dict()
        result.metadata["artifacts"] = artifacts
    return result


def _index_raw_csv(raw: pd.DataFrame) -> pd.DataFrame:
    output = raw.copy()
    if not output.empty:
        first_column = output.columns[0]
        parsed = pd.to_datetime(output[first_column], errors="coerce", utc=True)
        if parsed.notna().mean() > 0.8:
            output.index = parsed
    return output


def _storable_raw_frame(raw: pd.DataFrame) -> pd.DataFrame:
    output = raw.copy()
    if isinstance(output.columns, pd.MultiIndex):
        output.columns = ["_".join(str(part) for part in column if part) for column in output.columns]
    return output
