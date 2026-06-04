from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
                "capabilities": {
                    "intraday": True,
                    "production_sla": False,
                    "explicit_regular_session_flags": False,
                    "extended_intraday_history": False,
                    "notes": "Yahoo intraday history is capped and best treated as a research or validation source.",
                },
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
                "capabilities": {
                    "intraday": True,
                    "production_sla": None,
                    "explicit_regular_session_flags": None,
                    "extended_intraday_history": None,
                    "notes": "CSV quality depends on the upstream vendor and file construction.",
                },
            },
        )


class PolygonProvider:
    name = "polygon"
    base_url = "https://api.polygon.io"

    def __init__(self, api_key: str | None = None) -> None:
        _load_env_file()
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        if not self.api_key:
            raise RuntimeError("Provider `polygon` requires POLYGON_API_KEY.")
        multiplier, timespan = _polygon_interval(request.interval)
        start = request.start or "2000-01-01"
        end = request.end or pd.Timestamp.utcnow().date().isoformat()
        url = (
            f"{self.base_url}/v2/aggs/ticker/{request.ticker.upper()}/range/"
            f"{multiplier}/{timespan}/{start}/{end}?"
            + urlencode({"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": self.api_key})
        )
        rows: list[dict[str, Any]] = []
        next_url: str | None = url
        while next_url:
            payload = _get_json(next_url)
            rows.extend(payload.get("results", []) or [])
            next_url = payload.get("next_url")
            if next_url and "apiKey=" not in next_url:
                next_url = next_url + ("&" if "?" in next_url else "?") + urlencode({"apiKey": self.api_key})
        if not rows:
            raise ValueError(f"Polygon returned no bars for {request.ticker.upper()}.")
        raw = pd.DataFrame(rows)
        normalized = _normalize_polygon_bars(raw, target_column=request.target_column)
        return ProviderResult(
            frame=normalized,
            raw_frame=raw,
            metadata={
                "provider": self.name,
                "ticker": request.ticker.upper(),
                "request": request.to_dict(),
                "adjustment_policy": "adjusted",
                "raw_data_hash": frame_sha256(raw),
                "normalized_data_hash": frame_sha256(normalized),
                "cache_hit": False,
                "capabilities": {
                    "intraday": True,
                    "production_sla": True,
                    "explicit_regular_session_flags": False,
                    "extended_intraday_history": True,
                    "notes": "Polygon aggregate bars are suitable as a production-grade intraday source when subscribed to the required plan.",
                },
            },
        )


class AlpacaProvider:
    name = "alpaca"
    base_url = "https://data.alpaca.markets"

    def __init__(self, key_id: str | None = None, secret_key: str | None = None, feed: str | None = None) -> None:
        _load_env_file()
        self.key_id = key_id or os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        self.feed = feed or os.getenv("ALPACA_DATA_FEED", "iex")
        self.crypto_location = os.getenv("ALPACA_CRYPTO_LOCATION", "us")

    def fetch_prices(self, request: DataRequest) -> ProviderResult:
        if not self.key_id or not self.secret_key:
            raise RuntimeError("Provider `alpaca` requires ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.")
        if _is_crypto_symbol(request.ticker):
            return self._fetch_crypto_prices(request)
        return self._fetch_stock_prices(request)

    def _fetch_stock_prices(self, request: DataRequest) -> ProviderResult:
        params = {
            "timeframe": _alpaca_timeframe(request.interval),
            "start": _iso_utc(request.start),
            "adjustment": "all" if request.adjustment_policy in {"auto_adjust", "adjusted", "adjusted_close"} else "raw",
            "feed": self.feed,
            "limit": 10000,
        }
        if request.end:
            params["end"] = _iso_utc(request.end)
        url = f"{self.base_url}/v2/stocks/{request.ticker.upper()}/bars?" + urlencode(params)
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        rows: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while True:
            page_url = url
            if next_page_token:
                page_url += "&" + urlencode({"page_token": next_page_token})
            payload = _get_json(page_url, headers=headers)
            rows.extend(payload.get("bars", []) or [])
            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break
        if not rows:
            raise ValueError(f"Alpaca returned no bars for {request.ticker.upper()}.")
        raw = pd.DataFrame(rows)
        normalized = _normalize_alpaca_bars(raw, target_column=request.target_column)
        return ProviderResult(
            frame=normalized,
            raw_frame=raw,
            metadata={
                "provider": self.name,
                "ticker": request.ticker.upper(),
                "request": request.to_dict(),
                "feed": self.feed,
                "adjustment_policy": params["adjustment"],
                "raw_data_hash": frame_sha256(raw),
                "normalized_data_hash": frame_sha256(normalized),
                "cache_hit": False,
                "capabilities": {
                    "intraday": True,
                    "production_sla": True,
                    "explicit_regular_session_flags": True,
                    "extended_intraday_history": True,
                    "notes": "Alpaca historical bars are suitable for production-grade intraday research when using an appropriate market-data subscription/feed.",
                },
            },
        )

    def _fetch_crypto_prices(self, request: DataRequest) -> ProviderResult:
        symbol = _alpaca_crypto_symbol(request.ticker)
        params = {
            "symbols": symbol,
            "timeframe": _alpaca_timeframe(request.interval),
            "start": _iso_utc(request.start),
            "limit": 10000,
        }
        if request.end:
            params["end"] = _iso_utc(request.end)
        url = f"{self.base_url}/v1beta3/crypto/{self.crypto_location}/bars?" + urlencode(params)
        headers = {
            "APCA-API-KEY-ID": self.key_id,
            "APCA-API-SECRET-KEY": self.secret_key,
        }
        rows: list[dict[str, Any]] = []
        next_page_token: str | None = None
        while True:
            page_url = url
            if next_page_token:
                page_url += "&" + urlencode({"page_token": next_page_token})
            payload = _get_json(page_url, headers=headers)
            bars = payload.get("bars", []) or []
            if isinstance(bars, dict):
                rows.extend(bars.get(symbol, []) or [])
            else:
                rows.extend(bars)
            next_page_token = payload.get("next_page_token")
            if not next_page_token:
                break
        if not rows:
            raise ValueError(f"Alpaca returned no crypto bars for {symbol}.")
        raw = pd.DataFrame(rows)
        raw["symbol"] = symbol
        normalized = _normalize_alpaca_bars(raw, target_column=request.target_column)
        return ProviderResult(
            frame=normalized,
            raw_frame=raw,
            metadata={
                "provider": self.name,
                "ticker": symbol,
                "request": request.to_dict(),
                "asset_class": "crypto",
                "crypto_location": self.crypto_location,
                "raw_data_hash": frame_sha256(raw),
                "normalized_data_hash": frame_sha256(normalized),
                "cache_hit": False,
                "capabilities": {
                    "intraday": True,
                    "production_sla": True,
                    "explicit_regular_session_flags": False,
                    "extended_intraday_history": True,
                    "continuous_24_7_market": True,
                    "notes": "Alpaca crypto historical bars use the v1beta3 crypto market-data endpoint and are suitable for production research with the required subscription.",
                },
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
        return PolygonProvider()
    if normalized == "alpaca":
        return AlpacaProvider()
    if normalized == "iex":
        return NotConfiguredProvider("iex", "Add an IEX Cloud or IEX-compatible API-key-backed provider implementation.")
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


def _get_json(url: str, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_env_file() -> None:
    for path in _env_search_paths():
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def _env_search_paths() -> list[Path]:
    cwd = Path.cwd()
    return [cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]]


def _polygon_interval(interval: str) -> tuple[int, str]:
    value = interval.strip().lower()
    if value.endswith("m"):
        return int(float(value[:-1])), "minute"
    if value.endswith("h"):
        return int(float(value[:-1])), "hour"
    if value.endswith("d"):
        return int(float(value[:-1])), "day"
    raise ValueError(f"Unsupported Polygon interval `{interval}`.")


def _alpaca_timeframe(interval: str) -> str:
    value = interval.strip().lower()
    if value.endswith("m"):
        return f"{int(float(value[:-1]))}Min"
    if value.endswith("h"):
        return f"{int(float(value[:-1]))}Hour"
    if value.endswith("d"):
        return f"{int(float(value[:-1]))}Day"
    raise ValueError(f"Unsupported Alpaca interval `{interval}`.")


def _is_crypto_symbol(ticker: str) -> bool:
    normalized = ticker.strip().upper()
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        return bool(base) and quote in {"USD", "USDT", "USDC", "BTC", "ETH"}
    if "-" in normalized:
        base, quote = normalized.split("-", 1)
        return bool(base) and quote in {"USD", "USDT", "USDC", "BTC", "ETH"}
    return normalized in {"BTCUSD", "ETHUSD", "LTCUSD", "DOGEUSD", "SOLUSD"}


def _alpaca_crypto_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        return f"{base}/{quote}"
    if "-" in normalized:
        base, quote = normalized.split("-", 1)
        return f"{base}/{quote}"
    if normalized.endswith("USDT"):
        return f"{normalized[:-4]}/USDT"
    if normalized.endswith("USDC"):
        return f"{normalized[:-4]}/USDC"
    if normalized.endswith("USD"):
        return f"{normalized[:-3]}/USD"
    raise ValueError(f"Unsupported Alpaca crypto symbol `{ticker}`.")


def _iso_utc(value: str | None) -> str:
    timestamp = pd.Timestamp(value or "2000-01-01")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat().replace("+00:00", "Z")


def _normalize_polygon_bars(raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": raw["o"].to_numpy(),
            "high": raw["h"].to_numpy(),
            "low": raw["l"].to_numpy(),
            "close": raw["c"].to_numpy(),
            "volume": raw["v"].to_numpy(),
        },
        index=pd.to_datetime(raw["t"], unit="ms", utc=True).dt.tz_convert(None),
    )
    return normalize_price_frame(frame, target_column=target_column)


def _normalize_alpaca_bars(raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": raw["o"].to_numpy(),
            "high": raw["h"].to_numpy(),
            "low": raw["l"].to_numpy(),
            "close": raw["c"].to_numpy(),
            "volume": raw["v"].to_numpy(),
        },
        index=pd.to_datetime(raw["t"], utc=True).dt.tz_convert(None),
    )
    return normalize_price_frame(frame, target_column=target_column)


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
