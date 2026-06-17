from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MASSIVE_BASE_URL = "https://api.massive.com"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class MassiveEndpoint:
    name: str
    path: str
    params: dict[str, Any]


class MassiveFundamentalsClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | None = None,
        base_url: str = MASSIVE_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _load_env_file(env_file)
        self.api_key = api_key or os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY is required in the environment or .env file.")

    def fetch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = {key: value for key, value in (params or {}).items() if value is not None}
        query["apiKey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}?{urlencode(query)}"
        return _get_json(url, timeout=self.timeout)

    def fetch_company_context(
        self,
        symbol: str,
        *,
        limit: int = 5,
        include_financials: bool = True,
        include_filings_text: bool = True,
    ) -> dict[str, Any]:
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required.")

        endpoints = [
            MassiveEndpoint("ticker_overview", f"v3/reference/tickers/{symbol}", {}),
            MassiveEndpoint("previous_day_bar", f"v2/aggs/ticker/{symbol}/prev", {"adjusted": "true"}),
            MassiveEndpoint("dividends", "v3/reference/dividends", {"ticker": symbol, "limit": limit}),
            MassiveEndpoint("news", "v2/reference/news", {"ticker": symbol, "limit": limit}),
            MassiveEndpoint(
                "sma_20",
                f"v1/indicators/sma/{symbol}",
                {"timespan": "day", "adjusted": "true", "window": 20, "series_type": "close", "limit": limit},
            ),
            MassiveEndpoint("short_interest", "stocks/v1/short-interest", {"ticker": symbol, "limit": limit, "sort": "settlement_date.desc"}),
        ]
        if include_financials:
            endpoints.extend(
                [
                    MassiveEndpoint("ratios", "stocks/financials/v1/ratios", {"ticker": symbol, "limit": 1}),
                    MassiveEndpoint(
                        "income_statement",
                        "stocks/financials/v1/income-statements",
                        {"tickers": symbol, "limit": 1, "sort": "period_end.desc"},
                    ),
                    MassiveEndpoint(
                        "balance_sheet",
                        "stocks/financials/v1/balance-sheets",
                        {"tickers": symbol, "limit": 1, "sort": "period_end.desc"},
                    ),
                    MassiveEndpoint(
                        "cash_flow",
                        "stocks/financials/v1/cash-flow-statements",
                        {"tickers": symbol, "limit": 1, "sort": "period_end.desc"},
                    ),
                ]
            )
        if include_filings_text:
            endpoints.extend(
                [
                    MassiveEndpoint("ten_k_business", "stocks/filings/10-K/vX/sections", {"ticker": symbol, "limit": 1, "section": "business"}),
                    MassiveEndpoint("ten_k_risks", "stocks/filings/10-K/vX/sections", {"ticker": symbol, "limit": 1, "section": "risk_factors"}),
                    MassiveEndpoint("eight_k_text", "stocks/filings/8-K/vX/text", {"ticker": symbol, "limit": 1}),
                ]
            )

        raw: dict[str, Any] = {}
        endpoint_status: dict[str, Any] = {}
        for endpoint in endpoints:
            try:
                payload = self.fetch_json(endpoint.path, endpoint.params)
                raw[endpoint.name] = payload
                endpoint_status[endpoint.name] = _payload_status(payload)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                raw[endpoint.name] = None
                endpoint_status[endpoint.name] = {"status": "error", "error": _safe_error(exc)}

        normalized = normalize_massive_company_context(symbol, raw)
        return {
            "symbol": symbol,
            "source": "massive",
            "source_base_url": self.base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_key_fingerprint": _key_fingerprint(self.api_key),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_massive_company_context(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    overview = _results(raw.get("ticker_overview"))
    prev_bar = _first_result(raw.get("previous_day_bar"))
    dividends = _results_list(raw.get("dividends"))
    news = _results_list(raw.get("news"))
    sma_values = _nested_values(raw.get("sma_20"), "results", "values")
    short_interest = _first_result(raw.get("short_interest"))
    ratios = _first_result(raw.get("ratios"))
    income = _first_result(raw.get("income_statement"))
    balance = _first_result(raw.get("balance_sheet"))
    cash_flow = _first_result(raw.get("cash_flow"))
    ten_k_business = _first_result(raw.get("ten_k_business"))
    ten_k_risks = _first_result(raw.get("ten_k_risks"))
    eight_k = _first_result(raw.get("eight_k_text"))

    return {
        "identity": _compact_dict(
            {
                "symbol": symbol,
                "name": overview.get("name"),
                "market": overview.get("market"),
                "locale": overview.get("locale"),
                "primary_exchange": overview.get("primary_exchange"),
                "type": overview.get("type"),
                "active": overview.get("active"),
                "currency": overview.get("currency_name"),
                "cik": overview.get("cik"),
                "composite_figi": overview.get("composite_figi"),
                "share_class_figi": overview.get("share_class_figi"),
                "market_cap": overview.get("market_cap"),
                "sic_description": overview.get("sic_description"),
                "homepage_url": overview.get("homepage_url"),
                "description": overview.get("description"),
            }
        ),
        "quote": _compact_dict(
            {
                "ticker": prev_bar.get("T"),
                "timestamp_ms": prev_bar.get("t"),
                "open": prev_bar.get("o"),
                "high": prev_bar.get("h"),
                "low": prev_bar.get("l"),
                "close": prev_bar.get("c"),
                "volume": prev_bar.get("v"),
                "vwap": prev_bar.get("vw"),
                "transactions": prev_bar.get("n"),
            }
        ),
        "valuation": _compact_dict(
            {
                "market_cap": overview.get("market_cap") or ratios.get("market_cap"),
                "enterprise_value": ratios.get("enterprise_value"),
                "pe": _pick(ratios, "price_to_earnings_ratio", "pe_ratio"),
                "pb": _pick(ratios, "price_to_book_ratio", "pb_ratio"),
                "ps": _pick(ratios, "price_to_sales_ratio", "ps_ratio"),
                "ev_to_ebitda": _pick(ratios, "enterprise_value_to_ebitda_ratio", "ev_to_ebitda"),
            }
        ),
        "income_statement": _compact_dict(
            {
                "period_end": income.get("period_end"),
                "timeframe": income.get("timeframe"),
                "revenue": income.get("revenues"),
                "gross_profit": income.get("gross_profit"),
                "operating_income": income.get("operating_income_loss"),
                "net_income": income.get("net_income_loss"),
                "eps_basic": income.get("basic_earnings_per_share"),
                "eps_diluted": income.get("diluted_earnings_per_share"),
            }
        ),
        "balance_sheet": _compact_dict(
            {
                "period_end": balance.get("period_end"),
                "timeframe": balance.get("timeframe"),
                "assets": balance.get("assets"),
                "liabilities": balance.get("liabilities"),
                "equity": balance.get("equity"),
                "cash": balance.get("cash_and_cash_equivalents_at_carrying_value"),
                "debt": _pick(balance, "debt", "long_term_debt", "short_term_borrowings"),
            }
        ),
        "cash_flow": _compact_dict(
            {
                "period_end": cash_flow.get("period_end"),
                "timeframe": cash_flow.get("timeframe"),
                "operating_cash_flow": cash_flow.get("net_cash_from_operating_activities"),
                "investing_cash_flow": cash_flow.get("net_cash_from_investing_activities"),
                "financing_cash_flow": cash_flow.get("net_cash_from_financing_activities"),
                "capital_expenditure": cash_flow.get("payments_to_acquire_property_plant_and_equipment"),
            }
        ),
        "technical": _compact_dict(
            {
                "sma_20_latest": _first_value(sma_values, "value"),
                "sma_20_timestamp_ms": _first_value(sma_values, "timestamp"),
            }
        ),
        "short_interest": _compact_dict(
            {
                "settlement_date": short_interest.get("settlement_date"),
                "short_interest": short_interest.get("short_interest"),
                "avg_daily_volume": short_interest.get("avg_daily_volume"),
                "days_to_cover": short_interest.get("days_to_cover"),
            }
        ),
        "dividends": [
            _compact_dict(
                {
                    "ex_dividend_date": item.get("ex_dividend_date"),
                    "pay_date": item.get("pay_date"),
                    "cash_amount": item.get("cash_amount"),
                    "currency": item.get("currency"),
                    "frequency": item.get("frequency"),
                    "dividend_type": item.get("dividend_type"),
                }
            )
            for item in dividends[:10]
            if isinstance(item, dict)
        ],
        "recent_news": [
            _compact_dict(
                {
                    "title": item.get("title"),
                    "published_utc": item.get("published_utc"),
                    "article_url": item.get("article_url"),
                    "author": item.get("author"),
                    "publisher": _nested(item, "publisher", "name"),
                    "tickers": item.get("tickers"),
                    "description": item.get("description"),
                }
            )
            for item in news[:10]
            if isinstance(item, dict)
        ],
        "filings": _compact_dict(
            {
                "latest_10k_business": _filing_summary(ten_k_business, "text"),
                "latest_10k_risk_factors": _filing_summary(ten_k_risks, "text"),
                "latest_8k": _filing_summary(eight_k, "items_text"),
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize Massive company context for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--no-financials", action="store_true")
    parser.add_argument("--no-filings-text", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-raw", action="store_true", help="Print raw endpoint payloads in stdout output.")
    args = parser.parse_args(argv)

    client = MassiveFundamentalsClient(env_file=args.env_file)
    result = client.fetch_company_context(
        args.symbol,
        limit=args.limit,
        include_financials=not args.no_financials,
        include_filings_text=not args.no_filings_text,
    )
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    display = result if args.include_raw else {key: value for key, value in result.items() if key != "raw"}
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    request = Request(url, headers={"User-Agent": "market-forecasting-engine/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Massive returned non-JSON response: {payload[:200]}") from exc


def _load_env_file(env_file: str | None = None) -> None:
    paths = [Path(env_file).expanduser()] if env_file else []
    cwd = Path.cwd()
    paths.extend([cwd / ".env", *[parent / ".env" for parent in cwd.parents[:4]]])
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _results(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        return payload["results"]
    return payload if isinstance(payload, dict) else {}


def _results_list(payload: Any) -> list[Any]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return payload["results"]
    return payload if isinstance(payload, list) else []


def _first_result(payload: Any) -> dict[str, Any]:
    results = _results_list(payload)
    for item in results:
        if isinstance(item, dict):
            return item
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        return payload["results"]
    return {}


def _nested_values(payload: Any, *keys: str) -> list[Any]:
    value = payload
    for key in keys:
        if not isinstance(value, dict):
            return []
        value = value.get(key)
    return value if isinstance(value, list) else []


def _first_value(rows: list[Any], key: str) -> Any:
    for row in rows:
        if isinstance(row, dict) and row.get(key) is not None:
            return row.get(key)
    return None


def _nested(row: dict[str, Any], *keys: str) -> Any:
    value: Any = row
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _filing_summary(row: dict[str, Any], text_key: str) -> dict[str, Any] | None:
    text = row.get(text_key)
    if not isinstance(text, str):
        return None
    return _compact_dict(
        {
            "ticker": row.get("ticker"),
            "cik": row.get("cik"),
            "form_type": row.get("form_type"),
            "section": row.get("section"),
            "filing_date": row.get("filing_date"),
            "period_end": row.get("period_end"),
            "accession_number": row.get("accession_number"),
            "text_excerpt": text[:4000],
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    )


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != {} and value != ""}


def _payload_status(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        status: dict[str, Any] = {
            "status": "ok" if payload.get("status") in {None, "OK"} else str(payload.get("status")).lower(),
            "type": "dict",
            "keys": len(payload),
        }
        if isinstance(payload.get("results"), list):
            status["items"] = len(payload["results"])
        elif isinstance(payload.get("results"), dict):
            status["result_keys"] = len(payload["results"])
        if payload.get("request_id"):
            status["request_id"] = payload.get("request_id")
        return status
    if isinstance(payload, list):
        return {"status": "ok", "type": "list", "items": len(payload)}
    return {"status": "ok", "type": type(payload).__name__}


def _safe_error(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    return exc.__class__.__name__


def _key_fingerprint(api_key: str) -> str:
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
