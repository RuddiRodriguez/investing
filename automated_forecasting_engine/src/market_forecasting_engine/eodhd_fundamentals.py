from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen


EODHD_BASE_URL = "https://eodhd.com/api"
DEFAULT_TIMEOUT_SECONDS = 30.0


class EodhdFundamentalsClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | None = None,
        base_url: str = EODHD_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _load_env_file(env_file)
        self.api_key = api_key or os.getenv("EODHD_API_KEY") or os.getenv("EOD_HISTORICAL_DATA_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("EODHD_API_KEY is required in the environment or .env file.")

    def fetch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = {key: value for key, value in (params or {}).items() if value is not None}
        query["api_token"] = self.api_key
        query.setdefault("fmt", "json")
        url = f"{self.base_url}/{path.lstrip('/')}?{urlencode(query)}"
        return _get_json(url, timeout=self.timeout)

    def fetch_company_context(
        self,
        symbol: str,
        *,
        from_date: str | None = None,
        to_date: str | None = None,
        news_limit: int = 5,
    ) -> dict[str, Any]:
        eodhd_symbol = normalize_eodhd_symbol(symbol)
        from_date = from_date or (datetime.now(timezone.utc).date() - timedelta(days=21)).isoformat()
        raw: dict[str, Any] = {}
        endpoint_status: dict[str, Any] = {}
        endpoints = {
            "fundamentals": (f"fundamentals/{quote(eodhd_symbol)}", {}),
            "eod_prices": (f"eod/{quote(eodhd_symbol)}", {"from": from_date, "to": to_date}),
            "real_time": (f"real-time/{quote(eodhd_symbol)}", {}),
            "news": ("news", {"s": eodhd_symbol, "limit": news_limit}),
            "earnings_calendar": ("calendar/earnings", {"symbols": eodhd_symbol, "from": from_date, "to": to_date}),
            "sma_20": (f"technical/{quote(eodhd_symbol)}", {"function": "sma", "period": 20, "from": from_date, "to": to_date}),
        }
        for name, (path, params) in endpoints.items():
            try:
                payload = self.fetch_json(path, params)
                raw[name] = payload
                endpoint_status[name] = _payload_status(payload)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                raw[name] = None
                endpoint_status[name] = {"status": "error", "error": _safe_error(exc)}

        normalized = normalize_eodhd_context(eodhd_symbol, raw)
        return {
            "symbol": symbol.upper().strip(),
            "eodhd_symbol": eodhd_symbol,
            "source": "eodhd",
            "source_base_url": self.base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_key_fingerprint": _key_fingerprint(self.api_key),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_eodhd_context(eodhd_symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    fundamentals = raw.get("fundamentals") if isinstance(raw.get("fundamentals"), dict) else {}
    general = fundamentals.get("General") if isinstance(fundamentals.get("General"), dict) else {}
    highlights = fundamentals.get("Highlights") if isinstance(fundamentals.get("Highlights"), dict) else {}
    valuation = fundamentals.get("Valuation") if isinstance(fundamentals.get("Valuation"), dict) else {}
    shares = fundamentals.get("SharesStats") if isinstance(fundamentals.get("SharesStats"), dict) else {}
    technicals = fundamentals.get("Technicals") if isinstance(fundamentals.get("Technicals"), dict) else {}
    analyst = fundamentals.get("AnalystRatings") if isinstance(fundamentals.get("AnalystRatings"), dict) else {}
    financials = fundamentals.get("Financials") if isinstance(fundamentals.get("Financials"), dict) else {}

    prices = raw.get("eod_prices") if isinstance(raw.get("eod_prices"), list) else []
    latest_price = _last_dict(prices)
    real_time = raw.get("real_time") if isinstance(raw.get("real_time"), dict) else {}
    news = raw.get("news") if isinstance(raw.get("news"), list) else []
    earnings = raw.get("earnings_calendar") if isinstance(raw.get("earnings_calendar"), list) else []
    sma = raw.get("sma_20") if isinstance(raw.get("sma_20"), list) else []
    latest_sma = _last_dict(sma)

    income = _latest_financial_statement(financials, "Income_Statement")
    balance = _latest_financial_statement(financials, "Balance_Sheet")
    cash_flow = _latest_financial_statement(financials, "Cash_Flow")

    return {
        "identity": _compact_dict(
            {
                "symbol": eodhd_symbol,
                "code": general.get("Code"),
                "exchange": general.get("Exchange"),
                "name": general.get("Name"),
                "type": general.get("Type"),
                "currency": general.get("CurrencyCode"),
                "country": general.get("CountryName"),
                "sector": general.get("Sector"),
                "industry": general.get("Industry"),
                "isin": general.get("ISIN"),
                "figi": general.get("FIGI"),
                "lei": general.get("LEI"),
                "website": general.get("WebURL"),
                "description": general.get("Description"),
            }
        ),
        "quote": _compact_dict(
            {
                "date": latest_price.get("date") or real_time.get("timestamp"),
                "open": latest_price.get("open") or real_time.get("open"),
                "high": latest_price.get("high") or real_time.get("high"),
                "low": latest_price.get("low") or real_time.get("low"),
                "close": latest_price.get("close") or real_time.get("close"),
                "adjusted_close": latest_price.get("adjusted_close"),
                "volume": latest_price.get("volume") or real_time.get("volume"),
                "previous_close": real_time.get("previousClose"),
                "change": real_time.get("change"),
                "change_pct": real_time.get("change_p"),
            }
        ),
        "valuation": _compact_dict(
            {
                "market_cap": highlights.get("MarketCapitalization"),
                "enterprise_value": valuation.get("EnterpriseValue"),
                "pe": highlights.get("PERatio") or valuation.get("TrailingPE"),
                "forward_pe": valuation.get("ForwardPE"),
                "peg": highlights.get("PEGRatio"),
                "pb": highlights.get("PriceBookMRQ"),
                "ps": highlights.get("PriceSalesTTM"),
                "ev_revenue": valuation.get("EnterpriseValueRevenue"),
                "ev_ebitda": valuation.get("EnterpriseValueEbitda"),
            }
        ),
        "profitability": _compact_dict(
            {
                "profit_margin": highlights.get("ProfitMargin"),
                "operating_margin": highlights.get("OperatingMarginTTM"),
                "return_on_assets": highlights.get("ReturnOnAssetsTTM"),
                "return_on_equity": highlights.get("ReturnOnEquityTTM"),
                "revenue_ttm": highlights.get("RevenueTTM"),
                "gross_profit_ttm": highlights.get("GrossProfitTTM"),
                "eps_diluted_ttm": highlights.get("DilutedEpsTTM"),
                "quarterly_revenue_growth_yoy": highlights.get("QuarterlyRevenueGrowthYOY"),
                "quarterly_earnings_growth_yoy": highlights.get("QuarterlyEarningsGrowthYOY"),
            }
        ),
        "shares": _compact_dict(
            {
                "shares_outstanding": shares.get("SharesOutstanding"),
                "shares_float": shares.get("SharesFloat"),
                "percent_insiders": shares.get("PercentInsiders"),
                "percent_institutions": shares.get("PercentInstitutions"),
            }
        ),
        "technicals": _compact_dict(
            {
                "beta": technicals.get("Beta"),
                "52_week_high": technicals.get("52WeekHigh"),
                "52_week_low": technicals.get("52WeekLow"),
                "sma_20_latest": latest_sma.get("sma") or latest_sma.get("value"),
                "sma_20_date": latest_sma.get("date"),
            }
        ),
        "analyst": _compact_dict(
            {
                "rating": analyst.get("Rating"),
                "target_price": analyst.get("TargetPrice"),
                "strong_buy": analyst.get("StrongBuy"),
                "buy": analyst.get("Buy"),
                "hold": analyst.get("Hold"),
                "sell": analyst.get("Sell"),
                "strong_sell": analyst.get("StrongSell"),
            }
        ),
        "income_statement": _compact_dict(
            {
                "date": income.get("date"),
                "revenue": income.get("totalRevenue"),
                "gross_profit": income.get("grossProfit"),
                "operating_income": income.get("operatingIncome"),
                "net_income": income.get("netIncome"),
                "ebitda": income.get("ebitda"),
            }
        ),
        "balance_sheet": _compact_dict(
            {
                "date": balance.get("date"),
                "total_assets": balance.get("totalAssets"),
                "total_liabilities": balance.get("totalLiab"),
                "total_equity": balance.get("totalStockholderEquity"),
                "cash": balance.get("cash"),
                "short_long_term_debt_total": balance.get("shortLongTermDebtTotal"),
            }
        ),
        "cash_flow": _compact_dict(
            {
                "date": cash_flow.get("date"),
                "operating_cash_flow": cash_flow.get("totalCashFromOperatingActivities"),
                "capital_expenditures": cash_flow.get("capitalExpenditures"),
                "free_cash_flow": cash_flow.get("freeCashFlow"),
                "dividends_paid": cash_flow.get("dividendsPaid"),
            }
        ),
        "earnings": [
            _compact_dict(
                {
                    "date": item.get("report_date") or item.get("date"),
                    "before_after_market": item.get("before_after_market"),
                    "currency": item.get("currency"),
                    "eps_actual": item.get("eps_actual"),
                    "eps_estimate": item.get("eps_estimate"),
                    "revenue_actual": item.get("revenue_actual"),
                    "revenue_estimate": item.get("revenue_estimate"),
                }
            )
            for item in earnings[:10]
            if isinstance(item, dict)
        ],
        "recent_news": [
            _compact_dict(
                {
                    "date": item.get("date"),
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "source": item.get("source"),
                    "symbols": item.get("symbols"),
                    "sentiment": item.get("sentiment"),
                    "content_excerpt": str(item.get("content") or "")[:1000],
                }
            )
            for item in news[:10]
            if isinstance(item, dict)
        ],
        "coverage": {
            "eod_price_rows": len(prices),
            "news_rows": len(news),
            "earnings_rows": len(earnings),
            "sma_rows": len(sma),
            "fundamentals_available": bool(fundamentals),
        },
    }


def normalize_eodhd_symbol(symbol: str) -> str:
    clean = symbol.upper().strip()
    if not clean:
        raise ValueError("symbol is required.")
    return clean if "." in clean else f"{clean}.US"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize EODHD company context for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--from-date", default=None)
    parser.add_argument("--to-date", default=None)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-raw", action="store_true", help="Print raw endpoint payloads in stdout output.")
    args = parser.parse_args(argv)

    client = EodhdFundamentalsClient(env_file=args.env_file)
    result = client.fetch_company_context(args.symbol, from_date=args.from_date, to_date=args.to_date, news_limit=args.news_limit)
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
        raise ValueError(f"EODHD returned non-JSON response: {payload[:200]}") from exc


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


def _latest_financial_statement(financials: dict[str, Any], statement_name: str) -> dict[str, Any]:
    section = financials.get(statement_name)
    if not isinstance(section, dict):
        return {}
    for key in ("quarterly", "yearly"):
        rows = section.get(key)
        if isinstance(rows, dict) and rows:
            date, row = max(rows.items(), key=lambda item: str(item[0]))
            if isinstance(row, dict):
                return {"date": date, **row}
    return {}


def _last_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in reversed(payload):
            if isinstance(item, dict):
                return item
    if isinstance(payload, dict):
        return payload
    return {}


def _compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None and value != ""}


def _payload_status(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        return {"status": "ok", "type": "list", "items": len(payload)}
    if isinstance(payload, dict):
        return {"status": "ok", "type": "dict", "keys": len(payload)}
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
