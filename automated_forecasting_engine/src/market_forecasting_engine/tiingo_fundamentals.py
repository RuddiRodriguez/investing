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


TIINGO_BASE_URL = "https://api.tiingo.com"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class TiingoEndpoint:
    name: str
    path: str
    params: dict[str, Any]


class TiingoFundamentalsClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | None = None,
        base_url: str = TIINGO_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _load_env_file(env_file)
        self.api_key = api_key or os.getenv("TIINGO_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("TIINGO_API_KEY is required in the environment or .env file.")

    def fetch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = {key: value for key, value in (params or {}).items() if value is not None}
        url = f"{self.base_url}/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"
        return _get_json(url, api_key=self.api_key, timeout=self.timeout)

    def fetch_company_fundamentals(
        self,
        symbol: str,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        price_limit: int = 30,
        news_limit: int = 5,
        include_definitions: bool = False,
    ) -> dict[str, Any]:
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required.")
        price_params: dict[str, Any] = {"format": "json"}
        if start_date:
            price_params["startDate"] = start_date
        if end_date:
            price_params["endDate"] = end_date
        endpoints = [
            TiingoEndpoint("metadata", f"tiingo/daily/{symbol}", {}),
            TiingoEndpoint("prices", f"tiingo/daily/{symbol}/prices", price_params),
            TiingoEndpoint("fundamentals_daily", f"tiingo/fundamentals/{symbol}/daily", {}),
            TiingoEndpoint("fundamentals_statements", f"tiingo/fundamentals/{symbol}/statements", {}),
            TiingoEndpoint("news", "tiingo/news", {"tickers": symbol, "limit": news_limit}),
        ]
        if include_definitions:
            endpoints.append(TiingoEndpoint("fundamentals_definitions", "tiingo/fundamentals/definitions", {}))

        raw: dict[str, Any] = {}
        endpoint_status: dict[str, Any] = {}
        for endpoint in endpoints:
            try:
                payload = self.fetch_json(endpoint.path, endpoint.params)
                if endpoint.name == "prices" and isinstance(payload, list) and price_limit > 0:
                    payload = payload[-price_limit:]
                raw[endpoint.name] = payload
                endpoint_status[endpoint.name] = _payload_status(payload)
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                raw[endpoint.name] = None
                endpoint_status[endpoint.name] = {"status": "error", "error": _safe_error(exc)}

        normalized = normalize_tiingo_fundamentals(symbol, raw)
        return {
            "symbol": symbol,
            "source": "tiingo",
            "source_base_url": self.base_url,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "api_key_fingerprint": _key_fingerprint(self.api_key),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_tiingo_fundamentals(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    metadata = _as_dict(raw.get("metadata"))
    prices = raw.get("prices") if isinstance(raw.get("prices"), list) else []
    fundamentals_daily = raw.get("fundamentals_daily") if isinstance(raw.get("fundamentals_daily"), list) else []
    statements = raw.get("fundamentals_statements") if isinstance(raw.get("fundamentals_statements"), list) else []
    news = raw.get("news") if isinstance(raw.get("news"), list) else []

    latest_price = _last_dict(prices)
    latest_daily = _last_dict(fundamentals_daily)
    latest_statement = _first_dict(statements)
    statement_data = _statement_data_map(latest_statement)
    income = statement_data.get("incomeStatement", {})
    balance = statement_data.get("balanceSheet", {})
    cash_flow = statement_data.get("cashFlow", {})
    overview = statement_data.get("overview", {})
    revenue = _num(income.get("revenue"))
    gross_profit = _num(income.get("grossProfit"))
    operating_income = _num(income.get("opinc"))
    net_income = _num(income.get("netinc"))

    return {
        "identity": _compact_dict(
            {
                "symbol": symbol,
                "name": metadata.get("name"),
                "exchange_code": metadata.get("exchangeCode"),
                "start_date": metadata.get("startDate"),
                "end_date": metadata.get("endDate"),
                "description": metadata.get("description"),
            }
        ),
        "quote": _compact_dict(
            {
                "date": latest_price.get("date"),
                "open": latest_price.get("open"),
                "high": latest_price.get("high"),
                "low": latest_price.get("low"),
                "close": latest_price.get("close"),
                "volume": latest_price.get("volume"),
                "adj_open": latest_price.get("adjOpen"),
                "adj_high": latest_price.get("adjHigh"),
                "adj_low": latest_price.get("adjLow"),
                "adj_close": latest_price.get("adjClose"),
                "adj_volume": latest_price.get("adjVolume"),
                "div_cash": latest_price.get("divCash"),
                "split_factor": latest_price.get("splitFactor"),
            }
        ),
        "valuation": _compact_dict(
            {
                "date": latest_daily.get("date"),
                "market_cap": latest_daily.get("marketCap"),
                "enterprise_value": latest_daily.get("enterpriseVal"),
                "pe": latest_daily.get("peRatio"),
                "pb": latest_daily.get("pbRatio"),
                "trailing_peg_1y": latest_daily.get("trailingPEG1Y"),
            }
        ),
        "income_statement": _compact_dict(
            {
                "date": latest_statement.get("date"),
                "year": latest_statement.get("year"),
                "quarter": latest_statement.get("quarter"),
                "revenue": income.get("revenue"),
                "gross_profit": income.get("grossProfit"),
                "operating_income": income.get("opinc"),
                "net_income": income.get("netinc"),
                "eps": income.get("eps"),
                "eps_diluted": income.get("epsDil"),
                "shares_weighted_average": income.get("shareswa"),
                "shares_weighted_average_diluted": income.get("shareswaDil"),
            }
        ),
        "balance_sheet": _compact_dict(
            {
                "date": latest_statement.get("date"),
                "total_assets": _pick(balance, "totalAssets", "assets"),
                "total_liabilities": _pick(balance, "totalLiabilities", "liabilities"),
                "total_equity": balance.get("equity"),
                "cash_and_equivalents": balance.get("cashAndEq"),
                "total_debt": balance.get("debt"),
            }
        ),
        "cash_flow": _compact_dict(
            {
                "date": latest_statement.get("date"),
                "operating_cash_flow": cash_flow.get("ncfo"),
                "capital_expenditure": cash_flow.get("capex"),
                "free_cash_flow": cash_flow.get("freeCashFlow"),
                "investing_cash_flow": cash_flow.get("ncfi"),
                "financing_cash_flow": cash_flow.get("ncff"),
            }
        ),
        "profitability": _compact_dict(
            {
                "gross_margin": _safe_ratio(gross_profit, revenue) or overview.get("grossMargin"),
                "operating_margin": _safe_ratio(operating_income, revenue),
                "net_margin": _safe_ratio(net_income, revenue) or overview.get("profitMargin"),
                "roa": overview.get("roa"),
                "roe": overview.get("roe"),
                "roic": overview.get("roic"),
            }
        ),
        "recent_news": [
            _compact_dict(
                {
                    "title": item.get("title"),
                    "published_date": item.get("publishedDate"),
                    "source": item.get("source"),
                    "url": item.get("url"),
                    "description": item.get("description"),
                }
            )
            for item in news[:5]
            if isinstance(item, dict)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize Tiingo EOD and fundamentals for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--price-limit", type=int, default=30)
    parser.add_argument("--news-limit", type=int, default=5)
    parser.add_argument("--include-definitions", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--include-raw", action="store_true", help="Print raw endpoint payloads in stdout output.")
    args = parser.parse_args(argv)

    client = TiingoFundamentalsClient(env_file=args.env_file)
    result = client.fetch_company_fundamentals(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        price_limit=args.price_limit,
        news_limit=args.news_limit,
        include_definitions=args.include_definitions,
    )
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    display = result if args.include_raw else {key: value for key, value in result.items() if key != "raw"}
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


def _get_json(url: str, *, api_key: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    request = Request(
        url,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "market-forecasting-engine/0.1",
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tiingo returned non-JSON response: {payload[:200]}") from exc


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


def _statement_data_map(statement: dict[str, Any]) -> dict[str, dict[str, Any]]:
    statement_data = statement.get("statementData")
    if not isinstance(statement_data, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for section, rows in statement_data.items():
        if not isinstance(rows, list):
            continue
        result[section] = {
            str(item.get("dataCode")): item.get("value")
            for item in rows
            if isinstance(item, dict) and item.get("dataCode")
        }
    return result


def _as_dict(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _first_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
    if isinstance(payload, dict):
        return payload
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


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in {None, 0.0}:
        return None
    return numerator / denominator


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
