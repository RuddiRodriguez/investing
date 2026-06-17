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


FMP_BASE_URL = "https://financialmodelingprep.com/stable"
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FmpEndpoint:
    name: str
    path: str
    params: dict[str, Any]


class FinancialModelingPrepClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        env_file: str | None = None,
        base_url: str = FMP_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        _load_env_file(env_file)
        self.api_key = api_key or os.getenv("FMP_API_KEY") or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("FMP_API_KEY is required in the environment or .env file.")

    def fetch_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = dict(params or {})
        query["apikey"] = self.api_key
        url = f"{self.base_url}/{path.lstrip('/')}?{urlencode(query)}"
        return _get_json(url, timeout=self.timeout)

    def fetch_company_fundamentals(self, symbol: str, *, limit: int = 5) -> dict[str, Any]:
        symbol = symbol.upper().strip()
        if not symbol:
            raise ValueError("symbol is required.")
        endpoints = [
            FmpEndpoint("quote", "quote", {"symbol": symbol}),
            FmpEndpoint("profile", "profile", {"symbol": symbol}),
            FmpEndpoint("key_metrics_ttm", "key-metrics-ttm", {"symbol": symbol}),
            FmpEndpoint("ratios_ttm", "ratios-ttm", {"symbol": symbol}),
            FmpEndpoint("financial_scores", "financial-scores", {"symbol": symbol}),
            FmpEndpoint("income_statement", "income-statement", {"symbol": symbol, "limit": limit}),
            FmpEndpoint("balance_sheet", "balance-sheet-statement", {"symbol": symbol, "limit": limit}),
            FmpEndpoint("cash_flow", "cash-flow-statement", {"symbol": symbol, "limit": limit}),
            FmpEndpoint("income_statement_growth", "income-statement-growth", {"symbol": symbol, "limit": limit}),
            FmpEndpoint("analyst_estimates", "analyst-estimates", {"symbol": symbol, "limit": limit}),
            FmpEndpoint("price_target_consensus", "price-target-consensus", {"symbol": symbol}),
            FmpEndpoint("earnings", "earnings", {"symbol": symbol}),
            FmpEndpoint("company_notes", "company-notes", {"symbol": symbol}),
        ]
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

        normalized = normalize_fmp_fundamentals(symbol, raw)
        fetched_at = datetime.now(timezone.utc).isoformat()
        return {
            "symbol": symbol,
            "source": "financial_modeling_prep",
            "source_base_url": self.base_url,
            "fetched_at": fetched_at,
            "api_key_fingerprint": _key_fingerprint(self.api_key),
            "endpoint_status": endpoint_status,
            "normalized": normalized,
            "raw": raw,
            "raw_sha256": _json_sha256(raw),
        }


def normalize_fmp_fundamentals(symbol: str, raw: dict[str, Any]) -> dict[str, Any]:
    quote = _first_dict(raw.get("quote"))
    profile = _first_dict(raw.get("profile"))
    metrics = _first_dict(raw.get("key_metrics_ttm"))
    ratios = _first_dict(raw.get("ratios_ttm"))
    scores = _first_dict(raw.get("financial_scores"))
    income = _first_dict(raw.get("income_statement"))
    balance = _first_dict(raw.get("balance_sheet"))
    cash_flow = _first_dict(raw.get("cash_flow"))
    growth = _first_dict(raw.get("income_statement_growth"))
    estimates = _first_dict(raw.get("analyst_estimates"))
    price_target = _first_dict(raw.get("price_target_consensus"))
    earnings = _first_dict(raw.get("earnings"))

    return {
        "identity": _compact_dict(
            {
                "symbol": symbol,
                "name": quote.get("name") or profile.get("companyName") or profile.get("name"),
                "exchange": quote.get("exchange") or profile.get("exchange"),
                "currency": quote.get("currency") or profile.get("currency"),
                "sector": profile.get("sector"),
                "industry": profile.get("industry"),
                "country": profile.get("country"),
                "website": profile.get("website"),
                "description": profile.get("description"),
            }
        ),
        "quote": _compact_dict(
            {
                "price": _pick(quote, "price", "previousClose"),
                "market_cap": _pick(quote, "marketCap", "marketCapTTM"),
                "volume": quote.get("volume"),
                "avg_volume": _pick(quote, "avgVolume", "averageVolume"),
                "day_low": _pick(quote, "dayLow", "low"),
                "day_high": _pick(quote, "dayHigh", "high"),
                "year_low": _pick(quote, "yearLow", "yearLowPrice"),
                "year_high": _pick(quote, "yearHigh", "yearHighPrice"),
            }
        ),
        "valuation": _compact_dict(
            {
                "market_cap": _pick(quote, "marketCap", "marketCapTTM") or metrics.get("marketCapTTM"),
                "enterprise_value": _pick(metrics, "enterpriseValueTTM", "enterpriseValue"),
                "pe": _pick(ratios, "priceEarningsRatioTTM", "peRatioTTM") or _pick(metrics, "peRatioTTM", "peRatio"),
                "forward_pe": estimates.get("estimatedPeAvg"),
                "pb": _pick(ratios, "priceToBookRatioTTM", "pbRatioTTM") or metrics.get("pbRatioTTM"),
                "ps": _pick(ratios, "priceToSalesRatioTTM", "priceToSalesRatioTTM") or metrics.get("priceToSalesRatioTTM"),
                "ev_to_ebitda": _pick(metrics, "enterpriseValueOverEBITDATTM", "evToEBITDATTM"),
                "pfcf": _pick(metrics, "pfcfRatioTTM", "priceToFreeCashFlowsRatioTTM"),
            }
        ),
        "income_statement": _compact_dict(
            {
                "date": income.get("date"),
                "period": income.get("period"),
                "revenue": income.get("revenue"),
                "gross_profit": income.get("grossProfit"),
                "operating_income": income.get("operatingIncome"),
                "net_income": income.get("netIncome"),
                "eps": _pick(income, "eps", "epsdiluted") or metrics.get("netIncomePerShareTTM"),
                "eps_diluted": _pick(income, "epsdiluted", "epsDiluted"),
            }
        ),
        "cash_flow": _compact_dict(
            {
                "date": cash_flow.get("date"),
                "operating_cash_flow": cash_flow.get("operatingCashFlow"),
                "capital_expenditure": cash_flow.get("capitalExpenditure"),
                "free_cash_flow": cash_flow.get("freeCashFlow"),
                "net_cash_used_for_investing": cash_flow.get("netCashUsedForInvestingActivites"),
                "net_cash_used_for_financing": cash_flow.get("netCashUsedProvidedByFinancingActivities"),
            }
        ),
        "balance_sheet": _compact_dict(
            {
                "date": balance.get("date"),
                "total_assets": balance.get("totalAssets"),
                "total_debt": balance.get("totalDebt"),
                "total_equity": balance.get("totalEquity"),
                "cash_and_short_term_investments": balance.get("cashAndShortTermInvestments"),
            }
        ),
        "profitability": _compact_dict(
            {
                "gross_margin": _pick(ratios, "grossProfitMarginTTM", "grossProfitMargin"),
                "operating_margin": _pick(ratios, "operatingProfitMarginTTM", "operatingProfitMargin"),
                "net_margin": _pick(ratios, "netProfitMarginTTM", "netProfitMargin"),
                "roa": _pick(ratios, "returnOnAssetsTTM", "returnOnAssets"),
                "roe": _pick(ratios, "returnOnEquityTTM", "returnOnEquity"),
                "roic": _pick(ratios, "returnOnInvestedCapitalTTM", "returnOnInvestedCapital"),
            }
        ),
        "growth": _compact_dict(
            {
                "revenue_growth": growth.get("growthRevenue"),
                "gross_profit_growth": growth.get("growthGrossProfit"),
                "operating_income_growth": growth.get("growthOperatingIncome"),
                "net_income_growth": growth.get("growthNetIncome"),
                "eps_growth": _pick(growth, "growthEPS", "growthEPSDiluted"),
            }
        ),
        "analyst": _compact_dict(
            {
                "estimated_revenue_avg": estimates.get("estimatedRevenueAvg"),
                "estimated_eps_avg": estimates.get("estimatedEpsAvg"),
                "estimated_ebitda_avg": estimates.get("estimatedEbitdaAvg"),
                "estimated_net_income_avg": estimates.get("estimatedNetIncomeAvg"),
                "target_low": _pick(price_target, "targetLow", "low"),
                "target_high": _pick(price_target, "targetHigh", "high"),
                "target_median": _pick(price_target, "targetMedian", "median"),
                "target_consensus": _pick(price_target, "targetConsensus", "consensus", "targetMean"),
            }
        ),
        "events": _compact_dict(
            {
                "next_earnings_date": _pick(earnings, "date", "fiscalDateEnding", "reportDate"),
                "eps_estimated": earnings.get("epsEstimated"),
                "revenue_estimated": earnings.get("revenueEstimated"),
            }
        ),
        "scores": _compact_dict(
            {
                "altman_z_score": _pick(scores, "altmanZScore", "altmanZScoreTTM"),
                "piotroski_score": _pick(scores, "piotroskiScore", "piotroskiScoreTTM"),
                "working_capital": scores.get("workingCapital"),
            }
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and normalize FMP fundamentals for one ticker.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--include-raw", action="store_true", help="Print raw endpoint payloads in stdout output.")
    args = parser.parse_args(argv)

    client = FinancialModelingPrepClient(env_file=args.env_file)
    result = client.fetch_company_fundamentals(args.symbol, limit=args.limit)
    if args.output:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    display = result if args.include_raw else {k: v for k, v in result.items() if k != "raw"}
    print(json.dumps(display, indent=2, sort_keys=True))
    return 0


def _get_json(url: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Any:
    request = Request(url, headers={"User-Agent": "market-forecasting-engine/0.1"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"FMP returned non-JSON response: {payload[:200]}") from exc


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


def _first_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _pick(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


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
