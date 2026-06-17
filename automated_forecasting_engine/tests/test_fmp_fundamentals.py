from __future__ import annotations

import json

import pytest

from market_forecasting_engine import fmp_fundamentals
from market_forecasting_engine.fmp_fundamentals import FinancialModelingPrepClient, normalize_fmp_fundamentals


def test_normalize_fmp_fundamentals_compacts_core_sections() -> None:
    raw = {
        "quote": [{"symbol": "AAPL", "name": "Apple Inc.", "price": 291.13, "marketCap": 4275929952280}],
        "profile": [{"symbol": "AAPL", "companyName": "Apple Inc.", "sector": "Technology", "currency": "USD"}],
        "key_metrics_ttm": [{"enterpriseValueTTM": 4324312952280, "enterpriseValueOverEBITDATTM": 26.97}],
        "ratios_ttm": [{"priceEarningsRatioTTM": 31.2, "grossProfitMarginTTM": 0.47}],
        "income_statement": [{"date": "2025-09-27", "revenue": 416161000000, "netIncome": 112010000000}],
        "cash_flow": [{"freeCashFlow": 98767000000}],
        "balance_sheet": [{"totalDebt": 112377000000}],
        "income_statement_growth": [{"growthRevenue": 0.064}],
        "price_target_consensus": [{"targetLow": 250, "targetConsensus": 320, "targetHigh": 380}],
        "earnings": [{"date": "2026-01-29", "epsEstimated": 2.3}],
        "financial_scores": [{"altmanZScore": 12.3, "piotroskiScore": 9}],
    }

    normalized = normalize_fmp_fundamentals("AAPL", raw)

    assert normalized["identity"]["name"] == "Apple Inc."
    assert normalized["quote"]["price"] == 291.13
    assert normalized["valuation"]["enterprise_value"] == 4324312952280
    assert normalized["profitability"]["gross_margin"] == 0.47
    assert normalized["growth"]["revenue_growth"] == 0.064
    assert normalized["analyst"]["target_consensus"] == 320
    assert normalized["events"]["next_earnings_date"] == "2026-01-29"
    assert normalized["scores"]["piotroski_score"] == 9


def test_client_fetch_company_fundamentals_records_endpoint_status(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "quote": [{"price": 291.13}],
        "profile": [{"companyName": "Apple Inc.", "sector": "Technology"}],
        "key-metrics-ttm": [{"enterpriseValueTTM": 1}],
        "ratios-ttm": [{"priceEarningsRatioTTM": 2}],
        "financial-scores": [{"piotroskiScore": 9}],
        "income-statement": [{"revenue": 3}],
        "balance-sheet-statement": [{"totalDebt": 4}],
        "cash-flow-statement": [{"freeCashFlow": 5}],
        "income-statement-growth": [{"growthRevenue": 0.1}],
        "analyst-estimates": [{"estimatedEpsAvg": 6}],
        "price-target-consensus": [{"targetConsensus": 7}],
        "earnings": [{"date": "2026-01-29"}],
        "company-notes": [],
    }

    def fake_get_json(url: str, *, timeout: float = 30.0):
        for path, payload in responses.items():
            if f"/{path}?" in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setattr(fmp_fundamentals, "_get_json", fake_get_json)

    result = FinancialModelingPrepClient().fetch_company_fundamentals("aapl")

    assert result["symbol"] == "AAPL"
    assert result["endpoint_status"]["quote"]["items"] == 1
    assert result["normalized"]["analyst"]["target_consensus"] == 7
    assert result["api_key_fingerprint"].startswith("sha256:")
    json.dumps(result)
