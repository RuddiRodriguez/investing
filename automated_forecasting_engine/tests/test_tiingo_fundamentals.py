from __future__ import annotations

import json

import pytest

from market_forecasting_engine import tiingo_fundamentals
from market_forecasting_engine.tiingo_fundamentals import TiingoFundamentalsClient, normalize_tiingo_fundamentals


def test_normalize_tiingo_fundamentals_compacts_core_sections() -> None:
    raw = {
        "metadata": {
            "ticker": "AAPL",
            "name": "Apple Inc",
            "exchangeCode": "NASDAQ",
            "description": "Consumer electronics company.",
        },
        "prices": [
            {
                "date": "2026-06-12T00:00:00.000Z",
                "open": 290,
                "high": 295,
                "low": 288,
                "close": 291.13,
                "volume": 37905580,
                "adjClose": 291.13,
                "divCash": 0,
                "splitFactor": 1,
            }
        ],
        "fundamentals_daily": [
            {
                "date": "2026-06-12T00:00:00.000Z",
                "marketCap": 4275929952280,
                "enterpriseVal": 4324312952280,
                "peRatio": 31.2,
                "pbRatio": 40.2,
            }
        ],
        "fundamentals_statements": [
            {
                "date": "2026-03-28",
                "year": 2026,
                "quarter": 2,
                "statementData": {
                    "incomeStatement": [
                        {"dataCode": "revenue", "value": 1000},
                        {"dataCode": "grossProfit", "value": 500},
                        {"dataCode": "opinc", "value": 300},
                        {"dataCode": "netinc", "value": 250},
                        {"dataCode": "epsDil", "value": 2.01},
                    ],
                    "balanceSheet": [
                        {"dataCode": "assets", "value": 2000},
                        {"dataCode": "debt", "value": 400},
                    ],
                    "cashFlow": [
                        {"dataCode": "ncfo", "value": 350},
                        {"dataCode": "capex", "value": -50},
                    ],
                    "overview": [
                        {"dataCode": "roe", "value": 0.18},
                        {"dataCode": "roa", "value": 0.09},
                    ],
                },
            }
        ],
        "news": [{"title": "Apple news", "publishedDate": "2026-06-12T10:00:00Z", "source": "Example"}],
    }

    normalized = normalize_tiingo_fundamentals("AAPL", raw)

    assert normalized["identity"]["name"] == "Apple Inc"
    assert normalized["quote"]["close"] == 291.13
    assert normalized["valuation"]["market_cap"] == 4275929952280
    assert normalized["income_statement"]["revenue"] == 1000
    assert normalized["balance_sheet"]["total_debt"] == 400
    assert normalized["cash_flow"]["operating_cash_flow"] == 350
    assert normalized["profitability"]["roe"] == 0.18
    assert normalized["recent_news"][0]["title"] == "Apple news"


def test_client_fetch_company_fundamentals_records_endpoint_status(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "tiingo/daily/AAPL/prices?": [{"date": "2026-06-12T00:00:00.000Z", "close": 291.13}],
        "tiingo/daily/AAPL": {"ticker": "AAPL", "name": "Apple Inc"},
        "tiingo/fundamentals/AAPL/daily": [{"marketCap": 1}],
        "tiingo/fundamentals/AAPL/statements": [{"statementData": {}}],
        "tiingo/news?": [{"title": "News"}],
    }

    def fake_get_json(url: str, *, api_key: str, timeout: float = 30.0):
        for marker, payload in responses.items():
            if marker in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setenv("TIINGO_API_KEY", "test-key")
    monkeypatch.setattr(tiingo_fundamentals, "_get_json", fake_get_json)

    result = TiingoFundamentalsClient().fetch_company_fundamentals("aapl", start_date="2026-06-01")

    assert result["symbol"] == "AAPL"
    assert result["endpoint_status"]["metadata"]["status"] == "ok"
    assert result["normalized"]["quote"]["close"] == 291.13
    assert result["api_key_fingerprint"].startswith("sha256:")
    json.dumps(result)
