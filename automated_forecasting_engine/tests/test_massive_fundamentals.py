from __future__ import annotations

import json

import pytest

from market_forecasting_engine import massive_fundamentals
from market_forecasting_engine.massive_fundamentals import MassiveFundamentalsClient, normalize_massive_company_context


def test_normalize_massive_company_context_compacts_core_sections() -> None:
    raw = {
        "ticker_overview": {
            "status": "OK",
            "results": {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "market": "stocks",
                "primary_exchange": "XNAS",
                "currency_name": "usd",
                "cik": "0000320193",
                "market_cap": 4275929952280,
                "description": "Technology company.",
            },
        },
        "previous_day_bar": {
            "status": "OK",
            "results": [{"T": "AAPL", "o": 296.03, "h": 297.14, "l": 289.62, "c": 291.13, "v": 38784789, "t": 1781294400000}],
        },
        "dividends": {"results": [{"cash_amount": 0.27, "ex_dividend_date": "2026-05-11", "currency": "USD"}]},
        "news": {"results": [{"title": "Apple news", "published_utc": "2026-06-14T11:30:28Z", "publisher": {"name": "Benzinga"}}]},
        "sma_20": {"results": {"values": [{"timestamp": 1781236800000, "value": 303.88}]}},
        "short_interest": {"results": [{"settlement_date": "2026-05-29", "short_interest": 155886024, "days_to_cover": 3.38}]},
        "ten_k_business": {"results": [{"ticker": "AAPL", "section": "business", "filing_date": "2025-10-31", "text": "Item 1. Business"}]},
        "eight_k_text": {"results": [{"ticker": "AAPL", "form_type": "8-K", "filing_date": "2026-04-20", "items_text": "Item 5.02 CEO transition"}]},
    }

    normalized = normalize_massive_company_context("AAPL", raw)

    assert normalized["identity"]["name"] == "Apple Inc."
    assert normalized["quote"]["close"] == 291.13
    assert normalized["valuation"]["market_cap"] == 4275929952280
    assert normalized["technical"]["sma_20_latest"] == 303.88
    assert normalized["short_interest"]["days_to_cover"] == 3.38
    assert normalized["dividends"][0]["cash_amount"] == 0.27
    assert normalized["recent_news"][0]["publisher"] == "Benzinga"
    assert normalized["filings"]["latest_10k_business"]["text_excerpt"] == "Item 1. Business"


def test_client_fetch_company_context_records_endpoint_status(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "v3/reference/tickers/AAPL?": {"status": "OK", "results": {"ticker": "AAPL", "name": "Apple Inc."}},
        "v2/aggs/ticker/AAPL/prev?": {"status": "OK", "results": [{"c": 291.13}]},
        "v3/reference/dividends?": {"status": "OK", "results": []},
        "v2/reference/news?": {"status": "OK", "results": []},
        "v1/indicators/sma/AAPL?": {"status": "OK", "results": {"values": []}},
        "stocks/v1/short-interest?": {"status": "OK", "results": []},
    }

    def fake_get_json(url: str, *, timeout: float = 30.0):
        for marker, payload in responses.items():
            if marker in url:
                return payload
        raise AssertionError(url)

    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    monkeypatch.setattr(massive_fundamentals, "_get_json", fake_get_json)

    result = MassiveFundamentalsClient().fetch_company_context("aapl", include_financials=False, include_filings_text=False)

    assert result["symbol"] == "AAPL"
    assert result["endpoint_status"]["ticker_overview"]["status"] == "ok"
    assert result["normalized"]["identity"]["name"] == "Apple Inc."
    assert result["api_key_fingerprint"].startswith("sha256:")
    json.dumps(result)
