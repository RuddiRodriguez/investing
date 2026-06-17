from __future__ import annotations

import json

import pytest

from market_forecasting_engine import eodhd_fundamentals
from market_forecasting_engine.eodhd_fundamentals import EodhdFundamentalsClient, normalize_eodhd_context, normalize_eodhd_symbol


def test_normalize_eodhd_symbol_adds_us_only_when_missing_exchange() -> None:
    assert normalize_eodhd_symbol("AAPL") == "AAPL.US"
    assert normalize_eodhd_symbol("ASML.AS") == "ASML.AS"


def test_normalize_eodhd_context_compacts_available_sections() -> None:
    raw = {
        "fundamentals": {
            "General": {"Code": "AAPL", "Exchange": "US", "Name": "Apple Inc.", "Sector": "Technology"},
            "Highlights": {"MarketCapitalization": 1, "PERatio": 30, "RevenueTTM": 100},
            "Valuation": {"EnterpriseValue": 2},
            "SharesStats": {"SharesOutstanding": 3},
            "Technicals": {"Beta": 1.2},
            "AnalystRatings": {"TargetPrice": 320},
        },
        "eod_prices": [{"date": "2026-06-12", "open": 296.03, "high": 297.14, "low": 289.62, "close": 291.13, "volume": 38742100}],
        "real_time": {"previousClose": 295.63, "change": -4.5, "change_p": -1.5222},
        "news": [{"date": "2026-06-14T09:20:00+00:00", "title": "Apple news", "source": "Example", "content": "Long text"}],
        "earnings_calendar": [{"report_date": "2026-07-30", "eps_estimate": 1.86}],
        "sma_20": [{"date": "2026-06-12", "sma": 303.88}],
    }

    normalized = normalize_eodhd_context("AAPL.US", raw)

    assert normalized["identity"]["name"] == "Apple Inc."
    assert normalized["quote"]["close"] == 291.13
    assert normalized["valuation"]["market_cap"] == 1
    assert normalized["technicals"]["sma_20_latest"] == 303.88
    assert normalized["analyst"]["target_price"] == 320
    assert normalized["earnings"][0]["eps_estimate"] == 1.86
    assert normalized["recent_news"][0]["title"] == "Apple news"


def test_client_fetch_company_context_records_endpoint_status(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_json(url: str, *, timeout: float = 30.0):
        if "api_token=test-key" not in url:
            raise AssertionError(url)
        if "/eod/AAPL.US?" in url:
            return [{"date": "2026-06-12", "close": 291.13}]
        if "/real-time/AAPL.US?" in url:
            return {"close": 291.13}
        if "/news?" in url:
            return [{"title": "News"}]
        return {}

    monkeypatch.setenv("EODHD_API_KEY", "test-key")
    monkeypatch.setattr(eodhd_fundamentals, "_get_json", fake_get_json)

    result = EodhdFundamentalsClient().fetch_company_context("aapl", from_date="2026-06-01")

    assert result["symbol"] == "AAPL"
    assert result["eodhd_symbol"] == "AAPL.US"
    assert result["endpoint_status"]["eod_prices"]["items"] == 1
    assert result["normalized"]["quote"]["close"] == 291.13
    assert result["api_key_fingerprint"].startswith("sha256:")
    json.dumps(result)
