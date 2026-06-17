from __future__ import annotations

import json

from market_forecasting_engine import long_term_sources
from market_forecasting_engine.long_term_sources import LongTermSourceRequest, collect_long_term_source_context
from market_forecasting_engine.stockanalysis_fundamentals import normalize_stockanalysis_context


def test_normalize_stockanalysis_context_extracts_visible_sections() -> None:
    raw = {
        "overview": """
        <html><head><title>Example (EXM)</title></head><body>
        <h1>Example Corp. (EXM)</h1>
        <div>NASDAQ: EXM · Real-Time Price · USD 42.50</div>
        <div>Market Cap 4.2B Revenue (ttm) 1.1B Net Income 120M EPS 2.30 PE Ratio 18.2 Price Target 55.00 Analysts Buy Volume 1.2M Open 41.00 Previous Close 40.00</div>
        <h2>About EXM</h2><p>Example builds industrial software.</p>
        <h2>News</h2><a href="/news/example">Example wins contract</a>
        </body></html>
        """,
        "financials": """
        <table>
        <thead><tr><th>Metric</th><th>TTM</th><th>2025</th></tr></thead>
        <tbody>
        <tr><td>Revenue</td><td>1.1B</td><td>900M</td></tr>
        <tr><td>Gross Profit</td><td>600M</td><td>500M</td></tr>
        <tr><td>Operating Income</td><td>220M</td><td>180M</td></tr>
        <tr><td>Net Income</td><td>120M</td><td>90M</td></tr>
        <tr><td>Free Cash Flow</td><td>100M</td><td>80M</td></tr>
        <tr><td>Gross Margin</td><td>54.5%</td><td>55.5%</td></tr>
        <tr><td>Operating Margin</td><td>20.0%</td><td>20.0%</td></tr>
        <tr><td>Profit Margin</td><td>10.9%</td><td>10.0%</td></tr>
        </tbody></table>
        <div>Last updated: Jun 1, 2026</div>
        """,
        "forecast": """
        <h1>Example Stock Forecast</h1>
        <p>According to 7 analysts, EXM has a consensus rating of "Buy". The average price target is $55.00.</p>
        <div>Target Low Average Median High Price$45.00$55.00$56.00$70.00</div>
        """,
        "statistics": """
        <div>Enterprise Value 4.5B PB Ratio 3.1 PS Ratio 3.8 EV / EBITDA 11.2 Current Ratio 1.8 Debt / Equity 0.4</div>
        """,
        "dividend": """
        <div>Annual Dividend 1.20 Dividend Yield 2.5% Ex-Dividend Date Jun 20, 2026 Payout Ratio 40%</div>
        <table><tr><th>Ex-Dividend Date</th><th>Cash Amount</th></tr><tr><td>Jun 20, 2026</td><td>0.30</td></tr></table>
        """,
        "company": """
        <h1>Company Description</h1><p>Example Corp makes durable software.</p>
        <div>Country United States Founded 2010 IPO Date Jan 1, 2020 Industry Software Sector Technology Employees 5000 CEO Jane Doe Website example.com Exchange NASDAQ CIK Code 123456 ISIN Number US0000000000</div>
        <h2>Latest SEC Filings</h2><div>Jun 1, 2026 10-Q Quarterly report</div>
        """,
        "transcripts": '<a href="/stocks/exm/transcripts/1/">Earnings Call: Q1 2026</a><p>Revenue grew.</p>',
        "filings": "<div>Q1 2026 - Results - EXM Jun 1, 2026</div>",
        "visible_analyst_ratings": [
            {
                "analyst": {
                    "rank": 1,
                    "name": "Andrew Obin",
                    "company": "BofA Securities",
                    "sector": "Industrials",
                    "success_rate": 0.61,
                    "average_return": 0.13,
                    "url": "https://stockanalysis.com/analysts/andrew-obin/",
                },
                "source_symbol": "EXM",
                "company_name": "Example Corp",
                "rating_action": "Maintained:",
                "rating": "Buy",
                "price_target": "$57 → $68",
                "current_price": "$59.03",
                "upside": "15.2%",
                "updated": "Jun 12, 2026",
                "source_url": "https://stockanalysis.com/stocks/exm/",
            }
        ],
    }

    normalized = normalize_stockanalysis_context("exm", raw)

    assert normalized["identity"]["name"] == "Example Corp. (EXM)"
    assert normalized["valuation"]["market_cap"] == 4_200_000_000
    assert normalized["income_statement"]["revenue"] == 1_100_000_000
    assert normalized["profitability"]["gross_margin"] == 0.545
    assert normalized["analyst"]["target_consensus"] == 55.0
    assert normalized["analyst"]["target_high"] == 70.0
    assert normalized["analyst"]["visible_analyst_ratings"][0]["analyst_name"] == "Andrew Obin"
    assert normalized["analyst"]["visible_analyst_ratings"][0]["updated"] == "Jun 12, 2026"
    assert normalized["dividends"]["dividend_yield"] == 0.025
    assert normalized["transcripts"]["items"][0]["title"] == "Earnings Call: Q1 2026"
    assert normalized["recent_news"][0]["title"] == "Example wins contract"


def test_long_term_sources_include_stockanalysis_provider_context(monkeypatch) -> None:
    def fake_fetch(provider: str, request: LongTermSourceRequest):
        assert provider == "stockanalysis"
        return {
            "symbol": "EXM",
            "source": "stockanalysis",
            "source_base_url": "https://stockanalysis.com",
            "fetched_at": "2026-06-14T10:00:00+00:00",
            "endpoint_status": {"overview": {"status": "ok", "bytes": 100, "tables": 1}},
            "normalized": {
                "identity": {"symbol": "EXM", "name": "Example Corp", "sector": "Technology"},
                "valuation": {"market_cap": 1000, "pe": 20},
                "analyst": {"rating": "Buy", "target_consensus": 30},
                "transcripts": {"items": [{"title": "Q1 2026", "summary": "Margins improved."}]},
                "filings": {"items": [{"title": "Q1 2026 - Results"}]},
                "data_quality": {"source_note": "HTML-derived public StockAnalysis pages; tables and visible sections only."},
            },
            "raw": {"overview": "<html></html>"},
            "raw_sha256": "abc",
        }

    monkeypatch.setattr(long_term_sources, "_fetch_provider_payload", fake_fetch)

    context = collect_long_term_source_context(LongTermSourceRequest(ticker="EXM", providers=("stockanalysis",)))

    assert context["status"] == "ok"
    assert context["consolidated"]["provider_contexts"]["stockanalysis"]["analyst"]["rating"] == "Buy"
    assert context["consolidated"]["provider_contexts"]["stockanalysis"]["transcripts"]["items"][0]["title"] == "Q1 2026"
    assert "stockanalysis" in context["providers_requested"]
    json.dumps(context)
