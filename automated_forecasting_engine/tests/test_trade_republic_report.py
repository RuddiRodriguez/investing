from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from market_forecasting_engine.trade_republic_report import build_report, ticker_resolution_items


def test_build_report_combines_portfolio_transactions_and_historical_prices(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text(
        "Name;ISIN;quantity;price;avgCost;netValue\n"
        "Tesla;US88160R1014;2;260;200;520\n"
        "Nvidia;US67066G1040;1;120;100;120\n",
        encoding="utf-8",
    )
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text(
        "Date;Type;Value;Note;ISIN;Shares;Fees;Taxes;ISIN 2;Shares 2\n"
        "2026-01-02T10:00:00;Buy;-400;Tesla;US88160R1014;2;1;0;;\n"
        "2026-01-03T10:00:00;Buy;-100;Nvidia;US67066G1040;1;1;0;;\n",
        encoding="utf-8",
    )
    isin_map = tmp_path / "isin_map.csv"
    isin_map.write_text("isin,ticker,name\nUS88160R1014,TSLA,Tesla\nUS67066G1040,NVDA,Nvidia\n", encoding="utf-8")
    prices = tmp_path / "prices.csv"
    prices.write_text("ticker,date,close\nTSLA,2026-01-02,390\nNVDA,2026-01-03,99\n", encoding="utf-8")

    report = build_report(
        portfolio_path=portfolio,
        transactions_path=transactions,
        isin_map_path=isin_map,
        price_history_path=prices,
    )

    assert report["summary"]["total_open_cost_basis"] == 500.0
    assert report["summary"]["total_current_value"] == 640.0
    rows = {row["isin"]: row for row in report["holdings"]}
    assert rows["US88160R1014"]["ticker"] == "TSLA"
    assert rows["US88160R1014"]["unrealized_pl"] == 120.0
    assert rows["US88160R1014"]["weighted_paid_price"] == 200.0
    assert rows["US88160R1014"]["weighted_market_price_at_buy"] == 390.0
    assert rows["US88160R1014"]["historical_price_status"] == "matched"


def test_build_report_marks_missing_historical_prices(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text("Name;ISIN;quantity;price;avgCost;netValue\nTesla;US88160R1014;1;260;200;260\n", encoding="utf-8")
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text("Date;Type;Value;Note;ISIN;Shares\n2026-01-02T10:00:00;Buy;-200;Tesla;US88160R1014;1\n", encoding="utf-8")

    report = build_report(portfolio_path=portfolio, transactions_path=transactions)

    assert report["holdings"][0]["historical_price_status"] == "missing_price_history"


def test_build_report_can_fetch_yahoo_prices(tmp_path: Path, monkeypatch) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text("Name;ISIN;quantity;price;avgCost;netValue\nTesla;US88160R1014;1;260;200;260\n", encoding="utf-8")
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text("Date;Type;Value;Note;ISIN;Shares\n2026-01-02T10:00:00;Buy;-200;Tesla;US88160R1014;1\n", encoding="utf-8")
    isin_map = tmp_path / "isin_map.csv"
    isin_map.write_text("isin,ticker,name\nUS88160R1014,TSLA,Tesla\n", encoding="utf-8")

    def fake_download(*args, **kwargs):
        return pd.DataFrame({"Close": [390.0]}, index=pd.to_datetime(["2026-01-02"]))

    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(download=fake_download))

    report = build_report(
        portfolio_path=portfolio,
        transactions_path=transactions,
        isin_map_path=isin_map,
        fetch_yahoo=True,
    )

    assert report["holdings"][0]["weighted_market_price_at_buy"] == 390.0
    assert report["holdings"][0]["historical_price_status"] == "matched"


def test_build_report_can_fetch_alpaca_minute_price(tmp_path: Path, monkeypatch) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text("Name;ISIN;quantity;price;avgCost;netValue\nTesla;US88160R1014;1;260;200;260\n", encoding="utf-8")
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text("Date;Type;Value;Note;ISIN;Shares\n2026-01-02T10:00:00;Buy;-200;Tesla;US88160R1014;1\n", encoding="utf-8")
    isin_map = tmp_path / "isin_map.csv"
    isin_map.write_text("isin,ticker,alpaca_ticker,name\nUS88160R1014,TSLA,TSLA,Tesla\n", encoding="utf-8")

    class FakeAlpacaBroker:
        def stock_bars(self, symbol, *, start, end, timeframe="1Min", limit=1000):
            assert symbol == "TSLA"
            return [
                {"t": "2026-01-02T09:59:00Z", "c": 198.0},
                {"t": "2026-01-02T10:00:00Z", "c": 199.0},
            ]

    import market_forecasting_engine.alpaca_broker as alpaca_broker

    monkeypatch.setattr(alpaca_broker, "AlpacaPaperBroker", FakeAlpacaBroker)

    report = build_report(
        portfolio_path=portfolio,
        transactions_path=transactions,
        isin_map_path=isin_map,
        fetch_alpaca=True,
    )

    row = report["holdings"][0]
    assert row["alpaca_status"] == "matched"
    assert row["alpaca_weighted_price_at_buy_time"] == 199.0
    assert row["alpaca_paid_vs_market_at_buy_time"] == 1.0


def test_ticker_resolution_items_use_company_name_when_ticker_missing() -> None:
    items = ticker_resolution_items(
        [{"Name": "Eli Lilly & Co", "ISIN": "US5324571083"}],
        [],
        {},
    )

    assert items == [
        {
            "isin": "US5324571083",
            "name": "Eli Lilly & Co",
            "existing_yahoo_ticker": "",
            "existing_alpaca_ticker": "",
            "needs_yahoo": "true",
            "needs_alpaca": "true",
        }
    ]


def test_build_report_resolves_missing_ticker_with_name_based_resolver(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text("Name;ISIN;quantity;price;avgCost;netValue\nEli Lilly & Co;US5324571083;1;923;900;923\n", encoding="utf-8")
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text("Date;Type;Value;Note;ISIN;Shares\n2026-01-02T10:00:00;Buy;-900;Eli Lilly & Co;US5324571083;1\n", encoding="utf-8")

    def fake_resolver(items, *, model=None):
        assert items[0]["name"] == "Eli Lilly & Co"
        return [
            {
                "isin": "US5324571083",
                "name": "Eli Lilly & Co",
                "yahoo_ticker": "LLY",
                "alpaca_ticker": "LLY",
                "confidence": 0.98,
                "reason": "US company name resolves to NYSE LLY.",
            }
        ]

    report = build_report(
        portfolio_path=portfolio,
        transactions_path=transactions,
        resolve_tickers_llm=True,
        ticker_resolver=fake_resolver,
    )

    row = report["holdings"][0]
    assert row["ticker"] == "LLY"
    assert row["alpaca_ticker"] == "LLY"
    assert row["ticker_resolution_source"] == "llm_name_resolution"
    assert report["ticker_resolution"][0]["reason"] == "US company name resolves to NYSE LLY."


def test_build_report_can_persist_llm_resolved_tickers(tmp_path: Path) -> None:
    portfolio = tmp_path / "portfolio.csv"
    portfolio.write_text("Name;ISIN;quantity;price;avgCost;netValue\nEli Lilly & Co;US5324571083;1;923;900;923\n", encoding="utf-8")
    transactions = tmp_path / "account_transactions.csv"
    transactions.write_text("Date;Type;Value;Note;ISIN;Shares\n2026-01-02T10:00:00;Buy;-900;Eli Lilly & Co;US5324571083;1\n", encoding="utf-8")
    isin_map = tmp_path / "isin_map.csv"
    isin_map.write_text("isin,ticker,alpaca_ticker,name,source,confidence\n", encoding="utf-8")

    def fake_resolver(items, *, model=None):
        return [
            {
                "isin": "US5324571083",
                "name": "Eli Lilly & Co",
                "yahoo_ticker": "LLY",
                "alpaca_ticker": "LLY",
                "confidence": 0.98,
                "reason": "US company name resolves to NYSE LLY.",
            }
        ]

    build_report(
        portfolio_path=portfolio,
        transactions_path=transactions,
        isin_map_path=isin_map,
        resolve_tickers_llm=True,
        ticker_resolver=fake_resolver,
        update_isin_map=True,
    )

    saved = isin_map.read_text(encoding="utf-8")
    assert "US5324571083,LLY,LLY,Eli Lilly & Co,llm_name_resolution,0.98" in saved
