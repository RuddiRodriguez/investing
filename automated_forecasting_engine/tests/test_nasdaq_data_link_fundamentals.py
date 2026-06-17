from __future__ import annotations

import json

import pytest

from market_forecasting_engine import nasdaq_data_link_fundamentals
from market_forecasting_engine.nasdaq_data_link_fundamentals import (
    NasdaqDataLinkFundamentalsClient,
    normalize_nasdaq_data_link_context,
)


def _table(columns: list[str], rows: list[list[object]]) -> dict[str, object]:
    return {"datatable": {"columns": [{"name": name, "type": "text"} for name in columns], "data": rows}, "meta": {"next_cursor_id": None}}


def test_normalize_nasdaq_data_link_context_compacts_zacks_sections() -> None:
    raw = {
        "zacks_master": _table(
            ["ticker", "comp_name_2", "exchange", "currency_code", "comp_url", "zacks_x_sector_desc", "comp_cik"],
            [["AAPL", "Apple Inc.", "NASDAQ", "USD", "https://www.apple.com", "Computer and Technology", "0000320193"]],
        ),
        "zacks_fundamentals": _table(
            ["ticker", "per_end_date", "per_type", "tot_revnu", "gross_profit", "oper_income", "diluted_net_eps", "tot_asset"],
            [
                ["AAPL", "2018-09-30", "A", 265595.0, 101839.0, 70898.0, 2.9775, 365725.0],
                ["AAPL", "2018-12-31", "Q", 84310.0, 32031.0, 23346.0, 1.045, 373719.0],
            ],
        ),
        "zacks_ratios": _table(
            ["ticker", "per_end_date", "per_type", "curr_ratio", "gross_margin", "profit_margin", "ret_equity"],
            [["AAPL", "2018-12-31", "Q", 1.3006, 38.3, 22.4, 55.5]],
        ),
        "zacks_market_value": _table(
            ["ticker", "per_end_date", "per_type", "mkt_val", "ep_val"],
            [["AAPL", "2018-12-31", "Q", 743788.25, 750350.25]],
        ),
        "zacks_eps_estimates": _table(
            ["ticker", "per_end_date", "per_type", "eps_mean_est", "eps_high_est", "eps_cnt_est"],
            [["AAPL", "2026-09-30", "A", 8.7454, 8.91, 13]],
        ),
        "zacks_eps_surprises": _table(
            ["ticker", "per_end_date", "per_type", "act_rpt_date", "eps_mean_est", "eps_act", "eps_pct_diff_surp"],
            [["AAPL", "2018-03-31", "Q", "2018-05-01", 0.67, 0.68, 1.49]],
        ),
    }

    normalized = normalize_nasdaq_data_link_context("AAPL", raw)

    assert normalized["identity"]["name"] == "Apple Inc."
    assert normalized["latest_annual_fundamentals"]["revenue"] == 265595.0
    assert normalized["latest_quarter_fundamentals"]["eps_diluted"] == 1.045
    assert normalized["latest_quarter_ratios"]["current_ratio"] == 1.3006
    assert normalized["market_value"]["enterprise_value"] == 750350.25
    assert normalized["eps_estimates"]["annual"]["eps_mean_estimate"] == 8.7454
    assert normalized["latest_eps_surprise"]["eps_surprise_pct"] == 1.49


def test_client_fetch_company_context_records_table_status(monkeypatch: pytest.MonkeyPatch) -> None:
    table_payload = _table(["ticker"], [["AAPL"]])

    def fake_get_json(url: str, *, timeout: float = 30.0):
        if "api_key=test-key" not in url:
            raise AssertionError(url)
        return table_payload

    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "test-key")
    monkeypatch.setattr(nasdaq_data_link_fundamentals, "_get_json", fake_get_json)

    result = NasdaqDataLinkFundamentalsClient().fetch_company_context("aapl")

    assert result["symbol"] == "AAPL"
    assert result["endpoint_status"]["zacks_master"]["items"] == 1
    assert result["normalized"]["coverage"]["zacks_master_rows"] == 1
    assert result["api_key_fingerprint"].startswith("sha256:")
    json.dumps(result)
