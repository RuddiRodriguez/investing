from __future__ import annotations

import json
from pathlib import Path

from market_forecasting_engine.trade_republic_dashboard import build_dashboard_state


def test_build_dashboard_state_summarizes_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "summary": {
                    "report_timestamp": "2026-06-02T12:00:00+00:00",
                    "holding_count": 1,
                    "total_open_cost_basis": 100.0,
                    "total_current_value": 112.0,
                    "total_unrealized_pl": 12.0,
                    "total_unrealized_pl_pct": 12.0,
                    "total_historical_buy_cash": 100.0,
                    "total_historical_sell_cash": 0.0,
                    "ticker_resolution_count": 2,
                },
                "holdings": [
                    {
                        "name": "NVIDIA",
                        "isin": "US67066G1040",
                        "ticker": "NVD.DE",
                        "alpaca_ticker": "NVDA",
                        "ticker_resolution_source": "manual_map",
                        "current_quantity": 0.1,
                        "current_price": 195.04,
                        "current_value": 19.5,
                        "open_cost_basis": 18.0,
                        "unrealized_pl": 1.5,
                        "unrealized_pl_pct": 8.3333,
                        "weighted_paid_price": 180.0,
                        "alpaca_weighted_price_at_buy_time": 181.0,
                        "alpaca_paid_vs_market_at_buy_time_pct": -0.55,
                        "alpaca_status": "matched",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    state = build_dashboard_state(report)

    assert state["summary"]["total_current_value"] == 112.0
    assert state["summary"]["total_unrealized_pl_pct"] == 12.0
    assert state["holdings"][0]["name"] == "NVIDIA"
    assert state["holdings"][0]["alpaca_status"] == "matched"


def test_build_dashboard_state_reports_missing_file(tmp_path: Path) -> None:
    state = build_dashboard_state(tmp_path / "missing.json")

    assert state["report_error"].startswith("report_not_found")
    assert state["holdings"] == []
