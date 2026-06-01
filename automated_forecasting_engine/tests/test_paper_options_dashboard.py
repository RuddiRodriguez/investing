from __future__ import annotations

import json

from market_forecasting_engine.paper_options_dashboard import build_dashboard_state, read_history, read_latest_report


def test_options_dashboard_reads_ticker_specific_report_and_history(tmp_path) -> None:
    report = {
        "checked_at": "2026-06-01T10:00:00+00:00",
        "ticker": "AAPL",
        "selected_forecast": {"expected_direction": "Upward", "spot": 200.0, "predicted_price": 205.0},
        "market_clock": {"is_open": True, "next_open": "2026-06-02T09:30:00-04:00"},
        "option_trade_plan": {
            "action": "buy_option",
            "option_type": "call",
            "selected_contract": {"symbol": "AAPL260605C00205000", "name": "AAPL Jun 05 2026 205 Call"},
            "order": {"type": "limit", "limit_price": 2.1},
            "exit_plan": {
                "take_profit": {"limit_price": 3.2},
                "stop_loss": {"stop_price": 1.4, "limit_price": 1.3},
            },
            "risk": {"estimated_debit": 210.0},
        },
        "execution_blocks": [],
        "order_result": {"submitted": False},
    }
    (tmp_path / "AAPL_options_agent_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "AAPL_20260601.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    payload = build_dashboard_state(state_dir=tmp_path, ticker="AAPL")

    assert payload["ticker"] == "AAPL"
    assert payload["summary"]["contract"] == "AAPL260605C00205000"
    assert payload["summary"]["take_profit"] == 3.2
    assert payload["summary"]["stop_limit"] == 1.3
    assert payload["history"][0]["ticker"] == "AAPL"


def test_options_dashboard_missing_ticker_is_empty_not_tsla_specific(tmp_path) -> None:
    report, path = read_latest_report(tmp_path, "MSFT")
    history, log_path = read_history(tmp_path, "MSFT", max_history=10)

    assert report == {}
    assert path is None
    assert history == []
    assert log_path is None
