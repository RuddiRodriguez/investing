from __future__ import annotations

import json

from market_forecasting_engine.paper_options_dashboard import (
    build_dashboard_state,
    _recent_stock_bar_points,
    read_history,
    read_latest_report,
    summarize_position_pl,
)


def test_options_dashboard_reads_ticker_specific_report_and_history(tmp_path) -> None:
    report = {
        "checked_at": "2026-06-01T10:00:00+00:00",
        "ticker": "AAPL",
        "selected_forecast": {"expected_direction": "Upward", "spot": 200.0, "predicted_price": 205.0},
        "forecast_plan": {
            "as_of": "2026-06-01T10:00:00+00:00",
            "latest_price": 200.0,
            "forecasts": [
                {
                    "horizon_hours": 0.25,
                    "forecast_timestamp": "2026-06-01T10:15:00+00:00",
                    "predicted_price": 201.0,
                    "lower_price": 199.0,
                    "upper_price": 203.0,
                },
                {
                    "horizon_hours": 0.5,
                    "forecast_timestamp": "2026-06-01T10:30:00+00:00",
                    "predicted_price": 202.0,
                    "lower_price": 198.5,
                    "upper_price": 204.0,
                },
                {
                    "horizon_hours": 0.75,
                    "forecast_timestamp": "2026-06-01T10:45:00+00:00",
                    "predicted_price": 203.0,
                    "lower_price": 198.0,
                    "upper_price": 205.0,
                },
                {
                    "horizon_hours": 1.0,
                    "forecast_timestamp": "2026-06-01T11:00:00+00:00",
                    "predicted_price": 205.0,
                    "lower_price": 202.0,
                    "upper_price": 208.0,
                },
            ],
        },
        "market_clock": {"is_open": True, "next_open": "2026-06-02T09:30:00-04:00"},
        "option_trade_plan": {
            "action": "buy_option",
            "option_type": "call",
            "selected_contract": {
                "symbol": "AAPL260605C00205000",
                "name": "AAPL Jun 05 2026 205 Call",
                "spread_pct": 0.08,
                "open_interest": 250,
                "greeks": {
                    "delta": 0.41,
                    "gamma": 0.012,
                    "theta": -0.20,
                    "vega": 0.34,
                    "theta_decay_usd_for_horizon": 1.67,
                    "theta_premium_pct_per_day": 0.095,
                    "forecast_edge_usd_delta_adjusted": 205.0,
                },
            },
            "order": {"type": "limit", "limit_price": 2.1},
            "exit_plan": {
                "take_profit": {"limit_price": 3.2},
                "stop_loss": {"stop_price": 1.4, "limit_price": 1.3},
            },
            "risk": {"estimated_debit": 210.0},
        },
        "execution_blocks": [],
        "order_result": {"submitted": False},
        "option_positions": [
            {
                "symbol": "AAPL260605C00205000",
                "qty": "2",
                "avg_entry_price": "2.10",
                "current_price": "2.50",
                "cost_basis": "420",
                "market_value": "500",
                "unrealized_pl": "80",
            }
        ],
    }
    (tmp_path / "AAPL_options_agent_report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "AAPL_20260601.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    payload = build_dashboard_state(state_dir=tmp_path, ticker="AAPL", include_live_bars=False)

    assert payload["ticker"] == "AAPL"
    assert payload["summary"]["contract"] == "AAPL260605C00205000"
    assert payload["summary"]["option_type_label"] == "CALL"
    assert payload["summary"]["take_profit"] == 3.2
    assert payload["summary"]["stop_limit"] == 1.3
    assert payload["summary"]["greeks"]["delta"] == 0.41
    assert payload["summary"]["spread_pct"] == 0.08
    assert payload["summary"]["open_interest"] == 250
    assert payload["summary"]["position_pl"]["status"] == "winning"
    assert payload["summary"]["position_pl"]["total_unrealized_pl"] == 80
    assert payload["history"][0]["ticker"] == "AAPL"
    assert payload["chart"]["as_of"] == "2026-06-01T10:00:00+00:00"
    assert payload["chart"]["as_of_price"] == 200.0
    assert payload["chart"]["actual_points"] == [{"timestamp": "2026-06-01T10:00:00+00:00", "price": 200.0}]
    assert payload["chart"]["latest_actual"] == {"timestamp": "2026-06-01T10:00:00+00:00", "price": 200.0}
    assert payload["chart"]["source"] == "agent_history"
    assert [row["horizon_hours"] for row in payload["chart"]["forecast_points"]] == [0.25, 0.5, 0.75, 1.0]
    assert payload["chart"]["forecast_points"][-1]["predicted_price"] == 205.0


def test_options_dashboard_missing_ticker_is_empty_not_tsla_specific(tmp_path) -> None:
    report, path = read_latest_report(tmp_path, "MSFT")
    history, log_path = read_history(tmp_path, "MSFT", max_history=10)

    assert report == {}
    assert path is None
    assert history == []
    assert log_path is None


def test_recent_stock_bar_points_falls_back_to_iex(monkeypatch) -> None:
    class FakeBroker:
        def stock_bars(self, symbol, *, feed=None, **kwargs):
            if feed is None:
                raise RuntimeError("SIP forbidden")
            assert feed == "iex"
            return [{"t": "2026-06-04T15:00:00Z", "c": 420.25}]

    monkeypatch.setattr("market_forecasting_engine.paper_options_dashboard.AlpacaPaperBroker", FakeBroker)

    points = _recent_stock_bar_points("TSLA")
    assert points[0]["timestamp"] == "2026-06-04T15:00:00Z"
    assert points[0]["price"] == 420.25
    assert points[0]["close"] == 420.25


def test_summarize_position_pl_calculates_plain_english_status() -> None:
    payload = summarize_position_pl(
        [
            {
                "symbol": "TSLA260603P00420000",
                "qty": "2",
                "avg_entry_price": "3.65",
                "current_price": "4.20",
            },
            {
                "symbol": "TSLA260605P00417500",
                "qty": "1",
                "avg_entry_price": "5.50",
                "current_price": "5.30",
            },
        ]
    )

    assert payload["status"] == "winning"
    assert payload["total_cost"] == 1280
    assert payload["total_value"] == 1370
    assert payload["total_unrealized_pl"] == 90
