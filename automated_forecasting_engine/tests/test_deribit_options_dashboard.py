from __future__ import annotations

import json

from market_forecasting_engine.deribit_options_dashboard import build_dashboard_state, build_forecast_chart, summarize_position_pl


def test_deribit_dashboard_reads_report_state_and_history(tmp_path) -> None:
    state_dir = tmp_path / "runs"
    (state_dir / "logs").mkdir(parents=True)
    (state_dir / "state").mkdir(parents=True)
    report = {
        "checked_at": "2026-06-01T10:00:00+00:00",
        "currency": "ETH",
        "forecast_created_at_utc": "2026-06-01T10:00:00+00:00",
        "selected_forecast": {"expected_direction": "Upward", "spot": 2000, "predicted_price": 2100},
        "forecast_plan": {
            "latest_price": 2000,
            "forecasts": [
                {
                    "horizon_hours": 1,
                    "forecast_timestamp": "2026-06-01T11:00:00+00:00",
                    "predicted_price": 2100,
                    "lower_price": 2075,
                    "upper_price": 2125,
                }
            ],
        },
        "account": {"equity": 10, "available_funds": 9},
        "option_trade_plan": {
            "action": "buy_option",
            "option_type": "call",
            "fibonacci_analysis": {
                "enabled": True,
                "status": "ok",
                "confirmation": "supportive",
                "nearest_support": 1980,
                "nearest_resistance": 2020,
                "nearest_target_price": 2100,
            },
            "selected_contract": {
                "instrument_name": "ETH-5JUN26-2100-C",
                "strike": 2100,
                "expiration_utc": "2026-06-05T09:00:00+00:00",
                "hours_to_expiry": 95,
                "required_hours_to_expiry_for_entry": 20,
                "bid": 0.05,
                "ask": 0.055,
                "greeks": {"delta": 0.45, "gamma": 0.006, "theta": -2.0, "vega": 0.5},
                "liquidity": {"volume": 10, "open_interest": 100},
            },
            "order": {"price": 0.054, "amount": 0.1},
            "risk": {"estimated_debit_usd": 10.8, "estimated_debit_base": 0.0054},
            "exit_plan": {"take_profit": {"price": 0.08}, "stop_loss": {"price": 0.035}},
        },
        "open_option_orders": [{"instrument_name": "ETH-5JUN26-2100-C"}],
        "option_positions": [
            {
                "instrument_name": "ETH-5JUN26-2100-C",
                "size": 1,
                "average_price": 0.05,
                "mark_price": 0.06,
                "floating_profit_loss": 0.01,
                "expiration_utc": "2026-06-05T09:00:00+00:00",
            }
        ],
        "execution_blocks": [],
        "order_result": {"submitted": False},
    }
    (state_dir / "ETH_deribit_options_agent_report.json").write_text(json.dumps(report), encoding="utf-8")
    (state_dir / "state" / "ETH_deribit_options_agent_state.json").write_text(json.dumps({"active_trade": None}), encoding="utf-8")
    (state_dir / "logs" / "ETH_20260601.jsonl").write_text(json.dumps(report) + "\n", encoding="utf-8")

    dashboard = build_dashboard_state(state_dir=state_dir, currency="ETH")

    assert dashboard["summary"]["contract"] == "ETH-5JUN26-2100-C"
    assert dashboard["summary"]["estimated_debit_usd"] == 10.8
    assert dashboard["summary"]["open_order_count"] == 1
    assert dashboard["summary"]["position_pl"]["status"] == "winning"
    assert dashboard["summary"]["position_pl"]["total_unrealized_pl_usd"] == 20
    assert dashboard["summary"]["hours_to_expiry"] == 95
    assert dashboard["summary"]["position_pl"]["rows"][0]["expiration_utc"] == "2026-06-05T09:00:00+00:00"
    assert dashboard["summary"]["greeks"]["delta"] == 0.45
    assert dashboard["summary"]["fibonacci"]["confirmation"] == "supportive"
    assert dashboard["summary"]["liquidity"]["open_interest"] == 100
    assert len(dashboard["history"]) == 1
    assert dashboard["forecast_chart"]["actual_count"] == 1
    assert dashboard["forecast_chart"]["forecast_count"] == 1
    assert dashboard["forecast_chart"]["latest_forecast"][0]["predicted_price"] == 2100


def test_summarize_deribit_position_pl_splits_winners_and_losers() -> None:
    payload = summarize_position_pl(
        [
            {"instrument_name": "ETH-3JUN26-2000-P", "size": 1, "average_price": 0.018, "mark_price": 0.02},
            {"instrument_name": "ETH-4JUN26-1950-P", "size": 2, "average_price": 0.01, "mark_price": 0.009},
        ],
        underlying_price_usd=2000.0,
    )

    assert payload["status"] == "flat"
    assert payload["winning_positions"] == 1
    assert payload["losing_positions"] == 1
    assert payload["total_unrealized_pl_base"] == 0
    assert payload["total_cost_usd"] == 76
    assert payload["total_value_usd"] == 76


def test_summarize_usdc_settled_position_pl_does_not_multiply_by_underlying() -> None:
    payload = summarize_position_pl(
        [
            {
                "instrument_name": "ETH_USDC-8JUN26-1650-P",
                "size": 0.3,
                "average_price": 35.2,
                "mark_price": 34.5,
                "floating_profit_loss": -0.21,
            }
        ],
        underlying_price_usd=1665.0,
    )

    assert payload["display_currency"] == "USDC"
    assert payload["total_cost_base"] == 10.56
    assert payload["total_value_base"] == 10.35
    assert payload["total_cost_usd"] == 10.56
    assert payload["total_value_usd"] == 10.35
    assert payload["total_unrealized_pl_usd"] == -0.21
    assert payload["rows"][0]["value_currency"] == "USDC"


def test_build_forecast_chart_uses_latest_forecast_path() -> None:
    first = {
        "checked_at": "2026-06-01T09:00:00+00:00",
        "forecast_created_at_utc": "2026-06-01T09:00:00+00:00",
        "selected_forecast": {"spot": 1980},
        "forecast_plan": {
            "latest_price": 1980,
            "forecasts": [{"horizon_hours": 1, "forecast_timestamp": "2026-06-01T10:00:00+00:00", "predicted_price": 1990}],
        },
    }
    latest = {
        "checked_at": "2026-06-01T10:00:00+00:00",
        "forecast_created_at_utc": "2026-06-01T10:00:00+00:00",
        "selected_forecast": {"spot": 2000},
        "forecast_plan": {
            "latest_price": 2000,
            "forecasts": [
                {"horizon_hours": 1, "forecast_timestamp": "2026-06-01T11:00:00+00:00", "predicted_price": 2010},
                {"horizon_hours": 2, "forecast_timestamp": "2026-06-01T12:00:00+00:00", "predicted_price": 2020},
            ],
        },
    }

    chart = build_forecast_chart(history=[first], latest_report=latest)

    assert chart["actual_count"] == 2
    assert chart["forecast_count"] == 2
    assert chart["as_of"]["price"] == 2000
    assert [point["predicted_price"] for point in chart["latest_forecast"]] == [2010, 2020]


def test_build_forecast_chart_uses_created_at_plus_horizon_for_crypto_options() -> None:
    latest = {
        "checked_at": "2026-06-03T22:25:35+00:00",
        "forecast_created_at_utc": "2026-06-03T22:25:00+00:00",
        "selected_forecast": {"spot": 1831.0},
        "forecast_plan": {
            "latest_price": 1831.0,
            "forecasts": [
                {
                    "horizon_hours": 0.25,
                    "forecast_timestamp": "2026-06-04T00:14:00",
                    "predicted_price": 1815.0,
                }
            ],
        },
    }

    chart = build_forecast_chart(history=[], latest_report=latest)

    assert chart["forecast_count"] == 1
    assert chart["latest_forecast"][0]["target_time"] == "2026-06-03T22:40:00+00:00"
