from __future__ import annotations

import json

import math
import pandas as pd

from market_forecasting_engine.paper_trader_dashboard import (
    actual_at_or_after,
    build_forecast_points,
    discover_ordered_forecasts,
    read_agent_state,
    read_latest_agent_log,
    summarize_decision,
    summarize_forecast,
    _strict_json_value,
)


def test_reads_agent_state_and_latest_log(tmp_path) -> None:
    state_dir = tmp_path / "paper"
    (state_dir / "state").mkdir(parents=True)
    (state_dir / "logs").mkdir(parents=True)
    (state_dir / "state" / "ETH_USD_aggressive.json").write_text(
        json.dumps({"last_forecast": {"created_at_utc": "2026-06-01T08:00:00+00:00"}}),
        encoding="utf-8",
    )
    log_path = state_dir / "logs" / "ETH_USD_aggressive_20260601.jsonl"
    log_path.write_text(
        json.dumps({"checked_at": "2026-06-01T08:00:00+00:00", "decision": {"action": "hold"}})
        + "\n"
        + json.dumps({"checked_at": "2026-06-01T08:01:00+00:00", "decision": {"action": "submit_order"}})
        + "\n",
        encoding="utf-8",
    )

    state, state_path = read_agent_state(state_dir, "ETH-USD", "aggressive")
    latest, latest_path = read_latest_agent_log(state_dir, "ETH-USD", "aggressive")

    assert state_path.name == "ETH_USD_aggressive.json"
    assert state["last_forecast"]["created_at_utc"] == "2026-06-01T08:00:00+00:00"
    assert latest_path == log_path
    assert latest["decision"]["action"] == "submit_order"


def test_forecast_points_mark_matured_against_actual_series() -> None:
    actual = pd.DataFrame(
        {"close": [100.0, 104.0, 105.0]},
        index=pd.to_datetime(["2026-06-01T08:00:00", "2026-06-01T09:01:00", "2026-06-01T09:02:00"]),
    )
    forecast_record = {
        "forecast": {
            "horizon_hours": 1.0,
            "forecast_date": "2026-06-01T09:00:00",
            "spot": 100.0,
            "predicted_price": 103.0,
            "lower_price": 99.0,
            "upper_price": 106.0,
            "expected_direction": "Upward",
            "directional_confidence": 0.62,
        }
    }

    points = build_forecast_points(forecast_record, actual)

    assert points == [
        {
            "horizon_hours": 1.0,
            "timestamp": "2026-06-01T09:00:00",
            "predicted_price": 103.0,
            "lower_price": 99.0,
            "upper_price": 106.0,
            "expected_direction": "Upward",
            "directional_confidence": 0.62,
            "spot": 100.0,
            "source": None,
            "source_path": None,
            "source_as_of": None,
            "matured": True,
            "actual_price": 104.0,
            "error": 1.0,
            "direction_hit": True,
        }
    ]


def test_actual_at_or_after_returns_none_for_pending_forecast() -> None:
    actual = pd.DataFrame({"close": [100.0]}, index=pd.to_datetime(["2026-06-01T08:00:00"]))

    assert actual_at_or_after(actual, "2026-06-01T09:00:00") is None


def test_summarizers_keep_llm_entry_plan_and_order_context() -> None:
    forecast_record = {
        "created_at_utc": "2026-06-01T08:00:00+00:00",
        "horizon_minutes": 15.0,
        "spot_plan": {"as_of": "2026-06-01T08:00:00", "latest_price": 100.0},
        "forecast": {
            "forecast_date": "2026-06-01T08:15:00",
            "spot": 100.0,
            "predicted_price": 101.0,
            "lower_price": 99.5,
            "upper_price": 102.0,
            "expected_direction": "Upward",
            "directional_confidence": 0.6,
        },
        "llm_trader": {
            "status": "executed",
            "decision": {
                "decision": "Hold",
                "confidence": 0.7,
                "entry_plan": {"entry_style": "wait_for_pullback", "buy_near": 98.0},
            },
        },
        "mean_reversion_dip_buy": {"best_setup": {"entry_price": 98.0}},
    }
    decision = {
        "action": "hold",
        "side": "none",
        "decision_source": "llm_forecast_cache",
        "reasons": ["no_directional_edge"],
        "llm_status": "executed",
        "llm_decision": {"decision": "Hold", "confidence": 0.7, "entry_plan": {"buy_near": 98.0}},
        "order_plan": {"entry_order": {"type": "none"}},
    }

    forecast_summary = summarize_forecast(forecast_record)
    decision_summary = summarize_decision(decision)

    assert forecast_summary["llm_decision"] == "Hold"
    assert forecast_summary["entry_plan"]["entry_style"] == "wait_for_pullback"
    assert forecast_summary["dip_buy"]["entry_price"] == 98.0
    assert decision_summary["order_plan"]["entry_order"]["type"] == "none"


def test_strict_json_value_replaces_nan_with_null() -> None:
    payload = {"spot_plan": {"vwap": math.nan}, "rows": [{"value": 1.0}, {"value": math.nan}]}

    cleaned = _strict_json_value(payload)
    encoded = json.dumps(cleaned, allow_nan=False)

    assert '"vwap": null' in encoded
    assert cleaned["rows"][1]["value"] is None


def test_discover_ordered_forecasts_merges_live_and_saved_horizons(tmp_path) -> None:
    report_dir = tmp_path / "ETH_alpaca_48h"
    report_dir.mkdir()
    (report_dir / "daily_trade_report.json").write_text(
        json.dumps(
            {
                "ticker": "ETH-USD",
                "as_of_timestamp": "2026-05-31T17:11:00",
                "current_price": 1997.0,
                "forecasts": [
                    {"horizon_hours": 1.0, "forecast_date": "2026-05-31T18:11:00", "predicted_price": 1999.0},
                    {"horizon_hours": 48.0, "forecast_date": "2026-06-02T17:11:00", "predicted_price": 1940.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    agent = {
        "created_at_utc": "2026-06-01T08:00:00+00:00",
        "spot_plan": {"as_of": "2026-06-01T08:00:00", "latest_price": 1975.0},
        "forecast": {
            "horizon_hours": 0.25,
            "forecast_date": "2026-06-01T08:15:00",
            "predicted_price": 1977.0,
            "spot": 1975.0,
        },
    }

    forecasts = discover_ordered_forecasts(runs_dir=tmp_path, ticker="ETH-USD", agent_forecast_record=agent)

    assert [row["horizon_hours"] for row in forecasts] == [0.25, 1.0, 48.0]
    assert forecasts[0]["source"] == "live_agent"
    assert forecasts[-1]["source_path"].endswith("daily_trade_report.json")
