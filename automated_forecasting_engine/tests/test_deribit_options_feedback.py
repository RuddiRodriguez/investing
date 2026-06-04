from __future__ import annotations

import json
from datetime import UTC, datetime

from market_forecasting_engine.deribit_options_feedback import append_decision_to_ledger, update_feedback_loop


def test_feedback_loop_matures_forecast_horizons_and_blocks_weak_accuracy(tmp_path) -> None:
    output_dir = tmp_path / "runs"
    record = {
        "checked_at": "2026-06-03T10:00:00+00:00",
        "venue": "deribit_testnet",
        "currency": "ETH",
        "forecast_created_at_utc": "2026-06-03T10:00:00+00:00",
        "selected_forecast": {"spot": 100.0, "expected_direction": "Upward"},
        "forecast_plan": {
            "latest_price": 100.0,
            "forecasts": [
                {"horizon_hours": 1, "forecast_timestamp": "2026-06-03T11:00:00+00:00", "predicted_price": 110.0},
                {"horizon_hours": 2, "forecast_timestamp": "2026-06-03T12:00:00+00:00", "predicted_price": 112.0},
            ],
        },
        "option_trade_plan": {"action": "buy_option", "option_type": "call", "selected_contract": {"instrument_name": "ETH-TEST-C"}},
        "order_result": {"submitted": True, "order": {"trades": [{"profit_loss": -0.01, "fee": 0.001}]}},
    }

    append_result = append_decision_to_ledger(output_dir=output_dir, currency="ETH", record=record)
    feedback = update_feedback_loop(
        output_dir=output_dir,
        currency="ETH",
        now=datetime(2026, 6, 3, 12, 30, tzinfo=UTC),
        actual_price=95.0,
        min_matured=1,
        min_direction_accuracy=0.5,
        max_abs_pct_error=0.50,
        window=10,
    )

    assert append_result["horizon_count"] == 2
    assert feedback["metrics"]["matured_horizon_count"] == 2
    assert feedback["metrics"]["direction_accuracy"] == 0
    assert "feedback_direction_accuracy_below_min" in feedback["blocks"]
    rows = [json.loads(line) for line in (output_dir / "feedback" / "ETH_decision_ledger.jsonl").read_text().splitlines()]
    assert rows[0]["horizons"][0]["matured"] is True
    assert rows[0]["horizons"][0]["actual_price"] == 95.0


def test_feedback_loop_uses_created_at_plus_horizon_for_short_horizon_targets(tmp_path) -> None:
    output_dir = tmp_path / "runs"
    record = {
        "checked_at": "2026-06-03T22:25:35+00:00",
        "venue": "deribit_testnet",
        "currency": "ETH",
        "forecast_created_at_utc": "2026-06-03T22:25:00+00:00",
        "selected_forecast": {"spot": 1831.0, "expected_direction": "Downward"},
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
        "option_trade_plan": {"action": "buy_option", "option_type": "put", "selected_contract": {"instrument_name": "ETH-TEST-P"}},
        "order_result": {"submitted": True, "order": {"trades": []}},
    }

    append_decision_to_ledger(output_dir=output_dir, currency="ETH", record=record)
    ledger_path = output_dir / "feedback" / "ETH_decision_ledger.jsonl"
    before = [json.loads(line) for line in ledger_path.read_text().splitlines()]

    assert before[0]["horizons"][0]["target_time"] == "2026-06-03T22:40:00+00:00"
    assert before[0]["horizons"][0]["payload_target_time"] == "2026-06-04T00:14:00+00:00"
    assert before[0]["horizons"][0]["target_time_source"] == "forecast_created_at_plus_horizon_hours"

    too_early = update_feedback_loop(
        output_dir=output_dir,
        currency="ETH",
        now=datetime(2026, 6, 3, 22, 39, tzinfo=UTC),
        actual_price=1820.0,
    )
    assert too_early["metrics"]["matured_horizon_count"] == 0

    matured = update_feedback_loop(
        output_dir=output_dir,
        currency="ETH",
        now=datetime(2026, 6, 3, 22, 41, tzinfo=UTC),
        actual_price=1818.0,
    )
    assert matured["metrics"]["matured_horizon_count"] == 1
    assert matured["metrics"]["direction_accuracy"] == 1
