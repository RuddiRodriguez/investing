from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent import (
    decide_from_cached_ceo,
    load_forecast_decision,
    protection_orders_from_advice,
)


def _args(**overrides):
    values = {
        "instrument": "ETH_USDC",
        "ticker": "ETH-USDC",
        "base_currency": "ETH",
        "quote_currency": "USDC",
        "active_timezone": "Europe/Amsterdam",
        "max_notional_usdc": 100.0,
        "max_base_position": 0.25,
        "min_order_base_amount": 0.0001,
        "max_spread_pct": 0.003,
        "entry_price_tolerance_pct": 0.0015,
        "stop_protection_coverage_ratio": 0.95,
        "inventory_scope": "codex_only",
        "managed_base_balance": None,
        "price_tick_size": 0.01,
        "base_amount_step": 0.000001,
        "replace_protection": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_daily_agent_requires_llm_final_decision(tmp_path: Path) -> None:
    report_path = tmp_path / "forecast_report.json"
    report_path.write_text(json.dumps({"ticker": "ETH-USDC", "suggested_action": "Hold"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mandatory llm_final_decision"):
        load_forecast_decision(report_path)


def test_daily_agent_uses_llm_final_decision_buy_lower_trigger(tmp_path: Path) -> None:
    report_path = tmp_path / "forecast_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ticker": "ETH-USDC",
                "current_price": 1900,
                "risk_level": "Medium",
                "llm_final_decision": {
                    "decision": "Hold",
                    "confidence": 0.8,
                    "final_advice": {
                        "action_now": "hold",
                        "buy_lower_price": 1800.0,
                        "stop_loss_price": 1740.0,
                        "take_profit_price": 1940.0,
                    },
                },
                "forecasts": [],
            }
        ),
        encoding="utf-8",
    )
    forecast = load_forecast_decision(report_path)
    market = {
        "latest_price": 1801.0,
        "spread_pct": 0.0005,
        "quote": {"bid": 1800.5, "ask": 1801.5, "mid": 1801.0},
        "instrument_details": {"tick_size": 0.01, "contract_size": 0.000001, "min_trade_amount": 0.0001},
        "account": {"ETH": {"balance": 0.0}, "USDC": {"available_funds": 1000.0}},
        "open_orders": [],
    }

    plan = decide_from_cached_ceo(args=_args(), state={}, forecast_record=forecast, market_packet=market)

    assert plan["action"] == "buy_spot"
    assert plan["reason"] == "cached_ceo_buy_lower_trigger_reached"
    assert plan["execution_allowed"] is True
    assert plan["entry_order"]["type"] == "limit"
    assert plan["post_fill_protection_plan"]["stop_loss"]["type"] == "stop_limit"


def test_daily_agent_stop_protection_is_stop_limit() -> None:
    args = _args()

    orders = protection_orders_from_advice(
        args=args,
        amount=0.02,
        entry_reference=1800.0,
        advice={"stop_loss_price": 1740.0, "take_profit_price": 1940.0},
    )

    assert orders["take_profit"] == {"side": "sell", "type": "limit", "amount": 0.02, "price": 1940.0}
    assert orders["stop_loss"]["type"] == "stop_limit"
    assert orders["stop_loss"]["trigger_price"] == 1740.0
    assert orders["stop_loss"]["price"] < orders["stop_loss"]["trigger_price"]
