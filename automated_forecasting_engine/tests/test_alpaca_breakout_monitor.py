from __future__ import annotations

from argparse import Namespace

from market_forecasting_engine.live_trading.alpaca_breakout_monitor import (
    build_profit_target_payload,
    build_protective_order_payload,
    evaluate_breakout_confirmation,
    is_profit_target_order,
    manage_existing_position,
    simple_stop_from_rejected_oco,
    size_entry,
)
from datetime import UTC, datetime


def test_breakout_requires_hold_and_volume_pace() -> None:
    opening = [{"c": 293.0, "v": 1000, "t": f"open-{idx}"} for idx in range(60)]
    recent = [{"c": 294.8, "v": 1300, "t": f"recent-{idx}"} for idx in range(10)]
    result = evaluate_breakout_confirmation(
        bars=opening + recent,
        trigger_price=294.5,
        hold_bars=3,
        volume_window_bars=10,
        opening_hour_bars=60,
        volume_pace_multiplier=1.05,
    )
    assert result["confirmed"] is True
    assert result["hold_confirmed"] is True
    assert result["volume_confirmed"] is True


def test_breakout_blocks_when_volume_pace_is_weaker_than_opening_hour() -> None:
    opening = [{"c": 293.0, "v": 1000, "t": f"open-{idx}"} for idx in range(60)]
    recent = [{"c": 294.8, "v": 900, "t": f"recent-{idx}"} for idx in range(10)]
    result = evaluate_breakout_confirmation(
        bars=opening + recent,
        trigger_price=294.5,
        hold_bars=3,
        volume_window_bars=10,
        opening_hour_bars=60,
        volume_pace_multiplier=1.05,
    )
    assert result["confirmed"] is False
    assert result["hold_confirmed"] is True
    assert result["volume_confirmed"] is False


def test_size_entry_uses_fractional_qty_from_live_buying_power() -> None:
    result = size_entry(
        buying_power=4.63,
        latest_price=295.0,
        max_notional=None,
        buying_power_fraction=0.95,
        min_notional=1.0,
    )
    assert result["planned_notional"] == 4.4
    assert result["qty"] == 0.01491
    assert result["blocked"] is False


def test_existing_position_exit_uses_stop_invalidation_and_targets() -> None:
    args = Namespace(
        symbol="IWM",
        stop_price=286.0,
        invalidation_price=292.8,
        target1_price=305.0,
        target2_price=312.0,
        exit_limit_offset_pct=0.003,
    )
    state = {}
    action, reason, payload = manage_existing_position(args=args, position_qty=0.02, latest_price=292.5, state=state)
    assert action == "sell_invalidation"
    assert reason == "rejected_back_below_invalidation"
    assert payload is not None
    assert payload["side"] == "sell"


def test_existing_position_protection_payload_is_oco_sell() -> None:
    args = Namespace(
        symbol="IWM",
        protective_order_style="oco",
        target1_price=305.0,
        invalidation_price=292.8,
        stop_limit_offset_pct=0.003,
    )
    payload = build_protective_order_payload(args=args, position_qty=0.01491, now=datetime(2026, 6, 12, tzinfo=UTC))
    assert payload is not None
    assert payload["side"] == "sell"
    assert payload["order_class"] == "oco"
    assert payload["qty"] == "0.01491"
    assert payload["take_profit"]["limit_price"] == "305.0"
    assert payload["stop_loss"]["stop_price"] == "292.8"
    assert payload["stop_loss"]["limit_price"] == "291.92"


def test_fractional_oco_fallback_builds_simple_stop_limit() -> None:
    fallback = simple_stop_from_rejected_oco(
        {
            "symbol": "IWM",
            "side": "sell",
            "qty": "0.01491",
            "client_order_id": "iwm-breakout-oco-IWM-20260612150846",
            "stop_loss": {"stop_price": "292.8", "limit_price": "291.92"},
        }
    )
    assert fallback == {
        "symbol": "IWM",
        "side": "sell",
        "type": "stop_limit",
        "qty": "0.01491",
        "time_in_force": "day",
        "stop_price": "292.8",
        "limit_price": "291.92",
        "client_order_id": "iwm-breakout-oco-IWM-20260612150846-stp",
    }


def test_profit_target_payload_sells_full_fractional_position() -> None:
    args = Namespace(symbol="IWM")
    payload = build_profit_target_payload(
        args=args,
        position_qty=0.01491,
        now=datetime(2026, 6, 12, tzinfo=UTC),
        target_price=305.0,
    )
    assert payload["symbol"] == "IWM"
    assert payload["side"] == "sell"
    assert payload["type"] == "limit"
    assert payload["qty"] == "0.01491"
    assert payload["limit_price"] == "305.0"
    assert payload["time_in_force"] == "day"


def test_profit_target_order_classifier_separates_stop_from_target() -> None:
    stop_order = {"type": "stop_limit", "side": "sell", "limit_price": "291.92", "stop_price": "292.8"}
    target_order = {"type": "limit", "side": "sell", "limit_price": "305.0", "stop_price": None}
    assert is_profit_target_order(stop_order, target_price=305.0) is False
    assert is_profit_target_order(target_order, target_price=305.0) is True
