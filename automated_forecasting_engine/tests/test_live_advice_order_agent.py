from __future__ import annotations

from datetime import UTC, datetime

from market_forecasting_engine.live_trading.stocks.advice_order_agent import (
    advice_entry,
    expired_orders_by_symbol,
    order_payload,
)


def test_advice_entry_builds_buy_lower_limit_from_hold_advice() -> None:
    entry = advice_entry(
        {
            "forecast": {"current_price": 218.75},
            "advice": {
                "final_advice": {
                    "action_now": "hold",
                    "buy_lower_price": 214.5,
                    "buy_above_breakout_price": 224.5,
                }
            },
        },
        max_breakout_chase_pct=0.003,
    )

    assert entry["style"] == "buy_lower_resting_limit"
    assert entry["limit_price"] == 214.5
    assert entry["blocks"] == []


def test_advice_entry_blocks_breakout_chase_when_price_too_far_above_trigger() -> None:
    entry = advice_entry(
        {
            "forecast": {"current_price": 230.0},
            "advice": {"final_advice": {"action_now": "hold", "buy_above_breakout_price": 224.5}},
        },
        max_breakout_chase_pct=0.003,
    )

    assert entry["style"] == "buy_breakout"
    assert "breakout_too_far_above_advice_price" in entry["blocks"]


def test_order_payload_is_day_limit_buy_only() -> None:
    payload, blocks = order_payload(
        ticker="WM",
        entry={"style": "buy_lower_resting_limit", "limit_price": 214.5},
        notional=4.0,
        min_notional=1.0,
        prefix="liveadvice",
        now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
    )

    assert blocks == []
    assert payload == {
        "symbol": "WM",
        "side": "buy",
        "type": "limit",
        "qty": "0.018648",
        "limit_price": "214.5",
        "time_in_force": "day",
        "client_order_id": "liveadvice_WM_buy_lower_re_20260624120000",
    }


def test_order_payload_blocks_when_notional_too_small() -> None:
    payload, blocks = order_payload(
        ticker="ASML",
        entry={"style": "buy_lower_resting_limit", "limit_price": 1765.0},
        notional=0.5,
        min_notional=1.0,
        prefix="liveadvice",
        now=datetime(2026, 6, 24, 12, 0, tzinfo=UTC),
    )

    assert payload is None
    assert blocks == ["insufficient_buying_power_for_min_notional"]


def test_expired_orders_by_symbol_only_uses_watched_recent_orders() -> None:
    now = datetime(2026, 6, 24, 12, 0, tzinfo=UTC)
    result = expired_orders_by_symbol(
        [
            {
                "symbol": "WM",
                "status": "expired",
                "client_order_id": "liveadvice_WM_buy_lower_r_20260623150000",
                "expired_at": "2026-06-23T20:00:00Z",
            },
            {
                "symbol": "MMM",
                "status": "expired",
                "client_order_id": "manual_order",
                "expired_at": "2026-06-23T20:00:00Z",
            },
            {
                "symbol": "ASML",
                "status": "filled",
                "client_order_id": "liveadvice_ASML_buy_lower_r_20260623150000",
                "expired_at": None,
            },
        ],
        tickers=["ASML", "MMM", "WM"],
        lookback_hours=30,
        now=now,
    )

    assert [order["symbol"] for order in result["WM"]] == ["WM"]
    assert result["MMM"] == []
    assert result["ASML"] == []
