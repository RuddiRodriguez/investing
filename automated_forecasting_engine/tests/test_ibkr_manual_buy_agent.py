from __future__ import annotations

from argparse import Namespace

import pytest

from market_forecasting_engine.live_trading.ibkr_manual_buy_agent import (
    execution_blocks,
    matching_open_order,
    parse_plain_english_plan,
    run_cycle,
    validate_plan,
)


def test_parse_plain_english_buy_plan() -> None:
    plan = parse_plain_english_plan("Use paper account. Buy VWCE if price is 120.50 or lower. Use 500 EUR. If order expires resubmit.")

    assert plan == {
        "account_mode": "paper",
        "orders": [
            {
                "symbol": "VWCE",
                "exchange": "SMART",
                "currency": "EUR",
                "asset_type": "ETF",
                "side": "BUY",
                "amount_cash": 500.0,
                "condition": {"type": "price_at_or_below", "price": 120.5},
                "order_type": "LMT",
                "limit_price": 120.5,
                "time_in_force": "DAY",
                "resubmit_if_expired": True,
            }
        ],
    }


def test_validate_plan_blocks_sell_and_market_orders() -> None:
    with pytest.raises(ValueError, match="buy-only"):
        validate_plan({"account_mode": "paper", "orders": [{"symbol": "VWCE", "side": "SELL", "order_type": "LMT", "amount_cash": 100, "condition": {"type": "price_at_or_below", "price": 100}, "limit_price": 100}]})

    with pytest.raises(ValueError, match="Market orders are blocked"):
        validate_plan({"account_mode": "paper", "orders": [{"symbol": "VWCE", "side": "BUY", "order_type": "MKT", "amount_cash": 100, "condition": {"type": "price_at_or_below", "price": 100}, "limit_price": 100}]})


def test_live_mode_requires_extra_confirmation() -> None:
    args = Namespace(execute_orders=True, confirm_risk=True, confirm_live_risk=False)

    assert execution_blocks(args=args, account_mode="live") == ["live_requires_execute_orders_and_confirm_live_risk"]


def test_matching_open_order_finds_duplicate_buy_limit() -> None:
    duplicate = matching_open_order(
        [
            {"contract": {"symbol": "VWCE"}, "order": {"action": "BUY", "orderType": "LMT"}},
            {"contract": {"symbol": "IWDA"}, "order": {"action": "BUY", "orderType": "LMT"}},
        ],
        {"symbol": "VWCE"},
    )

    assert duplicate is not None
    assert duplicate["contract"]["symbol"] == "VWCE"


def test_run_cycle_dry_run_builds_limit_buy_payload_when_condition_met() -> None:
    args = Namespace(execute_orders=False, confirm_risk=False, confirm_live_risk=False, quote_wait_seconds=0)
    plan = {
        "account_mode": "paper",
        "orders": [
            {
                "symbol": "VWCE",
                "exchange": "SMART",
                "currency": "EUR",
                "side": "BUY",
                "amount_cash": 500,
                "condition": {"type": "price_at_or_below", "price": 120.5},
                "order_type": "LMT",
                "limit_price": 120.5,
                "time_in_force": "DAY",
                "resubmit_if_expired": True,
            }
        ],
    }

    report = run_cycle(args=args, plan=plan, broker=FakeBroker(), terminal=set())
    row = report["orders"][0]

    assert row["payload"] == {
        "symbol": "VWCE",
        "exchange": "SMART",
        "currency": "EUR",
        "action": "BUY",
        "orderType": "LMT",
        "totalQuantity": 4.149378,
        "lmtPrice": 120.5,
        "tif": "DAY",
    }
    assert row["effects"] == [{"action": "would_submit_limit_buy", "payload": row["payload"]}]


class FakeBroker:
    def account_values(self):
        return []

    def open_orders(self):
        return []

    def make_stock_contract(self, *, symbol: str, exchange: str, currency: str):
        return type("Contract", (), {"symbol": symbol, "secType": "STK", "exchange": exchange, "currency": currency, "conId": 123})()

    def snapshot_quote(self, contract, *, wait_seconds: float):
        return {"ask": 120.0, "last": 119.9, "marketPrice": 120.0}
