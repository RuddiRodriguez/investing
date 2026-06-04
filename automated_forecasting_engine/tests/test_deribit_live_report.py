from __future__ import annotations

import pytest

from market_forecasting_engine.deribit_broker import DeribitReadOnlyBroker
from market_forecasting_engine.live_trading.deribit_report import build_deribit_account_report


class FakeDeribitReadOnlyBroker:
    account_mode = "live"
    base_url = "https://www.deribit.com/api/v2"

    def account_summary(self, *, currency: str = "ETH"):
        return {
            "currency": currency,
            "equity": 10,
            "balance": 9,
            "available_funds": 8,
            "initial_margin": 1,
            "maintenance_margin": 0.5,
            "options_value": 0.03,
            "total_pl": -0.01,
            "delta_total": -0.45,
        }

    def positions(self, *, currency: str = "ETH", kind: str = "any"):
        if kind == "option":
            return [
                {
                    "instrument_name": "ETH-12JUN26-1800-P",
                    "kind": "option",
                    "currency": currency,
                    "direction": "buy",
                    "size": 1,
                    "average_price": 0.0345,
                    "mark_price": 0.0302,
                    "floating_profit_loss": -0.0043,
                    "total_profit_loss": -0.0043,
                    "delta": -0.45,
                    "gamma": 0.00255,
                    "vega": 1.09,
                }
            ]
        if kind == "future":
            return [
                {
                    "instrument_name": "ETH-PERPETUAL",
                    "kind": "future",
                    "currency": currency,
                    "direction": "buy",
                    "size": 100,
                    "average_price": 1800,
                    "mark_price": 1810,
                    "floating_profit_loss": 0.01,
                    "total_profit_loss": 0.01,
                    "delta": 0.05,
                }
            ]
        return []

    def open_orders(self, *, currency: str = "ETH", kind: str = "any"):
        if kind == "option":
            return [
                {
                    "order_id": "open-option",
                    "instrument_name": "ETH-12JUN26-1800-P",
                    "kind": "option",
                    "direction": "buy",
                    "order_type": "limit",
                    "order_state": "open",
                    "amount": 1,
                    "filled_amount": 0,
                    "price": 0.03,
                    "creation_timestamp": 1781200000000,
                }
            ]
        return []

    def order_history(self, *, currency: str = "ETH", kind: str = "any", count: int = 100):
        return []

    def user_trades(self, *, currency: str = "ETH", kind: str = "any", count: int = 100):
        if kind == "option":
            return [
                {
                    "trade_id": "trade-1",
                    "instrument_name": "ETH-12JUN26-1800-P",
                    "kind": "option",
                    "direction": "buy",
                    "price": 0.0345,
                    "amount": 1,
                    "fee": 0.0001,
                    "fee_currency": "ETH",
                    "timestamp": 1781200000000,
                }
            ]
        return []


def test_deribit_report_splits_options_and_non_options_and_is_read_only() -> None:
    report = build_deribit_account_report(
        FakeDeribitReadOnlyBroker(),
        currencies=["ETH"],
        kinds=["option", "future"],
        history_count=25,
    )

    assert report["mode"] == "read_only_deribit_account_report"
    assert report["venue"] == "deribit_live"
    assert report["safety"]["order_submission_enabled"] is False
    assert report["overview"]["option_position_count"] == 1
    assert report["overview"]["non_option_position_count"] == 1
    assert report["overview"]["option_open_order_count"] == 1
    assert report["options"]["positions"][0]["option_details"] == {
        "underlying": "ETH",
        "expiration": "2026-06-12",
        "option_type": "put",
        "strike": 1800.0,
    }
    assert report["non_options"]["positions"][0]["instrument_name"] == "ETH-PERPETUAL"


def test_deribit_live_read_only_broker_rejects_testnet_endpoint() -> None:
    with pytest.raises(RuntimeError, match="live mode with the testnet endpoint"):
        DeribitReadOnlyBroker(
            account_mode="live",
            base_url="https://test.deribit.com/api/v2",
            client_id="id",
            client_secret="secret",
        )


def test_deribit_testnet_read_only_broker_rejects_live_endpoint() -> None:
    with pytest.raises(RuntimeError, match="testnet mode with a non-testnet endpoint"):
        DeribitReadOnlyBroker(
            account_mode="testnet",
            base_url="https://www.deribit.com/api/v2",
            client_id="id",
            client_secret="secret",
        )
