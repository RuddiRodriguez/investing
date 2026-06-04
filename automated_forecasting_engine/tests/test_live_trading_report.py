from __future__ import annotations

from market_forecasting_engine.live_trading.report import build_live_account_report
from market_forecasting_engine.live_trading.cli import _base_url, _credentials


class FakeReadOnlyBroker:
    def account(self):
        return {
            "status": "ACTIVE",
            "currency": "USD",
            "cash": "1000",
            "buying_power": "2000",
            "equity": "5000",
            "portfolio_value": "5000",
        }

    def positions(self):
        return [
            {
                "symbol": "AAPL",
                "asset_class": "us_equity",
                "qty": "2",
                "avg_entry_price": "180",
                "current_price": "190",
                "market_value": "380",
                "cost_basis": "360",
                "unrealized_pl": "20",
                "unrealized_plpc": "0.055555",
            },
            {
                "symbol": "AAPL260619C00200000",
                "asset_class": "option",
                "qty": "1",
                "avg_entry_price": "4.5",
                "current_price": "5.0",
                "market_value": "500",
                "cost_basis": "450",
                "unrealized_pl": "50",
                "unrealized_plpc": "0.111111",
            },
        ]

    def orders(self, *, status: str = "open", limit: int = 50):
        if status == "open":
            return [
                {
                    "id": "stock-order",
                    "symbol": "AAPL",
                    "asset_class": "us_equity",
                    "side": "buy",
                    "type": "limit",
                    "qty": "1",
                    "filled_qty": "0",
                    "limit_price": "185",
                    "status": "accepted",
                    "submitted_at": "2026-06-04T13:00:00Z",
                }
            ]
        return [
            {
                "id": "option-order",
                "symbol": "AAPL260619P00170000",
                "asset_class": "option",
                "side": "buy",
                "type": "limit",
                "qty": "1",
                "filled_qty": "1",
                "filled_avg_price": "3.2",
                "status": "filled",
                "submitted_at": "2026-06-04T12:00:00Z",
                "filled_at": "2026-06-04T12:01:00Z",
            }
        ]


def test_live_account_report_splits_stocks_options_and_is_read_only() -> None:
    report = build_live_account_report(FakeReadOnlyBroker(), venue="alpaca_live")

    assert report["mode"] == "read_only_live_account_report"
    assert report["safety"]["order_submission_enabled"] is False
    assert report["overview"]["stock_position_count"] == 1
    assert report["overview"]["option_position_count"] == 1
    assert report["stocks"]["summary"]["unrealized_pl"] == 20
    assert report["options"]["summary"]["unrealized_pl"] == 50
    assert report["options"]["positions"][0]["option_details"] == {
        "underlying": "AAPL",
        "expiration": "2026-06-19",
        "option_type": "call",
        "strike": 200.0,
    }
    assert report["options"]["recent_orders"][0]["option_details"]["option_type"] == "put"


def test_live_cli_uses_distinct_live_endpoint_and_credentials(monkeypatch) -> None:
    from argparse import Namespace

    args = Namespace(
        account_mode="live",
        base_url=None,
        live_key_id_env="ALPACA_LIVE_API_KEY_ID",
        live_secret_key_env="ALPACA_LIVE_API_SECRET_KEY",
        allow_generic_live_credentials=False,
    )
    monkeypatch.setenv("ALPACA_LIVE_API_KEY_ID", "live-key")
    monkeypatch.setenv("ALPACA_LIVE_API_SECRET_KEY", "live-secret")

    assert _base_url(args) == "https://api.alpaca.markets"
    assert _credentials(args) == ("live-key", "live-secret")


def test_live_cli_rejects_paper_endpoint_for_live_mode() -> None:
    from argparse import Namespace
    import pytest

    args = Namespace(account_mode="live", base_url="https://paper-api.alpaca.markets")

    with pytest.raises(SystemExit, match="paper Alpaca endpoint"):
        _base_url(args)
