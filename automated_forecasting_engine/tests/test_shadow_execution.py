from __future__ import annotations

from pathlib import Path

from market_forecasting_engine.llm_options_trader.alpaca_shadow_ledger import (
    load_and_update_alpaca_shadow_state,
    record_simulated_alpaca_order,
)
from market_forecasting_engine.llm_options_trader.shadow_ledger import (
    load_and_update_shadow_state,
    record_simulated_order,
)


class FakeDeribitBroker:
    def __init__(self) -> None:
        self.trades: list[dict] = []

    def order_book(self, instrument_name: str, *, depth: int = 5) -> dict:
        return {
            "best_bid_price": 13.2,
            "best_ask_price": 15.6,
            "mark_price": 14.4,
        }

    def public_get(self, method: str, params: dict | None = None) -> dict:
        assert method == "get_last_trades_by_instrument"
        return {"trades": self.trades}


class FakeAlpacaBroker:
    def __init__(self) -> None:
        self.latest_trade: dict = {}

    def option_snapshots(self, symbols: list[str], *, feed: str | None = None) -> dict:
        return {
            symbols[0]: {
                "latestQuote": {"bp": 4.9, "ap": 5.4},
                "latestTrade": self.latest_trade,
            }
        }


def test_deribit_shadow_post_only_buy_fills_when_public_sell_trade_touches_bid(tmp_path: Path) -> None:
    broker = FakeDeribitBroker()
    record_simulated_order(
        output_dir=tmp_path,
        currency="ETH",
        broker=broker,
        checked_at_utc="2026-06-07T17:13:21+00:00",
        decision={"reason": "test"},
        validated_order={
            "instrument_name": "ETH_USDC-8JUN26-1650-C",
            "side": "buy",
            "amount": 0.1,
            "price": 13.4,
            "post_only": True,
        },
    )

    broker.trades = [
        {
            "trade_id": "t-1",
            "timestamp": 1780852442500,
            "price": 13.4,
            "direction": "sell",
        }
    ]
    summary = load_and_update_shadow_state(output_dir=tmp_path, currency="ETH", broker=broker)

    assert summary["open_order_count"] == 0
    assert summary["position_count"] == 1
    assert summary["trades"][-1]["fill_reason"] == "public_trade_touched_passive_bid"
    assert summary["trades"][-1]["price"] == 13.4


def test_deribit_shadow_does_not_backfill_trade_before_order_creation(tmp_path: Path) -> None:
    broker = FakeDeribitBroker()
    broker.trades = [
            {
                "trade_id": "old",
                "timestamp": 1780852400000,
                "price": 13.4,
                "direction": "sell",
            }
    ]
    record_simulated_order(
        output_dir=tmp_path,
        currency="ETH",
        broker=broker,
        checked_at_utc="2026-06-07T17:13:21+00:00",
        decision={"reason": "test"},
        validated_order={
            "instrument_name": "ETH_USDC-8JUN26-1650-C",
            "side": "buy",
            "amount": 0.1,
            "price": 13.4,
            "post_only": True,
        },
    )
    summary = load_and_update_shadow_state(output_dir=tmp_path, currency="ETH", broker=broker)

    assert summary["open_order_count"] == 1
    assert summary["position_count"] == 0
    assert summary["trades"] == []


def test_alpaca_shadow_uses_same_passive_trade_touch_rule(tmp_path: Path) -> None:
    broker = FakeAlpacaBroker()
    record_simulated_alpaca_order(
        output_dir=tmp_path,
        ticker="TSLA",
        broker=broker,
        checked_at_utc="2026-06-07T17:13:21+00:00",
        decision={"reason": "test"},
        validated_order={
            "symbol": "TSLA260605C00425000",
            "side": "buy",
            "qty": 1,
            "limit_price": 5.0,
            "asset_class": "us_option",
        },
    )

    broker.latest_trade = {
        "i": "alpaca-trade-1",
        "t": "2026-06-07T17:13:23Z",
        "p": 5.0,
        "side": "sell",
    }
    summary = load_and_update_alpaca_shadow_state(output_dir=tmp_path, ticker="TSLA", broker=broker)

    assert summary["open_order_count"] == 0
    assert summary["position_count"] == 1
    assert summary["trades"][-1]["fill_reason"] == "public_trade_touched_passive_bid"
    assert summary["trades"][-1]["notional"] == 500.0
