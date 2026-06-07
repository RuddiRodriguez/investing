from __future__ import annotations

from pathlib import Path

from market_forecasting_engine.llm_options_trader.alpaca_common import AlpacaLLMOptionsRuntimeConfig, compact_alpaca_market_packet, validate_alpaca_order_payload
from market_forecasting_engine.llm_options_trader.alpaca_agent import _entry_blocked_by_market
from market_forecasting_engine.llm_options_trader.alpaca_shadow_ledger import (
    load_and_update_alpaca_shadow_state,
    record_simulated_alpaca_order,
)


class FakeAlpacaBroker:
    def __init__(self) -> None:
        self.quotes = {
            "NVDA260612C00210000": {"bid": 4.4, "ask": 4.5},
        }

    def option_snapshots(self, symbols: list[str], *, feed: str | None = None) -> dict:
        output = {}
        for symbol in symbols:
            quote = self.quotes.get(symbol, {"bid": 4.4, "ask": 4.5})
            output[symbol] = {"latestQuote": {"bp": quote["bid"], "ap": quote["ask"]}}
        return output


def test_validate_alpaca_order_payload_uses_contract_qty_and_limits() -> None:
    config = AlpacaLLMOptionsRuntimeConfig(ticker="NVDA", max_order_qty=1, max_order_price=10.0, max_order_debit=1000.0)

    result = validate_alpaca_order_payload(
        {
            "symbol": "NVDA260612C00210000",
            "side": "buy",
            "type": "limit",
            "qty": 1,
            "limit_price": 4.5,
            "time_in_force": "day",
        },
        config=config,
        require_exit=False,
    )

    assert result["blocks"] == []
    assert result["order"]["qty"] == 1


def test_alpaca_shadow_ledger_uses_option_contract_multiplier(tmp_path: Path) -> None:
    broker = FakeAlpacaBroker()
    order = {
        "symbol": "NVDA260612C00210000",
        "side": "buy",
        "type": "limit",
        "qty": 1,
        "limit_price": 4.5,
        "time_in_force": "day",
    }

    result = record_simulated_alpaca_order(
        output_dir=tmp_path,
        ticker="NVDA",
        broker=broker,
        validated_order=order,
        decision={"reason": "test"},
        checked_at_utc="2026-06-07T12:00:00+00:00",
    )

    assert result["shadow_state"]["position_count"] == 1
    position = result["shadow_state"]["positions"][0]
    assert position["average_price"] == 4.5
    assert position["unrealized_pnl"] == -5.0

    broker.quotes["NVDA260612C00210000"] = {"bid": 5.0, "ask": 5.1}
    updated = load_and_update_alpaca_shadow_state(output_dir=tmp_path, ticker="NVDA", broker=broker)
    assert updated["unrealized_pnl"] == 55.0


def test_validate_alpaca_crypto_spot_order_allows_decimal_qty() -> None:
    config = AlpacaLLMOptionsRuntimeConfig(ticker="ETH-USD", max_crypto_notional=100.0, max_order_price=10000.0)

    result = validate_alpaca_order_payload(
        {
            "symbol": "ETH/USD",
            "side": "buy",
            "type": "limit",
            "qty": 0.02,
            "limit_price": 2000.0,
            "time_in_force": "gtc",
        },
        config=config,
        require_exit=False,
    )

    assert result["blocks"] == []
    assert result["order"]["asset_class"] == "crypto_spot"
    assert result["order"]["qty"] == 0.02


def test_non_crypto_entry_blocks_when_market_closed() -> None:
    config = AlpacaLLMOptionsRuntimeConfig(ticker="NVDA")
    decision = {"intent": "open_call", "order": {"side": "buy"}}

    blocked = _entry_blocked_by_market(
        packet_market_policy={"can_open_new_shadow_entries": False},
        config=config,
        decision=decision,
    )

    assert blocked is True


def test_crypto_entry_does_not_block_when_equity_market_closed() -> None:
    config = AlpacaLLMOptionsRuntimeConfig(ticker="ETH-USD")
    decision = {"intent": "open_call", "order": {"side": "buy"}}

    blocked = _entry_blocked_by_market(
        packet_market_policy={"can_open_new_shadow_entries": False},
        config=config,
        decision=decision,
    )

    assert blocked is False


def test_compact_alpaca_packet_keeps_strategy_mode() -> None:
    packet = compact_alpaca_market_packet(
        {
            "venue": "alpaca",
            "strategy_mode": {"name": "crypto_spot_probe", "allowed_entry_intents": ["open_spot_long", "hold"]},
            "recent_price_bars": [{"close": 1}, {"close": 2}],
            "option_chain": [],
        }
    )

    assert packet["strategy_mode"]["name"] == "crypto_spot_probe"
