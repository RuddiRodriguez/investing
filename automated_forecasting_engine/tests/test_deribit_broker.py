from __future__ import annotations

import pytest

from market_forecasting_engine.deribit_broker import DeribitTestnetBroker, summarize_instrument
from market_forecasting_engine.deribit_test_cli import _safe_account_summary


def test_deribit_broker_is_locked_to_testnet() -> None:
    with pytest.raises(RuntimeError, match="testnet"):
        DeribitTestnetBroker(base_url="https://www.deribit.com/api/v2", client_id="id", client_secret="secret")


def test_summarize_instrument_keeps_trading_fields() -> None:
    summary = summarize_instrument(
        {
            "instrument_name": "ETH-27JUN26-3000-C",
            "base_currency": "ETH",
            "quote_currency": "ETH",
            "kind": "option",
            "option_type": "call",
            "strike": 3000.0,
            "expiration_timestamp": 1782528000000,
            "is_active": True,
            "min_trade_amount": 0.1,
            "tick_size": 0.0005,
            "ignored": "x",
        }
    )

    assert summary == {
        "instrument_name": "ETH-27JUN26-3000-C",
        "base_currency": "ETH",
        "quote_currency": "ETH",
        "kind": "option",
        "option_type": "call",
        "strike": 3000.0,
        "expiration_timestamp": 1782528000000,
        "is_active": True,
        "min_trade_amount": 0.1,
        "tick_size": 0.0005,
    }


def test_safe_account_summary_does_not_return_credentials_or_extra_fields() -> None:
    safe = _safe_account_summary(
        {
            "currency": "ETH",
            "balance": 10,
            "equity": 11,
            "api_key": "secret",
            "limits": {"matching_engine": {"trading": {"total": 20}}},
        }
    )

    assert safe["currency"] == "ETH"
    assert safe["balance"] == 10
    assert "api_key" not in safe
