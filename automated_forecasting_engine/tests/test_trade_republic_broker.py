from __future__ import annotations

from pathlib import Path

import pytest

from market_forecasting_engine.trade_republic_broker import (
    TradeRepublicReadOnlyBroker,
    _normalize_isin,
    summarize_instrument,
    summarize_ticker,
)


class FakeTrApi:
    login_calls = 0

    def __init__(self, number: str, pin: str, timeout: float = 20.0, locale: str = "en") -> None:
        self.number = number
        self.pin = pin
        self.timeout = timeout
        self.locale = locale

    def instrument(self, isin: str) -> dict:
        return {"isin": isin, "name": "Tesla Inc", "type": "stock", "ignored": "x"}

    def ticker(self, isin: str, exchange: str = "LSX") -> dict:
        return {"key": f"ticker {isin} {exchange}", "bid": {"price": 100.0}, "ask": {"price": 100.2}, "ignored": "x"}

    def neon_search(self, **kwargs: object) -> dict:
        return {"query": kwargs["query"], "pageSize": kwargs["page_size"]}

    def login(self) -> None:
        self.__class__.login_calls += 1

    def cash(self) -> dict:
        return {"amount": 1000}

    def available_cash(self) -> dict:
        return {"amount": 900}

    def available_cash_for_payout(self) -> dict:
        return {"amount": 800}

    def portfolio(self) -> dict:
        return {"positions": []}

    def orders(self) -> list:
        return []

    def timeline(self, after: str | None = None) -> dict:
        return {
            "data": [
                {
                    "type": "timelineEvent",
                    "data": {
                        "id": after or "movement-1",
                        "timestamp": 1780000000,
                        "title": "Einzahlung",
                        "body": "Geldeingang",
                        "cashChangeAmount": 100.0,
                    },
                }
            ]
        }

    def timeline_detail(self, movement_id: str) -> dict:
        return {"id": movement_id, "sections": []}


def test_normalize_isin_requires_real_isin_shape() -> None:
    assert _normalize_isin(" us88160r1014 ") == "US88160R1014"
    with pytest.raises(ValueError, match="ISIN"):
        _normalize_isin("TSLA")


def test_public_probe_does_not_require_credentials() -> None:
    broker = TradeRepublicReadOnlyBroker(api_factory=FakeTrApi)

    assert broker.public_instrument("US88160R1014")["isin"] == "US88160R1014"
    assert broker.public_ticker("US88160R1014", exchange="lsx")["ask"]["price"] == 100.2
    assert broker.public_search("tesla", page_size=5)["pageSize"] == 5


def test_account_snapshot_requires_explicit_login_flag(tmp_path: Path) -> None:
    broker = TradeRepublicReadOnlyBroker(
        number="+491234567",
        pin="1234",
        key_dir=tmp_path,
        api_factory=FakeTrApi,
    )

    with pytest.raises(RuntimeError, match="--allow-login"):
        broker.account_snapshot()


def test_account_snapshot_blocks_registration_without_explicit_flag(tmp_path: Path) -> None:
    broker = TradeRepublicReadOnlyBroker(
        number="+491234567",
        pin="1234",
        key_dir=tmp_path,
        api_factory=FakeTrApi,
    )

    with pytest.raises(RuntimeError, match="key file is missing"):
        broker.account_snapshot(allow_login=True)


def test_account_snapshot_is_read_only_after_key_exists(tmp_path: Path) -> None:
    (tmp_path / "key").write_text("fake-key", encoding="utf-8")
    FakeTrApi.login_calls = 0
    broker = TradeRepublicReadOnlyBroker(
        number="+491234567",
        pin="1234",
        key_dir=tmp_path,
        api_factory=FakeTrApi,
    )

    snapshot = broker.account_snapshot(allow_login=True)

    assert FakeTrApi.login_calls == 1
    assert snapshot == {
        "cash": {"amount": 1000},
        "available_cash": {"amount": 900},
        "portfolio": {"positions": []},
        "orders": [],
    }


def test_cash_portfolio_orders_and_timeline_are_read_only_after_key_exists(tmp_path: Path) -> None:
    (tmp_path / "key").write_text("fake-key", encoding="utf-8")
    broker = TradeRepublicReadOnlyBroker(
        number="+491234567",
        pin="1234",
        key_dir=tmp_path,
        api_factory=FakeTrApi,
    )

    assert broker.cash_snapshot(allow_login=True)["available_cash_for_payout"] == {"amount": 800}
    assert broker.portfolio_snapshot(allow_login=True) == {"positions": []}
    assert broker.order_snapshot(allow_login=True) == []
    assert broker.timeline_movements(after="cursor-1", allow_login=True)["data"][0]["data"]["id"] == "cursor-1"
    assert broker.timeline_detail("movement-1", allow_login=True) == {"id": "movement-1", "sections": []}


def test_summarizers_keep_auditable_trading_fields() -> None:
    assert summarize_ticker({"bid": 1, "ask": 2, "last": 3, "ignored": 4}) == {"bid": 1, "ask": 2, "last": 3}
    assert summarize_instrument({"isin": "US88160R1014", "name": "Tesla", "type": "stock", "ignored": "x"}) == {
        "isin": "US88160R1014",
        "name": "Tesla",
        "type": "stock",
    }
