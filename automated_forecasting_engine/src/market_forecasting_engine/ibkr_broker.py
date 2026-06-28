from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IBKRConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 7
    account_mode: str = "paper"
    readonly: bool = False


class IBKRBroker:
    """Small IBKR wrapper around ib_insync.

    It expects IB Gateway or TWS to be already open and logged in.
    No username, password, or 2FA code is stored here.
    """

    def __init__(self, config: IBKRConnectionConfig) -> None:
        self.config = config
        try:
            from ib_insync import IB
        except ImportError as exc:
            raise RuntimeError("Install ib_insync to use IBKR trading: python -m pip install ib_insync") from exc
        self.ib = IB()

    def connect(self) -> None:
        self.ib.connect(self.config.host, self.config.port, clientId=self.config.client_id, readonly=self.config.readonly)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    def account_values(self) -> list[dict[str, Any]]:
        return [
            {
                "account": getattr(row, "account", None),
                "tag": getattr(row, "tag", None),
                "value": getattr(row, "value", None),
                "currency": getattr(row, "currency", None),
                "modelCode": getattr(row, "modelCode", None),
            }
            for row in self.ib.accountValues()
        ]

    def positions(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in self.ib.positions():
            output.append(
                {
                    "account": row.account,
                    "symbol": getattr(row.contract, "symbol", None),
                    "secType": getattr(row.contract, "secType", None),
                    "exchange": getattr(row.contract, "exchange", None),
                    "currency": getattr(row.contract, "currency", None),
                    "position": row.position,
                    "avgCost": row.avgCost,
                }
            )
        return output

    def open_orders(self) -> list[dict[str, Any]]:
        return [trade_to_dict(trade) for trade in self.ib.openTrades()]

    def make_stock_contract(self, *, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        from ib_insync import Stock

        contract = Stock(symbol.upper(), exchange, currency.upper())
        qualified = self.ib.qualifyContracts(contract)
        if not qualified:
            raise RuntimeError(f"IBKR could not qualify contract {symbol} {exchange} {currency}.")
        return qualified[0]

    def snapshot_quote(self, contract: Any, *, wait_seconds: float = 2.0) -> dict[str, Any]:
        ticker = self.ib.reqMktData(contract, "", False, False)
        self.ib.sleep(wait_seconds)
        quote = {
            "bid": clean_number(ticker.bid),
            "ask": clean_number(ticker.ask),
            "last": clean_number(ticker.last),
            "close": clean_number(ticker.close),
            "marketPrice": clean_number(ticker.marketPrice()),
        }
        self.ib.cancelMktData(contract)
        return quote

    def place_limit_buy(self, contract: Any, *, quantity: float, limit_price: float, tif: str = "DAY") -> dict[str, Any]:
        from ib_insync import LimitOrder

        order = LimitOrder("BUY", quantity, limit_price, tif=tif)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1.0)
        return trade_to_dict(trade)

    def sleep(self, seconds: float) -> None:
        self.ib.sleep(seconds)


def trade_to_dict(trade: Any) -> dict[str, Any]:
    contract = getattr(trade, "contract", None)
    order = getattr(trade, "order", None)
    status = getattr(trade, "orderStatus", None)
    return {
        "contract": {
            "symbol": getattr(contract, "symbol", None),
            "secType": getattr(contract, "secType", None),
            "exchange": getattr(contract, "exchange", None),
            "currency": getattr(contract, "currency", None),
            "conId": getattr(contract, "conId", None),
        },
        "order": {
            "orderId": getattr(order, "orderId", None),
            "clientId": getattr(order, "clientId", None),
            "action": getattr(order, "action", None),
            "orderType": getattr(order, "orderType", None),
            "totalQuantity": getattr(order, "totalQuantity", None),
            "lmtPrice": getattr(order, "lmtPrice", None),
            "tif": getattr(order, "tif", None),
        },
        "status": {
            "status": getattr(status, "status", None),
            "filled": getattr(status, "filled", None),
            "remaining": getattr(status, "remaining", None),
            "avgFillPrice": getattr(status, "avgFillPrice", None),
        },
    }


def clean_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number
