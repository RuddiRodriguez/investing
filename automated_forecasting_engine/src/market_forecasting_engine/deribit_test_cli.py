from __future__ import annotations

import argparse
import json

from market_forecasting_engine.deribit_broker import DeribitTestnetBroker, summarize_instrument


def main() -> None:
    args = build_parser().parse_args()
    broker = DeribitTestnetBroker()
    payload: dict[str, object] = {"testnet": True, "currency": args.currency.upper()}
    if args.instruments:
        instruments = broker.instruments(currency=args.currency, kind=args.kind, expired=False)
        payload["instrument_count"] = len(instruments)
        payload["instruments"] = [summarize_instrument(item) for item in instruments[: int(args.limit)]]
    if args.ticker:
        payload["ticker"] = broker.ticker(args.ticker)
    if args.order_book:
        payload["order_book"] = broker.order_book(args.order_book, depth=args.depth)
    if args.account:
        summary = broker.account_summary(currency=args.currency)
        payload["account"] = _safe_account_summary(summary)
    print(json.dumps(payload, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify Deribit testnet connectivity and account access.")
    parser.add_argument("--currency", default="ETH", choices=("BTC", "ETH", "USDC", "USDT"))
    parser.add_argument("--kind", default="option", choices=("option", "future", "spot", "future_combo", "option_combo"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--instruments", action="store_true", help="List public testnet instruments.")
    parser.add_argument("--ticker", default=None, help="Fetch public ticker for one instrument.")
    parser.add_argument("--order-book", default=None, help="Fetch public order book for one instrument.")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--account", action="store_true", help="Fetch private account summary using testnet credentials.")
    return parser


def _safe_account_summary(summary: dict) -> dict:
    allowed = [
        "currency",
        "balance",
        "available_funds",
        "available_withdrawal_funds",
        "equity",
        "margin_balance",
        "initial_margin",
        "maintenance_margin",
        "options_session_rpl",
        "options_session_upl",
        "futures_session_rpl",
        "futures_session_upl",
        "limits",
    ]
    return {key: summary.get(key) for key in allowed if key in summary}


if __name__ == "__main__":
    main()
