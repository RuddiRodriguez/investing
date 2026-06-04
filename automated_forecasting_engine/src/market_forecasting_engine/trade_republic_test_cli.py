from __future__ import annotations

import argparse
import json
from typing import Any

from market_forecasting_engine.trade_republic_broker import (
    TradeRepublicReadOnlyBroker,
    summarize_instrument,
    summarize_timeline,
    summarize_ticker,
)


def main() -> None:
    args = build_parser().parse_args()
    payload: dict[str, Any] = {
        "broker": "traderepublic",
        "unofficial_api": True,
        "dry_run": True,
        "execution_enabled": False,
        "errors": [],
    }
    if args.search:
        _record_probe(
            payload,
            "search",
            lambda: _new_broker(args).public_search(
                args.search,
                page=args.page,
                page_size=args.page_size,
                instrument_type=args.instrument_type,
                jurisdiction=args.jurisdiction,
            ),
        )
    if args.instrument:
        _record_probe(payload, "instrument", lambda: summarize_instrument(_new_broker(args).public_instrument(args.instrument)))
    if args.ticker:
        _record_probe(payload, "ticker", lambda: summarize_ticker(_new_broker(args).public_ticker(args.ticker, exchange=args.exchange)))
    if args.account:
        _record_probe(
            payload,
            "account",
            lambda: _new_broker(args).account_snapshot(
                allow_login=args.allow_login,
                allow_device_registration=args.allow_device_registration,
            ),
        )
    if args.cash:
        _record_probe(
            payload,
            "cash",
            lambda: _new_broker(args).cash_snapshot(
                allow_login=args.allow_login,
                allow_device_registration=args.allow_device_registration,
            ),
        )
    if args.portfolio:
        _record_probe(
            payload,
            "portfolio",
            lambda: _new_broker(args).portfolio_snapshot(
                allow_login=args.allow_login,
                allow_device_registration=args.allow_device_registration,
            ),
        )
    if args.orders:
        _record_probe(
            payload,
            "orders",
            lambda: _new_broker(args).order_snapshot(
                allow_login=args.allow_login,
                allow_device_registration=args.allow_device_registration,
            ),
        )
    if args.timeline:
        _record_probe(
            payload,
            "timeline",
            lambda: summarize_timeline(
                _new_broker(args).timeline_movements(
                    after=args.after,
                    allow_login=args.allow_login,
                    allow_device_registration=args.allow_device_registration,
                ),
                limit=args.limit,
            ),
        )
    if args.timeline_detail:
        _record_probe(
            payload,
            "timeline_detail",
            lambda: _new_broker(args).timeline_detail(
                args.timeline_detail,
                allow_login=args.allow_login,
                allow_device_registration=args.allow_device_registration,
            ),
        )
    print(json.dumps(payload, indent=2, default=str, ensure_ascii=True))
    if payload["errors"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dry-run Trade Republic API connectivity probe.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--search", default=None, help="Run a public instrument search query.")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument("--instrument-type", default="stock", choices=("stock", "fund", "derivative", "crypto"))
    parser.add_argument("--jurisdiction", default="DE")
    parser.add_argument("--instrument", default=None, help="Fetch public instrument metadata by ISIN.")
    parser.add_argument("--ticker", default=None, help="Fetch public ticker by ISIN.")
    parser.add_argument("--exchange", default="LSX")
    parser.add_argument("--account", action="store_true", help="Fetch read-only cash, available cash, portfolio, and orders. Requires --allow-login.")
    parser.add_argument("--cash", action="store_true", help="Fetch read-only cash balances. Requires --allow-login.")
    parser.add_argument("--portfolio", action="store_true", help="Fetch read-only portfolio positions. Requires --allow-login.")
    parser.add_argument("--orders", action="store_true", help="Fetch read-only orders as exposed by the upstream API. Requires --allow-login.")
    parser.add_argument("--timeline", action="store_true", help="Fetch read-only portfolio/cash movement timeline. Requires --allow-login.")
    parser.add_argument("--after", default=None, help="Timeline pagination cursor/event id passed to Trade Republic.")
    parser.add_argument("--limit", type=int, default=25, help="Maximum summarized timeline events to print.")
    parser.add_argument("--timeline-detail", default=None, help="Fetch read-only details for one timeline movement id. Requires --allow-login.")
    parser.add_argument("--allow-login", action="store_true", help="Permit Trade Republic authentication for read-only account probes.")
    parser.add_argument(
        "--allow-device-registration",
        action="store_true",
        help="Permit upstream interactive device registration if no key file exists.",
    )
    return parser


def _record_probe(payload: dict[str, Any], name: str, fn: Any) -> None:
    try:
        payload[name] = fn()
    except Exception as exc:
        payload["errors"].append(
            {
                "probe": name,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        )


def _new_broker(args: argparse.Namespace) -> TradeRepublicReadOnlyBroker:
    return TradeRepublicReadOnlyBroker(timeout=args.timeout)


if __name__ == "__main__":
    main()
