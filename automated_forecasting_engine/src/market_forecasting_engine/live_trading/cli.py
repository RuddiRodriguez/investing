from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker, load_env_file
from market_forecasting_engine.live_trading.report import build_live_account_report, write_live_account_report


LIVE_BASE_URL = "https://api.alpaca.markets"
PAPER_BASE_URL = "https://paper-api.alpaca.markets"


def main() -> None:
    args = build_parser().parse_args()
    load_env_file()
    base_url = _base_url(args)
    key_id, secret_key = _credentials(args)
    broker = AlpacaPaperBroker(base_url=base_url, key_id=key_id, secret_key=secret_key)
    while True:
        report = build_live_account_report(
            broker,
            venue=f"alpaca_{args.account_mode}",
            order_limit=int(args.order_limit),
            include_closed_orders=not args.open_orders_only,
        )
        output_path = write_live_account_report(report, Path(args.output_dir))
        print(
            json.dumps(
                {
                    "report": str(output_path),
                    "mode": report["mode"],
                    "venue": report["venue"],
                    "safety": report["safety"],
                    "overview": report["overview"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(max(5, int(args.refresh_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a read-only live trading account report.")
    parser.add_argument("--account-mode", choices=("paper", "live"), default="paper")
    parser.add_argument("--base-url", default=None, help="Override Alpaca trading API base URL. Use only when you know which account it targets.")
    parser.add_argument("--live-key-id-env", default="ALPACA_LIVE_API_KEY_ID")
    parser.add_argument("--live-secret-key-env", default="ALPACA_LIVE_API_SECRET_KEY")
    parser.add_argument("--paper-key-id-env", default="ALPACA_API_KEY_ID")
    parser.add_argument("--paper-secret-key-env", default="ALPACA_API_SECRET_KEY")
    parser.add_argument(
        "--allow-generic-live-credentials",
        action="store_true",
        help="Allow live mode to fall back to ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY. Not recommended.",
    )
    parser.add_argument("--order-limit", type=int, default=100)
    parser.add_argument("--open-orders-only", action="store_true")
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/live_trading")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh the read-only report file.")
    parser.add_argument("--refresh-seconds", type=int, default=60)
    return parser


def _base_url(args: argparse.Namespace) -> str:
    if args.base_url:
        url = str(args.base_url).rstrip("/")
    else:
        url = LIVE_BASE_URL if args.account_mode == "live" else PAPER_BASE_URL
    if args.account_mode == "live" and "paper-api" in url:
        raise SystemExit("Refusing live mode with a paper Alpaca endpoint.")
    if args.account_mode == "paper" and "paper-api" not in url:
        raise SystemExit("Refusing paper mode with a non-paper Alpaca endpoint.")
    return url


def _credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.account_mode == "live":
        key_id = os.getenv(str(args.live_key_id_env))
        secret_key = os.getenv(str(args.live_secret_key_env))
        if (not key_id or not secret_key) and args.allow_generic_live_credentials:
            key_id = key_id or os.getenv("ALPACA_API_KEY_ID") or os.getenv("APCA_API_KEY_ID")
            secret_key = secret_key or os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        if not key_id or not secret_key:
            raise SystemExit(
                "Live mode requires live credentials in "
                f"{args.live_key_id_env} and {args.live_secret_key_env}. "
                "Do not use paper API keys for real-money reporting."
            )
        return key_id, secret_key
    key_id = os.getenv(str(args.paper_key_id_env)) or os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv(str(args.paper_secret_key_env)) or os.getenv("APCA_API_SECRET_KEY")
    return key_id, secret_key


if __name__ == "__main__":
    main()
