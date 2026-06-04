from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from market_forecasting_engine.deribit_broker import DeribitReadOnlyBroker
from market_forecasting_engine.live_trading.deribit_report import build_deribit_account_report, write_deribit_account_report


def main() -> None:
    args = build_parser().parse_args()
    currencies = _csv(args.currencies)
    kinds = _csv(args.kinds)
    broker = DeribitReadOnlyBroker(account_mode=args.account_mode, base_url=args.base_url)
    while True:
        report = build_deribit_account_report(
            broker,
            currencies=currencies,
            kinds=kinds,
            history_count=int(args.history_count),
        )
        name = f"deribit_{args.account_mode}_account_report"
        output_path = write_deribit_account_report(report, Path(args.output_dir), name=name)
        print(
            json.dumps(
                {
                    "report": str(output_path),
                    "mode": report["mode"],
                    "venue": report["venue"],
                    "endpoint": report["endpoint"],
                    "safety": report["safety"],
                    "overview": report["overview"],
                    "access_issues": report["access_issues"],
                },
                indent=2,
                default=str,
            ),
            flush=True,
        )
        if not args.watch:
            break
        time.sleep(max(10, int(args.refresh_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a read-only Deribit account report for testnet or live.")
    parser.add_argument("--account-mode", choices=("testnet", "live"), default="testnet")
    parser.add_argument("--base-url", default=None, help="Override Deribit API URL. Must match account mode.")
    parser.add_argument("--currencies", default="ETH,BTC,USDC")
    parser.add_argument("--kinds", default="option,future,spot")
    parser.add_argument("--history-count", type=int, default=100)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/live_trading")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh the read-only report file.")
    parser.add_argument("--refresh-seconds", type=int, default=60)
    return parser


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


if __name__ == "__main__":
    main()
