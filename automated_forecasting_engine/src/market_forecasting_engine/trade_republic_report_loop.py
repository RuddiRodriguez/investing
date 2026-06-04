from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.trade_republic_report import build_report, load_env_file, write_report


DEFAULT_EXPORT_DIR = Path("trade_republic_exports")
DEFAULT_PORTFOLIO = DEFAULT_EXPORT_DIR / "portfolio.csv"
DEFAULT_TRANSACTIONS = DEFAULT_EXPORT_DIR / "account_transactions.csv"
DEFAULT_ISIN_MAP = DEFAULT_EXPORT_DIR / "isin_map.csv"
DEFAULT_OUTPUT = DEFAULT_EXPORT_DIR / "investment_report_latest.json"


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()
    run_loop(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Continuously refresh the read-only Trade Republic dashboard report.")
    parser.add_argument("--portfolio", type=Path, default=DEFAULT_PORTFOLIO)
    parser.add_argument("--transactions", type=Path, default=DEFAULT_TRANSACTIONS)
    parser.add_argument("--isin-map", type=Path, default=DEFAULT_ISIN_MAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--fetch-yahoo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fetch-alpaca", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-portfolio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--refresh-movements-every", type=int, default=15)
    parser.add_argument("--pytr-timeout-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    return parser


def run_loop(args: argparse.Namespace) -> None:
    iteration = 0
    while True:
        iteration += 1
        started = datetime.now(UTC)
        status: dict[str, Any] = {"iteration": iteration, "started_at": started.isoformat(), "events": []}
        try:
            if args.refresh_portfolio:
                refresh_portfolio(args, status)
            if args.refresh_movements_every > 0 and (iteration == 1 or iteration % args.refresh_movements_every == 0):
                refresh_movements(args, status)
            report = build_report(
                portfolio_path=args.portfolio,
                transactions_path=args.transactions,
                isin_map_path=args.isin_map,
                fetch_yahoo=args.fetch_yahoo,
                fetch_alpaca=args.fetch_alpaca,
            )
            atomic_write_report(report, args.output)
            summary = report.get("summary", {})
            status["events"].append(
                {
                    "event": "report_written",
                    "output": str(args.output),
                    "report_timestamp": summary.get("report_timestamp"),
                    "total_current_value": summary.get("total_current_value"),
                    "total_unrealized_pl": summary.get("total_unrealized_pl"),
                }
            )
        except Exception as exc:  # noqa: BLE001
            status["events"].append({"event": "refresh_failed", "error": str(exc)})
        status["finished_at"] = datetime.now(UTC).isoformat()
        print(json.dumps(status, ensure_ascii=True), flush=True)
        if args.once:
            return
        elapsed = (datetime.now(UTC) - started).total_seconds()
        time.sleep(max(1, int(args.interval_seconds) - elapsed))


def refresh_portfolio(args: argparse.Namespace, status: dict[str, Any]) -> None:
    run_readonly_export(
        [
            "portfolio",
            "--output",
            str(args.portfolio),
            "--lang",
            "en",
            "--no-decimal-localization",
            "--no-include-watchlist",
        ],
        timeout_seconds=args.pytr_timeout_seconds,
    )
    status["events"].append({"event": "portfolio_exported", "path": str(args.portfolio)})


def refresh_movements(args: argparse.Namespace, status: dict[str, Any]) -> None:
    run_readonly_export(
        [
            "movements",
            "--output-dir",
            str(args.transactions.parent),
            "--output-file",
            str(args.transactions),
            "--format",
            "csv",
            "--last-days",
            "0",
            "--lang",
            "en",
        ],
        timeout_seconds=args.pytr_timeout_seconds,
    )
    status["events"].append({"event": "movements_exported", "path": str(args.transactions)})


def run_readonly_export(args: list[str], *, timeout_seconds: int) -> None:
    command = [
        sys.executable,
        "-m",
        "market_forecasting_engine.trade_republic_readonly_cli",
        "--allow-login",
        "--store-credentials",
        *args,
    ]
    result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=timeout_seconds)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"read_only_export_failed rc={result.returncode}: {message[:500]}")


def atomic_write_report(report: dict[str, Any], output: Path) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    write_report(report, tmp, "json")
    tmp.replace(output)


if __name__ == "__main__":
    main()
