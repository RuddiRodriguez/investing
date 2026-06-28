from __future__ import annotations

import argparse
import asyncio
import csv
import os
import json
import threading
import time
from decimal import ROUND_HALF_UP, Decimal
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from market_forecasting_engine import snaptrade_readonly_cli
from market_forecasting_engine import trade_republic_report_loop
from market_forecasting_engine.llm_handler import LLMProviderNotConfigured, LLMRequest, default_llm_handler, resolve_llm_client_profile
from market_forecasting_engine.openai_responses import response_payload
from market_forecasting_engine.unified_portfolio import (
    DEFAULT_ASSET_PRICE_HISTORY,
    DEFAULT_BENCHMARK_HISTORY,
    DEFAULT_BENCHMARK_NAME,
    DEFAULT_BENCHMARK_SYMBOL,
    DEFAULT_CLASSIFICATION_CACHE,
    DEFAULT_ALERT_EVENTS,
    DEFAULT_ALLOCATION_TARGETS,
    DEFAULT_PRICE_ALERTS,
    DEFAULT_SCENARIO_HISTORY,
    DEFAULT_SNAPTRADE_SNAPSHOT,
    DEFAULT_TRADE_REPUBLIC_REPORT,
    DEFAULT_TRADE_REPUBLIC_TRANSACTIONS,
    DEFAULT_UNIFIED_HISTORY,
    DEFAULT_WEEKLY_CLEANUP_REVIEW,
    build_unified_portfolio_state,
    create_price_alert,
    refresh_asset_classification_cache,
    refresh_asset_price_history_cache,
    refresh_benchmark_history_cache,
    refresh_scenario_history_cache,
    save_allocation_targets,
    update_price_alert_status,
)


DEFAULT_REFRESH_SECONDS = 60


def main() -> None:
    args = build_parser().parse_args()
    if args.refresh_trade_republic_on_start:
        refresh_trade_republic_once(
            output=args.trade_republic_report,
            transactions=args.trade_republic_transactions,
            pytr_timeout_seconds=args.trade_republic_timeout_seconds,
            refresh_movements=args.refresh_trade_republic_movements,
        )
    if args.refresh_scenario_history_on_start:
        refresh_scenario_history_once(
            trade_republic_report=args.trade_republic_report,
            trade_republic_transactions=args.trade_republic_transactions,
            snaptrade_snapshot=args.snaptrade_snapshot,
            portfolio_history=args.portfolio_history,
            scenario_history=args.scenario_history,
            benchmark_history=args.benchmark_history,
        )
    if args.refresh_benchmark_history_on_start:
        refresh_benchmark_history_once(
            benchmark_symbol=args.benchmark_symbol,
            benchmark_history=args.benchmark_history,
            portfolio_history=args.portfolio_history,
        )
    if args.refresh_classifications_on_start:
        refresh_classifications_once(
            trade_republic_report=args.trade_republic_report,
            trade_republic_transactions=args.trade_republic_transactions,
            snaptrade_snapshot=args.snaptrade_snapshot,
            portfolio_history=args.portfolio_history,
            scenario_history=args.scenario_history,
            benchmark_history=args.benchmark_history,
            classification_cache=args.classification_cache,
            classification_model=args.classification_model,
        )
    if args.refresh_price_history_on_start:
        refresh_asset_price_history_once(
            trade_republic_report=args.trade_republic_report,
            trade_republic_transactions=args.trade_republic_transactions,
            snaptrade_snapshot=args.snaptrade_snapshot,
            portfolio_history=args.portfolio_history,
            scenario_history=args.scenario_history,
            benchmark_history=args.benchmark_history,
            classification_cache=args.classification_cache,
            asset_price_history=args.asset_price_history,
            lookback_days=args.price_history_lookback_days,
        )
    run_server(
        trade_republic_report=args.trade_republic_report,
        trade_republic_transactions=args.trade_republic_transactions,
        snaptrade_snapshot=args.snaptrade_snapshot,
        portfolio_history=args.portfolio_history,
        scenario_history=args.scenario_history,
        benchmark_history=args.benchmark_history,
        classification_cache=args.classification_cache,
        asset_price_history=args.asset_price_history,
        price_alerts=args.price_alerts,
        alert_events=args.alert_events,
        allocation_targets=args.allocation_targets,
        weekly_cleanup_review=args.weekly_cleanup_review,
        assistant_llm_provider=args.assistant_llm_provider,
        assistant_llm_model=args.assistant_llm_model,
        assistant_llm_env_file=args.assistant_llm_env_file,
        assistant_llm_timeout_seconds=args.assistant_llm_timeout_seconds,
        manual_refresh_trade_republic=args.manual_refresh_trade_republic,
        manual_refresh_snaptrade=args.manual_refresh_snaptrade,
        manual_refresh_snaptrade_activities_days=args.manual_refresh_snaptrade_activities_days,
        trade_republic_timeout_seconds=args.trade_republic_timeout_seconds,
        refresh_trade_republic_movements=args.refresh_trade_republic_movements,
        host=args.host,
        port=args.port,
        refresh_seconds=args.refresh_seconds,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified read-only portfolio dashboard for Trade Republic plus SnapTrade brokers.")
    parser.add_argument("--trade-republic-report", type=Path, default=DEFAULT_TRADE_REPUBLIC_REPORT)
    parser.add_argument("--trade-republic-transactions", type=Path, default=DEFAULT_TRADE_REPUBLIC_TRANSACTIONS)
    parser.add_argument("--snaptrade-snapshot", type=Path, default=DEFAULT_SNAPTRADE_SNAPSHOT)
    parser.add_argument("--portfolio-history", type=Path, default=DEFAULT_UNIFIED_HISTORY)
    parser.add_argument("--scenario-history", type=Path, default=DEFAULT_SCENARIO_HISTORY)
    parser.add_argument("--benchmark-history", type=Path, default=DEFAULT_BENCHMARK_HISTORY)
    parser.add_argument("--classification-cache", type=Path, default=DEFAULT_CLASSIFICATION_CACHE)
    parser.add_argument("--asset-price-history", type=Path, default=DEFAULT_ASSET_PRICE_HISTORY)
    parser.add_argument("--price-alerts", type=Path, default=DEFAULT_PRICE_ALERTS)
    parser.add_argument("--alert-events", type=Path, default=DEFAULT_ALERT_EVENTS)
    parser.add_argument("--allocation-targets", type=Path, default=DEFAULT_ALLOCATION_TARGETS)
    parser.add_argument("--weekly-cleanup-review", type=Path, default=DEFAULT_WEEKLY_CLEANUP_REVIEW)
    parser.add_argument("--benchmark-symbol", default=DEFAULT_BENCHMARK_SYMBOL)
    parser.add_argument("--classification-model", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--refresh-seconds", type=int, default=DEFAULT_REFRESH_SECONDS)
    parser.add_argument(
        "--refresh-trade-republic-on-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one read-only Trade Republic report refresh before starting the dashboard.",
    )
    parser.add_argument(
        "--refresh-trade-republic-movements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh Trade Republic movements/transactions during the startup refresh.",
    )
    parser.add_argument("--trade-republic-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--refresh-scenario-history-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch/cache Yahoo Finance historical prices for scenario stress-test windows before starting.",
    )
    parser.add_argument(
        "--refresh-benchmark-history-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch/cache Yahoo Finance benchmark prices before starting.",
    )
    parser.add_argument(
        "--refresh-classifications-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Classify uncached assets with the configured LLM and cache the results before starting.",
    )
    parser.add_argument(
        "--refresh-price-history-on-start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fetch/cache Yahoo Finance daily prices for current holdings before starting.",
    )
    parser.add_argument("--price-history-lookback-days", type=int, default=420)
    parser.add_argument("--assistant-llm-provider", default=None, help="Provider for the dashboard AI assistant. Defaults to LLM_PROVIDER or openai.")
    parser.add_argument("--assistant-llm-model", default=None, help="Model for the dashboard AI assistant. Defaults to provider-specific env/default model.")
    parser.add_argument("--assistant-llm-env-file", type=Path, default=Path(".env"), help="Optional env file with LLM credentials/settings.")
    parser.add_argument("--assistant-llm-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--manual-refresh-trade-republic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the dashboard Refresh Portfolios button to run the read-only Trade Republic refresh.",
    )
    parser.add_argument(
        "--manual-refresh-snaptrade",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow the dashboard Refresh Portfolios button to run the read-only SnapTrade snapshot refresh.",
    )
    parser.add_argument("--manual-refresh-snaptrade-activities-days", type=int, default=365)
    return parser


def refresh_trade_republic_once(
    *,
    output: Path,
    transactions: Path = trade_republic_report_loop.DEFAULT_TRANSACTIONS,
    pytr_timeout_seconds: int,
    refresh_movements: bool,
) -> None:
    print("Starting read-only Trade Republic refresh before dashboard startup...", flush=True)
    trade_republic_report_loop.load_env_file()
    args = argparse.Namespace(
        portfolio=trade_republic_report_loop.DEFAULT_PORTFOLIO,
        transactions=transactions,
        isin_map=trade_republic_report_loop.DEFAULT_ISIN_MAP,
        output=output,
        interval_seconds=60,
        fetch_yahoo=True,
        fetch_alpaca=True,
        refresh_portfolio=True,
        refresh_movements_every=1 if refresh_movements else 0,
        pytr_timeout_seconds=max(10, int(pytr_timeout_seconds)),
        once=True,
    )
    trade_republic_report_loop.run_loop(args)


def refresh_trade_republic_candidate(
    *,
    output: Path,
    transactions: Path,
    pytr_timeout_seconds: int,
    refresh_movements: bool,
) -> dict[str, Any]:
    candidate_report = output.with_name(output.stem + ".manual_refresh" + output.suffix)
    candidate_transactions = transactions.with_name(transactions.stem + ".manual_refresh" + transactions.suffix)
    try:
        refresh_trade_republic_once(
            output=candidate_report,
            transactions=candidate_transactions,
            pytr_timeout_seconds=pytr_timeout_seconds,
            refresh_movements=refresh_movements,
        )
        if refresh_movements and (not candidate_transactions.exists() or candidate_transactions.stat().st_size <= 0):
            raise RuntimeError("Trade Republic refresh produced an empty movements file; keeping the last good report.")
        report = json.loads(candidate_report.read_text(encoding="utf-8"))
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        holding_count = int(summary.get("holding_count") or len(report.get("holdings") or []))
        total_value = float(summary.get("total_current_value") or 0.0)
        if holding_count <= 0 and total_value <= 0:
            raise RuntimeError("Trade Republic refresh produced an empty report; keeping the last good report.")
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate_report.replace(output)
        if candidate_transactions.exists() and candidate_transactions.stat().st_size > 0:
            candidate_transactions.replace(transactions)
        return {"holding_count": holding_count, "total_current_value": total_value}
    finally:
        for path in (candidate_report, candidate_transactions):
            if path.exists():
                try:
                    path.unlink()
                except OSError:
                    pass


class TradeRepublicSessionWorker:
    def __init__(
        self,
        *,
        portfolio_path: Path,
        transactions_path: Path,
        isin_map_path: Path,
        output_path: Path,
        timeout_seconds: int,
        refresh_movements: bool,
    ) -> None:
        self.portfolio_path = portfolio_path
        self.transactions_path = transactions_path
        self.isin_map_path = isin_map_path
        self.output_path = output_path
        self.timeout_seconds = max(10, int(timeout_seconds))
        self.refresh_movements = refresh_movements
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="trade-republic-session-worker", daemon=True)
        self._thread.start()
        self._lock = threading.Lock()
        self._tr: Any | None = None
        self._last_refresh: dict[str, Any] | None = None

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def refresh_report(self, *, output: Path | None = None, transactions: Path | None = None) -> dict[str, Any]:
        with self._lock:
            target_output = output or self.output_path
            target_transactions = transactions or self.transactions_path
            future = asyncio.run_coroutine_threadsafe(
                self._refresh_report_async(target_output=target_output, target_transactions=target_transactions),
                self._loop,
            )
            result = future.result(timeout=self.timeout_seconds + 60)
            self._last_refresh = result
            return result

    async def _refresh_report_async(self, *, target_output: Path, target_transactions: Path) -> dict[str, Any]:
        started = time.time()
        events: list[dict[str, Any]] = []
        try:
            await self._export_portfolio_csv(self.portfolio_path)
            events.append({"event": "portfolio_exported_persistent_session", "path": str(self.portfolio_path)})
        except Exception as exc:  # noqa: BLE001
            await self._close_tr()
            raise RuntimeError(f"persistent Trade Republic portfolio refresh failed: {exc}") from exc

        if self.refresh_movements:
            try:
                await asyncio.to_thread(self._refresh_movements_subprocess, target_transactions)
                events.append({"event": "movements_exported", "path": str(target_transactions)})
            except Exception as exc:  # noqa: BLE001
                events.append({"event": "movements_export_failed", "error": str(exc)})

        report = await asyncio.to_thread(
            trade_republic_report_loop.build_report,
            portfolio_path=self.portfolio_path,
            transactions_path=target_transactions,
            isin_map_path=self.isin_map_path,
            fetch_yahoo=True,
            fetch_alpaca=True,
            reconstruct_positions_from_transactions=False,
        )
        await asyncio.to_thread(trade_republic_report_loop.atomic_write_report, report, target_output)
        summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
        events.append(
            {
                "event": "report_written",
                "path": str(target_output),
                "report_timestamp": summary.get("report_timestamp"),
                "total_current_value": summary.get("total_current_value"),
                "total_unrealized_pl": summary.get("total_unrealized_pl"),
            }
        )
        return {
            "holding_count": int(summary.get("holding_count") or len(report.get("holdings") or [])),
            "total_current_value": float(summary.get("total_current_value") or 0.0),
            "elapsed_seconds": round(time.time() - started, 2),
            "session": "persistent_pytr_websocket",
            "events": events,
        }

    def _refresh_movements_subprocess(self, target_transactions: Path) -> None:
        args = argparse.Namespace(
            transactions=target_transactions,
            pytr_timeout_seconds=self.timeout_seconds,
        )
        status: dict[str, Any] = {"events": []}
        trade_republic_report_loop.refresh_movements(args, status)

    async def _get_tr(self) -> Any:
        if self._tr is not None:
            return self._tr
        trade_republic_report_loop.load_env_file()
        from pytr.account import login

        self._tr = await asyncio.to_thread(login, store_credentials=True, waf_token="playwright")
        return self._tr

    async def _close_tr(self) -> None:
        if self._tr is None:
            return
        try:
            await self._tr.close()
        finally:
            self._tr = None

    async def _export_portfolio_csv(self, output: Path) -> None:
        tr = await self._get_tr()
        rows = await self._read_portfolio_rows(tr)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(["Name", "ISIN", "quantity", "price", "avgCost", "netValue"])
            for row in sorted(rows, key=lambda item: Decimal(str(item.get("netValue") or "0")), reverse=True):
                writer.writerow(
                    [
                        row.get("name") or row.get("instrumentId"),
                        row.get("instrumentId"),
                        decimal_text(row.get("netSize"), precision=6),
                        decimal_text(row.get("price"), precision=4),
                        decimal_text(row.get("averageBuyIn"), precision=4),
                        decimal_text(row.get("netValue"), precision=2),
                    ]
                )

    async def _read_portfolio_rows(self, tr: Any) -> list[dict[str, Any]]:
        subscription_id = await tr.compact_portfolio()
        response_subscription_id, subscription, response = await tr.recv()
        if response_subscription_id != subscription_id:
            raise RuntimeError(f"unexpected subscription response: {subscription}")
        await tr.unsubscribe(subscription_id)
        positions = compact_portfolio_positions(response)
        if not positions:
            raise RuntimeError("Trade Republic compact portfolio returned no positions.")

        instrument_subscriptions: dict[str, dict[str, Any]] = {}
        for position in positions:
            isin = position.get("instrumentId")
            if not isin:
                continue
            sub_id = await tr.instrument_details(isin)
            instrument_subscriptions[sub_id] = position
        while instrument_subscriptions:
            sub_id, subscription, response = await tr.recv()
            if subscription.get("type") != "instrument":
                continue
            await tr.unsubscribe(sub_id)
            position = instrument_subscriptions.pop(sub_id, None)
            if position is None:
                continue
            position["name"] = response.get("shortName") or position.get("name") or position.get("instrumentId")
            position["exchangeIds"] = response.get("exchangeIds") or []

        ticker_subscriptions: dict[str, dict[str, Any]] = {}
        for position in positions:
            exchange_ids = position.get("exchangeIds") or []
            if not exchange_ids:
                continue
            sub_id = await tr.ticker(position["instrumentId"], exchange=exchange_ids[0])
            ticker_subscriptions[sub_id] = position
        deadline = time.time() + min(20, self.timeout_seconds)
        while ticker_subscriptions and time.time() < deadline:
            try:
                sub_id, subscription, response = await asyncio.wait_for(tr.recv(), timeout=5)
            except asyncio.TimeoutError:
                break
            if subscription.get("type") != "ticker":
                continue
            await tr.unsubscribe(sub_id)
            position = ticker_subscriptions.pop(sub_id, None)
            if position is None:
                continue
            last = response.get("last") if isinstance(response, dict) else None
            pre = response.get("pre") if isinstance(response, dict) else None
            price = (last or pre or {}).get("price")
            if price is None:
                continue
            position["price"] = str(price)
            if "netSize" not in position:
                position["netSize"] = "0"
            if "averageBuyIn" not in position or position.get("averageBuyIn") is None:
                position["averageBuyIn"] = position["price"]
            position["netValue"] = str(
                (Decimal(str(position["price"])) * Decimal(str(position["netSize"]))).quantize(
                    Decimal("0.01"),
                    rounding=ROUND_HALF_UP,
                )
            )

        return [row for row in positions if row.get("price") is not None and row.get("netValue") is not None]


def compact_portfolio_positions(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, dict) and isinstance(response.get("positions"), list):
        return [dict(row) for row in response["positions"] if isinstance(row, dict)]
    rows: list[dict[str, Any]] = []
    if isinstance(response, dict) and isinstance(response.get("categories"), list):
        for category in response.get("categories") or []:
            if not isinstance(category, dict):
                continue
            for item in category.get("positions") or []:
                if not isinstance(item, dict):
                    continue
                row = dict(item)
                row.setdefault("instrumentId", row.get("isin"))
                rows.append(row)
    return rows


def decimal_text(value: Any, *, precision: int) -> str:
    if value is None:
        return ""
    amount = Decimal(str(value)).quantize(Decimal("1." + ("0" * precision)), rounding=ROUND_HALF_UP)
    text = f"{amount:.{precision}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def refresh_snaptrade_once(
    *,
    output: Path,
    include_activities: bool = True,
    activities_days: int = 365,
    include_orders: bool = True,
    orders_days: int = 90,
    include_balance_history: bool = True,
) -> None:
    print("Starting read-only SnapTrade snapshot refresh...", flush=True)
    snaptrade_readonly_cli.load_env_file()
    args = argparse.Namespace(
        command="snapshot",
        client_id=None,
        consumer_key=None,
        user_id=None,
        user_secret=None,
        output=output,
        format="json",
        include_orders=include_orders,
        include_activities=include_activities,
        include_balance_history=include_balance_history,
        activities_days=max(1, int(activities_days)),
        orders_days=max(1, min(90, int(orders_days))),
    )
    payload = snaptrade_readonly_cli.run_command(args)
    if payload.get("error"):
        raise RuntimeError(json.dumps(payload.get("error"), ensure_ascii=True, default=str))
    snaptrade_readonly_cli.write_payload(payload, output, "json")


def refresh_snaptrade_candidate(
    *,
    output: Path,
    activities_days: int,
) -> dict[str, Any]:
    candidate = output.with_name(output.stem + ".manual_refresh" + output.suffix)
    attempts = [
        {
            "include_activities": True,
            "activities_days": activities_days,
            "include_orders": True,
            "orders_days": 90,
            "include_balance_history": True,
            "mode": "full",
        },
        {
            "include_activities": False,
            "activities_days": 1,
            "include_orders": True,
            "orders_days": 30,
            "include_balance_history": True,
            "mode": "no_activities",
        },
        {
            "include_activities": False,
            "activities_days": 1,
            "include_orders": False,
            "orders_days": 1,
            "include_balance_history": False,
            "mode": "positions_only",
        },
    ]
    errors: list[dict[str, str]] = []
    try:
        for attempt in attempts:
            try:
                refresh_snaptrade_once(output=candidate, **{key: value for key, value in attempt.items() if key != "mode"})
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                accounts = payload.get("accounts") if isinstance(payload.get("accounts"), list) else []
                if not accounts:
                    raise RuntimeError("SnapTrade refresh produced no accounts.")
                output.parent.mkdir(parents=True, exist_ok=True)
                candidate.replace(output)
                return {
                    "mode": attempt["mode"],
                    "account_count": len(accounts),
                    "fetched_at": payload.get("fetched_at"),
                }
            except Exception as exc:  # noqa: BLE001
                errors.append({"mode": str(attempt["mode"]), "error_type": exc.__class__.__name__, "error": str(exc)})
        raise RuntimeError(json.dumps({"message": "SnapTrade refresh failed; kept last good snapshot.", "attempts": errors}, ensure_ascii=True))
    finally:
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass


def refresh_scenario_history_once(
    *,
    trade_republic_report: Path,
    trade_republic_transactions: Path,
    snaptrade_snapshot: Path,
    portfolio_history: Path,
    scenario_history: Path,
    benchmark_history: Path,
) -> None:
    print("Refreshing scenario historical prices from Yahoo Finance...", flush=True)
    state = build_unified_portfolio_state(
        trade_republic_report=trade_republic_report,
        trade_republic_transactions=trade_republic_transactions,
        snaptrade_snapshot=snaptrade_snapshot,
        portfolio_history_path=portfolio_history,
        scenario_history_path=scenario_history,
        benchmark_history_path=benchmark_history,
        persist_history=False,
    )
    payload = refresh_scenario_history_cache(state.get("holdings") or [], scenario_history)
    scenario_count = len(payload.get("scenarios") or {})
    symbol_count = len({row.get("ticker") or row.get("broker_symbol") for row in state.get("holdings") or [] if row.get("ticker") or row.get("broker_symbol")})
    print(f"Scenario history refreshed: {scenario_history} ({scenario_count} scenarios, {symbol_count} symbols)", flush=True)


def refresh_benchmark_history_once(*, benchmark_symbol: str, benchmark_history: Path, portfolio_history: Path) -> None:
    print(f"Refreshing benchmark history for {benchmark_symbol} from Yahoo Finance...", flush=True)
    start = None
    rows = []
    if portfolio_history.exists():
        try:
            rows = [
                json.loads(line)
                for line in portfolio_history.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, json.JSONDecodeError):
            rows = []
    dates = sorted(str(row.get("date") or str(row.get("timestamp") or "")[:10]) for row in rows if isinstance(row, dict) and (row.get("date") or row.get("timestamp")))
    if dates:
        start = dates[0]
    payload = refresh_benchmark_history_cache(
        symbol=benchmark_symbol,
        name=DEFAULT_BENCHMARK_NAME if benchmark_symbol == DEFAULT_BENCHMARK_SYMBOL else benchmark_symbol,
        path=benchmark_history,
        start=start,
    )
    print(f"Benchmark history refreshed: {benchmark_history} ({len(payload.get('prices') or [])} rows)", flush=True)


def refresh_classifications_once(
    *,
    trade_republic_report: Path,
    trade_republic_transactions: Path,
    snaptrade_snapshot: Path,
    portfolio_history: Path,
    scenario_history: Path,
    benchmark_history: Path,
    classification_cache: Path,
    classification_model: str | None,
) -> None:
    print("Refreshing asset classifications with cache/LLM fallback...", flush=True)
    trade_republic_report_loop.load_env_file()
    state = build_unified_portfolio_state(
        trade_republic_report=trade_republic_report,
        trade_republic_transactions=trade_republic_transactions,
        snaptrade_snapshot=snaptrade_snapshot,
        portfolio_history_path=portfolio_history,
        scenario_history_path=scenario_history,
        benchmark_history_path=benchmark_history,
        classification_cache_path=classification_cache,
        persist_history=False,
    )
    payload = refresh_asset_classification_cache(state.get("holdings") or [], classification_cache, model=classification_model)
    print(f"Asset classifications refreshed: {classification_cache} ({len(payload.get('classifications') or {})} cached assets)", flush=True)


def refresh_asset_price_history_once(
    *,
    trade_republic_report: Path,
    trade_republic_transactions: Path,
    snaptrade_snapshot: Path,
    portfolio_history: Path,
    scenario_history: Path,
    benchmark_history: Path,
    classification_cache: Path,
    asset_price_history: Path,
    lookback_days: int,
) -> None:
    print("Refreshing asset price history from Yahoo Finance...", flush=True)
    state = build_unified_portfolio_state(
        trade_republic_report=trade_republic_report,
        trade_republic_transactions=trade_republic_transactions,
        snaptrade_snapshot=snaptrade_snapshot,
        portfolio_history_path=portfolio_history,
        scenario_history_path=scenario_history,
        benchmark_history_path=benchmark_history,
        classification_cache_path=classification_cache,
        asset_price_history_path=asset_price_history,
        persist_history=False,
    )
    payload = refresh_asset_price_history_cache(state.get("holdings") or [], asset_price_history, lookback_days=lookback_days)
    prices = payload.get("prices") if isinstance(payload.get("prices"), dict) else {}
    errors = payload.get("errors") if isinstance(payload.get("errors"), dict) else {}
    print(f"Asset price history refreshed: {asset_price_history} ({len(prices)} symbols, {len(errors)} errors)", flush=True)


def holdings_for_dashboard_state(state: dict[str, Any], portfolio_key: str) -> list[dict[str, Any]]:
    portfolios = state.get("portfolio_summaries") if isinstance(state.get("portfolio_summaries"), list) else []
    selected = next((item for item in portfolios if isinstance(item, dict) and item.get("key") == portfolio_key), None)
    holdings = [item for item in state.get("holdings") or [] if isinstance(item, dict)]
    if selected and selected.get("kind") == "combined":
        included = {str(item.get("key")) for item in portfolios if isinstance(item, dict) and item.get("include_in_combined") and item.get("kind") != "combined"}
        return [holding for holding in holdings if str(holding.get("account_key")) in included]
    return [holding for holding in holdings if str(holding.get("account_key")) == portfolio_key]


def holding_current_price(holding: dict[str, Any]) -> float | None:
    price = holding.get("current_price")
    try:
        return float(price)
    except (TypeError, ValueError):
        pass
    try:
        value = float(holding.get("current_value"))
        quantity = abs(float(holding.get("quantity")))
    except (TypeError, ValueError):
        return None
    if quantity <= 0:
        return None
    return value / quantity


def load_env_file(path: Path | None) -> None:
    env_path = Path(path or ".env")
    if not env_path.exists() and not env_path.is_absolute():
        parent_env = Path.cwd().parent / env_path.name
        if parent_env.exists():
            env_path = parent_env
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or "=" not in clean:
            continue
        name, value = clean.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def portfolio_assistant_context(state: dict[str, Any], portfolio_key: str) -> dict[str, Any]:
    portfolios = [item for item in state.get("portfolio_summaries") or [] if isinstance(item, dict)]
    selected = next((item for item in portfolios if str(item.get("key")) == portfolio_key), None)
    if selected is None:
        selected = state.get("combined_portfolio") if isinstance(state.get("combined_portfolio"), dict) else {}
        portfolio_key = str(selected.get("key") or portfolio_key)
    holdings = holdings_for_dashboard_state(state, portfolio_key)
    holdings_sorted = sorted(holdings, key=lambda row: float(row.get("current_value") or 0), reverse=True)
    top_holdings = [
        {
            "ticker": row.get("ticker") or row.get("broker_symbol") or row.get("isin"),
            "name": row.get("name"),
            "quantity": row.get("quantity"),
            "current_price": row.get("current_price"),
            "average_cost": row.get("average_cost"),
            "current_value": row.get("current_value"),
            "cost_basis": row.get("cost_basis"),
            "unrealized_pl": row.get("unrealized_pl"),
            "unrealized_pl_pct": row.get("unrealized_pl_pct"),
            "today_gain": row.get("today_gain"),
            "today_gain_pct": row.get("today_gain_pct"),
            "currency": row.get("currency"),
            "broker": row.get("broker") or row.get("institution"),
            "classification": row.get("classification"),
        }
        for row in holdings_sorted[:35]
    ]
    total_value = float(selected.get("total_worth") or 0)
    allocation = []
    for row in holdings_sorted[:20]:
        value = float(row.get("current_value") or 0)
        allocation.append(
            {
                "ticker": row.get("ticker") or row.get("broker_symbol") or row.get("isin"),
                "name": row.get("name"),
                "weight_pct": (value / total_value * 100.0) if total_value else 0.0,
                "value": value,
                "asset_class": (row.get("classification") or {}).get("asset_class") if isinstance(row.get("classification"), dict) else row.get("asset_type"),
                "sector": (row.get("classification") or {}).get("sector") if isinstance(row.get("classification"), dict) else None,
                "geography": (row.get("classification") or {}).get("geography") if isinstance(row.get("classification"), dict) else None,
            }
        )
    rebalancing = state.get("rebalancing") if isinstance(state.get("rebalancing"), dict) else {}
    alerts = state.get("price_alerts") if isinstance(state.get("price_alerts"), dict) else {}
    return {
        "selected_portfolio": selected,
        "portfolio_key": portfolio_key,
        "goal_plan": state.get("goal_plan"),
        "source_timestamps": state.get("source_timestamps"),
        "holdings_count": len(holdings),
        "top_holdings": top_holdings,
        "allocation": allocation,
        "rebalancing": rebalancing.get(portfolio_key),
        "alerts_summary": alerts.get("summary") if isinstance(alerts, dict) else None,
        "history_points": [row for row in state.get("portfolio_history") or [] if isinstance(row, dict) and row.get("portfolio_key") == portfolio_key][-60:],
        "data_limits": [
            "Use only this provided local dashboard state.",
            "If a field is missing, say it is missing instead of guessing.",
            "This assistant is read-only and cannot place broker orders.",
        ],
    }


ASSISTANT_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "name": "portfolio_assistant_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "title": {"type": "string"},
            "answer": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "numbers_used": {"type": "array", "items": {"type": "string"}},
            "risks_or_limitations": {"type": "array", "items": {"type": "string"}},
            "follow_up_questions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "answer", "key_points", "numbers_used", "risks_or_limitations", "follow_up_questions"],
    },
}


def ask_portfolio_assistant(
    *,
    state: dict[str, Any],
    portfolio_key: str,
    question: str,
    provider: str | None,
    model: str | None,
    env_file: Path | None,
    timeout_seconds: float,
    handler: Any | None = None,
) -> dict[str, Any]:
    load_env_file(env_file)
    profile = resolve_llm_client_profile(provider=provider, model=model)
    context = portfolio_assistant_context(state, portfolio_key)
    system_message = (
        "You are a read-only AI portfolio assistant similar in scope to AllInvestView's AI assistant. "
        "Answer natural-language questions about holdings, returns, dividends, allocation, risk exposure, diversification gaps, "
        "benchmarks, alerts, and rebalancing using only the provided local normalized portfolio state. "
        "Write in very plain English for beginners, non-technical people, beginner investors, and people who are not investors. "
        "Avoid jargon. When you must use an investing term, explain it in one short sentence. Use simple examples and short sentences. "
        "Do not assume the user understands cost basis, unrealized profit/loss, realized profit/loss, yield, allocation, volatility, beta, alpha, or benchmark. "
        "Do not invent missing data. Do not recommend placing trades as instructions; "
        "you may explain what the data suggests and what the user may want to review. Never claim you executed an action."
    )
    user_message = (
        "User question:\n"
        f"{question}\n\n"
        "Local portfolio state snapshot, JSON:\n"
        f"{json.dumps(context, ensure_ascii=True, default=str)}"
    )
    payload = response_payload(
        model=profile.model,
        system_message=system_message,
        user_message=user_message,
        json_schema=ASSISTANT_JSON_SCHEMA,
        reasoning_effort="low",
        tools=[],
    )
    openai_client = None
    if profile.provider == "openai":
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMProviderNotConfigured("Install the openai package to use provider `openai`.") from exc
        openai_client = OpenAI(timeout=timeout_seconds)
    llm_handler = handler or default_llm_handler(openai_client=openai_client)
    result = llm_handler.predict(
        LLMRequest(
            provider=profile.provider,
            model=profile.model,
            payload=payload,
            usage_context={
                "purpose": "unified_portfolio_ai_assistant",
                "portfolio_key": portfolio_key,
                "question_preview": question[:160],
            },
            timeout_seconds=timeout_seconds,
        )
    )
    parsed = dict(result.parsed)
    parsed["provider"] = result.provider
    parsed["model"] = result.model
    parsed["portfolio_key"] = portfolio_key
    return parsed


def run_server(
    *,
    trade_republic_report: Path,
    trade_republic_transactions: Path,
    snaptrade_snapshot: Path,
    portfolio_history: Path,
    scenario_history: Path,
    benchmark_history: Path,
    classification_cache: Path,
    asset_price_history: Path,
    price_alerts: Path,
    alert_events: Path,
    allocation_targets: Path,
    weekly_cleanup_review: Path,
    assistant_llm_provider: str | None,
    assistant_llm_model: str | None,
    assistant_llm_env_file: Path | None,
    assistant_llm_timeout_seconds: float,
    manual_refresh_trade_republic: bool,
    manual_refresh_snaptrade: bool,
    manual_refresh_snaptrade_activities_days: int,
    trade_republic_timeout_seconds: int,
    refresh_trade_republic_movements: bool,
    host: str,
    port: int,
    refresh_seconds: int,
) -> None:
    trade_republic_worker = TradeRepublicSessionWorker(
        portfolio_path=trade_republic_report_loop.DEFAULT_PORTFOLIO,
        transactions_path=trade_republic_transactions,
        isin_map_path=trade_republic_report_loop.DEFAULT_ISIN_MAP,
        output_path=trade_republic_report,
        timeout_seconds=trade_republic_timeout_seconds,
        refresh_movements=refresh_trade_republic_movements,
    )

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_html(dashboard_html(refresh_seconds))
                return
            if parsed.path == "/api/state":
                params = parse_qs(parsed.query)
                tr_path = Path(params.get("trade_republic_report", [str(trade_republic_report)])[0])
                tr_transactions_path = Path(params.get("trade_republic_transactions", [str(trade_republic_transactions)])[0])
                snap_path = Path(params.get("snaptrade_snapshot", [str(snaptrade_snapshot)])[0])
                history_path = Path(params.get("portfolio_history", [str(portfolio_history)])[0])
                scenario_history_path = Path(params.get("scenario_history", [str(scenario_history)])[0])
                benchmark_history_path = Path(params.get("benchmark_history", [str(benchmark_history)])[0])
                classification_cache_path = Path(params.get("classification_cache", [str(classification_cache)])[0])
                asset_price_history_path = Path(params.get("asset_price_history", [str(asset_price_history)])[0])
                price_alerts_path = Path(params.get("price_alerts", [str(price_alerts)])[0])
                alert_events_path = Path(params.get("alert_events", [str(alert_events)])[0])
                allocation_targets_path = Path(params.get("allocation_targets", [str(allocation_targets)])[0])
                weekly_cleanup_review_path = Path(params.get("weekly_cleanup_review", [str(weekly_cleanup_review)])[0])
                self._send_json(
                    build_unified_portfolio_state(
                        trade_republic_report=tr_path,
                        trade_republic_transactions=tr_transactions_path,
                        snaptrade_snapshot=snap_path,
                        portfolio_history_path=history_path,
                        scenario_history_path=scenario_history_path,
                        benchmark_history_path=benchmark_history_path,
                        classification_cache_path=classification_cache_path,
                        asset_price_history_path=asset_price_history_path,
                        price_alerts_path=price_alerts_path,
                        alert_events_path=alert_events_path,
                        allocation_targets_path=allocation_targets_path,
                        weekly_cleanup_review_path=weekly_cleanup_review_path,
                        persist_history=True,
                    )
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                payload = self._read_json_body()
            except ValueError as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            if parsed.path == "/api/alerts":
                symbol = str(payload.get("symbol") or "").strip().upper()
                target = payload.get("target_price")
                if not symbol:
                    self.send_error(HTTPStatus.BAD_REQUEST, "symbol is required")
                    return
                try:
                    target_price = float(target)
                except (TypeError, ValueError):
                    self.send_error(HTTPStatus.BAD_REQUEST, "target_price must be a number")
                    return
                alert = create_price_alert(
                    price_alerts,
                    portfolio_key=str(payload.get("portfolio_key") or "combined:live"),
                    symbol=symbol,
                    name=payload.get("name") or symbol,
                    target_price=target_price,
                    direction=str(payload.get("direction") or "below"),
                    basis_price=payload.get("basis_price"),
                    threshold_pct=payload.get("threshold_pct"),
                    scope=str(payload.get("scope") or "single"),
                    currency=payload.get("currency"),
                )
                self._send_json({"ok": True, "alert": alert})
                return
            if parsed.path == "/api/alerts/bulk":
                state = build_unified_portfolio_state(
                    trade_republic_report=trade_republic_report,
                    trade_republic_transactions=trade_republic_transactions,
                    snaptrade_snapshot=snaptrade_snapshot,
                    portfolio_history_path=portfolio_history,
                    scenario_history_path=scenario_history,
                    benchmark_history_path=benchmark_history,
                    classification_cache_path=classification_cache,
                    asset_price_history_path=asset_price_history,
                    price_alerts_path=price_alerts,
                    alert_events_path=alert_events,
                    allocation_targets_path=allocation_targets,
                    weekly_cleanup_review_path=weekly_cleanup_review,
                    persist_history=False,
                )
                portfolio_key = str(payload.get("portfolio_key") or "combined:live")
                direction = str(payload.get("direction") or "below")
                threshold_pct = float(payload.get("threshold_pct") or 10.0)
                created = []
                for holding in holdings_for_dashboard_state(state, portfolio_key):
                    price = holding_current_price(holding)
                    symbol = str(holding.get("ticker") or holding.get("broker_symbol") or holding.get("isin") or "").upper()
                    if not symbol or price is None:
                        continue
                    multiplier = 1 + threshold_pct / 100.0 if direction == "above" else 1 - threshold_pct / 100.0
                    created.append(
                        create_price_alert(
                            price_alerts,
                            portfolio_key=portfolio_key,
                            symbol=symbol,
                            name=holding.get("name") or symbol,
                            target_price=price * multiplier,
                            direction=direction,
                            basis_price=price,
                            threshold_pct=threshold_pct,
                            scope=str(payload.get("scope") or "bulk"),
                            currency=holding.get("currency"),
                        )
                    )
                self._send_json({"ok": True, "created_count": len(created), "alerts": created})
                return
            if parsed.path.startswith("/api/alerts/"):
                parts = [part for part in parsed.path.split("/") if part]
                if len(parts) == 3:
                    status = str(payload.get("status") or "paused")
                    updated = update_price_alert_status(price_alerts, parts[2], status)
                    self._send_json({"ok": updated is not None, "alert": updated})
                    return
            if parsed.path == "/api/allocation-targets":
                portfolio_key = str(payload.get("portfolio_key") or "combined:live")
                rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
                saved = save_allocation_targets(allocation_targets, portfolio_key, rows)
                self._send_json({"ok": True, "target": saved})
                return
            if parsed.path == "/api/refresh":
                events: list[dict[str, Any]] = []
                if manual_refresh_trade_republic:
                    try:
                        result = trade_republic_worker.refresh_report(output=trade_republic_report, transactions=trade_republic_transactions)
                        events.append({"source": "trade_republic", "status": "refreshed", "path": str(trade_republic_report), **result})
                    except Exception as exc:
                        events.append(
                            {
                                "source": "trade_republic",
                                "status": "failed",
                                "path": str(trade_republic_report),
                                "error": str(exc),
                                "error_type": exc.__class__.__name__,
                            }
                        )
                else:
                    events.append({"source": "trade_republic", "status": "skipped", "reason": "manual refresh disabled"})
                if manual_refresh_snaptrade:
                    try:
                        result = refresh_snaptrade_candidate(
                            output=snaptrade_snapshot,
                            activities_days=manual_refresh_snaptrade_activities_days,
                        )
                        events.append({"source": "snaptrade", "status": "refreshed", "path": str(snaptrade_snapshot), **result})
                    except Exception as exc:
                        events.append(
                            {
                                "source": "snaptrade",
                                "status": "stale",
                                "path": str(snaptrade_snapshot),
                                "error": str(exc),
                                "error_type": exc.__class__.__name__,
                                "message": "Kept last good SnapTrade snapshot.",
                            }
                        )
                else:
                    events.append({"source": "snaptrade", "status": "skipped", "reason": "manual refresh disabled"})
                state = build_unified_portfolio_state(
                    trade_republic_report=trade_republic_report,
                    trade_republic_transactions=trade_republic_transactions,
                    snaptrade_snapshot=snaptrade_snapshot,
                    portfolio_history_path=portfolio_history,
                    scenario_history_path=scenario_history,
                    benchmark_history_path=benchmark_history,
                    classification_cache_path=classification_cache,
                    asset_price_history_path=asset_price_history,
                    price_alerts_path=price_alerts,
                    alert_events_path=alert_events,
                    allocation_targets_path=allocation_targets,
                    weekly_cleanup_review_path=weekly_cleanup_review,
                    persist_history=True,
                )
                ok = any(event.get("status") == "refreshed" for event in events)
                self._send_json({"ok": ok, "events": events, "state": state})
                return
            if parsed.path == "/api/assistant":
                question = str(payload.get("question") or "").strip()
                if not question:
                    self.send_error(HTTPStatus.BAD_REQUEST, "question is required")
                    return
                portfolio_key = str(payload.get("portfolio_key") or "combined:live")
                state = build_unified_portfolio_state(
                    trade_republic_report=trade_republic_report,
                    trade_republic_transactions=trade_republic_transactions,
                    snaptrade_snapshot=snaptrade_snapshot,
                    portfolio_history_path=portfolio_history,
                    scenario_history_path=scenario_history,
                    benchmark_history_path=benchmark_history,
                    classification_cache_path=classification_cache,
                    asset_price_history_path=asset_price_history,
                    price_alerts_path=price_alerts,
                    alert_events_path=alert_events,
                    allocation_targets_path=allocation_targets,
                    weekly_cleanup_review_path=weekly_cleanup_review,
                    persist_history=False,
                )
                try:
                    answer = ask_portfolio_assistant(
                        state=state,
                        portfolio_key=portfolio_key,
                        question=question,
                        provider=assistant_llm_provider,
                        model=assistant_llm_model,
                        env_file=assistant_llm_env_file,
                        timeout_seconds=assistant_llm_timeout_seconds,
                    )
                except Exception as exc:
                    self._send_json({"ok": False, "error": str(exc), "error_type": exc.__class__.__name__})
                    return
                self._send_json({"ok": True, "answer": answer})
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True, default=str).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or "0")
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON body") from exc
            if not isinstance(parsed, dict):
                raise ValueError("JSON body must be an object")
            return parsed

        def _send_html(self, html: str) -> None:
            body = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer((host, int(port)), Handler)
    print(f"Unified portfolio dashboard at http://{host}:{port}")
    print(f"Trade Republic report: {trade_republic_report}")
    print(f"Trade Republic transactions: {trade_republic_transactions}")
    print(f"SnapTrade snapshot: {snaptrade_snapshot}")
    print(f"Portfolio history: {portfolio_history}")
    print(f"Scenario history: {scenario_history}")
    print(f"Benchmark history: {benchmark_history}")
    print(f"Classification cache: {classification_cache}")
    print(f"Asset price history: {asset_price_history}")
    print(f"Price alerts: {price_alerts}")
    print(f"Alert events: {alert_events}")
    print(f"Allocation targets: {allocation_targets}")
    print(f"Weekly cleanup review: {weekly_cleanup_review}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def dashboard_html(refresh_seconds: int) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Unified Portfolio</title>
  <style>
    :root {{
      --bg: #090d13;
      --panel: #171c23;
      --panel-2: #1d232c;
      --text: #d9dee7;
      --muted: #9aa3b2;
      --line: #2b323d;
      --blue: #4d83ff;
      --blue-soft: #10244f;
      --green: #42c764;
      --red: #ef5350;
      --shadow: 0 12px 30px rgba(0, 0, 0, .28);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      padding-bottom: 62px;
      background: radial-gradient(circle at 12% -10%, rgba(77,131,255,.12), transparent 28%), var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .shell {{ max-width: 2048px; margin: 0 auto; padding: 26px 30px 44px; }}
    .metric-row {{
      display: grid;
      grid-template-columns: repeat(4, minmax(240px, 1fr));
      gap: 34px;
      margin-bottom: 50px;
    }}
    .metric-card, .portfolio-card, .placeholder, .dashboard-panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: var(--shadow);
    }}
    .metric-card {{ min-height: 158px; padding: 30px 28px; }}
    .metric-label {{ display: flex; align-items: center; gap: 12px; font-size: 18px; font-weight: 760; margin-bottom: 14px; }}
    .metric-icon {{ color: var(--blue); width: 22px; text-align: center; }}
    .metric-value {{ font-size: 28px; font-weight: 780; line-height: 1.15; margin-bottom: 16px; }}
    .metric-sub {{ color: var(--muted); font-size: 17px; line-height: 1.45; }}
    .goal-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); margin: -22px 0 26px; overflow: hidden; }}
    .goal-header {{ min-height: 68px; padding: 18px 24px; border-bottom: 1px solid var(--line); display: flex; justify-content: space-between; gap: 16px; align-items: center; }}
    .goal-title {{ margin: 0; font-size: 20px; font-weight: 800; }}
    .goal-status {{ border: 1px solid var(--line); border-radius: 999px; padding: 6px 12px; color: var(--muted); font-size: 13px; font-weight: 750; text-transform: uppercase; }}
    .goal-grid {{ display: grid; grid-template-columns: repeat(5, minmax(160px, 1fr)); gap: 0; }}
    .goal-cell {{ min-height: 118px; padding: 18px 20px; border-right: 1px solid var(--line); display: grid; align-content: center; gap: 8px; }}
    .goal-cell:last-child {{ border-right: 0; }}
    .goal-label {{ color: var(--muted); font-size: 13px; font-weight: 760; text-transform: uppercase; }}
    .goal-value {{ font-size: 23px; font-weight: 820; line-height: 1.2; }}
    .goal-note {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    .goal-action-box {{ display: grid; grid-template-columns: minmax(260px, .8fr) minmax(320px, 1.2fr); gap: 18px; padding: 20px 24px 24px; border-top: 1px solid var(--line); }}
    .goal-action-card {{ border: 1px solid var(--line); border-radius: 8px; background: #111821; padding: 16px; }}
    .goal-action-card.warning {{ border-color: rgba(239,83,80,.45); background: rgba(239,83,80,.07); }}
    .goal-action-title {{ font-size: 16px; font-weight: 820; margin-bottom: 8px; }}
    .goal-action-list {{ margin: 10px 0 0; padding-left: 18px; color: var(--text); font-size: 14px; line-height: 1.55; }}
    .goal-action-list li {{ margin-bottom: 6px; }}
    .cleanup-table-wrap {{ overflow-x: auto; margin-top: 10px; }}
    .cleanup-table {{ width: 100%; min-width: 620px; border-collapse: collapse; }}
    .cleanup-table th, .cleanup-table td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); font-size: 13px; text-align: left; vertical-align: top; }}
    .cleanup-table th {{ color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: .05em; }}
    .cleanup-symbol {{ font-weight: 840; color: var(--text); }}
    .cleanup-action {{ color: var(--muted); line-height: 1.35; }}
    .action-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 112px; border: 1px solid var(--line); border-radius: 999px; padding: 6px 10px; font-size: 12px; font-weight: 780; text-transform: uppercase; color: var(--text); background: #151b23; }}
    .action-pill.core {{ border-color: rgba(66,199,100,.45); color: var(--green); }}
    .action-pill.opportunity {{ border-color: rgba(77,131,255,.55); color: var(--blue); }}
    .action-pill.cleanup, .action-pill.review {{ border-color: rgba(239,83,80,.55); color: var(--red); }}
    .action-reasons {{ color: var(--muted); margin-top: 5px; font-size: 12px; line-height: 1.35; max-width: 360px; }}
    .positive {{ color: var(--green); }}
    .negative {{ color: var(--red); }}
    .info {{ color: var(--muted); font-size: .78em; margin-left: 4px; }}
    .tabs {{
      display: flex;
      align-items: stretch;
      gap: 46px;
      border-bottom: 1px solid var(--line);
      margin: 0 -18px 22px;
      padding-left: 18px;
    }}
    .tab {{
      min-width: 198px;
      height: 90px;
      border: 0;
      border-radius: 0;
      border-left: 5px solid transparent;
      border-bottom: 4px solid transparent;
      background: transparent;
      color: var(--muted);
      font: inherit;
      font-size: 21px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
    }}
    .tab.active {{
      color: var(--text);
      background: var(--panel);
      border-left-color: #174081;
      border-bottom-color: var(--blue);
      font-weight: 780;
    }}
    .portfolio-grid {{
      display: grid;
      grid-template-columns: repeat(5, minmax(260px, 305px));
      gap: 34px;
      padding-top: 0;
    }}
    .portfolio-card {{
      min-height: 326px;
      padding: 30px 28px;
      color: var(--text);
      text-align: left;
      cursor: pointer;
      position: relative;
      overflow: hidden;
    }}
    .portfolio-card.active {{ border-color: var(--blue); box-shadow: inset 0 0 0 1px var(--blue), var(--shadow); }}
    .portfolio-card.paper {{ border-style: dashed; }}
    .portfolio-title {{ display: flex; align-items: flex-start; gap: 12px; min-height: 56px; font-size: 19px; font-weight: 760; line-height: 1.38; }}
    .portfolio-edit {{ position: absolute; top: 34px; right: 28px; border: 2px solid var(--blue); border-radius: 10px; color: var(--blue); width: 70px; height: 56px; display: grid; place-items: center; font-size: 24px; }}
    .portfolio-worth {{ font-size: 27px; font-weight: 790; margin: 14px 0 16px; }}
    .portfolio-lines {{ font-size: 17px; line-height: 1.55; }}
    .portfolio-lines strong {{ color: var(--text); }}
    .portfolio-muted {{ color: var(--muted); margin-top: 14px; font-size: 16px; line-height: 1.55; }}
    .portfolio-note {{ color: var(--muted); margin-top: 10px; font-size: 14px; line-height: 1.45; }}
    .dashboard-stack {{ display: grid; gap: 28px; }}
    .performance-strip {{ background: var(--panel); border: 1px solid #c5c9d0; border-radius: 4px; overflow-x: auto; }}
    .performance-grid {{ min-width: 1560px; display: grid; grid-template-columns: repeat(11, minmax(120px, 1fr)); }}
    .perf-cell {{ min-height: 100px; padding: 20px 18px; border-right: 1px solid var(--line); display: grid; align-content: center; gap: 8px; text-align: center; }}
    .perf-cell:last-child {{ border-right: 0; }}
    .perf-head {{ border-bottom: 1px solid var(--line); color: var(--text); font-size: 17px; font-weight: 760; }}
    .perf-value {{ font-size: 17px; font-weight: 740; line-height: 1.4; }}
    .perf-sub {{ color: var(--muted); font-size: 14px; line-height: 1.35; }}
    .dashboard-main {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(360px, 480px); gap: 34px; align-items: start; }}
    .allocation-full {{ grid-column: 1 / -1; }}
    .dashboard-panel {{ min-height: 300px; overflow: hidden; }}
    .panel-header {{ min-height: 80px; padding: 22px 28px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
    .panel-title {{ margin: 0; font-size: 21px; font-weight: 760; }}
    .manage-link {{ color: var(--blue); font-size: 17px; font-weight: 650; }}
    .allocation-tabs {{ display: flex; gap: 32px; padding: 26px 30px 0; border-bottom: 1px solid var(--line); overflow-x: auto; }}
    .allocation-tab {{ color: var(--muted); padding: 0 18px 20px; border-bottom: 3px solid transparent; font-size: 17px; font-weight: 680; white-space: nowrap; background: transparent; border-left: 0; border-top: 0; border-right: 0; cursor: pointer; }}
    .allocation-tab.active {{ color: var(--blue); border-color: var(--blue); }}
    .allocation-body {{ display: grid; grid-template-columns: minmax(300px, 46%) minmax(360px, 1fr); gap: 42px; padding: 70px 46px 68px; align-items: center; }}
    .donut {{ width: min(340px, 100%); aspect-ratio: 1; border-radius: 50%; margin: 0 auto; position: relative; background: conic-gradient(#4d83ff 0 35%, #1f4d94 35% 70%, #102b59 70% 100%); }}
    .donut::after {{ content: ""; position: absolute; inset: 25%; border-radius: 50%; background: var(--panel); border: 2px solid #c8ced8; }}
    .allocation-list {{ display: grid; gap: 0; }}
    .allocation-row {{ display: grid; grid-template-columns: minmax(120px, 1fr) minmax(180px, 2fr) minmax(120px, .8fr); gap: 18px; align-items: center; padding: 20px 0; border-bottom: 1px solid var(--line); }}
    .allocation-name {{ font-size: 18px; font-weight: 760; }}
    .allocation-pct {{ color: var(--muted); font-size: 15px; margin-top: 6px; }}
    .allocation-bar {{ height: 24px; border-radius: 999px; background: #414a59; overflow: hidden; }}
    .allocation-fill {{ height: 100%; border-radius: 999px; background: var(--blue); }}
    .allocation-value {{ font-size: 16px; font-weight: 740; text-align: right; }}
    .stock-grid {{ display: grid; grid-template-columns: repeat(4, minmax(220px, 1fr)); gap: 16px; padding: 42px 30px 52px; }}
    .stock-card {{ min-height: 128px; border: 1px solid #151b23; border-radius: 10px; background: #151a21; padding: 16px 16px 14px; display: grid; gap: 12px; box-shadow: inset 0 0 0 1px rgba(255,255,255,.015); }}
    .stock-title {{ display: flex; align-items: center; gap: 12px; min-width: 0; font-size: 16px; font-weight: 690; }}
    .stock-avatar {{ width: 36px; height: 36px; border-radius: 999px; display: grid; place-items: center; flex: 0 0 auto; color: white; background: #284c8f; font-size: 13px; font-weight: 800; }}
    .stock-name {{ overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }}
    .stock-pct {{ font-size: 25px; font-weight: 780; }}
    .stock-bar {{ height: 10px; border-radius: 999px; background: #111821; overflow: hidden; }}
    .stock-fill {{ height: 100%; border-radius: 999px; background: var(--green); }}
    .dashboard-bottom {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0; }}
    .market-panel {{ min-height: 360px; border: 1px solid #c5c9d0; border-radius: 14px; background: var(--panel); overflow: hidden; }}
    .market-panel + .market-panel {{ margin-left: -1px; }}
    .market-header {{ min-height: 76px; padding: 18px 24px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; }}
    .market-title {{ margin: 0; font-size: 21px; font-weight: 780; text-align: center; width: 100%; }}
    .market-close {{ color: var(--text); font-size: 26px; font-weight: 800; margin-left: 12px; }}
    .market-list {{ max-height: 300px; overflow-y: auto; padding: 18px 24px; }}
    .market-row {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 18px; align-items: center; min-height: 104px; padding: 16px 0; border-bottom: 1px solid #c5c9d0; }}
    .market-row:last-child {{ border-bottom: 0; }}
    .market-name {{ display: flex; align-items: center; gap: 14px; min-width: 0; font-size: 19px; font-weight: 660; }}
    .market-name-text {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .market-values {{ text-align: right; font-size: 18px; line-height: 1.55; font-weight: 700; }}
    .transaction-row {{ grid-template-columns: minmax(0, 1.2fr) minmax(110px, .8fr) minmax(110px, .8fr); }}
    .badge {{ display: inline-flex; align-items: center; border-radius: 7px; padding: 5px 10px; background: var(--red); color: white; font-weight: 740; font-size: 15px; }}
    .heatmap-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; }}
    .heatmap-header {{ min-height: 62px; display: grid; place-items: center; border-bottom: 1px solid var(--line); font-weight: 780; font-size: 17px; }}
    .heatmap-subtitle {{ text-align: center; font-weight: 740; padding: 28px 20px 16px; }}
    .heatmap-grid {{ display: grid; grid-template-columns: repeat(12, minmax(0, 1fr)); grid-auto-rows: 74px; gap: 2px; padding: 0 34px 46px; }}
    .heatmap-cell {{ grid-column: span var(--span); min-height: 74px; background: var(--heat); border: 1px solid rgba(255,255,255,.55); display: grid; place-items: center; text-align: center; padding: 8px; color: white; overflow: hidden; }}
    .heatmap-label {{ max-width: 100%; font-size: 13px; line-height: 1.2; overflow: hidden; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }}
	    .heatmap-controls {{ display: flex; justify-content: center; align-items: center; gap: 18px; padding: 10px 20px 30px; color: var(--text); font-weight: 700; }}
	    .return-select {{ background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 5px; min-height: 42px; padding: 0 14px; font: inherit; font-weight: 500; }}
	    .reports-toolbar {{ min-height: 42px; border: 1px solid #123552; border-radius: 8px; background: #0d1a26; display: flex; align-items: center; gap: 8px; padding: 7px 12px; margin-bottom: 16px; overflow-x: auto; }}
	    .reports-label {{ display: inline-flex; align-items: center; gap: 7px; color: var(--muted); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; white-space: nowrap; }}
	    .report-chip {{ min-height: 28px; border: 1px solid #2d4054; border-radius: 6px; background: #17212c; color: var(--text); display: inline-flex; align-items: center; gap: 6px; padding: 0 11px; font-size: 13px; font-weight: 650; white-space: nowrap; }}
	    .reports-close {{ margin-left: auto; color: var(--muted); font-size: 18px; font-weight: 700; }}
	    .insights-report {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 18px; box-shadow: var(--shadow); }}
	    .insights-header {{ display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-bottom: 12px; }}
	    .insights-title {{ margin: 0; font-size: 18px; font-weight: 780; }}
	    .insights-range {{ color: var(--muted); font-weight: 650; margin-left: 6px; }}
	    .notes-pill {{ border: 1px solid var(--line); border-radius: 999px; color: var(--muted); font-size: 12px; padding: 6px 10px; white-space: nowrap; }}
	    .insight-card-grid {{ display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px; }}
	    .insight-card {{ min-height: 134px; border: 1px solid var(--line); border-radius: 10px; padding: 14px; background: #1a2028; display: grid; align-content: start; gap: 8px; }}
	    .insight-card.wide {{ grid-column: span 2; }}
	    .insight-card.full {{ grid-column: 1 / -1; }}
	    .insight-label {{ color: var(--muted); font-size: 13px; font-weight: 760; }}
	    .insight-empty-value {{ color: var(--text); font-size: 22px; font-weight: 780; }}
	    .insight-note {{ color: var(--muted); font-size: 13px; line-height: 1.45; border: 1px dashed #313b47; border-radius: 8px; padding: 8px 10px; }}
	    .insights-layout {{ display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(420px, .8fr); gap: 14px; margin-top: 12px; }}
	    .insight-section {{ border: 1px solid var(--line); border-radius: 10px; padding: 16px; background: #1a2028; }}
	    .insight-section-title {{ margin: 0 0 14px; font-size: 18px; font-weight: 780; }}
	    .insight-definition-list {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }}
	    .definition-item {{ border: 1px dashed #313b47; border-radius: 8px; padding: 10px; color: var(--muted); font-size: 13px; line-height: 1.45; }}
	    .definition-item strong {{ display: block; color: var(--text); font-size: 14px; margin-bottom: 4px; }}
	    .growth-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }}
	    .growth-header {{ min-height: 60px; padding: 16px 20px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 18px; }}
	    .growth-title {{ margin: 0; font-size: 16px; font-weight: 780; }}
	    .growth-actions {{ display: flex; align-items: center; gap: 12px; color: var(--muted); font-size: 13px; white-space: nowrap; }}
	    .portfolio-select {{ background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 6px; min-height: 30px; max-width: 260px; padding: 0 10px; font: inherit; font-size: 13px; }}
	    .toggle-button {{ width: 38px; height: 20px; border-radius: 999px; border: 1px solid #4b5563; background: #303846; position: relative; cursor: pointer; padding: 0; }}
	    .toggle-button::after {{ content: ""; position: absolute; width: 14px; height: 14px; left: 3px; top: 2px; border-radius: 50%; background: #f3f7fb; transition: left .12s ease; }}
	    .toggle-button.active {{ border-color: #275eaa; background: #2764bb; box-shadow: inset 0 0 0 1px rgba(66,199,100,.4); }}
	    .toggle-button.active::after {{ left: 19px; }}
	    .growth-body {{ padding: 18px 20px 20px; }}
	    .growth-summary {{ display: flex; align-items: center; gap: 10px; font-size: 15px; margin-bottom: 10px; }}
	    .growth-badge {{ border-radius: 7px; padding: 4px 8px; color: #fff; font-size: 12px; font-weight: 760; }}
	    .growth-badge.positive {{ background: var(--green); }}
	    .growth-badge.negative {{ background: var(--red); }}
	    .growth-ranges {{ display: flex; justify-content: flex-end; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
	    .growth-range {{ border: 1px solid transparent; background: transparent; color: var(--blue); font: inherit; font-size: 12px; cursor: pointer; padding: 4px 6px; border-radius: 5px; }}
	    .growth-range.active {{ border-color: var(--blue); background: rgba(77,131,255,.12); color: var(--text); }}
	    .growth-chart-wrap {{ min-height: 320px; position: relative; }}
	    .growth-chart {{ width: 100%; min-height: 320px; display: block; }}
	    .growth-grid {{ stroke: #29323d; stroke-width: 1; }}
	    .growth-axis {{ fill: var(--muted); font-size: 12px; }}
	    .growth-line-portfolio {{ fill: none; stroke: #2fb8c5; stroke-width: 3; }}
	    .growth-line-invested {{ fill: none; stroke: #8b3f91; stroke-width: 3; }}
	    .growth-hit {{ fill: transparent; stroke: transparent; cursor: crosshair; pointer-events: all; }}
	    .growth-tooltip {{ position: fixed; z-index: 20; pointer-events: none; min-width: 190px; padding: 9px 11px; border-radius: 8px; border: 1px solid var(--line); background: #080b10; color: var(--text); box-shadow: var(--shadow); font-size: 12px; line-height: 1.45; opacity: 0; transform: translate(-50%, calc(-100% - 14px)); transition: opacity .08s ease; }}
	    .growth-tooltip.visible {{ opacity: 1; }}
	    .growth-tooltip strong {{ display: block; font-size: 13px; margin-bottom: 3px; }}
	    .growth-legend {{ display: flex; justify-content: center; gap: 20px; color: var(--text); font-size: 12px; margin-top: -4px; }}
	    .legend-item {{ display: inline-flex; align-items: center; gap: 7px; }}
	    .legend-swatch {{ width: 38px; height: 12px; border: 2px solid currentColor; }}
	    .growth-note {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 12px; border: 1px dashed #313b47; border-radius: 8px; padding: 10px 12px; }}
	    .dividend-tabs {{ display: inline-flex; align-items: center; margin-bottom: 14px; border-radius: 8px; overflow: hidden; border: 1px solid #151b23; }}
	    .dividend-tab {{ border: 0; min-height: 44px; padding: 0 20px; background: #0c1118; color: var(--muted); font: inherit; font-size: 15px; font-weight: 760; cursor: pointer; }}
	    .dividend-tab.active {{ background: var(--blue); color: white; }}
	    .dividend-layout {{ display: grid; grid-template-columns: minmax(230px, 360px) minmax(0, 1fr); gap: 36px; align-items: start; }}
	    .dividend-summary {{ display: grid; gap: 28px; }}
	    .dividend-stat {{ min-height: 150px; padding: 28px 24px; display: grid; place-items: center; text-align: center; background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: var(--shadow); }}
	    .dividend-stat-label {{ display: flex; align-items: center; justify-content: center; gap: 10px; font-size: 18px; font-weight: 780; }}
	    .dividend-pill {{ display: inline-flex; align-items: center; justify-content: center; margin-top: 14px; min-width: 72px; min-height: 30px; border-radius: 7px; padding: 4px 10px; background: var(--blue); color: white; font-weight: 780; }}
	    .dividend-pill.muted {{ background: #7d8792; }}
	    .dividend-pill.green {{ background: var(--green); }}
	    .dividend-pill.cyan {{ background: #25bcd3; }}
	    .dividend-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; min-height: 560px; padding: 32px; box-shadow: var(--shadow); }}
	    .dividend-panel-header {{ display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 28px; }}
	    .dividend-title {{ margin: 0; font-size: 22px; font-weight: 780; }}
	    .dividend-total {{ display: inline-flex; gap: 12px; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 14px 22px; background: #1b222b; font-size: 18px; font-weight: 780; }}
	    .dividend-controls {{ display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 28px; flex-wrap: wrap; }}
	    .dividend-mode {{ display: inline-flex; border: 1px solid var(--blue); border-radius: 5px; overflow: hidden; }}
	    .dividend-mode button {{ border: 0; padding: 10px 16px; background: transparent; color: var(--blue); font: inherit; font-size: 16px; cursor: pointer; }}
	    .dividend-mode button.active {{ background: var(--blue); color: white; }}
	    .dividend-chart {{ width: 100%; min-height: 360px; display: block; }}
	    .dividend-grid {{ stroke: #29323d; stroke-width: 1; }}
	    .dividend-axis {{ fill: var(--text); font-size: 12px; }}
	    .dividend-axis-muted {{ fill: var(--muted); font-size: 12px; }}
	    .dividend-bar-after {{ fill: #3e9293; }}
	    .dividend-bar-tax {{ fill: var(--red); }}
	    .dividend-stock-list {{ display: grid; gap: 12px; }}
	    .dividend-stock-row {{ display: grid; grid-template-columns: minmax(160px, 1fr) minmax(220px, 2fr) minmax(120px, .7fr); gap: 16px; align-items: center; padding: 14px 0; border-bottom: 1px solid var(--line); }}
	    .dividend-stock-name {{ font-size: 16px; font-weight: 760; }}
	    .dividend-stock-sub {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
	    .dividend-stock-bar {{ height: 20px; border-radius: 999px; background: #414a59; overflow: hidden; }}
	    .dividend-stock-fill {{ height: 100%; background: #3e9293; border-radius: 999px; }}
	    .dividend-legend {{ display: flex; justify-content: center; gap: 24px; margin-top: 16px; color: var(--muted); font-size: 13px; }}
	    .dividend-legend span::before {{ content: ""; display: inline-block; width: 30px; height: 12px; margin-right: 8px; vertical-align: -1px; }}
	    .dividend-legend .after::before {{ background: #3e9293; }}
	    .dividend-legend .tax::before {{ background: var(--red); }}
	    .analytics-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }}
	    .analytics-heading {{ display: flex; align-items: center; gap: 22px; padding: 26px 34px; }}
	    .analytics-icon {{ color: var(--blue); font-size: 24px; width: 28px; text-align: center; }}
	    .analytics-title {{ margin: 0; font-size: 19px; font-weight: 780; }}
	    .analytics-subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
	    .analytics-table-wrap {{ margin: 0 20px 34px; border: 1px solid var(--line); border-radius: 6px; overflow: auto; max-height: 560px; }}
	    .analytics-table {{ width: 100%; min-width: 980px; border-collapse: collapse; }}
	    .analytics-table th {{ height: 42px; padding: 0 12px; border-bottom: 2px solid var(--line); color: var(--text); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; text-align: right; white-space: nowrap; }}
	    .analytics-table th:first-child {{ text-align: left; }}
	    .analytics-table td {{ padding: 14px 12px; border-bottom: 1px solid var(--line); font-size: 14px; text-align: right; vertical-align: middle; }}
	    .analytics-table tr:last-child td {{ border-bottom: 0; }}
	    .analytics-asset {{ display: flex; align-items: center; gap: 12px; text-align: left; min-width: 280px; }}
	    .analytics-avatar {{ width: 28px; height: 28px; border-radius: 4px; display: grid; place-items: center; background: #123266; color: white; font-size: 13px; font-weight: 800; flex: 0 0 auto; }}
	    .analytics-name {{ max-width: 360px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
	    .analytics-symbol {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
	    .analytics-note {{ color: var(--muted); font-size: 13px; line-height: 1.45; padding: 0 34px 28px; }}
	    .sharpe-warn {{ color: #ffb000; }}
	    .analytics-method-tabs {{ display: flex; align-items: center; gap: 8px; margin: 24px 0 22px; padding: 8px; border-radius: 10px; background: #171d25; overflow-x: auto; }}
	    .analytics-method-tab {{ min-height: 46px; border: 0; border-radius: 8px; padding: 0 18px; background: transparent; color: var(--muted); font: inherit; font-size: 14px; font-weight: 720; white-space: nowrap; }}
	    .analytics-method-tab.active {{ background: var(--blue); color: white; box-shadow: 0 8px 18px rgba(77,131,255,.25); }}
	    .monte-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 26px 20px 34px; }}
	    .monte-header {{ display: flex; align-items: center; gap: 20px; padding: 0 14px 24px; }}
	    .monte-title {{ margin: 0; font-size: 20px; font-weight: 780; }}
	    .monte-subtitle {{ margin-top: 5px; color: var(--muted); font-size: 13px; }}
		    .monte-cards {{ display: grid; grid-template-columns: repeat(6, minmax(150px, 1fr)); gap: 24px; margin-bottom: 24px; }}
	    .monte-card {{ min-height: 94px; border-radius: 10px; background: #222932; display: grid; place-items: center; text-align: center; padding: 14px; }}
	    .monte-card-label {{ color: var(--muted); font-size: 12px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }}
	    .monte-card-value {{ margin-top: 12px; font-size: 22px; font-weight: 820; }}
	    .monte-card-tag {{ display: inline-flex; margin-top: 10px; min-height: 22px; align-items: center; border-radius: 5px; padding: 0 9px; color: var(--muted); background: #343c46; font-size: 11px; }}
	    .monte-chart-wrap {{ padding: 0 28px; }}
		    .monte-chart-title {{ text-align: center; font-size: 12px; font-weight: 780; margin-bottom: 4px; }}
		    .monte-chart {{ width: 100%; min-height: 320px; display: block; }}
	    .monte-grid {{ stroke: #29323d; stroke-width: 1; }}
	    .monte-axis-line {{ stroke: #596272; stroke-width: 1.2; }}
	    .monte-tick {{ stroke: #596272; stroke-width: 1; }}
	    .monte-path {{ fill: none; stroke: rgba(45, 112, 220, .28); stroke-width: 1; }}
	    .monte-path-highlight {{ fill: none; stroke: rgba(77, 131, 255, .9); stroke-width: 3; }}
	    .monte-start-line {{ stroke: rgba(230,236,245,.6); stroke-width: 2; stroke-dasharray: 5 6; }}
		    .monte-axis {{ fill: var(--text); font-size: 10px; }}
		    .monte-axis-title {{ fill: var(--text); font-size: 11px; font-weight: 760; }}
		    .monte-note {{ margin: 18px 28px 0; color: var(--muted); font-size: 13px; line-height: 1.45; }}
		    .risk-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 38px 40px 44px; }}
		    .risk-grid {{ display: grid; grid-template-columns: minmax(320px, 1fr) minmax(360px, 1fr); gap: 64px; align-items: start; }}
		    .risk-section-title {{ display: flex; align-items: center; gap: 16px; margin-bottom: 18px; }}
		    .risk-icon {{ width: 40px; height: 40px; border-radius: 9px; display: grid; place-items: center; background: rgba(255, 80, 88, .18); color: var(--red); font-size: 20px; font-weight: 900; }}
		    .risk-icon.orange {{ background: rgba(255, 176, 0, .12); color: #ffb000; }}
		    .risk-title {{ margin: 0; font-size: 17px; font-weight: 800; }}
		    .risk-subtitle {{ margin-top: 5px; color: var(--muted); font-size: 13px; }}
		    .risk-var-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px; }}
		    .risk-box {{ min-height: 80px; border-radius: 8px; background: #22272e; padding: 16px; }}
		    .risk-label {{ color: var(--muted); font-size: 12px; font-weight: 760; letter-spacing: .05em; text-transform: uppercase; }}
		    .risk-value {{ margin-top: 10px; font-size: 25px; font-weight: 840; }}
		    .risk-help {{ margin-top: 16px; color: var(--muted); font-size: 13px; }}
		    .drawdown-hero {{ text-align: center; padding-top: 12px; }}
		    .drawdown-value {{ color: var(--red); font-size: 38px; font-weight: 860; }}
		    .drawdown-sub {{ color: var(--muted); font-size: 13px; margin-top: 8px; }}
		    .drawdown-line {{ margin-top: 34px; display: grid; grid-template-columns: 1fr 40px 1fr; gap: 22px; align-items: center; }}
		    .drawdown-segment {{ height: 4px; border-radius: 999px; background: linear-gradient(90deg, var(--green), var(--red)); }}
		    .drawdown-segment.recovery {{ background: linear-gradient(90deg, var(--red), var(--green)); }}
		    .drawdown-dot {{ width: 10px; height: 10px; border-radius: 999px; background: var(--red); box-shadow: 0 0 0 6px rgba(255,80,88,.14); justify-self: center; }}
		    .drawdown-dates {{ display: grid; grid-template-columns: 1fr 1fr 1fr; margin-top: 12px; color: var(--muted); font-size: 12px; text-align: center; }}
		    .risk-summary {{ margin-top: 72px; }}
		    .risk-summary-header {{ display: flex; align-items: center; gap: 18px; margin-bottom: 30px; }}
		    .risk-summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 28px; text-align: center; }}
		    .risk-summary-value {{ font-size: 29px; font-weight: 850; }}
		    .risk-summary-label {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
		    .frontier-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 26px 34px 36px; }}
		    .frontier-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 18px; }}
		    .secondary-btn {{ border: 1px solid var(--blue); border-radius: 8px; background: rgba(77,131,255,.08); color: var(--blue); min-height: 38px; padding: 0 14px; font: inherit; font-size: 13px; font-weight: 760; }}
		    .frontier-info {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-bottom: 18px; }}
		    .frontier-info-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #171d25; }}
		    .frontier-info-title {{ font-size: 13px; font-weight: 800; margin-bottom: 7px; }}
		    .frontier-info-text {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
		    .frontier-chart-wrap {{ border: 1px solid var(--line); border-radius: 8px; padding: 18px 18px 12px; background: #171d25; }}
		    .frontier-chart {{ width: 100%; min-height: 420px; display: block; }}
		    .frontier-grid {{ stroke: #2b333e; stroke-width: 1; }}
		    .frontier-line {{ fill: none; stroke: var(--blue); stroke-width: 3; }}
		    .frontier-asset {{ fill: #ef514d; stroke: #ffe4df; stroke-width: 2; }}
		    .frontier-current {{ fill: #111820; stroke: #f7c948; stroke-width: 4; }}
		    .frontier-sharpe {{ fill: #22c55e; stroke: #dbffe7; stroke-width: 3; }}
		    .frontier-axis {{ fill: var(--text); font-size: 10px; }}
		    .frontier-axis-title {{ fill: var(--text); font-size: 12px; font-weight: 780; }}
		    .frontier-table {{ width: 100%; margin-top: 18px; border-collapse: collapse; }}
		    .frontier-table th, .frontier-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: right; font-size: 13px; }}
		    .frontier-table th:first-child, .frontier-table td:first-child {{ text-align: left; }}
		    .frontier-legend {{ display: flex; gap: 18px; align-items: center; justify-content: center; color: var(--muted); font-size: 12px; margin-bottom: 8px; }}
		    .legend-dot {{ width: 11px; height: 11px; border-radius: 50%; display: inline-block; margin-right: 6px; vertical-align: -1px; }}
		    .legend-dot.blue {{ background: var(--blue); }}
		    .legend-dot.yellow {{ background: #f7c948; }}
		    .legend-dot.green {{ background: var(--green); }}
		    .legend-dot.red {{ background: #ef514d; }}
		    .scenario-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 34px 34px 32px; }}
		    .scenario-heading {{ display: flex; align-items: center; gap: 20px; margin-bottom: 28px; }}
		    .scenario-icon {{ width: 34px; height: 34px; display: grid; place-items: center; color: #ffb000; font-size: 24px; }}
		    .scenario-grid {{ display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 24px; }}
		    .scenario-card {{ border: 1px solid var(--line); border-radius: 8px; background: #171d25; overflow: hidden; }}
		    .scenario-card-head {{ padding: 16px 18px; background: #616a78; color: #f2f5fa; font-weight: 800; display: flex; align-items: center; justify-content: space-between; gap: 12px; }}
		    .scenario-card-body {{ padding: 20px; }}
		    .scenario-description {{ color: var(--muted); font-size: 13px; margin-bottom: 18px; }}
		    .scenario-return {{ font-size: 30px; font-weight: 860; }}
		    .scenario-subgrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
		    .scenario-kpi {{ border: 1px solid var(--line); border-radius: 7px; padding: 10px; background: #111820; }}
		    .scenario-kpi-label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }}
		    .scenario-kpi-value {{ margin-top: 5px; font-size: 15px; font-weight: 780; }}
		    .scenario-contributors {{ margin-top: 18px; border-top: 1px solid var(--line); padding-top: 12px; }}
		    .scenario-contributor {{ display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 12px; color: var(--muted); font-size: 12px; padding: 5px 0; }}
		    .scenario-note {{ margin-top: 28px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
		    .benchmark-panel {{ display: grid; gap: 28px; }}
		    .benchmark-top {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; }}
		    .benchmark-card {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }}
		    .benchmark-card-title {{ min-height: 44px; display: flex; align-items: center; gap: 8px; padding: 0 22px; color: #fff; font-size: 15px; font-weight: 800; }}
		    .benchmark-card-title.blue {{ background: #3c82f6; }}
		    .benchmark-card-title.cyan {{ background: #11aec3; color: #071014; }}
		    .benchmark-card-body {{ padding: 22px; }}
		    .benchmark-subtitle {{ color: var(--muted); font-size: 13px; margin-bottom: 18px; }}
		    .benchmark-table {{ width: 100%; border-collapse: collapse; }}
		    .benchmark-table th, .benchmark-table td {{ padding: 12px 10px; border-top: 1px solid var(--line); text-align: center; font-size: 14px; }}
		    .benchmark-table th {{ color: var(--text); text-transform: uppercase; letter-spacing: .05em; font-size: 12px; }}
		    .benchmark-table td:first-child {{ font-weight: 760; }}
		    .benchmark-excess {{ margin-top: 16px; font-size: 13px; }}
		    .benchmark-metric-list {{ display: grid; }}
		    .benchmark-metric-row {{ display: grid; grid-template-columns: minmax(220px, 1fr) auto; gap: 18px; padding: 13px 0; border-top: 1px solid var(--line); align-items: center; font-size: 14px; }}
		    .benchmark-metric-label {{ text-align: center; font-weight: 760; }}
		    .benchmark-chart-card {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); overflow: hidden; }}
		    .benchmark-chart-title {{ min-height: 54px; border-bottom: 1px solid var(--line); display: flex; align-items: center; padding: 0 22px; font-size: 16px; font-weight: 780; }}
		    .benchmark-chart-wrap {{ padding: 22px 22px 28px; position: relative; }}
		    .benchmark-chart {{ width: 100%; min-height: 360px; display: block; }}
		    .benchmark-grid {{ stroke: #29323d; stroke-width: 1; }}
		    .benchmark-axis {{ fill: var(--text); font-size: 11px; }}
		    .benchmark-axis-title {{ fill: var(--text); font-size: 12px; font-weight: 780; }}
		    .benchmark-line-portfolio {{ fill: none; stroke: var(--blue); stroke-width: 3; }}
		    .benchmark-line-index {{ fill: none; stroke: var(--red); stroke-width: 3; }}
		    .benchmark-area-portfolio {{ fill: rgba(77, 131, 255, .16); }}
		    .benchmark-area-index {{ fill: rgba(239, 83, 80, .10); }}
		    .benchmark-legend {{ display: flex; justify-content: center; gap: 22px; color: var(--text); font-size: 12px; margin-bottom: 10px; }}
		    .benchmark-hit {{ fill: transparent; stroke: transparent; cursor: crosshair; pointer-events: all; }}
		    .benchmark-tooltip {{ position: fixed; z-index: 20; pointer-events: none; min-width: 190px; padding: 9px 11px; border-radius: 8px; border: 1px solid var(--line); background: #080b10; color: var(--text); box-shadow: var(--shadow); font-size: 12px; line-height: 1.45; opacity: 0; transform: translate(-50%, calc(-100% - 14px)); transition: opacity .08s ease; }}
		    .benchmark-tooltip.visible {{ opacity: 1; }}
		    .correlation-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 28px 34px 36px; }}
		    .correlation-head {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 18px; }}
		    .correlation-info-grid {{ display: grid; grid-template-columns: repeat(4, minmax(170px, 1fr)); gap: 14px; margin-bottom: 22px; }}
		    .correlation-info-card {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #171d25; }}
		    .correlation-info-title {{ font-size: 13px; font-weight: 820; margin-bottom: 7px; }}
		    .correlation-info-text {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
		    .correlation-chart-card {{ border: 1px solid var(--line); border-radius: 8px; background: #171d25; padding: 18px; overflow: auto; }}
		    .correlation-title {{ margin: 0 0 16px; font-size: 20px; font-weight: 820; }}
		    .correlation-matrix {{ display: grid; gap: 2px; width: max-content; min-width: 100%; align-items: stretch; }}
		    .correlation-corner, .correlation-label, .correlation-cell {{ min-width: 76px; min-height: 44px; display: grid; place-items: center; padding: 7px; font-size: 12px; font-weight: 760; }}
		    .correlation-label {{ color: var(--text); background: #111820; border: 1px solid #25303b; position: sticky; z-index: 2; }}
		    .correlation-label.top {{ top: 0; }}
		    .correlation-label.left {{ left: 0; justify-items: end; text-align: right; min-width: 96px; }}
		    .correlation-corner {{ background: #111820; border: 1px solid #25303b; position: sticky; left: 0; top: 0; z-index: 3; }}
		    .correlation-cell {{ color: #f7fbff; border: 1px solid rgba(255,255,255,.16); font-variant-numeric: tabular-nums; }}
		    .correlation-legend {{ display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 12px; margin-top: 14px; flex-wrap: wrap; }}
		    .correlation-ramp {{ width: 260px; max-width: 100%; height: 12px; border-radius: 999px; background: linear-gradient(90deg, #3855a4, #1f9a94, #f2dd2f); border: 1px solid rgba(255,255,255,.22); }}
		    .correlation-table-card {{ margin-top: 24px; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: #171d25; }}
		    .correlation-table-title {{ min-height: 54px; display: flex; align-items: center; padding: 0 20px; border-bottom: 1px solid var(--line); font-size: 16px; font-weight: 820; }}
		    .correlation-table {{ width: 100%; min-width: 860px; border-collapse: collapse; }}
		    .correlation-table th, .correlation-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; }}
		    .correlation-table th {{ color: var(--text); letter-spacing: .06em; text-transform: uppercase; font-size: 11px; }}
		    .correlation-table td:nth-child(3), .correlation-table th:nth-child(3) {{ text-align: right; font-variant-numeric: tabular-nums; }}
		    .correlation-note {{ margin-top: 16px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
		    .trade-alpha-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 32px 34px 36px; }}
		    .trade-alpha-head {{ display: flex; align-items: flex-start; gap: 22px; margin-bottom: 26px; }}
		    .trade-alpha-icon {{ width: 40px; height: 40px; display: grid; place-items: center; color: #ffb000; font-size: 28px; }}
		    .trade-alpha-title {{ margin: 0; font-size: 25px; font-weight: 850; }}
		    .trade-alpha-subtitle {{ color: var(--muted); font-size: 15px; margin-top: 6px; }}
		    .trade-alpha-summary {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 18px; margin: 18px 0 30px; }}
		    .trade-alpha-kpi {{ min-height: 120px; display: grid; place-items: center; text-align: center; background: #151b23; border-radius: 8px; border: 1px solid #18212b; padding: 16px; }}
		    .trade-alpha-label {{ color: var(--muted); font-size: 12px; font-weight: 820; letter-spacing: .08em; text-transform: uppercase; }}
		    .trade-alpha-value {{ margin-top: 12px; font-size: 31px; font-weight: 860; }}
		    .trade-alpha-sub {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
		    .trade-alpha-explainer {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 14px; margin-bottom: 24px; }}
		    .trade-alpha-info {{ border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: #171d25; }}
		    .trade-alpha-info strong {{ display: block; margin-bottom: 6px; }}
		    .trade-alpha-info span {{ color: var(--muted); font-size: 12px; line-height: 1.45; }}
		    .trade-alpha-comparison {{ display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 20px; align-items: center; margin: 20px 0 28px; }}
		    .trade-alpha-compare-card {{ text-align: center; color: var(--muted); }}
		    .trade-alpha-compare-value {{ margin-top: 10px; font-size: 24px; font-weight: 850; }}
		    .trade-alpha-pill {{ display: inline-flex; min-height: 36px; align-items: center; border-radius: 8px; padding: 0 18px; background: var(--blue); color: white; font-weight: 800; }}
		    .trade-alpha-pill.gray {{ background: #6b737d; }}
		    .trade-alpha-table-wrap {{ border: 1px solid #c5c9d0; border-radius: 5px; overflow: auto; }}
		    .trade-alpha-table {{ width: 100%; min-width: 1120px; border-collapse: collapse; }}
		    .trade-alpha-table th {{ padding: 16px 14px; border-bottom: 2px solid var(--line); color: var(--text); font-size: 12px; letter-spacing: .08em; text-transform: uppercase; text-align: right; white-space: nowrap; }}
		    .trade-alpha-table th:first-child {{ text-align: left; }}
		    .trade-alpha-table td {{ padding: 14px; border-bottom: 1px solid var(--line); font-size: 14px; text-align: right; vertical-align: middle; }}
		    .trade-alpha-table td:first-child {{ text-align: left; }}
		    .trade-alpha-holding {{ display: flex; align-items: center; gap: 12px; min-width: 300px; }}
		    .trade-alpha-date {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
		    .alpha-badge {{ display: inline-flex; align-items: center; justify-content: center; min-width: 78px; min-height: 34px; border-radius: 8px; padding: 0 10px; font-weight: 820; }}
		    .alpha-badge.positive {{ background: rgba(66,199,100,.18); color: var(--green); }}
		    .alpha-badge.negative {{ background: rgba(239,83,80,.2); color: var(--red); }}
		    .status-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 70px; min-height: 30px; border-radius: 8px; background: var(--blue); color: white; font-weight: 800; }}
		    .status-pill.muted {{ background: #46505c; color: var(--text); }}
		    .trade-alpha-note {{ margin-top: 16px; color: var(--muted); font-size: 13px; line-height: 1.5; }}
		    .diversification-panel {{ background: var(--panel); border: 1px solid var(--line); box-shadow: var(--shadow); padding: 38px 40px 34px; }}
		    .diversification-grid {{ display: grid; grid-template-columns: minmax(280px, .82fr) minmax(330px, 1fr) minmax(330px, 1fr); gap: 24px; align-items: stretch; }}
		    .diversification-card {{ min-height: 330px; background: #171c23; border-radius: 8px; padding: 26px 24px; }}
		    .diversification-heading {{ display: flex; align-items: flex-start; gap: 16px; margin-bottom: 24px; }}
		    .diversification-icon {{ width: 32px; height: 32px; display: grid; place-items: center; color: var(--green); font-size: 22px; }}
		    .diversification-title {{ margin: 0; font-size: 17px; font-weight: 820; }}
		    .diversification-subtitle {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
		    .diversification-score {{ min-height: 130px; display: flex; align-items: center; justify-content: center; gap: 12px; }}
		    .diversification-score-value {{ font-size: 58px; line-height: 1; font-weight: 880; color: #6671ff; }}
		    .diversification-score-label {{ color: var(--muted); font-size: 13px; align-self: center; margin-top: 30px; }}
		    .diversification-divider {{ border-top: 1px solid var(--line); margin: 18px 0 24px; }}
		    .diversification-subscore-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; text-align: center; }}
		    .diversification-subscore-value {{ font-size: 24px; font-weight: 820; }}
		    .diversification-subscore-label {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
		    .diversification-footnote {{ margin-top: 22px; text-align: center; color: var(--muted); font-size: 12px; }}
		    .diversification-chart-card {{ min-height: 330px; background: #171c23; border-radius: 8px; padding: 26px 24px; }}
		    .diversification-chart-body {{ display: grid; grid-template-columns: minmax(150px, .9fr) minmax(128px, .7fr); gap: 14px; align-items: center; }}
		    .diversification-donut {{ width: min(190px, 100%); aspect-ratio: 1; border-radius: 50%; margin: 0 auto; position: relative; background: var(--donut); }}
		    .diversification-donut::after {{ content: ""; position: absolute; inset: 34%; border-radius: 50%; background: #171c23; }}
		    .diversification-legend {{ display: grid; gap: 9px; align-content: center; }}
		    .diversification-legend-row {{ display: grid; grid-template-columns: 11px minmax(0, 1fr) auto; gap: 7px; align-items: center; color: var(--text); font-size: 11px; }}
		    .diversification-swatch {{ width: 12px; height: 12px; border-radius: 3px; background: var(--swatch); }}
		    .diversification-legend-name {{ white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
		    .diversification-chart-note {{ margin-top: 14px; color: var(--muted); text-align: center; font-size: 12px; }}
		    .diversification-recommendations {{ margin-top: 28px; border-left: 4px solid #ffb020; border-radius: 8px; background: #242321; padding: 22px 26px; }}
		    .diversification-rec-title {{ display: flex; align-items: center; gap: 12px; margin: 0 0 16px; font-size: 15px; font-weight: 820; }}
		    .diversification-rec-list {{ margin: 0; padding-left: 18px; color: var(--muted); line-height: 1.7; font-size: 14px; }}
		    .diversification-breakdowns {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 26px; margin-top: 28px; }}
		    .breakdown-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); overflow: hidden; }}
		    .breakdown-title {{ min-height: 54px; display: flex; align-items: center; gap: 8px; padding: 0 20px; border-bottom: 1px solid var(--line); font-size: 15px; font-weight: 820; }}
		    .breakdown-body {{ padding: 18px 20px 22px; overflow-x: auto; }}
		    .breakdown-table {{ width: 100%; min-width: 520px; border-collapse: collapse; }}
		    .breakdown-table th {{ padding: 12px 10px; border-bottom: 2px solid var(--line); color: var(--text); text-transform: uppercase; letter-spacing: .08em; font-size: 12px; text-align: center; }}
		    .breakdown-table td {{ padding: 14px 10px; border-bottom: 1px solid var(--line); font-size: 14px; text-align: center; vertical-align: middle; }}
		    .breakdown-name {{ max-width: 260px; margin: 0 auto; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
		    .breakdown-weight {{ font-variant-numeric: tabular-nums; text-align: right; }}
		    .breakdown-bar {{ height: 15px; border-radius: 999px; background: #121820; overflow: hidden; min-width: 220px; }}
		    .breakdown-fill {{ height: 100%; border-radius: 999px; background: var(--bar-color); width: var(--bar-width); }}
		    .diversification-note {{ margin-top: 26px; border-radius: 8px; background: #20262e; color: var(--text); padding: 18px 20px; font-size: 14px; line-height: 1.55; }}
		    .holdings-panel, .alerts-panel, .assistant-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; box-shadow: var(--shadow); }}
		    .holdings-header, .alerts-header, .assistant-header {{ min-height: 64px; padding: 18px 22px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
		    .holdings-title, .alerts-title, .assistant-title {{ margin: 0; font-size: 18px; font-weight: 800; }}
		    .holdings-tools, .alerts-tools {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
		    .mini-input, .mini-select {{ min-height: 34px; border: 1px solid var(--line); border-radius: 7px; background: #111820; color: var(--text); padding: 0 10px; font: inherit; font-size: 13px; }}
		    .mini-input {{ width: 130px; }}
		    .mini-select {{ width: 112px; }}
		    .action-btn {{ min-height: 34px; border: 1px solid var(--blue); border-radius: 7px; background: rgba(77,131,255,.13); color: var(--blue); padding: 0 12px; font: inherit; font-size: 13px; font-weight: 760; cursor: pointer; }}
		    .action-btn.danger {{ border-color: rgba(239,83,80,.65); color: #ff7a78; background: rgba(239,83,80,.1); }}
		    .action-btn.green {{ border-color: rgba(66,199,100,.75); color: var(--green); background: rgba(66,199,100,.1); }}
		    .holdings-table-wrap, .alerts-table-wrap, .rebalance-table-wrap {{ overflow: auto; max-height: 520px; }}
		    .holdings-table {{ width: 100%; min-width: 1320px; border-collapse: collapse; }}
		    .alerts-table, .rebalance-table {{ width: 100%; min-width: 1100px; border-collapse: collapse; }}
		    .holdings-table th, .holdings-table td, .alerts-table th, .alerts-table td, .rebalance-table th, .rebalance-table td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: middle; text-align: right; }}
		    .holdings-table th, .alerts-table th, .rebalance-table th {{ height: 42px; color: var(--text); font-size: 11px; letter-spacing: .08em; text-transform: uppercase; background: #151b23; white-space: nowrap; }}
		    .holdings-table th:first-child, .holdings-table td:first-child, .alerts-table th:first-child, .alerts-table td:first-child, .rebalance-table th:first-child, .rebalance-table td:first-child {{ text-align: left; }}
		    .holding-cell {{ display: flex; align-items: center; gap: 12px; min-width: 260px; }}
		    .holding-logo {{ width: 34px; height: 34px; border-radius: 8px; display: grid; place-items: center; background: #123266; color: white; font-size: 12px; font-weight: 840; flex: 0 0 auto; }}
		    .holding-name {{ max-width: 260px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-weight: 760; }}
		    .holding-sub {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
		    .sparkline {{ width: 104px; height: 30px; display: block; }}
		    .sparkline path {{ fill: none; stroke: var(--blue); stroke-width: 2; }}
		    .sparkline.negative path {{ stroke: var(--red); }}
		    .alert-summary-grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 16px; margin-bottom: 18px; }}
		    .alert-card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; min-height: 100px; padding: 18px; box-shadow: var(--shadow); }}
		    .alert-label {{ color: var(--muted); font-size: 12px; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }}
		    .alert-value {{ margin-top: 10px; font-size: 27px; font-weight: 840; }}
		    .alert-status {{ display: inline-flex; align-items: center; min-height: 24px; border-radius: 999px; padding: 0 9px; font-size: 12px; font-weight: 780; }}
		    .alert-status.active {{ background: rgba(66,199,100,.16); color: var(--green); }}
		    .alert-status.triggered {{ background: rgba(239,83,80,.16); color: #ff7a78; }}
		    .alert-status.paused {{ background: rgba(154,163,178,.18); color: var(--muted); }}
		    .alerts-actions {{ display: flex; justify-content: flex-end; gap: 8px; }}
		    .alert-form-grid {{ display: grid; grid-template-columns: repeat(7, minmax(110px, 1fr)); gap: 10px; align-items: end; padding: 18px 20px; border-bottom: 1px solid var(--line); background: #111820; }}
		    .form-field {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; font-weight: 760; }}
		    .form-field .mini-input, .form-field .mini-select {{ width: 100%; }}
		    .rebalance-card {{ background: #111820; border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
		    .rebalance-head {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 12px; }}
		    .rebalance-page {{ display: grid; gap: 22px; }}
		    .rebalance-title-block {{ display: grid; gap: 6px; }}
		    .rebalance-title {{ margin: 0; font-size: 27px; font-weight: 860; }}
		    .rebalance-subtitle {{ color: var(--muted); font-size: 14px; line-height: 1.45; }}
		    .rebalance-subtitle a {{ color: var(--blue); text-decoration: none; }}
		    .rebalance-summary {{ display: grid; grid-template-columns: repeat(3, minmax(180px, 1fr)); gap: 18px; }}
		    .rebalance-kpis {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-bottom: 14px; }}
		    .rebalance-kpi, .rebalance-summary-card {{ background: #171d25; border: 1px solid var(--line); border-radius: 10px; padding: 16px; }}
		    .rebalance-summary-card {{ min-height: 116px; display: grid; align-content: center; gap: 12px; }}
		    .rebalance-summary-value {{ font-size: 28px; line-height: 1.1; font-weight: 860; }}
		    .rebalance-workspace {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(330px, 420px); gap: 22px; align-items: start; }}
		    .rebalance-visual-panel, .rebalance-options-panel, .rebalance-breakdown-panel, .rebalance-results-panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }}
		    .rebalance-visual-panel {{ padding: 22px; min-height: 380px; }}
		    .rebalance-options-panel {{ padding: 24px; position: sticky; top: 16px; }}
		    .rebalance-chart-grid {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 18px; align-items: center; }}
		    .rebalance-chart-title {{ display: flex; align-items: center; gap: 8px; font-size: 16px; font-weight: 820; margin-bottom: 12px; }}
		    .rebalance-donut {{ width: min(280px, 100%); aspect-ratio: 1; border-radius: 50%; margin: 0 auto; background: var(--donut); position: relative; }}
		    .rebalance-donut::after {{ content: ""; position: absolute; inset: 31%; border-radius: 50%; background: var(--panel); box-shadow: inset 0 0 0 3px #0d1219; }}
		    .rebalance-donut-center {{ position: absolute; inset: 0; display: grid; place-content: center; text-align: center; z-index: 1; pointer-events: none; }}
		    .rebalance-donut-value {{ font-size: 17px; font-weight: 850; }}
		    .rebalance-donut-label {{ color: var(--muted); font-size: 12px; margin-top: 3px; }}
		    .rebalance-option-field {{ display: grid; gap: 8px; margin: 18px 0; }}
		    .rebalance-option-label {{ color: var(--muted); font-size: 12px; font-weight: 820; letter-spacing: .08em; text-transform: uppercase; }}
		    .rebalance-amount-input {{ width: 100%; min-height: 46px; border: 1px solid var(--line); border-radius: 9px; background: #0d1219; color: var(--text); padding: 0 14px; font: inherit; font-size: 17px; }}
		    .rebalance-segment {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }}
		    .rebalance-segment button {{ min-height: 43px; border: 0; background: #0d1219; color: var(--muted); font: inherit; font-weight: 760; cursor: pointer; }}
		    .rebalance-segment button.active {{ background: var(--blue); color: white; }}
		    .rebalance-check {{ display: grid; grid-template-columns: 24px minmax(0, 1fr); gap: 12px; align-items: start; padding: 16px 0; border-top: 1px solid var(--line); color: var(--text); cursor: pointer; }}
		    .rebalance-check input {{ width: 20px; height: 20px; accent-color: var(--blue); }}
		    .rebalance-check-title {{ font-size: 15px; font-weight: 760; }}
		    .rebalance-check-sub {{ color: var(--muted); font-size: 12px; margin-top: 4px; line-height: 1.4; }}
		    .rebalance-primary {{ width: 100%; min-height: 48px; border: 0; border-radius: 9px; background: var(--blue); color: white; font: inherit; font-weight: 850; font-size: 16px; margin-top: 16px; cursor: pointer; }}
		    .rebalance-secondary {{ width: 100%; min-height: 42px; border: 1px solid var(--line); border-radius: 9px; background: #111820; color: var(--muted); font: inherit; font-weight: 780; margin-top: 10px; }}
		    .rebalance-breakdown-panel {{ padding: 22px; }}
		    .rebalance-breakdown-header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-bottom: 16px; }}
		    .rebalance-breakdown-sub {{ color: var(--muted); font-size: 13px; margin-top: 6px; }}
		    .rebalance-breakdown-tools {{ display: flex; gap: 8px; flex-wrap: wrap; }}
		    .rebalance-mode-chip {{ min-height: 34px; border: 1px solid var(--line); border-radius: 999px; background: #111820; color: var(--muted); padding: 0 14px; font: inherit; font-size: 13px; font-weight: 760; }}
		    .rebalance-mode-chip.active {{ border-color: var(--blue); color: white; background: rgba(77,131,255,.35); }}
		    .rebalance-allocation-row {{ display: grid; grid-template-columns: minmax(170px, 1fr) minmax(240px, 2fr) 80px 98px 96px; gap: 12px; align-items: center; padding: 10px 0; border-top: 1px solid var(--line); }}
		    .rebalance-allocation-name {{ display: flex; align-items: center; gap: 10px; font-weight: 800; }}
		    .rebalance-chevron {{ color: var(--muted); font-size: 18px; }}
		    .rebalance-progress {{ position: relative; height: 26px; border-radius: 8px; background: #29313b; overflow: hidden; }}
		    .rebalance-progress-fill {{ position: absolute; inset: 0 auto 0 0; width: var(--current-width); background: var(--bar-color); border-radius: 8px; }}
		    .rebalance-progress-target {{ position: absolute; top: -4px; bottom: -4px; left: var(--target-left); width: 3px; border-radius: 999px; background: #a8b0bb; box-shadow: 0 -6px 0 #a8b0bb, 0 6px 0 #a8b0bb; }}
		    .rebalance-target-inline {{ display: inline-grid; grid-template-columns: 78px 18px; gap: 6px; align-items: center; }}
		    .rebalance-drift-pill {{ display: inline-flex; justify-content: center; align-items: center; min-height: 30px; border-radius: 999px; padding: 0 10px; background: rgba(66,199,100,.14); color: var(--green); font-weight: 820; }}
		    .rebalance-drift-pill.negative {{ background: rgba(239,83,80,.14); color: #ff7775; }}
		    .rebalance-results-panel {{ overflow: hidden; }}
		    .rebalance-results-head {{ padding: 18px 22px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 14px; }}
		    .rebalance-results-summary {{ color: var(--muted); font-size: 13px; }}
		    .rebalance-action-pill {{ display: inline-flex; align-items: center; justify-content: center; min-width: 72px; min-height: 28px; border-radius: 999px; font-size: 12px; font-weight: 820; }}
		    .rebalance-action-pill.buy {{ background: rgba(66,199,100,.16); color: var(--green); }}
		    .rebalance-action-pill.sell {{ background: rgba(239,83,80,.16); color: #ff7775; }}
		    .rebalance-action-pill.hold {{ background: rgba(154,163,178,.16); color: var(--muted); }}
		    .rebalance-cheapest {{ color: #ffb020; font-size: 11px; font-weight: 820; margin-left: 6px; }}
		    .target-input {{ width: 82px; min-height: 30px; border: 1px solid var(--line); border-radius: 6px; background: #0d1219; color: var(--text); text-align: right; padding: 0 8px; font: inherit; font-size: 13px; }}
		    .assistant-layout {{ display: grid; grid-template-columns: minmax(280px, 420px) minmax(0, 1fr); gap: 22px; }}
		    .assistant-panel {{ padding-bottom: 20px; }}
		    .assistant-main {{ padding: 20px; display: grid; gap: 16px; }}
		    .assistant-prompts {{ display: grid; gap: 10px; }}
		    .prompt-btn {{ text-align: left; min-height: 42px; border: 1px solid var(--line); border-radius: 8px; background: #111820; color: var(--text); padding: 10px 12px; font: inherit; font-size: 13px; cursor: pointer; }}
		    .assistant-question-box {{ display: grid; gap: 10px; }}
		    .assistant-question-input {{ width: 100%; min-height: 92px; resize: vertical; border: 1px solid var(--line); border-radius: 9px; background: #0d1219; color: var(--text); padding: 12px; font: inherit; font-size: 14px; line-height: 1.45; }}
		    .assistant-actions {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }}
		    .assistant-answer {{ min-height: 260px; border: 1px solid var(--line); border-radius: 10px; background: #111820; padding: 18px; line-height: 1.55; font-size: 14px; }}
		    .assistant-answer h3 {{ margin: 0 0 10px; font-size: 17px; }}
		    .assistant-answer ul {{ margin: 8px 0 0; padding-left: 18px; }}
		    .assistant-answer-section {{ margin-top: 14px; }}
		    .assistant-answer-section strong {{ display: block; margin-bottom: 6px; }}
		    .assistant-note {{ color: var(--muted); font-size: 13px; line-height: 1.45; border: 1px dashed #313b47; border-radius: 8px; padding: 10px 12px; }}
		    @media (max-width: 1300px) {{ .monte-cards {{ grid-template-columns: repeat(3, minmax(150px, 1fr)); }} }}
		    @media (max-width: 1100px) {{ .risk-grid, .risk-summary-grid, .frontier-info, .scenario-grid, .benchmark-top, .diversification-grid, .diversification-breakdowns, .assistant-layout, .rebalance-workspace, .rebalance-chart-grid {{ grid-template-columns: minmax(0, 1fr); gap: 24px; }} .rebalance-options-panel {{ position: static; }} }}
		    @media (max-width: 720px) {{ .monte-cards, .risk-var-grid {{ grid-template-columns: minmax(0, 1fr); }} .monte-chart-wrap {{ padding: 0; }} }}
	    .empty-panel {{ padding: 42px 30px; color: var(--muted); font-size: 17px; line-height: 1.5; }}
    .placeholder {{ margin-top: 24px; padding: 40px; min-height: 260px; color: var(--muted); font-size: 18px; }}
    .status-line {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 200; color: var(--muted); font-size: 14px; display: flex; align-items: center; justify-content: space-between; gap: 14px; flex-wrap: wrap; padding: 9px 30px; background: rgba(11, 15, 22, .96); border-top: 1px solid var(--line); box-shadow: 0 -10px 22px rgba(0,0,0,.22); }}
    .status-text {{ min-width: 220px; }}
    .refresh-btn {{ min-height: 34px; border: 1px solid var(--blue); border-radius: 7px; background: rgba(77,131,255,.13); color: var(--blue); padding: 0 12px; font: inherit; font-size: 13px; font-weight: 760; cursor: pointer; }}
    @media (max-width: 1300px) {{
      .metric-row {{ grid-template-columns: repeat(2, minmax(240px, 1fr)); gap: 18px; margin-bottom: 30px; }}
      .goal-grid {{ grid-template-columns: repeat(2, minmax(160px, 1fr)); }}
      .goal-cell {{ border-bottom: 1px solid var(--line); }}
      .portfolio-grid {{ grid-template-columns: repeat(2, minmax(260px, 1fr)); }}
	      .dashboard-main, .allocation-body, .dividend-layout {{ grid-template-columns: minmax(0, 1fr); }}
	      .insights-layout, .insight-definition-list {{ grid-template-columns: minmax(0, 1fr); }}
	      .insight-card-grid {{ grid-template-columns: repeat(2, minmax(180px, 1fr)); }}
	      .dashboard-bottom {{ grid-template-columns: minmax(0, 1fr); gap: 18px; }}
      .market-panel + .market-panel {{ margin-left: 0; }}
      .tabs {{ gap: 12px; overflow-x: auto; }}
    }}
    @media (max-width: 720px) {{
      .shell {{ padding: 16px 12px 28px; }}
      .metric-row, .portfolio-grid, .rebalance-summary {{ grid-template-columns: minmax(0, 1fr); }}
      .goal-grid {{ grid-template-columns: minmax(0, 1fr); }}
      .goal-header {{ align-items: flex-start; flex-direction: column; }}
      .metric-card, .portfolio-card {{ padding: 22px 20px; }}
      .tab {{ min-width: 170px; font-size: 20px; }}
      .allocation-body {{ padding: 36px 22px; }}
      .allocation-row {{ grid-template-columns: minmax(0, 1fr); }}
      .allocation-value {{ text-align: left; }}
      .stock-grid {{ grid-template-columns: minmax(0, 1fr); padding: 28px 18px; }}
	      .transaction-row, .market-row {{ grid-template-columns: minmax(0, 1fr); }}
	      .market-values {{ text-align: left; }}
	      .heatmap-grid {{ grid-template-columns: repeat(4, minmax(0, 1fr)); padding: 0 16px 28px; }}
	      .heatmap-cell {{ grid-column: span 2; }}
	      .heatmap-controls {{ flex-direction: column; align-items: stretch; }}
	      .insight-card-grid {{ grid-template-columns: minmax(0, 1fr); }}
	      .insight-card.wide {{ grid-column: auto; }}
	      .breakdown-body {{ padding: 14px 12px 18px; overflow-x: visible; }}
	      .breakdown-table {{ min-width: 0; table-layout: fixed; }}
	      .breakdown-table th, .breakdown-table td {{ padding: 12px 6px; font-size: 12px; }}
	      .breakdown-table th:nth-child(1), .breakdown-table td:nth-child(1) {{ width: 42%; }}
	      .breakdown-table th:nth-child(2), .breakdown-table td:nth-child(2) {{ width: 24%; }}
	      .breakdown-table th:nth-child(3), .breakdown-table td:nth-child(3) {{ width: 34%; }}
	      .breakdown-bar {{ min-width: 0; height: 13px; }}
	      .rebalance-allocation-row {{ grid-template-columns: minmax(0, 1fr); gap: 8px; }}
	      .rebalance-target-inline {{ grid-template-columns: 82px 18px; }}
	    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="metric-row" id="metricRow"></section>
    <section id="goalPanel"></section>
    <nav class="tabs" id="tabs"></nav>
    <section id="tabContent"></section>
    <div class="status-line" id="statusLine"></div>
  </div>
  <script>
    const refreshMs = {refresh_seconds * 1000};
    const tabs = [
      ["portfolios", "💼", "Portfolios"],
      ["dashboard", "⌁", "Dashboard"],
      ["insights", "▤", "Insights"],
      ["dividends", "◉", "Dividends"],
      ["analytics", "◌", "Analytics"],
      ["alerts", "⚑", "Alerts"],
      ["assistant", "✦", "Assistant"],
      ["screener", "◔", "Screener"],
    ];
    let state = null;
    let activeTab = location.hash ? location.hash.replace("#", "") : "portfolios";
    let allocationMode = "category";
    let heatmapReturnType = "daily";
    let growthRange = "max";
    let includeRealizedGrowth = true;
    let dividendMode = "monthly";
    let analyticsMode = "monte_carlo";
    let rebalanceAmount = 0;
    let rebalanceOperation = "deposit";
    let rebalanceAllowSelling = false;
    let rebalanceMinimizeTransactions = false;
    let rebalancePricePriority = false;
    let lastAssistantQuestion = localStorage.getItem("lastAssistantQuestion") || "Summarize this selected portfolio and tell me the main risks.";
    let lastAssistantAnswer = null;
    let lastAssistantError = "";
    let assistantIsLoading = false;
    let manualRefreshLoading = false;
    let lastRefreshMessage = "";
    let selectedKey = localStorage.getItem("selectedPortfolioKey") || "combined:live";
    const moneyFmt = new Intl.NumberFormat(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
    function money(value, currency = "EUR") {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${{moneyFmt.format(Number(value))}} ${{currency || "EUR"}}`;
    }}
    function pct(value) {{
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${{Number(value).toFixed(2)}}%`;
    }}
    function sign(value) {{ return Number(value || 0) >= 0 ? "+" : ""; }}
    function klass(value) {{ return Number(value || 0) >= 0 ? "positive" : "negative"; }}
    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, (char) => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }}[char]));
    }}
    async function apiPost(url, payload) {{
      const response = await fetch(url, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload || {{}}),
      }});
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }}
    function portfolioList() {{ return state?.portfolio_summaries || []; }}
    function selectedPortfolio() {{
      return portfolioList().find((item) => item.key === selectedKey) || state?.combined_portfolio || portfolioList()[0] || {{}};
    }}
    function renderMetrics() {{
      const p = selectedPortfolio();
      const currency = p.currency || "EUR";
      const gainClass = klass(p.capital_gain);
      const gainLabel = Number(p.capital_gain || 0) >= 0 ? "Capital Gain" : "Capital Loss";
      const holdingsCost = Number(p.holdings_cost_basis ?? p.initially_invested ?? 0);
      const unrealized = Number(p.unrealized_pl ?? p.capital_gain ?? 0);
      const realized = Number(p.realized_pl || 0);
      const investedLine = Math.abs(holdingsCost - Number(p.initially_invested || 0)) > 0.01
        ? `Invested capital: ${{money(p.initially_invested, currency)}} · Holdings cost: ${{money(holdingsCost, currency)}}`
        : `Holdings cost: ${{money(holdingsCost, currency)}}`;
      const totalWorthTitle = `Current portfolio value including equities, cash and bonds. Equities: ${{money(p.equities, currency)}}. Cash/Bonds: ${{money(p.cash, currency)}}.`;
      const investedTitle = `Invested capital is account-level net money in the account when available. Holdings cost basis excludes cash and includes only currently owned holdings: ${{money(holdingsCost, currency)}}.`;
      const growthTitle = `Total growth equals open holdings P/L plus realized P/L when available. Open holdings P/L: ${{money(unrealized, currency)}}. Realized P/L: ${{money(realized, currency)}}.`;
      document.getElementById("metricRow").innerHTML = `
        <article class="metric-card">
          <div class="metric-label"><span class="metric-icon">▰</span> Simple Return <span>⌄</span></div>
          <div class="metric-value">${{pct(p.simple_return_pct)}}</div>
          <div class="metric-sub" title="${{escapeHtml(growthTitle)}}">Total growth: ${{money(p.capital_gain, currency)}} <span class="info">●</span></div>
        </article>
        <article class="metric-card" title="${{escapeHtml(totalWorthTitle)}}">
          <div class="metric-label"><span class="metric-icon">▣</span> ${{p.name || "Portfolio"}} <span class="info">●</span></div>
          <div class="metric-value">${{money(p.total_worth, currency)}}</div>
          <div class="metric-sub" title="${{escapeHtml(investedTitle)}}">${{investedLine}} <span class="info">●</span></div>
        </article>
        <article class="metric-card">
          <div class="metric-label"><span class="metric-icon">⌁</span> ${{gainLabel}} <span class="info">●</span></div>
          <div class="metric-value ${{gainClass}}">${{sign(p.capital_gain)}} ${{money(p.capital_gain, currency)}} <span>${{Number(p.capital_gain || 0) >= 0 ? "↑" : "↓"}}</span> <span style="font-size:16px">(${{pct(p.simple_return_pct)}})</span></div>
          <div class="metric-sub">Open holdings P/L: <span class="${{klass(unrealized)}}">${{sign(unrealized)}} ${{money(unrealized, currency)}}</span> · Realized: <span class="${{klass(realized)}}">${{sign(realized)}} ${{money(realized, currency)}}</span></div>
        </article>
        <article class="metric-card">
          <div class="metric-label"><span class="metric-icon" style="color:var(--green)">$</span> Dividend <span class="info">●</span></div>
          <div class="metric-value">${{pct(p.dividend_yield_pct)}}</div>
          <div class="metric-sub">Yield on Cost: ${{pct(p.dividend_yield_on_cost_pct ?? p.dividend_yield_pct)}} <span class="info">●</span></div>
        </article>`;
    }}
    function renderGoalPanel() {{
      const goal = state?.goal_plan;
      const panel = document.getElementById("goalPanel");
      if (!panel || !goal) {{
        if (panel) panel.innerHTML = "";
        return;
      }}
      const currency = "EUR";
      const statusLabel = String(goal.status || "monitoring").replace(/_/g, " ");
      const remaining = Number(goal.remaining_gain || 0);
      const current = Number(goal.current_gain || 0);
      const cashCushion = Number(goal.cash_cushion || 0);
      const stopGain = Number(goal.stop_adding_risk_at_gain || 0);
      const cleanup = state?.weekly_cleanup_review || {{}};
      const candidates = Array.isArray(cleanup.cleanup_candidates) ? cleanup.cleanup_candidates : [];
      const notUrgent = Array.isArray(cleanup.not_urgent) ? cleanup.not_urgent : [];
      const candidateRows = candidates.length ? candidates.map((row) => `
        <tr>
          <td><span class="cleanup-symbol">${{escapeHtml(row.symbol || "-")}}</span><div class="perf-sub">${{escapeHtml(row.name || "")}}</div></td>
          <td>${{money(row.current_value, currency)}}</td>
          <td class="${{klass(row.unrealized_pl_pct)}}">${{pct(row.unrealized_pl_pct)}}</td>
          <td class="cleanup-action">${{escapeHtml(row.plain_action || "Review on cleanup date.")}}</td>
        </tr>`).join("") : `<tr><td colspan="4" class="perf-sub">No tiny losing cleanup candidates in latest review.</td></tr>`;
      const notUrgentText = notUrgent.length
        ? notUrgent.slice(0, 6).map((row) => `${{escapeHtml(row.symbol || "-")}} (${{money(row.current_value, currency)}}, ${{pct(row.unrealized_pl_pct)}})`).join(", ")
        : "None right now.";
      panel.innerHTML = `
        <article class="goal-panel">
          <div class="goal-header">
            <div>
              <h2 class="goal-title">${{escapeHtml(goal.name || "100 EUR Net Gain Plan")}}</h2>
              <div class="goal-note">Read-only monitor. Paper portfolios are excluded. Deposits do not count as profit.</div>
            </div>
            <div class="goal-status">${{escapeHtml(statusLabel)}}</div>
          </div>
          <div class="goal-grid">
            <div class="goal-cell">
              <div class="goal-label">Current gain</div>
              <div class="goal-value ${{klass(current)}}">${{sign(current)}}${{money(current, currency)}}</div>
              <div class="goal-note">Target is ${{money(goal.target_gain, currency)}} net gain.</div>
            </div>
            <div class="goal-cell">
              <div class="goal-label">Remaining target</div>
              <div class="goal-value">${{money(remaining, currency)}}</div>
              <div class="goal-note">${{pct(goal.return_needed_pct)}} return still needed from current value.</div>
            </div>
            <div class="goal-cell">
              <div class="goal-label">Value needed</div>
              <div class="goal-value">${{money(goal.target_total_worth, currency)}}</div>
              <div class="goal-note">Assumes no new deposits count as profit.</div>
            </div>
            <div class="goal-cell">
              <div class="goal-label">Drawdown budget</div>
              <div class="goal-value negative">${{money(goal.drawdown_budget, currency)}}</div>
              <div class="goal-note">Stop adding risk if gain reaches ${{money(stopGain, currency)}}.</div>
            </div>
            <div class="goal-cell">
              <div class="goal-label">Cash floor</div>
              <div class="goal-value ${{klass(cashCushion)}}">${{money(goal.cash, currency)}}</div>
              <div class="goal-note">Floor: ${{money(goal.cash_floor_value, currency)}}. Cushion: ${{money(cashCushion, currency)}}.</div>
            </div>
          </div>
          <div class="goal-action-box">
            <div class="goal-action-card warning">
              <div class="goal-action-title">Next cleanup review: ${{escapeHtml(cleanup.review_date || "next Tuesday")}}</div>
              <div class="goal-note">${{escapeHtml(cleanup.summary?.plain || "Later means the next planned weekly cleanup review, not a random day.")}}</div>
              <ul class="goal-action-list">
                <li>Refresh portfolio first.</li>
                <li>Look at tiny positions under 2 EUR/USD.</li>
                <li>If still under 2 and still worse than -7%, mark for sell/cleanup.</li>
                <li>If fee or spread costs more than saved value, leave it alone, but never add.</li>
                <li>Until then: do not add more money to tiny losing positions.</li>
              </ul>
              <div class="goal-note">Automation: ${{escapeHtml(cleanup.automation_id || "weekly-portfolio-cleanup-review")}} · ${{escapeHtml(cleanup.schedule || "Every Tuesday 09:00 Amsterdam time")}}</div>
            </div>
            <div class="goal-action-card">
              <div class="goal-action-title">Likely cleanup candidates</div>
              <div class="goal-note">Action is not a trade order. It is a checklist for the review date.</div>
              <div class="cleanup-table-wrap">
                <table class="cleanup-table">
                  <thead><tr><th>Ticker</th><th>Value</th><th>P/L %</th><th>Specific action</th></tr></thead>
                  <tbody>${{candidateRows}}</tbody>
                </table>
              </div>
              <div class="goal-note"><strong>Not urgent:</strong> ${{notUrgentText}}</div>
            </div>
          </div>
        </article>`;
    }}
    function renderTabs() {{
      document.getElementById("tabs").innerHTML = tabs.map(([key, icon, label]) => `
        <button class="tab ${{activeTab === key ? "active" : ""}}" data-tab="${{key}}"><span>${{icon}}</span>${{label}}</button>
      `).join("");
      document.querySelectorAll(".tab").forEach((button) => button.addEventListener("click", () => {{
        activeTab = button.dataset.tab;
        history.replaceState(null, "", `#${{activeTab}}`);
        render();
      }}));
    }}
    function renderPortfolioCard(p) {{
      const currency = p.currency || "EUR";
      const active = p.key === selectedKey ? "active" : "";
      const paper = p.is_paper ? "paper" : "";
      const icon = p.kind === "combined" ? "▣" : "▣";
      const title = p.kind === "combined" ? "Portfolio Combined Total Worth" : p.name;
      const note = p.note ? `<div class="portfolio-note">${{p.note}}</div>` : "";
      const holdingsCost = Number(p.holdings_cost_basis ?? p.initially_invested ?? 0);
      const edit = p.kind === "combined" ? "" : `<div class="portfolio-edit">✎</div>`;
      return `
        <button class="portfolio-card ${{active}} ${{paper}}" data-portfolio="${{p.key}}">
          <div class="portfolio-title"><span class="metric-icon">${{icon}}</span><span>${{title}}</span></div>
          ${{edit}}
          <div class="portfolio-worth">${{money(p.total_worth, currency)}}</div>
          <div class="portfolio-lines">
            <div><strong>Equities:</strong> ${{money(p.equities, currency)}}</div>
            <div><strong>Cash:</strong> ${{money(p.cash, currency)}}</div>
          </div>
          <div class="portfolio-muted">Invested capital: ${{money(p.initially_invested, currency)}}</div>
          <div class="portfolio-muted">Holdings cost basis: ${{money(holdingsCost, currency)}}</div>
          <div class="portfolio-note">${{p.include_in_combined ? "Included in combined portfolio" : "Not included in combined portfolio"}}</div>
          ${{note}}
        </button>`;
    }}
    function renderPortfoliosTab() {{
      const portfolios = portfolioList();
      document.getElementById("tabContent").innerHTML = `
        <section class="portfolio-grid">${{portfolios.map(renderPortfolioCard).join("")}}</section>`;
      document.querySelectorAll(".portfolio-card").forEach((card) => card.addEventListener("click", () => {{
        selectedKey = card.dataset.portfolio;
        localStorage.setItem("selectedPortfolioKey", selectedKey);
        render();
      }}));
    }}
    function selectedHoldings() {{
      const p = selectedPortfolio();
      const holdings = state?.holdings || [];
      if (p.kind === "combined") {{
        const includedKeys = new Set(portfolioList().filter((item) => item.include_in_combined && item.kind !== "combined").map((item) => item.key));
        return holdings.filter((row) => includedKeys.has(row.account_key));
      }}
      return holdings.filter((row) => row.account_key === p.key);
    }}
    function selectedDividends() {{
      const p = selectedPortfolio();
      const dividends = state?.dividends || [];
      if (p.kind === "combined") {{
        const includedKeys = new Set(portfolioList().filter((item) => item.include_in_combined && item.kind !== "combined").map((item) => item.key));
        return dividends.filter((row) => includedKeys.has(row.account_key));
      }}
      return dividends.filter((row) => row.account_key === p.key);
    }}
    function assetCategory(holding) {{
      const cls = holding.classification || {{}};
      if (cls.asset_class) return cls.asset_class;
      const type = String(holding.asset_type || "Other").toLowerCase();
      if (type.includes("etf")) return "ETF";
      if (type.includes("crypto")) return "Crypto";
      if (type.includes("fund")) return "Fund";
      if (type.includes("bond")) return "Bond";
      if (type.includes("cash")) return "Cash";
      return "Stock";
    }}
    function assetTypeLabel(holding) {{
      const category = assetCategory(holding);
      if (category !== "Stock") return category;
      const type = String(holding.asset_type || "").toLowerCase();
      if (type.includes("adr")) return "ADR";
      if (type.includes("security")) return "Security";
      if (type.includes("stock")) return "Stock";
      return category;
    }}
    function countryFor(holding) {{
      const cls = holding.classification || {{}};
      if (cls.geography) return cls.geography;
      const raw = holding.raw || {{}};
      const instrument = raw.instrument || {{}};
      const exchange = String(instrument.exchange || "").toUpperCase();
      const currency = String(holding.currency || instrument.currency || "").toUpperCase();
      const isin = String(holding.isin || "").toUpperCase();
      const symbol = String(holding.ticker || "").toUpperCase();
      const isinCountry = {{ US: "United States", DE: "Germany", NL: "Netherlands", IE: "Ireland", FR: "France", GB: "United Kingdom", CA: "Canada", TW: "Taiwan", JP: "Japan", CH: "Switzerland" }};
      const exchangeCountry = {{ XNAS: "United States", XNYS: "United States", ARC: "United States", XETR: "Germany", XPAR: "France", XLON: "United Kingdom", XAMS: "Netherlands" }};
      if (isinCountry[isin.slice(0, 2)]) return isinCountry[isin.slice(0, 2)];
      if (exchangeCountry[exchange]) return exchangeCountry[exchange];
      if (symbol.endsWith(".DE")) return "Germany";
      if (currency === "USD") return "United States";
      if (currency === "EUR") return "Europe";
      return "Unknown";
    }}
    function sectorFor(holding) {{
      const cls = holding.classification || {{}};
      if (cls.sector) return cls.sector;
      const raw = holding.raw || {{}};
      const instrument = raw.instrument || {{}};
      const direct = raw.sector || instrument.sector || instrument.industry;
      if (direct) return String(direct);
      const text = `${{holding.name || ""}} ${{holding.ticker || ""}}`.toLowerCase();
      if (/asml|semiconductor|nvidia|tsmc|taiwan semiconductor/.test(text)) return "Semiconductors";
      if (/airbus|lockheed|raytheon|rtx|rheinmetall|leidos|defence|defense|heico|3m/.test(text)) return "Industrials";
      if (/rare earth|metals|antimony|niocorp|energy fuels|resources|tmc|idaho/.test(text)) return "Materials";
      if (/lilly|eli/.test(text)) return "Healthcare";
      if (/bigbear|crowdstrike|automation|keysight/.test(text)) return "Technology";
      if (/etf|ishares|vaneck|vanguard/.test(text)) return "ETF / Fund";
      return "Unclassified";
    }}
    function allocationKeyFor(holding, mode) {{
      if (mode === "stock") return holding.name || holding.ticker || "Unknown";
      if (mode === "asset") return assetTypeLabel(holding);
      if (mode === "country") return countryFor(holding);
      if (mode === "sector") return sectorFor(holding);
      if (mode === "broker") return holding.institution || holding.broker || "Unknown";
      if (mode === "currency") return holding.currency || "Unknown";
      return assetCategory(holding);
    }}
    function cashAllocationLabel(portfolio, mode) {{
      if (mode === "broker") return portfolio.institution || portfolio.name || "Cash";
      if (mode === "currency") return portfolio.currency || "Unknown";
      if (mode === "country") {{
        if (portfolio.currency === "USD") return "United States";
        if (portfolio.currency === "EUR") return "Europe";
      }}
      return "Cash";
    }}
    function allocationRows(mode = allocationMode) {{
      const p = selectedPortfolio();
      const rows = new Map();
      for (const holding of selectedHoldings()) {{
        const label = allocationKeyFor(holding, mode);
        const value = Number(holding.current_value || 0);
        const invested = Number(holding.cost_basis || 0);
        const existing = rows.get(label) || {{ name: label, value: 0, invested: 0, ticker: holding.ticker, currency: holding.currency }};
        existing.value += value;
        existing.invested += invested;
        rows.set(label, existing);
      }}
      const cashSources = p.kind === "combined"
        ? portfolioList().filter((item) => item.include_in_combined && item.kind !== "combined")
        : [p];
      for (const source of cashSources) {{
        if (Number(source.cash || 0) <= 0) continue;
        const cashLabel = cashAllocationLabel(source, mode);
        const existing = rows.get(cashLabel) || {{ name: cashLabel, value: 0, invested: 0 }};
        existing.value += Number(source.cash || 0);
        existing.invested += Number(source.cash || 0);
        rows.set(cashLabel, existing);
      }}
      const total = Array.from(rows.values()).reduce((sum, row) => sum + row.value, 0);
      return Array.from(rows.values()).filter((row) => row.value > 0).sort((a, b) => b.value - a.value).map((row) => ({{
        ...row,
        pct: total ? row.value / total * 100 : 0,
        gainPct: row.invested ? (row.value - row.invested) / row.invested * 100 : 0,
      }}));
    }}
    function allocationGradient(rows) {{
      if (!rows.length) return "#263142";
      const colors = ["#4d83ff", "#1f4d94", "#102b59", "#739fff", "#2c6ed8", "#384257"];
      let cursor = 0;
      return `conic-gradient(${{rows.map((row, index) => {{
        const start = cursor;
        cursor += row.pct;
        return `${{colors[index % colors.length]}} ${{start.toFixed(2)}}% ${{cursor.toFixed(2)}}%`;
      }}).join(", ")}})`;
    }}
    function renderPerformanceStrip() {{
      const p = selectedPortfolio();
      const currency = p.currency || "EUR";
      const daily = Number(p.today_gain || 0);
      const total = Number(p.capital_gain || 0);
      return `
        <section class="performance-strip">
          <div class="performance-grid">
            <div class="perf-cell perf-head">Daily<br>Profit/Loss</div>
            <div class="perf-cell perf-head">MTD vs VOO <span class="info">●</span></div>
            <div class="perf-cell perf-head">Weekly vs<br>VOO <span class="info">●</span></div>
            <div class="perf-cell perf-head">Initial<br>Balance</div>
            <div class="perf-cell perf-head">Current<br>Balance</div>
            <div class="perf-cell perf-head">Total Return</div>
            <div class="perf-cell perf-head">Capital<br>Return</div>
            <div class="perf-cell perf-head">Realized<br>Return</div>
            <div class="perf-cell perf-head">Unrealized<br>Return</div>
            <div class="perf-cell perf-head">IRR for<br>Dividends</div>
            <div class="perf-cell perf-head">Dividend<br>Yield</div>
            <div class="perf-cell"><div class="perf-value ${{klass(daily)}}">${{Number(daily) >= 0 ? "↑" : "↓"}} ${{money(daily, currency)}}</div><div class="perf-sub">(${{pct(p.today_gain_pct)}})</div></div>
            <div class="perf-cell"><div class="perf-value negative">-</div><div class="perf-sub">vs benchmark</div></div>
            <div class="perf-cell"><div class="perf-value negative">-</div><div class="perf-sub">vs benchmark</div></div>
            <div class="perf-cell"><div class="perf-value">${{money(p.initially_invested, currency)}}</div></div>
            <div class="perf-cell"><div class="perf-value">${{money(p.total_worth, currency)}}</div></div>
            <div class="perf-cell"><div class="perf-value ${{klass(total)}}">${{Number(total) >= 0 ? "↑" : "↓"}} ${{money(total, currency)}}</div><div class="perf-sub">(${{pct(p.simple_return_pct)}})</div></div>
            <div class="perf-cell"><div class="perf-value ${{klass(total)}}">${{Number(total) >= 0 ? "↑" : "↓"}} ${{money(total, currency)}}</div><div class="perf-sub">(${{pct(p.simple_return_pct)}})</div></div>
            <div class="perf-cell"><div class="perf-value">0.00 ${{currency}}</div><div class="perf-sub">(0.00%)</div></div>
            <div class="perf-cell"><div class="perf-value ${{klass(total)}}">${{Number(total) >= 0 ? "↑" : "↓"}} ${{money(total, currency)}}</div><div class="perf-sub">(${{pct(p.simple_return_pct)}})</div></div>
            <div class="perf-cell"><div class="perf-value">-</div></div>
            <div class="perf-cell"><div class="perf-value">${{pct(p.dividend_yield_pct)}} <span class="info">●</span></div></div>
          </div>
        </section>`;
    }}
    function allocationModeLabel(mode) {{
      return {{
        category: "By Category",
        stock: "By Stock",
        asset: "By Asset",
        country: "By Country",
        sector: "By Sector",
        broker: "By Broker",
        currency: "By Currency",
      }}[mode] || "By Category";
    }}
    function stockInitials(row) {{
      const ticker = String(row.ticker || "").replace(/\\W+/g, "");
      if (ticker) return ticker.slice(0, 3).toUpperCase();
      return String(row.name || "?").split(/\\s+/).filter(Boolean).slice(0, 2).map((part) => part[0]).join("").toUpperCase() || "?";
    }}
    function renderStockCards(rows) {{
      if (!rows.length) return `<div class="empty-panel">No stock allocation data is available for this portfolio yet.</div>`;
      return `<div class="stock-grid">${{rows.map((row) => `
        <article class="stock-card">
          <div class="stock-title"><div class="stock-avatar">${{stockInitials(row)}}</div><div class="stock-name">${{row.name}}</div></div>
          <div class="stock-pct">${{pct(row.pct)}}</div>
          <div class="stock-bar"><div class="stock-fill" style="width:${{Math.max(2, Math.min(100, row.pct))}}%"></div></div>
        </article>`).join("")}}</div>`;
    }}
    function holdingDisplayName(holding) {{
      return holding.name || holding.ticker || holding.isin || "Unknown";
    }}
    function holdingInitials(holding) {{
      return stockInitials({{ name: holdingDisplayName(holding), ticker: holding.ticker }});
    }}
    function topMovers(direction) {{
      const rows = selectedHoldings()
        .filter((holding) => Number.isFinite(Number(holding.unrealized_pl)))
        .map((holding) => ({{
          ...holding,
          pl: Number(holding.unrealized_pl || 0),
          plPct: Number(holding.unrealized_pl_pct || 0),
          value: Number(holding.current_value || 0),
          currency: holding.currency || selectedPortfolio().currency || "EUR",
        }}));
      return rows
        .filter((holding) => direction === "gain" ? holding.pl >= 0 : holding.pl < 0)
        .sort((a, b) => direction === "gain" ? b.plPct - a.plPct : a.plPct - b.plPct)
        .slice(0, 8);
    }}
    function renderMoverRows(rows, emptyText) {{
      if (!rows.length) return `<div class="empty-panel">${{emptyText}}</div>`;
      return `<div class="market-list">${{rows.map((row) => `
        <div class="market-row">
          <div class="market-name">
            <div class="stock-avatar">${{holdingInitials(row)}}</div>
            <div class="market-name-text">${{holdingDisplayName(row)}}</div>
          </div>
          <div class="market-values ${{klass(row.pl)}}">
            <div>${{money(row.value, row.currency)}}</div>
            <div>${{sign(row.pl)}}${{money(row.pl, row.currency)}}</div>
            <div>(${{pct(row.plPct)}})</div>
          </div>
        </div>`).join("")}}</div>`;
    }}
    function recentTransactions() {{
      return selectedHoldings()
        .filter((holding) => Number(holding.quantity || 0) > 0)
        .slice()
        .sort((a, b) => Number(b.current_value || 0) - Number(a.current_value || 0))
        .slice(0, 10)
        .map((holding) => ({{
          name: holding.ticker || holdingDisplayName(holding),
          label: holdingDisplayName(holding),
          date: (state?.source_timestamps?.snaptrade || state?.source_timestamps?.trade_republic || state?.generated_at || "").slice(0, 10),
          quantity: holding.quantity,
          price: holding.current_price,
          currency: holding.currency || selectedPortfolio().currency || "EUR",
          type: "Position",
        }}));
    }}
    function renderTransactionRows(rows) {{
      if (!rows.length) return `<div class="empty-panel">No recent transaction rows are available for this portfolio yet.</div>`;
      return `<div class="market-list">${{rows.map((row) => `
        <div class="market-row transaction-row">
          <div class="market-name">
            <div class="stock-avatar">${{stockInitials({{ name: row.name, ticker: row.name }})}}</div>
            <div>
              <div class="market-name-text">${{row.name}}</div>
              <div class="perf-sub">${{row.date || "-"}}</div>
              <div class="badge">${{row.type}}</div>
            </div>
          </div>
          <div class="market-values">
            <div>Quantity:</div>
            <div class="perf-sub">${{Number(row.quantity || 0).toFixed(6)}}</div>
          </div>
          <div class="market-values">
            <div>Price:</div>
            <div class="perf-sub">${{money(row.price, row.currency)}}</div>
          </div>
        </div>`).join("")}}</div>`;
    }}
    function renderBottomPanels() {{
      return `
        <section class="dashboard-bottom">
          <article class="market-panel">
            <div class="market-header"><h2 class="market-title">Daily Top Gainers</h2><span class="market-close">×</span></div>
            ${{renderMoverRows(topMovers("gain"), "No gainers in this selected portfolio yet.")}}
          </article>
          <article class="market-panel">
            <div class="market-header"><h2 class="market-title">Daily Top Losers</h2><span class="market-close">×</span></div>
            ${{renderMoverRows(topMovers("loss"), "No losers in this selected portfolio yet.")}}
          </article>
          <article class="market-panel">
            <div class="market-header"><h2 class="market-title">Recent Transactions</h2><span class="market-close">×</span></div>
            ${{renderTransactionRows(recentTransactions())}}
          </article>
        </section>`;
    }}
    function heatmapReturnValue(holding, type) {{
      if (type === "daily") return Number(holding.today_gain_pct || holding.daily_return_pct || 0);
      const total = Number(holding.unrealized_pl_pct || 0);
      if (type === "annual") return total;
      return total;
    }}
    function heatColor(value) {{
      const bounded = Math.max(-20, Math.min(20, Number(value || 0)));
      if (Math.abs(bounded) < 0.05) return "#43525b";
      if (bounded > 0) {{
        const alpha = 0.48 + Math.min(0.42, bounded / 45);
        return `rgba(22, 169, 119, ${{alpha.toFixed(2)}})`;
      }}
      const alpha = 0.48 + Math.min(0.42, Math.abs(bounded) / 45);
      return `rgba(196, 70, 63, ${{alpha.toFixed(2)}})`;
    }}
    function heatmapRows() {{
      const holdings = selectedHoldings()
        .filter((holding) => Number(holding.current_value || 0) > 0)
        .slice()
        .sort((a, b) => Number(b.current_value || 0) - Number(a.current_value || 0))
        .slice(0, 30);
      const maxValue = Math.max(1, ...holdings.map((holding) => Number(holding.current_value || 0)));
      return holdings.map((holding) => {{
        const value = Number(holding.current_value || 0);
        const span = Math.max(1, Math.min(6, Math.ceil(value / maxValue * 6)));
        const returnValue = heatmapReturnValue(holding, heatmapReturnType);
        return {{
          name: holdingDisplayName(holding),
          ticker: holding.ticker,
          value,
          span,
          returnValue,
          color: heatColor(returnValue),
        }};
      }});
    }}
    function heatmapReturnLabel() {{
      return {{
        daily: "Daily Return",
        total: "Total Return",
        annual: "Annualized Return",
      }}[heatmapReturnType] || "Daily Return";
    }}
    function renderHeatmapPanel() {{
      const rows = heatmapRows();
      const cells = rows.length ? rows.map((row) => `
        <div class="heatmap-cell" style="--span:${{row.span}}; --heat:${{row.color}}">
          <div class="heatmap-label">${{row.name}}<br>${{pct(row.returnValue)}}</div>
        </div>`).join("") : `<div class="empty-panel">No holdings are available for the heatmap.</div>`;
      return `
        <section class="heatmap-panel">
          <div class="heatmap-header">Stock Performance Heatmap</div>
          <div class="heatmap-subtitle">${{heatmapReturnLabel()}}</div>
          <div class="heatmap-grid">${{cells}}</div>
          <div class="heatmap-controls">
            <label for="returnType">Select Return Type: <span class="info">●</span></label>
            <select id="returnType" class="return-select">
              <option value="daily" ${{heatmapReturnType === "daily" ? "selected" : ""}}>Daily Return</option>
              <option value="total" ${{heatmapReturnType === "total" ? "selected" : ""}}>Total Return</option>
              <option value="annual" ${{heatmapReturnType === "annual" ? "selected" : ""}}>Annualized Return</option>
            </select>
          </div>
        </section>`;
    }}
    function holdingPrice(holding) {{
      const direct = Number(holding.current_price);
      if (Number.isFinite(direct)) return direct;
      const value = Number(holding.current_value || 0);
      const quantity = Math.abs(Number(holding.quantity || 0));
      return quantity ? value / quantity : null;
    }}
    function averageCost(holding) {{
      const cost = Number(holding.cost_basis || 0);
      const quantity = Math.abs(Number(holding.quantity || 0));
      return quantity ? cost / quantity : null;
    }}
    function miniSparkline(holding) {{
      const current = Number(holdingPrice(holding) || 0);
      const avg = Number(averageCost(holding) || current || 0);
      const gainPct = Number(holding.unrealized_pl_pct || 0);
      const values = [avg, avg * .995, avg * (1 + gainPct / 350), avg * (1 + gainPct / 220), avg * (1 + gainPct / 160), current || avg].filter(Number.isFinite);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = Math.max(.0001, max - min);
      const width = 104;
      const height = 30;
      const path = values.map((value, index) => {{
        const x = 2 + index / Math.max(1, values.length - 1) * (width - 4);
        const y = height - 3 - ((value - min) / span) * (height - 6);
        return `${{index === 0 ? "M" : "L"}} ${{x.toFixed(1)}} ${{y.toFixed(1)}}`;
      }}).join(" ");
      return `<svg class="sparkline ${{gainPct < 0 ? "negative" : ""}}" viewBox="0 0 ${{width}} ${{height}}" aria-label="Mini price trend"><path d="${{path}}"></path></svg>`;
    }}
    function actionLabel(action) {{
      return {{
        hold_core: "Hold core",
        watch_opportunity: "Watch setup",
        cleanup_watch: "Cleanup watch",
        review_loss: "Review loss",
        no_trade: "No trade",
      }}[action] || "No trade";
    }}
    function renderGoalAction(holding) {{
      const action = holding.goal_action || {{}};
      const bucket = action.bucket || "watch";
      const reasons = Array.isArray(action.reasons) ? action.reasons : [];
      return `
        <span class="action-pill ${{escapeHtml(bucket)}}">${{escapeHtml(actionLabel(action.action))}}</span>
        <div class="action-reasons">${{reasons.map(escapeHtml).join(" ") || "No automatic add rule applies."}}</div>`;
    }}
    function renderHoldingsTablePanel() {{
      const p = selectedPortfolio();
      const currency = p.currency || "EUR";
      const rows = selectedHoldings()
        .filter((holding) => Number(holding.current_value || 0) !== 0 || Number(holding.quantity || 0) !== 0)
        .slice()
        .sort((a, b) => Number(b.current_value || 0) - Number(a.current_value || 0));
      const body = rows.length ? rows.map((holding) => {{
        const rowCurrency = holding.currency || currency;
        const avg = averageCost(holding);
        const price = holdingPrice(holding);
        const unrealized = Number(holding.unrealized_pl || 0);
        const daily = Number(holding.today_gain || 0);
        return `
          <tr>
            <td>
              <div class="holding-cell">
                <div class="holding-logo">${{holdingInitials(holding)}}</div>
                <div>
                  <div class="holding-name" title="${{holdingDisplayName(holding)}}">${{holdingDisplayName(holding)}}</div>
                  <div class="holding-sub">${{holding.ticker || holding.broker_symbol || holding.isin || "-"}} · ${{holding.institution || holding.broker || "-"}}</div>
                </div>
              </div>
            </td>
            <td>${{Number(holding.quantity || 0).toFixed(6)}}</td>
            <td>${{avg === null ? "-" : money(avg, rowCurrency)}}</td>
            <td>${{price === null ? "-" : money(price, rowCurrency)}}</td>
            <td>${{money(holding.current_value, rowCurrency)}}</td>
            <td class="${{klass(unrealized)}}">${{sign(unrealized)}}${{money(unrealized, rowCurrency)}}<div class="holding-sub">${{pct(holding.unrealized_pl_pct)}}</div></td>
            <td class="${{klass(daily)}}">${{sign(daily)}}${{money(daily, rowCurrency)}}<div class="holding-sub">${{pct(holding.today_gain_pct || 0)}}</div></td>
            <td>${{holding.broker || "-"}}</td>
            <td>${{rowCurrency}}</td>
            <td>${{renderGoalAction(holding)}}</td>
            <td>${{miniSparkline(holding)}}</td>
          </tr>`;
      }}).join("") : `<tr><td colspan="11"><div class="empty-panel">No holdings are available for this selected portfolio yet.</div></td></tr>`;
      return `
        <section class="holdings-panel">
          <div class="holdings-header">
            <div>
              <h2 class="holdings-title">Holdings</h2>
              <div class="analytics-subtitle">${{p.name || "Selected portfolio"}} · sortable table wiring comes next</div>
            </div>
            <div class="holdings-tools"><span class="perf-sub">${{rows.length}} rows</span></div>
          </div>
          <div class="holdings-table-wrap">
            <table class="holdings-table">
              <thead><tr><th>Holding</th><th>Quantity</th><th>Avg Cost</th><th>Price</th><th>Market Value</th><th>Unrealized P/L</th><th>Daily P/L</th><th>Broker</th><th>Currency</th><th>Goal Action</th><th>Sparkline</th></tr></thead>
              <tbody>${{body}}</tbody>
            </table>
          </div>
        </section>`;
    }}
    function selectedRebalancing() {{
      const payload = state?.rebalancing || {{}};
      return payload[selectedKey] || payload[state?.combined_portfolio?.key] || null;
    }}
    function rebalanceColor(index) {{
      return ["#4d83ff", "#18be8a", "#f4a20b", "#8b5cf6", "#14b8c7", "#ef5350", "#6b7280"][index % 7];
    }}
    function rebalanceDonutGradient(rows, field) {{
      const cleaned = (rows || []).filter((row) => Number(row[field] || 0) > 0);
      if (!cleaned.length) return "#263142";
      let cursor = 0;
      return `conic-gradient(${{cleaned.map((row, index) => {{
        const pctValue = Math.max(0, Number(row[field] || 0));
        const start = cursor;
        cursor += pctValue;
        return `${{rebalanceColor(index)}} ${{start.toFixed(2)}}% ${{Math.min(100, cursor).toFixed(2)}}%`;
      }}).join(", ")}})`;
    }}
    function rebalancePlan(rows, portfolioValue, amount, operation, allowSelling, minimizeTransactions, pricePriority) {{
      const safeRows = (rows || []).map((row) => ({{
        ...row,
        current_value: Number(row.current_value || 0),
        current_pct: Number(row.current_pct || 0),
        target_pct: Number(row.target_pct || 0),
        current_price: Number(row.current_price || 0),
      }}));
      const cashAmount = Math.max(0, Number(amount || 0));
      const startValue = Math.max(0, Number(portfolioValue || 0));
      const finalValue = Math.max(0, operation === "withdrawal" ? startValue - cashAmount : startValue + cashAmount);
      const targetRows = safeRows.map((row) => {{
        const targetValue = finalValue * row.target_pct / 100;
        const gap = targetValue - row.current_value;
        return {{ ...row, target_value_after_cash: targetValue, raw_trade_value: gap }};
      }});
      const threshold = minimizeTransactions ? Math.max(1, finalValue * 0.005) : 0.01;
      let buyBudget = operation === "deposit" ? cashAmount : 0;
      let sellBudget = operation === "withdrawal" ? cashAmount : 0;
      let planned = targetRows.map((row) => ({{
        ...row,
        trade_value: allowSelling ? row.raw_trade_value : 0,
      }}));
      if (!allowSelling) {{
        if (operation === "deposit") {{
          const candidates = targetRows.filter((row) => row.raw_trade_value > threshold);
          candidates.sort((a, b) => pricePriority
            ? ((a.current_price || Number.MAX_SAFE_INTEGER) - (b.current_price || Number.MAX_SAFE_INTEGER))
            : b.raw_trade_value - a.raw_trade_value);
          const totalNeed = candidates.reduce((sum, row) => sum + Math.max(0, row.raw_trade_value), 0);
          const scale = totalNeed > 0 ? Math.min(1, buyBudget / totalNeed) : 0;
          planned = targetRows.map((row) => {{
            const match = candidates.find((item) => item.symbol === row.symbol);
            return {{ ...row, trade_value: match ? Math.max(0, row.raw_trade_value) * scale : 0 }};
          }});
        }} else {{
          const candidates = targetRows.filter((row) => row.raw_trade_value < -threshold).sort((a, b) => a.raw_trade_value - b.raw_trade_value);
          const totalNeed = candidates.reduce((sum, row) => sum + Math.abs(row.raw_trade_value), 0);
          const scale = totalNeed > 0 ? Math.min(1, sellBudget / totalNeed) : 0;
          planned = targetRows.map((row) => {{
            const match = candidates.find((item) => item.symbol === row.symbol);
            return {{ ...row, trade_value: match ? -Math.abs(row.raw_trade_value) * scale : 0 }};
          }});
        }}
      }}
      planned = planned.map((row) => {{
        const small = Math.abs(row.trade_value) < threshold;
        const tradeValue = small ? 0 : row.trade_value;
        const expectedValue = Math.max(0, row.current_value + tradeValue);
        const expectedPct = finalValue ? expectedValue / finalValue * 100 : 0;
        const qty = row.current_price ? Math.abs(tradeValue) / row.current_price : null;
        return {{
          ...row,
          trade_value: tradeValue,
          expected_value: expectedValue,
          expected_pct: expectedPct,
          suggested_action: tradeValue > 0 ? "Buy" : tradeValue < 0 ? "Sell" : "Hold",
          suggested_quantity: qty,
        }};
      }});
      if (pricePriority) {{
        const buyRows = planned.filter((row) => row.suggested_action === "Buy" && row.current_price > 0);
        const cheapestPrice = buyRows.length ? Math.min(...buyRows.map((row) => row.current_price)) : null;
        planned = planned.map((row) => ({{
          ...row,
          cheapest: cheapestPrice !== null && row.suggested_action === "Buy" && row.current_price === cheapestPrice,
        }}));
      }}
      const buyTotal = planned.reduce((sum, row) => sum + Math.max(0, row.trade_value), 0);
      const sellTotal = planned.reduce((sum, row) => sum + Math.abs(Math.min(0, row.trade_value)), 0);
      return {{
        rows: planned,
        finalValue,
        buyTotal,
        sellTotal,
        buyCount: planned.filter((row) => row.suggested_action === "Buy").length,
        sellCount: planned.filter((row) => row.suggested_action === "Sell").length,
        tradeCount: planned.filter((row) => row.suggested_action !== "Hold").length,
        threshold,
      }};
    }}
    function renderRebalancePanel() {{
      const rb = selectedRebalancing();
      const p = selectedPortfolio();
      const currency = rb?.currency || p.currency || "EUR";
      const rows = rb?.rows || [];
      const targetTotal = rows.reduce((sum, row) => sum + Number(row.target_pct || 0), 0);
      const driftAbsPct = rows.length ? rows.reduce((sum, row) => sum + Math.abs(Number(row.drift_pct || 0)), 0) / rows.length : 0;
      const currentRows = rows.map((row) => ({{
        name: row.name,
        symbol: row.symbol,
        current_pct: Number(row.current_pct || 0),
        target_pct: Number(row.target_pct || 0),
        drift_pct: Number(row.drift_pct || 0),
      }}));
      const plan = rebalancePlan(rows, Number(rb?.portfolio_value || 0), rebalanceAmount, rebalanceOperation, rebalanceAllowSelling, rebalanceMinimizeTransactions, rebalancePricePriority);
      const actionsText = `<span class="positive">${{plan.buyCount}} to buy</span> / <span class="negative">${{plan.sellCount}} to sell</span>`;
      const allocationRowsHtml = currentRows.length ? currentRows.map((row, index) => {{
        const barColor = rebalanceColor(index);
        const currentWidth = Math.max(2, Math.min(100, row.current_pct));
        const targetLeft = Math.max(0, Math.min(100, row.target_pct));
        const driftClass = row.drift_pct >= 0 ? "" : "negative";
        return `
          <div class="rebalance-allocation-row">
            <div class="rebalance-allocation-name"><span class="rebalance-chevron">›</span><span>${{row.name}}</span></div>
            <div class="rebalance-progress" style="--bar-color:${{barColor}}; --current-width:${{currentWidth}}%; --target-left:${{targetLeft}}%">
              <div class="rebalance-progress-fill"></div>
              <div class="rebalance-progress-target"></div>
            </div>
            <div>${{pct(row.current_pct)}}</div>
            <label class="rebalance-target-inline"><input class="target-input" data-target-symbol="${{row.symbol}}" value="${{Number(row.target_pct || 0).toFixed(2)}}"><span>%</span></label>
            <div><span class="rebalance-drift-pill ${{driftClass}}">${{sign(row.drift_pct)}}${{pct(row.drift_pct)}}</span></div>
          </div>`;
      }}).join("") : `<div class="empty-panel">No allocation rows are available for this selected portfolio yet.</div>`;
      const resultRows = plan.rows.length ? plan.rows.slice().sort((a, b) => Math.abs(b.trade_value) - Math.abs(a.trade_value)).map((row) => `
        <tr>
          <td>
            <div class="holding-cell">
              <div class="holding-logo">${{stockInitials({{ ticker: row.symbol, name: row.name }})}}</div>
              <div><div class="holding-name">${{row.name}}</div><div class="holding-sub">${{row.symbol}}</div></div>
            </div>
          </td>
          <td>${{pct(row.current_pct)}}</td>
          <td>${{pct(row.target_pct)}}</td>
          <td>${{money(row.current_value, row.currency || currency)}}</td>
          <td>${{money(row.target_value_after_cash, row.currency || currency)}}</td>
          <td><span class="rebalance-action-pill ${{row.suggested_action.toLowerCase()}}">${{row.suggested_action}}</span></td>
          <td class="${{klass(row.trade_value)}}">${{row.trade_value ? `${{sign(row.trade_value)}}${{money(row.trade_value, row.currency || currency)}}` : "-"}}</td>
          <td>${{row.suggested_quantity ? Number(row.suggested_quantity).toFixed(6) : "-"}}</td>
          <td>${{pct(row.expected_pct)}}</td>
          <td>${{row.current_price ? money(row.current_price, row.currency || currency) : "-"}}${{row.cheapest ? `<span class="rebalance-cheapest">Cheapest</span>` : ""}}</td>
        </tr>`).join("") : `<tr><td colspan="10"><div class="empty-panel">No rebalancing rows are available for this selected portfolio yet.</div></td></tr>`;
      return `
        <section class="rebalance-page">
          <div class="rebalance-title-block">
            <h2 class="rebalance-title">Portfolio Rebalancing</h2>
            <div class="panel-title">${{p.name || "Selected portfolio"}}</div>
            <div class="rebalance-subtitle">Recommendations are based on your <a href="javascript:void(0)">target allocations</a>. This is read-only analysis: it does not place broker orders.</div>
          </div>
          <div class="rebalance-summary">
            <article class="rebalance-summary-card"><div class="alert-label">Total Portfolio Value</div><div class="rebalance-summary-value">${{money(rb?.portfolio_value, currency)}}</div></article>
            <article class="rebalance-summary-card"><div class="alert-label">Average Drift</div><div class="rebalance-summary-value negative">${{pct(driftAbsPct)}}</div></article>
            <article class="rebalance-summary-card"><div class="alert-label">Actions Needed</div><div class="rebalance-summary-value">${{actionsText}}</div></article>
          </div>
          <div class="rebalance-workspace">
            <article class="rebalance-visual-panel">
              <div class="rebalance-chart-grid">
                <div>
                  <div class="rebalance-chart-title"><span class="metric-icon">◔</span>Current Allocation</div>
                  <div class="rebalance-donut" style="--donut:${{rebalanceDonutGradient(rows, "current_pct")}}">
                    <div class="rebalance-donut-center"><div class="rebalance-donut-value">${{money(rb?.portfolio_value, currency)}}</div><div class="rebalance-donut-label">Total</div></div>
                  </div>
                </div>
                <div>
                  <div class="rebalance-chart-title"><span class="metric-icon">◎</span>Target Allocation</div>
                  <div class="rebalance-donut" style="--donut:${{rebalanceDonutGradient(rows, "target_pct")}}">
                    <div class="rebalance-donut-center"><div class="rebalance-donut-value">${{pct(targetTotal)}}</div><div class="rebalance-donut-label">Target</div></div>
                  </div>
                </div>
              </div>
            </article>
            <aside class="rebalance-options-panel">
              <h3 class="panel-title"><span class="metric-icon">⚙</span> Rebalancing Options</h3>
              <label class="rebalance-option-field">
                <span class="rebalance-option-label">Amount (${{currency}})</span>
                <input id="rebalanceAmount" class="rebalance-amount-input" type="number" min="0" step="0.01" value="${{Number(rebalanceAmount || 0).toFixed(2)}}">
              </label>
              <div class="rebalance-option-field">
                <span class="rebalance-option-label">Transaction Type</span>
                <div class="rebalance-segment">
                  <button id="rebalanceDeposit" class="${{rebalanceOperation === "deposit" ? "active" : ""}}" type="button">↓ Deposit</button>
                  <button id="rebalanceWithdrawal" class="${{rebalanceOperation === "withdrawal" ? "active" : ""}}" type="button">↑ Withdrawal</button>
                </div>
              </div>
              <label class="rebalance-check"><input id="rebalanceAllowSelling" type="checkbox" ${{rebalanceAllowSelling ? "checked" : ""}}><span><span class="rebalance-check-title">Allow Selling</span><span class="rebalance-check-sub">Sell overallocated holdings to buy underallocated holdings.</span></span></label>
              <label class="rebalance-check"><input id="rebalanceMinimizeTransactions" type="checkbox" ${{rebalanceMinimizeTransactions ? "checked" : ""}}><span><span class="rebalance-check-title">Minimize Transactions</span><span class="rebalance-check-sub">Ignore very small adjustments and show the most meaningful trades.</span></span></label>
              <label class="rebalance-check"><input id="rebalancePricePriority" type="checkbox" ${{rebalancePricePriority ? "checked" : ""}}><span><span class="rebalance-check-title">Recommend by Price</span><span class="rebalance-check-sub">Prioritize cheaper underallocated holdings when using new cash.</span></span></label>
              <button class="rebalance-primary" id="rebalanceRunBtn" type="button">⚖ Rebalance</button>
              <button class="rebalance-secondary" type="button">▣ Compare Scenarios</button>
            </aside>
          </div>
          <article class="rebalance-breakdown-panel">
            <div class="rebalance-breakdown-header">
              <div>
                <h3 class="panel-title"><span class="metric-icon">☷</span> Allocation Breakdown</h3>
                <div class="rebalance-breakdown-sub">Viewing: target allocations - ${{p.name || "Selected portfolio"}}</div>
              </div>
              <div class="rebalance-breakdown-tools">
                <button class="rebalance-mode-chip active" type="button">Target Allocations</button>
                <button class="rebalance-mode-chip" type="button">Manage Categories</button>
                <button class="action-btn green" id="saveTargetsBtn">Save Targets</button>
              </div>
            </div>
            ${{allocationRowsHtml}}
            <div style="text-align:right; margin-top:14px;" class="${{Math.abs(targetTotal - 100) < 0.01 ? "positive" : "negative"}}">Sum: ${{pct(targetTotal)}}</div>
          </article>
          <article class="rebalance-results-panel">
            <div class="rebalance-results-head">
              <div>
                <h3 class="panel-title">Rebalance Recommendations</h3>
                <div class="rebalance-results-summary">Expected final value ${{money(plan.finalValue, currency)}} · Buy ${{money(plan.buyTotal, currency)}} · Sell ${{money(plan.sellTotal, currency)}} · ${{plan.tradeCount}} suggested actions</div>
              </div>
              <div class="analytics-subtitle">${{rb?.note || "Suggestions are estimates only and never place broker orders."}}</div>
            </div>
            <div class="rebalance-table-wrap">
            <table class="rebalance-table">
              <thead><tr><th>Asset</th><th>Current %</th><th>Target %</th><th>Current Value</th><th>Target Value</th><th>Action</th><th>Amount</th><th>Qty</th><th>Expected %</th><th>Price</th></tr></thead>
              <tbody>${{resultRows}}</tbody>
            </table>
            </div>
          </article>
        </section>`;
    }}
	    function renderDashboardTab() {{
	      const p = selectedPortfolio();
	      const currency = p.currency || "EUR";
      const rows = allocationRows(allocationMode);
      const list = rows.length ? rows.map((row) => `
        <div class="allocation-row">
          <div>
            <div class="allocation-name">${{row.name}}</div>
            <div class="allocation-pct">${{pct(row.pct)}}</div>
          </div>
          <div>
            <div class="allocation-bar"><div class="allocation-fill" style="width:${{Math.max(2, Math.min(100, row.pct))}}%"></div></div>
            <div class="perf-sub">Invested: ${{money(row.invested, currency)}} <span class="${{klass(row.gainPct)}}">(${{pct(row.gainPct)}})</span></div>
          </div>
          <div class="allocation-value">${{money(row.value, currency)}}</div>
        </div>`).join("") : `<div class="empty-panel">No allocation data is available for this portfolio yet.</div>`;
      const allocationTabs = ["category", "stock", "asset", "country", "sector", "broker", "currency"].map((mode) => `
        <button class="allocation-tab ${{allocationMode === mode ? "active" : ""}}" data-allocation-mode="${{mode}}">${{allocationModeLabel(mode)}}</button>
      `).join("");
      const allocationContent = allocationMode === "stock"
        ? renderStockCards(rows)
        : `<div class="allocation-body">
            <div class="donut" style="background:${{allocationGradient(rows)}}"></div>
            <div class="allocation-list">${{list}}</div>
          </div>`;
      document.getElementById("tabContent").innerHTML = `
        <div class="dashboard-stack">
          ${{renderPerformanceStrip()}}
          <section class="dashboard-main">
            <article class="dashboard-panel allocation-full">
              <div class="panel-header">
                <h2 class="panel-title">Asset Allocation <span class="metric-icon">↻</span></h2>
                <div class="manage-link">Manage Allocations ⚒</div>
              </div>
              <div class="allocation-tabs">${{allocationTabs}}</div>
              ${{allocationContent}}
            </article>
          </section>
          ${{renderRebalancePanel()}}
          ${{renderHoldingsTablePanel()}}
          ${{renderBottomPanels()}}
          ${{renderHeatmapPanel()}}
        </div>`;
      document.querySelectorAll(".allocation-tab").forEach((button) => button.addEventListener("click", () => {{
        allocationMode = button.dataset.allocationMode;
        renderDashboardTab();
        renderStatus();
      }}));
      const returnSelect = document.getElementById("returnType");
      if (returnSelect) returnSelect.addEventListener("change", () => {{
        heatmapReturnType = returnSelect.value;
        renderDashboardTab();
        renderStatus();
	      }});
      const saveTargetsBtn = document.getElementById("saveTargetsBtn");
      if (saveTargetsBtn) saveTargetsBtn.addEventListener("click", async () => {{
        const rows = Array.from(document.querySelectorAll("[data-target-symbol]")).map((input) => ({{
          symbol: input.dataset.targetSymbol,
          target_pct: Number(input.value || 0),
        }}));
        saveTargetsBtn.textContent = "Saving...";
        await apiPost("/api/allocation-targets", {{ portfolio_key: selectedKey, rows }});
        await load();
      }});
      const amountInput = document.getElementById("rebalanceAmount");
      if (amountInput) amountInput.addEventListener("change", () => {{
        rebalanceAmount = Number(amountInput.value || 0);
        renderDashboardTab();
        renderStatus();
      }});
      const depositBtn = document.getElementById("rebalanceDeposit");
      if (depositBtn) depositBtn.addEventListener("click", () => {{
        rebalanceOperation = "deposit";
        renderDashboardTab();
        renderStatus();
      }});
      const withdrawalBtn = document.getElementById("rebalanceWithdrawal");
      if (withdrawalBtn) withdrawalBtn.addEventListener("click", () => {{
        rebalanceOperation = "withdrawal";
        renderDashboardTab();
        renderStatus();
      }});
      const allowSelling = document.getElementById("rebalanceAllowSelling");
      if (allowSelling) allowSelling.addEventListener("change", () => {{
        rebalanceAllowSelling = allowSelling.checked;
        renderDashboardTab();
        renderStatus();
      }});
      const minimizeTransactions = document.getElementById("rebalanceMinimizeTransactions");
      if (minimizeTransactions) minimizeTransactions.addEventListener("change", () => {{
        rebalanceMinimizeTransactions = minimizeTransactions.checked;
        renderDashboardTab();
        renderStatus();
      }});
      const pricePriority = document.getElementById("rebalancePricePriority");
      if (pricePriority) pricePriority.addEventListener("change", () => {{
        rebalancePricePriority = pricePriority.checked;
        renderDashboardTab();
        renderStatus();
      }});
      const runBtn = document.getElementById("rebalanceRunBtn");
      if (runBtn) runBtn.addEventListener("click", () => {{
        rebalanceAmount = Number(document.getElementById("rebalanceAmount")?.value || rebalanceAmount || 0);
        renderDashboardTab();
        renderStatus();
      }});
	    }}
	    function renderReportsToolbar() {{
	      return `
	        <div class="reports-toolbar">
	          <span class="reports-label">◉ Reports</span>
	          <span class="report-chip">⌁ Investment & Tax</span>
	          <span class="report-chip">◉ Dividends</span>
	          <span class="report-chip">▰ Holdings</span>
	          <span class="report-chip">▣ Options</span>
	          <span class="report-chip">▥ Bonds</span>
	          <span class="reports-close">×</span>
	        </div>`;
	    }}
	    function insightCard(label, note, extraClass = "") {{
	      return `
	        <article class="insight-card ${{extraClass}}">
	          <div class="insight-label">${{label}} <span class="info">●</span></div>
	          <div class="insight-empty-value">-</div>
	          <div class="insight-note">${{note}}</div>
	        </article>`;
	    }}
	    function definitionItem(label, text) {{
	      return `<div class="definition-item"><strong>${{label}}</strong>${{text}}</div>`;
	    }}
	    function pointTime(point) {{
	      const parsed = new Date(point.timestamp || point.label).getTime();
	      return Number.isFinite(parsed) ? parsed : 0;
	    }}
	    function timeExtent(points) {{
	      const times = points.map(pointTime).filter((value) => Number.isFinite(value) && value > 0);
	      if (!times.length) return {{ minTime: 0, maxTime: 1 }};
	      const minTime = Math.min(...times);
	      const maxTime = Math.max(...times);
	      return {{ minTime, maxTime: maxTime === minTime ? minTime + 24 * 3600 * 1000 : maxTime }};
	    }}
	    function scaledGrowthPoint(point, index, points, width, height, pad, minY, maxY, extent = null) {{
	      const span = Math.max(1, maxY - minY);
	      const time = pointTime(point);
	      const range = extent || timeExtent(points);
	      const x = pad + ((time - range.minTime) / Math.max(1, range.maxTime - range.minTime)) * (width - pad * 2);
	      const y = height - pad - ((Number(point.value || 0) - minY) / span) * (height - pad * 2);
	      return {{ x, y }};
	    }}
	    function growthPath(points, width, height, pad, minY, maxY, stepped = false) {{
	      const extent = timeExtent(points);
	      const scaled = points.map((point, index) => scaledGrowthPoint(point, index, points, width, height, pad, minY, maxY, extent));
	      if (!stepped) return scaled.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{point.x.toFixed(1)}} ${{point.y.toFixed(1)}}`).join(" ");
	      return scaled.map((point, index) => {{
	        if (index === 0) return `M ${{point.x.toFixed(1)}} ${{point.y.toFixed(1)}}`;
	        const previous = scaled[index - 1];
	        return `L ${{point.x.toFixed(1)}} ${{previous.y.toFixed(1)}} L ${{point.x.toFixed(1)}} ${{point.y.toFixed(1)}}`;
	      }}).join(" ");
	    }}
	    function growthHitAreas(portfolioPoints, investedPoints, width, height, pad, minY, maxY) {{
	      const extent = timeExtent(portfolioPoints);
	      return portfolioPoints.map((point, index) => {{
	        const scaled = scaledGrowthPoint(point, index, portfolioPoints, width, height, pad, minY, maxY, extent);
	        const invested = investedPoints[index] || investedPoints[investedPoints.length - 1] || {{}};
	        const portfolioValue = Number(point.value || 0);
	        const investedValue = Number(invested.value || 0);
	        const realizedValue = Number(point.realized || 0);
	        const diff = portfolioValue - investedValue;
	        const diffPct = investedValue ? (diff / investedValue) * 100 : 0;
	        const tooltip = [
	          point.label,
	          `Portfolio: ${{money(portfolioValue, point.currency)}}`,
	          `Invested: ${{money(investedValue, invested.currency || point.currency)}}`,
	          `Realized P/L: ${{sign(realizedValue)}}${{money(realizedValue, point.currency)}}`,
	          `Difference: ${{sign(diff)}}${{money(diff, point.currency)}} (${{pct(diffPct)}})`,
	        ].join("|");
	        return `<circle class="growth-hit" cx="${{scaled.x.toFixed(1)}}" cy="${{scaled.y.toFixed(1)}}" r="13" data-tooltip="${{tooltip}}"></circle>`;
	      }}).join("");
	    }}
	    function selectedHistoryRows() {{
	      const rows = (state?.portfolio_history || [])
	        .filter((row) => row.portfolio_key === selectedKey && Number.isFinite(Number(row.total_worth)))
	        .slice()
	        .sort((a, b) => new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date));
	      const uniqueDates = new Set(rows.map((row) => row.date || String(row.timestamp || "").slice(0, 10)));
	      if (uniqueDates.size < 2) return rows;
	      const latestByDate = new Map();
	      rows.forEach((row) => {{
	        const dateKey = row.date || String(row.timestamp || "").slice(0, 10);
	        const previous = latestByDate.get(dateKey);
	        if (!previous || new Date(row.timestamp || row.date) >= new Date(previous.timestamp || previous.date)) {{
	          latestByDate.set(dateKey, row);
	        }}
	      }});
	      return Array.from(latestByDate.values()).sort((a, b) => new Date(a.timestamp || a.date) - new Date(b.timestamp || b.date));
	    }}
	    function filterGrowthRows(rows) {{
	      if (growthRange === "max" || rows.length < 2) return rows;
	      const days = {{ mtd: null, ytd: null, "1m": 31, "3m": 93, "1y": 366, "2y": 732, "3y": 1098, "5y": 1830 }}[growthRange];
	      const lastTime = new Date(rows[rows.length - 1].timestamp || rows[rows.length - 1].date).getTime();
	      let startTime = -Infinity;
	      if (growthRange === "mtd") {{
	        const last = new Date(lastTime);
	        startTime = new Date(last.getFullYear(), last.getMonth(), 1).getTime();
	      }} else if (growthRange === "ytd") {{
	        const last = new Date(lastTime);
	        startTime = new Date(last.getFullYear(), 0, 1).getTime();
	      }} else if (days) {{
	        startTime = lastTime - days * 24 * 3600 * 1000;
	      }}
	      const filtered = rows.filter((row) => new Date(row.timestamp || row.date).getTime() >= startTime);
	      return filtered.length >= 2 ? filtered : rows;
	    }}
	    function historyPoint(row, field, currency) {{
	      const baseValue = Number(row[field] || 0);
	      const realized = Number(row.realized_pl || 0);
	      return {{
	        label: row.date || String(row.timestamp || "").slice(0, 10),
	        timestamp: row.timestamp || row.date,
	        value: field === "total_worth" && includeRealizedGrowth ? baseValue + realized : baseValue,
	        realized,
	        currency: row.currency || currency,
	      }};
	    }}
	    function renderGrowthRanges(hasEnoughHistory) {{
	      if (!hasEnoughHistory) return "";
	      return `<div class="growth-ranges">${{[
	        ["ytd", "YTD"], ["mtd", "MTD"], ["1m", "1M"], ["3m", "3M"], ["1y", "1Y"], ["2y", "2Y"], ["3y", "3Y"], ["5y", "5Y"], ["max", "Max"],
	      ].map(([key, label]) => `<button class="growth-range ${{growthRange === key ? "active" : ""}}" data-growth-range="${{key}}">${{label}}</button>`).join("")}}</div>`;
	    }}
	    function renderPortfolioGrowthPanel() {{
	      const p = selectedPortfolio();
	      const currency = p.currency || "EUR";
	      const portfolioOptions = portfolioList().map((item) => `<option value="${{item.key}}" ${{item.key === selectedKey ? "selected" : ""}}>${{item.name || item.key}}</option>`).join("");
	      const allRows = selectedHistoryRows();
	      const rows = filterGrowthRows(allRows);
	      const hasEnoughHistory = allRows.length > 2;
	      const realizedTotal = Number(p.realized_pl || 0);
	      const invested = Math.max(0, Number(rows[rows.length - 1]?.initially_invested ?? p.initially_invested ?? 0));
	      const current = Math.max(0, Number(rows[rows.length - 1]?.total_worth ?? p.total_worth ?? 0)) + (includeRealizedGrowth ? realizedTotal : 0);
	      const gain = current - invested;
	      const gainPct = invested ? (gain / invested) * 100 : Number(p.simple_return_pct || 0);
	      const width = 1200;
	      const height = 320;
	      const pad = 44;
	      const portfolioPoints = rows.length
	        ? rows.map((row) => historyPoint(row, "total_worth", currency))
	        : [{{ label: "Initial", value: invested, currency }}, {{ label: "Now", value: current, currency }}];
	      const investedPoints = rows.length
	        ? rows.map((row) => historyPoint({{ ...row, initially_invested: row.initially_invested ?? invested }}, "initially_invested", currency))
	        : [{{ label: "Initial invested", value: invested, currency }}, {{ label: "Current invested", value: invested, currency }}];
	      const values = [0, ...portfolioPoints.map((point) => point.value), ...investedPoints.map((point) => point.value)].filter((value) => Number.isFinite(value));
	      const maxY = Math.max(1, ...values) * 1.12;
	      const minY = 0;
	      const grid = [0, .25, .5, .75, 1].map((ratio) => {{
	        const y = height - pad - ratio * (height - pad * 2);
	        const labelValue = minY + ratio * (maxY - minY);
	        return `<line class="growth-grid" x1="${{pad}}" y1="${{y.toFixed(1)}}" x2="${{width - pad}}" y2="${{y.toFixed(1)}}"></line><text class="growth-axis" x="8" y="${{(y + 4).toFixed(1)}}">${{money(labelValue, currency)}}</text>`;
	      }}).join("");
	      const deltaIcon = gain >= 0 ? "↑" : "↓";
	      const firstLabel = portfolioPoints[0]?.label || "Initial";
	      const lastLabel = portfolioPoints[portfolioPoints.length - 1]?.label || "Now";
	      const historyNote = hasEnoughHistory
	        ? `Showing ${{rows.length}} stored/API history points for the selected portfolio. ${{includeRealizedGrowth ? "Realized P/L is added to the portfolio line." : "Realized P/L is excluded from the portfolio line."}} New points are appended automatically on every refresh.`
	        : "Not enough historical points yet for date ranges. The dashboard will append a new point on every refresh; SnapTrade balance history is used when the broker/API returns it.";
	      return `
	        <section class="growth-panel">
	          <div class="growth-header">
	            <h3 class="growth-title">Portfolio Growth</h3>
	            <div class="growth-actions">
	              <select id="growthPortfolioSelect" class="portfolio-select" aria-label="Selected portfolio">${{portfolioOptions}}</select>
	              <button class="toggle-button ${{includeRealizedGrowth ? "active" : ""}}" id="realizedToggle" aria-pressed="${{includeRealizedGrowth ? "true" : "false"}}"></button>
	              <span>Include Realized</span>
	              <span>✣</span>
	            </div>
	          </div>
	          <div class="growth-body">
	            <div class="growth-summary">
	              <span class="${{klass(gain)}}">${{deltaIcon}} ${{money(gain, currency)}}</span>
	              <span class="growth-badge ${{klass(gain)}}">${{pct(gainPct)}}</span>
	            </div>
	            ${{renderGrowthRanges(hasEnoughHistory)}}
	            <div class="growth-chart-wrap">
	              <svg class="growth-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Portfolio growth for ${{p.name || "selected portfolio"}}">
	                ${{grid}}
	                <path class="growth-line-invested" d="${{growthPath(investedPoints, width, height, pad, minY, maxY, true)}}"></path>
	                <path class="growth-line-portfolio" d="${{growthPath(portfolioPoints, width, height, pad, minY, maxY)}}"></path>
	                ${{growthHitAreas(portfolioPoints, investedPoints, width, height, pad, minY, maxY)}}
	                <text class="growth-axis" x="${{pad}}" y="${{height - 12}}">${{firstLabel}}</text>
	                <text class="growth-axis" x="${{width - pad - 84}}" y="${{height - 12}}">${{lastLabel}}</text>
	              </svg>
	            </div>
	            <div class="growth-legend">
	              <span class="legend-item" style="color:#2fb8c5"><span class="legend-swatch"></span>Portfolio</span>
	              <span class="legend-item" style="color:#8b3f91"><span class="legend-swatch"></span>Invested</span>
	            </div>
	            <div class="growth-note">${{historyNote}}</div>
	            <div class="growth-tooltip" id="growthTooltip"></div>
	          </div>
	        </section>`;
	    }}
	    function renderInsightsTab() {{
	      const p = selectedPortfolio();
	      const generated = String(state?.generated_at || "");
	      const fromDate = generated.slice(0, 10) || "start date";
	      const toDate = new Date().toISOString().slice(0, 10);
	      document.getElementById("tabContent").innerHTML = `
	        <div class="dashboard-stack">
	          ${{renderReportsToolbar()}}
	          <section class="insights-report">
	            <div class="insights-header">
	              <h2 class="insights-title">Portfolio Insights for ${{p.name || "Selected Portfolio"}} <span class="insights-range">from ${{fromDate}} to ${{toDate}}</span></h2>
	              <span class="notes-pill">Notes: on</span>
	            </div>
	            <div class="insight-card-grid">
	              ${{insightCard("MTD Return", "How much the selected portfolio has gained or lost since the start of this month.")}}
	              ${{insightCard("Weekly Return", "How much the selected portfolio has gained or lost over the last week.")}}
	              ${{insightCard("CAGR / Time-Weighted Return", "The yearly pace of growth after removing the effect of deposits and withdrawals. It helps compare performance fairly.")}}
	              ${{insightCard("Capture Up / Down", "How much of the market's good days and bad days your portfolio follows. Lower downside capture usually means it falls less when the market falls.")}}
	            </div>
	            <div class="insights-layout">
	              <section class="insight-section">
	                <h3 class="insight-section-title">Annualized Volatility</h3>
	                <div class="insight-note">How much the portfolio value normally moves up and down in a year. Higher volatility means a bumpier ride, not automatically a bad investment.</div>
	                <div class="insight-card-grid" style="margin-top:12px">
	                  ${{insightCard("Tracking Error", "How differently your portfolio moves compared with the benchmark. Higher means it behaves less like the benchmark.", "wide")}}
	                  ${{insightCard("Beta / Correlation", "Beta shows how strongly your portfolio moves when the market moves. Correlation shows how closely the direction matches.", "wide")}}
	                </div>
	              </section>
	              <section class="insight-section">
	                <h3 class="insight-section-title">Risk Highlights</h3>
	                <div class="insight-card-grid">
	                  ${{insightCard("Max Drawdown", "The biggest fall from a previous high point to a later low point.")}}
	                  ${{insightCard("Current Drawdown", "How far the portfolio is below its most recent high right now.")}}
	                  ${{insightCard("Hit Rate", "The percentage of measured periods where the portfolio made money.")}}
	                  ${{insightCard("Information Ratio", "How much extra return the portfolio earns compared with the benchmark for each unit of extra risk.")}}
	                  ${{insightCard("Best Period", "The strongest measured period in the report window.", "wide")}}
	                  ${{insightCard("Worst Period", "The weakest measured period in the report window.", "wide")}}
	                </div>
		              </section>
		            </div>
		          </section>
		          ${{renderPortfolioGrowthPanel()}}
		        </div>`;
	      document.querySelectorAll(".growth-range").forEach((button) => button.addEventListener("click", () => {{
	        growthRange = button.dataset.growthRange || "max";
	        renderInsightsTab();
	        renderStatus();
	      }}));
	      const realizedToggle = document.getElementById("realizedToggle");
	      if (realizedToggle) realizedToggle.addEventListener("click", () => {{
	        includeRealizedGrowth = !includeRealizedGrowth;
	        renderInsightsTab();
	        renderStatus();
	      }});
	      const portfolioSelect = document.getElementById("growthPortfolioSelect");
	      if (portfolioSelect) portfolioSelect.addEventListener("change", () => {{
	        selectedKey = portfolioSelect.value;
	        localStorage.setItem("selectedPortfolioKey", selectedKey);
	        render();
	      }});
	      const tooltip = document.getElementById("growthTooltip");
	      document.querySelectorAll(".growth-hit").forEach((point) => {{
	        point.addEventListener("mouseenter", () => {{
	          if (!tooltip) return;
	          const parts = String(point.dataset.tooltip || "").split("|");
	          tooltip.innerHTML = `<strong>${{parts[0] || ""}}</strong>${{parts.slice(1).map((part) => `<div>${{part}}</div>`).join("")}}`;
	          tooltip.classList.add("visible");
	        }});
	        point.addEventListener("mousemove", (event) => {{
	          if (!tooltip) return;
	          tooltip.style.left = `${{event.clientX}}px`;
	          tooltip.style.top = `${{event.clientY}}px`;
	        }});
	        point.addEventListener("mouseleave", () => {{
	          if (!tooltip) return;
	          tooltip.classList.remove("visible");
	        }});
	      }});
	    }}
	    function dividendTotals(rows) {{
	      return rows.reduce((acc, row) => {{
	        acc.afterTax += Number(row.after_tax_amount || 0);
	        acc.tax += Number(row.tax_amount || 0);
	        acc.gross += Number(row.gross_amount || 0);
	        return acc;
	      }}, {{ afterTax: 0, tax: 0, gross: 0 }});
	    }}
	    function monthlyDividendRows(rows) {{
	      const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
	      const output = months.map((month, index) => ({{ label: month, month: index, afterTax: 0, tax: 0, gross: 0 }}));
	      rows.forEach((row) => {{
	        const date = new Date(row.date || row.timestamp || "");
	        if (Number.isNaN(date.getTime())) return;
	        const item = output[date.getMonth()];
	        item.afterTax += Number(row.after_tax_amount || 0);
	        item.tax += Number(row.tax_amount || 0);
	        item.gross += Number(row.gross_amount || 0);
	      }});
	      return output;
	    }}
	    function stockDividendRows(rows) {{
	      const grouped = new Map();
	      rows.forEach((row) => {{
	        const key = row.ticker || row.isin || row.name || "Unknown";
	        const existing = grouped.get(key) || {{ key, name: row.name || key, ticker: row.ticker || "", afterTax: 0, tax: 0, gross: 0, count: 0, currency: row.currency || "EUR" }};
	        existing.afterTax += Number(row.after_tax_amount || 0);
	        existing.tax += Number(row.tax_amount || 0);
	        existing.gross += Number(row.gross_amount || 0);
	        existing.count += 1;
	        grouped.set(key, existing);
	      }});
	      return Array.from(grouped.values()).sort((a, b) => b.gross - a.gross);
	    }}
	    function renderDividendChart(rows, currency) {{
	      const width = 960;
	      const height = 360;
	      const pad = 44;
	      const maxValue = Math.max(1, ...rows.map((row) => Number(row.gross || 0)));
	      const grid = [0, .25, .5, .75, 1].map((ratio) => {{
	        const y = height - pad - ratio * (height - pad * 2);
	        return `<line class="dividend-grid" x1="${{pad}}" y1="${{y.toFixed(1)}}" x2="${{width - pad}}" y2="${{y.toFixed(1)}}"></line><text class="dividend-axis-muted" x="8" y="${{(y + 4).toFixed(1)}}">${{money(maxValue * ratio, currency)}}</text>`;
	      }}).join("");
	      const slot = (width - pad * 2) / rows.length;
	      const bars = rows.map((row, index) => {{
	        const x = pad + index * slot + slot * .22;
	        const barWidth = Math.max(12, slot * .44);
	        const afterHeight = (Number(row.afterTax || 0) / maxValue) * (height - pad * 2);
	        const taxHeight = (Number(row.tax || 0) / maxValue) * (height - pad * 2);
	        const afterY = height - pad - afterHeight;
	        const taxY = afterY - taxHeight;
	        return `
	          <rect class="dividend-bar-tax" x="${{x.toFixed(1)}}" y="${{taxY.toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{Math.max(0, taxHeight).toFixed(1)}}"></rect>
	          <rect class="dividend-bar-after" x="${{x.toFixed(1)}}" y="${{afterY.toFixed(1)}}" width="${{barWidth.toFixed(1)}}" height="${{Math.max(0, afterHeight).toFixed(1)}}"></rect>
	          <text class="dividend-axis" x="${{(x + barWidth / 2).toFixed(1)}}" y="${{height - 12}}" text-anchor="middle">${{row.label}}</text>`;
	      }}).join("");
	      return `
	        <svg class="dividend-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Monthly dividends">
	          ${{grid}}
	          ${{bars}}
	        </svg>
	        <div class="dividend-legend"><span class="after">After-Tax</span><span class="tax">Tax Paid</span></div>`;
	    }}
	    function renderDividendStockList(rows, currency) {{
	      if (!rows.length) return `<div class="empty-panel">No dividend payments are available for this selected portfolio yet.</div>`;
	      const maxValue = Math.max(1, ...rows.map((row) => row.gross));
	      return `<div class="dividend-stock-list">${{rows.map((row) => `
	        <div class="dividend-stock-row">
	          <div>
	            <div class="dividend-stock-name">${{row.name}}</div>
	            <div class="dividend-stock-sub">${{row.ticker || row.key}} · ${{row.count}} payment${{row.count === 1 ? "" : "s"}}</div>
	          </div>
	          <div>
	            <div class="dividend-stock-bar"><div class="dividend-stock-fill" style="width:${{Math.max(2, Math.min(100, row.gross / maxValue * 100))}}%"></div></div>
	            <div class="dividend-stock-sub">After tax ${{money(row.afterTax, row.currency || currency)}} · Tax ${{money(row.tax, row.currency || currency)}}</div>
	          </div>
	          <div class="allocation-value">${{money(row.gross, row.currency || currency)}}</div>
	        </div>`).join("")}}</div>`;
	    }}
	    function renderDividendsTab() {{
	      const p = selectedPortfolio();
	      const currency = p.currency || "EUR";
	      const rows = selectedDividends();
	      const totals = dividendTotals(rows);
	      const annual = totals.afterTax;
	      const monthly = annual / 12;
	      const content = dividendMode === "stock"
	        ? renderDividendStockList(stockDividendRows(rows), currency)
	        : renderDividendChart(monthlyDividendRows(rows), currency);
	      document.getElementById("tabContent").innerHTML = `
	        <div class="dividend-tabs">
	          <button class="dividend-tab active">▦ Overview</button>
	          <button class="dividend-tab">▤ Full Dividend Report</button>
	        </div>
	        <section class="dividend-layout">
	          <aside class="dividend-summary">
	            <article class="dividend-stat">
	              <div>
	                <div class="dividend-stat-label"><span class="metric-icon">$</span>Dividend Yield</div>
	                <div class="dividend-pill">${{pct(p.dividend_yield_pct)}}</div>
	              </div>
	            </article>
	            <article class="dividend-stat">
	              <div>
	                <div class="dividend-stat-label">⚖ Yield on Cost</div>
	                <div class="dividend-pill muted">${{pct(p.dividend_yield_on_cost_pct ?? p.dividend_yield_pct)}}</div>
	              </div>
	            </article>
	            <article class="dividend-stat">
	              <div>
	                <div class="dividend-stat-label">▦ Annual</div>
	                <div class="dividend-pill green">${{money(annual, currency)}}</div>
	              </div>
	            </article>
	            <article class="dividend-stat">
	              <div>
	                <div class="dividend-stat-label">▣ Monthly</div>
	                <div class="dividend-pill cyan">${{money(monthly, currency)}}</div>
	              </div>
	            </article>
	          </aside>
	          <article class="dividend-panel">
	            <div class="dividend-panel-header">
	              <h2 class="dividend-title">▧ Historical Dividends</h2>
	              <div class="dividend-total">◉ Total: ${{money(totals.afterTax, currency)}}</div>
	            </div>
	            <div class="dividend-controls">
	              <div></div>
	              <div class="dividend-mode">
	                <button class="${{dividendMode === "monthly" ? "active" : ""}}" data-dividend-mode="monthly">▣ Monthly</button>
	                <button class="${{dividendMode === "stock" ? "active" : ""}}" data-dividend-mode="stock">▤ Per Stock</button>
	              </div>
	              <div class="perf-sub">Currency: ${{currency}}</div>
	            </div>
	            ${{content}}
	          </article>
	        </section>`;
	      document.querySelectorAll("[data-dividend-mode]").forEach((button) => button.addEventListener("click", () => {{
	        dividendMode = button.dataset.dividendMode || "monthly";
	        renderDividendsTab();
	        renderStatus();
	      }}));
	    }}
	    function analyticsRows() {{
	      return selectedHoldings()
	        .filter((holding) => Number(holding.current_value || 0) > 0)
	        .map((holding) => {{
	          const meanReturn = Number(holding.unrealized_pl_pct || 0);
	          const volatility = Math.abs(meanReturn) > 0.01 ? Math.max(2, Math.abs(meanReturn) * 1.35) : 0;
	          const sharpe = volatility > 0 ? meanReturn / volatility : 0;
	          return {{
	            name: holdingDisplayName(holding),
	            ticker: holding.ticker || holding.broker_symbol || holding.isin || "",
	            meanReturn,
	            volatility,
	            sharpe,
	            value: Number(holding.current_value || 0),
	          }};
	        }})
	        .sort((a, b) => b.meanReturn - a.meanReturn);
	    }}
	    function selectedMonteCarlo() {{
	      const simulations = state?.analytics?.monte_carlo || {{}};
	      return simulations[selectedKey] || simulations[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedRisk() {{
	      const risks = state?.analytics?.risk || {{}};
	      return risks[selectedKey] || risks[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedFrontier() {{
	      const frontiers = state?.analytics?.frontier || {{}};
	      return frontiers[selectedKey] || frontiers[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedScenarios() {{
	      const scenarios = state?.analytics?.scenarios || {{}};
	      return scenarios[selectedKey] || scenarios[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedBenchmark() {{
	      const benchmarks = state?.analytics?.benchmark || {{}};
	      return benchmarks[selectedKey] || benchmarks[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedCorrelation() {{
	      const correlations = state?.analytics?.correlation || {{}};
	      return correlations[selectedKey] || correlations[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedTradeAlpha() {{
	      const tradeAlpha = state?.analytics?.trade_alpha || {{}};
	      return tradeAlpha[selectedKey] || tradeAlpha[state?.combined_portfolio?.key] || null;
	    }}
	    function selectedDiversification() {{
	      const p = selectedPortfolio();
	      const holdings = selectedHoldings().filter((holding) => Number(holding.current_value || 0) > 0);
	      const cashRows = diversificationCashRows(p);
	      const inputs = [...holdings, ...cashRows];
	      const sectorRows = groupedDiversificationRows(inputs, diversificationSectorFor);
	      const geographicRows = groupedDiversificationRows(inputs, diversificationCountryFor);
	      const sectorScore = diversificationSubscore(sectorRows);
	      const geographicScore = diversificationSubscore(geographicRows);
	      const score = Math.round((sectorScore * 0.5 + geographicScore * 0.5) * 10) / 10;
	      const recommendations = diversificationRecommendations(sectorRows, geographicRows, holdings.length);
	      return {{
	        holdings,
	        cashRows,
	        sectorRows,
	        geographicRows,
	        sectorScore,
	        geographicScore,
	        score,
	        recommendations,
	      }};
	    }}
	    function diversificationCashRows(portfolio) {{
	      const cashSources = portfolio.kind === "combined"
	        ? portfolioList().filter((item) => item.include_in_combined && item.kind !== "combined")
	        : [portfolio];
	      return cashSources
	        .filter((item) => Number(item.cash || 0) > 0)
	        .map((item) => ({{
	          name: "Cash",
	          ticker: "CASH",
	          current_value: Number(item.cash || 0),
	          currency: item.currency || portfolio.currency || "EUR",
	          asset_type: "cash",
	          account_key: item.key,
	          __diversification_sector: "Cash",
	          __diversification_country: "Cash",
	        }}));
	    }}
	    function diversificationSectorFor(holding) {{
	      if (holding.__diversification_sector) return holding.__diversification_sector;
	      return sectorFor(holding);
	    }}
	    function diversificationCountryFor(holding) {{
	      if (holding.__diversification_country) return holding.__diversification_country;
	      return countryFor(holding);
	    }}
	    function groupedDiversificationRows(holdings, labelFn) {{
	      const grouped = new Map();
	      holdings.forEach((holding) => {{
	        const label = labelFn(holding) || "Other";
	        const existing = grouped.get(label) || {{ name: label, value: 0, count: 0 }};
	        existing.value += Number(holding.current_value || 0);
	        existing.count += 1;
	        grouped.set(label, existing);
	      }});
	      const total = Array.from(grouped.values()).reduce((sum, row) => sum + row.value, 0);
	      return Array.from(grouped.values())
	        .filter((row) => row.value > 0)
	        .map((row) => ({{ ...row, pct: total ? row.value / total * 100 : 0 }}))
	        .sort((a, b) => b.value - a.value);
	    }}
	    function diversificationSubscore(rows) {{
	      if (!rows.length) return 0;
	      const hhi = rows.reduce((sum, row) => sum + Math.pow(Number(row.pct || 0) / 100, 2), 0);
	      return Math.max(0, Math.min(100, Math.round((1 - hhi) * 1000) / 10));
	    }}
	    function diversificationRecommendations(sectorRows, geographicRows, holdingCount) {{
	      const recommendations = [];
	      const topSector = sectorRows.find((row) => row.name !== "Cash");
	      const topRegion = geographicRows.find((row) => row.name !== "Cash");
	      if (topSector && topSector.pct >= 40) recommendations.push(`High sector concentration: ${{topSector.name}} at ${{pct(topSector.pct)}}.`);
	      if (topRegion && topRegion.pct >= 50) recommendations.push(`High geographic concentration: ${{topRegion.name}} at ${{pct(topRegion.pct)}}.`);
	      if (holdingCount > 0 && holdingCount < 8) recommendations.push(`Low number of holdings: only ${{holdingCount}} positions in this selected portfolio.`);
	      if (!recommendations.length) recommendations.push("No major concentration warning for the selected portfolio based on current holdings.");
	      return recommendations;
	    }}
	    function compactDiversificationRows(rows, limit = 7) {{
	      if (rows.length <= limit) return rows;
	      const visible = rows.slice(0, limit - 1);
	      const other = rows.slice(limit - 1).reduce((acc, row) => {{
	        acc.value += row.value;
	        acc.pct += row.pct;
	        acc.count += row.count || 0;
	        return acc;
	      }}, {{ name: "Other", value: 0, pct: 0, count: 0 }});
	      return [...visible, other].filter((row) => row.value > 0);
	    }}
	    function diversificationGradient(rows) {{
	      if (!rows.length) return "#263142";
	      const colors = ["#3d78d8", "#22b45c", "#e79a13", "#d8443f", "#14a6b8", "#7c5bd6", "#d64e95", "#12a37f", "#5b62d6"];
	      let cursor = 0;
	      return `conic-gradient(${{rows.map((row, index) => {{
	        const start = cursor;
	        cursor += Number(row.pct || 0);
	        return `${{colors[index % colors.length]}} ${{start.toFixed(2)}}% ${{cursor.toFixed(2)}}%`;
	      }}).join(", ")}})`;
	    }}
	    function renderDiversificationDonut(title, icon, rows, note) {{
	      const compactRows = compactDiversificationRows(rows);
	      const colors = ["#3d78d8", "#22b45c", "#e79a13", "#d8443f", "#14a6b8", "#7c5bd6", "#d64e95", "#12a37f", "#5b62d6"];
	      const legend = compactRows.map((row, index) => `
	        <div class="diversification-legend-row">
	          <span class="diversification-swatch" style="--swatch:${{colors[index % colors.length]}}"></span>
	          <span class="diversification-legend-name">${{row.name}}</span>
	          <strong>${{pct(row.pct)}}</strong>
	        </div>`).join("");
	      return `
	        <article class="diversification-chart-card">
	          <div class="diversification-heading">
	            <div class="diversification-icon">${{icon}}</div>
	            <div><h2 class="diversification-title">${{title}}</h2></div>
	          </div>
	          <div class="diversification-chart-body">
	            <div class="diversification-donut" style="--donut:${{diversificationGradient(compactRows)}}"></div>
	            <div class="diversification-legend">${{legend || `<div class="empty-panel">No allocation rows.</div>`}}</div>
	          </div>
	          <div class="diversification-chart-note"><span class="info">●</span> ${{note}}</div>
	        </article>`;
	    }}
	    function renderDiversificationBreakdown(title, icon, firstColumn, rows, color) {{
	      const body = rows.length ? rows.map((row) => `
	        <tr>
	          <td><div class="breakdown-name" title="${{row.name}}">${{row.name}}</div></td>
	          <td class="breakdown-weight">${{pct(row.pct)}}</td>
	          <td>
	            <div class="breakdown-bar">
	              <div class="breakdown-fill" style="--bar-color:${{color}}; --bar-width:${{Math.max(1.5, Math.min(100, row.pct)).toFixed(2)}}%"></div>
	            </div>
	          </td>
	        </tr>`).join("") : `<tr><td colspan="3"><div class="empty-panel">No breakdown rows available for this portfolio.</div></td></tr>`;
	      return `
	        <article class="breakdown-card">
	          <div class="breakdown-title">${{icon}} ${{title}}</div>
	          <div class="breakdown-body">
	            <table class="breakdown-table">
	              <thead><tr><th>${{firstColumn}}</th><th>Weight</th><th>Allocation</th></tr></thead>
	              <tbody>${{body}}</tbody>
	            </table>
	          </div>
	        </article>`;
	    }}
	    function renderAnalyticsMethodTabs() {{
	      const methods = [
	        ["monte_carlo", "⌘", "Monte Carlo"],
	        ["risk", "⬟", "Risk"],
	        ["frontier", "⌁", "Frontier"],
	        ["scenarios", "ϟ", "Scenarios"],
	        ["benchmark", "⚖", "Benchmark"],
	        ["diversification", "◔", "Diversification"],
	        ["correlation", "▦", "Correlation"],
	        ["data", "▤", "Data"],
	        ["trade_alpha", "♜", "Trade Alpha"],
	      ];
	      return `<nav class="analytics-method-tabs">${{methods.map(([key, icon, label]) => `
	        <button class="analytics-method-tab ${{analyticsMode === key ? "active" : ""}}" data-analytics-mode="${{key}}">${{icon}} ${{label}}</button>
	      `).join("")}}</nav>`;
	    }}
	    function renderAnalyticsMethodPanel() {{
	      if (analyticsMode === "risk") return renderRiskPanel();
	      if (analyticsMode === "frontier") return renderFrontierPanel();
	      if (analyticsMode === "scenarios") return renderScenariosPanel();
	      if (analyticsMode === "benchmark") return renderBenchmarkPanel();
	      if (analyticsMode === "diversification") return renderDiversificationPanel();
	      if (analyticsMode === "correlation") return renderCorrelationPanel();
	      if (analyticsMode === "monte_carlo") return renderMonteCarloPanel();
	      if (analyticsMode === "data") return renderAssetPerformancePanel();
	      if (analyticsMode === "trade_alpha") return renderTradeAlphaPanel();
	      return `<section class="placeholder">${{analyticsMode.replaceAll("_", " ")}} will be built next for the selected portfolio.</section>`;
	    }}
		    function montePointPath(path, width, chart, minY, maxY) {{
		      if (!path || !path.length) return "";
		      const step = Math.max(1, Math.ceil(path.length / 120));
		      const sampled = path.filter((_, index) => index % step === 0 || index === path.length - 1);
		      const span = Math.max(1, maxY - minY);
		      return sampled.map((value, index) => {{
		        const x = chart.left + (index / Math.max(1, sampled.length - 1)) * (width - chart.left - chart.right);
		        const y = chart.bottom - ((Number(value || 0) - minY) / span) * (chart.bottom - chart.top);
		        return `${{index === 0 ? "M" : "L"}} ${{x.toFixed(1)}} ${{y.toFixed(1)}}`;
		      }}).join(" ");
		    }}
	    function renderMonteCarloPanel() {{
	      const p = selectedPortfolio();
	      const currency = p.currency || "EUR";
	      const mc = selectedMonteCarlo();
	      if (!mc) return `<section class="monte-panel"><div class="empty-panel">No Monte Carlo simulation is available for this selected portfolio yet.</div></section>`;
	      const percentiles = mc.percentiles || {{}};
	      const cards = [
	        ["Starting", mc.starting_value, ""],
	        ["5th %ile", percentiles.p5, "Worst case"],
	        ["25th %ile", percentiles.p25, ""],
	        ["Median", percentiles.median, "Expected"],
	        ["75th %ile", percentiles.p75, ""],
	        ["95th %ile", percentiles.p95, "Best case"],
	      ];
	      const paths = mc.sample_paths || [];
	      const flatValues = [mc.starting_value, ...Object.values(percentiles), ...paths.flat()].map(Number).filter(Number.isFinite);
	      const minY = Math.max(0, Math.min(...flatValues) * 0.82);
	      const maxY = Math.max(1, Math.max(...flatValues) * 1.08);
		      const width = 1200;
		      const height = 330;
		      const chart = {{ left: 96, right: 28, top: 26, bottom: 276 }};
		      const grid = [0, .25, .5, .75, 1].map((ratio) => {{
		        const y = chart.bottom - ratio * (chart.bottom - chart.top);
		        const value = minY + ratio * (maxY - minY);
		        return `<line class="monte-grid" x1="${{chart.left}}" y1="${{y.toFixed(1)}}" x2="${{width - chart.right}}" y2="${{y.toFixed(1)}}"></line><text class="monte-axis" x="${{chart.left - 8}}" y="${{(y + 3).toFixed(1)}}" text-anchor="end">${{money(value, currency)}}</text>`;
		      }}).join("");
	      const horizonDays = Number(mc.horizon_days || 252);
	      const xTicks = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, horizonDays]
	        .filter((value, index, items) => value <= horizonDays && items.indexOf(value) === index);
		      const xTickLabels = xTicks.map((day) => {{
		        const x = chart.left + (day / Math.max(1, horizonDays)) * (width - chart.left - chart.right);
		        return `<line class="monte-tick" x1="${{x.toFixed(1)}}" y1="${{chart.bottom}}" x2="${{x.toFixed(1)}}" y2="${{chart.bottom + 5}}"></line><text class="monte-axis" x="${{x.toFixed(1)}}" y="${{chart.bottom + 24}}" text-anchor="middle">${{day}}</text>`;
		      }}).join("");
		      const pathLines = paths.map((path, index) => `<path class="${{index === 0 ? "monte-path-highlight" : "monte-path"}}" d="${{montePointPath(path, width, chart, minY, maxY)}}"></path>`).join("");
		      const startY = chart.bottom - ((Number(mc.starting_value || 0) - minY) / Math.max(1, maxY - minY)) * (chart.bottom - chart.top);
	      return `
	        <section class="monte-panel">
	          <div class="monte-header">
	            <div class="analytics-icon">⌘</div>
	            <div>
	              <h2 class="monte-title">Monte Carlo Simulation</h2>
	              <div class="monte-subtitle">${{Number(mc.path_count || 0).toLocaleString()}} paths · 1-year projection · source: ${{mc.source || "portfolio history"}}</div>
	            </div>
	          </div>
	          <div class="monte-cards">
	            ${{cards.map(([label, value, tag], index) => `
	              <article class="monte-card">
	                <div>
	                  <div class="monte-card-label">${{label}}</div>
	                  <div class="monte-card-value ${{index === 1 ? "negative" : index === 2 ? "sharpe-warn" : index >= 4 ? "positive" : ""}}">${{money(value, currency)}}</div>
	                  ${{tag ? `<div class="monte-card-tag">${{tag}}</div>` : ""}}
	                </div>
	              </article>`).join("")}}
	          </div>
	          <div class="monte-chart-wrap">
	            <div class="monte-chart-title">1-Year Portfolio Value Projection (${{Number(mc.path_count || 0).toLocaleString()}} simulations)</div>
	            <svg class="monte-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Monte Carlo portfolio value projection">
	              ${{grid}}
		              <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.bottom}}" x2="${{width - chart.right}}" y2="${{chart.bottom}}"></line>
		              <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.top}}" x2="${{chart.left}}" y2="${{chart.bottom}}"></line>
	              ${{xTickLabels}}
	              ${{pathLines}}
		              <line class="monte-start-line" x1="${{chart.left}}" y1="${{startY.toFixed(1)}}" x2="${{width - chart.right}}" y2="${{startY.toFixed(1)}}"></line>
		              <text class="monte-axis" x="${{width / 2 - 64}}" y="${{(startY - 8).toFixed(1)}}">Starting Value: ${{money(mc.starting_value, currency)}}</text>
		              <text class="monte-axis-title" x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle">Trading Days</text>
		              <text class="monte-axis-title" x="0" y="0" text-anchor="middle" transform="translate(18 ${{(chart.top + chart.bottom) / 2}}) rotate(-90)">Portfolio Value (${{currency}})</text>
	            </svg>
	          </div>
	          <div class="monte-note">This simulation uses NumPy in the Python backend. It projects daily portfolio returns from stored portfolio history when available; if history is too short it uses a conservative current-return proxy until more refresh points accumulate.</div>
	        </section>`;
	    }}
	    function renderRiskPanel() {{
	      const p = selectedPortfolio();
	      const risk = selectedRisk();
	      const currency = risk?.currency || p.currency || "EUR";
	      if (!risk) return `<section class="risk-panel"><div class="empty-panel">No risk metrics are available for this selected portfolio yet.</div></section>`;
	      const var95 = Number(risk.var_95_pct || 0);
	      const var99 = Number(risk.var_99_pct || 0);
	      const drawdown = Number(risk.max_drawdown_pct || 0);
	      const peak = risk.drawdown_peak_date || "No peak yet";
	      const trough = risk.drawdown_trough_date || "No trough yet";
	      const recovered = risk.drawdown_recovered_date || "Not recovered";
	      return `
	        <section class="risk-panel">
	          <div class="risk-grid">
	            <div>
	              <div class="risk-section-title">
	                <div class="risk-icon">▲</div>
	                <div>
	                  <h2 class="risk-title">Value at Risk (VaR)</h2>
	                  <div class="risk-subtitle">Daily maximum expected loss</div>
	                </div>
	              </div>
	              <div class="risk-var-grid">
	                <article class="risk-box">
	                  <div class="risk-label">95% VaR <span class="info">●</span></div>
	                  <div class="risk-value negative">${{pct(var95)}}</div>
	                </article>
	                <article class="risk-box">
	                  <div class="risk-label">99% VaR <span class="info">●</span></div>
	                  <div class="risk-value negative">${{pct(var99)}}</div>
	                </article>
	              </div>
	              <div class="risk-help"><span class="info">●</span> 95% confident daily loss should not exceed ${{pct(Math.abs(var95))}} based on the available portfolio history/proxy.</div>
	            </div>
	            <div>
	              <div class="risk-section-title">
	                <div class="risk-icon orange">⌞</div>
	                <div>
	                  <h2 class="risk-title">Maximum Drawdown</h2>
	                  <div class="risk-subtitle">Peak-to-trough decline</div>
	                </div>
	              </div>
	              <div class="drawdown-hero">
	                <div class="drawdown-value">${{pct(drawdown)}}</div>
	                <div class="drawdown-sub">Maximum Portfolio Decline</div>
	              </div>
	              <div class="drawdown-line">
	                <div class="drawdown-segment"></div>
	                <div class="drawdown-dot"></div>
	                <div class="drawdown-segment recovery"></div>
	              </div>
	              <div class="drawdown-dates">
	                <div>${{peak}}<br>Peak</div>
	                <div>${{trough}}<br>Trough</div>
	                <div>${{recovered}}<br>Recovered</div>
	              </div>
	            </div>
	          </div>
	          <div class="risk-summary">
	            <div class="risk-summary-header">
	              <div class="analytics-icon">◌</div>
	              <h2 class="risk-title">Risk Summary</h2>
	            </div>
	            <div class="risk-summary-grid">
	              <article>
	                <div class="risk-summary-value positive">${{pct(risk.volatility_pct)}}</div>
	                <div class="risk-summary-label">Volatility <span class="info">●</span></div>
	              </article>
	              <article>
	                <div class="risk-summary-value positive">${{pct(risk.expected_return_pct)}}</div>
	                <div class="risk-summary-label">Expected Return <span class="info">●</span></div>
	              </article>
	              <article>
	                <div class="risk-summary-value positive">${{Number(risk.sharpe_ratio || 0).toFixed(2)}}</div>
	                <div class="risk-summary-label">Sharpe Ratio <span class="info">●</span></div>
	              </article>
	              <article>
	                <div class="risk-summary-value">${{money(risk.portfolio_value, currency)}}</div>
	                <div class="risk-summary-label">Portfolio Value</div>
	              </article>
	            </div>
	          </div>
	          <div class="monte-note">Risk metrics are read-only and calculated for the selected active portfolio. Source: ${{risk.source || "portfolio history"}} · history points: ${{risk.history_point_count || 0}}.</div>
	        </section>`;
	    }}
	    function frontierPointSvg(point, scale, className, radius = 6) {{
	      if (!point) return "";
	      const x = scale.x(Number(point.volatility_pct || 0));
	      const y = scale.y(Number(point.mean_return_pct || 0));
	      return `<circle class="${{className}}" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="${{radius}}"></circle>`;
	    }}
	    function renderFrontierPanel() {{
	      const p = selectedPortfolio();
	      const frontier = selectedFrontier();
	      if (!frontier || !frontier.frontier_points?.length) {{
	        return `<section class="frontier-panel"><div class="empty-panel">${{frontier?.note || "Efficient frontier needs at least two priced assets in the selected portfolio."}}</div></section>`;
	      }}
	      const points = frontier.frontier_points || [];
	      const assetPoints = frontier.asset_points || [];
	      const current = frontier.current_portfolio;
	      const maxSharpe = frontier.max_sharpe;
	      const allPoints = [...points, ...assetPoints, current, maxSharpe].filter(Boolean);
	      const maxVol = Math.max(5, ...allPoints.map((point) => Number(point.volatility_pct || 0))) * 1.12;
	      const minReturn = Math.min(-5, ...allPoints.map((point) => Number(point.mean_return_pct || 0))) * 1.08;
	      const maxReturn = Math.max(5, ...allPoints.map((point) => Number(point.mean_return_pct || 0))) * 1.12;
	      const width = 1200;
	      const height = 430;
	      const chart = {{ left: 82, right: 28, top: 28, bottom: 360 }};
	      const scale = {{
	        x: (value) => chart.left + (value / maxVol) * (width - chart.left - chart.right),
	        y: (value) => chart.bottom - ((value - minReturn) / Math.max(1, maxReturn - minReturn)) * (chart.bottom - chart.top),
	      }};
	      const xTicks = [0, .25, .5, .75, 1].map((ratio) => maxVol * ratio);
	      const yTicks = [0, .25, .5, .75, 1].map((ratio) => minReturn + (maxReturn - minReturn) * ratio);
	      const grid = [
	        ...xTicks.map((tick) => `<line class="frontier-grid" x1="${{scale.x(tick).toFixed(1)}}" y1="${{chart.top}}" x2="${{scale.x(tick).toFixed(1)}}" y2="${{chart.bottom}}"></line><text class="frontier-axis" x="${{scale.x(tick).toFixed(1)}}" y="${{chart.bottom + 22}}" text-anchor="middle">${{pct(tick)}}</text>`),
	        ...yTicks.map((tick) => `<line class="frontier-grid" x1="${{chart.left}}" y1="${{scale.y(tick).toFixed(1)}}" x2="${{width - chart.right}}" y2="${{scale.y(tick).toFixed(1)}}"></line><text class="frontier-axis" x="${{chart.left - 8}}" y="${{(scale.y(tick) + 3).toFixed(1)}}" text-anchor="end">${{pct(tick)}}</text>`),
	      ].join("");
	      const linePath = points.map((point, index) => `${{index === 0 ? "M" : "L"}} ${{scale.x(Number(point.volatility_pct || 0)).toFixed(1)}} ${{scale.y(Number(point.mean_return_pct || 0)).toFixed(1)}}`).join(" ");
	      const assetDots = assetPoints.slice(0, 60).map((point) => frontierPointSvg(point, scale, "frontier-asset", 4.5)).join("");
	      const dataRows = [
	        ["Your Portfolio", current],
	        ["Max Sharpe", maxSharpe],
	        ...points.filter((_, index) => index % Math.max(1, Math.ceil(points.length / 5)) === 0).slice(0, 5).map((point, index) => [`Frontier ${{index + 1}}`, point]),
	      ].filter(([, point]) => point).map(([label, point]) => `
	        <tr>
	          <td>${{label}}</td>
	          <td>${{pct(point.volatility_pct)}}</td>
	          <td>${{pct(point.mean_return_pct)}}</td>
	          <td>${{Number(point.sharpe_ratio || 0).toFixed(2)}}</td>
	        </tr>`).join("");
	      return `
	        <section class="frontier-panel">
	          <div class="frontier-head">
	            <div>
	              <h2 class="analytics-title">Efficient Frontier</h2>
	              <div class="analytics-subtitle">${{p.name || "Selected portfolio"}} risk-return optimization view</div>
	            </div>
	            <button class="secondary-btn">Export PDF</button>
	          </div>
	          <div class="frontier-info">
	            <div class="frontier-info-card"><div class="frontier-info-title">What is it?</div><div class="frontier-info-text">The frontier shows portfolios with the highest expected return for each level of risk.</div></div>
	            <div class="frontier-info-card"><div class="frontier-info-title">Your Portfolio</div><div class="frontier-info-text">The yellow point shows your current allocation. Closer to the blue line means more efficient risk use.</div></div>
	            <div class="frontier-info-card"><div class="frontier-info-title">Dive Deeper</div><div class="frontier-info-text">The data table lists risk, return, and Sharpe values for key frontier portfolios.</div></div>
	            <div class="frontier-info-card"><div class="frontier-info-title">Optimization Insight</div><div class="frontier-info-text">Use this to compare whether more return is possible without taking much more risk.</div></div>
	          </div>
	          <div class="frontier-chart-wrap">
	            <div class="frontier-legend">
	              <span><span class="legend-dot blue"></span>Efficient Frontier</span>
	              <span><span class="legend-dot yellow"></span>Your Portfolio</span>
	              <span><span class="legend-dot green"></span>Max Sharpe</span>
	              <span><span class="legend-dot red"></span>Assets</span>
	            </div>
	            <svg class="frontier-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Efficient frontier chart">
	              ${{grid}}
	              <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.bottom}}" x2="${{width - chart.right}}" y2="${{chart.bottom}}"></line>
	              <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.top}}" x2="${{chart.left}}" y2="${{chart.bottom}}"></line>
	              <path class="frontier-line" d="${{linePath}}"></path>
	              ${{assetDots}}
	              ${{frontierPointSvg(current, scale, "frontier-current", 7)}}
	              ${{frontierPointSvg(maxSharpe, scale, "frontier-sharpe", 7)}}
	              <text class="frontier-axis-title" x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle">Volatility (%)</text>
	              <text class="frontier-axis-title" x="0" y="0" text-anchor="middle" transform="translate(20 ${{(chart.top + chart.bottom) / 2}}) rotate(-90)">Expected Return (%)</text>
	            </svg>
	          </div>
	          <table class="frontier-table">
	            <thead><tr><th>Data Point</th><th>Volatility</th><th>Expected Return</th><th>Sharpe Ratio</th></tr></thead>
	            <tbody>${{dataRows}}</tbody>
	          </table>
	          <div class="monte-note">${{frontier.note || ""}}</div>
	        </section>`;
	    }}
	    function benchmarkPath(series, field, width, chart, minY, maxY) {{
	      if (!series.length) return "";
	      const span = Math.max(1, maxY - minY);
	      return series.map((row, index) => {{
	        const x = chart.left + (index / Math.max(1, series.length - 1)) * (width - chart.left - chart.right);
	        const y = chart.bottom - ((Number(row[field] || 0) - minY) / span) * (chart.bottom - chart.top);
	        return `${{index === 0 ? "M" : "L"}} ${{x.toFixed(1)}} ${{y.toFixed(1)}}`;
	      }}).join(" ");
	    }}
	    function benchmarkArea(series, field, width, chart, minY, maxY) {{
	      const path = benchmarkPath(series, field, width, chart, minY, maxY);
	      if (!path) return "";
	      return `${{path}} L ${{width - chart.right}} ${{chart.bottom}} L ${{chart.left}} ${{chart.bottom}} Z`;
	    }}
	    function benchmarkHitAreas(series, width, chart, minY, maxY, currency, benchmarkSymbol) {{
	      const span = Math.max(1, maxY - minY);
	      return series.map((row, index) => {{
	        const x = chart.left + (index / Math.max(1, series.length - 1)) * (width - chart.left - chart.right);
	        const y = chart.bottom - ((Number(row.portfolio_return_pct || 0) - minY) / span) * (chart.bottom - chart.top);
	        const tooltip = [
	          row.date,
	          `Portfolio return: ${{pct(row.portfolio_return_pct)}}`,
	          `${{benchmarkSymbol}} return: ${{pct(row.benchmark_return_pct)}}`,
	          `Portfolio value: ${{money(row.portfolio_value, currency)}}`,
	        ].join("|");
	        return `<circle class="benchmark-hit" cx="${{x.toFixed(1)}}" cy="${{y.toFixed(1)}}" r="12" data-tooltip="${{tooltip}}"></circle>`;
	      }}).join("");
	    }}
	    function renderBenchmarkPanel() {{
	      const p = selectedPortfolio();
	      const benchmark = selectedBenchmark();
	      if (!benchmark || !benchmark.series?.length) {{
	        return `<section class="benchmark-panel"><div class="empty-panel">${{benchmark?.note || "No benchmark comparison is available yet. Refresh the benchmark history cache, then let the dashboard append portfolio history points."}}</div></section>`;
	      }}
	      const currency = benchmark.currency || p.currency || "EUR";
	      const metrics = benchmark.metrics || {{}};
	      const symbol = benchmark.benchmark_symbol || "VOO";
	      const name = benchmark.benchmark_name || "Vanguard 500 Index Fund";
	      const rows = [
	        ["Total Return", metrics.total_return_pct, metrics.benchmark_total_return_pct, "pct"],
	        ["Annual Return", metrics.annual_return_pct, metrics.benchmark_annual_return_pct, "pct"],
	        ["Volatility", metrics.volatility_pct, metrics.benchmark_volatility_pct, "pct"],
	        ["Sharpe Ratio", metrics.sharpe_ratio, metrics.benchmark_sharpe_ratio, "number"],
	        ["Sortino Ratio", metrics.sortino_ratio, metrics.benchmark_sortino_ratio, "number"],
	      ];
	      const valueCell = (value, type) => type === "pct" ? `<span class="${{klass(value)}}">${{sign(value)}}${{pct(value)}}</span>` : Number(value || 0).toFixed(2);
	      const metricRows = rows.map(([label, portfolioValue, benchmarkValue, type]) => `
	        <tr>
	          <td>${{label}}</td>
	          <td>${{valueCell(portfolioValue, type)}}</td>
	          <td>${{valueCell(benchmarkValue, type)}}</td>
	        </tr>`).join("");
	      const advancedRows = [
	        ["Alpha (Jensen's)", metrics.alpha_pct, "pct"],
	        ["Beta", metrics.beta, "number"],
	        ["Correlation", metrics.correlation, "number"],
	        ["R-Squared", metrics.r_squared_pct, "pct"],
	        ["Tracking Error", metrics.tracking_error_pct, "pct"],
	        ["Information Ratio", metrics.information_ratio, "number"],
	        ["Treynor Ratio", metrics.treynor_ratio, "number"],
	      ].map(([label, value, type]) => `
	        <div class="benchmark-metric-row">
	          <div class="benchmark-metric-label">${{label}} <span class="info">●</span></div>
	          <div class="${{type === "pct" ? klass(value) : ""}}">${{type === "pct" ? pct(value) : Number(value || 0).toFixed(2)}}</div>
	        </div>`).join("");
	      const series = benchmark.series || [];
	      const width = 1200;
	      const height = 360;
	      const chart = {{ left: 70, right: 28, top: 30, bottom: 292 }};
	      const values = series.flatMap((row) => [Number(row.portfolio_return_pct || 0), Number(row.benchmark_return_pct || 0)]).filter(Number.isFinite);
	      const minY = Math.min(-5, ...values) * 1.08;
	      const maxY = Math.max(5, ...values) * 1.08;
	      const yTicks = [0, .25, .5, .75, 1].map((ratio) => minY + (maxY - minY) * ratio);
	      const grid = yTicks.map((tick) => {{
	        const y = chart.bottom - ((tick - minY) / Math.max(1, maxY - minY)) * (chart.bottom - chart.top);
	        return `<line class="benchmark-grid" x1="${{chart.left}}" y1="${{y.toFixed(1)}}" x2="${{width - chart.right}}" y2="${{y.toFixed(1)}}"></line><text class="benchmark-axis" x="${{chart.left - 8}}" y="${{(y + 3).toFixed(1)}}" text-anchor="end">${{pct(tick)}}</text>`;
	      }}).join("");
	      const dateTicks = [0, .25, .5, .75, 1].map((ratio) => Math.min(series.length - 1, Math.round((series.length - 1) * ratio)));
	      const xLabels = [...new Set(dateTicks)].map((index) => {{
	        const x = chart.left + (index / Math.max(1, series.length - 1)) * (width - chart.left - chart.right);
	        return `<text class="benchmark-axis" x="${{x.toFixed(1)}}" y="${{chart.bottom + 24}}" text-anchor="middle">${{series[index]?.date || ""}}</text>`;
	      }}).join("");
	      return `
	        <section class="benchmark-panel">
	          <div class="benchmark-top">
	            <article class="benchmark-card">
	              <div class="benchmark-card-title blue">⌁ Performance Comparison vs ${{name}}</div>
	              <div class="benchmark-card-body">
	                <div class="benchmark-subtitle">Comparing ${{p.name || "your portfolio"}} against ${{symbol}} (${{benchmark.aligned_point_count || series.length}} trading days)</div>
	                <table class="benchmark-table">
	                  <thead><tr><th>Metric</th><th>Your Portfolio</th><th>${{symbol}}</th></tr></thead>
	                  <tbody>${{metricRows}}</tbody>
	                </table>
	                <div class="benchmark-excess">Excess Return: <span class="${{klass(metrics.excess_return_pct)}}">${{sign(metrics.excess_return_pct)}}${{pct(metrics.excess_return_pct)}}</span> versus benchmark</div>
	              </div>
	            </article>
	            <article class="benchmark-card">
	              <div class="benchmark-card-title cyan">⚖ Risk-Adjusted Metrics</div>
	              <div class="benchmark-card-body">
	                <div class="benchmark-subtitle">Advanced metrics measuring portfolio performance relative to ${{symbol}}.</div>
	                <div class="benchmark-metric-list">${{advancedRows}}</div>
	              </div>
	            </article>
	          </div>
	          <article class="benchmark-chart-card">
	            <div class="benchmark-chart-title">▰ Cumulative Returns: Portfolio vs ${{symbol}}</div>
	            <div class="benchmark-chart-wrap">
	              <div class="benchmark-legend">
	                <span class="legend-item" style="color:var(--blue)"><span class="legend-swatch"></span>Your Portfolio</span>
	                <span class="legend-item" style="color:var(--red)"><span class="legend-swatch"></span>${{symbol}}</span>
	              </div>
	              <svg class="benchmark-chart" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Benchmark cumulative return chart">
	                ${{grid}}
	                <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.bottom}}" x2="${{width - chart.right}}" y2="${{chart.bottom}}"></line>
	                <line class="monte-axis-line" x1="${{chart.left}}" y1="${{chart.top}}" x2="${{chart.left}}" y2="${{chart.bottom}}"></line>
	                <path class="benchmark-area-portfolio" d="${{benchmarkArea(series, "portfolio_return_pct", width, chart, minY, maxY)}}"></path>
	                <path class="benchmark-area-index" d="${{benchmarkArea(series, "benchmark_return_pct", width, chart, minY, maxY)}}"></path>
	                <path class="benchmark-line-portfolio" d="${{benchmarkPath(series, "portfolio_return_pct", width, chart, minY, maxY)}}"></path>
	                <path class="benchmark-line-index" d="${{benchmarkPath(series, "benchmark_return_pct", width, chart, minY, maxY)}}"></path>
	                ${{benchmarkHitAreas(series, width, chart, minY, maxY, currency, symbol)}}
	                ${{xLabels}}
	                <text class="benchmark-axis-title" x="${{width / 2}}" y="${{height - 8}}" text-anchor="middle">Date</text>
	                <text class="benchmark-axis-title" x="0" y="0" text-anchor="middle" transform="translate(18 ${{(chart.top + chart.bottom) / 2}}) rotate(-90)">Cumulative Return (%)</text>
	              </svg>
	              <div class="benchmark-tooltip" id="benchmarkTooltip"></div>
	            </div>
	          </article>
	          <div class="monte-note">${{benchmark.note || ""}} Source: ${{benchmark.source || "benchmark cache"}} · fetched: ${{benchmark.fetched_at || "not fetched yet"}} · window: ${{benchmark.start_date || "-"}} to ${{benchmark.end_date || "-"}} · status: ${{benchmark.status || "-"}}.</div>
	        </section>`;
	    }}
	    function renderScenariosPanel() {{
	      const p = selectedPortfolio();
	      const scenarioPayload = selectedScenarios();
	      const currency = scenarioPayload?.currency || p.currency || "EUR";
	      const rows = scenarioPayload?.scenarios || [];
	      const cards = rows.length ? rows.map((row) => {{
	        const status = row.status || "missing";
	        const returnValue = Number(row.estimated_return_pct || 0);
	        const contributors = row.top_contributors || [];
	        const contributorRows = contributors.length
	          ? contributors.map((item) => `
	              <div class="scenario-contributor">
	                <span>${{item.ticker || item.name || "Asset"}} · ${{pct(item.event_return_pct)}}</span>
	                <strong class="${{klass(item.portfolio_impact)}}">${{money(item.portfolio_impact, currency)}}</strong>
	              </div>`).join("")
	          : `<div class="scenario-contributor"><span>No historical ticker data available</span><strong>-</strong></div>`;
	        return `
	          <article class="scenario-card">
	            <div class="scenario-card-head">
	              <span><span class="info">●</span> ${{row.name}}</span>
	              <span>${{status}}</span>
	            </div>
	            <div class="scenario-card-body">
	              <div class="scenario-description">${{row.description}}</div>
	              <div class="scenario-return ${{klass(returnValue)}}">${{sign(returnValue)}}${{pct(returnValue)}}</div>
	              <div class="scenario-subgrid">
	                <div class="scenario-kpi">
	                  <div class="scenario-kpi-label">Value Change</div>
	                  <div class="scenario-kpi-value ${{klass(row.estimated_value_change)}}">${{money(row.estimated_value_change, currency)}}</div>
	                </div>
	                <div class="scenario-kpi">
	                  <div class="scenario-kpi-label">End Value</div>
	                  <div class="scenario-kpi-value">${{money(row.estimated_end_value, currency)}}</div>
	                </div>
	                <div class="scenario-kpi">
	                  <div class="scenario-kpi-label">Coverage</div>
	                  <div class="scenario-kpi-value">${{pct(row.coverage_pct)}}</div>
	                </div>
	                <div class="scenario-kpi">
	                  <div class="scenario-kpi-label">Assets Matched</div>
	                  <div class="scenario-kpi-value">${{row.available_asset_count || 0}} / ${{row.holding_count || 0}}</div>
	                </div>
	              </div>
	              <div class="scenario-contributors">
	                <div class="scenario-kpi-label">Top Portfolio Impacts</div>
	                ${{contributorRows}}
	              </div>
	            </div>
	          </article>`;
	      }}).join("") : `<div class="empty-panel">No scenario data is available yet. Restart with --refresh-scenario-history-on-start to collect Yahoo historical data.</div>`;
	      return `
	        <section class="scenario-panel">
	          <div class="scenario-heading">
	            <div class="scenario-icon">ϟ</div>
	            <div>
	              <h2 class="analytics-title">Historical Stress Tests</h2>
	              <div class="analytics-subtitle">How ${{p.name || "the selected portfolio"}} would have performed during major market events</div>
	            </div>
	          </div>
	          <div class="scenario-grid">${{cards}}</div>
	          <div class="scenario-note"><span class="info">●</span> Scenario analysis replays historical ticker returns against your current allocation. It needs Yahoo historical prices for each current holding; coverage below 100% means some holdings did not have data for that historical period. Source: ${{scenarioPayload?.source || "missing cache"}} · fetched: ${{scenarioPayload?.fetched_at || "not fetched yet"}}.</div>
	        </section>`;
	    }}
	    function correlationCellColor(value) {{
	      const v = Math.max(-1, Math.min(1, Number(value || 0)));
	      if (v >= 0) {{
	        const t = v;
	        const r = Math.round(32 + 210 * t);
	        const g = Math.round(154 + 66 * t);
	        const b = Math.round(148 - 105 * t);
	        return `rgb(${{r}}, ${{g}}, ${{b}})`;
	      }}
	      const t = Math.abs(v);
	      const r = Math.round(32 + 36 * t);
	      const g = Math.round(154 - 74 * t);
	      const b = Math.round(148 + 70 * t);
	      return `rgb(${{r}}, ${{g}}, ${{b}})`;
	    }}
	    function correlationText(value) {{
	      const n = Number(value);
	      if (!Number.isFinite(n)) return "-";
	      return n.toFixed(3);
	    }}
	    function renderCorrelationPanel() {{
	      const p = selectedPortfolio();
	      const corr = selectedCorrelation();
	      if (!corr || !corr.matrix?.length) {{
	        return `<section class="correlation-panel">
	          <div class="correlation-head">
	            <div>
	              <h2 class="analytics-title">Correlation Matrix</h2>
		              <div class="analytics-subtitle">${{p.name || "Selected portfolio"}} · ${{corr?.status || "missing cache"}}</div>
	            </div>
	          </div>
	          <div class="empty-panel">${{corr?.note || "No correlation matrix is available yet. Restart with --refresh-price-history-on-start to download daily prices for the current holdings."}}</div>
	        </section>`;
	      }}
	      const symbols = corr.symbols || [];
	      const count = symbols.length;
	      const headerCells = symbols.map((item) => `<div class="correlation-label top" title="${{item.name || item.symbol}}">${{item.symbol}}</div>`).join("");
	      const rows = symbols.map((rowSymbol, rowIndex) => {{
	        const cells = (corr.matrix[rowIndex] || []).map((value, columnIndex) => `
	          <div class="correlation-cell" title="${{rowSymbol.symbol}} vs ${{symbols[columnIndex]?.symbol || ""}}: ${{correlationText(value)}}" style="background:${{correlationCellColor(value)}}">${{correlationText(value)}}</div>
	        `).join("");
	        return `<div class="correlation-label left" title="${{rowSymbol.name || rowSymbol.symbol}}">${{rowSymbol.symbol}}</div>${{cells}}`;
	      }}).join("");
	      const pairRows = (corr.pairs || []).slice(0, 18).map((pair) => `
	        <tr>
	          <td><strong>${{pair.asset_a}}</strong><div class="analytics-symbol">${{pair.name_a || ""}}</div></td>
	          <td><strong>${{pair.asset_b}}</strong><div class="analytics-symbol">${{pair.name_b || ""}}</div></td>
	          <td>${{correlationText(pair.correlation)}}</td>
	          <td>${{pair.relationship || ""}}</td>
	        </tr>`).join("");
	      return `
	        <section class="correlation-panel">
	          <div class="correlation-head">
	            <div>
	              <h2 class="analytics-title">Correlation Matrix</h2>
	              <div class="analytics-subtitle">${{p.name || "Selected portfolio"}} · ${{corr.used_asset_count || count}} assets · ${{corr.overlap_days || 0}} overlapping return days</div>
	            </div>
	            <button class="secondary-btn">Export PDF</button>
	          </div>
	          <div class="correlation-info-grid">
	            <div class="correlation-info-card"><div class="correlation-info-title">Heatmap Overview</div><div class="correlation-info-text">Each square shows how two assets moved together historically. Brighter positive cells mean they often moved in the same direction.</div></div>
	            <div class="correlation-info-card"><div class="correlation-info-title">1, 0, and -1</div><div class="correlation-info-text">1 means two assets moved together, 0 means no clear relationship, and -1 means they moved in opposite directions.</div></div>
	            <div class="correlation-info-card"><div class="correlation-info-title">Diversification</div><div class="correlation-info-text">Low or negative correlations can reduce portfolio swings because not every asset reacts the same way.</div></div>
	            <div class="correlation-info-card"><div class="correlation-info-title">Risk Management</div><div class="correlation-info-text">High positive correlations warn that positions may behave like one large bet during stress.</div></div>
	          </div>
	          <article class="correlation-chart-card">
	            <h3 class="correlation-title">Correlation Heatmap</h3>
	            <div class="correlation-matrix" style="grid-template-columns: 96px repeat(${{count}}, minmax(76px, 1fr));">
	              <div class="correlation-corner"></div>
	              ${{headerCells}}
	              ${{rows}}
	            </div>
	            <div class="correlation-legend"><span>-1 opposite</span><span class="correlation-ramp"></span><span>+1 together</span></div>
	          </article>
	          <article class="correlation-table-card">
	            <div class="correlation-table-title">Strongest Asset Relationships</div>
	            <div class="analytics-table-wrap" style="margin:0; border:0; border-radius:0;">
	              <table class="correlation-table">
	                <thead><tr><th>Asset A</th><th>Asset B</th><th>Correlation</th><th>Plain-English Meaning</th></tr></thead>
	                <tbody>${{pairRows || `<tr><td colspan="4">No pair relationships available.</td></tr>`}}</tbody>
	              </table>
	            </div>
	          </article>
	          <div class="correlation-note">${{corr.note || ""}} Source: ${{corr.source || "asset price cache"}} · fetched: ${{corr.fetched_at || "not fetched yet"}} · window: ${{corr.start_date || "-"}} to ${{corr.end_date || "-"}} · status: ${{corr.status || "-"}}.</div>
	        </section>`;
	    }}
	    function renderTradeAlphaPanel() {{
	      const p = selectedPortfolio();
	      const alpha = selectedTradeAlpha();
	      if (!alpha || !alpha.rows) {{
	        return `<section class="trade-alpha-panel"><div class="empty-panel">Trade Alpha is not available yet. Refresh benchmark history first so the dashboard can compare holdings against VOO.</div></section>`;
	      }}
	      const currency = alpha.currency || p.currency || "EUR";
	      const benchmark = alpha.benchmark_symbol || "VOO";
	      const rows = alpha.rows || [];
	      const rowHtml = rows.length ? rows.map((row) => {{
	        const hasAlpha = row.alpha_value !== null && row.alpha_value !== undefined;
	        const alphaClass = hasAlpha ? klass(row.alpha_value) : "muted";
	        const statusLabel = row.status === "beat" ? "Beat" : row.status === "below" ? "Below" : "Missing";
	        return `
	          <tr>
	            <td>
	              <div class="trade-alpha-holding">
	                <div class="analytics-avatar">${{stockInitials(row)}}</div>
	                <div>
	                  <div class="analytics-name">${{row.name}}</div>
	                  <div class="trade-alpha-date">${{row.ticker || "-"}} · ${{row.buy_date || "buy date missing"}}${{row.buy_date_source && row.buy_date_source !== "missing" ? "" : " · estimated/missing"}}</div>
	                </div>
	              </div>
	            </td>
	            <td>${{money(row.cost_basis, currency)}}</td>
	            <td>${{money(row.current_value, currency)}}</td>
	            <td><strong class="${{klass(row.your_return_value)}}">${{money(row.your_return_value, currency)}}</strong><div class="${{klass(row.your_return_pct)}}">${{pct(row.your_return_pct)}}</div></td>
	            <td>${{row.benchmark_return_value === null || row.benchmark_return_value === undefined ? "-" : money(row.benchmark_return_value, currency)}}<div>${{row.benchmark_return_pct === null || row.benchmark_return_pct === undefined ? "-" : pct(row.benchmark_return_pct)}}</div></td>
	            <td>${{hasAlpha ? `<span class="alpha-badge ${{alphaClass}}">${{sign(row.alpha_pct)}}${{pct(row.alpha_pct)}}</span><div class="${{alphaClass}}">${{money(row.alpha_value, currency)}}</div>` : "-"}}</td>
	            <td><span class="status-pill ${{hasAlpha ? "" : "muted"}}">${{statusLabel}}</span></td>
	          </tr>`;
	      }}).join("") : `<tr><td colspan="7"><div class="empty-panel">No open holdings available for Trade Alpha.</div></td></tr>`;
	      return `
	        <section class="trade-alpha-panel">
	          <div class="trade-alpha-head">
	            <div class="trade-alpha-icon">♜</div>
	            <div>
	              <h2 class="trade-alpha-title">Trade Alpha vs ${{benchmark}}</h2>
	              <div class="trade-alpha-subtitle">Compare ${{p.name || "your selected portfolio"}} against ${{alpha.benchmark_name || "Vanguard 500 Index Fund"}}</div>
	            </div>
	          </div>
	          <div class="trade-alpha-explainer">
	            <div class="trade-alpha-info"><strong>What is Trade Alpha?</strong><span>Alpha is the extra money you made or lost versus a simple benchmark alternative. Here the benchmark is ${{benchmark}}.</span></div>
	            <div class="trade-alpha-info"><strong>How it is calculated</strong><span>For each holding, the dashboard compares your return with what the same cost basis would have returned in ${{benchmark}} since your buy date.</span></div>
	            <div class="trade-alpha-info"><strong>How to read it</strong><span>Positive alpha means the holding beat ${{benchmark}} over your holding period. Negative alpha means ${{benchmark}} would have done better.</span></div>
	          </div>
	          <div class="trade-alpha-summary">
	            <article class="trade-alpha-kpi">
	              <div>
	                <div class="trade-alpha-label">Total Alpha</div>
	                <div class="trade-alpha-value ${{klass(alpha.alpha_value)}}">${{money(alpha.alpha_value, currency)}}</div>
	                <div class="trade-alpha-sub ${{klass(alpha.alpha_pct)}}">${{pct(alpha.alpha_pct)}}</div>
	              </div>
	            </article>
	            <article class="trade-alpha-kpi">
	              <div>
	                <div class="trade-alpha-label">↑ Beat Benchmark</div>
	                <div class="trade-alpha-value positive">${{alpha.beat_count || 0}}</div>
	                <div class="trade-alpha-sub">holdings</div>
	              </div>
	            </article>
	            <article class="trade-alpha-kpi">
	              <div>
	                <div class="trade-alpha-label">↓ Below Benchmark</div>
	                <div class="trade-alpha-value negative">${{alpha.below_count || 0}}</div>
	                <div class="trade-alpha-sub">holdings</div>
	              </div>
	            </article>
	            <article class="trade-alpha-kpi">
	              <div>
	                <div class="trade-alpha-label">Cost Basis</div>
	                <div class="trade-alpha-value">${{money(alpha.cost_basis, currency)}}</div>
	                <div class="trade-alpha-sub">${{alpha.holding_count || 0}} holdings</div>
	              </div>
	            </article>
	          </div>
	          <div class="trade-alpha-comparison">
	            <div class="trade-alpha-compare-card">
	              <div>Your Total Return <span class="info">●</span></div>
	              <div class="trade-alpha-compare-value ${{klass(alpha.your_return_value)}}">${{money(alpha.your_return_value, currency)}}</div>
	              <div class="${{klass(alpha.your_return_pct)}}">${{pct(alpha.your_return_pct)}}</div>
	            </div>
	            <div class="trade-alpha-compare-card">
	              <div>Performance Comparison</div>
	              <div style="display:flex; justify-content:center; gap:10px; margin-top:12px;"><span class="trade-alpha-pill">You</span><span style="align-self:center;">vs</span><span class="trade-alpha-pill gray">${{benchmark}}</span></div>
	            </div>
	            <div class="trade-alpha-compare-card">
	              <div>${{benchmark}} Would Return</div>
	              <div class="trade-alpha-compare-value ${{klass(alpha.benchmark_return_value)}}">${{money(alpha.benchmark_return_value, currency)}}</div>
	              <div class="${{klass(alpha.benchmark_return_pct)}}">${{pct(alpha.benchmark_return_pct)}}</div>
	            </div>
	          </div>
	          <div class="trade-alpha-table-wrap">
	            <table class="trade-alpha-table">
	              <thead>
	                <tr><th>Holding</th><th>Cost Basis</th><th>Current</th><th>Your Return</th><th>${{benchmark}}</th><th>Alpha</th><th>Status</th></tr>
	              </thead>
	              <tbody>${{rowHtml}}</tbody>
	            </table>
	          </div>
	          <div class="trade-alpha-note">${{alpha.note || ""}} Source: ${{alpha.source || "benchmark cache"}} · benchmark window: ${{alpha.benchmark_start_date || "-"}} to ${{alpha.benchmark_end_date || "-"}} · covered cost basis: ${{money(alpha.covered_cost_basis, currency)}}. Missing rows usually mean the buy date or benchmark price was not available.</div>
	        </section>`;
	    }}
	    function renderDiversificationPanel() {{
	      const p = selectedPortfolio();
	      const data = selectedDiversification();
	      const sectorCount = data.sectorRows.length;
	      const regionCount = data.geographicRows.length;
	      const recommendationRows = data.recommendations.map((item) => `<li>${{item}}</li>`).join("");
	      return `
	        <section class="diversification-panel">
	          <div class="diversification-grid">
	            <article class="diversification-card">
	              <div class="diversification-heading">
	                <div class="diversification-icon">◔</div>
	                <div>
	                  <h2 class="diversification-title">Diversification Score</h2>
	                  <div class="diversification-subtitle">HHI-based analysis for ${{p.name || "selected portfolio"}}</div>
	                </div>
	              </div>
	              <div class="diversification-score">
	                <div class="diversification-score-value">${{data.score.toFixed(1)}}</div>
	                <div class="diversification-score-label">Out of 100</div>
	              </div>
	              <div class="diversification-divider"></div>
	              <div class="diversification-subscore-grid">
	                <div>
	                  <div class="diversification-subscore-value">${{data.sectorScore.toFixed(1)}}</div>
	                  <div class="diversification-subscore-label">Sector <span class="info">●</span></div>
	                </div>
	                <div>
	                  <div class="diversification-subscore-value">${{data.geographicScore.toFixed(1)}}</div>
	                  <div class="diversification-subscore-label">Geographic <span class="info">●</span></div>
	                </div>
	              </div>
	              <div class="diversification-footnote">${{sectorCount}} sectors/cash buckets across ${{regionCount}} regions/cash buckets · ${{data.holdings.length}} holdings</div>
	            </article>
	            ${{renderDiversificationDonut("Sector Allocation", "▰", data.sectorRows, "Top sectors shown")}}
	            ${{renderDiversificationDonut("Geographic Allocation", "◎", data.geographicRows, "Top regions shown")}}
	          </div>
	          <aside class="diversification-recommendations">
	            <h3 class="diversification-rec-title"><span style="color:#ffb020">▲</span>Diversification Recommendations</h3>
	            <ul class="diversification-rec-list">${{recommendationRows}}</ul>
	          </aside>
	          <div class="diversification-breakdowns">
	            ${{renderDiversificationBreakdown("Sector Breakdown", "☷", "Sector", data.sectorRows, "#f2a40e")}}
	            ${{renderDiversificationBreakdown("Geographic Breakdown", "⚑", "Country", data.geographicRows, "#13b8c8")}}
	          </div>
	          <div class="diversification-note"><strong><span class="info">●</span> Note:</strong> Diversification score is based on the Herfindahl-Hirschman Index (HHI), measuring concentration across sectors and geographies. A higher score indicates better diversification. Scores above 60 are considered well-diversified.</div>
	          <div class="monte-note">The score uses current selected-portfolio holdings only. It does not judge whether the assets are good or bad investments.</div>
	        </section>`;
	    }}
	    function renderAssetPerformancePanel() {{
	      const p = selectedPortfolio();
	      const generated = String(state?.source_timestamps?.trade_republic || state?.source_timestamps?.snaptrade || state?.generated_at || "");
	      const since = selectedHistoryRows()[0]?.date || generated.slice(0, 10) || "first snapshot";
	      const rows = analyticsRows();
	      const tableRows = rows.length ? rows.map((row) => `
	        <tr>
	          <td>
	            <div class="analytics-asset">
	              <div class="analytics-avatar">${{stockInitials(row)}}</div>
	              <div>
	                <div class="analytics-name">${{row.name}}</div>
	                <div class="analytics-symbol">${{row.ticker || "-"}}</div>
	              </div>
	            </div>
	          </td>
	          <td class="${{klass(row.meanReturn)}}">${{sign(row.meanReturn)}}${{pct(row.meanReturn)}}</td>
	          <td>${{pct(row.volatility)}}</td>
	          <td class="${{row.sharpe >= 1 ? "positive" : row.sharpe < 0 ? "negative" : "sharpe-warn"}}">${{row.sharpe.toFixed(2)}}</td>
	        </tr>`).join("") : `<tr><td colspan="4"><div class="empty-panel">No asset performance rows are available for this selected portfolio yet.</div></td></tr>`;
	      return `
	        <section class="analytics-panel">
	          <div class="analytics-heading">
	            <div class="analytics-icon">▤</div>
	            <div>
	              <h2 class="analytics-title">Asset Performance</h2>
	              <div class="analytics-subtitle">${{p.name || "Selected portfolio"}} returns and risk metrics since ${{since}}</div>
	            </div>
	          </div>
	          <div class="analytics-table-wrap">
	            <table class="analytics-table">
	              <thead>
	                <tr>
	                  <th>Asset</th>
	                  <th>Mean Return <span class="info">●</span></th>
	                  <th>Volatility <span class="info">●</span></th>
	                  <th>Sharpe Ratio <span class="info">●</span></th>
	                </tr>
	              </thead>
	              <tbody>${{tableRows}}</tbody>
	            </table>
	          </div>
	          <div class="analytics-note">Mean return uses the current holding return for the selected portfolio. Volatility and Sharpe are proxy values until per-asset historical price series are wired into the dashboard.</div>
	        </section>`;
	    }}
	    function selectedAlerts() {{
	      const alerts = state?.price_alerts?.alerts || [];
	      return alerts.filter((alert) => (alert.portfolio_key || "combined:live") === selectedKey || selectedPortfolio().kind === "combined" && alert.portfolio_key === "combined:live");
	    }}
	    function renderAlertsTab() {{
	      const p = selectedPortfolio();
	      const currency = p.currency || "EUR";
	      const summary = state?.price_alerts?.summary || {{}};
	      const rows = selectedAlerts();
	      const closest = summary.closest;
	      const alertRows = rows.length ? rows.map((alert) => `
	        <tr>
	          <td>
	            <div class="holding-cell">
	              <div class="holding-logo">${{stockInitials({{ ticker: alert.symbol, name: alert.name }})}}</div>
	              <div><div class="holding-name">${{alert.name || alert.symbol}}</div><div class="holding-sub">${{alert.symbol}} · ${{alert.scope || "single"}}</div></div>
	            </div>
	          </td>
	          <td><span class="alert-status ${{alert.status || "paused"}}">${{alert.status || "paused"}}</span></td>
	          <td>${{alert.direction === "above" ? "Above" : "Below"}}</td>
	          <td>${{money(alert.target_price, alert.currency || currency)}}</td>
	          <td>${{alert.last_price === null || alert.last_price === undefined ? "-" : money(alert.last_price, alert.currency || currency)}}</td>
	          <td class="${{klass(-(Number(alert.distance_pct || 0)))}}">${{alert.distance_pct === null || alert.distance_pct === undefined ? "-" : pct(alert.distance_pct)}}</td>
	          <td>${{String(alert.created_at || "").slice(0, 19).replace("T", " ")}}</td>
	          <td>${{alert.triggered_at ? String(alert.triggered_at).slice(0, 19).replace("T", " ") : "-"}}</td>
	          <td>
	            <div class="alerts-actions">
	              <button class="action-btn" data-alert-status="${{alert.status === "active" ? "paused" : "active"}}" data-alert-id="${{alert.id}}">${{alert.status === "active" ? "Pause" : "Activate"}}</button>
	              <button class="action-btn danger" data-alert-status="deleted" data-alert-id="${{alert.id}}">Delete</button>
	            </div>
	          </td>
	        </tr>`).join("") : `<tr><td colspan="9"><div class="empty-panel">No alerts for this selected portfolio yet.</div></td></tr>`;
	      const holdingOptions = selectedHoldings().map((holding) => {{
	        const symbol = holding.ticker || holding.broker_symbol || holding.isin || "";
	        const price = holdingPrice(holding);
	        return symbol ? `<option value="${{symbol}}" data-name="${{holdingDisplayName(holding)}}" data-price="${{price || ""}}" data-currency="${{holding.currency || currency}}">${{symbol}} · ${{holdingDisplayName(holding)}}</option>` : "";
	      }}).join("");
	      document.getElementById("tabContent").innerHTML = `
	        <div class="dashboard-stack">
	          <section class="alert-summary-grid">
	            <article class="alert-card"><div class="alert-label">Total Alerts</div><div class="alert-value">${{summary.total || 0}}</div></article>
	            <article class="alert-card"><div class="alert-label">Active</div><div class="alert-value positive">${{summary.active || 0}}</div></article>
	            <article class="alert-card"><div class="alert-label">Triggered</div><div class="alert-value negative">${{summary.triggered || 0}}</div></article>
	            <article class="alert-card"><div class="alert-label">Closest</div><div class="alert-value">${{closest ? `${{closest.symbol}} ${{pct(closest.distance_pct)}}` : "-"}}</div></article>
	          </section>
	          <section class="alerts-panel">
	            <div class="alerts-header">
	              <div>
	                <h2 class="alerts-title">Price Alerts</h2>
	                <div class="analytics-subtitle">Read-only local alerts for ${{p.name || "selected portfolio"}}. Triggered alerts are stored in the local event log.</div>
	              </div>
	            </div>
	            <div class="alert-form-grid">
	              <label class="form-field">Holding<select class="mini-select" id="alertSymbol">${{holdingOptions}}</select></label>
	              <label class="form-field">Direction<select class="mini-select" id="alertDirection"><option value="below">Below</option><option value="above">Above</option></select></label>
	              <label class="form-field">Target Price<input class="mini-input" id="alertTarget" type="number" step="0.0001"></label>
	              <button class="action-btn green" id="createAlertBtn">Create Alert</button>
	              <label class="form-field">Bulk %<input class="mini-input" id="bulkPct" type="number" step="0.1" value="10"></label>
	              <button class="action-btn" id="bulkDropBtn">Bulk Drop</button>
	              <button class="action-btn danger" id="protectionBtn">Protection -15%</button>
	            </div>
	            <div class="alerts-table-wrap">
	              <table class="alerts-table">
	                <thead><tr><th>Alert</th><th>Status</th><th>Rule</th><th>Target</th><th>Last Price</th><th>Distance</th><th>Created</th><th>Triggered</th><th>Actions</th></tr></thead>
	                <tbody>${{alertRows}}</tbody>
	              </table>
	            </div>
	          </section>
	          <section class="alerts-panel">
	            <div class="alerts-header"><h2 class="alerts-title">Trigger History</h2><span class="perf-sub">${{(state?.price_alerts?.events || []).length}} recent events</span></div>
	            <div class="alerts-table-wrap">
	              <table class="alerts-table">
	                <thead><tr><th>Event</th><th>Portfolio</th><th>Symbol</th><th>Rule</th><th>Target</th><th>Triggered Price</th><th>Time</th></tr></thead>
	                <tbody>${{(state?.price_alerts?.events || []).map((event) => `
	                  <tr><td>${{event.event}}</td><td>${{event.portfolio_key}}</td><td>${{event.symbol}}</td><td>${{event.direction}}</td><td>${{money(event.target_price, event.currency || currency)}}</td><td>${{money(event.trigger_price, event.currency || currency)}}</td><td>${{String(event.triggered_at || "").slice(0, 19).replace("T", " ")}}</td></tr>
	                `).join("") || `<tr><td colspan="7"><div class="empty-panel">No alert triggers yet.</div></td></tr>`}}</tbody>
	              </table>
	            </div>
	          </section>
	        </div>`;
	      const symbolSelect = document.getElementById("alertSymbol");
	      const targetInput = document.getElementById("alertTarget");
	      if (symbolSelect && targetInput) {{
	        const fillPrice = () => {{
	          const selected = symbolSelect.selectedOptions[0];
	          targetInput.value = selected?.dataset.price ? Number(selected.dataset.price).toFixed(4) : "";
	        }};
	        symbolSelect.addEventListener("change", fillPrice);
	        fillPrice();
	      }}
	      document.getElementById("createAlertBtn")?.addEventListener("click", async () => {{
	        const selected = symbolSelect?.selectedOptions[0];
	        await apiPost("/api/alerts", {{
	          portfolio_key: selectedKey,
	          symbol: symbolSelect?.value,
	          name: selected?.dataset.name,
	          target_price: Number(targetInput?.value || 0),
	          direction: document.getElementById("alertDirection")?.value || "below",
	          basis_price: Number(selected?.dataset.price || 0),
	          currency: selected?.dataset.currency || currency,
	        }});
	        await load();
	      }});
	      document.getElementById("bulkDropBtn")?.addEventListener("click", async () => {{
	        await apiPost("/api/alerts/bulk", {{ portfolio_key: selectedKey, direction: "below", threshold_pct: Number(document.getElementById("bulkPct")?.value || 10), scope: "bulk_drop" }});
	        await load();
	      }});
	      document.getElementById("protectionBtn")?.addEventListener("click", async () => {{
	        await apiPost("/api/alerts/bulk", {{ portfolio_key: selectedKey, direction: "below", threshold_pct: 15, scope: "portfolio_protection" }});
	        await load();
	      }});
	      document.querySelectorAll("[data-alert-id]").forEach((button) => button.addEventListener("click", async () => {{
	        await apiPost(`/api/alerts/${{button.dataset.alertId}}`, {{ status: button.dataset.alertStatus }});
	        await load();
	      }}));
	    }}
	    function assistantFacts() {{
	      const p = selectedPortfolio();
	      const holdings = selectedHoldings();
	      const sortedByPl = holdings.slice().sort((a, b) => Number(a.unrealized_pl || 0) - Number(b.unrealized_pl || 0));
	      const worst = sortedByPl[0];
	      const best = sortedByPl[sortedByPl.length - 1];
	      const allocation = allocationRows("stock").slice(0, 5);
	      return {{ p, holdings, worst, best, allocation }};
	    }}
	    function answerAssistant(questionKey) {{
	      const {{ p, holdings, worst, best, allocation }} = assistantFacts();
	      const currency = p.currency || "EUR";
	      if (questionKey === "worst") {{
	        if (!worst) return "<h3>No holdings found</h3><p>The selected portfolio has no position rows to analyze.</p>";
	        return `<h3>Worst performer</h3><p>${{holdingDisplayName(worst)}} is currently the weakest open holding by unrealized P/L.</p><ul><li>Unrealized P/L: <span class="${{klass(worst.unrealized_pl)}}">${{sign(worst.unrealized_pl)}}${{money(worst.unrealized_pl, worst.currency || currency)}} (${{pct(worst.unrealized_pl_pct)}})</span></li><li>Market value: ${{money(worst.current_value, worst.currency || currency)}}</li><li>Broker: ${{worst.institution || worst.broker || "-"}}</li></ul>`;
	      }}
	      if (questionKey === "today") {{
	        return `<h3>Why is the portfolio moving today?</h3><p>The local data currently reports daily portfolio P/L of <span class="${{klass(p.today_gain)}}">${{sign(p.today_gain)}}${{money(p.today_gain, currency)}} (${{pct(p.today_gain_pct)}})</span>.</p><p>Most broker snapshots do not yet provide full per-position daily movement, so this answer is limited to the normalized fields available in the dashboard state.</p>`;
	      }}
	      if (questionKey === "allocation") {{
	        return `<h3>Largest allocations</h3><p>These are the largest current weights in the selected portfolio:</p><ul>${{allocation.map((row) => `<li>${{row.name}}: ${{pct(row.pct)}} · ${{money(row.value, currency)}}</li>`).join("")}}</ul>`;
	      }}
	      if (questionKey === "best") {{
	        if (!best) return "<h3>No holdings found</h3><p>The selected portfolio has no position rows to analyze.</p>";
	        return `<h3>Best performer</h3><p>${{holdingDisplayName(best)}} is currently the strongest open holding by unrealized P/L.</p><ul><li>Unrealized P/L: <span class="${{klass(best.unrealized_pl)}}">${{sign(best.unrealized_pl)}}${{money(best.unrealized_pl, best.currency || currency)}} (${{pct(best.unrealized_pl_pct)}})</span></li><li>Market value: ${{money(best.current_value, best.currency || currency)}}</li></ul>`;
	      }}
	      return `<h3>Portfolio summary</h3><p>${{p.name || "The selected portfolio"}} has ${{holdings.length}} holdings, total worth ${{money(p.total_worth, currency)}}, and total capital gain <span class="${{klass(p.capital_gain)}}">${{sign(p.capital_gain)}}${{money(p.capital_gain, currency)}}</span>.</p>`;
	    }}
	    function renderAssistantAnswer(payload) {{
	      if (!payload) return answerAssistant("summary");
	      const list = (items) => Array.isArray(items) && items.length ? `<ul>${{items.map((item) => `<li>${{escapeHtml(item)}}</li>`).join("")}}</ul>` : `<div class="perf-sub">No items returned.</div>`;
	      return `
	        <h3>${{escapeHtml(payload.title || "Portfolio answer")}}</h3>
	        <p>${{escapeHtml(payload.answer || "")}}</p>
	        <div class="assistant-answer-section"><strong>Key points</strong>${{list(payload.key_points)}}</div>
	        <div class="assistant-answer-section"><strong>Numbers used</strong>${{list(payload.numbers_used)}}</div>
	        <div class="assistant-answer-section"><strong>Limits and risks</strong>${{list(payload.risks_or_limitations)}}</div>
	        <div class="assistant-answer-section"><strong>Follow-up questions</strong>${{list(payload.follow_up_questions)}}</div>
	        <div class="assistant-answer-section perf-sub">Model: ${{escapeHtml(payload.provider || "-")}} / ${{escapeHtml(payload.model || "-")}}</div>`;
	    }}
    async function askAssistant(question) {{
      const answer = document.getElementById("assistantAnswer");
      const askBtn = document.getElementById("assistantAskBtn");
      if (!answer) return;
      const cleanQuestion = String(question || "").trim();
      if (!cleanQuestion) {{
        answer.innerHTML = `<h3>Ask a question</h3><p>Type a portfolio question or choose one of the prompt buttons.</p>`;
        return;
      }}
      lastAssistantQuestion = cleanQuestion;
      localStorage.setItem("lastAssistantQuestion", cleanQuestion);
      assistantIsLoading = true;
      lastAssistantError = "";
      lastAssistantAnswer = null;
      answer.innerHTML = `<h3>Thinking...</h3><p>Reading the selected portfolio state and asking the configured LLM.</p>`;
      if (askBtn) askBtn.textContent = "Asking...";
      try {{
        const response = await apiPost("/api/assistant", {{ portfolio_key: selectedKey, question: cleanQuestion }});
        if (!response.ok) {{
          lastAssistantError = response.error || "The LLM call failed.";
          answer.innerHTML = `<h3>Assistant unavailable</h3><p>${{escapeHtml(lastAssistantError)}}</p><p class="perf-sub">Check assistant provider credentials or local model availability.</p>`;
          return;
        }}
        lastAssistantAnswer = response.answer;
        answer.innerHTML = renderAssistantAnswer(response.answer);
      }} catch (error) {{
        lastAssistantError = error?.message || String(error);
        answer.innerHTML = `<h3>Assistant unavailable</h3><p>${{escapeHtml(lastAssistantError)}}</p>`;
      }} finally {{
        assistantIsLoading = false;
        if (askBtn) askBtn.textContent = "Ask AI";
      }}
    }}
    function currentAssistantHtml() {{
      if (assistantIsLoading) return `<h3>Thinking...</h3><p>Reading the selected portfolio state and asking the configured LLM.</p>`;
      if (lastAssistantError) return `<h3>Assistant unavailable</h3><p>${{escapeHtml(lastAssistantError)}}</p>`;
      if (lastAssistantAnswer) return renderAssistantAnswer(lastAssistantAnswer);
      return answerAssistant("summary");
    }}
	    function renderAssistantTab() {{
	      document.getElementById("tabContent").innerHTML = `
	        <section class="assistant-layout">
	          <aside class="assistant-panel">
	            <div class="assistant-header"><h2 class="assistant-title">AI Portfolio Assistant</h2></div>
	            <div class="assistant-main">
	              <div class="assistant-note">This assistant reads the local normalized portfolio state, then asks the configured LLM for a plain-English answer. It is read-only and does not place trades.</div>
	              <div class="assistant-question-box">
	                <textarea id="assistantQuestion" class="assistant-question-input" placeholder="Ask anything about this selected portfolio...">${{escapeHtml(lastAssistantQuestion)}}</textarea>
	                <div class="assistant-actions">
	                  <button class="action-btn green" id="assistantAskBtn">Ask AI</button>
	                  <span class="perf-sub">Grounded on ${{selectedPortfolio().name || "selected portfolio"}}</span>
	                </div>
	              </div>
	              <div class="assistant-prompts">
	                <button class="prompt-btn" data-question="Summarize this selected portfolio and explain the main risks in plain English.">Summarize this selected portfolio</button>
	                <button class="prompt-btn" data-question="What is my worst performer and how much is it losing?">What is my worst performer?</button>
	                <button class="prompt-btn" data-question="What is my best performer and how much is it winning?">What is my best performer?</button>
	                <button class="prompt-btn" data-question="Why is my portfolio down or up today based on the available data?">Why is my portfolio down or up today?</button>
	                <button class="prompt-btn" data-question="Where is my money concentrated by stock, sector, country, and asset class?">Where is my money concentrated?</button>
	                <button class="prompt-btn" data-question="Review my rebalancing drift and explain what I should review before making changes.">Explain my rebalance drift</button>
	              </div>
	            </div>
	          </aside>
	          <article class="assistant-panel">
	            <div class="assistant-header"><h2 class="assistant-title">Answer</h2><span class="perf-sub">${{selectedPortfolio().name || "Selected portfolio"}}</span></div>
	            <div class="assistant-main"><div class="assistant-answer" id="assistantAnswer">${{currentAssistantHtml()}}</div></div>
	          </article>
	        </section>`;
	      document.getElementById("assistantQuestion")?.addEventListener("input", (event) => {{
	        lastAssistantQuestion = event.target.value || "";
	        localStorage.setItem("lastAssistantQuestion", lastAssistantQuestion);
	      }});
	      document.getElementById("assistantAskBtn")?.addEventListener("click", () => askAssistant(document.getElementById("assistantQuestion")?.value || ""));
	      document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => {{
	        const question = button.dataset.question || "";
	        const input = document.getElementById("assistantQuestion");
	        if (input) input.value = question;
	        askAssistant(question);
	      }}));
	    }}
	    function renderAnalyticsTab() {{
	      document.getElementById("tabContent").innerHTML = `
	        ${{renderAnalyticsMethodTabs()}}
	        ${{renderAnalyticsMethodPanel()}}`;
	      document.querySelectorAll("[data-analytics-mode]").forEach((button) => button.addEventListener("click", () => {{
	        analyticsMode = button.dataset.analyticsMode || "monte_carlo";
	        renderAnalyticsTab();
	        renderStatus();
	      }}));
	      const benchmarkTooltip = document.getElementById("benchmarkTooltip");
	      document.querySelectorAll(".benchmark-hit").forEach((point) => {{
	        point.addEventListener("mouseenter", () => {{
	          if (!benchmarkTooltip) return;
	          const parts = String(point.dataset.tooltip || "").split("|");
	          benchmarkTooltip.innerHTML = `<strong>${{parts[0] || ""}}</strong>${{parts.slice(1).map((part) => `<div>${{part}}</div>`).join("")}}`;
	          benchmarkTooltip.classList.add("visible");
	        }});
	        point.addEventListener("mousemove", (event) => {{
	          if (!benchmarkTooltip) return;
	          benchmarkTooltip.style.left = `${{event.clientX}}px`;
	          benchmarkTooltip.style.top = `${{event.clientY}}px`;
	        }});
	        point.addEventListener("mouseleave", () => {{
	          if (!benchmarkTooltip) return;
	          benchmarkTooltip.classList.remove("visible");
	        }});
	      }});
	    }}
	    function renderPlaceholderTab() {{
	      const tab = tabs.find(([key]) => key === activeTab);
	      document.getElementById("tabContent").innerHTML = `<section class="placeholder">${{tab?.[2] || "Tab"}} will be built next. The selected portfolio metric row stays visible here.</section>`;
    }}
    function renderStatus() {{
      const errors = state?.source_errors || [];
      const timestamps = Object.entries(state?.source_timestamps || {{}}).map(([key, value]) => `${{key}}: ${{value || "-"}}`).join(" | ");
      const sourceText = `${{timestamps || "No source timestamps"}}${{errors.length ? " | Source issues: " + errors.length : ""}}`;
      const refreshText = lastRefreshMessage ? ` | ${{lastRefreshMessage}}` : " | Manual refresh only";
      const buttonText = manualRefreshLoading ? "Refreshing..." : "Refresh portfolios";
      const line = document.getElementById("statusLine");
      line.innerHTML = `
        <span class="status-text">${{escapeHtml(sourceText + refreshText)}}</span>
        <button class="refresh-btn" id="refreshNowBtn" type="button" ${{manualRefreshLoading ? "disabled" : ""}}>${{buttonText}}</button>`;
      document.getElementById("refreshNowBtn")?.addEventListener("click", refreshNow);
    }}
    function render() {{
      if (!state) return;
      if (!portfolioList().some((item) => item.key === selectedKey)) selectedKey = state.combined_portfolio?.key || portfolioList()[0]?.key;
      renderMetrics();
      renderGoalPanel();
	      renderTabs();
	      if (activeTab === "portfolios") renderPortfoliosTab();
	      else if (activeTab === "dashboard") renderDashboardTab();
	      else if (activeTab === "insights") renderInsightsTab();
	      else if (activeTab === "dividends") renderDividendsTab();
	      else if (activeTab === "analytics") renderAnalyticsTab();
	      else if (activeTab === "alerts") renderAlertsTab();
	      else if (activeTab === "assistant") renderAssistantTab();
	      else renderPlaceholderTab();
      renderStatus();
    }}
    async function load() {{
      const response = await fetch("/api/state", {{ cache: "no-store" }});
      state = await response.json();
      render();
    }}
    function formatRefreshEvents(events) {{
      if (!Array.isArray(events) || !events.length) return "Refresh finished";
      return events.map((event) => {{
        const source = event.source || "source";
        const status = event.status || "unknown";
        if (status === "failed") return `${{source}} failed: ${{event.error_type || "error"}}`;
        if (status === "stale") return `${{source}} kept last good data: ${{event.error_type || "error"}}`;
        if (status === "skipped") return `${{source}} skipped`;
        return `${{source}} refreshed`;
      }}).join(" | ");
    }}
    async function refreshNow() {{
      if (manualRefreshLoading) return;
      manualRefreshLoading = true;
      lastRefreshMessage = "Refreshing Trade Republic and SnapTrade...";
      renderStatus();
      try {{
        const response = await apiPost("/api/refresh", {{}});
        if (response.state) state = response.state;
        else await load();
        lastRefreshMessage = formatRefreshEvents(response.events);
        render();
      }} catch (error) {{
        lastRefreshMessage = `Refresh failed: ${{error?.message || String(error)}}`;
        renderStatus();
      }} finally {{
        manualRefreshLoading = false;
        renderStatus();
      }}
    }}
    load().catch((error) => {{
      document.getElementById("statusLine").textContent = String(error);
    }});
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
