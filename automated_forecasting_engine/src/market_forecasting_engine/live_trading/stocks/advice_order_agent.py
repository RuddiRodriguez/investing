from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker, load_env_file


LIVE_BASE_URL = "https://api.alpaca.markets"
DEFAULT_TICKERS = ("ASML", "MMM", "WM")
DEFAULT_ORDER_PREFIX = "liveadvice"
WATCHED_EXPIRED_PREFIXES = ("liveadvice", "livepllmdip", "pllmdip")


def main() -> None:
    args = build_parser().parse_args()
    load_env_file()
    broker = live_broker(args)
    report = run_once(args=args, broker=broker)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "live_advice_order_agent_report.json"
    write_json(report_path, report)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True, default=str), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-shot live Alpaca advice-order agent.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/live_advice_order_agent")
    parser.add_argument("--forecast-output-root", default="automated_forecasting_engine/runs/pure_llm_stock_forecasts")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--provider", default="alpaca")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--bars", type=int, default=30)
    parser.add_argument("--llm-provider", default="openai")
    parser.add_argument("--llm-model", default="gpt-5.4-mini-2026-03-17")
    parser.add_argument("--ceo-llm-provider", default="openai")
    parser.add_argument("--ceo-llm-model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--trader-profile", choices=("conservative", "medium", "aggressive"), default="medium")
    parser.add_argument("--only-if-expired", action="store_true", help="Skip forecasting unless a watched order expired.")
    parser.add_argument("--force-forecast", action="store_true", help="Forecast all configured tickers without requiring an expired order.")
    parser.add_argument("--expired-lookback-hours", type=float, default=30.0)
    parser.add_argument("--buying-power-fraction", type=float, default=0.95)
    parser.add_argument("--min-notional", type=float, default=1.0)
    parser.add_argument("--max-notional-per-order", type=float, default=None)
    parser.add_argument("--max-symbol-market-value-pct-equity", type=float, default=0.60)
    parser.add_argument("--breakout-chase-pct", type=float, default=0.003)
    parser.add_argument("--order-prefix", default=DEFAULT_ORDER_PREFIX)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-order-risk", action="store_true")
    return parser


def run_once(*, args: argparse.Namespace, broker: AlpacaPaperBroker) -> dict[str, Any]:
    now = datetime.now(UTC)
    tickers = parse_tickers(args.tickers)
    account = broker.account()
    clock = broker.clock()
    positions = broker.positions()
    open_orders = broker.orders(status="open", limit=500, direction="desc", nested=True)
    recent_orders = broker.orders(status="all", limit=500, direction="desc", nested=True)
    assets = {ticker: safe_asset(broker, ticker) for ticker in tickers}
    position_by_symbol = {str(row.get("symbol") or "").upper(): row for row in positions if isinstance(row, dict)}
    open_by_symbol = orders_by_symbol(open_orders)
    expired_by_symbol = expired_orders_by_symbol(recent_orders, tickers=tickers, lookback_hours=float(args.expired_lookback_hours), now=now)
    execution_blocks = base_execution_blocks(args=args, account=account, clock=clock)

    reports: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    forecast_dir = Path(args.forecast_output_root) / now.strftime("live_advice_%Y%m%d_%H%M%S")
    for ticker in tickers:
        ticker_report = {
            "ticker": ticker,
            "open_orders": compact_orders(open_by_symbol.get(ticker, [])),
            "expired_orders": compact_orders(expired_by_symbol.get(ticker, [])),
            "position": compact_position(position_by_symbol.get(ticker)),
            "asset": compact_asset(assets.get(ticker)),
            "forecast": None,
            "entry": None,
            "blocks": [],
            "effects": [],
        }
        symbol_blocks = symbol_execution_blocks(asset=assets.get(ticker), open_orders=open_by_symbol.get(ticker, []), ticker=ticker)
        should_forecast = should_run_forecast(
            args=args,
            ticker=ticker,
            open_orders=open_by_symbol.get(ticker, []),
            expired_orders=expired_by_symbol.get(ticker, []),
        )
        if not should_forecast:
            ticker_report["blocks"].append("forecast_not_needed")
            reports.append(ticker_report)
            continue
        forecast_path = run_forecast(args=args, ticker=ticker, position=position_by_symbol.get(ticker), account=account, output_dir=forecast_dir)
        forecast_record = read_json(forecast_path)
        ticker_report["forecast"] = compact_forecast_record(forecast_record, forecast_path)
        entry = advice_entry(forecast_record, max_breakout_chase_pct=float(args.breakout_chase_pct))
        ticker_report["entry"] = entry
        blocks = [*execution_blocks, *symbol_blocks, *entry.get("blocks", [])]
        ticker_report["blocks"] = blocks
        if not blocks:
            candidates.append({"ticker": ticker, "entry": entry, "report": ticker_report, "position": position_by_symbol.get(ticker)})
        reports.append(ticker_report)

    buying_power = number(account.get("buying_power")) or 0.0
    equity = number(account.get("equity") or account.get("portfolio_value")) or 0.0
    allocatable = max(0.0, buying_power * max(0.0, min(1.0, float(args.buying_power_fraction))))
    per_order = allocatable / len(candidates) if candidates else 0.0
    if args.max_notional_per_order is not None:
        per_order = min(per_order, float(args.max_notional_per_order))

    side_effects: list[dict[str, Any]] = []
    for item in candidates:
        ticker = item["ticker"]
        ticker_report = item["report"]
        symbol_cap = remaining_symbol_capacity(
            equity=equity,
            position=item.get("position"),
            max_symbol_market_value_pct_equity=float(args.max_symbol_market_value_pct_equity),
        )
        notional = min(per_order, symbol_cap)
        payload, sizing_blocks = order_payload(
            ticker=ticker,
            entry=item["entry"],
            notional=notional,
            min_notional=float(args.min_notional),
            prefix=str(args.order_prefix),
            now=now,
        )
        ticker_report["sizing"] = {"allocatable": round(allocatable, 2), "per_order": round(per_order, 2), "symbol_capacity": round(symbol_cap, 2)}
        if sizing_blocks:
            ticker_report["blocks"].extend(sizing_blocks)
            continue
        ticker_report["order_payload"] = payload
        effects = submit_if_allowed(broker=broker, args=args, payload=payload)
        ticker_report["effects"] = effects
        side_effects.extend(effects)

    return {
        "generated_at": now.isoformat(),
        "mode": "live_advice_order_agent",
        "dry_run": not bool(args.execute_live_orders),
        "policy": {
            "only_limit_buy_orders": True,
            "no_market_orders": True,
            "default_dry_run": True,
            "live_submit_requires_execute_live_orders_and_confirm_live_order_risk": True,
            "forecast_rerun_required_when_watched_order_expired": True,
        },
        "account": {
            "status": account.get("status"),
            "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
            "equity": account.get("equity"),
            "portfolio_value": account.get("portfolio_value"),
            "trading_blocked": account.get("trading_blocked"),
            "account_blocked": account.get("account_blocked"),
        },
        "clock": {key: clock.get(key) for key in ("timestamp", "is_open", "next_open", "next_close")},
        "execution_blocks": execution_blocks,
        "tickers": reports,
        "side_effects": side_effects,
    }


def should_run_forecast(*, args: argparse.Namespace, ticker: str, open_orders: list[dict[str, Any]], expired_orders: list[dict[str, Any]]) -> bool:
    if open_orders:
        return False
    if args.force_forecast:
        return True
    if expired_orders:
        return True
    return not bool(args.only_if_expired)


def run_forecast(*, args: argparse.Namespace, ticker: str, position: dict[str, Any] | None, account: dict[str, Any], output_dir: Path) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "market_forecasting_engine.pure_llm_stock_forecaster",
        "--ticker",
        ticker,
        "--company",
        ticker,
        "--provider",
        str(args.provider),
        "--start",
        str(args.start),
        "--interval",
        str(args.interval),
        "--bars",
        str(int(args.bars)),
        "--output-dir",
        str(output_dir),
        "--llm-env-file",
        str(Path(args.env_file).expanduser().resolve()),
        "--llm-provider",
        str(args.llm_provider),
        "--llm-model",
        str(args.llm_model),
        "--fallback-llm-provider",
        "openai",
        "--fallback-llm-model",
        "gpt-5.4-mini-2026-03-17",
        "--ceo-llm-provider",
        str(args.ceo_llm_provider),
        "--ceo-llm-model",
        str(args.ceo_llm_model),
        "--trader-profile",
        str(args.trader_profile),
        "--holding-status",
        "owned" if position else "not_owned",
        "--account-equity",
        str(account.get("equity") or account.get("portfolio_value") or ""),
        "--portfolio-notes",
        "Live Alpaca account context. One-shot advice order agent. No market orders.",
    ]
    if position:
        append_position_args(cmd, position)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3])
    if env.get("ALPACA_API_KEY_ID_LIVE") and not env.get("ALPACA_API_KEY_ID"):
        env["ALPACA_API_KEY_ID"] = env["ALPACA_API_KEY_ID_LIVE"]
    if env.get("ALPACA_API_SECRET_KEY_LIVE") and not env.get("ALPACA_API_SECRET_KEY"):
        env["ALPACA_API_SECRET_KEY"] = env["ALPACA_API_SECRET_KEY_LIVE"]
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[5], env=env, check=True)
    return output_dir / f"{safe_name(ticker)}_pure_llm_stock_forecast.json"


def advice_entry(record: dict[str, Any], *, max_breakout_chase_pct: float) -> dict[str, Any]:
    advice = record.get("advice") if isinstance(record.get("advice"), dict) else {}
    final = advice.get("final_advice") if isinstance(advice.get("final_advice"), dict) else {}
    forecast = record.get("forecast") if isinstance(record.get("forecast"), dict) else {}
    current = number(forecast.get("current_price"))
    action_now = str(final.get("action_now") or "").lower()
    buy_now = number(final.get("buy_now_price"))
    buy_lower = number(final.get("buy_lower_price") or final.get("buy_lower_zone_high"))
    breakout = number(final.get("buy_above_breakout_price"))
    blocks: list[str] = []
    if action_now in {"avoid", "sell_now", "trim_or_reduce"}:
        blocks.append(f"advice_action_{action_now}")
    if action_now == "buy_now" and buy_now:
        return {"style": "buy_now", "limit_price": round(buy_now, 2), "blocks": blocks, "advice": compact_final_advice(final)}
    if current is not None and breakout is not None and current >= breakout:
        max_price = breakout * (1.0 + max(0.0, max_breakout_chase_pct))
        if current > max_price:
            blocks.append("breakout_too_far_above_advice_price")
        return {"style": "buy_breakout", "limit_price": round(min(current, max_price), 2), "blocks": blocks, "advice": compact_final_advice(final)}
    if buy_lower is not None:
        return {"style": "buy_lower_resting_limit", "limit_price": round(buy_lower, 2), "blocks": blocks, "advice": compact_final_advice(final)}
    blocks.append("no_buy_price_in_advice")
    return {"style": "no_entry", "limit_price": None, "blocks": blocks, "advice": compact_final_advice(final)}


def order_payload(*, ticker: str, entry: dict[str, Any], notional: float, min_notional: float, prefix: str, now: datetime) -> tuple[dict[str, Any] | None, list[str]]:
    limit_price = number(entry.get("limit_price"))
    if limit_price is None or limit_price <= 0:
        return None, ["missing_limit_price"]
    if notional < min_notional:
        return None, ["insufficient_buying_power_for_min_notional"]
    qty = round(notional / limit_price, 6)
    if qty <= 0:
        return None, ["calculated_qty_zero"]
    return {
        "symbol": ticker,
        "side": "buy",
        "type": "limit",
        "qty": str(qty),
        "limit_price": str(round(limit_price, 2)),
        "time_in_force": "day",
        "client_order_id": client_order_id(prefix, ticker, entry.get("style") or "entry", now),
    }, []


def submit_if_allowed(*, broker: AlpacaPaperBroker, args: argparse.Namespace, payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if not args.execute_live_orders:
        return [{"action": "would_submit_live_limit_buy", "payload": payload}]
    if not args.confirm_live_order_risk:
        return [{"action": "submit_blocked", "reason": "missing_confirm_live_order_risk", "payload": payload}]
    try:
        return [{"action": "submitted_live_limit_buy", "order": broker._request("POST", "/v2/orders", payload), "payload": payload}]
    except RuntimeError as exc:
        return [{"action": "live_limit_buy_rejected", "error": str(exc), "payload": payload}]


def live_broker(args: argparse.Namespace) -> AlpacaPaperBroker:
    base_url = str(args.base_url or os.getenv("ALPACA_TRADING_BASE_URL_LIVE") or os.getenv("ALPACA_API_BASE_URL_LIVE") or LIVE_BASE_URL).rstrip("/")
    if "paper-api" in base_url:
        raise SystemExit("Refusing to run live advice order agent against paper Alpaca endpoint.")
    key = os.getenv("ALPACA_API_KEY_ID_LIVE") or os.getenv("APCA_API_KEY_ID_LIVE") or os.getenv("ALPACA_LIVE_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY_LIVE") or os.getenv("APCA_API_SECRET_KEY_LIVE") or os.getenv("ALPACA_LIVE_API_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("Live Alpaca credentials are required.")
    return AlpacaPaperBroker(base_url=base_url, key_id=key, secret_key=secret)


def base_execution_blocks(*, args: argparse.Namespace, account: dict[str, Any], clock: dict[str, Any]) -> list[str]:
    blocks: list[str] = []
    if not bool(clock.get("is_open")):
        blocks.append("market_closed")
    if str(account.get("status") or "").upper() != "ACTIVE":
        blocks.append("account_not_active")
    if account.get("trading_blocked") or account.get("account_blocked") or account.get("trade_suspended_by_user"):
        blocks.append("account_trading_blocked")
    if args.execute_live_orders and not args.confirm_live_order_risk:
        blocks.append("missing_confirm_live_order_risk")
    return blocks


def symbol_execution_blocks(*, asset: dict[str, Any] | None, open_orders: list[dict[str, Any]], ticker: str) -> list[str]:
    blocks: list[str] = []
    if open_orders:
        blocks.append("existing_open_order_for_symbol")
    if asset is None:
        blocks.append("asset_not_found")
    else:
        if not asset.get("tradable"):
            blocks.append("asset_not_tradable")
        if not asset.get("fractionable"):
            blocks.append("asset_not_fractionable_for_small_account")
    return blocks


def expired_orders_by_symbol(
    orders: list[dict[str, Any]],
    *,
    tickers: list[str],
    lookback_hours: float,
    now: datetime,
) -> dict[str, list[dict[str, Any]]]:
    cutoff = now - timedelta(hours=max(1.0, lookback_hours))
    wanted = set(tickers)
    output = {ticker: [] for ticker in tickers}
    for order in orders:
        symbol = str(order.get("symbol") or "").upper()
        if symbol not in wanted:
            continue
        if str(order.get("status") or "").lower() != "expired":
            continue
        client_id = str(order.get("client_order_id") or "").lower()
        if not any(client_id.startswith(prefix) for prefix in WATCHED_EXPIRED_PREFIXES):
            continue
        expired_at = parse_time(order.get("expired_at") or order.get("updated_at") or order.get("submitted_at"))
        if expired_at is not None and expired_at < cutoff:
            continue
        output[symbol].append(order)
    return output


def orders_by_symbol(orders: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for order in orders:
        symbol = str(order.get("symbol") or "").upper()
        if symbol:
            output.setdefault(symbol, []).append(order)
    return output


def remaining_symbol_capacity(*, equity: float, position: dict[str, Any] | None, max_symbol_market_value_pct_equity: float) -> float:
    if equity <= 0:
        return 0.0
    cap = equity * max(0.0, max_symbol_market_value_pct_equity)
    current = number((position or {}).get("market_value")) or 0.0
    return max(0.0, cap - current)


def safe_asset(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any] | None:
    try:
        return broker._request("GET", f"/v2/assets/{symbol.upper()}")
    except RuntimeError:
        return None


def append_position_args(command: list[str], position: dict[str, Any]) -> None:
    for flag, key in [("--entry-price", "avg_entry_price"), ("--quantity", "qty"), ("--position-value", "market_value")]:
        value = position.get(key)
        if value is not None:
            command.extend([flag, str(value)])


def parse_tickers(raw: str) -> list[str]:
    return [ticker for ticker in dict.fromkeys(part.strip().upper() for part in raw.split(",") if part.strip())]


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(UTC)
    except ValueError:
        return None


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value.upper())


def client_order_id(prefix: str, symbol: str, style: str, now: datetime) -> str:
    clean_style = "".join(char for char in str(style).lower() if char.isalnum() or char == "_")[:12] or "entry"
    return f"{prefix}_{symbol.upper()}_{clean_style}_{now.strftime('%Y%m%d%H%M%S')}"[:48]


def compact_final_advice(final: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "action_now",
        "headline",
        "buy_now_price",
        "buy_lower_price",
        "buy_lower_zone_low",
        "buy_lower_zone_high",
        "buy_above_breakout_price",
        "invalidation_price",
        "sell_or_trim_price",
        "why_not_buy_now",
    ]
    return {key: final.get(key) for key in keys if key in final}


def compact_forecast_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    advice = record.get("advice") if isinstance(record.get("advice"), dict) else {}
    forecast = record.get("forecast") if isinstance(record.get("forecast"), dict) else {}
    return {
        "path": str(path),
        "decision": advice.get("decision"),
        "current_price": forecast.get("current_price"),
        "final_advice": compact_final_advice(advice.get("final_advice") if isinstance(advice.get("final_advice"), dict) else {}),
    }


def compact_orders(orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": short_id(order.get("id")),
            "client_order_id": order.get("client_order_id"),
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "type": order.get("type"),
            "qty": order.get("qty"),
            "limit_price": order.get("limit_price"),
            "status": order.get("status"),
            "submitted_at": order.get("submitted_at"),
            "expired_at": order.get("expired_at"),
        }
        for order in orders
    ]


def compact_position(position: dict[str, Any] | None) -> dict[str, Any] | None:
    if not position:
        return None
    return {
        "symbol": position.get("symbol"),
        "qty": position.get("qty"),
        "market_value": position.get("market_value"),
        "avg_entry_price": position.get("avg_entry_price"),
        "unrealized_pl": position.get("unrealized_pl"),
    }


def compact_asset(asset: dict[str, Any] | None) -> dict[str, Any] | None:
    if not asset:
        return None
    return {"symbol": asset.get("symbol"), "status": asset.get("status"), "tradable": asset.get("tradable"), "fractionable": asset.get("fractionable")}


def short_id(value: Any) -> str | None:
    if not value:
        return None
    text = str(value)
    return text[:8] + "..." if len(text) > 8 else text


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": report["mode"],
        "dry_run": report["dry_run"],
        "buying_power": report["account"]["buying_power"],
        "market_open": report["clock"]["is_open"],
        "execution_blocks": report["execution_blocks"],
        "tickers": [
            {
                "ticker": row["ticker"],
                "decision": (row.get("forecast") or {}).get("decision"),
                "entry": row.get("entry"),
                "blocks": row.get("blocks"),
                "effects": row.get("effects"),
            }
            for row in report["tickers"]
        ],
    }


if __name__ == "__main__":
    main()
