from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.alpaca_options_trader import (
    OptionExecutionConfig,
    build_real_option_trade_plan,
    build_option_exit_plan,
    submit_option_order,
)
from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    while True:
        record = run_once(args)
        report_path = write_record(output_dir, args.ticker, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real Alpaca paper options decision for one underlying.")
    parser.add_argument("--ticker", default="TSLA")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--provider", default="alpaca")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--forecast-hours", default="1,2,4")
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--forecast-refresh-seconds", type=int, default=900)
    parser.add_argument("--max-training-rows", type=int, default=3500)
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=14)
    parser.add_argument("--allow-0dte", action="store_true")
    parser.add_argument("--max-contract-premium", type=float, default=None, help="Optional per-contract debit cap. Default lets sizing use actual premium, risk budget, and max total debit.")
    parser.add_argument("--max-total-debit", type=float, default=1500.0)
    parser.add_argument("--risk-budget-pct", type=float, default=None, help="Equity fraction used to size option debit; defaults from risk profile.")
    parser.add_argument("--max-position-equity-pct", type=float, default=0.02)
    parser.add_argument("--max-spread-pct", type=float, default=None)
    parser.add_argument("--max-contracts", type=int, default=1)
    parser.add_argument("--target-delta", type=float, default=0.35)
    parser.add_argument("--max-delta-distance", type=float, default=0.28)
    parser.add_argument("--min-open-interest", type=int, default=0)
    parser.add_argument("--limit-price-offset-pct", type=float, default=0.03)
    parser.add_argument("--entry-order-policy", choices=("auto", "limit"), default="auto")
    parser.add_argument("--exit-order-policy", choices=("auto", "stop_limit", "trailing_stop", "take_profit"), default="auto")
    parser.add_argument("--stop-loss-pct", type=float, default=0.35)
    parser.add_argument("--take-profit-pct", type=float, default=0.55)
    parser.add_argument("--stop-limit-offset-pct", type=float, default=0.08)
    parser.add_argument("--abandon-entry-after-seconds", type=int, default=300)
    parser.add_argument("--require-market-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-duplicate-contract-order", action="store_true")
    parser.add_argument("--max-open-option-orders", type=int, default=1)
    parser.add_argument("--execute-paper-orders", action="store_true")
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/paper_options_agent")
    parser.add_argument("--once", action="store_true")
    return parser


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    state = read_state(output_dir, args.ticker)
    broker = AlpacaPaperBroker()
    account = broker.account()
    clock = broker._request("GET", "/v2/clock")
    asset = broker._request("GET", f"/v2/assets/{args.ticker.upper()}")
    forecast_bundle = state.get("last_forecast")
    now = datetime.now(UTC)
    if forecast_bundle is None or _forecast_is_stale(forecast_bundle, now, int(args.forecast_refresh_seconds)):
        prices = _load_prices(args)
        plan = build_daily_trade_plan(
            prices,
            DailyTradeConfig(
                ticker=args.ticker.upper(),
                interval=args.interval,
                forecast_hours=tuple(float(value.strip()) for value in args.forecast_hours.split(",") if value.strip()),
                minimum_score_to_trade=1.5 if args.risk_profile == "aggressive" else 2.0,
            ),
        )
        forecast = _primary_forecast(plan)
        forecast["account_equity"] = _float_or_none(account.get("equity"))
        forecast_bundle = {
            "created_at_utc": now.isoformat(),
            "price_rows": len(prices),
            "forecast_plan": plan,
            "selected_forecast": forecast,
        }
        state["last_forecast"] = forecast_bundle
    else:
        plan = forecast_bundle["forecast_plan"]
        forecast = forecast_bundle["selected_forecast"]
        forecast["account_equity"] = _float_or_none(account.get("equity"))
    profile = risk_profile_for_name(args.risk_profile)
    config = OptionExecutionConfig(
        underlying=args.ticker.upper(),
        risk_profile=args.risk_profile,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        allow_0dte=bool(args.allow_0dte),
        max_contract_premium=None if args.max_contract_premium is None else float(args.max_contract_premium),
        max_total_debit=float(args.max_total_debit),
        risk_budget_pct=float(args.risk_budget_pct if args.risk_budget_pct is not None else profile.risk_budget_pct),
        max_position_equity_pct=float(args.max_position_equity_pct),
        max_spread_pct=float(args.max_spread_pct if args.max_spread_pct is not None else profile.options_max_spread_pct),
        max_contracts=int(args.max_contracts),
        target_delta=float(args.target_delta),
        max_delta_distance=float(args.max_delta_distance),
        min_open_interest=int(args.min_open_interest),
        limit_price_offset_pct=float(args.limit_price_offset_pct),
        stop_loss_pct=float(args.stop_loss_pct),
        take_profit_pct=float(args.take_profit_pct),
        stop_limit_offset_pct=float(args.stop_limit_offset_pct),
        abandon_entry_after_seconds=int(args.abandon_entry_after_seconds),
        entry_order_policy=args.entry_order_policy,
        exit_order_policy=args.exit_order_policy,
    )
    trade_plan = build_real_option_trade_plan(
        broker=broker,
        underlying=args.ticker.upper(),
        underlying_price=float(plan["latest_price"]),
        forecast=forecast,
        config=config,
    )
    raw_open_orders = broker.orders(status="open", limit=50)
    open_orders = _open_option_orders(raw_open_orders, args.ticker.upper())
    option_positions = _option_positions(broker.positions(), args.ticker.upper())
    management_actions = manage_existing_option_orders_and_positions(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        state=state,
        now=now,
    )
    execution_blocks = execution_block_reasons(
        args=args,
        clock=clock,
        trade_plan=trade_plan,
        open_orders=open_orders,
    )
    order_result = {"submitted": False, "reason": "execution_disabled" if not args.execute_paper_orders else "execution_blocked", "blocks": execution_blocks}
    if args.execute_paper_orders and trade_plan.get("action") == "buy_option" and not execution_blocks:
        client_order_id = f"opt-{args.ticker.upper()}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        try:
            order_result = {
                "submitted": True,
                "order": submit_option_order(broker, trade_plan["order"], client_order_id=client_order_id),
            }
        except RuntimeError as exc:
            order_result = {"submitted": False, "reason": "broker_rejected_option_order", "error": str(exc)}
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "ticker": args.ticker.upper(),
        "risk_profile": profile.to_dict(),
        "account": {
            key: account.get(key)
            for key in [
                "status",
                "trading_blocked",
                "account_blocked",
                "options_approved_level",
                "options_trading_level",
                "options_buying_power",
                "buying_power",
                "cash",
                "equity",
                "pattern_day_trader",
            ]
        },
        "market_clock": clock,
        "asset": {key: asset.get(key) for key in ["symbol", "status", "tradable", "attributes"]},
        "forecast_created_at_utc": forecast_bundle.get("created_at_utc"),
        "forecast_cache_status": _forecast_cache_status(forecast_bundle, now, int(args.forecast_refresh_seconds)),
        "price_rows": forecast_bundle.get("price_rows"),
        "forecast_plan": plan,
        "selected_forecast": forecast,
        "option_execution_config": config.__dict__,
        "option_trade_plan": trade_plan,
        "open_option_orders": open_orders,
        "option_positions": option_positions,
        "management_actions": management_actions,
        "execution_blocks": execution_blocks,
        "execute_paper_orders": bool(args.execute_paper_orders),
        "order_result": order_result,
    }
    if order_result.get("submitted") and trade_plan.get("action") == "buy_option":
        state["active_trade"] = {
            "entry_order": order_result.get("order"),
            "trade_plan": trade_plan,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "entry_submitted",
        }
    state["last_record_summary"] = _summary(record)
    write_state(output_dir, args.ticker, state)
    append_log(output_dir, args.ticker, record)
    return record


def write_record(output_dir: Path, ticker: str, record: dict[str, Any]) -> Path:
    report_path = output_dir / f"{ticker.upper()}_options_agent_report.json"
    report_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return report_path


def read_state(output_dir: Path, ticker: str) -> dict[str, Any]:
    path = _state_path(output_dir, ticker)
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_state(output_dir: Path, ticker: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def append_log(output_dir: Path, ticker: str, record: dict[str, Any]) -> None:
    path = output_dir / "logs" / f"{ticker.upper()}_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _state_path(output_dir: Path, ticker: str) -> Path:
    return output_dir / "state" / f"{ticker.upper()}_options_agent_state.json"


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    start = (datetime.now(UTC) - timedelta(days=int(args.lookback_days))).isoformat().replace("+00:00", "Z")
    result = load_prices_with_provider(
        args.provider,
        DataRequest(ticker=args.ticker.upper(), start=start, interval=args.interval, target_column="close"),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    prices = normalize_price_frame(result.frame, target_column="close")
    if args.max_training_rows and len(prices) > int(args.max_training_rows):
        prices = prices.tail(int(args.max_training_rows))
    return prices


def execution_block_reasons(
    *,
    args: argparse.Namespace,
    clock: dict[str, Any],
    trade_plan: dict[str, Any],
    open_orders: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    if trade_plan.get("action") != "buy_option":
        reasons.append(str(trade_plan.get("reason") or "no_buy_option_plan"))
    if args.require_market_open and not bool(clock.get("is_open")):
        reasons.append("market_closed")
    if len(open_orders) >= int(args.max_open_option_orders):
        reasons.append("max_open_option_orders_reached")
    planned_symbol = ((trade_plan.get("order") or {}).get("symbol") or "").upper()
    if planned_symbol and not args.allow_duplicate_contract_order:
        for order in open_orders:
            if str(order.get("symbol") or "").upper() == planned_symbol:
                reasons.append("same_contract_order_already_open")
                break
    return list(dict.fromkeys(reasons))


def manage_existing_option_orders_and_positions(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    actions.extend(_abandon_stale_entry_orders(broker=broker, args=args, open_orders=open_orders, state=state, now=now))
    if not args.execute_paper_orders:
        return actions
    active_trade = state.get("active_trade") or {}
    trade_plan = active_trade.get("trade_plan") or {}
    for position in option_positions:
        symbol = str(position.get("symbol") or "")
        qty = _int_float(position.get("qty"))
        if not symbol or qty <= 0:
            continue
        has_exit = any(str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell" for order in open_orders)
        if has_exit:
            continue
        entry_price = _float_or_none(position.get("avg_entry_price")) or _entry_price_from_state(active_trade)
        if entry_price is None:
            continue
        exit_plan = (trade_plan.get("exit_plan") or build_option_exit_plan(
            entry_limit_price=entry_price,
            qty=qty,
            config=OptionExecutionConfig(
                underlying=args.ticker.upper(),
                stop_loss_pct=float(args.stop_loss_pct),
                take_profit_pct=float(args.take_profit_pct),
                stop_limit_offset_pct=float(args.stop_limit_offset_pct),
            ),
        ))
        selected_exit = dict(exit_plan.get("primary_exit") or exit_plan["stop_loss"])
        try:
            order = broker.submit_order(
                symbol=symbol,
                side=selected_exit.get("side", "sell"),
                order_type=selected_exit.get("type", "stop_limit"),
                qty=qty,
                stop_price=_float_or_none(selected_exit.get("stop_price")),
                limit_price=_float_or_none(selected_exit.get("limit_price")),
                trail_percent=_float_or_none(selected_exit.get("trail_percent")),
                trail_price=_float_or_none(selected_exit.get("trail_price")),
                time_in_force=selected_exit.get("time_in_force", "day"),
                client_order_id=f"opt-stop-{args.ticker.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
            )
            actions.append({"action": "submitted_primary_exit", "exit_type": selected_exit.get("type"), "symbol": symbol, "order": order})
        except RuntimeError as exc:
            fallback = dict(exit_plan["stop_loss"])
            if selected_exit.get("type") == "stop_limit":
                actions.append({"action": "primary_exit_rejected", "symbol": symbol, "error": str(exc)})
                continue
            try:
                order = broker.submit_order(
                    symbol=symbol,
                    side="sell",
                    order_type="stop_limit",
                    qty=qty,
                    stop_price=float(fallback["stop_price"]),
                    limit_price=float(fallback["limit_price"]),
                    time_in_force=fallback.get("time_in_force", "day"),
                    client_order_id=f"opt-stop-fallback-{args.ticker.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
                )
                actions.append({
                    "action": "submitted_fallback_stop_limit",
                    "requested_exit_type": selected_exit.get("type"),
                    "symbol": symbol,
                    "order": order,
                    "rejection": str(exc),
                })
            except RuntimeError as fallback_exc:
                actions.append({"action": "primary_and_fallback_exit_rejected", "symbol": symbol, "error": str(exc), "fallback_error": str(fallback_exc)})
    return actions


def _abandon_stale_entry_orders(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    active_trade = state.get("active_trade") or {}
    entry_order = active_trade.get("entry_order") or {}
    entry_id = entry_order.get("id")
    if not entry_id:
        return []
    submitted_at = entry_order.get("submitted_at") or active_trade.get("created_at_utc")
    age = _age_seconds(submitted_at, now)
    if age is None or age < int(args.abandon_entry_after_seconds):
        return []
    matching = [order for order in open_orders if order.get("id") == entry_id]
    if not matching:
        return []
    if not args.execute_paper_orders:
        return [{"action": "would_cancel_stale_entry", "order_id": entry_id, "age_seconds": age}]
    try:
        broker.cancel_order(str(entry_id))
        state["active_trade"] = {**active_trade, "status": "entry_cancelled_stale", "cancelled_at_utc": now.isoformat()}
        return [{"action": "cancelled_stale_entry", "order_id": entry_id, "age_seconds": age}]
    except RuntimeError as exc:
        return [{"action": "cancel_stale_entry_failed", "order_id": entry_id, "age_seconds": age, "error": str(exc)}]


def _open_option_orders(orders: list[dict[str, Any]], underlying: str) -> list[dict[str, Any]]:
    output = []
    for order in orders:
        symbol = str(order.get("symbol") or "").upper()
        if not symbol.startswith(underlying.upper()):
            continue
        if not _looks_like_option_symbol(symbol):
            continue
        output.append(
            {
                "id": order.get("id"),
                "symbol": symbol,
                "side": order.get("side"),
                "type": order.get("type"),
                "qty": order.get("qty"),
                "limit_price": order.get("limit_price"),
                "status": order.get("status"),
                "submitted_at": order.get("submitted_at"),
            }
        )
    return output


def _option_positions(positions: list[dict[str, Any]], underlying: str) -> list[dict[str, Any]]:
    output = []
    for position in positions:
        symbol = str(position.get("symbol") or "").upper()
        if symbol.startswith(underlying.upper()) and _looks_like_option_symbol(symbol):
            output.append(
                {
                    "symbol": symbol,
                    "qty": position.get("qty"),
                    "avg_entry_price": position.get("avg_entry_price"),
                    "market_value": position.get("market_value"),
                    "unrealized_pl": position.get("unrealized_pl"),
                }
            )
    return output


def _looks_like_option_symbol(symbol: str) -> bool:
    return len(symbol) >= 15 and ("C" in symbol[4:] or "P" in symbol[4:])


def _age_seconds(timestamp: Any, now: datetime) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - parsed).total_seconds()


def _entry_price_from_state(active_trade: dict[str, Any]) -> float | None:
    order = ((active_trade.get("trade_plan") or {}).get("order") or {})
    return _float_or_none(order.get("limit_price"))


def _int_float(value: Any) -> int:
    parsed = _float_or_none(value)
    return 0 if parsed is None else int(abs(parsed))


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _forecast_is_stale(forecast_bundle: dict[str, Any], now: datetime, refresh_seconds: int) -> bool:
    created = forecast_bundle.get("created_at_utc")
    if not created:
        return True
    try:
        created_at = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError:
        return True
    return (now - created_at).total_seconds() >= max(1, int(refresh_seconds))


def _forecast_cache_status(forecast_bundle: dict[str, Any], now: datetime, refresh_seconds: int) -> str:
    created = forecast_bundle.get("created_at_utc")
    if not created:
        return "missing"
    try:
        created_at = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
    except ValueError:
        return "invalid_created_at"
    age = (now - created_at).total_seconds()
    return "stale" if age >= max(1, int(refresh_seconds)) else f"cached_age_seconds={age:.0f}"


def _primary_forecast(plan: dict[str, Any]) -> dict[str, Any]:
    forecasts = sorted(plan.get("forecasts", []), key=lambda row: float(row.get("horizon_hours") or 0.0))
    if not forecasts:
        raise RuntimeError("No forecasts were generated.")
    forecast = dict(forecasts[0])
    forecast.setdefault("spot", plan.get("latest_price"))
    forecast.setdefault("forecast_date", forecast.get("forecast_timestamp"))
    expected_return = float(forecast.get("expected_return") or 0.0)
    forecast["expected_direction"] = "Upward" if expected_return > 0 else "Downward" if expected_return < 0 else "Flat"
    return forecast


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    trade_plan = record.get("option_trade_plan") or {}
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    return {
        "market_is_open": (record.get("market_clock") or {}).get("is_open"),
        "forecast_direction": (record.get("selected_forecast") or {}).get("expected_direction"),
        "forecast_price": (record.get("selected_forecast") or {}).get("predicted_price"),
        "action": trade_plan.get("action"),
        "reason": trade_plan.get("reason"),
        "contract": selected.get("symbol"),
        "contract_name": selected.get("name"),
        "limit_price": order.get("limit_price"),
        "estimated_debit": (trade_plan.get("risk") or {}).get("estimated_debit"),
        "order_submitted": (record.get("order_result") or {}).get("submitted"),
    }


if __name__ == "__main__":
    main()
