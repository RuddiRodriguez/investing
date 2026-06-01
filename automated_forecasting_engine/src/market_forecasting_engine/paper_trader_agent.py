from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.daily_trade_cli import _annotate_mean_reversion_dip_buy, _run_daily_llm_decision
from market_forecasting_engine.options_decision import OptionsDecisionConfig, build_options_decision
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL, DEFAULT_REASONING_EFFORT
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def main() -> None:
    args = build_parser().parse_args()
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    while True:
        record = run_once(args, state_dir)
        print(json.dumps(record, indent=2, default=str))
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a one-minute Alpaca paper trading agent.")
    parser.add_argument("--ticker", default="ETH-USD", help="Ticker such as ETH-USD or BTC-USD.")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--data-interval", default="1m", help="Alpaca bar interval. 1m is recommended for the agent.")
    parser.add_argument("--initial-lookback-hours", type=float, default=72.0)
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--forecast-refresh-seconds", type=int, default=900, help="Refresh forecast every N seconds; checks still run every minute.")
    parser.add_argument("--forecast-horizon-minutes", type=float, default=15.0)
    parser.add_argument("--minimum-cache-rows", type=int, default=120)
    parser.add_argument("--state-dir", default="automated_forecasting_engine/runs/paper_trader_agent")
    parser.add_argument("--execute-paper-orders", action="store_true", help="Actually submit Alpaca paper orders. Default is decision-only.")
    parser.add_argument("--max-notional", type=float, default=25.0, help="Hard cap per paper order.")
    parser.add_argument("--min-notional", type=float, default=5.0, help="Minimum order notional.")
    parser.add_argument("--entry-order-type", choices=("market", "limit"), default="limit", help="Entry order type for new spot positions.")
    parser.add_argument("--exit-order-type", choices=("market", "limit", "stop", "stop_limit", "trailing_stop"), default="trailing_stop", help="Exit/protective order type for existing spot positions.")
    parser.add_argument("--entry-limit-offset-bps", type=float, default=8.0, help="Limit-entry tolerance from latest price in bps.")
    parser.add_argument("--stop-buffer-bps", type=float, default=8.0, help="Extra stop buffer beyond the model stop/lower bound in bps.")
    parser.add_argument("--stop-limit-offset-bps", type=float, default=5.0, help="Stop-limit price offset from stop price in bps.")
    parser.add_argument("--trailing-stop-percent", type=float, default=None, help="Trailing stop percent. Defaults to forecast/trade-plan derived value.")
    parser.add_argument("--enable-protective-exit", action="store_true", help="Submit protective exit orders for existing filled positions.")
    parser.add_argument("--allow-short", action="store_true", help="Allow sell signals to open short positions when broker supports it.")
    parser.add_argument("--allow-multiple-open-orders", action="store_true", help="Allow new paper orders while other orders are open.")
    parser.add_argument("--max-open-orders", type=int, default=3, help="Maximum open orders allowed when --allow-multiple-open-orders is set.")
    parser.add_argument("--enable-llm-decision", action="store_true", help="Use a cached forecast-time LLM decision as the final trade intent.")
    parser.add_argument("--llm-dry-run", action="store_true", help="Build the LLM packet at forecast refresh but do not call the LLM.")
    parser.add_argument("--llm-model", default=None, help=f"Defaults to OPENAI_MODEL or {DEFAULT_OPENAI_MODEL}.")
    parser.add_argument("--llm-reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--llm-timeout", type=int, default=120)
    parser.add_argument("--llm-env-file", default=None)
    parser.add_argument("--llm-no-web-search", action="store_true")
    parser.add_argument("--llm-search-context-size", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--trader-name", default="paper_intraday_trader")
    parser.add_argument("--portfolio-notes", default="")
    parser.add_argument("--quiet-progress", action="store_true", help="Disable step-by-step terminal progress messages.")
    parser.add_argument("--once", action="store_true")
    return parser


def run_once(args: argparse.Namespace, state_dir: Path) -> dict[str, Any]:
    _progress(args, "START", "agent check started", ticker=args.ticker, profile=args.risk_profile, state_dir=str(state_dir))
    profile = risk_profile_for_name(args.risk_profile)
    symbol = _alpaca_symbol(args.ticker)
    _progress(args, "DATA", "updating Alpaca price cache", symbol=symbol, interval=args.data_interval)
    prices = update_cached_prices(args, state_dir)
    latest_price = float(prices["close"].iloc[-1])
    _progress(args, "DATA", "price cache updated", rows=len(prices), latest_price=round(latest_price, 6), latest_timestamp=str(prices.index[-1]))
    state = _read_state(state_dir, args.ticker, args.risk_profile)
    forecast_record = state.get("last_forecast")
    now = datetime.now(UTC)
    _progress(args, "BROKER", "reading Alpaca paper account, position, and open orders", symbol=symbol)
    broker_state = read_broker_state(symbol)
    _progress(
        args,
        "BROKER",
        "broker state loaded",
        position_symbol=(broker_state.get("position") or {}).get("symbol"),
        position_qty=(broker_state.get("position") or {}).get("qty"),
        open_orders=len(broker_state.get("open_orders", []) or []),
    )
    stale = _forecast_is_stale(forecast_record, now, args.forecast_refresh_seconds)
    if stale:
        _progress(args, "FORECAST", "forecast refresh required", reason=_forecast_cache_status(forecast_record, now, args.forecast_refresh_seconds))
        forecast_record = build_forecast_decision(args, prices, broker_state=broker_state)
        state["last_forecast"] = forecast_record
        _progress(
            args,
            "FORECAST",
            "forecast refresh finished",
            forecast_price=round(float(forecast_record.get("forecast", {}).get("predicted_price") or 0.0), 6),
            forecast_time=forecast_record.get("forecast", {}).get("forecast_date"),
            llm_status=(forecast_record.get("llm_trader") or {}).get("status"),
        )
    else:
        _progress(args, "FORECAST", "reusing cached forecast", reason=_forecast_cache_status(forecast_record, now, args.forecast_refresh_seconds))

    _progress(args, "DECISION", "building order decision", decision_source="llm_forecast_cache" if args.enable_llm_decision else "deterministic_forecast")
    decision = decide_order(
        args=args,
        profile_name=args.risk_profile,
        profile_budget=profile.risk_budget_pct,
        symbol=symbol,
        latest_price=latest_price,
        forecast_record=forecast_record,
        broker_state=broker_state,
    )
    _progress(
        args,
        "DECISION",
        "order decision built",
        action=decision.get("action"),
        side=decision.get("side"),
        reasons=",".join(decision.get("reasons") or []) or "none",
        llm_status=decision.get("llm_status"),
    )
    _progress(args, "ORDER", "evaluating order submission", execute_paper_orders=bool(args.execute_paper_orders))
    order_result = maybe_submit_order(args, symbol, decision)
    _progress(args, "ORDER", "order step finished", submitted=order_result.get("submitted"), reason=order_result.get("reason"))
    record = {
        "checked_at": now.isoformat(),
        "ticker": args.ticker.upper(),
        "alpaca_symbol": symbol,
        "risk_profile": profile.to_dict(),
        "latest_price": latest_price,
        "cache_rows": int(len(prices)),
        "forecast": forecast_record,
        "broker_state": _broker_state_for_log(broker_state),
        "decision": decision,
        "order_result": order_result,
    }
    append_log(state_dir, args.ticker, args.risk_profile, record)
    _progress(args, "LOG", "decision log written", log_file=str(_log_path(state_dir, args.ticker, args.risk_profile)))
    _write_state(state_dir, args.ticker, args.risk_profile, state)
    _progress(args, "DONE", "agent check finished")
    return record


def update_cached_prices(args: argparse.Namespace, state_dir: Path) -> pd.DataFrame:
    cache_path = _cache_path(state_dir, args.ticker, args.data_interval)
    existing = _read_cache(cache_path)
    if existing.empty:
        start = (datetime.now(UTC) - pd.Timedelta(hours=float(args.initial_lookback_hours))).isoformat().replace("+00:00", "Z")
    else:
        last_timestamp = pd.Timestamp(existing.index[-1])
        start = (last_timestamp - pd.Timedelta(minutes=2)).isoformat()
    result = load_prices_with_provider(
        "alpaca",
        DataRequest(ticker=args.ticker, start=start, interval=args.data_interval, target_column="close"),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    fetched = normalize_price_frame(result.frame, target_column="close")
    combined = pd.concat([existing, fetched]).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.tail(max(int(args.minimum_cache_rows) * 10, 5000))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path, index_label="timestamp")
    return combined


def build_forecast_decision(args: argparse.Namespace, prices: pd.DataFrame, broker_state: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = risk_profile_for_name(args.risk_profile)
    horizon_hours = float(args.forecast_horizon_minutes) / 60.0
    _progress(args, "FORECAST", "building intraday forecast", horizon_minutes=args.forecast_horizon_minutes, rows=len(prices))
    plan = build_daily_trade_plan(
        prices,
        DailyTradeConfig(
            ticker=args.ticker,
            interval=args.data_interval,
            forecast_hours=(horizon_hours,),
            minimum_score_to_trade=1.5 if args.risk_profile == "aggressive" else 2.0,
        ),
    )
    forecast = plan["forecasts"][0]
    forecast_timestamp = (pd.Timestamp(plan["as_of"]) + pd.Timedelta(minutes=float(args.forecast_horizon_minutes))).isoformat()
    forecast = {**forecast, "forecast_timestamp": forecast_timestamp}
    report = {
        "ticker": args.ticker.upper(),
        "as_of_timestamp": plan["as_of"],
        "current_price": plan["latest_price"],
        "forecasts": [
            {
                "horizon_hours": horizon_hours,
                "forecast_date": forecast["forecast_timestamp"],
                "predicted_price": forecast["predicted_price"],
                "lower_price": forecast["lower_price"],
                "upper_price": forecast["upper_price"],
                "expected_return": forecast["expected_return"],
                "expected_direction": _direction_from_return(float(forecast["expected_return"])),
                "directional_confidence": _confidence_from_interval(forecast),
                "spot": plan["latest_price"],
                "validation_metrics": {"mae": _interval_mae_proxy(forecast, plan["latest_price"])},
            }
        ],
    }
    _progress(args, "FORECAST", "scoring synthetic options context")
    options = build_options_decision(
        report,
        prices,
        config=OptionsDecisionConfig(
            ticker=args.ticker,
            risk_profile=args.risk_profile,
            min_edge_pct=profile.options_min_edge_pct,
            max_spread_pct=profile.options_max_spread_pct,
            min_probability_above_breakeven=profile.options_min_probability_breakeven,
        ),
    )
    report["options_decision"] = options
    _progress(args, "FORECAST", "evaluating conditional dip-buy context")
    _annotate_mean_reversion_dip_buy(report, prices, "close", risk_profile_name=args.risk_profile)
    derivative_intent = _derivative_intent(report["forecasts"][0], profile.minimum_edge_fraction)
    llm_trader = None
    llm_derivative_intent = None
    if getattr(args, "enable_llm_decision", False):
        llm_args = _forecast_llm_args(args, broker_state or {})
        _progress(
            args,
            "LLM",
            "running forecast-time LLM decision" if not getattr(args, "llm_dry_run", False) else "building LLM decision packet only",
            dry_run=bool(getattr(args, "llm_dry_run", False)),
            model=args.llm_model or DEFAULT_OPENAI_MODEL,
        )
        llm_trader = _run_daily_llm_decision(report, llm_args, dry_run=bool(getattr(args, "llm_dry_run", False)))
        llm_derivative_intent = _intent_from_llm_decision(llm_trader, report["forecasts"][0])
        _progress(
            args,
            "LLM",
            "forecast-time LLM decision finished",
            status=llm_trader.get("status"),
            decision=(llm_trader.get("decision") or {}).get("decision"),
            intent=llm_derivative_intent.get("action"),
        )
    else:
        _progress(args, "LLM", "LLM decision disabled for this forecast")
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "horizon_minutes": float(args.forecast_horizon_minutes),
        "spot_plan": plan,
        "forecast": report["forecasts"][0],
        "options_decision": {
            "mode": options.get("mode"),
            "best_trade": options.get("best_trade"),
            "top_candidates": options.get("top_candidates", [])[:3],
        },
        "mean_reversion_dip_buy": report.get("decision_view", {}).get("mean_reversion_dip_buy", {}),
        "llm_trader": llm_trader,
        "llm_derivative_intent": llm_derivative_intent,
        "derivative_intent": derivative_intent,
    }


def _forecast_llm_args(args: argparse.Namespace, broker_state: dict[str, Any]) -> argparse.Namespace:
    position = broker_state.get("position") or {}
    holding_status = "owned" if position else "not_owned"
    return SimpleNamespace(
        ticker=args.ticker,
        risk_profile=args.risk_profile,
        trader_name=args.trader_name,
        holding_status=holding_status,
        entry_price=_float_or_none(position.get("avg_entry_price")),
        quantity=_float_or_none(position.get("qty")),
        position_value=_float_or_none(position.get("market_value")),
        account_equity=_float_or_none((broker_state.get("account") or {}).get("equity")),
        portfolio_notes=args.portfolio_notes,
        llm_model=args.llm_model,
        llm_reasoning_effort=args.llm_reasoning_effort,
        llm_timeout=args.llm_timeout,
        llm_env_file=args.llm_env_file,
        llm_no_web_search=args.llm_no_web_search,
        llm_search_context_size=args.llm_search_context_size,
    )


def _intent_from_llm_decision(llm_trader: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any]:
    if llm_trader.get("status") != "executed":
        return {
            "action": "no_trade",
            "reason": f"llm_decision_unavailable:{llm_trader.get('status')}",
            "expected_return": forecast.get("expected_return"),
        }
    decision = str((llm_trader.get("decision") or {}).get("decision") or "").strip().lower()
    if decision == "buy":
        return {"action": "buy_call", "reason": "llm_final_decision_buy", "expected_return": forecast.get("expected_return")}
    if decision == "sell":
        return {"action": "buy_put", "reason": "llm_final_decision_sell", "expected_return": forecast.get("expected_return")}
    return {"action": "no_trade", "reason": "llm_final_decision_hold", "expected_return": forecast.get("expected_return")}


def read_broker_state(symbol: str) -> dict[str, Any]:
    broker = AlpacaPaperBroker()
    account = broker.account()
    position = broker.position(symbol)
    open_orders = broker.orders(status="open", limit=20)
    return {"account": account, "position": position, "open_orders": open_orders}


def decide_order(
    *,
    args: argparse.Namespace,
    profile_name: str,
    profile_budget: float,
    symbol: str,
    latest_price: float,
    forecast_record: dict[str, Any],
    broker_state: dict[str, Any],
) -> dict[str, Any]:
    forecast = forecast_record.get("forecast", {})
    if getattr(args, "enable_llm_decision", False):
        intent = forecast_record.get("llm_derivative_intent") or {"action": "no_trade", "reason": "missing_cached_llm_decision"}
    else:
        intent = forecast_record.get("derivative_intent", {})
    account = broker_state.get("account", {}) or {}
    position = broker_state.get("position")
    open_orders = broker_state.get("open_orders", []) or []
    equity = float(account.get("equity") or account.get("cash") or 0.0)
    notional = min(float(args.max_notional), max(float(args.min_notional), equity * float(profile_budget)))
    reasons = []
    if open_orders and not args.allow_multiple_open_orders:
        reasons.append("open_order_exists")
    if args.allow_multiple_open_orders and len(open_orders) >= int(args.max_open_orders):
        reasons.append("max_open_orders_reached")
    if len(forecast_record.get("spot_plan", {}).get("forecasts", [])) == 0:
        reasons.append("missing_forecast")
    if intent.get("action") == "buy_call":
        side = "buy"
        if position is not None:
            reasons.append("already_has_position")
    elif intent.get("action") == "buy_put":
        if position is not None:
            side = "sell"
            notional = min(notional, abs(float(position.get("market_value") or 0.0)))
        elif args.allow_short:
            side = "sell"
        else:
            side = "none"
            reasons.append("put_signal_without_short_or_position")
    else:
        side = "none"
        reasons.append("no_directional_edge")
    if notional < float(args.min_notional):
        reasons.append("notional_below_minimum")
    order_plan = build_order_plan(
        args=args,
        symbol=symbol,
        side=side,
        latest_price=latest_price,
        notional=notional,
        forecast_record=forecast_record,
        position=position,
        open_orders=open_orders,
    )
    reasons.extend(_open_order_conflict_reasons(symbol=symbol, side=side, open_orders=open_orders))
    action = "submit_order" if side in {"buy", "sell"} and not reasons else "hold"
    return {
        "action": action,
        "symbol": symbol,
        "side": side,
        "notional": round(notional, 2),
        "latest_price": latest_price,
        "profile": profile_name,
        "derivative_intent": intent,
        "decision_source": "llm_forecast_cache" if getattr(args, "enable_llm_decision", False) else "deterministic_forecast",
        "llm_decision": (forecast_record.get("llm_trader") or {}).get("decision"),
        "llm_status": (forecast_record.get("llm_trader") or {}).get("status"),
        "forecast_price": forecast.get("predicted_price"),
        "forecast_time": forecast.get("forecast_date"),
        "reasons": reasons,
        "execute_paper_orders": bool(args.execute_paper_orders),
        "open_order_count": len(open_orders),
        "allow_multiple_open_orders": bool(args.allow_multiple_open_orders),
        "max_open_orders": int(args.max_open_orders),
        "order_plan": order_plan,
    }


def maybe_submit_order(args: argparse.Namespace, symbol: str, decision: dict[str, Any]) -> dict[str, Any]:
    if decision["action"] != "submit_order":
        return {"submitted": False, "reason": "decision_hold"}
    if not args.execute_paper_orders:
        return {"submitted": False, "reason": "execution_disabled"}
    broker = AlpacaPaperBroker()
    order_id = f"agent-{symbol.replace('/', '')}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    entry_plan = decision.get("order_plan", {}).get("entry_order", {})
    if entry_plan.get("type") in {None, "none"}:
        return {"submitted": False, "reason": "no_executable_order_plan", "entry_order": entry_plan}
    try:
        order = broker.submit_order(
            symbol=entry_plan.get("symbol", symbol),
            side=entry_plan.get("side", decision["side"]),
            order_type=entry_plan.get("type", "market"),
            notional=entry_plan.get("notional"),
            qty=entry_plan.get("qty"),
            limit_price=entry_plan.get("limit_price"),
            stop_price=entry_plan.get("stop_price"),
            trail_percent=entry_plan.get("trail_percent"),
            trail_price=entry_plan.get("trail_price"),
            client_order_id=order_id,
            time_in_force=entry_plan.get("time_in_force", "gtc"),
        )
    except RuntimeError as exc:
        return {"submitted": False, "reason": "broker_rejected_entry_order", "error": str(exc), "entry_order": entry_plan}
    protective_result = None
    protective_plan = decision.get("order_plan", {}).get("protective_exit_order")
    if protective_plan and protective_plan.get("submit_now"):
        protective_id = f"protect-{symbol.replace('/', '')}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        try:
            protective_result = broker.submit_order(
                symbol=protective_plan.get("symbol", symbol),
                side=protective_plan.get("side", "sell"),
                order_type=protective_plan.get("type", "trailing_stop"),
                qty=protective_plan.get("qty"),
                limit_price=protective_plan.get("limit_price"),
                stop_price=protective_plan.get("stop_price"),
                trail_percent=protective_plan.get("trail_percent"),
                trail_price=protective_plan.get("trail_price"),
                client_order_id=protective_id,
                time_in_force=protective_plan.get("time_in_force", "gtc"),
            )
        except RuntimeError as exc:
            protective_result = {
                "submitted": False,
                "reason": "broker_rejected_protective_order",
                "error": str(exc),
                "protective_order": protective_plan,
            }
    return {"submitted": True, "entry_order": order, "protective_order": protective_result}


def build_order_plan(
    *,
    args: argparse.Namespace,
    symbol: str,
    side: str,
    latest_price: float,
    notional: float,
    forecast_record: dict[str, Any],
    position: dict[str, Any] | None,
    open_orders: list[dict[str, Any]],
) -> dict[str, Any]:
    forecast = forecast_record.get("forecast", {}) or {}
    trade_plan = forecast_record.get("spot_plan", {}).get("trade_plan", {}) or {}
    entry_order = _entry_order_plan(
        args=args,
        symbol=symbol,
        side=side,
        latest_price=latest_price,
        notional=notional,
        position=position,
    )
    exiting_existing_position = side == "sell" and position is not None
    protective_order = None
    if not exiting_existing_position:
        protective_order = _protective_exit_plan(
            args=args,
            symbol=symbol,
            latest_price=latest_price,
            forecast=forecast,
            trade_plan=trade_plan,
            position=position,
            open_orders=open_orders,
        )
    return {
        "entry_order": entry_order,
        "protective_exit_order": protective_order,
        "forecast_levels": {
            "predicted_price": forecast.get("predicted_price"),
            "lower_price": forecast.get("lower_price"),
            "upper_price": forecast.get("upper_price"),
            "trade_plan_stop": trade_plan.get("stop"),
            "trade_plan_take_profit": trade_plan.get("take_profit"),
        },
        "policy": (
            "Entry price and protective levels are derived from the latest price, forecast interval, "
            "and intraday trade plan. Protective exits are submitted immediately only for existing filled positions."
        ),
    }


def _open_order_conflict_reasons(*, symbol: str, side: str, open_orders: list[dict[str, Any]]) -> list[str]:
    if side not in {"buy", "sell"}:
        return []
    opposite = "sell" if side == "buy" else "buy"
    conflicts = [
        order
        for order in open_orders
        if _is_active_order(order)
        and _same_alpaca_symbol(str(order.get("symbol") or ""), symbol)
        and str(order.get("side") or "").lower() == opposite
    ]
    if not conflicts:
        return []
    return [f"opposite_open_{opposite}_order_exists"]


def _is_active_order(order: dict[str, Any]) -> bool:
    return str(order.get("status") or "").lower() in {"new", "accepted", "pending_new", "partially_filled", "held"}


def _same_alpaca_symbol(left: str, right: str) -> bool:
    return _compact_symbol(left) == _compact_symbol(right)


def _compact_symbol(symbol: str) -> str:
    return symbol.strip().upper().replace("/", "").replace("-", "")


def _entry_order_plan(
    *,
    args: argparse.Namespace,
    symbol: str,
    side: str,
    latest_price: float,
    notional: float,
    position: dict[str, Any] | None,
) -> dict[str, Any]:
    if side not in {"buy", "sell"}:
        return {"type": "none", "reason": "no executable side"}
    requested_order_type = args.exit_order_type if side == "sell" and position is not None else args.entry_order_type
    order_type = _effective_order_type(symbol, requested_order_type)
    plan: dict[str, Any] = {
        "symbol": symbol,
        "side": side,
        "type": order_type,
        "time_in_force": "gtc",
    }
    if side == "sell" and position is not None:
        plan["qty"] = abs(float(position.get("qty") or 0.0))
    else:
        if order_type == "market":
            plan["notional"] = round(float(notional), 2)
        else:
            reference_price = _entry_limit_price(side, latest_price, float(args.entry_limit_offset_bps)) if order_type == "limit" else latest_price
            plan["qty"] = _qty_from_notional(notional, reference_price)
    if order_type == "limit":
        plan["limit_price"] = _entry_limit_price(side, latest_price, float(args.entry_limit_offset_bps))
    elif order_type in {"stop", "stop_limit", "trailing_stop"}:
        plan.update(_exit_order_prices(args=args, order_type=order_type, side=side, latest_price=latest_price, stop_price=None))
    if order_type != requested_order_type:
        plan["requested_type"] = requested_order_type
        plan["order_type_policy"] = f"{requested_order_type} is not supported for this symbol; using {order_type}."
    return plan


def _protective_exit_plan(
    *,
    args: argparse.Namespace,
    symbol: str,
    latest_price: float,
    forecast: dict[str, Any],
    trade_plan: dict[str, Any],
    position: dict[str, Any] | None,
    open_orders: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not args.enable_protective_exit or position is None:
        return None
    qty = abs(float(position.get("qty") or 0.0))
    if qty <= 0:
        return None
    if any(str(order.get("side")) == "sell" and str(order.get("status")) in {"new", "accepted", "pending_new"} for order in open_orders):
        return {"submit_now": False, "reason": "protective_sell_order_already_open"}
    stop_price = _model_stop_price(latest_price=latest_price, forecast=forecast, trade_plan=trade_plan)
    requested_order_type = args.exit_order_type
    order_type = _effective_order_type(symbol, requested_order_type)
    plan = {
        "submit_now": True,
        "symbol": symbol,
        "side": "sell",
        "type": order_type,
        "qty": qty,
        "time_in_force": "gtc",
    }
    plan.update(_exit_order_prices(args=args, order_type=order_type, side="sell", latest_price=latest_price, stop_price=stop_price))
    if order_type != requested_order_type:
        plan["requested_type"] = requested_order_type
        plan["order_type_policy"] = f"{requested_order_type} is not supported for this symbol; using {order_type}."
    return plan


def _effective_order_type(symbol: str, requested_order_type: str) -> str:
    if "/" in symbol and requested_order_type not in {"market", "limit", "stop_limit"}:
        return "stop_limit"
    return requested_order_type


def _entry_limit_price(side: str, latest_price: float, offset_bps: float) -> float:
    offset = float(offset_bps) / 10_000.0
    if side == "buy":
        return round(float(latest_price) * (1.0 + offset), 2)
    return round(float(latest_price) * (1.0 - offset), 2)


def _qty_from_notional(notional: float, price: float) -> float:
    qty = float(notional) / max(float(price), 1e-9)
    return float(f"{qty:.9f}")


def _model_stop_price(*, latest_price: float, forecast: dict[str, Any], trade_plan: dict[str, Any]) -> float:
    candidates = [
        _float_or_none(trade_plan.get("stop")),
        _float_or_none(forecast.get("lower_price")),
        latest_price * 0.995,
    ]
    below = [value for value in candidates if value is not None and value < latest_price]
    return max(below) if below else latest_price * 0.995


def _exit_order_prices(
    *,
    args: argparse.Namespace,
    order_type: str,
    side: str,
    latest_price: float,
    stop_price: float | None,
) -> dict[str, Any]:
    if order_type == "market":
        return {}
    if order_type == "limit":
        return {"limit_price": _entry_limit_price(side, latest_price, float(args.entry_limit_offset_bps))}
    if order_type == "stop":
        return {"stop_price": round(float(stop_price or latest_price), 2)}
    if order_type == "stop_limit":
        stop = float(stop_price or latest_price)
        offset = float(args.stop_limit_offset_bps) / 10_000.0
        limit = stop * (1.0 - offset) if side == "sell" else stop * (1.0 + offset)
        return {"stop_price": round(stop, 2), "limit_price": round(limit, 2)}
    if order_type == "trailing_stop":
        trail_percent = args.trailing_stop_percent
        if trail_percent is None:
            reference_stop = float(stop_price or latest_price * 0.995)
            trail_percent = max(0.1, min(3.0, abs(latest_price - reference_stop) / max(latest_price, 1e-9) * 100.0))
        return {"trail_percent": round(float(trail_percent), 4)}
    return {}


def _float_or_none(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(output):
        return None
    return output


def _progress(args: argparse.Namespace, stage: str, message: str, **fields: Any) -> None:
    if getattr(args, "quiet_progress", False):
        return
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    suffix = ""
    if fields:
        compact = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        suffix = f" | {compact}" if compact else ""
    print(f"[{timestamp}] [{stage}] {message}{suffix}", file=sys.stderr, flush=True)


def append_log(state_dir: Path, ticker: str, profile: str, record: dict[str, Any]) -> None:
    path = _log_path(state_dir, ticker, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _log_path(state_dir: Path, ticker: str, profile: str) -> Path:
    return state_dir / "logs" / f"{_safe(ticker)}_{profile}_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"


def _forecast_is_stale(forecast_record: dict[str, Any] | None, now: datetime, refresh_seconds: int) -> bool:
    if not forecast_record:
        return True
    created = forecast_record.get("created_at_utc")
    if not created:
        return True
    return (now - datetime.fromisoformat(created.replace("Z", "+00:00"))).total_seconds() >= max(1, refresh_seconds)


def _forecast_cache_status(forecast_record: dict[str, Any] | None, now: datetime, refresh_seconds: int) -> str:
    if not forecast_record:
        return "no_cached_forecast"
    created = forecast_record.get("created_at_utc")
    if not created:
        return "cached_forecast_missing_created_at"
    age = (now - datetime.fromisoformat(str(created).replace("Z", "+00:00"))).total_seconds()
    refresh = max(1, int(refresh_seconds))
    if age >= refresh:
        return f"cached_forecast_age_seconds={age:.0f} >= refresh_seconds={refresh}"
    return f"cached_forecast_age_seconds={age:.0f} < refresh_seconds={refresh}"


def _read_cache(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    frame = frame.set_index("timestamp")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    return normalize_price_frame(frame, target_column="close")


def _read_state(state_dir: Path, ticker: str, profile: str) -> dict[str, Any]:
    path = _state_path(state_dir, ticker, profile)
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _write_state(state_dir: Path, ticker: str, profile: str, state: dict[str, Any]) -> None:
    path = _state_path(state_dir, ticker, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _broker_state_for_log(state: dict[str, Any]) -> dict[str, Any]:
    account = state.get("account", {}) or {}
    position = state.get("position")
    return {
        "account": {
            "status": account.get("status"),
            "equity": account.get("equity"),
            "cash": account.get("cash"),
            "buying_power": account.get("buying_power"),
        },
        "position": None
        if position is None
        else {
            "symbol": position.get("symbol"),
            "qty": position.get("qty"),
            "market_value": position.get("market_value"),
            "unrealized_pl": position.get("unrealized_pl"),
        },
        "open_order_count": len(state.get("open_orders", []) or []),
    }


def _derivative_intent(forecast: dict[str, Any], edge_threshold: float) -> dict[str, Any]:
    expected_return = (float(forecast["predicted_price"]) / float(forecast.get("spot", 0.0) or forecast["predicted_price"])) - 1.0
    if "expected_return" in forecast:
        expected_return = float(forecast["expected_return"])
    if expected_return >= edge_threshold:
        return {"action": "buy_call", "reason": "forecast_up_edge", "expected_return": expected_return}
    if expected_return <= -edge_threshold:
        return {"action": "buy_put", "reason": "forecast_down_edge", "expected_return": expected_return}
    return {"action": "no_trade", "reason": "edge_below_threshold", "expected_return": expected_return}


def _direction_from_return(expected_return: float) -> str:
    if expected_return > 0.001:
        return "Upward"
    if expected_return < -0.001:
        return "Downward"
    return "Flat"


def _confidence_from_interval(forecast: dict[str, Any]) -> float:
    predicted = float(forecast["predicted_price"])
    lower = float(forecast["lower_price"])
    upper = float(forecast["upper_price"])
    width = max(upper - lower, 1e-9)
    edge = abs(predicted - (lower + upper) / 2.0)
    return float(min(0.75, max(0.50, 0.50 + edge / width)))


def _interval_mae_proxy(forecast: dict[str, Any], current_price: float) -> float:
    width = abs(float(forecast["upper_price"]) - float(forecast["lower_price"]))
    return max(width / max(current_price, 1e-9) / 4.0, 0.001)


def _alpaca_symbol(ticker: str) -> str:
    value = ticker.upper().strip()
    if "/" in value:
        return value
    if "-" in value:
        base, quote = value.split("-", 1)
        return f"{base}/{quote}"
    if value.endswith("USD"):
        return f"{value[:-3]}/USD"
    return value


def _cache_path(state_dir: Path, ticker: str, interval: str) -> Path:
    return state_dir / "cache" / f"{_safe(ticker)}_{interval}.csv"


def _state_path(state_dir: Path, ticker: str, profile: str) -> Path:
    return state_dir / "state" / f"{_safe(ticker)}_{profile}.json"


def _safe(value: str) -> str:
    return value.upper().replace("/", "_").replace("-", "_").replace(" ", "_")


if __name__ == "__main__":
    main()
