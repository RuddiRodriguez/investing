from __future__ import annotations

import argparse
import json
import re
import time
from uuid import uuid4
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


OPTION_AGENT_PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "aggressive": {
        "stop_loss_pct": 0.10,
        "take_profit_pct": 0.55,
        "profit_lock_trigger_pct": 0.08,
        "profit_lock_ratio": 0.75,
        "take_profit_position_pl": 50.0,
        "profit_retrace_from_peak_pct": 0.35,
        "max_spread_pct": 0.15,
        "max_theta_edge_ratio": 0.75,
        "max_theta_premium_pct_per_day": 0.35,
        "entry_cooldown_minutes": 1,
        "loss_cooldown_minutes": 2,
        "max_trades_per_day": 50,
        "max_consecutive_losses": 10,
        "max_open_option_positions": 1,
        "max_open_option_contracts": 2,
        "max_open_option_exposure": 2500.0,
        "max_realized_loss_per_day": 300.0,
        "max_position_unrealized_loss": 150.0,
        "one_trade_per_forecast": False,
    },
    "medium": {
        "stop_loss_pct": 0.08,
        "take_profit_pct": 0.40,
        "profit_lock_trigger_pct": 0.12,
        "profit_lock_ratio": 0.60,
        "take_profit_position_pl": 75.0,
        "profit_retrace_from_peak_pct": 0.30,
        "max_spread_pct": 0.10,
        "max_theta_edge_ratio": 0.75,
        "max_theta_premium_pct_per_day": 0.35,
        "entry_cooldown_minutes": 30,
        "loss_cooldown_minutes": 60,
        "max_trades_per_day": 3,
        "max_consecutive_losses": 2,
        "max_open_option_positions": 2,
        "max_open_option_contracts": 2,
        "max_open_option_exposure": 1500.0,
        "max_realized_loss_per_day": 150.0,
        "max_position_unrealized_loss": 100.0,
        "one_trade_per_forecast": True,
    },
    "conservative": {
        "stop_loss_pct": 0.06,
        "take_profit_pct": 0.30,
        "profit_lock_trigger_pct": 0.10,
        "profit_lock_ratio": 0.75,
        "take_profit_position_pl": 50.0,
        "profit_retrace_from_peak_pct": 0.25,
        "max_spread_pct": 0.08,
        "max_theta_edge_ratio": 0.60,
        "max_theta_premium_pct_per_day": 0.25,
        "entry_cooldown_minutes": 60,
        "loss_cooldown_minutes": 120,
        "max_trades_per_day": 2,
        "max_consecutive_losses": 1,
        "max_open_option_positions": 1,
        "max_open_option_contracts": 1,
        "max_open_option_exposure": 750.0,
        "max_realized_loss_per_day": 75.0,
        "max_position_unrealized_loss": 50.0,
        "one_trade_per_forecast": True,
    },
}


def main() -> None:
    args = apply_option_profile_defaults(build_parser().parse_args())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.liquidate_and_stop:
        record = liquidate_and_stop(args)
        report_path = write_record(output_dir, args.ticker, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        return
    if args.stop_agent:
        request = write_stop_request(output_dir, args.ticker, reason="manual_stop_request")
        print(json.dumps({"status": "stop_requested", "ticker": args.ticker.upper(), "request": request}, indent=2, default=str), flush=True)
        return
    if args.clear_stop_request:
        result = clear_trade_pause_state(output_dir, args.ticker)
        print(json.dumps({"status": "trade_pause_cleared", "ticker": args.ticker.upper(), **result}, indent=2, default=str), flush=True)
        return
    while True:
        stop_request = read_stop_request(output_dir, args.ticker)
        if stop_request and not args.once:
            print(json.dumps({"status": "stopped_by_request", "ticker": args.ticker.upper(), "request": stop_request}, indent=2, default=str), flush=True)
            break
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
    parser.add_argument("--forecast-hours", default="0.25,0.5,0.75,1", help="Forecast horizons in hours. Defaults match the ETH-style 15m/30m/45m/1h dashboard path; trading uses the shortest horizon first.")
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
    parser.add_argument("--target-delta", type=float, default=0.45)
    parser.add_argument("--max-delta-distance", type=float, default=0.30)
    parser.add_argument("--require-greeks", action=argparse.BooleanOptionalAction, default=True, help="Require live option delta/gamma/theta/vega before entry.")
    parser.add_argument("--max-theta-edge-ratio", type=float, default=None, help="Profile default: reject contracts when expected theta decay over the forecast horizon is too large versus delta-adjusted forecast edge.")
    parser.add_argument("--max-theta-premium-pct-per-day", type=float, default=None, help="Profile default: reject contracts when daily theta decay is too large versus option premium.")
    parser.add_argument("--min-open-interest", type=int, default=0)
    parser.add_argument("--limit-price-offset-pct", type=float, default=0.03)
    parser.add_argument("--entry-order-policy", choices=("auto", "limit"), default="auto")
    parser.add_argument("--exit-order-policy", choices=("auto", "stop_limit", "trailing_stop", "take_profit"), default="auto")
    parser.add_argument("--stop-loss-pct", type=float, default=None)
    parser.add_argument("--take-profit-pct", type=float, default=None)
    parser.add_argument("--profit-lock-trigger-pct", type=float, default=None, help="When an option is up by this fraction, raise the stop to protect part of the open profit.")
    parser.add_argument("--profit-lock-ratio", type=float, default=None, help="Fraction of the open per-contract gain to protect when raising the stop.")
    parser.add_argument("--take-profit-position-pl", type=float, default=None, help="Profile default: close an individual option position when that position's unrealized dollar P/L reaches this amount.")
    parser.add_argument("--profit-retrace-from-peak-pct", type=float, default=None, help="Profile default: after a position reaches the per-position profit target, close if it gives back this fraction of peak open profit.")
    parser.add_argument("--profit-close-limit-offset-pct", type=float, default=0.03, help="When take-profit is reached, sell with a limit this fraction below the current option mark to improve fill odds without using market orders.")
    parser.add_argument("--disable-profit-taking", action="store_true", help="Disable autonomous take-profit closes for open option positions.")
    parser.add_argument("--stop-limit-offset-pct", type=float, default=0.08)
    parser.add_argument("--max-total-unrealized-loss", type=float, default=None, help="Optional ticker-level dollar loss cutoff. If open option P/L for this ticker is <= -abs(value), close all open option positions for the ticker.")
    parser.add_argument("--total-loss-close-mode", choices=("all", "losing_only"), default="all", help="When max total unrealized loss is breached, close all ticker positions or only positions currently losing.")
    parser.add_argument("--max-total-unrealized-profit", type=float, default=None, help="Optional ticker-level dollar profit target. If open option P/L for this ticker is >= value, close ticker positions.")
    parser.add_argument("--total-profit-close-mode", choices=("all", "winning_only"), default="all", help="When max total unrealized profit is reached, close all ticker positions or only positions currently winning.")
    parser.add_argument("--max-position-unrealized-loss", type=float, default=None, help="Profile default: close an individual option position when its unrealized dollar P/L is <= -abs(value). Use 0 to close any losing position.")
    parser.add_argument("--max-position-unrealized-profit", type=float, default=None, help="Optional explicit per-position dollar profit target. Use the profile take-profit-position-pl by default; set this only to override with a hard P/L guard.")
    parser.add_argument("--enable-forecast-reversal-exit", action=argparse.BooleanOptionalAction, default=True, help="Close calls when the refreshed forecast turns bearish, and close puts when it turns bullish.")
    parser.add_argument("--min-reversal-edge-pct", type=float, default=0.001, help="Minimum underlying forecast move required before closing a position because the forecast reversed.")
    parser.add_argument("--close-before-expiry-hours", type=float, default=12.0, help="Close open option positions this many hours before option expiry.")
    parser.add_argument("--expiry-warning-hours", type=float, default=24.0, help="Write a visible warning when open option positions are this close to expiry.")
    parser.add_argument("--max-open-option-positions", type=int, default=None, help="Profile default: maximum same-ticker option positions allowed before blocking new entries.")
    parser.add_argument("--max-open-option-contracts", type=float, default=None, help="Profile default: maximum same-ticker open option contracts allowed before blocking new entries.")
    parser.add_argument("--max-open-option-exposure", type=float, default=None, help="Profile default: maximum same-ticker open option debit exposure before blocking new entries.")
    parser.add_argument("--max-realized-loss-per-day", type=float, default=None, help="Profile default: block new entries after same-ticker realized option losses reach this dollar amount for the UTC day.")
    parser.add_argument("--entry-cooldown-minutes", type=float, default=None, help="Profile default: wait this many minutes after any filled option order before opening a new entry.")
    parser.add_argument("--loss-cooldown-minutes", type=float, default=None, help="Profile default: wait this many minutes after a realized losing trade before opening a new entry.")
    parser.add_argument("--max-trades-per-day", type=int, default=None, help="Profile default: maximum filled buy entries per ticker per UTC day.")
    parser.add_argument("--max-consecutive-losses", type=int, default=None, help="Profile default: block new entries after this many consecutive realized losing trades.")
    parser.add_argument("--one-trade-per-forecast", action=argparse.BooleanOptionalAction, default=None, help="Profile default: allow only one entry per forecast refresh.")
    parser.add_argument("--liquidate-and-stop", action="store_true", help="Emergency command: request the running ticker agent to stop, cancel open ticker option orders, and close all open ticker option positions.")
    parser.add_argument("--stop-agent", action="store_true", help="Request the running ticker agent to stop without submitting liquidation orders.")
    parser.add_argument("--clear-stop-request", action="store_true", help="Clear a prior ticker stop request so the agent can run again.")
    parser.add_argument("--liquidation-limit-offset-pct", type=float, default=0.05, help="Emergency liquidation sell limit below current option mark. Example 0.05 means current mark minus 5%%.")
    parser.add_argument("--liquidation-retry-limit-offset-pct", type=float, default=0.15, help="If emergency liquidation is not flat after the first attempt, retry with a wider sell limit below current mark.")
    parser.add_argument("--liquidation-wait-seconds", type=float, default=30.0, help="After emergency liquidation, wait this many seconds while checking whether ticker option positions are flat.")
    parser.add_argument("--liquidation-poll-seconds", type=float, default=2.0, help="Polling interval while verifying emergency liquidation.")
    parser.add_argument("--abandon-entry-after-seconds", type=int, default=300)
    parser.add_argument("--require-market-open", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-duplicate-contract-order", action="store_true")
    parser.add_argument("--allow-mixed-option-direction", action="store_true", help="Allow opening calls while same-ticker puts are open, or puts while same-ticker calls are open.")
    parser.add_argument("--max-open-option-orders", type=int, default=1)
    parser.add_argument("--execute-paper-orders", action="store_true")
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/paper_options_agent")
    parser.add_argument("--once", action="store_true")
    return parser


def apply_option_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = OPTION_AGENT_PROFILE_DEFAULTS[str(args.risk_profile)]
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    ticker = str(getattr(args, "ticker", ""))
    if getattr(args, "max_total_unrealized_profit", None) is None:
        args.max_total_unrealized_profit = _default_total_profit_target(ticker, str(args.risk_profile))
    if getattr(args, "max_total_unrealized_loss", None) is None:
        args.max_total_unrealized_loss = _default_total_loss_cutoff(ticker, str(args.risk_profile))
    return args


def _default_total_profit_target(ticker: str, risk_profile: str) -> float:
    base_by_profile = {
        "aggressive": 150.0,
        "medium": 100.0,
        "conservative": 60.0,
    }
    ticker_adjustment = {
        "SPY": 0.75,
        "QQQ": 0.85,
        "NVDA": 1.00,
        "TSLA": 1.00,
    }
    base = base_by_profile.get(risk_profile, 100.0)
    multiplier = ticker_adjustment.get(ticker.upper(), 0.85)
    return round(base * multiplier, 2)


def _default_total_loss_cutoff(ticker: str, risk_profile: str) -> float:
    base_by_profile = {
        "aggressive": 225.0,
        "medium": 125.0,
        "conservative": 60.0,
    }
    ticker_adjustment = {
        "SPY": 0.75,
        "QQQ": 0.85,
        "NVDA": 1.00,
        "TSLA": 1.00,
    }
    base = base_by_profile.get(risk_profile, 125.0)
    multiplier = ticker_adjustment.get(ticker.upper(), 0.85)
    return round(base * multiplier, 2)


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
        require_greeks=bool(args.require_greeks),
        max_theta_edge_ratio=float(args.max_theta_edge_ratio),
        max_theta_premium_pct_per_day=float(args.max_theta_premium_pct_per_day),
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
    closed_orders = _closed_option_orders(broker.orders(status="closed", limit=500), args.ticker.upper())
    management_actions = manage_existing_option_orders_and_positions(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        state=state,
        now=now,
        forecast=forecast,
        underlying_price=float(plan["latest_price"]),
    )
    if read_stop_request(output_dir, args.ticker):
        management_actions.append({"action": "entry_blocked_by_stop_request", "ticker": args.ticker.upper()})
        trade_plan = {"action": "hold", "reason": "ticker_stop_requested"}
    if _protective_close_triggered(management_actions):
        forecast_key = _forecast_key(forecast_bundle)
        state["wait_until_next_forecast_after_close"] = {
            "ticker": args.ticker.upper(),
            "forecast_created_at_utc": forecast_bundle.get("created_at_utc"),
            "forecast_key": forecast_key,
            "set_at_utc": now.isoformat(),
            "reason": "protective_or_profit_close_submitted",
        }
    execution_blocks = execution_block_reasons(
        args=args,
        clock=clock,
        trade_plan=trade_plan,
        open_orders=open_orders,
        option_positions=option_positions,
    )
    entry_guard = option_entry_guard_reasons(
        args=args,
        closed_orders=closed_orders,
        state=state,
        forecast_bundle=forecast_bundle,
        now=now,
    )
    execution_blocks = list(dict.fromkeys([*execution_blocks, *entry_guard["reasons"]]))
    order_result = {"submitted": False, "reason": "execution_disabled" if not args.execute_paper_orders else "execution_blocked", "blocks": execution_blocks}
    if args.execute_paper_orders and trade_plan.get("action") == "buy_option" and not execution_blocks:
        client_order_id = _client_order_id("opt-entry", args.ticker.upper(), str((trade_plan.get("order") or {}).get("symbol") or ""), datetime.now(UTC))
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
        "option_agent_controls": option_agent_controls(args),
        "entry_guard": entry_guard,
        "option_trade_plan": trade_plan,
        "open_option_orders": open_orders,
        "option_positions": option_positions,
        "management_actions": management_actions,
        "execution_blocks": execution_blocks,
        "execute_paper_orders": bool(args.execute_paper_orders),
        "order_result": order_result,
    }
    if order_result.get("submitted") and trade_plan.get("action") == "buy_option":
        forecast_key = _forecast_key(forecast_bundle)
        state["active_trade"] = {
            "entry_order": order_result.get("order"),
            "trade_plan": trade_plan,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "forecast_key": forecast_key,
            "status": "entry_submitted",
        }
        if bool(args.one_trade_per_forecast) and forecast_key:
            traded = state.setdefault("traded_forecast_keys", [])
            if forecast_key not in traded:
                traded.append(forecast_key)
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


def liquidate_and_stop(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    stop_request = write_stop_request(output_dir, args.ticker, reason="liquidate_and_stop")
    broker = AlpacaPaperBroker()
    now = datetime.now(UTC)
    account = broker.account()
    clock = broker._request("GET", "/v2/clock")
    raw_open_orders = broker.orders(status="open", limit=100)
    open_orders = _open_option_orders(raw_open_orders, args.ticker.upper())
    option_positions = _option_positions(broker.positions(), args.ticker.upper())
    actions = _cancel_open_option_orders(
        broker=broker,
        args=args,
        open_orders=open_orders,
        cancel_action="cancelled_open_order_for_liquidate_and_stop",
    )
    actions.extend(_liquidate_option_positions(
        broker=broker,
        args=args,
        open_orders=[],
        option_positions=option_positions,
        now=now,
    ))
    verification = verify_liquidation_until_flat(
        broker=broker,
        args=args,
        started_at=now,
    )
    actions.extend(verification["actions"])
    final_open_orders = verification["open_orders"]
    final_option_positions = verification["option_positions"]
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "ticker": args.ticker.upper(),
        "command": "liquidate_and_stop",
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
        "stop_request": stop_request,
        "open_option_orders": open_orders,
        "option_positions": option_positions,
        "final_open_option_orders": final_open_orders,
        "final_option_positions": final_option_positions,
        "liquidation_verification": verification,
        "management_actions": actions,
        "execution_blocks": [],
        "execute_paper_orders": bool(args.execute_paper_orders),
        "order_result": {
            "submitted": bool(actions),
            "reason": "liquidate_and_stop",
            "flat": verification["flat"],
            "remaining_position_count": len(final_option_positions),
            "remaining_open_order_count": len(final_open_orders),
        },
    }
    append_log(output_dir, args.ticker, record)
    return record


def write_stop_request(output_dir: Path, ticker: str, *, reason: str) -> dict[str, Any]:
    payload = {
        "ticker": ticker.upper(),
        "reason": reason,
        "requested_at": datetime.now(UTC).isoformat(),
    }
    path = _stop_request_path(output_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {**payload, "path": str(path)}


def read_stop_request(output_dir: Path, ticker: str) -> dict[str, Any] | None:
    path = _stop_request_path(output_dir, ticker)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"ticker": ticker.upper(), "reason": "invalid_stop_request", "path": str(path)}
    return {**parsed, "path": str(path)} if isinstance(parsed, dict) else {"ticker": ticker.upper(), "reason": "invalid_stop_request", "path": str(path)}


def clear_trade_pause_state(output_dir: Path, ticker: str) -> dict[str, Any]:
    cleared_stop_request = clear_stop_request(output_dir, ticker)
    state = read_state(output_dir, ticker)
    removed_keys = []
    for key in ("wait_until_next_forecast_after_close", "traded_forecast_keys", "active_trade"):
        if key in state:
            state.pop(key, None)
            removed_keys.append(key)
    state["manual_trade_resume_at_utc"] = datetime.now(UTC).isoformat()
    write_state(output_dir, ticker, state)
    return {
        "cleared_stop_request": cleared_stop_request,
        "cleared_state_keys": removed_keys,
        "manual_trade_resume_at_utc": state["manual_trade_resume_at_utc"],
        "note": "Entry and loss cooldowns from broker fills before this timestamp are ignored; max trades per day is still enforced.",
    }


def clear_stop_request(output_dir: Path, ticker: str) -> bool:
    path = _stop_request_path(output_dir, ticker)
    if not path.exists():
        return False
    path.unlink()
    return True


def _state_path(output_dir: Path, ticker: str) -> Path:
    return output_dir / "state" / f"{ticker.upper()}_options_agent_state.json"


def _stop_request_path(output_dir: Path, ticker: str) -> Path:
    return output_dir / "control" / f"{ticker.upper()}_stop.json"


def _client_order_id(prefix: str, ticker: str, symbol: str | None, now: datetime) -> str:
    symbol_part = _client_symbol_part(symbol or ticker)
    return f"{prefix}-{ticker.upper()}-{symbol_part}-{now.strftime('%Y%m%d%H%M%S')}-{uuid4().hex[:6]}"


def _client_symbol_part(symbol: str) -> str:
    cleaned = "".join(char for char in str(symbol).upper() if char.isalnum())
    if not cleaned:
        return "ORDER"
    return cleaned[-18:]


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
    option_positions: list[dict[str, Any]],
) -> list[str]:
    reasons = []
    if trade_plan.get("action") != "buy_option":
        reasons.append(str(trade_plan.get("reason") or "no_buy_option_plan"))
    if args.require_market_open and not bool(clock.get("is_open")):
        reasons.append("market_closed")
    if len(open_orders) >= int(args.max_open_option_orders):
        reasons.append("max_open_option_orders_reached")
    open_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    if len(open_positions) >= int(args.max_open_option_positions):
        reasons.append("max_open_option_positions_reached")
    max_contracts = _float_or_none(getattr(args, "max_open_option_contracts", None))
    if max_contracts is not None and max_contracts > 0:
        current_contracts = sum(_int_float(position.get("qty")) for position in open_positions)
        planned_contracts = _int_float((trade_plan.get("order") or {}).get("qty"))
        if current_contracts >= max_contracts or current_contracts + planned_contracts > max_contracts:
            reasons.append("max_open_option_contracts_reached")
    current_exposure = _option_position_exposure(open_positions)
    planned_debit = _planned_option_debit(trade_plan)
    max_exposure = _float_or_none(getattr(args, "max_open_option_exposure", None))
    if max_exposure is not None and max_exposure > 0 and current_exposure + planned_debit > max_exposure:
        reasons.append("max_open_option_exposure_reached")
    planned_symbol = ((trade_plan.get("order") or {}).get("symbol") or "").upper()
    planned_option_type = _option_type_from_symbol(planned_symbol)
    if planned_option_type and not bool(getattr(args, "allow_mixed_option_direction", False)):
        existing_types = {
            option_type
            for option_type in (_option_type_from_symbol(str(position.get("symbol") or "")) for position in open_positions)
            if option_type
        }
        if existing_types and planned_option_type not in existing_types:
            reasons.append("mixed_option_direction_blocked")
    if planned_symbol and not args.allow_duplicate_contract_order:
        for order in open_orders:
            if str(order.get("symbol") or "").upper() == planned_symbol:
                reasons.append("same_contract_order_already_open")
                break
    return list(dict.fromkeys(reasons))


def option_agent_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "risk_profile": args.risk_profile,
        "stop_loss_pct": float(args.stop_loss_pct),
        "take_profit_pct": float(args.take_profit_pct),
        "profit_lock_trigger_pct": float(args.profit_lock_trigger_pct),
        "profit_lock_ratio": float(args.profit_lock_ratio),
        "take_profit_position_pl": float(args.take_profit_position_pl),
        "profit_retrace_from_peak_pct": float(args.profit_retrace_from_peak_pct),
        "max_spread_pct": args.max_spread_pct,
        "entry_cooldown_minutes": float(args.entry_cooldown_minutes),
        "loss_cooldown_minutes": float(args.loss_cooldown_minutes),
        "max_trades_per_day": int(args.max_trades_per_day),
        "max_consecutive_losses": int(args.max_consecutive_losses),
        "one_trade_per_forecast": bool(args.one_trade_per_forecast),
        "max_total_unrealized_loss": getattr(args, "max_total_unrealized_loss", None),
        "total_loss_close_mode": getattr(args, "total_loss_close_mode", None),
        "max_total_unrealized_profit": getattr(args, "max_total_unrealized_profit", None),
        "total_profit_close_mode": getattr(args, "total_profit_close_mode", None),
        "max_position_unrealized_loss": getattr(args, "max_position_unrealized_loss", None),
        "max_position_unrealized_profit": getattr(args, "max_position_unrealized_profit", None),
        "enable_forecast_reversal_exit": getattr(args, "enable_forecast_reversal_exit", None),
        "min_reversal_edge_pct": getattr(args, "min_reversal_edge_pct", None),
        "close_before_expiry_hours": getattr(args, "close_before_expiry_hours", None),
        "expiry_warning_hours": getattr(args, "expiry_warning_hours", None),
        "max_open_option_positions": int(args.max_open_option_positions),
        "max_open_option_contracts": float(args.max_open_option_contracts),
        "max_open_option_exposure": float(args.max_open_option_exposure),
        "max_realized_loss_per_day": float(args.max_realized_loss_per_day),
        "allow_mixed_option_direction": bool(getattr(args, "allow_mixed_option_direction", False)),
    }


def _option_position_exposure(option_positions: list[dict[str, Any]]) -> float:
    exposure = 0.0
    for position in option_positions:
        qty = _int_float(position.get("qty"))
        if qty <= 0:
            continue
        market_value = _float_or_none(position.get("market_value"))
        if market_value is not None:
            exposure += abs(market_value)
            continue
        avg_entry = _float_or_none(position.get("avg_entry_price"))
        if avg_entry is not None:
            exposure += abs(avg_entry * qty * 100.0)
    return round(exposure, 2)


def _planned_option_debit(trade_plan: dict[str, Any]) -> float:
    order = trade_plan.get("order") or {}
    qty = _int_float(order.get("qty"))
    if qty <= 0:
        return 0.0
    limit_price = _float_or_none(order.get("limit_price"))
    if limit_price is not None:
        return round(abs(limit_price * qty * 100.0), 2)
    estimated = _float_or_none(trade_plan.get("estimated_total_debit") or trade_plan.get("max_debit"))
    return round(abs(estimated), 2) if estimated is not None else 0.0


def option_entry_guard_reasons(
    *,
    args: argparse.Namespace,
    closed_orders: list[dict[str, Any]],
    state: dict[str, Any],
    forecast_bundle: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    metrics = option_entry_guard_metrics(closed_orders=closed_orders, now=now)
    reasons: list[str] = []
    resume_at = _parse_timestamp((state.get("manual_trade_resume_at_utc") if isinstance(state, dict) else None))
    last_order_age = _age_seconds(metrics.get("last_filled_order_at"), now)
    last_filled_at = _parse_timestamp(metrics.get("last_filled_order_at"))
    cooldown_overridden = bool(resume_at and last_filled_at and last_filled_at <= resume_at)
    if not cooldown_overridden and last_order_age is not None and last_order_age < float(args.entry_cooldown_minutes) * 60.0:
        reasons.append("entry_cooldown_active")
    last_loss_age = _age_seconds(metrics.get("last_losing_trade_exit_at"), now)
    last_loss_at = _parse_timestamp(metrics.get("last_losing_trade_exit_at"))
    loss_cooldown_overridden = bool(resume_at and last_loss_at and last_loss_at <= resume_at)
    if not loss_cooldown_overridden and last_loss_age is not None and last_loss_age < float(args.loss_cooldown_minutes) * 60.0:
        reasons.append("loss_cooldown_active")
    if int(metrics["buy_entries_today"]) >= int(args.max_trades_per_day):
        reasons.append("max_trades_per_day_reached")
    max_realized_loss = _float_or_none(getattr(args, "max_realized_loss_per_day", None))
    if max_realized_loss is not None and max_realized_loss > 0 and float(metrics["realized_pnl_today"]) <= -abs(max_realized_loss):
        reasons.append("max_realized_loss_per_day_reached")
    if int(metrics["consecutive_losses"]) >= int(args.max_consecutive_losses):
        reasons.append("max_consecutive_losses_reached")
    forecast_key = _forecast_key(forecast_bundle)
    traded_forecast_keys = state.get("traded_forecast_keys") or []
    if bool(args.one_trade_per_forecast) and forecast_key and forecast_key in traded_forecast_keys:
        reasons.append("one_trade_per_forecast_used")
    if _waiting_until_next_forecast(state, forecast_bundle):
        reasons.append("waiting_until_next_forecast_after_close")
    return {
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": metrics,
        "forecast_key": forecast_key,
        "profile_defaults": OPTION_AGENT_PROFILE_DEFAULTS[str(args.risk_profile)],
        "manual_trade_resume_at_utc": state.get("manual_trade_resume_at_utc"),
    }


def option_entry_guard_metrics(*, closed_orders: list[dict[str, Any]], now: datetime) -> dict[str, Any]:
    trades = match_option_round_trips(closed_orders)
    start_day = now.date().isoformat()
    buy_entries_today = sum(1 for order in closed_orders if order.get("side") == "buy" and str(order.get("filled_at") or "").startswith(start_day))
    last_filled_order_at = max((str(order.get("filled_at") or "") for order in closed_orders if order.get("filled_at")), default=None)
    losing_trades = [trade for trade in trades if float(trade["pnl"]) < 0.0]
    last_losing_trade_exit_at = max((str(trade.get("exit_time") or "") for trade in losing_trades), default=None)
    consecutive_losses = 0
    for trade in reversed(trades):
        if float(trade["pnl"]) < 0.0:
            consecutive_losses += 1
        else:
            break
    return {
        "buy_entries_today": buy_entries_today,
        "round_trips_today": sum(1 for trade in trades if str(trade.get("exit_time") or "").startswith(start_day)),
        "realized_pnl_today": round(sum(float(trade["pnl"]) for trade in trades if str(trade.get("exit_time") or "").startswith(start_day)), 2),
        "consecutive_losses": consecutive_losses,
        "last_filled_order_at": last_filled_order_at,
        "last_losing_trade_exit_at": last_losing_trade_exit_at,
    }


def match_option_round_trips(closed_orders: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lots: dict[str, list[dict[str, Any]]] = {}
    trades: list[dict[str, Any]] = []
    for order in sorted(closed_orders, key=lambda row: str(row.get("filled_at") or row.get("submitted_at") or "")):
        symbol = str(order.get("symbol") or "")
        qty = _int_float(order.get("filled_qty"))
        price = _float_or_none(order.get("filled_avg_price"))
        if not symbol or qty <= 0 or price is None:
            continue
        if order.get("side") == "buy":
            lots.setdefault(symbol, []).append({"qty": qty, "price": price, "filled_at": order.get("filled_at")})
            continue
        if order.get("side") != "sell":
            continue
        remaining = qty
        cost = 0.0
        matched = 0
        symbol_lots = lots.setdefault(symbol, [])
        while remaining > 0 and symbol_lots:
            lot = symbol_lots[0]
            take = min(remaining, int(lot["qty"]))
            cost += take * float(lot["price"]) * 100.0
            matched += take
            remaining -= take
            lot["qty"] = int(lot["qty"]) - take
            if int(lot["qty"]) <= 0:
                symbol_lots.pop(0)
        if matched:
            proceeds = matched * price * 100.0
            trades.append(
                {
                    "symbol": symbol,
                    "qty": matched,
                    "exit_time": order.get("filled_at"),
                    "cost": round(cost, 2),
                    "proceeds": round(proceeds, 2),
                    "pnl": round(proceeds - cost, 2),
                }
            )
    return trades


def manage_existing_option_orders_and_positions(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
    forecast: dict[str, Any] | None = None,
    underlying_price: float | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    actions.extend(_abandon_stale_entry_orders(broker=broker, args=args, open_orders=open_orders, state=state, now=now))
    if not args.execute_paper_orders:
        return actions
    total_loss_actions = _manage_total_unrealized_loss(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        now=now,
    )
    if total_loss_actions:
        actions.extend(total_loss_actions)
        return actions
    total_profit_actions = _manage_total_unrealized_profit(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        now=now,
    )
    if total_profit_actions:
        actions.extend(total_profit_actions)
        return actions
    position_guard_actions = _manage_position_unrealized_guards(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        now=now,
    )
    if position_guard_actions:
        actions.extend(position_guard_actions)
        if _protective_close_triggered(position_guard_actions):
            return actions
    expiry_actions = _manage_expiry_risk(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        now=now,
    )
    if expiry_actions:
        actions.extend(expiry_actions)
        if _protective_close_triggered(expiry_actions):
            return actions
    reversal_actions = _manage_forecast_reversal_exit(
        broker=broker,
        args=args,
        open_orders=open_orders,
        option_positions=option_positions,
        now=now,
        forecast=forecast or {},
        underlying_price=underlying_price,
    )
    if reversal_actions:
        actions.extend(reversal_actions)
        if _protective_close_triggered(reversal_actions):
            return actions
    _prune_position_profit_peaks(state, option_positions)
    active_trade = state.get("active_trade") or {}
    trade_plan = active_trade.get("trade_plan") or {}
    for position in option_positions:
        symbol = str(position.get("symbol") or "")
        qty = _int_float(position.get("qty"))
        if not symbol or qty <= 0:
            continue
        sell_orders = [
            order
            for order in open_orders
            if str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell"
        ]
        entry_price = _float_or_none(position.get("avg_entry_price")) or _entry_price_from_state(active_trade)
        if entry_price is None:
            continue
        profit_peak = _update_position_profit_peak(state, position=position, now=now)
        profit_action = _manage_take_profit(
            broker=broker,
            args=args,
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            position=position,
            sell_orders=sell_orders,
            profit_peak=profit_peak,
            now=now,
        )
        if profit_action:
            actions.extend(profit_action)
            continue
        risk_action = _manage_configured_stop_loss(
            broker=broker,
            args=args,
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            position=position,
            sell_orders=sell_orders,
            now=now,
        )
        if risk_action:
            actions.extend(risk_action)
            continue
        if sell_orders:
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
                client_order_id=_client_order_id("opt-stop", args.ticker.upper(), symbol, now),
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
                    client_order_id=_client_order_id("opt-stop-fallback", args.ticker.upper(), symbol, now),
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


def _manage_total_unrealized_loss(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    threshold = _float_or_none(getattr(args, "max_total_unrealized_loss", None))
    if threshold is None or threshold <= 0:
        return []
    open_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    if not open_positions:
        return []
    total_pl = sum(_float_or_none(position.get("unrealized_pl")) or 0.0 for position in open_positions)
    loss_cutoff = -abs(float(threshold))
    if total_pl > loss_cutoff:
        return []
    actions: list[dict[str, Any]] = [
        {
            "action": "max_total_unrealized_loss_triggered",
            "ticker": args.ticker.upper(),
            "total_unrealized_pl": round(total_pl, 2),
            "loss_cutoff": round(loss_cutoff, 2),
            "position_count": len(open_positions),
            "close_mode": args.total_loss_close_mode,
        }
    ]
    if not args.execute_paper_orders:
        actions.append({"action": "would_close_all_positions_for_total_loss", "ticker": args.ticker.upper()})
        return actions
    positions_to_close = _positions_for_total_loss_close(open_positions, str(args.total_loss_close_mode))
    if not positions_to_close:
        actions.append({"action": "total_loss_no_positions_match_close_mode", "close_mode": args.total_loss_close_mode})
        return actions
    for position in positions_to_close:
        symbol = str(position.get("symbol") or "")
        qty = _int_float(position.get("qty"))
        current_price = _current_option_price(position)
        if not symbol or qty <= 0 or current_price is None:
            actions.append({"action": "total_loss_close_skipped_missing_price_or_qty", "symbol": symbol, "qty": qty})
            continue
        sell_orders = [
            order
            for order in open_orders
            if str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell"
        ]
        close_limit = _profit_close_limit_price(current_price, float(args.profit_close_limit_offset_pct))
        actions.extend(
            _replace_sell_orders(
                broker=broker,
                args=args,
                symbol=symbol,
                qty=qty,
                sell_orders=sell_orders,
                replacement={
                    "symbol": symbol,
                    "side": "sell",
                    "order_type": "limit",
                    "qty": qty,
                    "limit_price": close_limit,
                    "time_in_force": "day",
                    "client_order_id": f"opt-maxloss-{args.ticker.upper()}-{symbol[4:10]}-{symbol[-8:]}-{now.strftime('%Y%m%d%H%M%S')}",
                },
                trigger_action={
                    "action": "total_loss_position_close_triggered",
                    "symbol": symbol,
                    "qty": qty,
                    "current_price": round(float(current_price), 4),
                    "close_limit_price": close_limit,
                    "position_unrealized_pl": _float_or_none(position.get("unrealized_pl")),
                },
                cancel_action="cancelled_existing_exit_for_total_loss",
                submit_action="submitted_total_loss_close",
                now=now,
            )
        )
    return actions


def _manage_total_unrealized_profit(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    threshold = _float_or_none(getattr(args, "max_total_unrealized_profit", None))
    if threshold is None or threshold <= 0:
        return []
    open_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    if not open_positions:
        return []
    total_pl = sum(_float_or_none(position.get("unrealized_pl")) or 0.0 for position in open_positions)
    if total_pl < float(threshold):
        return []
    close_mode = str(getattr(args, "total_profit_close_mode", "all"))
    actions: list[dict[str, Any]] = [
        {
            "action": "max_total_unrealized_profit_triggered",
            "ticker": args.ticker.upper(),
            "total_unrealized_pl": round(total_pl, 2),
            "profit_target": round(float(threshold), 2),
            "position_count": len(open_positions),
            "close_mode": close_mode,
        }
    ]
    positions_to_close = _positions_for_total_profit_close(open_positions, close_mode)
    if not positions_to_close:
        actions.append({"action": "total_profit_no_positions_match_close_mode", "close_mode": close_mode})
        return actions
    for position in positions_to_close:
        symbol = str(position.get("symbol") or "")
        qty = _int_float(position.get("qty"))
        current_price = _current_option_price(position)
        if not symbol or qty <= 0 or current_price is None:
            actions.append({"action": "total_profit_close_skipped_missing_price_or_qty", "symbol": symbol, "qty": qty})
            continue
        sell_orders = [
            order
            for order in open_orders
            if str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell"
        ]
        close_limit = _profit_close_limit_price(current_price, float(args.profit_close_limit_offset_pct))
        actions.extend(
            _replace_sell_orders(
                broker=broker,
                args=args,
                symbol=symbol,
                qty=qty,
                sell_orders=sell_orders,
                replacement={
                    "symbol": symbol,
                    "side": "sell",
                    "order_type": "limit",
                    "qty": qty,
                    "limit_price": close_limit,
                    "time_in_force": "day",
                    "client_order_id": f"opt-maxtp-{args.ticker.upper()}-{symbol[4:10]}-{symbol[-8:]}-{now.strftime('%Y%m%d%H%M%S')}",
                },
                trigger_action={
                    "action": "total_profit_position_close_triggered",
                    "symbol": symbol,
                    "qty": qty,
                    "current_price": round(float(current_price), 4),
                    "close_limit_price": close_limit,
                    "position_unrealized_pl": _float_or_none(position.get("unrealized_pl")),
                },
                cancel_action="cancelled_existing_exit_for_total_profit",
                submit_action="submitted_total_profit_close",
                now=now,
            )
        )
    return actions


def _manage_position_unrealized_guards(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    loss_threshold = _float_or_none(getattr(args, "max_position_unrealized_loss", None))
    profit_threshold = _float_or_none(getattr(args, "max_position_unrealized_profit", None))
    if loss_threshold is not None:
        loss_threshold = abs(float(loss_threshold))
    profit_enabled = profit_threshold is not None and float(profit_threshold) >= 0.0
    if loss_threshold is None and not profit_enabled:
        return []
    actions: list[dict[str, Any]] = []
    open_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    for position in open_positions:
        unrealized_pl = _float_or_none(position.get("unrealized_pl"))
        if unrealized_pl is None:
            actions.append({"action": "position_unrealized_guard_skipped_missing_pl", "symbol": position.get("symbol")})
            continue
        trigger_reason = None
        threshold_payload: dict[str, Any] = {}
        if loss_threshold is not None:
            loss_cutoff = -abs(float(loss_threshold))
            if (float(loss_threshold) == 0.0 and unrealized_pl < 0.0) or (float(loss_threshold) > 0.0 and unrealized_pl <= loss_cutoff):
                trigger_reason = "position_loss_position_close_triggered"
                threshold_payload = {"position_loss_cutoff": round(loss_cutoff, 2)}
        if trigger_reason is None and profit_enabled and profit_threshold is not None:
            if (float(profit_threshold) == 0.0 and unrealized_pl > 0.0) or (float(profit_threshold) > 0.0 and unrealized_pl >= float(profit_threshold)):
                trigger_reason = "position_profit_position_close_triggered"
                threshold_payload = {"position_profit_target": round(float(profit_threshold), 2)}
        if trigger_reason is None:
            actions.append(
                {
                    "action": "position_unrealized_guard_not_triggered",
                    "symbol": position.get("symbol"),
                    "position_unrealized_pl": round(float(unrealized_pl), 2),
                    "loss_guard": loss_threshold,
                    "profit_guard": profit_threshold,
                }
            )
            continue
        actions.extend(
            _close_position_with_limit(
                broker=broker,
                args=args,
                open_orders=open_orders,
                position=position,
                now=now,
                reason=trigger_reason,
                limit_offset_pct=float(getattr(args, "profit_close_limit_offset_pct", 0.03)),
                extra={
                    "position_unrealized_pl": round(float(unrealized_pl), 2),
                    **threshold_payload,
                },
            )
        )
    return actions


def _manage_forecast_reversal_exit(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
    forecast: dict[str, Any],
    underlying_price: float | None,
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "enable_forecast_reversal_exit", True)):
        return []
    spot = _float_or_none(forecast.get("spot")) or _float_or_none(underlying_price)
    predicted_price = _float_or_none(forecast.get("predicted_price"))
    if spot is None or spot <= 0 or predicted_price is None:
        return []
    forecast_move_pct = (float(predicted_price) - float(spot)) / float(spot)
    min_edge = abs(float(getattr(args, "min_reversal_edge_pct", 0.001)))
    if abs(forecast_move_pct) < min_edge:
        return [
            {
                "action": "forecast_reversal_exit_not_triggered_edge_too_small",
                "forecast_direction": forecast.get("expected_direction"),
                "forecast_move_pct": round(forecast_move_pct, 6),
                "min_reversal_edge_pct": min_edge,
            }
        ]
    expected_direction = str(forecast.get("expected_direction") or "").lower()
    desired_type = None
    if expected_direction == "upward" or forecast_move_pct > 0:
        desired_type = "call"
    elif expected_direction == "downward" or forecast_move_pct < 0:
        desired_type = "put"
    if desired_type is None:
        return []
    actions: list[dict[str, Any]] = []
    open_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    for position in open_positions:
        symbol = str(position.get("symbol") or "")
        current_type = _option_type_from_symbol(symbol)
        if not current_type:
            continue
        if current_type == desired_type:
            actions.append(
                {
                    "action": "forecast_reversal_exit_not_triggered_position_aligned",
                    "symbol": symbol,
                    "position_option_type": current_type,
                    "desired_option_type": desired_type,
                    "forecast_direction": forecast.get("expected_direction"),
                    "forecast_move_pct": round(forecast_move_pct, 6),
                }
            )
            continue
        actions.extend(
            _close_position_with_limit(
                broker=broker,
                args=args,
                open_orders=open_orders,
                position=position,
                now=now,
                reason="forecast_reversal_position_close_triggered",
                limit_offset_pct=float(getattr(args, "profit_close_limit_offset_pct", 0.03)),
                extra={
                    "position_option_type": current_type,
                    "desired_option_type": desired_type,
                    "forecast_direction": forecast.get("expected_direction"),
                    "forecast_spot": round(float(spot), 4),
                    "forecast_price": round(float(predicted_price), 4),
                    "forecast_move_pct": round(forecast_move_pct, 6),
                    "min_reversal_edge_pct": min_edge,
                },
            )
        )
    return actions


def _manage_expiry_risk(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    active_positions = [position for position in option_positions if _int_float(position.get("qty")) > 0]
    if not active_positions:
        return []
    close_before_hours = max(0.0, float(getattr(args, "close_before_expiry_hours", 2.0)))
    warning_hours = max(close_before_hours, float(getattr(args, "expiry_warning_hours", 24.0)))
    actions: list[dict[str, Any]] = []
    for position in active_positions:
        symbol = str(position.get("symbol") or "")
        expiry = _option_expiry_utc_from_symbol(symbol)
        hours_to_expiry = None if expiry is None else (expiry - now).total_seconds() / 3600.0
        if not symbol or hours_to_expiry is None:
            actions.append({"action": "expiry_risk_skipped_missing_symbol_or_expiry", "symbol": symbol})
            continue
        expiry_payload = {
            "symbol": symbol,
            "expiration_utc": expiry.isoformat(),
            "hours_to_expiry": round(hours_to_expiry, 2),
            "close_before_expiry_hours": close_before_hours,
            "expiry_warning_hours": warning_hours,
        }
        if hours_to_expiry <= close_before_hours:
            actions.extend(
                _close_position_with_limit(
                    broker=broker,
                    args=args,
                    open_orders=open_orders,
                    position=position,
                    now=now,
                    reason="expiry_position_close_triggered",
                    limit_offset_pct=float(getattr(args, "liquidation_limit_offset_pct", 0.05)),
                    extra=expiry_payload,
                )
            )
        elif hours_to_expiry <= warning_hours:
            actions.append({"action": "expiry_position_warning", **expiry_payload})
    return actions


def _close_position_with_limit(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    position: dict[str, Any],
    now: datetime,
    reason: str,
    limit_offset_pct: float,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    symbol = str(position.get("symbol") or "")
    qty = _int_float(position.get("qty"))
    current_price = _current_option_price(position)
    if not symbol or qty <= 0 or current_price is None:
        return [{"action": f"{reason}_skipped_missing_price_or_qty", "symbol": symbol, "qty": qty}]
    sell_orders = [
        order
        for order in open_orders
        if str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell"
    ]
    close_limit = _profit_close_limit_price(current_price, float(limit_offset_pct))
    return _replace_sell_orders(
        broker=broker,
        args=args,
        symbol=symbol,
        qty=qty,
        sell_orders=sell_orders,
        replacement={
            "symbol": symbol,
            "side": "sell",
            "order_type": "limit",
            "qty": qty,
            "limit_price": close_limit,
            "time_in_force": "day",
            "client_order_id": _client_order_id("opt-riskguard", args.ticker.upper(), symbol, now),
        },
        trigger_action={
            "action": reason,
            "symbol": symbol,
            "qty": qty,
            "current_price": round(float(current_price), 4),
            "close_limit_price": close_limit,
            "position_unrealized_pl": _float_or_none(position.get("unrealized_pl")),
            **(extra or {}),
        },
        cancel_action=f"cancelled_existing_exit_for_{reason}",
        submit_action=f"submitted_{reason}_close",
        now=now,
    )


def verify_liquidation_until_flat(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    started_at: datetime,
) -> dict[str, Any]:
    deadline = datetime.now(UTC).timestamp() + max(0.0, float(getattr(args, "liquidation_wait_seconds", 30.0)))
    poll_seconds = max(0.5, float(getattr(args, "liquidation_poll_seconds", 2.0)))
    actions: list[dict[str, Any]] = []
    retry_submitted = False
    open_orders: list[dict[str, Any]] = []
    option_positions: list[dict[str, Any]] = []
    while True:
        open_orders = _open_option_orders(broker.orders(status="open", limit=100), args.ticker.upper())
        option_positions = _option_positions(broker.positions(), args.ticker.upper())
        if not option_positions and not open_orders:
            actions.append({"action": "liquidation_verified_flat", "ticker": args.ticker.upper(), "checked_at": datetime.now(UTC).isoformat()})
            return {
                "flat": True,
                "actions": actions,
                "open_orders": open_orders,
                "option_positions": option_positions,
                "started_at": started_at.isoformat(),
                "finished_at": datetime.now(UTC).isoformat(),
            }
        if not option_positions and open_orders and args.execute_paper_orders:
            actions.append(
                {
                    "action": "liquidation_cancel_remaining_open_orders",
                    "ticker": args.ticker.upper(),
                    "remaining_open_order_count": len(open_orders),
                }
            )
            actions.extend(
                _cancel_open_option_orders(
                    broker=broker,
                    args=args,
                    open_orders=open_orders,
                    cancel_action="cancelled_remaining_open_order_for_liquidation",
                )
            )
        if option_positions and args.execute_paper_orders and not retry_submitted and datetime.now(UTC).timestamp() >= min(deadline, started_at.timestamp() + poll_seconds):
            actions.append(
                {
                    "action": "liquidation_retry_triggered",
                    "ticker": args.ticker.upper(),
                    "remaining_position_count": len(option_positions),
                    "remaining_open_order_count": len(open_orders),
                    "retry_limit_offset_pct": float(getattr(args, "liquidation_retry_limit_offset_pct", 0.15)),
                }
            )
            actions.extend(
                _cancel_open_option_orders(
                    broker=broker,
                    args=args,
                    open_orders=open_orders,
                    cancel_action="cancelled_open_order_for_liquidation_retry",
                )
            )
            retry_args = argparse.Namespace(**vars(args))
            retry_args.liquidation_limit_offset_pct = float(getattr(args, "liquidation_retry_limit_offset_pct", 0.15))
            actions.extend(
                _liquidate_option_positions(
                    broker=broker,
                    args=retry_args,
                    open_orders=[],
                    option_positions=option_positions,
                    now=datetime.now(UTC),
                )
            )
            retry_submitted = True
        if datetime.now(UTC).timestamp() >= deadline:
            break
        time.sleep(poll_seconds)
    open_orders = _open_option_orders(broker.orders(status="open", limit=100), args.ticker.upper())
    option_positions = _option_positions(broker.positions(), args.ticker.upper())
    if not option_positions and not open_orders:
        actions.append({"action": "liquidation_verified_flat", "ticker": args.ticker.upper(), "checked_at": datetime.now(UTC).isoformat()})
        return {
            "flat": True,
            "actions": actions,
            "open_orders": open_orders,
            "option_positions": option_positions,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
        }
    actions.append(
        {
            "action": "liquidation_not_flat_after_timeout",
            "ticker": args.ticker.upper(),
            "remaining_position_count": len(option_positions),
            "remaining_open_order_count": len(open_orders),
            "checked_at": datetime.now(UTC).isoformat(),
        }
    )
    return {
        "flat": False,
        "actions": actions,
        "open_orders": open_orders,
        "option_positions": option_positions,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }


def _liquidate_option_positions(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    option_positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "action": "manual_liquidate_and_stop_triggered",
            "ticker": args.ticker.upper(),
            "position_count": len([position for position in option_positions if _int_float(position.get("qty")) > 0]),
        }
    ]
    for position in option_positions:
        symbol = str(position.get("symbol") or "")
        qty = _int_float(position.get("qty"))
        current_price = _current_option_price(position)
        if not symbol or qty <= 0 or current_price is None:
            actions.append({"action": "manual_liquidation_skipped_missing_price_or_qty", "symbol": symbol, "qty": qty})
            continue
        sell_orders = [
            order
            for order in open_orders
            if str(order.get("symbol") or "") == symbol and str(order.get("side") or "").lower() == "sell"
        ]
        close_limit = _profit_close_limit_price(current_price, float(args.liquidation_limit_offset_pct))
        actions.extend(
            _replace_sell_orders(
                broker=broker,
                args=args,
                symbol=symbol,
                qty=qty,
                sell_orders=sell_orders,
                replacement={
                    "symbol": symbol,
                    "side": "sell",
                    "order_type": "limit",
                    "qty": qty,
                    "limit_price": close_limit,
                    "time_in_force": "day",
                    "client_order_id": _client_order_id("opt-emergency", args.ticker.upper(), symbol, now),
                },
                trigger_action={
                    "action": "manual_liquidation_position_close_triggered",
                    "symbol": symbol,
                    "qty": qty,
                    "current_price": round(float(current_price), 4),
                    "close_limit_price": close_limit,
                    "position_unrealized_pl": _float_or_none(position.get("unrealized_pl")),
                },
                cancel_action="cancelled_existing_exit_for_manual_liquidation",
                submit_action="submitted_manual_liquidation_close",
                now=now,
            )
        )
    return actions


def _cancel_open_option_orders(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    cancel_action: str,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not args.execute_paper_orders:
        return [{"action": f"would_{cancel_action}", "ticker": args.ticker.upper(), "order_count": len(open_orders)}] if open_orders else []
    for order in open_orders:
        order_id = order.get("id")
        if not order_id:
            continue
        try:
            broker.cancel_order(str(order_id))
            actions.append(
                {
                    "action": cancel_action,
                    "ticker": args.ticker.upper(),
                    "order_id": order_id,
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "type": order.get("type"),
                }
            )
        except RuntimeError as exc:
            actions.append(
                {
                    "action": f"{cancel_action}_failed",
                    "ticker": args.ticker.upper(),
                    "order_id": order_id,
                    "symbol": order.get("symbol"),
                    "error": str(exc),
                }
            )
    return actions


def _positions_for_total_loss_close(positions: list[dict[str, Any]], close_mode: str) -> list[dict[str, Any]]:
    if close_mode == "losing_only":
        return [position for position in positions if (_float_or_none(position.get("unrealized_pl")) or 0.0) < 0.0]
    return positions


def _positions_for_total_profit_close(positions: list[dict[str, Any]], close_mode: str) -> list[dict[str, Any]]:
    if close_mode == "winning_only":
        return [position for position in positions if (_float_or_none(position.get("unrealized_pl")) or 0.0) > 0.0]
    return positions


def _protective_close_triggered(actions: list[dict[str, Any]]) -> bool:
    close_markers = (
        "total_loss_position_close_triggered",
        "total_profit_position_close_triggered",
        "position_loss_position_close_triggered",
        "position_profit_position_close_triggered",
        "forecast_reversal_position_close_triggered",
        "expiry_position_close_triggered",
        "configured_stop_already_breached",
        "take_profit_triggered",
        "manual_liquidation_position_close_triggered",
    )
    would_markers = tuple(f"would_{marker}" for marker in close_markers)
    for action in actions:
        name = str(action.get("action") or "")
        if name in close_markers or name in would_markers:
            return True
    return False


def _waiting_until_next_forecast(state: dict[str, Any], forecast_bundle: dict[str, Any]) -> bool:
    wait = state.get("wait_until_next_forecast_after_close")
    if not isinstance(wait, dict):
        return False
    wait_forecast = wait.get("forecast_created_at_utc")
    current_forecast = forecast_bundle.get("created_at_utc")
    return bool(wait_forecast and current_forecast and str(wait_forecast) == str(current_forecast))


def _manage_configured_stop_loss(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    symbol: str,
    qty: int,
    entry_price: float,
    position: dict[str, Any],
    sell_orders: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    current_price = _current_option_price(position)
    if current_price is None:
        return []
    desired_stop = round(max(0.01, float(entry_price) * (1.0 - float(args.stop_loss_pct))), 2)
    desired_limit = round(max(0.01, desired_stop * (1.0 - float(args.stop_limit_offset_pct))), 2)
    existing_best_stop = max((_float_or_none(order.get("stop_price")) or 0.0 for order in sell_orders), default=0.0)
    if sell_orders and existing_best_stop >= desired_stop:
        return []
    if current_price <= desired_stop:
        close_limit = _profit_close_limit_price(current_price, float(args.profit_close_limit_offset_pct))
        return _replace_sell_orders(
            broker=broker,
            args=args,
            symbol=symbol,
            qty=qty,
            sell_orders=sell_orders,
            replacement={
                "symbol": symbol,
                "side": "sell",
                "order_type": "limit",
                "qty": qty,
                "limit_price": close_limit,
                "time_in_force": "day",
                    "client_order_id": _client_order_id("opt-risk", args.ticker.upper(), symbol, now),
            },
            trigger_action={
                "action": "configured_stop_already_breached",
                "symbol": symbol,
                "qty": qty,
                "entry_price": round(float(entry_price), 4),
                "current_price": round(float(current_price), 4),
                "desired_stop_price": desired_stop,
                "close_limit_price": close_limit,
            },
            cancel_action="cancelled_existing_exit_for_configured_stop",
            submit_action="submitted_configured_stop_risk_close",
            now=now,
        )
    return _replace_sell_orders(
        broker=broker,
        args=args,
        symbol=symbol,
        qty=qty,
        sell_orders=sell_orders,
        replacement={
            "symbol": symbol,
            "side": "sell",
            "order_type": "stop_limit",
            "qty": qty,
            "stop_price": desired_stop,
            "limit_price": desired_limit,
            "time_in_force": "day",
            "client_order_id": _client_order_id("opt-stopcfg", args.ticker.upper(), symbol, now),
        },
        trigger_action={
            "action": "configured_stop_update_triggered",
            "symbol": symbol,
            "qty": qty,
            "entry_price": round(float(entry_price), 4),
            "current_price": round(float(current_price), 4),
            "old_best_stop_price": existing_best_stop or None,
            "desired_stop_price": desired_stop,
            "desired_stop_limit": desired_limit,
        },
        cancel_action="cancelled_existing_exit_for_configured_stop",
        submit_action="submitted_configured_stop_update",
        now=now,
    )


def _manage_take_profit(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    symbol: str,
    qty: int,
    entry_price: float,
    position: dict[str, Any],
    sell_orders: list[dict[str, Any]],
    profit_peak: dict[str, Any] | None,
    now: datetime,
) -> list[dict[str, Any]]:
    if bool(getattr(args, "disable_profit_taking", False)):
        return []
    current_price = _current_option_price(position)
    if current_price is None:
        return []
    take_profit_price = round(float(entry_price) * (1.0 + float(args.take_profit_pct)), 2)
    unrealized_pl = _float_or_none(position.get("unrealized_pl"))
    if unrealized_pl is not None and unrealized_pl <= 0:
        return []
    position_profit_target = _float_or_none(getattr(args, "take_profit_position_pl", None))
    position_profit_reached = bool(position_profit_target is not None and position_profit_target > 0 and unrealized_pl is not None and unrealized_pl >= position_profit_target)
    peak_profit = _float_or_none((profit_peak or {}).get("peak_unrealized_pl"))
    retrace_pct = max(0.0, min(1.0, float(getattr(args, "profit_retrace_from_peak_pct", 0.35))))
    retrace_triggered = bool(
        position_profit_target is not None
        and position_profit_target > 0
        and peak_profit is not None
        and peak_profit >= position_profit_target
        and unrealized_pl is not None
        and unrealized_pl > 0
        and unrealized_pl <= peak_profit * (1.0 - retrace_pct)
    )
    price_profit_reached = current_price >= take_profit_price
    if not position_profit_reached and not price_profit_reached and not retrace_triggered:
        return _manage_profit_lock(
            broker=broker,
            args=args,
            symbol=symbol,
            qty=qty,
            entry_price=entry_price,
            current_price=current_price,
            sell_orders=sell_orders,
            now=now,
        )
    actions: list[dict[str, Any]] = [
        {
            "action": "take_profit_triggered",
            "symbol": symbol,
            "qty": qty,
            "entry_price": round(float(entry_price), 4),
            "current_price": round(float(current_price), 4),
            "take_profit_price": take_profit_price,
            "take_profit_position_pl": position_profit_target,
            "peak_unrealized_pl": peak_profit,
            "profit_retrace_from_peak_pct": retrace_pct,
            "trigger": "profit_retrace" if retrace_triggered else "position_pl" if position_profit_reached else "price_pct",
            "unrealized_pl": unrealized_pl,
        }
    ]
    if not args.execute_paper_orders:
        actions.append({"action": "would_submit_take_profit_close", "symbol": symbol, "qty": qty})
        return actions
    limit_price = _profit_close_limit_price(current_price, float(args.profit_close_limit_offset_pct))
    return _replace_sell_orders(
        broker=broker,
        args=args,
        symbol=symbol,
        qty=qty,
        sell_orders=sell_orders,
        replacement={
            "symbol": symbol,
            "side": "sell",
            "order_type": "limit",
            "qty": qty,
            "limit_price": limit_price,
            "time_in_force": "day",
            "client_order_id": _client_order_id("opt-tp", args.ticker.upper(), symbol, now),
        },
        trigger_action=actions[0],
        cancel_action="cancelled_existing_exit_for_take_profit",
        submit_action="submitted_take_profit_close",
        now=now,
    )


def _update_position_profit_peak(state: dict[str, Any], *, position: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    symbol = str(position.get("symbol") or "")
    unrealized_pl = _float_or_none(position.get("unrealized_pl"))
    current_price = _current_option_price(position)
    if not symbol or unrealized_pl is None:
        return None
    peaks = state.setdefault("option_position_profit_peaks", {})
    if not isinstance(peaks, dict):
        peaks = {}
        state["option_position_profit_peaks"] = peaks
    existing = peaks.get(symbol) if isinstance(peaks.get(symbol), dict) else {}
    existing_peak = _float_or_none(existing.get("peak_unrealized_pl"))
    if existing_peak is None or unrealized_pl > existing_peak:
        existing = {
            "symbol": symbol,
            "peak_unrealized_pl": round(float(unrealized_pl), 2),
            "peak_current_price": None if current_price is None else round(float(current_price), 4),
            "updated_at_utc": now.isoformat(),
        }
        peaks[symbol] = existing
    return existing


def _prune_position_profit_peaks(state: dict[str, Any], option_positions: list[dict[str, Any]]) -> None:
    peaks = state.get("option_position_profit_peaks")
    if not isinstance(peaks, dict):
        return
    open_symbols = {
        str(position.get("symbol") or "")
        for position in option_positions
        if _int_float(position.get("qty")) > 0 and position.get("symbol")
    }
    for symbol in list(peaks):
        if symbol not in open_symbols:
            peaks.pop(symbol, None)


def _manage_profit_lock(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    symbol: str,
    qty: int,
    entry_price: float,
    current_price: float,
    sell_orders: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    trigger_price = float(entry_price) * (1.0 + float(args.profit_lock_trigger_pct))
    if current_price < trigger_price:
        return []
    locked_stop = _profit_lock_stop_price(
        entry_price=entry_price,
        current_price=current_price,
        lock_ratio=float(args.profit_lock_ratio),
    )
    existing_best_stop = max((_float_or_none(order.get("stop_price")) or 0.0 for order in sell_orders), default=0.0)
    if existing_best_stop >= locked_stop:
        return []
    stop_limit = round(max(0.01, locked_stop * (1.0 - float(args.stop_limit_offset_pct))), 2)
    return _replace_sell_orders(
        broker=broker,
        args=args,
        symbol=symbol,
        qty=qty,
        sell_orders=sell_orders,
        replacement={
            "symbol": symbol,
            "side": "sell",
            "order_type": "stop_limit",
            "qty": qty,
            "stop_price": locked_stop,
            "limit_price": stop_limit,
            "time_in_force": "day",
            "client_order_id": _client_order_id("opt-lock", args.ticker.upper(), symbol, now),
        },
        trigger_action={
            "action": "profit_lock_triggered",
            "symbol": symbol,
            "qty": qty,
            "entry_price": round(float(entry_price), 4),
            "current_price": round(float(current_price), 4),
            "locked_stop_price": locked_stop,
            "locked_stop_limit": stop_limit,
        },
        cancel_action="cancelled_existing_exit_for_profit_lock",
        submit_action="submitted_profit_lock_stop",
        now=now,
    )


def _replace_sell_orders(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    symbol: str,
    qty: int,
    sell_orders: list[dict[str, Any]],
    replacement: dict[str, Any],
    trigger_action: dict[str, Any],
    cancel_action: str,
    submit_action: str,
    now: datetime,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [trigger_action]
    if not args.execute_paper_orders:
        actions.append({"action": f"would_{submit_action}", "symbol": symbol, "qty": qty})
        return actions
    for order in sell_orders:
        order_id = order.get("id")
        if not order_id:
            continue
        try:
            broker.cancel_order(str(order_id))
            actions.append({"action": cancel_action, "symbol": symbol, "order_id": order_id})
        except RuntimeError as exc:
            actions.append({"action": f"{cancel_action}_failed", "symbol": symbol, "order_id": order_id, "error": str(exc)})
            return actions
    try:
        order = broker.submit_order(**replacement)
        submitted = {
            "action": submit_action,
            "symbol": symbol,
            "qty": qty,
            "order": order,
        }
        for key in ("stop_price", "limit_price"):
            if key in replacement:
                submitted[key] = replacement[key]
        actions.append(submitted)
    except RuntimeError as exc:
        actions.append({"action": f"{submit_action}_rejected", "symbol": symbol, "error": str(exc), "replacement": replacement})
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
                "stop_price": order.get("stop_price"),
                "status": order.get("status"),
                "submitted_at": order.get("submitted_at"),
            }
        )
    return output


def _closed_option_orders(orders: list[dict[str, Any]], underlying: str) -> list[dict[str, Any]]:
    output = []
    for order in orders:
        symbol = str(order.get("symbol") or "").upper()
        if not symbol.startswith(underlying.upper()) or not _looks_like_option_symbol(symbol):
            continue
        if order.get("status") != "filled":
            continue
        output.append(
            {
                "id": order.get("id"),
                "symbol": symbol,
                "side": order.get("side"),
                "type": order.get("type"),
                "qty": order.get("qty"),
                "filled_qty": order.get("filled_qty"),
                "filled_avg_price": order.get("filled_avg_price"),
                "limit_price": order.get("limit_price"),
                "status": order.get("status"),
                "submitted_at": order.get("submitted_at"),
                "filled_at": order.get("filled_at"),
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
                    "current_price": position.get("current_price"),
                    "cost_basis": position.get("cost_basis"),
                    "market_value": position.get("market_value"),
                    "unrealized_pl": position.get("unrealized_pl"),
                    "unrealized_plpc": position.get("unrealized_plpc"),
                }
            )
    return output


def _looks_like_option_symbol(symbol: str) -> bool:
    return len(symbol) >= 15 and ("C" in symbol[4:] or "P" in symbol[4:])


def _age_seconds(timestamp: Any, now: datetime) -> float | None:
    parsed = _parse_timestamp(timestamp)
    if parsed is None:
        return None
    return (now - parsed).total_seconds()


def _parse_timestamp(timestamp: Any) -> datetime | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _entry_price_from_state(active_trade: dict[str, Any]) -> float | None:
    order = ((active_trade.get("trade_plan") or {}).get("order") or {})
    return _float_or_none(order.get("limit_price"))


def _forecast_key(forecast_bundle: dict[str, Any]) -> str | None:
    created = forecast_bundle.get("created_at_utc")
    selected = forecast_bundle.get("selected_forecast") or {}
    horizon = selected.get("horizon_hours")
    if not created:
        return None
    return f"{created}|{horizon}"


def _option_type_from_symbol(symbol: str) -> str | None:
    match = re.search(r"\d{6}([CP])\d{8}$", str(symbol or "").upper())
    if not match:
        return None
    return "call" if match.group(1) == "C" else "put"


def _option_expiry_utc_from_symbol(symbol: str) -> datetime | None:
    match = re.search(r"(\d{2})(\d{2})(\d{2})[CP]\d{8}$", str(symbol or "").upper())
    if not match:
        return None
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    try:
        return datetime(year, month, day, 20, 0, tzinfo=UTC)
    except ValueError:
        return None


def _current_option_price(position: dict[str, Any]) -> float | None:
    current_price = _float_or_none(position.get("current_price"))
    if current_price is not None and current_price > 0:
        return current_price
    market_value = _float_or_none(position.get("market_value"))
    qty = _float_or_none(position.get("qty"))
    if market_value is None or qty is None or qty == 0:
        return None
    return abs(market_value) / (abs(qty) * 100.0)


def _profit_close_limit_price(current_price: float, offset_pct: float) -> float:
    return round(max(0.01, float(current_price) * (1.0 - max(0.0, float(offset_pct)))), 2)


def _profit_lock_stop_price(*, entry_price: float, current_price: float, lock_ratio: float) -> float:
    protected_gain = max(0.0, float(current_price) - float(entry_price)) * max(0.0, min(1.0, float(lock_ratio)))
    return round(max(0.01, float(entry_price) + protected_gain), 2)


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
