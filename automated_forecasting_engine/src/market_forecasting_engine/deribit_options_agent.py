from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.chart_patterns import build_chart_pattern_analysis
from market_forecasting_engine.curve_shape import build_curve_shape_analysis
from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.deribit_broker import DeribitOptionsBroker, DeribitTestnetBroker
from market_forecasting_engine.deribit_options_feedback import append_decision_to_ledger, update_feedback_loop
from market_forecasting_engine.deribit_options_trader import (
    DeribitOptionExecutionConfig,
    build_fibonacci_analysis,
    build_deribit_exit_plan,
    build_deribit_option_trade_plan,
    build_options_market_regime,
    submit_deribit_limit_order,
)
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def main() -> None:
    args = build_parser().parse_args()
    if args.account_mode == "live":
        if args.execute_paper_orders:
            raise SystemExit("Use --execute-live-orders for live Deribit options, not --execute-paper-orders.")
        if args.execute_live_orders and not args.confirm_live_deribit_options_orders:
            raise SystemExit("Live Deribit options execution requires --confirm-live-deribit-options-orders.")
    if args.account_mode == "testnet" and args.execute_live_orders:
        raise SystemExit("--execute-live-orders is only valid with --account-mode live.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.liquidate_and_stop:
        record = liquidate_and_stop(args)
        report_path = write_record(output_dir, args.currency, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        return
    if args.stop_agent:
        request = write_stop_request(output_dir, args.currency, reason="manual_stop_request")
        print(json.dumps({"status": "stop_requested", "currency": args.currency.upper(), "request": request}, indent=2, default=str), flush=True)
        return
    if args.clear_stop_request:
        cleared = clear_stop_request(output_dir, args.currency)
        print(json.dumps({"status": "stop_request_cleared", "currency": args.currency.upper(), "cleared": cleared}, indent=2, default=str), flush=True)
        return
    while True:
        stop_request = read_stop_request(output_dir, args.currency)
        if stop_request and not args.once:
            print(json.dumps({"status": "stopped_by_request", "currency": args.currency.upper(), "request": stop_request}, indent=2, default=str), flush=True)
            break
        record = run_once(args)
        report_path = write_record(output_dir, args.currency, record)
        print(json.dumps({"report": str(report_path), **_summary(record)}, indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(1, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an autonomous Deribit crypto-options trading agent.")
    parser.add_argument("--account-mode", choices=("testnet", "live"), default="testnet")
    parser.add_argument("--currency", choices=("BTC", "ETH"), default="ETH")
    parser.add_argument("--instrument-currency", choices=("BTC", "ETH", "USDC"), default=None, help="Deribit option instrument universe/account currency. Use USDC for live ETH_USDC options.")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--data-provider", default="alpaca")
    parser.add_argument("--data-interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--forecast-hours", default="1,2,4")
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--forecast-refresh-seconds", type=int, default=900)
    parser.add_argument("--max-training-rows", type=int, default=3500)
    parser.add_argument("--min-dte", type=int, default=1)
    parser.add_argument("--max-dte", type=int, default=14)
    parser.add_argument("--allow-0dte", action="store_true")
    parser.add_argument("--max-total-debit-usd", type=float, default=1500.0)
    parser.add_argument("--risk-budget-pct", type=float, default=None)
    parser.add_argument("--max-position-equity-pct", type=float, default=0.02)
    parser.add_argument("--max-spread-pct", type=float, default=None)
    parser.add_argument("--max-contracts", type=float, default=1.0)
    parser.add_argument("--min-contract-amount", type=float, default=None)
    parser.add_argument("--target-delta", type=float, default=0.45)
    parser.add_argument("--max-delta-distance", type=float, default=0.30)
    parser.add_argument("--greeks-mode", choices=("required", "off"), default="required", help="Use strict Greek-letter gates or disable Greek-based screening/scoring.")
    parser.add_argument("--max-theta-edge-ratio", type=float, default=0.75, help="Reject options when expected theta decay over the forecast horizon is too large versus delta-adjusted forecast edge.")
    parser.add_argument("--max-theta-premium-pct-per-day", type=float, default=0.35, help="Reject options when daily theta decay is too large versus option premium.")
    parser.add_argument("--min-option-volume", type=float, default=0.0)
    parser.add_argument("--enable-fibonacci", action=argparse.BooleanOptionalAction, default=True, help="Include Fibonacci swing/retracement/extension analysis in the forecast and option scoring.")
    parser.add_argument("--require-fibonacci-confluence", action="store_true", help="Block entries when Fibonacci analysis conflicts with the forecast direction.")
    parser.add_argument("--fibonacci-lookback-rows", type=int, default=720)
    parser.add_argument("--max-fibonacci-distance-pct", type=float, default=0.006)
    parser.add_argument("--enable-chart-patterns", action=argparse.BooleanOptionalAction, default=True, help="Detect named chart patterns and include them in option-entry decisions.")
    parser.add_argument("--block-chart-pattern-conflicts", action=argparse.BooleanOptionalAction, default=True, help="Block entries when a confirmed high-confidence chart pattern conflicts with the forecast.")
    parser.add_argument("--chart-pattern-lookback-rows", type=int, default=240)
    parser.add_argument("--chart-pattern-level-tolerance-pct", type=float, default=0.006)
    parser.add_argument("--chart-pattern-breakout-buffer-pct", type=float, default=0.0015)
    parser.add_argument("--chart-pattern-min-volume-ratio", type=float, default=1.20)
    parser.add_argument("--min-chart-pattern-confidence", type=float, default=0.70)
    parser.add_argument("--enable-market-regime-filter", action=argparse.BooleanOptionalAction, default=True, help="Block directional option entries unless the market is trending or breaking out.")
    parser.add_argument("--allow-range-edge-reversal-entry", action="store_true", help="Allow optional contrarian entries near support/resistance in range-bound markets. Default is to wait.")
    parser.add_argument("--market-regime-lookback-rows", type=int, default=120)
    parser.add_argument("--market-regime-breakout-buffer-pct", type=float, default=0.001)
    parser.add_argument("--market-regime-middle-zone-width", type=float, default=0.30)
    parser.add_argument("--min-trend-strength-pct", type=float, default=0.003)
    parser.add_argument("--enable-impulse-entry", action=argparse.BooleanOptionalAction, default=True, help="Allow early option entries when recent bars show a strong forecast-aligned impulse.")
    parser.add_argument("--impulse-lookback-bars", type=int, default=12)
    parser.add_argument("--min-impulse-move-pct", type=float, default=0.006)
    parser.add_argument("--min-impulse-directional-bars", type=int, default=7)
    parser.add_argument("--enable-late-entry-filter", action=argparse.BooleanOptionalAction, default=True, help="Block new option entries when the impulse/trend is already extended or showing reversal candles.")
    parser.add_argument("--max-late-entry-move-pct", type=float, default=0.018)
    parser.add_argument("--max-ema-extension-pct", type=float, default=0.010)
    parser.add_argument("--exhaustion-reversal-bars", type=int, default=2)
    parser.add_argument("--close-before-expiry-hours", type=float, default=12.0, help="Automatically close open option positions this many hours before expiry.")
    parser.add_argument("--expiry-warning-hours", type=float, default=24.0, help="Log and display a warning when open positions are this close to expiry.")
    parser.add_argument("--entry-expiry-buffer-hours", type=float, default=4.0, help="Extra buffer added to forecast horizon and close window when screening new option entries.")
    parser.add_argument("--target-moneyness", type=float, default=0.02)
    parser.add_argument("--max-moneyness-distance", type=float, default=0.12)
    parser.add_argument("--limit-price-offset-pct", type=float, default=0.03)
    parser.add_argument("--entry-order-policy", choices=("auto", "limit", "post_only_limit"), default="auto")
    parser.add_argument("--exit-order-policy", choices=("auto", "agent_managed"), default="auto")
    parser.add_argument("--stop-loss-pct", type=float, default=0.35)
    parser.add_argument("--take-profit-pct", type=float, default=0.55)
    parser.add_argument("--max-total-unrealized-loss-usd", type=float, default=None, help="Optional currency-level USD loss cutoff. Negative values are treated as their absolute value; omit the flag to disable.")
    parser.add_argument("--total-loss-close-mode", choices=("all", "losing_only"), default="all", help="When max total unrealized loss is breached, close all positions or only positions currently losing.")
    parser.add_argument("--max-total-unrealized-profit-usd", type=float, default=None, help="Optional currency-level USD profit target. If open option P/L converted to USD is >= value, close open option positions.")
    parser.add_argument("--total-profit-close-mode", choices=("all", "winning_only"), default="all", help="When max total unrealized profit is reached, close all positions or only positions currently winning.")
    parser.add_argument("--max-position-unrealized-loss-usd", type=float, default=None, help="Optional per-position USD loss cutoff. 0 closes any losing option position; negative values are treated as their absolute value; omit to disable.")
    parser.add_argument("--max-position-unrealized-profit-usd", type=float, default=None, help="Optional per-position USD profit target. 0 closes any winning option position; negative disables.")
    parser.add_argument("--take-profit-position-pl-usd", type=float, default=None, help="Per-position open-profit target used for profit taking and retrace protection. Defaults to max-position-unrealized-profit-usd when omitted.")
    parser.add_argument("--profit-retrace-from-peak-pct", type=float, default=0.20, help="After a position reaches the per-position profit target, close if it gives back this fraction of peak open profit.")
    parser.add_argument("--profit-close-limit-offset-pct", type=float, default=0.01, help="Normal profit-close sell limit below current option mark. Separate from emergency liquidation offset.")
    parser.add_argument("--enable-forecast-reversal-exit", action=argparse.BooleanOptionalAction, default=True, help="Close open calls when the refreshed forecast turns bearish, and close open puts when it turns bullish.")
    parser.add_argument("--min-reversal-edge-pct", type=float, default=0.001, help="Minimum absolute forecast move versus spot required before closing an opposite option position.")
    parser.add_argument("--abandon-entry-after-seconds", type=int, default=300)
    parser.add_argument("--allow-duplicate-contract-order", action="store_true")
    parser.add_argument("--max-open-option-orders", type=int, default=1)
    parser.add_argument("--max-open-option-positions", type=int, default=1, help="Block new entries when this many option positions are already open.")
    parser.add_argument("--max-open-option-contracts", type=float, default=2.0, help="Block new entries when existing option contract amount is at or above this value.")
    parser.add_argument("--max-open-option-premium-usd", type=float, default=300.0, help="Block new entries when current open option premium exposure is at or above this amount.")
    parser.add_argument("--allow-mixed-option-direction", action="store_true", help="Allow opening calls while puts are open, or puts while calls are open.")
    parser.add_argument("--max-daily-realized-loss-usd", type=float, default=None, help="Block new option entries for the rest of the UTC day when realized option P/L is at or below this loss.")
    parser.add_argument("--loss-cooldown-minutes", type=float, default=0.0, help="Block new entries for this many minutes after a losing option close.")
    parser.add_argument("--min-entry-expected-return-pct", type=float, default=0.0, help="Minimum absolute underlying forecast return required before opening a new option position.")
    parser.add_argument("--enable-feedback-loop", action=argparse.BooleanOptionalAction, default=True, help="Evaluate matured forecasts and block new entries when recent performance is weak.")
    parser.add_argument("--feedback-min-matured", type=int, default=5)
    parser.add_argument("--feedback-min-direction-accuracy", type=float, default=0.45)
    parser.add_argument("--feedback-max-abs-pct-error", type=float, default=0.06)
    parser.add_argument("--feedback-ledger-window", type=int, default=50)
    parser.add_argument("--liquidate-and-stop", action="store_true", help="Emergency command: request this currency agent to stop, cancel open option orders, and close all open option positions.")
    parser.add_argument("--stop-agent", action="store_true", help="Request this currency agent to stop without submitting liquidation orders.")
    parser.add_argument("--clear-stop-request", action="store_true", help="Clear a prior stop request so the agent can run again.")
    parser.add_argument("--liquidation-limit-offset-pct", type=float, default=0.05, help="Emergency liquidation sell limit below current option mark. Example 0.05 means current mark minus 5%.")
    parser.add_argument("--execute-paper-orders", action="store_true")
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-deribit-options-orders", action="store_true")
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/deribit_options_agent")
    parser.add_argument("--once", action="store_true")
    return parser


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    state = read_state(output_dir, args.currency)
    broker = _broker_for_args(args)
    now = datetime.now(UTC)
    instrument_currency = _instrument_currency(args)
    account = broker.account_summary(currency=instrument_currency)
    open_orders = _option_open_orders(broker.open_orders(currency=instrument_currency, kind="option"), args.currency, instrument_currency=instrument_currency)
    positions = _option_positions(broker.positions(currency=instrument_currency, kind="option"), args.currency, instrument_currency=instrument_currency)
    forecast_bundle = state.get("last_forecast")
    if forecast_bundle is None or _forecast_is_stale(forecast_bundle, now, int(args.forecast_refresh_seconds)):
        prices = _load_prices(args)
        plan = build_daily_trade_plan(
            prices,
            DailyTradeConfig(
                ticker=f"{args.currency.upper()}-USD",
                interval=args.data_interval,
                forecast_hours=tuple(float(value.strip()) for value in args.forecast_hours.split(",") if value.strip()),
                forecast_calendar="continuous_24_7",
                minimum_score_to_trade=1.5 if args.risk_profile == "aggressive" else 2.0,
            ),
        )
        forecast = _primary_forecast(plan)
        forecast["account_equity"] = _float_or_none(account.get("equity"))
        forecast["fibonacci_analysis"] = (
            build_fibonacci_analysis(
                prices,
                current_price=float(plan["latest_price"]),
                forecast=forecast,
                lookback_rows=int(args.fibonacci_lookback_rows),
                max_distance_pct=float(args.max_fibonacci_distance_pct),
            )
            if bool(args.enable_fibonacci)
            else {"enabled": False}
        )
        forecast["chart_pattern_analysis"] = (
            build_chart_pattern_analysis(
                prices,
                current_price=float(plan["latest_price"]),
                forecast=forecast,
                lookback_rows=int(args.chart_pattern_lookback_rows),
                level_tolerance_pct=float(args.chart_pattern_level_tolerance_pct),
                breakout_buffer_pct=float(args.chart_pattern_breakout_buffer_pct),
                min_volume_ratio=float(args.chart_pattern_min_volume_ratio),
            )
            if bool(args.enable_chart_patterns)
            else {"enabled": False}
        )
        forecast["curve_shape_analysis"] = build_curve_shape_analysis(
            prices,
            current_price=float(plan["latest_price"]),
            lookback_rows=120,
            short_window=9,
            long_window=21,
            impulse_bars=int(args.impulse_lookback_bars),
            range_lookback=int(args.market_regime_lookback_rows),
        )
        forecast["market_regime"] = (
            build_options_market_regime(
                prices,
                current_price=float(plan["latest_price"]),
                forecast=forecast,
                lookback_rows=int(args.market_regime_lookback_rows),
                breakout_buffer_pct=float(args.market_regime_breakout_buffer_pct),
                middle_zone_width=float(args.market_regime_middle_zone_width),
                min_trend_strength_pct=float(args.min_trend_strength_pct),
                allow_range_edge_reversal_entry=bool(args.allow_range_edge_reversal_entry),
                enable_impulse_entry=bool(args.enable_impulse_entry),
                impulse_lookback_bars=int(args.impulse_lookback_bars),
                min_impulse_move_pct=float(args.min_impulse_move_pct),
                min_impulse_directional_bars=int(args.min_impulse_directional_bars),
                enable_late_entry_filter=bool(args.enable_late_entry_filter),
                max_late_entry_move_pct=float(args.max_late_entry_move_pct),
                max_ema_extension_pct=float(args.max_ema_extension_pct),
                exhaustion_reversal_bars=int(args.exhaustion_reversal_bars),
            )
            if bool(args.enable_market_regime_filter)
            else {"enabled": False}
        )
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
        if "fibonacci_analysis" not in forecast:
            forecast["fibonacci_analysis"] = {"enabled": False, "status": "missing_from_cached_forecast"}
        if "chart_pattern_analysis" not in forecast:
            forecast["chart_pattern_analysis"] = {"enabled": False, "status": "missing_from_cached_forecast"}
        if "curve_shape_analysis" not in forecast:
            forecast["curve_shape_analysis"] = {"enabled": False, "status": "missing_from_cached_forecast"}
        if "market_regime" not in forecast:
            forecast["market_regime"] = {"enabled": False, "status": "missing_from_cached_forecast"}
    profile = risk_profile_for_name(args.risk_profile)
    config = DeribitOptionExecutionConfig(
        currency=instrument_currency,
        underlying_currency=args.currency.upper(),
        risk_profile=args.risk_profile,
        min_dte=args.min_dte,
        max_dte=args.max_dte,
        allow_0dte=bool(args.allow_0dte),
        max_total_debit_usd=float(args.max_total_debit_usd),
        risk_budget_pct=float(args.risk_budget_pct if args.risk_budget_pct is not None else profile.risk_budget_pct),
        max_position_equity_pct=float(args.max_position_equity_pct),
        max_spread_pct=float(args.max_spread_pct if args.max_spread_pct is not None else profile.options_max_spread_pct),
        max_contracts=float(args.max_contracts),
        min_contract_amount=args.min_contract_amount,
        target_delta=float(args.target_delta),
        max_delta_distance=float(args.max_delta_distance),
        greeks_mode=args.greeks_mode,
        max_theta_edge_ratio=float(args.max_theta_edge_ratio),
        max_theta_premium_pct_per_day=float(args.max_theta_premium_pct_per_day),
        min_option_volume=float(args.min_option_volume),
        enable_fibonacci=bool(args.enable_fibonacci),
        require_fibonacci_confluence=bool(args.require_fibonacci_confluence),
        max_fibonacci_distance_pct=float(args.max_fibonacci_distance_pct),
        enable_chart_patterns=bool(args.enable_chart_patterns),
        block_chart_pattern_conflicts=bool(args.block_chart_pattern_conflicts),
        min_chart_pattern_confidence=float(args.min_chart_pattern_confidence),
        enable_market_regime_filter=bool(args.enable_market_regime_filter),
        allow_range_edge_reversal_entry=bool(args.allow_range_edge_reversal_entry),
        market_regime_lookback_rows=int(args.market_regime_lookback_rows),
        market_regime_breakout_buffer_pct=float(args.market_regime_breakout_buffer_pct),
        market_regime_middle_zone_width=float(args.market_regime_middle_zone_width),
        min_trend_strength_pct=float(args.min_trend_strength_pct),
        enable_impulse_entry=bool(args.enable_impulse_entry),
        impulse_lookback_bars=int(args.impulse_lookback_bars),
        min_impulse_move_pct=float(args.min_impulse_move_pct),
        min_impulse_directional_bars=int(args.min_impulse_directional_bars),
        enable_late_entry_filter=bool(args.enable_late_entry_filter),
        max_late_entry_move_pct=float(args.max_late_entry_move_pct),
        max_ema_extension_pct=float(args.max_ema_extension_pct),
        exhaustion_reversal_bars=int(args.exhaustion_reversal_bars),
        close_before_expiry_hours=float(args.close_before_expiry_hours),
        entry_expiry_buffer_hours=float(args.entry_expiry_buffer_hours),
        target_moneyness=float(args.target_moneyness),
        max_moneyness_distance=float(args.max_moneyness_distance),
        limit_price_offset_pct=float(args.limit_price_offset_pct),
        stop_loss_pct=float(args.stop_loss_pct),
        take_profit_pct=float(args.take_profit_pct),
        abandon_entry_after_seconds=int(args.abandon_entry_after_seconds),
        entry_order_policy=args.entry_order_policy,
        exit_order_policy=args.exit_order_policy,
    )
    underlying_price = float(plan["latest_price"])
    feedback_context = (
        update_feedback_loop(
            output_dir=output_dir,
            currency=args.currency,
            now=now,
            actual_price=underlying_price,
            min_matured=int(args.feedback_min_matured),
            min_direction_accuracy=float(args.feedback_min_direction_accuracy),
            max_abs_pct_error=float(args.feedback_max_abs_pct_error),
            window=int(args.feedback_ledger_window),
        )
        if bool(args.enable_feedback_loop)
        else {"enabled": False}
    )
    performance_context = _option_performance_context(broker=broker, currency=args.currency, instrument_currency=instrument_currency, now=now)
    trade_plan = build_deribit_option_trade_plan(
        broker=broker,
        currency=instrument_currency,
        underlying_price_usd=underlying_price,
        forecast=forecast,
        account=account,
        config=config,
    )
    management_actions = manage_existing_orders_and_positions(
        broker=broker,
        args=args,
        open_orders=open_orders,
        positions=positions,
        state=state,
        now=now,
        underlying_price_usd=underlying_price,
        forecast=forecast,
    )
    stop_request = read_stop_request(output_dir, args.currency)
    if stop_request:
        management_actions.append({"action": "entry_blocked_by_stop_request", "currency": args.currency.upper(), "request": stop_request})
        trade_plan = {"action": "hold", "reason": "currency_stop_requested"}
    if _protective_close_triggered(management_actions):
        state["wait_until_next_forecast_after_close"] = {
            "created_at_utc": now.isoformat(),
            "forecast_created_at_utc": forecast_bundle.get("created_at_utc"),
            "reason": "protective_close_triggered",
        }
        trade_plan = {"action": "hold", "reason": "protective_close_triggered_this_cycle"}
    elif _waiting_until_next_forecast(state, forecast_bundle):
        trade_plan = {"action": "hold", "reason": "waiting_until_next_forecast_after_protective_close"}
    elif (feedback_context.get("blocks") or []) and trade_plan.get("action") == "buy_option":
        trade_plan = {"action": "hold", "reason": "feedback_loop_blocked_entry", "feedback_blocks": feedback_context.get("blocks")}
    elif trade_plan.get("action") == "buy_option":
        entry_quality = _entry_quality_context(args=args, forecast=forecast, performance_context=performance_context, now=now)
        if entry_quality.get("blocks"):
            trade_plan = {"action": "hold", "reason": "entry_quality_blocked", "entry_quality_blocks": entry_quality.get("blocks")}
    else:
        state.pop("wait_until_next_forecast_after_close", None)
    entry_quality_context = _entry_quality_context(args=args, forecast=forecast, performance_context=performance_context, now=now)
    execution_blocks = execution_block_reasons(args=args, trade_plan=trade_plan, open_orders=open_orders, positions=positions, underlying_price_usd=underlying_price)
    order_result = {"submitted": False, "reason": "execution_disabled" if not _orders_enabled(args) else "execution_blocked", "blocks": execution_blocks}
    if _orders_enabled(args) and trade_plan.get("action") == "buy_option" and not execution_blocks:
        label = f"codex-{args.currency.upper()}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
        try:
            order_result = {
                "submitted": True,
                "order": submit_deribit_limit_order(broker, trade_plan["order"], label=label),
            }
        except RuntimeError as exc:
            order_result = {"submitted": False, "reason": "broker_rejected_deribit_order", "error": str(exc)}
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "venue": f"deribit_{args.account_mode}",
        "account_mode": args.account_mode,
        "currency": args.currency.upper(),
        "instrument_currency": instrument_currency,
        "ticker": f"{args.currency.upper()}-USD",
        "risk_profile": profile.to_dict(),
        "account": _safe_account(account),
        "forecast_created_at_utc": forecast_bundle.get("created_at_utc"),
        "forecast_cache_status": _forecast_cache_status(forecast_bundle, now, int(args.forecast_refresh_seconds)),
        "price_rows": forecast_bundle.get("price_rows"),
        "forecast_plan": plan,
        "selected_forecast": forecast,
        "option_execution_config": config.__dict__,
        "risk_control_config": _risk_control_config(args),
        "feedback_context": feedback_context,
        "performance_context": performance_context,
        "entry_quality_context": entry_quality_context,
        "option_trade_plan": trade_plan,
        "open_option_orders": open_orders,
        "option_positions": positions,
        "management_actions": management_actions,
        "execution_blocks": execution_blocks,
        "execute_paper_orders": bool(args.execute_paper_orders),
        "execute_live_orders": bool(args.execute_live_orders),
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
    write_state(output_dir, args.currency, state)
    append_log(output_dir, args.currency, record)
    if bool(args.enable_feedback_loop):
        record["feedback_ledger_append"] = append_decision_to_ledger(output_dir=output_dir, currency=args.currency, record=record)
        write_record(output_dir, args.currency, record)
    return record


def _option_performance_context(
    *,
    broker: Any,
    currency: str,
    instrument_currency: str,
    now: datetime,
    trade_count: int = 1000,
) -> dict[str, Any]:
    if not hasattr(broker, "user_trades"):
        return {"enabled": False, "status": "broker_trade_history_unavailable"}
    try:
        trades = broker.user_trades(currency=instrument_currency, kind="option", count=trade_count)
    except Exception as exc:
        return {"enabled": True, "status": "trade_history_error", "error": f"{type(exc).__name__}: {exc}"}
    day_start = datetime(now.year, now.month, now.day, tzinfo=UTC)
    closing_trades: list[dict[str, Any]] = []
    for trade in trades if isinstance(trades, list) else []:
        if not isinstance(trade, dict):
            continue
        instrument = str(trade.get("instrument_name") or "").upper()
        if not instrument.startswith(_instrument_prefix(currency, instrument_currency)):
            continue
        ts = _timestamp_ms_to_datetime(_float_or_none(trade.get("timestamp")))
        if ts is None or ts < day_start or ts > now:
            continue
        direction = str(trade.get("direction") or "").lower()
        if direction != "sell" and not bool(trade.get("reduce_only")):
            continue
        pl = _float_or_none(trade.get("profit_loss")) or 0.0
        closing_trades.append(
            {
                "timestamp_utc": ts.isoformat(),
                "instrument_name": trade.get("instrument_name"),
                "direction": direction,
                "amount": _float_or_none(trade.get("amount")),
                "price": _float_or_none(trade.get("price")),
                "profit_loss_usd": pl,
                "reduce_only": trade.get("reduce_only"),
                "label": trade.get("label"),
                "order_id": trade.get("order_id"),
            }
        )
    closing_trades.sort(key=lambda item: str(item.get("timestamp_utc") or ""))
    wins = [trade for trade in closing_trades if (_float_or_none(trade.get("profit_loss_usd")) or 0.0) > 0]
    losses = [trade for trade in closing_trades if (_float_or_none(trade.get("profit_loss_usd")) or 0.0) < 0]
    latest_losing_close = losses[-1] if losses else None
    latest_losing_close_age_minutes = None
    if latest_losing_close:
        latest_ts = _parse_time(latest_losing_close.get("timestamp_utc"))
        if latest_ts is not None:
            latest_losing_close_age_minutes = max(0.0, (now - latest_ts).total_seconds() / 60.0)
    return {
        "enabled": True,
        "status": "ok",
        "day_start_utc": day_start.isoformat(),
        "closing_trade_count": len(closing_trades),
        "winning_close_count": len(wins),
        "losing_close_count": len(losses),
        "gross_wins_usd": round(sum(_float_or_none(trade.get("profit_loss_usd")) or 0.0 for trade in wins), 6),
        "gross_losses_usd": round(sum(_float_or_none(trade.get("profit_loss_usd")) or 0.0 for trade in losses), 6),
        "net_realized_options_pl_usd": round(sum(_float_or_none(trade.get("profit_loss_usd")) or 0.0 for trade in closing_trades), 6),
        "latest_losing_close": latest_losing_close,
        "latest_losing_close_age_minutes": None if latest_losing_close_age_minutes is None else round(latest_losing_close_age_minutes, 2),
        "recent_closing_trades": closing_trades[-10:],
    }


def _entry_quality_context(
    *,
    args: argparse.Namespace,
    forecast: dict[str, Any],
    performance_context: dict[str, Any],
    now: datetime,
) -> dict[str, Any]:
    blocks: list[str] = []
    expected_return = _float_or_none(forecast.get("expected_return"))
    min_expected = _float_or_none(getattr(args, "min_entry_expected_return_pct", None))
    if min_expected is not None and min_expected > 0:
        if expected_return is None or abs(float(expected_return)) < float(min_expected):
            blocks.append("entry_expected_return_below_min")
    max_daily_loss = _abs_or_none(getattr(args, "max_daily_realized_loss_usd", None))
    net_realized = _float_or_none(performance_context.get("net_realized_options_pl_usd"))
    if max_daily_loss is not None and net_realized is not None and net_realized <= -float(max_daily_loss):
        blocks.append("daily_realized_loss_limit_reached")
    cooldown_minutes = _float_or_none(getattr(args, "loss_cooldown_minutes", None)) or 0.0
    latest_loss_age = _float_or_none(performance_context.get("latest_losing_close_age_minutes"))
    if cooldown_minutes > 0 and latest_loss_age is not None and latest_loss_age < cooldown_minutes:
        blocks.append("loss_cooldown_active")
    return {
        "enabled": True,
        "checked_at_utc": now.isoformat(),
        "blocks": blocks,
        "expected_return": expected_return,
        "min_entry_expected_return_pct": min_expected,
        "max_daily_realized_loss_usd": max_daily_loss,
        "loss_cooldown_minutes": cooldown_minutes,
        "latest_losing_close_age_minutes": latest_loss_age,
        "net_realized_options_pl_usd": net_realized,
    }


def execution_block_reasons(
    *,
    args: argparse.Namespace,
    trade_plan: dict[str, Any],
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]] | None = None,
    underlying_price_usd: float | None = None,
) -> list[str]:
    reasons = []
    if trade_plan.get("action") != "buy_option":
        reasons.append(str(trade_plan.get("reason") or "no_buy_option_plan"))
    if len(open_orders) >= int(args.max_open_option_orders):
        reasons.append("max_open_option_orders_reached")
    planned = ((trade_plan.get("order") or {}).get("instrument_name") or "").upper()
    active_positions = [position for position in (positions or []) if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    if len(active_positions) >= int(getattr(args, "max_open_option_positions", 1)):
        reasons.append("max_open_option_positions_reached")
    existing_contracts = sum(abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0) for position in active_positions)
    if existing_contracts >= float(getattr(args, "max_open_option_contracts", 2.0)):
        reasons.append("max_open_option_contracts_reached")
    open_premium_usd = _open_option_premium_usd(active_positions, underlying_price_usd)
    max_open_premium = _float_or_none(getattr(args, "max_open_option_premium_usd", None))
    if max_open_premium is not None and open_premium_usd is not None and open_premium_usd >= float(max_open_premium):
        reasons.append("max_open_option_premium_usd_reached")
    if planned and active_positions and not bool(getattr(args, "allow_mixed_option_direction", False)):
        planned_type = _option_type_from_instrument_name(planned)
        existing_types = {_option_type_from_instrument_name(str(position.get("instrument_name") or "")) for position in active_positions}
        existing_types.discard(None)
        if planned_type and existing_types and planned_type not in existing_types:
            reasons.append("mixed_option_direction_blocked")
    if planned and not args.allow_duplicate_contract_order:
        for order in open_orders:
            if str(order.get("instrument_name") or "").upper() == planned:
                reasons.append("same_contract_order_already_open")
                break
    return list(dict.fromkeys(reasons))


def _open_option_premium_usd(positions: list[dict[str, Any]], underlying_price_usd: float | None) -> float | None:
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return None
    total_usd = 0.0
    for position in positions:
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        mark = _float_or_none(position.get("mark_price"))
        average = _float_or_none(position.get("average_price"))
        price = mark if mark is not None else average
        if amount <= 0 or price is None:
            continue
        multiplier = 1.0 if _is_usdc_settled_instrument(str(position.get("instrument_name") or "")) else float(underlying_price_usd)
        total_usd += amount * price * multiplier
    return total_usd


def _option_type_from_instrument_name(instrument_name: str) -> str | None:
    suffix = str(instrument_name or "").upper().rsplit("-", 1)[-1]
    if suffix == "C":
        return "call"
    if suffix == "P":
        return "put"
    return None


def manage_existing_orders_and_positions(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
    underlying_price_usd: float | None = None,
    forecast: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    actions = _abandon_stale_entry_orders(broker=broker, args=args, open_orders=open_orders, state=state, now=now)
    if not _orders_enabled(args):
        actions.extend(
            _manage_position_unrealized_guards(
                broker=broker,
                args=args,
                open_orders=open_orders,
                positions=positions,
                state=state,
                now=now,
                underlying_price_usd=underlying_price_usd,
            )
        )
        actions.extend(_manage_expiry_risk(broker=broker, args=args, open_orders=open_orders, positions=positions, now=now))
        return actions
    total_loss_actions = _manage_total_unrealized_loss(
        broker=broker,
        args=args,
        open_orders=open_orders,
        positions=positions,
        now=now,
        underlying_price_usd=underlying_price_usd,
    )
    if total_loss_actions:
        actions.extend(total_loss_actions)
        return actions
    total_profit_actions = _manage_total_unrealized_profit(
        broker=broker,
        args=args,
        open_orders=open_orders,
        positions=positions,
        now=now,
        underlying_price_usd=underlying_price_usd,
    )
    if total_profit_actions:
        actions.extend(total_profit_actions)
        return actions
    position_guard_actions = _manage_position_unrealized_guards(
        broker=broker,
        args=args,
        open_orders=open_orders,
        positions=positions,
        state=state,
        now=now,
        underlying_price_usd=underlying_price_usd,
    )
    if position_guard_actions:
        actions.extend(position_guard_actions)
        if _protective_close_triggered(position_guard_actions):
            return actions
    reversal_actions = _manage_forecast_reversal_exit(
        broker=broker,
        args=args,
        open_orders=open_orders,
        positions=positions,
        now=now,
        underlying_price_usd=underlying_price_usd,
        forecast=forecast or {},
    )
    if reversal_actions:
        actions.extend(reversal_actions)
        if any(str(action.get("action") or "").startswith("forecast_reversal_position_close") for action in reversal_actions):
            return actions
    expiry_actions = _manage_expiry_risk(broker=broker, args=args, open_orders=open_orders, positions=positions, now=now)
    if expiry_actions:
        actions.extend(expiry_actions)
        if any(str(action.get("action") or "").startswith("expiry_position_close") for action in expiry_actions):
            return actions
    active_trade = state.get("active_trade") or {}
    for position in positions:
        instrument = str(position.get("instrument_name") or "")
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        if not instrument or amount <= 0:
            continue
        sell_orders = _sell_orders_for_instrument(open_orders, instrument)
        book = broker.order_book(instrument, depth=5)
        quote = _book_quote(book)
        mark = _float_or_none(book.get("mark_price")) or quote.get("bid") or quote.get("ask")
        entry_price = _position_entry_price(position) or _entry_price_from_state(active_trade)
        if entry_price is None or mark is None:
            continue
        current_exit_plan = build_deribit_exit_plan(
            instrument_name=instrument,
            entry_limit_price_base=entry_price,
            amount=amount,
            config=DeribitOptionExecutionConfig(
                currency=args.currency.upper(),
                stop_loss_pct=float(args.stop_loss_pct),
                take_profit_pct=float(args.take_profit_pct),
            ),
        )
        take_profit = current_exit_plan["take_profit"]
        stop_loss = current_exit_plan["stop_loss"]
        selected_exit = None
        reason = None
        if mark >= float(take_profit["price"]):
            selected_exit = dict(take_profit)
            selected_exit["price"] = max(float(quote.get("bid") or mark), float(take_profit["price"]))
            reason = "take_profit_triggered"
        elif mark <= float(stop_loss["price"]):
            selected_exit = dict(stop_loss)
            selected_exit["price"] = max(float(quote.get("bid") or mark), 1e-8)
            reason = "stop_loss_triggered"
        if selected_exit is None:
            stale_profit_action = _replace_stale_profit_order_if_config_changed(
                broker=broker,
                args=args,
                instrument=instrument,
                amount=amount,
                sell_orders=sell_orders,
                take_profit=take_profit,
                entry_price=entry_price,
                mark=mark,
                now=now,
            )
            if stale_profit_action:
                actions.extend(stale_profit_action)
                continue
            actions.append(
                {
                    "action": "hold_position",
                    "instrument_name": instrument,
                    "mark_price": mark,
                    "take_profit": take_profit["price"],
                    "stop_loss": stop_loss["price"],
                    "sell_order_count": len(sell_orders),
                }
            )
            continue
        actions.extend(
            _replace_sell_orders_with_exit(
                broker=broker,
                args=args,
                instrument=instrument,
                amount=amount,
                sell_orders=sell_orders,
                selected_exit=selected_exit,
                reason=str(reason),
                now=now,
            )
        )
    return actions


def _manage_expiry_risk(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    active_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    if not active_positions:
        return []
    close_before_hours = max(0.0, float(getattr(args, "close_before_expiry_hours", 12.0)))
    warning_hours = max(close_before_hours, float(getattr(args, "expiry_warning_hours", 24.0)))
    instruments = _instrument_metadata_by_name(broker=broker, currency=_instrument_currency(args))
    actions: list[dict[str, Any]] = []
    for position in active_positions:
        instrument = str(position.get("instrument_name") or "")
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        expiry = _position_expiry_utc(position, instruments)
        hours_to_expiry = None if expiry is None else (expiry - now).total_seconds() / 3600.0
        if not instrument or amount <= 0 or hours_to_expiry is None:
            actions.append({"action": "expiry_risk_skipped_missing_instrument_amount_or_expiry", "instrument_name": instrument, "amount": amount})
            continue
        expiry_payload = {
            "instrument_name": instrument,
            "expiration_utc": expiry.isoformat(),
            "hours_to_expiry": round(hours_to_expiry, 2),
            "close_before_expiry_hours": close_before_hours,
            "expiry_warning_hours": warning_hours,
        }
        if hours_to_expiry <= close_before_hours:
            book = broker.order_book(instrument, depth=5)
            quote = _book_quote(book)
            mark = _float_or_none(book.get("mark_price")) or quote.get("bid") or quote.get("ask") or _float_or_none(position.get("mark_price"))
            if mark is None:
                actions.append({"action": "expiry_position_close_skipped_missing_mark", **expiry_payload})
                continue
            tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument, price=mark)
            close_limit = _deribit_liquidation_limit_price(mark, float(getattr(args, "liquidation_limit_offset_pct", 0.05)), tick_size=tick_size)
            selected_exit = {
                "instrument_name": instrument,
                "side": "sell",
                "type": "limit",
                "amount": amount,
                "price": close_limit,
                "reduce_only": True,
            }
            sell_orders = _sell_orders_for_instrument(open_orders, instrument)
            if not _orders_enabled(args):
                actions.append({"action": "would_close_position_before_expiry", "amount": amount, "mark_price": round(float(mark), 8), "close_limit_price": close_limit, **expiry_payload})
                continue
            actions.extend(
                _replace_sell_orders_with_exit(
                    broker=broker,
                    args=args,
                    instrument=instrument,
                    amount=amount,
                    sell_orders=sell_orders,
                    selected_exit=selected_exit,
                    reason="expiry_position_close_triggered",
                    now=now,
                    extra={"mark_price": round(float(mark), 8), "close_limit_price": close_limit, **expiry_payload},
                )
            )
        elif hours_to_expiry <= warning_hours:
            actions.append({"action": "expiry_position_warning", **expiry_payload})
    return actions


def _replace_stale_profit_order_if_config_changed(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    instrument: str,
    amount: float,
    sell_orders: list[dict[str, Any]],
    take_profit: dict[str, Any],
    entry_price: float,
    mark: float,
    now: datetime,
) -> list[dict[str, Any]]:
    if not sell_orders:
        return []
    desired_price = float(take_profit["price"])
    existing_prices = [_float_or_none(order.get("price")) for order in sell_orders]
    existing_sell_limit = min((price for price in existing_prices if price is not None), default=None)
    if existing_sell_limit is not None and existing_sell_limit <= desired_price:
        return []
    replacement = {
        **take_profit,
        "amount": amount,
        "price": desired_price,
    }
    return _replace_sell_orders_with_exit(
        broker=broker,
        args=args,
        instrument=instrument,
        amount=amount,
        sell_orders=sell_orders,
        selected_exit=replacement,
        reason="exit_config_changed_replacing_profit_order",
        now=now,
        extra={
            "entry_price": round(float(entry_price), 8),
            "mark_price": round(float(mark), 8),
            "old_best_sell_price": existing_sell_limit,
            "new_take_profit_price": desired_price,
        },
    )


def _replace_sell_orders_with_exit(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    instrument: str,
    amount: float,
    sell_orders: list[dict[str, Any]],
    selected_exit: dict[str, Any],
    reason: str,
    now: datetime,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = [
        {
            "action": reason,
            "instrument_name": instrument,
            "amount": amount,
            "replacement_price": selected_exit.get("price"),
            **(extra or {}),
        }
    ]
    for order in sell_orders:
        order_id = order.get("order_id") or order.get("id")
        if not order_id:
            continue
        try:
            cancel_result = broker.cancel_order(str(order_id))
            actions.append({"action": "cancelled_existing_deribit_exit", "instrument_name": instrument, "order_id": order_id, "result": cancel_result})
        except RuntimeError as exc:
            actions.append({"action": "cancel_existing_deribit_exit_failed", "instrument_name": instrument, "order_id": order_id, "error": str(exc)})
            return actions
    actions.extend(_submit_exit_with_reduce_only_fallback(broker=broker, args=args, instrument=instrument, amount=amount, selected_exit=selected_exit, reason=reason, now=now))
    return actions


def _submit_exit_with_reduce_only_fallback(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    instrument: str,
    amount: float,
    selected_exit: dict[str, Any],
    reason: str,
    now: datetime,
) -> list[dict[str, Any]]:
    try:
        order = submit_deribit_limit_order(
            broker,
            {**selected_exit, "amount": min(amount, float(selected_exit["amount"]))},
            label=f"codex-exit-{args.currency.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
        )
        return [{"action": "submitted_reduce_only_exit", "reason": reason, "instrument_name": instrument, "order": order}]
    except RuntimeError as exc:
        fallback_order = {**selected_exit, "amount": min(amount, float(selected_exit["amount"])), "reduce_only": False}
        try:
            order = submit_deribit_limit_order(
                broker,
                fallback_order,
                label=f"codex-exit-fallback-{args.currency.upper()}-{now.strftime('%Y%m%d%H%M%S')}",
            )
            return [
                {
                    "action": "submitted_exit_without_reduce_only_after_rejection",
                    "reason": reason,
                    "instrument_name": instrument,
                    "order": order,
                    "rejection": str(exc),
                    "safety_note": "Fallback is only used for an existing long position amount and never for more than the current position size.",
                }
            ]
        except RuntimeError as fallback_exc:
            return [{"action": "exit_order_rejected", "reason": reason, "instrument_name": instrument, "error": str(exc), "fallback_error": str(fallback_exc)}]


def _sell_orders_for_instrument(open_orders: list[dict[str, Any]], instrument: str) -> list[dict[str, Any]]:
    return [
        order
        for order in open_orders
        if str(order.get("instrument_name") or "") == instrument
        and str(order.get("direction") or order.get("side") or "").lower() == "sell"
    ]


def _cancel_open_option_orders(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    cancel_action: str,
) -> list[dict[str, Any]]:
    if not open_orders:
        return []
    if not _orders_enabled(args):
        return [{"action": f"would_{cancel_action}", "currency": args.currency.upper(), "order_count": len(open_orders)}]
    actions: list[dict[str, Any]] = []
    for order in open_orders:
        order_id = order.get("order_id") or order.get("id")
        if not order_id:
            continue
        try:
            result = broker.cancel_order(str(order_id))
            actions.append(
                {
                    "action": cancel_action,
                    "currency": args.currency.upper(),
                    "order_id": order_id,
                    "instrument_name": order.get("instrument_name"),
                    "direction": order.get("direction") or order.get("side"),
                    "result": result,
                }
            )
        except RuntimeError as exc:
            actions.append(
                {
                    "action": f"{cancel_action}_failed",
                    "currency": args.currency.upper(),
                    "order_id": order_id,
                    "instrument_name": order.get("instrument_name"),
                    "error": str(exc),
                }
            )
    return actions


def _manage_total_unrealized_loss(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    now: datetime,
    underlying_price_usd: float | None,
) -> list[dict[str, Any]]:
    threshold = _float_or_none(getattr(args, "max_total_unrealized_loss_usd", None))
    if threshold is None:
        return []
    threshold = abs(float(threshold))
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return []
    open_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    if not open_positions:
        return []
    total_pl_base = sum(_position_pl_value(position) or 0.0 for position in open_positions)
    total_pl_usd = sum(_position_pl_usd(position, underlying_price_usd) or 0.0 for position in open_positions)
    loss_cutoff = -float(threshold)
    if float(threshold) == 0:
        if total_pl_usd >= 0:
            return []
    elif total_pl_usd > loss_cutoff:
        return []
    close_mode = str(getattr(args, "total_loss_close_mode", "all"))
    actions: list[dict[str, Any]] = [
        {
            "action": "max_total_unrealized_loss_usd_triggered",
            "currency": args.currency.upper(),
            "total_unrealized_pl_base": round(total_pl_base, 8),
            "total_unrealized_pl_usd": round(total_pl_usd, 2),
            "loss_cutoff_usd": round(loss_cutoff, 2),
            "position_count": len(open_positions),
            "close_mode": close_mode,
        }
    ]
    positions_to_close = _positions_for_total_loss_close(open_positions, close_mode)
    if not positions_to_close:
        actions.append({"action": "total_loss_no_positions_match_close_mode", "close_mode": close_mode})
        return actions
    for position in positions_to_close:
        instrument = str(position.get("instrument_name") or "")
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        mark = _float_or_none(position.get("mark_price"))
        if not instrument or amount <= 0 or mark is None:
            actions.append({"action": "total_loss_close_skipped_missing_mark_or_amount", "instrument_name": instrument, "amount": amount})
            continue
        sell_orders = _sell_orders_for_instrument(open_orders, instrument)
        tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument, price=mark)
        close_limit = _deribit_liquidation_limit_price(mark, float(args.liquidation_limit_offset_pct), tick_size=tick_size)
        selected_exit = {
            "instrument_name": instrument,
            "side": "sell",
            "type": "limit",
            "amount": amount,
            "price": close_limit,
            "reduce_only": True,
        }
        actions.extend(
            _replace_sell_orders_with_exit(
                broker=broker,
                args=args,
                instrument=instrument,
                amount=amount,
                sell_orders=sell_orders,
                selected_exit=selected_exit,
                reason="total_loss_position_close_triggered",
                now=now,
                extra={
                    "mark_price": round(mark, 8),
                    "close_limit_price": close_limit,
                    "position_unrealized_pl_base": _float_or_none(position.get("floating_profit_loss")),
                    "position_unrealized_pl_usd": None if _position_pl_usd(position, underlying_price_usd) is None else round(float(_position_pl_usd(position, underlying_price_usd)), 2),
                },
            )
        )
    return actions


def _manage_total_unrealized_profit(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    now: datetime,
    underlying_price_usd: float | None,
) -> list[dict[str, Any]]:
    threshold = _float_or_none(getattr(args, "max_total_unrealized_profit_usd", None))
    if threshold is None or threshold <= 0:
        return []
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return []
    open_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    if not open_positions:
        return []
    total_pl_base = sum(_position_pl_value(position) or 0.0 for position in open_positions)
    total_pl_usd = sum(_position_pl_usd(position, underlying_price_usd) or 0.0 for position in open_positions)
    if total_pl_usd < float(threshold):
        return []
    close_mode = str(getattr(args, "total_profit_close_mode", "all"))
    actions: list[dict[str, Any]] = [
        {
            "action": "max_total_unrealized_profit_usd_triggered",
            "currency": args.currency.upper(),
            "total_unrealized_pl_base": round(total_pl_base, 8),
            "total_unrealized_pl_usd": round(total_pl_usd, 2),
            "profit_target_usd": round(float(threshold), 2),
            "position_count": len(open_positions),
            "close_mode": close_mode,
        }
    ]
    positions_to_close = _positions_for_total_profit_close(open_positions, close_mode)
    if not positions_to_close:
        actions.append({"action": "total_profit_no_positions_match_close_mode", "close_mode": close_mode})
        return actions
    for position in positions_to_close:
        instrument = str(position.get("instrument_name") or "")
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        mark = _float_or_none(position.get("mark_price"))
        if not instrument or amount <= 0 or mark is None:
            actions.append({"action": "total_profit_close_skipped_missing_mark_or_amount", "instrument_name": instrument, "amount": amount})
            continue
        sell_orders = _sell_orders_for_instrument(open_orders, instrument)
        tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument, price=mark)
        close_limit = _deribit_liquidation_limit_price(mark, float(getattr(args, "profit_close_limit_offset_pct", 0.01)), tick_size=tick_size)
        selected_exit = {
            "instrument_name": instrument,
            "side": "sell",
            "type": "limit",
            "amount": amount,
            "price": close_limit,
            "reduce_only": True,
        }
        pl_base = _position_pl_value(position)
        actions.extend(
            _replace_sell_orders_with_exit(
                broker=broker,
                args=args,
                instrument=instrument,
                amount=amount,
                sell_orders=sell_orders,
                selected_exit=selected_exit,
                reason="total_profit_position_close_triggered",
                now=now,
                extra={
                    "mark_price": round(mark, 8),
                    "close_limit_price": close_limit,
                    "profit_close_limit_offset_pct": float(getattr(args, "profit_close_limit_offset_pct", 0.01)),
                    "position_unrealized_pl_base": pl_base,
                    "position_unrealized_pl_usd": None if _position_pl_usd(position, underlying_price_usd) is None else round(float(_position_pl_usd(position, underlying_price_usd)), 2),
                },
            )
        )
    return actions


def _manage_position_unrealized_guards(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
    underlying_price_usd: float | None,
) -> list[dict[str, Any]]:
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return []
    loss_threshold = _float_or_none(getattr(args, "max_position_unrealized_loss_usd", None))
    profit_threshold = _float_or_none(getattr(args, "max_position_unrealized_profit_usd", None))
    if loss_threshold is not None:
        loss_threshold = abs(float(loss_threshold))
    loss_enabled = loss_threshold is not None
    profit_enabled = profit_threshold is not None and profit_threshold >= 0
    profit_peak_target = _float_or_none(getattr(args, "take_profit_position_pl_usd", None))
    if profit_peak_target is None:
        profit_peak_target = profit_threshold
    retrace_pct = max(0.0, min(1.0, float(getattr(args, "profit_retrace_from_peak_pct", 0.20))))
    retrace_enabled = profit_peak_target is not None and profit_peak_target > 0 and retrace_pct > 0
    if not loss_enabled and not profit_enabled and not retrace_enabled:
        return []
    actions: list[dict[str, Any]] = []
    open_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    _prune_position_profit_peaks(state, open_positions)
    for position in open_positions:
        pl_base = _position_pl_value(position)
        if pl_base is None:
            continue
        pl_usd = _position_pl_usd(position, underlying_price_usd)
        if pl_usd is None:
            continue
        profit_peak = _update_position_profit_peak(state, position=position, underlying_price_usd=underlying_price_usd, now=now)
        peak_profit_usd = _float_or_none((profit_peak or {}).get("peak_unrealized_pl_usd"))
        trigger_reason = None
        threshold_payload: dict[str, Any] = {}
        if loss_enabled:
            loss_cutoff = -abs(float(loss_threshold))
            if (float(loss_threshold) == 0 and pl_usd < 0) or (float(loss_threshold) > 0 and pl_usd <= loss_cutoff):
                trigger_reason = "position_loss_position_close_triggered"
                threshold_payload = {"position_loss_cutoff_usd": round(loss_cutoff, 2)}
        if trigger_reason is None and profit_enabled:
            if (float(profit_threshold) == 0 and pl_usd > 0) or (float(profit_threshold) > 0 and pl_usd >= float(profit_threshold)):
                trigger_reason = "position_profit_position_close_triggered"
                threshold_payload = {"position_profit_target_usd": round(float(profit_threshold), 2)}
        if trigger_reason is None and retrace_enabled and peak_profit_usd is not None and peak_profit_usd >= float(profit_peak_target):
            protected_profit_usd = peak_profit_usd * (1.0 - retrace_pct)
            if pl_usd <= protected_profit_usd:
                trigger_reason = "position_profit_retrace_close_triggered"
                threshold_payload = {
                    "take_profit_position_pl_usd": round(float(profit_peak_target), 2),
                    "peak_unrealized_pl_usd": round(float(peak_profit_usd), 2),
                    "protected_profit_usd": round(float(protected_profit_usd), 2),
                    "profit_retrace_from_peak_pct": round(float(retrace_pct), 4),
                }
        if trigger_reason is None:
            actions.append(
                {
                    "action": "position_unrealized_guard_not_triggered",
                    "instrument_name": position.get("instrument_name"),
                    "position_unrealized_pl_base": round(float(pl_base), 8),
                    "position_unrealized_pl_usd": round(float(pl_usd), 2),
                    "loss_guard_usd": loss_threshold,
                    "profit_guard_usd": profit_threshold,
                    "take_profit_position_pl_usd": profit_peak_target,
                    "peak_unrealized_pl_usd": None if peak_profit_usd is None else round(float(peak_profit_usd), 2),
                    "profit_retrace_from_peak_pct": retrace_pct,
                }
            )
            continue
        is_profit_close = trigger_reason in {"position_profit_position_close_triggered", "position_profit_retrace_close_triggered"}
        actions.extend(
            _close_position_with_limit(
                broker=broker,
                args=args,
                open_orders=open_orders,
                position=position,
                now=now,
                reason=trigger_reason,
                limit_offset_pct=float(getattr(args, "profit_close_limit_offset_pct", 0.01)) if is_profit_close else None,
                extra={
                    "position_unrealized_pl_base": round(float(pl_base), 8),
                    "position_unrealized_pl_usd": round(float(pl_usd), 2),
                    **threshold_payload,
                },
            )
        )
    return actions


def _manage_forecast_reversal_exit(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    now: datetime,
    underlying_price_usd: float | None,
    forecast: dict[str, Any],
) -> list[dict[str, Any]]:
    if not bool(getattr(args, "enable_forecast_reversal_exit", True)):
        return []
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return []
    expected_direction = str(forecast.get("expected_direction") or "").lower()
    predicted_price = _float_or_none(forecast.get("predicted_price"))
    spot = _float_or_none(forecast.get("spot")) or float(underlying_price_usd)
    if predicted_price is None or spot <= 0:
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
    desired_type = None
    if expected_direction == "upward" or forecast_move_pct > 0:
        desired_type = "call"
    elif expected_direction == "downward" or forecast_move_pct < 0:
        desired_type = "put"
    if desired_type is None:
        return []
    actions: list[dict[str, Any]] = []
    open_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    for position in open_positions:
        instrument = str(position.get("instrument_name") or "")
        current_type = _option_type_from_instrument_name(instrument)
        if not current_type:
            continue
        if current_type == desired_type:
            actions.append(
                {
                    "action": "forecast_reversal_exit_not_triggered_position_aligned",
                    "instrument_name": instrument,
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


def _close_position_with_limit(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    position: dict[str, Any],
    now: datetime,
    reason: str,
    limit_offset_pct: float | None = None,
    extra: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    instrument = str(position.get("instrument_name") or "")
    amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
    mark = _float_or_none(position.get("mark_price"))
    if not instrument or amount <= 0 or mark is None:
        return [{"action": f"{reason}_skipped_missing_mark_or_amount", "instrument_name": instrument, "amount": amount}]
    sell_orders = _sell_orders_for_instrument(open_orders, instrument)
    tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument, price=mark)
    close_offset = float(limit_offset_pct) if limit_offset_pct is not None else float(getattr(args, "liquidation_limit_offset_pct", 0.05))
    close_limit = _deribit_liquidation_limit_price(mark, close_offset, tick_size=tick_size)
    selected_exit = {
        "instrument_name": instrument,
        "side": "sell",
        "type": "limit",
        "amount": amount,
        "price": close_limit,
        "reduce_only": True,
    }
    if not _orders_enabled(args):
        return [
            {
                "action": f"would_{reason}",
                "instrument_name": instrument,
                "amount": amount,
                "mark_price": round(mark, 8),
                "close_limit_price": close_limit,
                "close_limit_offset_pct": close_offset,
                **(extra or {}),
            }
        ]
    return _replace_sell_orders_with_exit(
        broker=broker,
        args=args,
        instrument=instrument,
        amount=amount,
        sell_orders=sell_orders,
        selected_exit=selected_exit,
        reason=reason,
        now=now,
        extra={"mark_price": round(mark, 8), "close_limit_price": close_limit, "close_limit_offset_pct": close_offset, **(extra or {})},
    )


def _liquidate_option_positions(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    now: datetime,
) -> list[dict[str, Any]]:
    active_positions = [position for position in positions if abs(_float_or_none(position.get("size")) or 0.0) > 0.0]
    actions: list[dict[str, Any]] = [
        {
            "action": "manual_liquidate_and_stop_triggered",
            "currency": args.currency.upper(),
            "position_count": len(active_positions),
        }
    ]
    for position in active_positions:
        instrument = str(position.get("instrument_name") or "")
        amount = abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0)
        mark = _float_or_none(position.get("mark_price"))
        if not instrument or amount <= 0 or mark is None:
            actions.append({"action": "manual_liquidation_skipped_missing_price_or_amount", "instrument_name": instrument, "amount": amount})
            continue
        sell_orders = _sell_orders_for_instrument(open_orders, instrument)
        tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument, price=mark)
        close_limit = _deribit_liquidation_limit_price(mark, float(args.liquidation_limit_offset_pct), tick_size=tick_size)
        selected_exit = {
            "instrument_name": instrument,
            "side": "sell",
            "type": "limit",
            "amount": amount,
            "price": close_limit,
            "reduce_only": True,
        }
        actions.extend(
            _replace_sell_orders_with_exit(
                broker=broker,
                args=args,
                instrument=instrument,
                amount=amount,
                sell_orders=sell_orders,
                selected_exit=selected_exit,
                reason="manual_liquidation_position_close_triggered",
                now=now,
                extra={
                    "mark_price": round(mark, 8),
                    "close_limit_price": close_limit,
                    "position_unrealized_pl": _float_or_none(position.get("floating_profit_loss")),
                },
            )
        )
    return actions


def _positions_for_total_loss_close(positions: list[dict[str, Any]], close_mode: str) -> list[dict[str, Any]]:
    if close_mode == "losing_only":
        return [position for position in positions if (_position_pl_value(position) or 0.0) < 0.0]
    return positions


def _positions_for_total_profit_close(positions: list[dict[str, Any]], close_mode: str) -> list[dict[str, Any]]:
    if close_mode == "winning_only":
        return [position for position in positions if (_position_pl_value(position) or 0.0) > 0.0]
    return positions


def _update_position_profit_peak(
    state: dict[str, Any],
    *,
    position: dict[str, Any],
    underlying_price_usd: float | None,
    now: datetime,
) -> dict[str, Any] | None:
    instrument = str(position.get("instrument_name") or "")
    pl_base = _position_pl_value(position)
    pl_usd = _position_pl_usd(position, underlying_price_usd)
    mark = _float_or_none(position.get("mark_price"))
    if not instrument or pl_base is None or pl_usd is None:
        return None
    peaks = state.setdefault("option_position_profit_peaks", {})
    if not isinstance(peaks, dict):
        peaks = {}
        state["option_position_profit_peaks"] = peaks
    existing = peaks.get(instrument) if isinstance(peaks.get(instrument), dict) else {}
    existing_peak_usd = _float_or_none(existing.get("peak_unrealized_pl_usd"))
    if existing_peak_usd is None or float(pl_usd) > existing_peak_usd:
        existing = {
            "instrument_name": instrument,
            "peak_unrealized_pl_base": round(float(pl_base), 8),
            "peak_unrealized_pl_usd": round(float(pl_usd), 2),
            "peak_mark_price": None if mark is None else round(float(mark), 8),
            "updated_at_utc": now.isoformat(),
        }
        peaks[instrument] = existing
    return existing


def _prune_position_profit_peaks(state: dict[str, Any], positions: list[dict[str, Any]]) -> None:
    peaks = state.get("option_position_profit_peaks")
    if not isinstance(peaks, dict):
        return
    open_instruments = {
        str(position.get("instrument_name") or "")
        for position in positions
        if abs(_float_or_none(position.get("size")) or _float_or_none(position.get("amount")) or 0.0) > 0.0 and position.get("instrument_name")
    }
    for instrument in list(peaks):
        if instrument not in open_instruments:
            peaks.pop(instrument, None)


def _position_pl_value(position: dict[str, Any]) -> float | None:
    return _float_or_none(position.get("floating_profit_loss")) if _float_or_none(position.get("floating_profit_loss")) is not None else _float_or_none(position.get("total_profit_loss"))


def _position_pl_usd(position: dict[str, Any], underlying_price_usd: float | None) -> float | None:
    explicit = _float_or_none(position.get("floating_profit_loss_usd"))
    if explicit is not None:
        return explicit
    explicit = _float_or_none(position.get("total_profit_loss_usd"))
    if explicit is not None:
        return explicit
    raw = _position_pl_value(position)
    if raw is None:
        return None
    if _is_usdc_settled_instrument(str(position.get("instrument_name") or "")):
        return raw
    if underlying_price_usd is None or underlying_price_usd <= 0:
        return None
    return raw * float(underlying_price_usd)


def _is_usdc_settled_instrument(instrument_name: str) -> bool:
    return "_" in str(instrument_name or "").split("-", 1)[0]


def _deribit_liquidation_limit_price(mark_price: float, offset_pct: float, *, tick_size: float | None = None) -> float:
    tick = max(float(tick_size or 0.0001), 0.00000001)
    raw = max(tick, float(mark_price) * (1.0 - max(0.0, float(offset_pct))))
    return round(max(tick, math.floor(raw / tick) * tick), 8)


def _instrument_tick_size(*, broker: DeribitTestnetBroker, instrument_name: str, price: float | None = None) -> float | None:
    currency = _instrument_currency_from_name(instrument_name)
    try:
        instruments = broker.instruments(currency=currency, kind="option", expired=False)
    except (RuntimeError, AttributeError):
        return None
    for instrument in instruments:
        if str(instrument.get("instrument_name") or "") == instrument_name:
            stepped = _tick_size_from_steps(instrument.get("tick_size_steps"), price)
            if stepped is not None:
                return stepped
            return _float_or_none(instrument.get("tick_size"))
    return None


def _instrument_metadata_by_name(*, broker: DeribitTestnetBroker, currency: str) -> dict[str, dict[str, Any]]:
    try:
        instruments = broker.instruments(currency=currency, kind="option", expired=False)
    except (RuntimeError, AttributeError):
        return {}
    return {str(instrument.get("instrument_name") or ""): instrument for instrument in instruments if instrument.get("instrument_name")}


def _position_expiry_utc(position: dict[str, Any], instruments: dict[str, dict[str, Any]]) -> datetime | None:
    instrument_name = str(position.get("instrument_name") or "")
    instrument = instruments.get(instrument_name) or {}
    timestamp = _float_or_none(instrument.get("expiration_timestamp")) or _float_or_none(position.get("expiration_timestamp"))
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp / 1000.0, tz=UTC)
    explicit = instrument.get("expiration_utc") or position.get("expiration_utc")
    if explicit:
        try:
            return datetime.fromisoformat(str(explicit).replace("Z", "+00:00"))
        except ValueError:
            pass
    return _expiry_from_deribit_instrument_name(instrument_name)


def _expiry_from_deribit_instrument_name(instrument_name: str) -> datetime | None:
    parts = str(instrument_name or "").split("-")
    if len(parts) < 2:
        return None
    try:
        parsed = datetime.strptime(parts[1].upper(), "%d%b%y")
    except ValueError:
        return None
    return parsed.replace(hour=8, minute=0, second=0, microsecond=0, tzinfo=UTC)


def _tick_size_from_steps(steps: Any, price: float | None) -> float | None:
    if price is None or not isinstance(steps, list):
        return None
    selected = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        above = _float_or_none(step.get("above_price"))
        tick = _float_or_none(step.get("tick_size"))
        if tick is None:
            continue
        if above is None or float(price) >= above:
            selected = tick
    return selected


def _abandon_stale_entry_orders(
    *,
    broker: DeribitTestnetBroker,
    args: argparse.Namespace,
    open_orders: list[dict[str, Any]],
    state: dict[str, Any],
    now: datetime,
) -> list[dict[str, Any]]:
    active_trade = state.get("active_trade") or {}
    entry_order = (active_trade.get("entry_order") or {}).get("order") or active_trade.get("entry_order") or {}
    order_id = entry_order.get("order_id") or entry_order.get("id")
    if not order_id:
        return []
    age = _age_seconds(entry_order.get("creation_timestamp") or active_trade.get("created_at_utc"), now)
    if age is None or age < int(args.abandon_entry_after_seconds):
        return []
    if not any(str(order.get("order_id") or order.get("id")) == str(order_id) for order in open_orders):
        return []
    if not _orders_enabled(args):
        return [{"action": "would_cancel_stale_entry", "order_id": order_id, "age_seconds": age}]
    try:
        result = broker.cancel_order(str(order_id))
        state["active_trade"] = {**active_trade, "status": "entry_cancelled_stale", "cancelled_at_utc": now.isoformat()}
        return [{"action": "cancelled_stale_entry", "order_id": order_id, "age_seconds": age, "result": result}]
    except RuntimeError as exc:
        return [{"action": "cancel_stale_entry_failed", "order_id": order_id, "age_seconds": age, "error": str(exc)}]


def write_record(output_dir: Path, currency: str, record: dict[str, Any]) -> Path:
    report_path = output_dir / f"{currency.upper()}_deribit_options_agent_report.json"
    report_path.write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
    return report_path


def read_state(output_dir: Path, currency: str) -> dict[str, Any]:
    path = _state_path(output_dir, currency)
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_state(output_dir: Path, currency: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def append_log(output_dir: Path, currency: str, record: dict[str, Any]) -> None:
    path = output_dir / "logs" / f"{currency.upper()}_{datetime.now(UTC).strftime('%Y%m%d')}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def liquidate_and_stop(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    stop_request = write_stop_request(output_dir, args.currency, reason="liquidate_and_stop")
    broker = _broker_for_args(args)
    now = datetime.now(UTC)
    instrument_currency = _instrument_currency(args)
    account = broker.account_summary(currency=instrument_currency)
    open_orders = _option_open_orders(broker.open_orders(currency=instrument_currency, kind="option"), args.currency, instrument_currency=instrument_currency)
    positions = _option_positions(broker.positions(currency=instrument_currency, kind="option"), args.currency, instrument_currency=instrument_currency)
    actions = _cancel_open_option_orders(
        broker=broker,
        args=args,
        open_orders=open_orders,
        cancel_action="cancelled_open_order_for_liquidate_and_stop",
    )
    actions.extend(
        _liquidate_option_positions(
            broker=broker,
            args=args,
            open_orders=[],
            positions=positions,
            now=now,
        )
    )
    record = {
        "checked_at": datetime.now(UTC).isoformat(),
        "venue": f"deribit_{args.account_mode}",
        "account_mode": args.account_mode,
        "currency": args.currency.upper(),
        "instrument_currency": instrument_currency,
        "ticker": f"{args.currency.upper()}-USD",
        "command": "liquidate_and_stop",
        "account": _safe_account(account),
        "stop_request": stop_request,
        "open_option_orders": open_orders,
        "option_positions": positions,
        "management_actions": actions,
        "execution_blocks": [],
        "execute_paper_orders": bool(args.execute_paper_orders),
        "execute_live_orders": bool(args.execute_live_orders),
        "order_result": {"submitted": bool(actions), "reason": "liquidate_and_stop"},
    }
    write_state(output_dir, args.currency, {"stop_request": stop_request, "last_record_summary": _summary(record)})
    append_log(output_dir, args.currency, record)
    return record


def write_stop_request(output_dir: Path, currency: str, *, reason: str) -> dict[str, Any]:
    payload = {
        "currency": currency.upper(),
        "reason": reason,
        "requested_at": datetime.now(UTC).isoformat(),
    }
    path = _stop_request_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {**payload, "path": str(path)}


def read_stop_request(output_dir: Path, currency: str) -> dict[str, Any] | None:
    path = _stop_request_path(output_dir, currency)
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"currency": currency.upper(), "reason": "invalid_stop_request", "path": str(path)}
    return {**parsed, "path": str(path)} if isinstance(parsed, dict) else {"currency": currency.upper(), "reason": "invalid_stop_request", "path": str(path)}


def clear_stop_request(output_dir: Path, currency: str) -> bool:
    path = _stop_request_path(output_dir, currency)
    if not path.exists():
        return False
    path.unlink()
    return True


def _state_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "state" / f"{currency.upper()}_deribit_options_agent_state.json"


def _stop_request_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "control" / f"{currency.upper()}_stop.json"


def _load_prices(args: argparse.Namespace) -> pd.DataFrame:
    start = (datetime.now(UTC) - timedelta(days=int(args.lookback_days))).isoformat().replace("+00:00", "Z")
    result = load_prices_with_provider(
        args.data_provider,
        DataRequest(ticker=f"{args.currency.upper()}-USD", start=start, interval=args.data_interval, target_column="close"),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    prices = normalize_price_frame(result.frame, target_column="close")
    if args.max_training_rows and len(prices) > int(args.max_training_rows):
        prices = prices.tail(int(args.max_training_rows))
    return prices


def _option_open_orders(orders: list[dict[str, Any]], currency: str, *, instrument_currency: str | None = None) -> list[dict[str, Any]]:
    prefix = _instrument_prefix(currency, instrument_currency)
    return [
        {
            "order_id": order.get("order_id"),
            "instrument_name": order.get("instrument_name"),
            "direction": order.get("direction"),
            "amount": order.get("amount"),
            "price": order.get("price"),
            "order_state": order.get("order_state"),
            "creation_timestamp": order.get("creation_timestamp"),
            "label": order.get("label"),
        }
        for order in orders
        if str(order.get("instrument_name") or "").upper().startswith(prefix)
    ]


def _option_positions(positions: list[dict[str, Any]], currency: str, *, instrument_currency: str | None = None) -> list[dict[str, Any]]:
    prefix = _instrument_prefix(currency, instrument_currency)
    rows = []
    for position in positions:
        instrument_name = str(position.get("instrument_name") or "")
        if not instrument_name.upper().startswith(prefix):
            continue
        size = _float_or_none(position.get("size"))
        if str(position.get("direction") or "").lower() == "zero" or size is None or abs(size) <= 1e-12:
            continue
        expiry = _position_expiry_utc(position, {})
        rows.append(
            {
                "instrument_name": position.get("instrument_name"),
                "size": position.get("size"),
                "average_price": position.get("average_price"),
                "floating_profit_loss": position.get("floating_profit_loss"),
                "floating_profit_loss_usd": position.get("floating_profit_loss_usd"),
                "total_profit_loss": position.get("total_profit_loss"),
                "total_profit_loss_usd": position.get("total_profit_loss_usd"),
                "mark_price": position.get("mark_price"),
                "average_price_usd": position.get("average_price_usd"),
                "expiration_timestamp": position.get("expiration_timestamp"),
                "expiration_utc": expiry.isoformat() if expiry else position.get("expiration_utc"),
                "delta": position.get("delta"),
                "gamma": position.get("gamma"),
                "theta": position.get("theta"),
                "vega": position.get("vega"),
            }
        )
    return rows


def _safe_account(account: dict[str, Any]) -> dict[str, Any]:
    allowed = ["currency", "balance", "equity", "available_funds", "available_withdrawal_funds", "margin_balance", "options_session_rpl", "options_session_upl"]
    return {key: account.get(key) for key in allowed if key in account}


def _broker_for_args(args: argparse.Namespace) -> DeribitTestnetBroker | DeribitOptionsBroker:
    if str(getattr(args, "account_mode", "testnet")) == "testnet":
        return DeribitTestnetBroker()
    return DeribitOptionsBroker(account_mode="live")


def _orders_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "execute_paper_orders", False) or getattr(args, "execute_live_orders", False))


def _instrument_currency(args: argparse.Namespace) -> str:
    return str(getattr(args, "instrument_currency", None) or getattr(args, "currency", "ETH")).upper()


def _instrument_prefix(currency: str, instrument_currency: str | None = None) -> str:
    underlying = str(currency or "").upper()
    instrument_ccy = str(instrument_currency or currency or "").upper()
    if instrument_ccy in {"USDC", "USDT", "USDE"}:
        return f"{underlying}_{instrument_ccy}-"
    return f"{instrument_ccy}-"


def _instrument_currency_from_name(instrument_name: str) -> str:
    prefix = str(instrument_name or "").split("-", 1)[0].upper()
    if "_" in prefix:
        return prefix.rsplit("_", 1)[-1]
    return prefix


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


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    trade_plan = record.get("option_trade_plan") or {}
    selected = trade_plan.get("selected_contract") or {}
    order = trade_plan.get("order") or {}
    return {
        "currency": record.get("currency"),
        "forecast_direction": (record.get("selected_forecast") or {}).get("expected_direction"),
        "forecast_price": (record.get("selected_forecast") or {}).get("predicted_price"),
        "action": trade_plan.get("action"),
        "reason": trade_plan.get("reason"),
        "contract": selected.get("instrument_name"),
        "limit_price_base": order.get("price"),
        "estimated_debit_usd": (trade_plan.get("risk") or {}).get("estimated_debit_usd"),
        "order_submitted": (record.get("order_result") or {}).get("submitted"),
        "execution_blocks": record.get("execution_blocks"),
    }


def _risk_control_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_total_unrealized_loss_usd": _abs_or_none(getattr(args, "max_total_unrealized_loss_usd", None)),
        "total_loss_close_mode": getattr(args, "total_loss_close_mode", None),
        "max_total_unrealized_profit_usd": getattr(args, "max_total_unrealized_profit_usd", None),
        "total_profit_close_mode": getattr(args, "total_profit_close_mode", None),
        "max_position_unrealized_loss_usd": _abs_or_none(getattr(args, "max_position_unrealized_loss_usd", None)),
        "max_position_unrealized_profit_usd": getattr(args, "max_position_unrealized_profit_usd", None),
        "take_profit_position_pl_usd": getattr(args, "take_profit_position_pl_usd", None),
        "profit_retrace_from_peak_pct": getattr(args, "profit_retrace_from_peak_pct", None),
        "profit_close_limit_offset_pct": getattr(args, "profit_close_limit_offset_pct", None),
        "enable_forecast_reversal_exit": getattr(args, "enable_forecast_reversal_exit", None),
        "min_reversal_edge_pct": getattr(args, "min_reversal_edge_pct", None),
        "enable_chart_patterns": getattr(args, "enable_chart_patterns", None),
        "block_chart_pattern_conflicts": getattr(args, "block_chart_pattern_conflicts", None),
        "chart_pattern_lookback_rows": getattr(args, "chart_pattern_lookback_rows", None),
        "chart_pattern_level_tolerance_pct": getattr(args, "chart_pattern_level_tolerance_pct", None),
        "chart_pattern_breakout_buffer_pct": getattr(args, "chart_pattern_breakout_buffer_pct", None),
        "chart_pattern_min_volume_ratio": getattr(args, "chart_pattern_min_volume_ratio", None),
        "min_chart_pattern_confidence": getattr(args, "min_chart_pattern_confidence", None),
        "close_before_expiry_hours": getattr(args, "close_before_expiry_hours", None),
        "expiry_warning_hours": getattr(args, "expiry_warning_hours", None),
        "liquidation_limit_offset_pct": getattr(args, "liquidation_limit_offset_pct", None),
        "max_open_option_positions": getattr(args, "max_open_option_positions", None),
        "max_open_option_contracts": getattr(args, "max_open_option_contracts", None),
        "max_open_option_premium_usd": getattr(args, "max_open_option_premium_usd", None),
        "allow_mixed_option_direction": getattr(args, "allow_mixed_option_direction", None),
        "enable_market_regime_filter": getattr(args, "enable_market_regime_filter", None),
        "allow_range_edge_reversal_entry": getattr(args, "allow_range_edge_reversal_entry", None),
        "market_regime_lookback_rows": getattr(args, "market_regime_lookback_rows", None),
        "market_regime_breakout_buffer_pct": getattr(args, "market_regime_breakout_buffer_pct", None),
        "market_regime_middle_zone_width": getattr(args, "market_regime_middle_zone_width", None),
        "min_trend_strength_pct": getattr(args, "min_trend_strength_pct", None),
        "enable_impulse_entry": getattr(args, "enable_impulse_entry", None),
        "impulse_lookback_bars": getattr(args, "impulse_lookback_bars", None),
        "min_impulse_move_pct": getattr(args, "min_impulse_move_pct", None),
        "min_impulse_directional_bars": getattr(args, "min_impulse_directional_bars", None),
        "enable_late_entry_filter": getattr(args, "enable_late_entry_filter", None),
        "max_late_entry_move_pct": getattr(args, "max_late_entry_move_pct", None),
        "max_ema_extension_pct": getattr(args, "max_ema_extension_pct", None),
        "exhaustion_reversal_bars": getattr(args, "exhaustion_reversal_bars", None),
        "enable_feedback_loop": getattr(args, "enable_feedback_loop", None),
        "feedback_min_matured": getattr(args, "feedback_min_matured", None),
        "feedback_min_direction_accuracy": getattr(args, "feedback_min_direction_accuracy", None),
        "feedback_max_abs_pct_error": getattr(args, "feedback_max_abs_pct_error", None),
        "feedback_ledger_window": getattr(args, "feedback_ledger_window", None),
    }


def _abs_or_none(value: Any) -> float | None:
    parsed = _float_or_none(value)
    return None if parsed is None else abs(float(parsed))


def _protective_close_triggered(actions: list[dict[str, Any]]) -> bool:
    close_markers = (
        "total_loss_position_close_triggered",
        "total_profit_position_close_triggered",
        "position_loss_position_close_triggered",
        "position_profit_position_close_triggered",
        "forecast_reversal_position_close_triggered",
        "expiry_position_close_triggered",
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


def _book_quote(order_book: dict[str, Any]) -> dict[str, float | None]:
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    return {
        "bid": _book_price(bids[0]) if bids else _float_or_none(order_book.get("best_bid_price")),
        "ask": _book_price(asks[0]) if asks else _float_or_none(order_book.get("best_ask_price")),
    }


def _book_price(level: Any) -> float | None:
    if isinstance(level, dict):
        return _float_or_none(level.get("price"))
    if isinstance(level, (list, tuple)) and level:
        return _float_or_none(level[0])
    return None


def _position_entry_price(position: dict[str, Any]) -> float | None:
    return _float_or_none(position.get("average_price")) or _float_or_none(position.get("average_price_usd"))


def _entry_price_from_state(active_trade: dict[str, Any]) -> float | None:
    order = ((active_trade.get("trade_plan") or {}).get("order") or {})
    return _float_or_none(order.get("price"))


def _age_seconds(timestamp: Any, now: datetime) -> float | None:
    if not timestamp:
        return None
    if isinstance(timestamp, (int, float)):
        return (now - datetime.fromtimestamp(float(timestamp) / 1000.0, tz=UTC)).total_seconds()
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except ValueError:
        return None
    return (now - parsed).total_seconds()


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _timestamp_ms_to_datetime(value: float | None) -> datetime | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


if __name__ == "__main__":
    main()
