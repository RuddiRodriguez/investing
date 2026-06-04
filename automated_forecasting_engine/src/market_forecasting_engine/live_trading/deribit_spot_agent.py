from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from market_forecasting_engine.daily_trade import DailyTradeConfig, build_daily_trade_plan
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.deribit_broker import DeribitLiveSpotBroker
from market_forecasting_engine.dip_buy import annotate_mean_reversion_dip_buy
from market_forecasting_engine.live_trading.deribit_report import write_deribit_account_report
from market_forecasting_engine.risk_profiles import risk_profile_for_name


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.execute_live_orders and not args.confirm_live_deribit_spot_orders:
        raise SystemExit("Live execution requires --confirm-live-deribit-spot-orders.")
    while True:
        record = run_once(args)
        path = write_spot_agent_report(record, output_dir)
        print(json.dumps({"report": str(path), **summary(record)}, indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(10, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live Deribit ETH_USDC spot trading agent.")
    parser.add_argument("--instrument", default="ETH_USDC")
    parser.add_argument("--base-currency", default="ETH")
    parser.add_argument("--quote-currency", default="USDC")
    parser.add_argument("--risk-profile", choices=("conservative", "medium", "aggressive"), default="aggressive")
    parser.add_argument("--data-provider", default="alpaca")
    parser.add_argument("--data-interval", default="1m")
    parser.add_argument("--lookback-days", type=int, default=20)
    parser.add_argument("--forecast-hours", default="0.25,0.5,1")
    parser.add_argument("--forecast-refresh-seconds", type=int, default=180)
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--max-training-rows", type=int, default=3500)
    parser.add_argument("--max-notional-usdc", type=float, default=25.0)
    parser.add_argument("--min-edge-pct", type=float, default=None)
    parser.add_argument("--max-spread-pct", type=float, default=0.005)
    parser.add_argument("--min-order-base-amount", type=float, default=0.0001)
    parser.add_argument("--max-base-position", type=float, default=0.05)
    parser.add_argument("--take-profit-pct", type=float, default=0.08)
    parser.add_argument("--stop-loss-pct", type=float, default=0.04)
    parser.add_argument("--enable-pullback-buy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pullback-min-reward-risk", type=float, default=1.25)
    parser.add_argument("--pullback-min-reversal-probability", type=float, default=0.45)
    parser.add_argument("--pullback-max-entry-distance-pct", type=float, default=0.05)
    parser.add_argument("--enable-scale-in-pullback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--spot-average-entry-price", type=float, default=None, help="Average entry price for existing live spot inventory. Required for scale-in unless explicitly overridden.")
    parser.add_argument("--allow-scale-in-without-entry-price", action="store_true", help="Allow scale-in without knowing the existing position average entry. Not recommended for live trading.")
    parser.add_argument("--scale-in-min-discount-from-entry-pct", type=float, default=0.02)
    parser.add_argument("--scale-in-max-existing-loss-pct", type=float, default=0.06)
    parser.add_argument("--scale-in-max-addition-fraction", type=float, default=0.35)
    parser.add_argument("--scale-in-max-notional-usdc", type=float, default=None)
    parser.add_argument("--replace-protection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--respect-existing-protection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--protected-position-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--sell-now-min-forecast-beyond-stop-pct", type=float, default=0.01)
    parser.add_argument("--tighten-stop-buffer-pct", type=float, default=0.003)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/live_deribit_spot_agent")
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-deribit-spot-orders", action="store_true")
    parser.add_argument("--once", action="store_true")
    return parser


def run_once(args: argparse.Namespace, broker: DeribitLiveSpotBroker | None = None) -> dict[str, Any]:
    broker = broker or DeribitLiveSpotBroker()
    now = datetime.now(UTC)
    instrument = args.instrument.upper()
    base_currency = args.base_currency.upper()
    quote_currency = args.quote_currency.upper()
    base_account = broker.account_summary(currency=base_currency)
    quote_account = broker.account_summary(currency=quote_currency)
    order_book = broker.order_book(instrument, depth=10)
    quote = book_quote(order_book)
    ticker = broker.ticker(instrument)
    open_orders = dedupe_orders(
        [
            *broker.open_orders(currency=base_currency, kind="spot"),
            *broker.open_orders(currency=quote_currency, kind="spot"),
        ]
    )
    price = quote["mid"] or quote["ask"] or quote["bid"] or float(ticker.get("last_price") or 0.0)
    forecast_bundle = build_forecast(args)
    forecast = forecast_bundle["selected_forecast"]
    plan = decide_spot_trade(
        args=args,
        base_account=base_account,
        quote_account=quote_account,
        quote=quote,
        latest_price=price,
        forecast=forecast,
        forecast_bundle=forecast_bundle,
        open_orders=open_orders,
    )
    order_results: list[dict[str, Any]] = []
    if args.execute_live_orders and plan["action"] in {"buy_spot", "place_pullback_buy", "place_scale_in_pullback_buy", "sell_spot", "protect_existing_position"}:
        label_base = f"codex-live-spot-{base_currency}-{now.strftime('%Y%m%d%H%M%S')}"
        order_results = execute_plan(broker=broker, plan=plan, open_orders=open_orders, label_base=label_base, replace_protection=bool(args.replace_protection))
    record = {
        "checked_at": now.isoformat(),
        "mode": "live_deribit_spot_agent",
        "venue": "deribit_live",
        "instrument": instrument,
        "base_currency": base_currency,
        "quote_currency": quote_currency,
        "safety": {
            "execute_live_orders": bool(args.execute_live_orders),
            "confirmation_required": True,
            "confirmation_provided": bool(args.confirm_live_deribit_spot_orders),
            "policy": "Dry-run by default. Live spot orders require --execute-live-orders and --confirm-live-deribit-spot-orders.",
        },
        "account": {
            base_currency: account_subset(base_account),
            quote_currency: account_subset(quote_account),
        },
        "market": {"order_book": order_book, "ticker": ticker, "quote": quote, "latest_price": price},
        "forecast": forecast,
        "forecast_bundle": forecast_bundle,
        "open_orders": open_orders,
        "decision": plan,
        "order_results": order_results,
    }
    return record


def build_forecast(args: argparse.Namespace) -> dict[str, Any]:
    prices = load_prices_with_provider(
        args.data_provider,
        DataRequest(
            ticker=f"{args.base_currency.upper()}-USD",
            start=(datetime.now(UTC) - timedelta(days=int(args.lookback_days))).isoformat(),
            interval=args.data_interval,
        ),
    ).frame
    if len(prices) > int(args.max_training_rows):
        prices = prices.tail(int(args.max_training_rows))
    plan = build_daily_trade_plan(
        prices,
        DailyTradeConfig(
            ticker=f"{args.base_currency.upper()}-USD",
            interval=args.data_interval,
            forecast_hours=tuple(float(value.strip()) for value in str(args.forecast_hours).split(",") if value.strip()),
            minimum_score_to_trade=1.5 if args.risk_profile == "aggressive" else 2.0,
        ),
    )
    normalize_crypto_forecast_timestamps(plan)
    attach_rolling_validation(plan.get("forecasts", []), prices)
    forecast = _primary_forecast(plan)
    dip_report = {
        "ticker": f"{args.base_currency.upper()}-USD",
        "current_price": plan["latest_price"],
        "forecasts": [_forecast_for_decision(row, plan["latest_price"]) for row in plan.get("forecasts", [])],
    }
    annotate_mean_reversion_dip_buy(dip_report, prices, "close", risk_profile_name=args.risk_profile)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "price_rows": len(prices),
        "price_tail": _price_tail(prices),
        "forecast_plan": plan,
        "selected_forecast": forecast,
        "mean_reversion_dip_buy": dip_report.get("decision_view", {}).get("mean_reversion_dip_buy", {}),
    }


def normalize_crypto_forecast_timestamps(plan: dict[str, Any]) -> None:
    """Use continuous clock horizons for crypto spot instead of equity-session bars."""

    as_of_raw = plan.get("as_of")
    try:
        as_of = datetime.fromisoformat(str(as_of_raw))
    except (TypeError, ValueError):
        return
    for forecast in plan.get("forecasts", []) or []:
        try:
            hours = float(forecast.get("horizon_hours") or 0.0)
        except (TypeError, ValueError):
            continue
        forecast["forecast_timestamp"] = (as_of + timedelta(hours=hours)).isoformat()
        forecast["timestamp_policy"] = "continuous_crypto_clock"


def attach_rolling_validation(forecasts: list[dict[str, Any]], prices: Any, max_samples: int = 120) -> None:
    close = _close_values(prices)
    if len(close) < 40:
        for forecast in forecasts:
            forecast["validation_metrics"] = {"status": "insufficient_history", "sample_count": 0}
        return
    for forecast in forecasts:
        horizon = int(float(forecast.get("horizon_bars") or 0))
        if horizon <= 0 or len(close) <= horizon + 14:
            forecast["validation_metrics"] = {"status": "insufficient_matured_horizon_history", "sample_count": 0}
            continue
        start = max(13, len(close) - horizon - max_samples)
        stop = len(close) - horizon
        price_errors: list[float] = []
        return_errors: list[float] = []
        direction_hits = 0
        samples = 0
        for idx in range(start, stop):
            anchor = close[idx]
            actual = close[idx + horizon]
            if anchor <= 0 or actual <= 0:
                continue
            returns = []
            for lookback in range(idx - 11, idx + 1):
                previous = close[lookback - 1]
                current = close[lookback]
                if previous > 0 and current > 0:
                    returns.append(math.log(current / previous))
            drift = sum(returns) / len(returns) if returns else 0.0
            predicted = anchor * math.exp(drift * horizon)
            price_errors.append(abs(predicted - actual))
            predicted_return = (predicted - anchor) / anchor
            actual_return = (actual - anchor) / anchor
            return_errors.append(abs(predicted_return - actual_return))
            direction_hits += int((predicted_return >= 0) == (actual_return >= 0))
            samples += 1
        forecast["validation_metrics"] = {
            "status": "measured" if samples else "no_valid_samples",
            "sample_count": samples,
            "price_mae": round(sum(price_errors) / samples, 4) if samples else None,
            "return_mae": round(sum(return_errors) / samples, 6) if samples else None,
            "directional_accuracy": round(direction_hits / samples, 4) if samples else None,
            "method": "rolling_matured_horizon_validation",
        }


def decide_spot_trade(
    *,
    args: argparse.Namespace,
    base_account: dict[str, Any],
    quote_account: dict[str, Any],
    quote: dict[str, float | None],
    latest_price: float,
    forecast: dict[str, Any],
    open_orders: list[dict[str, Any]],
    forecast_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile = risk_profile_for_name(args.risk_profile)
    min_edge = float(args.min_edge_pct if args.min_edge_pct is not None else max(profile.minimum_edge_fraction, 0.001))
    spread_pct = spread_fraction(quote)
    base_balance = _float(base_account.get("balance"))
    quote_available = _float(quote_account.get("available_funds"))
    predicted = _float(forecast.get("predicted_price"))
    if latest_price <= 0 or predicted <= 0:
        return hold_plan("missing_market_or_forecast_price", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct)
    edge_pct = (predicted - latest_price) / latest_price
    direction = "upward" if edge_pct > 0 else "downward" if edge_pct < 0 else "flat"
    if spread_pct is not None and spread_pct > float(args.max_spread_pct):
        return hold_plan("spread_too_wide", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
    if abs(edge_pct) < min_edge:
        return hold_plan("forecast_edge_too_small", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
    if direction == "upward":
        if has_open_spot_entry(open_orders, args.instrument):
            return hold_plan("open_entry_order_exists", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
        if base_balance >= float(args.max_base_position):
            return protection_plan(args=args, latest_price=latest_price, base_balance=base_balance, quote_available=quote_available, forecast=forecast, edge_pct=edge_pct, reason="max_base_position_reached")
        max_notional = min(float(args.max_notional_usdc), quote_available)
        amount = round_down(max_notional / float(quote["ask"] or latest_price), 6)
        if amount < float(args.min_order_base_amount):
            return hold_plan("not_enough_quote_available_for_min_order", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
        entry_price = round_price(float(quote["ask"] or latest_price) * 1.001)
        return {
            **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
            "action": "buy_spot",
            "reason": "bullish_forecast",
            "entry_order": {"side": "buy", "type": "limit", "amount": amount, "price": entry_price},
            "protection": {},
            "post_fill_protection_plan": protection_orders(args=args, amount=amount, entry_reference=entry_price),
        }
    if base_balance >= float(args.min_order_base_amount):
        scale_in = scale_in_pullback_buy_plan(
            args=args,
            latest_price=latest_price,
            forecast=forecast,
            forecast_bundle=forecast_bundle or {},
            base_balance=base_balance,
            quote_available=quote_available,
            spread_pct=spread_pct,
            edge_pct=edge_pct,
            open_orders=open_orders,
            quote=quote,
        )
        if scale_in is not None and scale_in.get("action") != "hold":
            return scale_in
        protection_decision = protection_aware_bearish_plan(
            args=args,
            latest_price=latest_price,
            forecast=forecast,
            base_balance=base_balance,
            quote_available=quote_available,
            spread_pct=spread_pct,
            edge_pct=edge_pct,
            open_orders=open_orders,
        )
        if protection_decision is not None:
            if scale_in is not None:
                protection_decision["scale_in_pullback"] = scale_in.get("scale_in_pullback", {})
            return protection_decision
        sell_price = round_price(float(quote["bid"] or latest_price) * 0.999)
        plan = {
            **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
            "action": "sell_spot",
            "reason": "bearish_forecast_sell_existing_base",
            "entry_order": {"side": "sell", "type": "limit", "amount": round_down(min(base_balance, float(args.max_base_position)), 6), "price": sell_price},
            "protection": {},
        }
        if scale_in is not None:
            plan["scale_in_pullback"] = scale_in.get("scale_in_pullback", {})
        return plan
    pullback = pullback_buy_plan(
        args=args,
        latest_price=latest_price,
        forecast=forecast,
        forecast_bundle=forecast_bundle or {},
        base_balance=base_balance,
        quote_available=quote_available,
        spread_pct=spread_pct,
        edge_pct=edge_pct,
        open_orders=open_orders,
        quote=quote,
    )
    if pullback is not None:
        return pullback
    return hold_plan("bearish_forecast_no_base_position", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)


def protection_aware_bearish_plan(
    *,
    args: argparse.Namespace,
    latest_price: float,
    forecast: dict[str, Any],
    base_balance: float,
    quote_available: float,
    spread_pct: float | None,
    edge_pct: float,
    open_orders: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "respect_existing_protection", True)):
        return None
    coverage = analyze_existing_protection(open_orders=open_orders, instrument=args.instrument, base_balance=base_balance, latest_price=latest_price)
    if not coverage["has_any_protection"]:
        return None
    predicted = _float(forecast.get("predicted_price"))
    stop_price = _float(coverage.get("nearest_stop_price"))
    coverage_ratio = _float(coverage.get("sell_coverage_ratio"))
    required_coverage = float(getattr(args, "protected_position_coverage_ratio", 0.95))
    if coverage_ratio < required_coverage:
        plan = protection_plan(args=args, latest_price=latest_price, base_balance=base_balance, quote_available=quote_available, forecast=forecast, edge_pct=edge_pct, reason="existing_protection_incomplete_replace_or_add")
        plan["existing_protection"] = coverage
        return plan
    if stop_price > 0 and predicted > 0:
        beyond_stop_pct = (stop_price - predicted) / stop_price
    else:
        beyond_stop_pct = 0.0
    if stop_price > 0 and predicted < stop_price and beyond_stop_pct >= float(getattr(args, "sell_now_min_forecast_beyond_stop_pct", 0.01)):
        sell_price = round_price(latest_price * 0.999)
        return {
            **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
            "action": "sell_spot",
            "reason": "bearish_forecast_breaks_materially_below_existing_stop",
            "entry_order": {"side": "sell", "type": "limit", "amount": round_down(min(base_balance, float(args.max_base_position)), 6), "price": sell_price},
            "protection": {},
            "existing_protection": coverage,
            "protection_decision": {
                "decision": "sell_now",
                "predicted_below_stop_pct": beyond_stop_pct,
                "policy": "Existing protection is respected unless the forecast is materially below the stop, making immediate risk reduction preferable.",
            },
        }
    if stop_price > 0 and latest_price > stop_price:
        current_stop_gap_pct = (latest_price - stop_price) / latest_price
    else:
        current_stop_gap_pct = None
    return {
        **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
        "action": "hold_with_existing_protection",
        "reason": "bearish_forecast_existing_protection_covers_position",
        "entry_order": None,
        "protection": {},
        "existing_protection": coverage,
        "protection_decision": {
            "decision": "respect_existing_protection",
            "current_stop_gap_pct": current_stop_gap_pct,
            "policy": "Prior protective sell orders cover the existing ETH position, so the agent does not override them with an immediate sell unless the forecast materially breaks below the stop or coverage is incomplete.",
        },
    }


def analyze_existing_protection(*, open_orders: list[dict[str, Any]], instrument: str, base_balance: float, latest_price: float) -> dict[str, Any]:
    sell_orders = []
    stop_orders = []
    take_profit_orders = []
    for order in open_orders:
        if str(order.get("instrument_name") or "").upper() != instrument.upper():
            continue
        if str(order.get("direction") or "").lower() != "sell":
            continue
        state = str(order.get("order_state") or "").lower()
        if state not in {"open", "untriggered"}:
            continue
        amount = max(0.0, _float(order.get("amount")) - _float(order.get("filled_amount")))
        if amount <= 0:
            continue
        order_type = str(order.get("order_type") or "").lower()
        payload = {
            "order_id": order.get("order_id"),
            "order_type": order_type,
            "order_state": order.get("order_state"),
            "amount": amount,
            "price": _float(order.get("price")) or None,
            "trigger_price": _float(order.get("trigger_price")) or None,
            "meaning": "stop_loss" if "stop" in order_type else "take_profit" if order_type == "limit" and _float(order.get("price")) > latest_price else "sell_order",
        }
        sell_orders.append(payload)
        if "stop" in order_type:
            stop_orders.append(payload)
        elif order_type == "limit" and _float(order.get("price")) > latest_price:
            take_profit_orders.append(payload)
    sell_amount = sum(float(order["amount"]) for order in sell_orders)
    stop_amount = sum(float(order["amount"]) for order in stop_orders)
    take_profit_amount = sum(float(order["amount"]) for order in take_profit_orders)
    nearest_stop = max((_float(order.get("trigger_price")) for order in stop_orders if _float(order.get("trigger_price")) > 0), default=None)
    nearest_take_profit = min((_float(order.get("price")) for order in take_profit_orders if _float(order.get("price")) > 0), default=None)
    denominator = max(base_balance, 1e-12)
    return {
        "has_any_protection": bool(stop_orders or take_profit_orders),
        "has_stop_protection": bool(stop_orders),
        "has_take_profit_protection": bool(take_profit_orders),
        "sell_orders": sell_orders,
        "stop_orders": stop_orders,
        "take_profit_orders": take_profit_orders,
        "sell_coverage_amount": sell_amount,
        "stop_coverage_amount": stop_amount,
        "take_profit_coverage_amount": take_profit_amount,
        "sell_coverage_ratio": min(1.0, sell_amount / denominator),
        "stop_coverage_ratio": min(1.0, stop_amount / denominator),
        "take_profit_coverage_ratio": min(1.0, take_profit_amount / denominator),
        "nearest_stop_price": nearest_stop,
        "nearest_take_profit_price": nearest_take_profit,
        "base_balance": base_balance,
        "policy": "Existing sell stops and take-profit limits are treated as prior agent/user protection decisions and are respected when coverage is sufficient.",
    }


def scale_in_pullback_buy_plan(
    *,
    args: argparse.Namespace,
    latest_price: float,
    forecast: dict[str, Any],
    forecast_bundle: dict[str, Any],
    base_balance: float,
    quote_available: float,
    spread_pct: float | None,
    edge_pct: float,
    open_orders: list[dict[str, Any]],
    quote: dict[str, float | None],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "enable_scale_in_pullback", True)):
        return None
    setup = ((forecast_bundle.get("mean_reversion_dip_buy") or {}).get("best_setup") or {})
    if not setup:
        return None
    entry_price = _float(setup.get("entry_price"))
    stop_price = _float(setup.get("stop_price"))
    target_price = _float(setup.get("target_price"))
    reward_risk = _float(setup.get("reward_risk"))
    reversal_probability = _float(setup.get("reversal_probability"))
    average_entry = _float(getattr(args, "spot_average_entry_price", None))
    reasons = []
    if has_open_spot_entry(open_orders, args.instrument):
        reasons.append("open_scale_in_entry_order_exists")
    if not bool(setup.get("allowed")):
        reasons.append("dip_buy_setup_not_allowed")
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        reasons.append("invalid_pullback_prices")
    if entry_price >= latest_price:
        reasons.append("pullback_entry_not_below_market")
    if reward_risk < float(getattr(args, "pullback_min_reward_risk", 1.25)):
        reasons.append("scale_in_reward_risk_too_low")
    if reversal_probability < float(getattr(args, "pullback_min_reversal_probability", 0.45)):
        reasons.append("scale_in_reversal_probability_too_low")
    if entry_price > 0 and latest_price > 0 and (latest_price - entry_price) / latest_price > float(getattr(args, "pullback_max_entry_distance_pct", 0.05)):
        reasons.append("scale_in_entry_too_far_below_market")
    if target_price <= entry_price or stop_price >= entry_price:
        reasons.append("invalid_scale_in_stop_target_geometry")
    if average_entry <= 0 and not bool(getattr(args, "allow_scale_in_without_entry_price", False)):
        reasons.append("missing_existing_average_entry_price")
    if average_entry > 0:
        existing_loss_pct = max(0.0, (average_entry - latest_price) / average_entry)
        entry_discount_pct = (average_entry - entry_price) / average_entry
        if existing_loss_pct > float(getattr(args, "scale_in_max_existing_loss_pct", 0.06)):
            reasons.append("existing_position_loss_too_large_to_average_down")
        if entry_discount_pct < float(getattr(args, "scale_in_min_discount_from_entry_pct", 0.02)):
            reasons.append("pullback_entry_not_discounted_enough_vs_average_entry")
    else:
        existing_loss_pct = None
        entry_discount_pct = None
    remaining_base_capacity = max(0.0, float(args.max_base_position) - base_balance)
    max_addition_by_fraction = max(0.0, base_balance * float(getattr(args, "scale_in_max_addition_fraction", 0.35)))
    max_notional = float(getattr(args, "scale_in_max_notional_usdc", None) or args.max_notional_usdc)
    max_notional = min(max_notional, quote_available)
    if entry_price > 0:
        amount = round_down(min(remaining_base_capacity, max_addition_by_fraction, max_notional / entry_price), 6)
    else:
        amount = 0.0
    if remaining_base_capacity <= 0:
        reasons.append("max_base_position_reached")
    if amount < float(args.min_order_base_amount):
        reasons.append("scale_in_amount_below_min_order")
    if reasons:
        plan = hold_plan("scale_in_pullback_blocked", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
        plan["scale_in_pullback"] = {
            "setup": setup,
            "average_entry_price": average_entry if average_entry > 0 else None,
            "existing_loss_pct": existing_loss_pct,
            "entry_discount_pct": entry_discount_pct,
            "blocking_reasons": reasons,
            "policy": "Scale-in is blocked unless the existing position loss, entry discount, exposure, and pullback reversal gates all pass.",
        }
        return plan
    entry_price = round_price(entry_price)
    stop_pct = max(0.0, (entry_price - stop_price) / entry_price)
    take_profit_pct = max(0.0, (target_price - entry_price) / entry_price)
    protection_args = argparse.Namespace(**vars(args))
    protection_args.stop_loss_pct = stop_pct if stop_pct > 0 else float(args.stop_loss_pct)
    protection_args.take_profit_pct = take_profit_pct if take_profit_pct > 0 else float(args.take_profit_pct)
    return {
        **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
        "action": "place_scale_in_pullback_buy",
        "reason": "existing_eth_strict_scale_in_at_lower_reversal_entry",
        "entry_order": {"side": "buy", "type": "limit", "amount": amount, "price": entry_price},
        "protection": {},
        "post_fill_protection_plan": protection_orders(args=protection_args, amount=amount, entry_reference=entry_price),
        "scale_in_pullback": {
            "setup": setup,
            "average_entry_price": average_entry if average_entry > 0 else None,
            "existing_loss_pct": existing_loss_pct,
            "entry_discount_pct": entry_discount_pct,
            "remaining_base_capacity": remaining_base_capacity,
            "max_addition_by_fraction": max_addition_by_fraction,
            "max_notional_usdc": max_notional,
            "policy": "Strict average-down only: add ETH at a lower limit after exposure, loss, discount, reward/risk, and reversal gates pass. Protection is submitted only after the scale-in buy fills.",
            "quote_reference": quote,
        },
    }


def pullback_buy_plan(
    *,
    args: argparse.Namespace,
    latest_price: float,
    forecast: dict[str, Any],
    forecast_bundle: dict[str, Any],
    base_balance: float,
    quote_available: float,
    spread_pct: float | None,
    edge_pct: float,
    open_orders: list[dict[str, Any]],
    quote: dict[str, float | None],
) -> dict[str, Any] | None:
    if not bool(getattr(args, "enable_pullback_buy", True)):
        return None
    if has_open_spot_entry(open_orders, args.instrument):
        return hold_plan("open_pullback_entry_order_exists", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
    setup = ((forecast_bundle.get("mean_reversion_dip_buy") or {}).get("best_setup") or {})
    if not setup:
        return None
    entry_price = _float(setup.get("entry_price"))
    stop_price = _float(setup.get("stop_price"))
    target_price = _float(setup.get("target_price"))
    reward_risk = _float(setup.get("reward_risk"))
    reversal_probability = _float(setup.get("reversal_probability"))
    if entry_price <= 0 or stop_price <= 0 or target_price <= 0:
        return None
    if entry_price >= latest_price:
        return None
    distance_pct = (latest_price - entry_price) / latest_price
    reasons = []
    if not bool(setup.get("allowed")):
        reasons.append("dip_buy_setup_not_allowed")
    if reward_risk < float(getattr(args, "pullback_min_reward_risk", 1.25)):
        reasons.append("pullback_reward_risk_too_low")
    if reversal_probability < float(getattr(args, "pullback_min_reversal_probability", 0.45)):
        reasons.append("pullback_reversal_probability_too_low")
    if distance_pct > float(getattr(args, "pullback_max_entry_distance_pct", 0.05)):
        reasons.append("pullback_entry_too_far_below_market")
    if target_price <= entry_price or stop_price >= entry_price:
        reasons.append("invalid_pullback_stop_target_geometry")
    if reasons:
        plan = hold_plan("pullback_buy_blocked", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
        plan["pullback_buy"] = {"setup": setup, "distance_pct": distance_pct, "blocking_reasons": reasons}
        return plan
    max_notional = min(float(args.max_notional_usdc), quote_available)
    amount = round_down(max_notional / entry_price, 6)
    if amount < float(args.min_order_base_amount):
        return hold_plan("not_enough_quote_available_for_pullback_order", args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct)
    entry_price = round_price(entry_price)
    stop_pct = max(0.0, (entry_price - stop_price) / entry_price)
    take_profit_pct = max(0.0, (target_price - entry_price) / entry_price)
    protection_args = argparse.Namespace(**vars(args))
    protection_args.stop_loss_pct = stop_pct if stop_pct > 0 else float(args.stop_loss_pct)
    protection_args.take_profit_pct = take_profit_pct if take_profit_pct > 0 else float(args.take_profit_pct)
    return {
        **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
        "action": "place_pullback_buy",
        "reason": "bearish_forecast_wait_for_lower_reversal_entry",
        "entry_order": {"side": "buy", "type": "limit", "amount": amount, "price": entry_price},
        "protection": {},
        "post_fill_protection_plan": protection_orders(args=protection_args, amount=amount, entry_reference=entry_price),
        "pullback_buy": {
            "setup": setup,
            "distance_pct": distance_pct,
            "policy": "Place a lower limit buy only if the forecast downside zone, reward/risk, and reversal probability clear the pullback gate. Protective sell orders are submitted only after the buy is filled.",
            "quote_reference": quote,
        },
    }


def protection_plan(*, args: argparse.Namespace, latest_price: float, base_balance: float, quote_available: float, forecast: dict[str, Any], edge_pct: float, reason: str) -> dict[str, Any]:
    amount = round_down(min(base_balance, float(args.max_base_position)), 6)
    return {
        **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=None, edge_pct=edge_pct),
        "action": "protect_existing_position" if amount >= float(args.min_order_base_amount) else "hold",
        "reason": reason if amount >= float(args.min_order_base_amount) else "existing_position_below_min_order",
        "entry_order": None,
        "protection": protection_orders(args=args, amount=amount, entry_reference=latest_price),
    }


def protection_orders(*, args: argparse.Namespace, amount: float, entry_reference: float) -> dict[str, Any]:
    return {
        "take_profit": {
            "side": "sell",
            "type": "limit",
            "amount": amount,
            "price": round_price(entry_reference * (1.0 + float(args.take_profit_pct))),
            "meaning": "Take profit if ETH_USDC rises to the limit price.",
        },
        "stop_loss": {
            "side": "sell",
            "type": "stop_market",
            "amount": amount,
            "trigger_price": round_price(entry_reference * (1.0 - float(args.stop_loss_pct))),
            "trigger": "index_price",
            "meaning": "Protective stop if ETH_USDC falls to the trigger price.",
        },
    }


def execute_plan(*, broker: DeribitLiveSpotBroker, plan: dict[str, Any], open_orders: list[dict[str, Any]], label_base: str, replace_protection: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    instrument = str(plan["instrument"])
    if replace_protection:
        for order in open_orders:
            if order.get("instrument_name") == instrument and order.get("direction") == "sell":
                results.append({"action": "cancel_existing_sell_order", "order_id": order.get("order_id"), "result": broker.cancel_order(str(order.get("order_id")))})
    entry = plan.get("entry_order")
    if entry:
        results.append(
            {
                "action": "submit_entry_order",
                "payload": entry,
                "result": broker.submit_spot_order(
                    side=entry["side"],
                    instrument_name=instrument,
                    amount=float(entry["amount"]),
                    order_type=entry["type"],
                    price=entry.get("price"),
                    label=f"{label_base}-entry",
                ),
            }
        )
    for name, order in (plan.get("protection") or {}).items():
        results.append(
            {
                "action": f"submit_{name}_order",
                "payload": order,
                "result": broker.submit_spot_order(
                    side=order["side"],
                    instrument_name=instrument,
                    amount=float(order["amount"]),
                    order_type=order["type"],
                    price=order.get("price"),
                    trigger_price=order.get("trigger_price"),
                    trigger=order.get("trigger"),
                    label=f"{label_base}-{name}",
                ),
            }
        )
    if entry and entry.get("side") == "buy" and plan.get("post_fill_protection_plan"):
        results.append(
            {
                "action": "post_fill_protection_pending",
                "reason": "protection_not_submitted_until_buy_fill_is_confirmed",
                "protection_plan": plan.get("post_fill_protection_plan"),
            }
        )
    return results


def write_spot_agent_report(record: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{record['instrument']}_spot_agent_report.json"
    payload = json.dumps(strict_json(record), indent=2, default=str, allow_nan=False)
    path.write_text(payload, encoding="utf-8")
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    checked = str(record.get("checked_at") or datetime.now(UTC).isoformat()).replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")
    (snapshots / f"{record['instrument']}_spot_agent_report_{checked}.json").write_text(payload, encoding="utf-8")
    return path


def summary(record: dict[str, Any]) -> dict[str, Any]:
    decision = record.get("decision") or {}
    return {
        "instrument": record.get("instrument"),
        "action": decision.get("action"),
        "reason": decision.get("reason"),
        "latest_price": (record.get("market") or {}).get("latest_price"),
        "forecast_direction": (record.get("forecast") or {}).get("expected_direction"),
        "forecast_price": (record.get("forecast") or {}).get("predicted_price"),
        "execute_live_orders": (record.get("safety") or {}).get("execute_live_orders"),
        "order_result_count": len(record.get("order_results") or []),
    }


def base_plan(*, args: argparse.Namespace, latest_price: float, forecast: dict[str, Any], base_balance: float, quote_available: float, spread_pct: float | None, edge_pct: float | None) -> dict[str, Any]:
    return {
        "instrument": args.instrument.upper(),
        "latest_price": latest_price,
        "forecast_price": forecast.get("predicted_price"),
        "forecast_direction": forecast.get("expected_direction"),
        "edge_pct": edge_pct,
        "spread_pct": spread_pct,
        "base_balance": base_balance,
        "quote_available": quote_available,
        "limits": {
            "max_notional_usdc": float(args.max_notional_usdc),
            "max_base_position": float(args.max_base_position),
            "take_profit_pct": float(args.take_profit_pct),
            "stop_loss_pct": float(args.stop_loss_pct),
        },
    }


def hold_plan(reason: str, *, args: argparse.Namespace, latest_price: float, forecast: dict[str, Any], base_balance: float, quote_available: float, spread_pct: float | None, edge_pct: float | None = None) -> dict[str, Any]:
    return {
        **base_plan(args=args, latest_price=latest_price, forecast=forecast, base_balance=base_balance, quote_available=quote_available, spread_pct=spread_pct, edge_pct=edge_pct),
        "action": "hold",
        "reason": reason,
        "entry_order": None,
        "protection": {},
    }


def book_quote(order_book: dict[str, Any]) -> dict[str, float | None]:
    bids = order_book.get("bids") or []
    asks = order_book.get("asks") or []
    bid = _float(bids[0][0]) if bids else None
    ask = _float(asks[0][0]) if asks else None
    mid = (bid + ask) / 2.0 if bid and ask else None
    return {"bid": bid, "ask": ask, "mid": mid}


def spread_fraction(quote: dict[str, float | None]) -> float | None:
    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = quote.get("mid")
    if bid is None or ask is None or mid is None or mid <= 0:
        return None
    return (ask - bid) / mid


def dedupe_orders(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for row in rows:
        order_id = row.get("order_id")
        if order_id and order_id in seen:
            continue
        if order_id:
            seen.add(order_id)
        output.append(row)
    return output


def has_open_spot_entry(open_orders: list[dict[str, Any]], instrument: str) -> bool:
    for order in open_orders:
        if order.get("instrument_name") == instrument.upper() and order.get("direction") == "buy":
            return True
    return False


def account_subset(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("currency", "equity", "balance", "available_funds", "available_withdrawal_funds", "initial_margin", "maintenance_margin")
    return {key: row.get(key) for key in keys if key in row}


def _primary_forecast(plan: dict[str, Any]) -> dict[str, Any]:
    forecasts = list(plan.get("forecasts") or [])
    if not forecasts:
        raise RuntimeError("Forecast plan did not produce forecasts.")
    forecast = dict(sorted(forecasts, key=lambda row: float(row.get("horizon_hours") or 999))[0])
    latest = float(plan["latest_price"])
    predicted = float(forecast["predicted_price"])
    forecast["spot"] = latest
    forecast["expected_direction"] = "Upward" if predicted > latest else "Downward" if predicted < latest else "Flat"
    return forecast


def _forecast_for_decision(forecast: dict[str, Any], latest_price: float) -> dict[str, Any]:
    row = dict(forecast)
    predicted = _float(row.get("predicted_price"))
    row["spot"] = latest_price
    row["expected_direction"] = "Upward" if predicted > latest_price else "Downward" if predicted < latest_price else "Flat"
    return row


def _price_tail(prices: Any, max_rows: int = 360) -> list[dict[str, Any]]:
    rows = []
    frame = prices.tail(max_rows)
    close_column = "close" if "close" in frame.columns else "Close" if "Close" in frame.columns else None
    if close_column is None:
        return rows
    for timestamp, row in frame.iterrows():
        price = _float(row.get(close_column))
        if price <= 0:
            continue
        rows.append({"time": str(timestamp), "close": price})
    return rows


def _close_values(prices: Any) -> list[float]:
    if not hasattr(prices, "columns"):
        return []
    close_column = "close" if "close" in prices.columns else "Close" if "Close" in prices.columns else None
    if close_column is None:
        return []
    values = []
    for value in prices[close_column].tolist():
        price = _float(value)
        if price > 0:
            values.append(price)
    return values


def round_price(value: float) -> float:
    return round(float(value), 2)


def round_down(value: float, decimals: int) -> float:
    factor = 10**decimals
    return math_floor(float(value) * factor) / factor


def math_floor(value: float) -> int:
    return int(value // 1)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def strict_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strict_json(item) for item in value]
    if isinstance(value, tuple):
        return [strict_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
