from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.deribit_broker import DeribitOptionsBroker, DeribitTestnetBroker


@dataclass(frozen=True)
class DeribitOptionExecutionConfig:
    currency: str
    underlying_currency: str | None = None
    risk_profile: str = "aggressive"
    min_dte: int = 1
    max_dte: int = 14
    allow_0dte: bool = False
    max_total_debit_usd: float = 1500.0
    max_position_equity_pct: float = 0.02
    risk_budget_pct: float = 0.0075
    max_spread_pct: float = 0.30
    max_contracts: float = 1.0
    min_contract_amount: float | None = None
    target_delta: float = 0.45
    max_delta_distance: float = 0.30
    greeks_mode: str = "required"
    max_theta_edge_ratio: float = 0.75
    max_theta_premium_pct_per_day: float = 0.35
    min_option_volume: float = 0.0
    enable_fibonacci: bool = True
    require_fibonacci_confluence: bool = False
    max_fibonacci_distance_pct: float = 0.006
    enable_chart_patterns: bool = True
    block_chart_pattern_conflicts: bool = True
    min_chart_pattern_confidence: float = 0.70
    enable_market_regime_filter: bool = True
    allow_range_edge_reversal_entry: bool = False
    market_regime_lookback_rows: int = 120
    market_regime_breakout_buffer_pct: float = 0.001
    market_regime_middle_zone_width: float = 0.30
    min_trend_strength_pct: float = 0.003
    enable_impulse_entry: bool = True
    impulse_lookback_bars: int = 12
    min_impulse_move_pct: float = 0.006
    min_impulse_directional_bars: int = 7
    enable_late_entry_filter: bool = True
    max_late_entry_move_pct: float = 0.018
    max_ema_extension_pct: float = 0.010
    exhaustion_reversal_bars: int = 2
    min_hours_to_expiry_for_entry: float = 18.0
    close_before_expiry_hours: float = 12.0
    entry_expiry_buffer_hours: float = 4.0
    target_moneyness: float = 0.02
    max_moneyness_distance: float = 0.12
    limit_price_offset_pct: float = 0.03
    stop_loss_pct: float = 0.35
    take_profit_pct: float = 0.55
    abandon_entry_after_seconds: int = 300
    entry_order_policy: str = "auto"
    exit_order_policy: str = "auto"


def build_deribit_option_trade_plan(
    *,
    broker: DeribitTestnetBroker,
    currency: str,
    underlying_price_usd: float,
    forecast: dict[str, Any],
    account: dict[str, Any],
    config: DeribitOptionExecutionConfig,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    option_type = _option_type_from_forecast(forecast)
    if option_type is None:
        return {"action": "hold", "reason": "forecast_has_no_directional_edge", "forecast": forecast}
    fibonacci = forecast.get("fibonacci_analysis") if isinstance(forecast.get("fibonacci_analysis"), dict) else {}
    if config.enable_fibonacci and config.require_fibonacci_confluence and fibonacci.get("confirmation") == "conflict":
        return {
            "action": "hold",
            "reason": "fibonacci_conflicts_with_forecast",
            "currency": currency.upper(),
            "option_type": option_type,
            "fibonacci_analysis": fibonacci,
            "forecast": forecast,
        }
    chart_patterns = forecast.get("chart_pattern_analysis") if isinstance(forecast.get("chart_pattern_analysis"), dict) else {}
    pattern_summary = chart_patterns.get("summary") if isinstance(chart_patterns.get("summary"), dict) else {}
    if (
        config.enable_chart_patterns
        and config.block_chart_pattern_conflicts
        and pattern_summary.get("permission") == "conflict"
        and (_float_or_none(pattern_summary.get("dominant_confidence")) or 0.0) >= float(config.min_chart_pattern_confidence)
        and pattern_summary.get("dominant_status") == "confirmed"
    ):
        return {
            "action": "hold",
            "reason": "chart_pattern_conflicts_with_forecast",
            "currency": currency.upper(),
            "option_type": option_type,
            "chart_pattern_analysis": chart_patterns,
            "fibonacci_analysis": fibonacci if config.enable_fibonacci else {"enabled": False},
            "forecast": forecast,
        }
    regime = forecast.get("market_regime") if isinstance(forecast.get("market_regime"), dict) else {}
    if config.enable_market_regime_filter and regime and regime.get("allow_directional_entry") is False:
        return {
            "action": "hold",
            "reason": "market_regime_blocks_directional_entry",
            "currency": currency.upper(),
            "option_type": option_type,
            "market_regime": regime,
            "fibonacci_analysis": fibonacci if config.enable_fibonacci else {"enabled": False},
            "chart_pattern_analysis": chart_patterns if config.enable_chart_patterns else {"enabled": False},
            "forecast": forecast,
        }
    instruments = broker.instruments(currency=currency, kind="option", expired=False)
    underlying_prefix = _option_instrument_prefix(currency=currency, underlying_currency=config.underlying_currency)
    candidates = score_deribit_option_contracts(
        broker=broker,
        instruments=instruments,
        instrument_prefix=underlying_prefix,
        underlying_price_usd=underlying_price_usd,
        forecast=forecast,
        option_type=option_type,
        config=config,
        now=now,
    )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        return {
            "action": "hold",
            "reason": "no_contract_passed_execution_gates",
            "currency": currency.upper(),
            "option_type": option_type,
            "candidate_count": len(candidates),
            "top_rejections": candidates[:10],
        }
    selected = None
    sizing = None
    for candidate in accepted:
        candidate_sizing = size_deribit_option_position(
            entry_limit_price_base=float(candidate["limit_price"]),
            underlying_price_usd=underlying_price_usd,
            account=account,
            config=config,
            premium_usd_per_contract=_float_or_none(candidate.get("estimated_premium_usd_per_contract")),
            min_trade_amount=_float_or_none(candidate.get("min_trade_amount")),
        )
        candidate["sizing"] = candidate_sizing
        if candidate_sizing["amount"] > 0:
            selected = candidate
            sizing = candidate_sizing
            break
    if selected is None or sizing is None:
        best = accepted[0]
        best["sizing"] = size_deribit_option_position(
            entry_limit_price_base=float(best["limit_price"]),
            underlying_price_usd=underlying_price_usd,
            account=account,
            config=config,
            premium_usd_per_contract=_float_or_none(best.get("estimated_premium_usd_per_contract")),
            min_trade_amount=_float_or_none(best.get("min_trade_amount")),
        )
        return {
            "action": "hold",
            "reason": "position_size_below_minimum_contract_amount",
            "selected_contract": best,
            "sizing": best["sizing"],
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "top_candidates": accepted[:5],
        }
    entry_order = choose_deribit_entry_order(selected=selected, amount=float(sizing["amount"]), config=config)
    exit_plan = build_deribit_exit_plan(
        instrument_name=selected["instrument_name"],
        entry_limit_price_base=float(entry_order["price"]),
        amount=float(sizing["amount"]),
        config=config,
    )
    return {
        "action": "buy_option",
        "currency": currency.upper(),
        "option_type": option_type,
        "fibonacci_analysis": fibonacci if config.enable_fibonacci else {"enabled": False},
        "chart_pattern_analysis": chart_patterns if config.enable_chart_patterns else {"enabled": False},
        "market_regime": regime if config.enable_market_regime_filter else {"enabled": False},
        "selected_contract": selected,
        "order": entry_order,
        "risk": {
            "estimated_debit_usd": round(float(sizing["estimated_debit_usd"]), 2),
            "estimated_debit_base": sizing["estimated_debit_base"],
            "risk_budget_pct": config.risk_budget_pct,
            "max_total_debit_usd": config.max_total_debit_usd,
            "max_position_equity_pct": config.max_position_equity_pct,
            "max_contracts": config.max_contracts,
            "policy": "Deribit option entries use real instruments, live order books, base-currency limit prices, and amount sizing from account equity and live premium.",
        },
        "exit_plan": exit_plan,
        "sizing": sizing,
        "order_policy_decision": {
            "entry_order_policy": config.entry_order_policy,
            "exit_order_policy": config.exit_order_policy,
            "available_order_types_considered": ["limit", "post_only_limit"],
            "selected_entry_type": entry_order["type"],
            "note": "Deribit crypto-option exits are managed by the agent loop with limit reduce-only orders when stop/take-profit triggers are reached.",
        },
        "abandon_plan": {
            "cancel_unfilled_entry_after_seconds": int(config.abandon_entry_after_seconds),
            "replace_policy": "Cancel stale entries and recompute contract, order book, and limit from fresh Deribit quotes.",
        },
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "top_candidates": accepted[:5],
    }


def score_deribit_option_contracts(
    *,
    broker: DeribitTestnetBroker,
    instruments: list[dict[str, Any]],
    instrument_prefix: str | None = None,
    underlying_price_usd: float,
    forecast: dict[str, Any],
    option_type: str,
    config: DeribitOptionExecutionConfig,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    scored: list[dict[str, Any]] = []
    forecast_price = _float_or_none(forecast.get("predicted_price"))
    for instrument in instruments:
        name = str(instrument.get("instrument_name") or "")
        if instrument_prefix and not name.upper().startswith(instrument_prefix.upper()):
            continue
        if str(instrument.get("option_type") or "").lower() != option_type:
            continue
        expiry = _expiry_from_ms(instrument.get("expiration_timestamp"))
        dte = None if expiry is None else max(0, (expiry.date() - now.date()).days)
        hours_to_expiry = None if expiry is None else (expiry - now).total_seconds() / 3600.0
        if dte is None or dte > int(config.max_dte) + 3:
            continue
        order_book = broker.order_book(name, depth=5) if name else {}
        quote = _book_quote(order_book)
        greeks = order_book.get("greeks") or {}
        stats = order_book.get("stats") or {}
        bid = quote.get("bid")
        ask = quote.get("ask")
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        spread_pct = None if bid is None or ask is None or mid is None else (ask - bid) / max(mid, 1e-12)
        strike = _float_or_none(instrument.get("strike"))
        min_trade_amount = _float_or_none(instrument.get("min_trade_amount"))
        tick_size = _float_or_none(instrument.get("tick_size")) or 0.0001
        reasons = []
        if not name:
            reasons.append("missing_instrument_name")
        if instrument.get("is_active") is False:
            reasons.append("contract_not_active")
        if strike is None:
            reasons.append("missing_strike")
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            reasons.append("missing_live_bid_ask")
        if spread_pct is None or spread_pct > config.max_spread_pct:
            reasons.append("spread_too_wide")
        if dte is not None:
            if dte == 0 and not config.allow_0dte:
                reasons.append("zero_dte_blocked")
            if dte < config.min_dte:
                reasons.append("dte_below_min")
            if dte > config.max_dte:
                reasons.append("dte_above_max")
        horizon_hours = _float_or_none(forecast.get("horizon_hours"))
        required_entry_hours = max(
            float(config.min_hours_to_expiry_for_entry),
            float(config.close_before_expiry_hours) + float(config.entry_expiry_buffer_hours) + max(float(horizon_hours or 0.0), 0.0),
        )
        if hours_to_expiry is None:
            reasons.append("missing_hours_to_expiry")
        elif hours_to_expiry <= required_entry_hours:
            reasons.append("expiry_too_close_for_new_entry")
            if horizon_hours is not None:
                reasons.append("expiry_too_close_for_entry_horizon")
        if min_trade_amount is None or min_trade_amount <= 0:
            reasons.append("missing_min_trade_amount")
        delta = _float_or_none(greeks.get("delta"))
        gamma = _float_or_none(greeks.get("gamma"))
        theta = _float_or_none(greeks.get("theta"))
        vega = _float_or_none(greeks.get("vega"))
        rho = _float_or_none(greeks.get("rho"))
        volume = _float_or_none(stats.get("volume"))
        premium_multiplier_usd = _premium_multiplier_usd(instrument=instrument, underlying_price_usd=underlying_price_usd)
        premium_usd = None if ask is None else ask * premium_multiplier_usd
        forecast_edge_usd = None
        theta_decay_usd_for_horizon = None
        theta_premium_pct_per_day = None
        if forecast_price is not None and delta is not None:
            forecast_edge_usd = abs(forecast_price - underlying_price_usd) * abs(delta)
        if theta is not None and horizon_hours is not None:
            theta_decay_usd_for_horizon = abs(theta) * max(horizon_hours, 0.0) / 24.0
        if theta is not None and premium_usd is not None and premium_usd > 0:
            theta_premium_pct_per_day = abs(theta) / premium_usd
        greeks_required = str(config.greeks_mode).lower() == "required"
        if greeks_required and (delta is None or gamma is None or theta is None or vega is None):
            reasons.append("missing_greeks")
        if greeks_required and delta is not None and abs(abs(delta) - config.target_delta) > config.max_delta_distance:
            reasons.append("delta_too_far_from_target")
        if volume is not None and volume < config.min_option_volume:
            reasons.append("volume_below_minimum")
        if (
            greeks_required
            and
            theta_decay_usd_for_horizon is not None
            and forecast_edge_usd is not None
            and theta_decay_usd_for_horizon > forecast_edge_usd * config.max_theta_edge_ratio
        ):
            reasons.append("theta_decay_too_large_vs_forecast_edge")
        if greeks_required and theta_premium_pct_per_day is not None and theta_premium_pct_per_day > config.max_theta_premium_pct_per_day:
            reasons.append("theta_too_large_vs_premium")
        intrinsic_ok = True
        if strike is not None and forecast_price is not None:
            intrinsic_ok = forecast_price > strike if option_type == "call" else forecast_price < strike
            if not intrinsic_ok:
                reasons.append("forecast_not_beyond_strike")
        limit_price = None
        if ask is not None and mid is not None:
            limit_price = _round_to_tick(min(ask, mid * (1.0 + config.limit_price_offset_pct)), tick_size)
        score = _candidate_score(
            strike=strike,
            underlying_price_usd=underlying_price_usd,
            forecast_price=forecast_price,
            spread_pct=spread_pct,
            delta=delta,
            gamma=gamma,
            theta_premium_pct_per_day=theta_premium_pct_per_day,
            dte=dte,
            option_type=option_type,
            config=config,
            fibonacci=forecast.get("fibonacci_analysis") if isinstance(forecast.get("fibonacci_analysis"), dict) else None,
        )
        scored.append(
            {
                "instrument_name": name,
                "option_type": option_type,
                "strike": strike,
                "expiration_timestamp": instrument.get("expiration_timestamp"),
                "expiration_utc": expiry.isoformat() if expiry else None,
                "dte": dte,
                "hours_to_expiry": None if hours_to_expiry is None else round(hours_to_expiry, 2),
                "required_hours_to_expiry_for_entry": round(required_entry_hours, 2),
                "min_trade_amount": min_trade_amount,
                "tick_size": tick_size,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "underlying_price_usd": underlying_price_usd,
                "estimated_premium_usd_per_contract": premium_usd,
                "premium_currency": _premium_currency(instrument),
                "premium_multiplier_usd": premium_multiplier_usd,
                "greeks": {
                    "mode": str(config.greeks_mode).lower(),
                    "delta": delta,
                    "gamma": gamma,
                    "theta": theta,
                    "vega": vega,
                    "rho": rho,
                    "theta_decay_usd_for_horizon": theta_decay_usd_for_horizon,
                    "theta_premium_pct_per_day": theta_premium_pct_per_day,
                    "forecast_edge_usd_delta_adjusted": forecast_edge_usd,
                    "horizon_hours": horizon_hours,
                },
                "liquidity": {
                    "volume": volume,
                    "volume_usd": _float_or_none(stats.get("volume_usd")),
                    "price_change": _float_or_none(stats.get("price_change")),
                    "open_interest": _float_or_none(order_book.get("open_interest")),
                },
                "limit_price": limit_price,
                "accepted": not reasons and limit_price is not None,
                "reasons": reasons,
                "score": score,
                "order_book": {
                    "timestamp": order_book.get("timestamp"),
                    "mark_price": order_book.get("mark_price"),
                    "underlying_price": order_book.get("underlying_price"),
                    "open_interest": order_book.get("open_interest"),
                },
            }
        )
    scored.sort(key=lambda item: (not item["accepted"], item["score"], item.get("estimated_premium_usd_per_contract") or float("inf")))
    return scored


def size_deribit_option_position(
    *,
    entry_limit_price_base: float,
    underlying_price_usd: float,
    account: dict[str, Any],
    config: DeribitOptionExecutionConfig,
    premium_usd_per_contract: float | None = None,
    min_trade_amount: float | None,
) -> dict[str, Any]:
    premium_usd = max(float(premium_usd_per_contract) if premium_usd_per_contract is not None else float(entry_limit_price_base) * max(float(underlying_price_usd), 1e-12), 1e-12)
    equity_base = _float_or_none(account.get("equity")) or _float_or_none(account.get("balance")) or 0.0
    account_currency = str(account.get("currency") or config.currency or "").upper()
    equity_usd = max(equity_base, 0.0) if account_currency in {"USD", "USDC", "USDT", "USDE"} else max(equity_base, 0.0) * max(float(underlying_price_usd), 0.0)
    risk_budget = equity_usd * float(config.risk_budget_pct) if equity_usd > 0 else premium_usd
    exposure_budget = equity_usd * float(config.max_position_equity_pct) if equity_usd > 0 else float(config.max_total_debit_usd)
    budget = max(0.0, min(float(config.max_total_debit_usd), exposure_budget, max(risk_budget, premium_usd)))
    raw_amount = min(float(config.max_contracts), budget / max(premium_usd, 1e-12))
    minimum = float(config.min_contract_amount or min_trade_amount or 1.0)
    amount = _round_amount(raw_amount, minimum)
    if amount < minimum:
        amount = 0.0
    return {
        "amount": amount,
        "min_trade_amount": minimum,
        "premium_usd_per_contract": round(premium_usd, 2),
        "premium_base_per_contract": entry_limit_price_base,
        "account_equity_base": equity_base,
        "account_equity_usd": round(equity_usd, 2),
        "risk_budget_usd": round(risk_budget, 2),
        "exposure_budget_usd": round(exposure_budget, 2),
        "max_total_debit_usd": float(config.max_total_debit_usd),
        "max_contracts": float(config.max_contracts),
        "budget_usd": round(budget, 2),
        "estimated_debit_usd": round(amount * premium_usd, 2),
        "estimated_debit_base": round(amount * entry_limit_price_base, 8),
        "reason": "sized_from_equity_risk_budget_and_live_premium" if amount > 0 else "budget_below_minimum_trade_amount",
    }


def choose_deribit_entry_order(
    *,
    selected: dict[str, Any],
    amount: float,
    config: DeribitOptionExecutionConfig,
) -> dict[str, Any]:
    if config.entry_order_policy not in {"auto", "limit", "post_only_limit"}:
        raise ValueError("Deribit options entry policy supports auto, limit, or post_only_limit.")
    return {
        "instrument_name": selected["instrument_name"],
        "side": "buy",
        "type": "limit",
        "amount": amount,
        "price": float(selected["limit_price"]),
        "post_only": config.entry_order_policy == "post_only_limit",
        "reduce_only": False,
        "time_in_force": "good_til_cancelled",
        "policy_reason": "Crypto-option entries use Deribit limit orders from the live order book to control spread slippage.",
    }


def build_deribit_exit_plan(
    *,
    instrument_name: str,
    entry_limit_price_base: float,
    amount: float,
    config: DeribitOptionExecutionConfig,
) -> dict[str, Any]:
    entry = max(float(entry_limit_price_base), 1e-12)
    take_profit = entry * (1.0 + float(config.take_profit_pct))
    stop_trigger = entry * (1.0 - float(config.stop_loss_pct))
    return {
        "take_profit": {
            "instrument_name": instrument_name,
            "side": "sell",
            "type": "limit",
            "amount": float(amount),
            "price": round(take_profit, 8),
            "reduce_only": True,
            "trigger_policy": "submit_reduce_only_limit_when_bid_or_mark_reaches_price",
        },
        "stop_loss": {
            "instrument_name": instrument_name,
            "side": "sell",
            "type": "limit",
            "amount": float(amount),
            "price": round(max(stop_trigger, 1e-8), 8),
            "reduce_only": True,
            "trigger_policy": "submit_reduce_only_limit_when_bid_or_mark_falls_to_or_below_price",
        },
        "primary_exit": "agent_managed_stop_or_take_profit",
        "policy": "Deribit testnet option exits are managed continuously by the agent. It submits reduce-only limit exits only after the trigger is observed in fresh order-book data.",
    }


def submit_deribit_limit_order(
    broker: DeribitTestnetBroker | DeribitOptionsBroker,
    order: dict[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    if order.get("type") != "limit":
        raise ValueError("Deribit options agent only submits limit orders.")
    side = str(order.get("side"))
    normalized_price = _normalize_limit_price_for_instrument(
        broker=broker,
        instrument_name=str(order["instrument_name"]),
        price=float(order["price"]),
        side=side,
    )
    if side == "buy":
        return broker.buy_limit(
            instrument_name=str(order["instrument_name"]),
            amount=float(order["amount"]),
            price=normalized_price,
            label=label,
            post_only=bool(order.get("post_only")),
            reduce_only=bool(order.get("reduce_only")),
        )
    if side == "sell":
        return broker.sell_limit(
            instrument_name=str(order["instrument_name"]),
            amount=float(order["amount"]),
            price=normalized_price,
            label=label,
            post_only=bool(order.get("post_only")),
            reduce_only=bool(order.get("reduce_only", True)),
        )
    raise ValueError("Deribit order side must be buy or sell.")


def _normalize_limit_price_for_instrument(
    *,
    broker: DeribitTestnetBroker | DeribitOptionsBroker,
    instrument_name: str,
    price: float,
    side: str,
) -> float:
    if price <= 0:
        raise ValueError("Deribit option limit orders require a positive price.")
    tick_size = _instrument_tick_size(broker=broker, instrument_name=instrument_name, price=price)
    if tick_size is None:
        return float(price)
    if str(side).lower() == "sell":
        return _round_down_to_tick(price, tick_size)
    return _round_to_tick(price, tick_size)


def build_fibonacci_analysis(
    prices: pd.DataFrame,
    *,
    current_price: float,
    forecast: dict[str, Any],
    lookback_rows: int = 720,
    max_distance_pct: float = 0.006,
) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"enabled": True, "status": "unavailable", "reason": "missing_price_history"}
    close = _price_series(prices).dropna()
    if len(close) < 20:
        return {"enabled": True, "status": "unavailable", "reason": "not_enough_price_history", "rows": len(close)}
    window = close.tail(max(20, int(lookback_rows)))
    swing_high = float(window.max())
    swing_low = float(window.min())
    high_time = window.idxmax()
    low_time = window.idxmin()
    price_range = swing_high - swing_low
    if price_range <= max(abs(float(current_price)) * 0.001, 1e-9):
        return {
            "enabled": True,
            "status": "unavailable",
            "reason": "swing_range_too_small",
            "swing_high": swing_high,
            "swing_low": swing_low,
            "rows": len(window),
        }
    expected_return = _float_or_none(forecast.get("expected_return")) or 0.0
    direction = "bullish" if expected_return > 0 else "bearish" if expected_return < 0 else "flat"
    trend = "upswing" if window.index.get_loc(high_time) > window.index.get_loc(low_time) else "downswing"
    retracement_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    extension_ratios = [1.272, 1.618, 2.0]
    if trend == "upswing":
        retracements = {str(ratio): swing_high - price_range * ratio for ratio in retracement_ratios}
        extensions = {str(ratio): swing_low + price_range * ratio for ratio in extension_ratios}
    else:
        retracements = {str(ratio): swing_low + price_range * ratio for ratio in retracement_ratios}
        extensions = {str(ratio): swing_high - price_range * ratio for ratio in extension_ratios}
    below = {name: price for name, price in retracements.items() if price <= current_price}
    above = {name: price for name, price in retracements.items() if price >= current_price}
    nearest_support = max(below.values(), default=swing_low)
    nearest_resistance = min(above.values(), default=swing_high)
    target_pool = [price for price in list(extensions.values()) + list(retracements.values()) if (price > current_price if direction == "bullish" else price < current_price)]
    nearest_target = min(target_pool, key=lambda price: abs(price - current_price), default=None)
    support_distance_pct = abs(current_price - nearest_support) / max(abs(current_price), 1e-9)
    resistance_distance_pct = abs(nearest_resistance - current_price) / max(abs(current_price), 1e-9)
    confirmation = "neutral"
    reason = "price_not_near_relevant_fibonacci_level"
    if direction == "bullish":
        if support_distance_pct <= max_distance_pct:
            confirmation = "supportive"
            reason = "bullish_forecast_near_fibonacci_support"
        elif resistance_distance_pct <= max_distance_pct:
            confirmation = "conflict"
            reason = "bullish_forecast_near_fibonacci_resistance"
    elif direction == "bearish":
        if resistance_distance_pct <= max_distance_pct:
            confirmation = "supportive"
            reason = "bearish_forecast_near_fibonacci_resistance"
        elif support_distance_pct <= max_distance_pct:
            confirmation = "conflict"
            reason = "bearish_forecast_near_fibonacci_support"
    return {
        "enabled": True,
        "status": "ok",
        "direction": direction,
        "trend": trend,
        "confirmation": confirmation,
        "reason": reason,
        "lookback_rows": len(window),
        "current_price": float(current_price),
        "swing_high": swing_high,
        "swing_low": swing_low,
        "swing_high_time": str(high_time),
        "swing_low_time": str(low_time),
        "range": price_range,
        "retracements": {key: round(value, 6) for key, value in retracements.items()},
        "extensions": {key: round(value, 6) for key, value in extensions.items()},
        "nearest_support": round(nearest_support, 6),
        "nearest_resistance": round(nearest_resistance, 6),
        "support_distance_pct": support_distance_pct,
        "resistance_distance_pct": resistance_distance_pct,
        "nearest_target_price": None if nearest_target is None else round(float(nearest_target), 6),
        "max_distance_pct": float(max_distance_pct),
    }


def build_options_market_regime(
    prices: pd.DataFrame,
    *,
    current_price: float,
    forecast: dict[str, Any],
    lookback_rows: int = 120,
    breakout_buffer_pct: float = 0.001,
    middle_zone_width: float = 0.30,
    min_trend_strength_pct: float = 0.003,
    allow_range_edge_reversal_entry: bool = False,
    enable_impulse_entry: bool = True,
    impulse_lookback_bars: int = 12,
    min_impulse_move_pct: float = 0.006,
    min_impulse_directional_bars: int = 7,
    enable_late_entry_filter: bool = True,
    max_late_entry_move_pct: float = 0.018,
    max_ema_extension_pct: float = 0.010,
    exhaustion_reversal_bars: int = 2,
) -> dict[str, Any]:
    if prices is None or prices.empty:
        return {"enabled": True, "status": "unavailable", "reason": "missing_price_history", "allow_directional_entry": False}
    close = _price_series(prices).dropna()
    if len(close) < 30:
        return {"enabled": True, "status": "unavailable", "reason": "not_enough_price_history", "rows": len(close), "allow_directional_entry": False}
    window = close.tail(max(30, int(lookback_rows)))
    prior = window.iloc[:-1] if len(window) > 1 else window
    support = float(prior.min())
    resistance = float(prior.max())
    latest = float(current_price)
    price_range = max(resistance - support, 0.0)
    range_width_pct = price_range / max(abs(latest), 1e-9)
    range_position = 0.5 if price_range <= 0 else max(0.0, min(1.0, (latest - support) / price_range))
    fast = window.ewm(span=min(9, len(window)), adjust=False).mean()
    slow = window.ewm(span=min(21, len(window)), adjust=False).mean()
    ema_spread_pct = float(fast.iloc[-1] / max(float(slow.iloc[-1]), 1e-9) - 1.0)
    slope_window = min(15, max(3, len(window) // 4))
    slope_pct = float(window.iloc[-1] / max(float(window.iloc[-slope_window]), 1e-9) - 1.0)
    trend_strength_pct = abs(ema_spread_pct) + abs(slope_pct)
    forecast_return = _float_or_none(forecast.get("expected_return")) or 0.0
    forecast_direction = "up" if forecast_return > 0 else "down" if forecast_return < 0 else "flat"
    breakout_up = latest > resistance * (1.0 + max(0.0, float(breakout_buffer_pct)))
    breakout_down = latest < support * (1.0 - max(0.0, float(breakout_buffer_pct)))
    aligned_up = forecast_direction == "up" and (breakout_up or (ema_spread_pct > 0 and slope_pct > 0 and range_position >= 0.60))
    aligned_down = forecast_direction == "down" and (breakout_down or (ema_spread_pct < 0 and slope_pct < 0 and range_position <= 0.40))
    strong_trend = trend_strength_pct >= max(0.0, float(min_trend_strength_pct))
    middle_half_width = max(0.0, min(1.0, float(middle_zone_width))) / 2.0
    middle_low = 0.5 - middle_half_width
    middle_high = 0.5 + middle_half_width
    in_middle = middle_low <= range_position <= middle_high
    near_support = range_position <= 0.20
    near_resistance = range_position >= 0.80
    impulse = _options_impulse_signal(
        window,
        forecast_direction=forecast_direction,
        lookback_bars=int(impulse_lookback_bars),
        min_move_pct=float(min_impulse_move_pct),
        min_directional_bars=int(min_impulse_directional_bars),
    ) if enable_impulse_entry else {"enabled": False, "allow_entry": False}
    exhaustion = _options_exhaustion_signal(
        window,
        forecast_direction=forecast_direction,
        support=support,
        resistance=resistance,
        lookback_bars=int(impulse_lookback_bars),
        max_late_entry_move_pct=float(max_late_entry_move_pct),
        max_ema_extension_pct=float(max_ema_extension_pct),
        reversal_bars=int(exhaustion_reversal_bars),
    ) if enable_late_entry_filter else {"enabled": False, "late_entry_block": False}
    range_edge_reversal = bool(
        allow_range_edge_reversal_entry
        and (
            (forecast_direction == "up" and near_support and slope_pct > 0)
            or (forecast_direction == "down" and near_resistance and slope_pct < 0)
        )
    )
    if breakout_up:
        regime = "trend_up"
        breakout_status = "confirmed_breakout"
    elif breakout_down:
        regime = "trend_down"
        breakout_status = "confirmed_breakdown"
    elif strong_trend and aligned_up:
        regime = "trend_up"
        breakout_status = "trend_continuation"
    elif strong_trend and aligned_down:
        regime = "trend_down"
        breakout_status = "trend_continuation"
    elif bool(impulse.get("allow_entry")) and forecast_direction == "up":
        regime = "trend_up"
        breakout_status = "impulse_up"
    elif bool(impulse.get("allow_entry")) and forecast_direction == "down":
        regime = "trend_down"
        breakout_status = "impulse_down"
    elif in_middle:
        regime = "range_bound"
        breakout_status = "inside_range_middle"
    else:
        regime = "range_edge"
        breakout_status = "inside_range_edge"
    allow_entry = bool(
        (forecast_direction == "up" and regime == "trend_up" and strong_trend)
        or (forecast_direction == "down" and regime == "trend_down" and strong_trend)
        or bool(impulse.get("allow_entry"))
        or range_edge_reversal
    )
    if allow_entry and bool(exhaustion.get("late_entry_block")):
        allow_entry = False
        regime = "late_trend"
        breakout_status = "late_entry_exhaustion"
    reason = "directional_trend_confirmed" if allow_entry and not range_edge_reversal and not bool(impulse.get("allow_entry")) else "impulse_entry_confirmed" if bool(impulse.get("allow_entry")) else "range_edge_reversal_allowed" if allow_entry else "range_or_noise_without_confirmed_direction"
    if bool(exhaustion.get("late_entry_block")):
        reason = str(exhaustion.get("reason") or "late_entry_exhaustion")
    if in_middle and not allow_entry:
        reason = "price_in_middle_of_range"
    return {
        "enabled": True,
        "status": "ok",
        "regime": regime,
        "reason": reason,
        "allow_directional_entry": allow_entry,
        "forecast_direction": forecast_direction,
        "support_level": support,
        "resistance_level": resistance,
        "range_position": round(float(range_position), 4),
        "range_width_pct": round(float(range_width_pct), 6),
        "breakout_status": breakout_status,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
        "ema_spread_pct": round(float(ema_spread_pct), 6),
        "slope_pct": round(float(slope_pct), 6),
        "trend_strength_pct": round(float(trend_strength_pct), 6),
        "min_trend_strength_pct": float(min_trend_strength_pct),
        "middle_zone": [round(float(middle_low), 4), round(float(middle_high), 4)],
        "near_support": near_support,
        "near_resistance": near_resistance,
        "range_edge_reversal_allowed": range_edge_reversal,
        "impulse": impulse,
        "exhaustion": exhaustion,
        "lookback_rows": len(window),
    }


def _options_impulse_signal(
    close: pd.Series,
    *,
    forecast_direction: str,
    lookback_bars: int,
    min_move_pct: float,
    min_directional_bars: int,
) -> dict[str, Any]:
    if len(close) < max(4, int(lookback_bars) + 1):
        return {"enabled": True, "status": "insufficient_history", "allow_entry": False}
    lookback = max(3, int(lookback_bars))
    segment = close.tail(lookback + 1)
    start = float(segment.iloc[0])
    end = float(segment.iloc[-1])
    move_pct = end / max(abs(start), 1e-9) - 1.0
    diffs = segment.diff().dropna()
    up_bars = int((diffs > 0).sum())
    down_bars = int((diffs < 0).sum())
    required_bars = max(1, min(int(min_directional_bars), lookback))
    if move_pct >= float(min_move_pct) and up_bars >= required_bars:
        direction = "up"
    elif move_pct <= -float(min_move_pct) and down_bars >= required_bars:
        direction = "down"
    else:
        direction = "none"
    allow_entry = bool(direction != "none" and direction == forecast_direction)
    return {
        "enabled": True,
        "status": "ok",
        "direction": direction,
        "allow_entry": allow_entry,
        "move_pct": round(float(move_pct), 6),
        "min_move_pct": float(min_move_pct),
        "lookback_bars": lookback,
        "up_bars": up_bars,
        "down_bars": down_bars,
        "min_directional_bars": required_bars,
        "forecast_direction": forecast_direction,
    }


def _options_exhaustion_signal(
    close: pd.Series,
    *,
    forecast_direction: str,
    support: float,
    resistance: float,
    lookback_bars: int,
    max_late_entry_move_pct: float,
    max_ema_extension_pct: float,
    reversal_bars: int,
) -> dict[str, Any]:
    if len(close) < max(8, int(lookback_bars) + 1):
        return {"enabled": True, "status": "insufficient_history", "late_entry_block": False}
    lookback = max(3, int(lookback_bars))
    segment = close.tail(lookback + 1)
    start = float(segment.iloc[0])
    latest = float(segment.iloc[-1])
    move_pct = latest / max(abs(start), 1e-9) - 1.0
    ema = close.ewm(span=min(9, len(close)), adjust=False).mean()
    ema_extension_pct = latest / max(float(ema.iloc[-1]), 1e-9) - 1.0
    diffs = close.diff().dropna()
    recent = diffs.tail(max(1, int(reversal_bars)))
    reversal_up = bool(len(recent) >= max(1, int(reversal_bars)) and (recent > 0).all())
    reversal_down = bool(len(recent) >= max(1, int(reversal_bars)) and (recent < 0).all())
    support_distance_pct = abs(latest - float(support)) / max(abs(latest), 1e-9)
    resistance_distance_pct = abs(float(resistance) - latest) / max(abs(latest), 1e-9)
    late_reasons: list[str] = []
    if forecast_direction == "down":
        if move_pct <= -abs(float(max_late_entry_move_pct)):
            late_reasons.append("down_move_already_extended")
        if ema_extension_pct <= -abs(float(max_ema_extension_pct)):
            late_reasons.append("price_extended_below_fast_ema")
        if reversal_up:
            late_reasons.append("recent_bullish_reversal_bars")
        if support_distance_pct <= 0.0025:
            late_reasons.append("price_near_support_after_drop")
    elif forecast_direction == "up":
        if move_pct >= abs(float(max_late_entry_move_pct)):
            late_reasons.append("up_move_already_extended")
        if ema_extension_pct >= abs(float(max_ema_extension_pct)):
            late_reasons.append("price_extended_above_fast_ema")
        if reversal_down:
            late_reasons.append("recent_bearish_reversal_bars")
        if resistance_distance_pct <= 0.0025:
            late_reasons.append("price_near_resistance_after_rally")
    has_reversal_warning = "recent_bullish_reversal_bars" in late_reasons or "recent_bearish_reversal_bars" in late_reasons
    late_entry_block = len(late_reasons) >= 2 and has_reversal_warning
    return {
        "enabled": True,
        "status": "ok",
        "late_entry_block": late_entry_block,
        "reason": "late_entry_exhaustion" if late_entry_block else "not_exhausted",
        "late_reasons": late_reasons,
        "move_pct": round(float(move_pct), 6),
        "max_late_entry_move_pct": float(max_late_entry_move_pct),
        "ema_extension_pct": round(float(ema_extension_pct), 6),
        "max_ema_extension_pct": float(max_ema_extension_pct),
        "support_distance_pct": round(float(support_distance_pct), 6),
        "resistance_distance_pct": round(float(resistance_distance_pct), 6),
        "reversal_up": reversal_up,
        "reversal_down": reversal_down,
        "reversal_bars": max(1, int(reversal_bars)),
        "forecast_direction": forecast_direction,
    }


def _option_type_from_forecast(forecast: dict[str, Any]) -> str | None:
    direction = str(forecast.get("expected_direction") or "").lower()
    expected_return = _float_or_none(forecast.get("expected_return"))
    if expected_return is not None:
        if expected_return > 0:
            return "call"
        if expected_return < 0:
            return "put"
    if direction.startswith("up"):
        return "call"
    if direction.startswith("down"):
        return "put"
    return None


def _option_instrument_prefix(*, currency: str, underlying_currency: str | None) -> str | None:
    currency = str(currency or "").upper()
    underlying = str(underlying_currency or "").upper()
    if currency in {"USDC", "USDT", "USDE"} and underlying:
        return f"{underlying}_{currency}-"
    return f"{currency}-" if currency else None


def _premium_currency(instrument: dict[str, Any]) -> str:
    return str(instrument.get("settlement_currency") or instrument.get("quote_currency") or instrument.get("counter_currency") or "").upper()


def _premium_multiplier_usd(*, instrument: dict[str, Any], underlying_price_usd: float) -> float:
    premium_currency = _premium_currency(instrument)
    if premium_currency in {"USD", "USDC", "USDT", "USDE"}:
        return 1.0
    return max(float(underlying_price_usd), 1e-12)


def _price_series(prices: pd.DataFrame) -> pd.Series:
    for column in ("close", "Close", "price", "Price"):
        if column in prices.columns:
            return pd.to_numeric(prices[column], errors="coerce")
    numeric = prices.select_dtypes(include=[np.number])
    if numeric.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")


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


def _expiry_from_ms(value: Any) -> datetime | None:
    parsed = _float_or_none(value)
    if parsed is None:
        return None
    return datetime.fromtimestamp(parsed / 1000.0, tz=UTC)


def _candidate_score(
    *,
    strike: float | None,
    underlying_price_usd: float,
    forecast_price: float | None,
    spread_pct: float | None,
    delta: float | None,
    gamma: float | None,
    theta_premium_pct_per_day: float | None,
    dte: int | None,
    option_type: str,
    config: DeribitOptionExecutionConfig,
    fibonacci: dict[str, Any] | None = None,
) -> float:
    score = 0.0
    if strike is not None:
        desired = underlying_price_usd * (1.0 + config.target_moneyness if option_type == "call" else 1.0 - config.target_moneyness)
        score += abs(strike - desired) / max(underlying_price_usd, 1e-9)
        distance = abs(strike - underlying_price_usd) / max(underlying_price_usd, 1e-9)
        if distance > config.max_moneyness_distance:
            score += 10.0
    if forecast_price is not None and strike is not None:
        score -= abs(forecast_price - strike) / max(underlying_price_usd, 1e-9) * 0.10
    if spread_pct is not None:
        score += spread_pct
    if str(config.greeks_mode).lower() == "required" and delta is not None:
        score += abs(abs(delta) - config.target_delta) * 0.60
    if str(config.greeks_mode).lower() == "required" and gamma is not None:
        score -= min(abs(gamma), 0.05) * 0.40
    if str(config.greeks_mode).lower() == "required" and theta_premium_pct_per_day is not None:
        score += theta_premium_pct_per_day * 0.35
    if fibonacci and config.enable_fibonacci:
        if fibonacci.get("confirmation") == "supportive":
            score -= 0.03
        elif fibonacci.get("confirmation") == "conflict":
            score += 0.08
        nearest_target = _float_or_none(fibonacci.get("nearest_target_price"))
        if nearest_target is not None and strike is not None:
            score += abs(strike - nearest_target) / max(underlying_price_usd, 1e-9) * 0.20
    if dte is not None:
        score += abs(dte - max(config.min_dte, 1)) * 0.01
    return float(score)


def _round_to_tick(value: float, tick_size: float) -> float:
    tick = max(float(tick_size), 1e-8)
    return float(round(float(np.ceil(float(value) / tick) * tick), 8))


def _round_down_to_tick(value: float, tick_size: float) -> float:
    tick = max(float(tick_size), 1e-8)
    return float(round(max(tick, float(np.floor(float(value) / tick) * tick)), 8))


def _instrument_tick_size(
    *,
    broker: DeribitTestnetBroker | DeribitOptionsBroker,
    instrument_name: str,
    price: float | None = None,
) -> float | None:
    currency = _instrument_currency_from_name(instrument_name)
    try:
        instruments = broker.instruments(currency=currency, kind="option", expired=False)
    except (RuntimeError, AttributeError):
        return None
    for instrument in instruments:
        if str(instrument.get("instrument_name") or "") != instrument_name:
            continue
        stepped = _tick_size_from_steps(instrument.get("tick_size_steps"), price)
        if stepped is not None:
            return stepped
        return _float_or_none(instrument.get("tick_size"))
    return None


def _instrument_currency_from_name(instrument_name: str) -> str:
    prefix = str(instrument_name or "").split("-", 1)[0].upper()
    if "_" in prefix:
        return prefix.rsplit("_", 1)[-1]
    return prefix


def _tick_size_from_steps(steps: Any, price: float | None) -> float | None:
    if not isinstance(steps, list) or price is None:
        return None
    selected: float | None = None
    for step in sorted(steps, key=lambda row: _float_or_none(row.get("above_price")) or 0.0):
        above = _float_or_none(step.get("above_price")) or 0.0
        tick = _float_or_none(step.get("tick_size"))
        if tick is not None and float(price) >= above:
            selected = tick
    return selected


def _round_amount(value: float, minimum: float) -> float:
    step = max(float(minimum), 1e-8)
    return float(round(float(np.floor(float(value) / step) * step), 8))


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
