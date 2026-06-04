from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.deribit_broker import DeribitTestnetBroker


@dataclass(frozen=True)
class DeribitOptionExecutionConfig:
    currency: str
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
    instruments = broker.instruments(currency=currency, kind="option", expired=False)
    candidates = score_deribit_option_contracts(
        broker=broker,
        instruments=instruments,
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
        if str(instrument.get("option_type") or "").lower() != option_type:
            continue
        name = str(instrument.get("instrument_name") or "")
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
        premium_usd = None if ask is None else ask * underlying_price_usd
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
    min_trade_amount: float | None,
) -> dict[str, Any]:
    premium_usd = max(float(entry_limit_price_base), 1e-12) * max(float(underlying_price_usd), 1e-12)
    equity_base = _float_or_none(account.get("equity")) or _float_or_none(account.get("balance")) or 0.0
    equity_usd = max(equity_base, 0.0) * max(float(underlying_price_usd), 0.0)
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
    broker: DeribitTestnetBroker,
    order: dict[str, Any],
    *,
    label: str | None = None,
) -> dict[str, Any]:
    if order.get("type") != "limit":
        raise ValueError("Deribit options agent only submits limit orders.")
    if str(order.get("side")) == "buy":
        return broker.buy_limit(
            instrument_name=str(order["instrument_name"]),
            amount=float(order["amount"]),
            price=float(order["price"]),
            label=label,
            post_only=bool(order.get("post_only")),
            reduce_only=bool(order.get("reduce_only")),
        )
    if str(order.get("side")) == "sell":
        return broker.sell_limit(
            instrument_name=str(order["instrument_name"]),
            amount=float(order["amount"]),
            price=float(order["price"]),
            label=label,
            post_only=bool(order.get("post_only")),
            reduce_only=bool(order.get("reduce_only", True)),
        )
    raise ValueError("Deribit order side must be buy or sell.")


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
