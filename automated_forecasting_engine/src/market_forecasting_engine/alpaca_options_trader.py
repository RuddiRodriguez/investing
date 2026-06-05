from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.options_market_regime import build_options_market_regime


@dataclass(frozen=True)
class OptionExecutionConfig:
    underlying: str
    risk_profile: str = "aggressive"
    min_dte: int = 1
    max_dte: int = 14
    allow_0dte: bool = False
    max_contract_premium: float | None = None
    max_total_debit: float = 1500.0
    risk_budget_pct: float = 0.0025
    max_position_equity_pct: float = 0.02
    max_spread_pct: float = 0.18
    max_contracts: int = 1
    target_delta: float = 0.35
    max_delta_distance: float = 0.28
    require_greeks: bool = True
    max_theta_edge_ratio: float = 0.75
    max_theta_premium_pct_per_day: float = 0.35
    min_open_interest: int = 0
    enable_market_regime_filter: bool = False
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
    limit_price_offset_pct: float = 0.03
    stop_loss_pct: float = 0.35
    take_profit_pct: float = 0.55
    stop_limit_offset_pct: float = 0.08
    abandon_entry_after_seconds: int = 300
    entry_order_policy: str = "auto"
    exit_order_policy: str = "auto"


def build_real_option_trade_plan(
    *,
    broker: AlpacaPaperBroker,
    underlying: str,
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    prices: pd.DataFrame | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    option_type = _option_type_from_forecast(forecast)
    if option_type is None:
        return {"action": "hold", "reason": "forecast_has_no_directional_edge", "forecast": forecast}
    market_regime = (
        build_options_market_regime(
            prices if prices is not None else pd.DataFrame(),
            current_price=float(underlying_price),
            forecast=forecast,
            lookback_rows=int(config.market_regime_lookback_rows),
            breakout_buffer_pct=float(config.market_regime_breakout_buffer_pct),
            middle_zone_width=float(config.market_regime_middle_zone_width),
            min_trend_strength_pct=float(config.min_trend_strength_pct),
            allow_range_edge_reversal_entry=bool(config.allow_range_edge_reversal_entry),
            enable_impulse_entry=bool(config.enable_impulse_entry),
            impulse_lookback_bars=int(config.impulse_lookback_bars),
            min_impulse_move_pct=float(config.min_impulse_move_pct),
            min_impulse_directional_bars=int(config.min_impulse_directional_bars),
            enable_late_entry_filter=bool(config.enable_late_entry_filter),
            max_late_entry_move_pct=float(config.max_late_entry_move_pct),
            max_ema_extension_pct=float(config.max_ema_extension_pct),
            exhaustion_reversal_bars=int(config.exhaustion_reversal_bars),
        )
        if config.enable_market_regime_filter
        else {"enabled": False}
    )
    if config.enable_market_regime_filter and market_regime.get("allow_directional_entry") is False:
        return {
            "action": "hold",
            "reason": "market_regime_blocks_directional_entry",
            "underlying": underlying.upper(),
            "option_type": option_type,
            "market_regime": market_regime,
            "forecast": forecast,
        }
    start_date, end_date = _expiration_window(now, config)
    contracts = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=start_date.isoformat(),
        expiration_date_lte=end_date.isoformat(),
        option_type=option_type,
        limit=1000,
    )
    if not contracts:
        return {
            "action": "hold",
            "reason": "no_tradable_option_contracts",
            "option_type": option_type,
            "expiration_date_gte": start_date.isoformat(),
            "expiration_date_lte": end_date.isoformat(),
        }
    symbols = [str(contract.get("symbol")) for contract in contracts if contract.get("symbol")]
    snapshots = broker.option_snapshots(symbols)
    candidates = score_option_contracts(
        contracts=contracts,
        snapshots=snapshots,
        underlying_price=underlying_price,
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
            "option_type": option_type,
            "candidate_count": len(candidates),
            "top_rejections": candidates[:10],
        }
    selected = None
    sizing = None
    for candidate in accepted:
        candidate_limit = round(float(candidate["limit_price"]), 2)
        candidate_sizing = size_option_position(
            entry_limit_price=candidate_limit,
            account_equity=_float_or_none(forecast.get("account_equity")),
            config=config,
        )
        candidate["sizing"] = candidate_sizing
        if candidate_sizing["qty"] > 0:
            selected = candidate
            sizing = candidate_sizing
            break
    if selected is None or sizing is None:
        best = accepted[0]
        best_limit = round(float(best["limit_price"]), 2)
        best["sizing"] = size_option_position(
            entry_limit_price=best_limit,
            account_equity=_float_or_none(forecast.get("account_equity")),
            config=config,
        )
        return {
            "action": "hold",
            "reason": "position_size_below_one_contract",
            "selected_contract": best,
            "sizing": best["sizing"],
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "top_candidates": accepted[:5],
        }
    limit_price = round(float(selected["limit_price"]), 2)
    qty = sizing["qty"]
    entry_order = choose_option_entry_order(selected=selected, qty=qty, config=config)
    exit_plan = choose_option_exit_orders(entry_limit_price=limit_price, qty=qty, config=config)
    return {
        "action": "buy_option",
        "underlying": underlying.upper(),
        "option_type": option_type,
        "market_regime": market_regime,
        "selected_contract": selected,
        "order": entry_order,
        "risk": {
            "max_contract_premium": config.max_contract_premium,
            "max_total_debit": config.max_total_debit,
            "risk_budget_pct": config.risk_budget_pct,
            "max_position_equity_pct": config.max_position_equity_pct,
            "estimated_debit": round(limit_price * qty * 100.0, 2),
            "max_loss_if_stop_fills": round((limit_price - exit_plan["stop_loss"]["limit_price"]) * qty * 100.0, 2),
            "max_contracts": config.max_contracts,
            "max_spread_pct": config.max_spread_pct,
            "policy": "Options use whole-contract qty and limit orders only. Notional orders are not used for options.",
        },
        "exit_plan": exit_plan,
        "sizing": sizing,
        "order_policy_decision": {
            "entry_order_policy": config.entry_order_policy,
            "exit_order_policy": config.exit_order_policy,
            "available_order_types_considered": ["limit", "stop", "stop_limit", "trailing_stop"],
            "unsupported_or_rejected_for_options": ["market_for_entry"],
            "selected_entry_type": entry_order["type"],
        },
        "abandon_plan": {
            "cancel_unfilled_entry_after_seconds": int(config.abandon_entry_after_seconds),
            "replace_policy": "If still valid, recompute contract and limit from fresh quotes instead of chasing the old order.",
        },
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "top_candidates": accepted[:5],
    }


def score_option_contracts(
    *,
    contracts: list[dict[str, Any]],
    snapshots: dict[str, Any],
    underlying_price: float,
    forecast: dict[str, Any],
    option_type: str,
    config: OptionExecutionConfig,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(UTC)
    forecast_price = _float_or_none(forecast.get("predicted_price"))
    scored = []
    for contract in contracts:
        symbol = str(contract.get("symbol") or "")
        quote = _snapshot_quote(snapshots.get(symbol) or {})
        greeks = _snapshot_greeks(snapshots.get(symbol) or {})
        strike = _float_or_none(contract.get("strike_price") or contract.get("strike"))
        expiry = _parse_expiry(contract.get("expiration_date"))
        bid = quote.get("bid")
        ask = quote.get("ask")
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        spread_pct = None if bid is None or ask is None or mid is None else (ask - bid) / max(mid, 1e-9)
        delta = _float_or_none(greeks.get("delta"))
        gamma = _float_or_none(greeks.get("gamma"))
        theta = _float_or_none(greeks.get("theta"))
        vega = _float_or_none(greeks.get("vega"))
        rho = _float_or_none(greeks.get("rho"))
        dte = None if expiry is None else max(0, (expiry - now.date()).days)
        premium = None if ask is None else ask * 100.0
        horizon_hours = _float_or_none(forecast.get("horizon_hours"))
        forecast_edge_usd = None
        theta_decay_usd_for_horizon = None
        theta_premium_pct_per_day = None
        if forecast_price is not None and delta is not None:
            forecast_edge_usd = abs(forecast_price - underlying_price) * abs(delta) * 100.0
        if theta is not None and horizon_hours is not None:
            theta_decay_usd_for_horizon = abs(theta) * 100.0 * max(horizon_hours, 0.0) / 24.0
        if theta is not None and premium is not None and premium > 0:
            theta_premium_pct_per_day = abs(theta) * 100.0 / premium
        reasons = []
        if not symbol:
            reasons.append("missing_symbol")
        if contract.get("tradable") is False:
            reasons.append("contract_not_tradable")
        if str(contract.get("status") or "").lower() not in {"", "active"}:
            reasons.append("contract_not_active")
        if strike is None:
            reasons.append("missing_strike")
        if expiry is None:
            reasons.append("missing_expiration")
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            reasons.append("missing_live_bid_ask")
        if spread_pct is None or spread_pct > config.max_spread_pct:
            reasons.append("spread_too_wide")
        if premium is None:
            reasons.append("missing_option_premium")
        elif config.max_contract_premium is not None and premium > config.max_contract_premium:
            reasons.append("premium_above_cap")
        if dte is not None:
            if dte == 0 and not config.allow_0dte:
                reasons.append("zero_dte_blocked")
            if dte < config.min_dte:
                reasons.append("dte_below_min")
            if dte > config.max_dte:
                reasons.append("dte_above_max")
        open_interest = _float_or_none(contract.get("open_interest"))
        if open_interest is not None and open_interest < config.min_open_interest:
            reasons.append("open_interest_below_min")
        if config.require_greeks and (delta is None or gamma is None or theta is None or vega is None):
            reasons.append("missing_greeks")
        if delta is not None and abs(abs(delta) - config.target_delta) > config.max_delta_distance:
            reasons.append("delta_too_far_from_target")
        if (
            theta_decay_usd_for_horizon is not None
            and forecast_edge_usd is not None
            and theta_decay_usd_for_horizon > forecast_edge_usd * config.max_theta_edge_ratio
        ):
            reasons.append("theta_decay_too_large_vs_forecast_edge")
        if theta_premium_pct_per_day is not None and theta_premium_pct_per_day > config.max_theta_premium_pct_per_day:
            reasons.append("theta_too_large_vs_premium")
        intrinsic_ok = True
        if strike is not None and forecast_price is not None:
            intrinsic_ok = forecast_price > strike if option_type == "call" else forecast_price < strike
            if not intrinsic_ok:
                reasons.append("forecast_not_beyond_strike")
        score = _candidate_score(
            strike=strike,
            underlying_price=underlying_price,
            forecast_price=forecast_price,
            delta=delta,
            gamma=gamma,
            theta_premium_pct_per_day=theta_premium_pct_per_day,
            spread_pct=spread_pct,
            dte=dte,
            config=config,
        )
        limit_price = None if ask is None or mid is None else min(ask, mid * (1.0 + config.limit_price_offset_pct))
        scored.append(
            {
                "symbol": symbol,
                "name": contract.get("name"),
                "option_type": option_type,
                "expiration_date": contract.get("expiration_date"),
                "dte": dte,
                "strike": strike,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "delta": delta,
                "greeks": {
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
                "open_interest": open_interest,
                "premium": premium,
                "limit_price": limit_price,
                "accepted": not reasons,
                "reasons": reasons,
                "score": score,
            }
        )
    scored.sort(key=lambda item: (not item["accepted"], item["score"], item.get("premium") or float("inf")))
    return scored


def submit_option_order(broker: AlpacaPaperBroker, order: dict[str, Any], *, client_order_id: str | None = None) -> dict[str, Any]:
    if order.get("type") not in {"limit", "stop", "stop_limit", "trailing_stop"}:
        raise ValueError("Unsupported option order type.")
    if "notional" in order:
        raise ValueError("Options orders must use whole-contract qty, not notional.")
    qty = int(order.get("qty") or 0)
    if qty <= 0:
        raise ValueError("Options orders require positive whole-contract qty.")
    return broker.submit_order(
        symbol=str(order["symbol"]),
        side=str(order["side"]),
        order_type=str(order["type"]),
        qty=qty,
        limit_price=_float_or_none(order.get("limit_price")),
        stop_price=_float_or_none(order.get("stop_price")),
        trail_percent=_float_or_none(order.get("trail_percent")),
        trail_price=_float_or_none(order.get("trail_price")),
        client_order_id=client_order_id,
        time_in_force=str(order.get("time_in_force") or "day"),
    )


def choose_option_entry_order(*, selected: dict[str, Any], qty: int, config: OptionExecutionConfig) -> dict[str, Any]:
    if config.entry_order_policy not in {"auto", "limit"}:
        raise ValueError("Options entry policy supports auto or limit.")
    ask = float(selected["ask"])
    mid = float(selected["mid"])
    limit_price = round(min(ask, mid * (1.0 + config.limit_price_offset_pct)), 2)
    return {
        "symbol": selected["symbol"],
        "side": "buy",
        "type": "limit",
        "qty": int(qty),
        "limit_price": limit_price,
        "time_in_force": "day",
        "policy_reason": "Options entries use limit orders to control spread slippage.",
    }


def size_option_position(
    *,
    entry_limit_price: float,
    account_equity: float | None,
    config: OptionExecutionConfig,
) -> dict[str, Any]:
    premium_per_contract = round(max(float(entry_limit_price), 0.01) * 100.0, 2)
    if config.max_contract_premium is not None and premium_per_contract > float(config.max_contract_premium):
        return {
            "qty": 0,
            "premium_per_contract": premium_per_contract,
            "budget": 0.0,
            "reason": "premium_per_contract_above_cap",
        }
    equity = max(float(account_equity or 0.0), 0.0)
    risk_budget = equity * float(config.risk_budget_pct) if equity > 0 else premium_per_contract
    exposure_budget = equity * float(config.max_position_equity_pct) if equity > 0 else float(config.max_total_debit)
    budget = max(0.0, min(float(config.max_total_debit), exposure_budget, max(risk_budget, premium_per_contract)))
    by_budget = int(budget // max(premium_per_contract, 1e-9))
    qty = max(0, min(int(config.max_contracts), by_budget))
    return {
        "qty": qty,
        "premium_per_contract": premium_per_contract,
        "account_equity": account_equity,
        "risk_budget": round(risk_budget, 2),
        "exposure_budget": round(exposure_budget, 2),
        "max_total_debit": float(config.max_total_debit),
        "max_contracts": int(config.max_contracts),
        "budget": round(budget, 2),
        "estimated_debit": round(qty * premium_per_contract, 2),
        "reason": "sized_from_equity_risk_budget" if qty > 0 else "budget_below_one_contract",
    }


def choose_option_exit_orders(*, entry_limit_price: float, qty: int, config: OptionExecutionConfig) -> dict[str, Any]:
    exit_plan = build_option_exit_plan(entry_limit_price=entry_limit_price, qty=qty, config=config)
    if config.exit_order_policy == "trailing_stop":
        exit_plan["primary_exit"] = {
            "side": "sell",
            "type": "trailing_stop",
            "qty": int(qty),
            "trail_percent": round(max(5.0, min(35.0, config.stop_loss_pct * 100.0)), 2),
            "time_in_force": "day",
            "policy_reason": "Trailing stop requested; agent will submit only if Alpaca accepts it for the option contract.",
        }
    elif config.exit_order_policy == "take_profit":
        exit_plan["primary_exit"] = exit_plan["take_profit"]
    else:
        exit_plan["primary_exit"] = exit_plan["stop_loss"]
    exit_plan["selected_policy"] = config.exit_order_policy
    return exit_plan


def build_option_exit_plan(*, entry_limit_price: float, qty: int, config: OptionExecutionConfig) -> dict[str, Any]:
    entry = max(float(entry_limit_price), 0.01)
    take_profit = round(entry * (1.0 + float(config.take_profit_pct)), 2)
    stop_price = round(max(0.01, entry * (1.0 - float(config.stop_loss_pct))), 2)
    stop_limit = round(max(0.01, stop_price * (1.0 - float(config.stop_limit_offset_pct))), 2)
    return {
        "take_profit": {
            "side": "sell",
            "type": "limit",
            "qty": int(qty),
            "limit_price": take_profit,
            "time_in_force": "day",
        },
        "stop_loss": {
            "side": "sell",
            "type": "stop_limit",
            "qty": int(qty),
            "stop_price": stop_price,
            "limit_price": stop_limit,
            "time_in_force": "day",
        },
        "policy": (
            "Entry is autonomous limit. After fill, the agent manages exits with a profit-taking limit "
            "and a stop-limit loss control because market option orders can cross wide spreads."
        ),
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


def _expiration_window(now: datetime, config: OptionExecutionConfig) -> tuple[date, date]:
    min_dte = 0 if config.allow_0dte else max(1, int(config.min_dte))
    return now.date() + timedelta(days=min_dte), now.date() + timedelta(days=max(min_dte, int(config.max_dte)))


def _snapshot_quote(snapshot: dict[str, Any]) -> dict[str, float | None]:
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or snapshot.get("quote") or {}
    return {
        "bid": _float_or_none(quote.get("bp") or quote.get("bid_price") or quote.get("bid")),
        "ask": _float_or_none(quote.get("ap") or quote.get("ask_price") or quote.get("ask")),
    }


def _snapshot_greeks(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("greeks") or snapshot.get("latestGreeks") or snapshot.get("latest_greeks") or {}


def _parse_expiry(value: Any) -> date | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _candidate_score(
    *,
    strike: float | None,
    underlying_price: float,
    forecast_price: float | None,
    delta: float | None,
    gamma: float | None,
    theta_premium_pct_per_day: float | None,
    spread_pct: float | None,
    dte: int | None,
    config: OptionExecutionConfig,
) -> float:
    score = 0.0
    if delta is not None:
        score += abs(abs(delta) - config.target_delta)
    if gamma is not None:
        score -= min(abs(gamma), 0.05) * 0.40
    if theta_premium_pct_per_day is not None:
        score += theta_premium_pct_per_day * 0.35
    if spread_pct is not None:
        score += spread_pct
    if strike is not None:
        score += abs(strike - underlying_price) / max(underlying_price, 1e-9) * 0.35
    if forecast_price is not None and strike is not None:
        score -= abs(forecast_price - strike) / max(underlying_price, 1e-9) * 0.15
    if dte is not None:
        score += abs(dte - max(config.min_dte, 1)) * 0.01
    return float(score)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
