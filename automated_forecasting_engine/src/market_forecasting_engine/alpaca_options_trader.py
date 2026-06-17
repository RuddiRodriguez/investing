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
    option_strategy_mode: str = "directional"
    enable_multi_leg: bool = False
    enable_short_option_strategies: bool = False
    max_legs: int = 2
    straddle_max_debit_multiplier: float = 1.0
    iron_butterfly_wing_width_pct: float = 0.05
    calendar_near_min_dte: int = 1
    calendar_near_max_dte: int = 7
    calendar_far_min_dte: int = 8
    calendar_far_max_dte: int = 35


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
        multi_leg_plan = _maybe_build_strategy_plan(
            broker=broker,
            underlying=underlying,
            underlying_price=underlying_price,
            forecast=forecast,
            config=config,
            now=now,
            market_regime=market_regime,
            reason="directional_entry_blocked_by_regime",
        )
        if multi_leg_plan is not None:
            return multi_leg_plan
        return {
            "action": "hold",
            "reason": "market_regime_blocks_directional_entry",
            "underlying": underlying.upper(),
            "option_type": option_type,
            "market_regime": market_regime,
            "forecast": forecast,
        }
    explicit_multi_leg_modes = {"long_straddle", "short_iron_butterfly", "long_call_calendar", "long_put_calendar"}
    if config.enable_multi_leg and config.option_strategy_mode in explicit_multi_leg_modes:
        multi_leg_plan = _maybe_build_strategy_plan(
            broker=broker,
            underlying=underlying,
            underlying_price=underlying_price,
            forecast=forecast,
            config=config,
            now=now,
            market_regime=market_regime,
            reason=f"explicit_{config.option_strategy_mode}_strategy",
        )
        if multi_leg_plan is not None:
            return multi_leg_plan
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
        "trade_quality": selected.get("trade_quality"),
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
        trade_quality = _single_leg_trade_quality(
            option_type=option_type,
            underlying_price=underlying_price,
            forecast_price=forecast_price,
            strike=strike,
            bid=bid,
            ask=ask,
            mid=mid,
            limit_price=limit_price,
            spread_pct=spread_pct,
            open_interest=open_interest,
            premium=premium,
            forecast_edge_usd=forecast_edge_usd,
            theta_decay_usd_for_horizon=theta_decay_usd_for_horizon,
            theta_premium_pct_per_day=theta_premium_pct_per_day,
            config=config,
        )
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
                "trade_quality": trade_quality,
            }
        )
    scored.sort(key=lambda item: (not item["accepted"], item["score"], item.get("premium") or float("inf")))
    return scored


def submit_option_order(broker: AlpacaPaperBroker, order: dict[str, Any], *, client_order_id: str | None = None) -> dict[str, Any]:
    if order.get("order_class") == "mleg":
        if order.get("type") == "market":
            raise ValueError("Market multi-leg option entries are disabled for this agent.")
        return broker.submit_multileg_option_order(
            legs=list(order.get("legs") or []),
            order_type=str(order.get("type") or "limit"),
            qty=int(order.get("qty") or 1),
            limit_price=_float_or_none(order.get("limit_price")),
            time_in_force=str(order.get("time_in_force") or "day"),
            client_order_id=client_order_id,
        )
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


def _maybe_build_strategy_plan(
    *,
    broker: AlpacaPaperBroker,
    underlying: str,
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
    market_regime: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    if not config.enable_multi_leg:
        return None
    mode = str(config.option_strategy_mode or "directional")
    expected_return = _float_or_none(forecast.get("expected_return")) or 0.0
    regime = str(market_regime.get("regime") or "").lower()
    planners: list[tuple[str, str | None]] = []
    if mode == "long_straddle":
        planners = [("long_straddle", None)]
    elif mode == "short_iron_butterfly":
        planners = [("short_iron_butterfly", None)]
    elif mode == "long_call_calendar":
        planners = [("long_calendar", "call")]
    elif mode == "long_put_calendar":
        planners = [("long_calendar", "put")]
    elif mode == "auto":
        if config.enable_short_option_strategies and regime in {"range_bound", "oscillating"} and abs(expected_return) < 0.004:
            planners.append(("short_iron_butterfly", None))
        if config.enable_short_option_strategies and abs(expected_return) < 0.008:
            planners.append(("long_calendar", "call" if expected_return >= 0 else "put"))
        planners.append(("long_straddle", None))
    for strategy, option_type in planners:
        if strategy == "short_iron_butterfly":
            plan = _maybe_build_short_iron_butterfly_plan(
                broker=broker,
                underlying=underlying,
                underlying_price=underlying_price,
                forecast=forecast,
                config=config,
                now=now,
                market_regime=market_regime,
                reason=reason,
            )
        elif strategy == "long_calendar":
            plan = _maybe_build_calendar_spread_plan(
                broker=broker,
                underlying=underlying,
                underlying_price=underlying_price,
                forecast=forecast,
                config=config,
                now=now,
                market_regime=market_regime,
                reason=reason,
                option_type=str(option_type or "call"),
            )
        else:
            plan = _maybe_build_long_straddle_plan(
                broker=broker,
                underlying=underlying,
                underlying_price=underlying_price,
                forecast=forecast,
                config=config,
                now=now,
                market_regime=market_regime,
                reason=reason,
            )
        if plan is not None and plan.get("action") != "hold":
            return plan
        if mode != "auto" and plan is not None:
            return plan
    return None


def _maybe_build_long_straddle_plan(
    *,
    broker: AlpacaPaperBroker,
    underlying: str,
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
    market_regime: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    if not config.enable_multi_leg or config.max_legs < 2 or config.option_strategy_mode not in {"auto", "long_straddle"}:
        return None
    start_date, end_date = _expiration_window(now, config)
    calls = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=start_date.isoformat(),
        expiration_date_lte=end_date.isoformat(),
        option_type="call",
        limit=1000,
    )
    puts = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=start_date.isoformat(),
        expiration_date_lte=end_date.isoformat(),
        option_type="put",
        limit=1000,
    )
    if not calls or not puts:
        return None
    symbols = [str(contract.get("symbol")) for contract in [*calls, *puts] if contract.get("symbol")]
    snapshots = broker.option_snapshots(symbols)
    pairs = _score_long_straddle_pairs(
        calls=calls,
        puts=puts,
        snapshots=snapshots,
        underlying_price=underlying_price,
        forecast=forecast,
        config=config,
        now=now,
    )
    accepted = [pair for pair in pairs if pair["accepted"]]
    if not accepted:
        return {
            "action": "hold",
            "reason": "no_long_straddle_pair_passed_execution_gates",
            "strategy": "long_straddle",
            "strategy_reason": reason,
            "market_regime": market_regime,
            "candidate_count": len(pairs),
            "top_rejections": pairs[:10],
        }
    selected = accepted[0]
    sizing = size_option_position(
        entry_limit_price=float(selected["combined_limit_price"]),
        account_equity=_float_or_none(forecast.get("account_equity")),
        config=OptionExecutionConfig(
            **{
                **config.__dict__,
                "max_total_debit": float(config.max_total_debit) * float(config.straddle_max_debit_multiplier),
                "max_contracts": int(config.max_contracts),
            }
        ),
    )
    selected["sizing"] = sizing
    if sizing["qty"] <= 0:
        return {
            "action": "hold",
            "reason": "multi_leg_position_size_below_one_strategy",
            "strategy": "long_straddle",
            "strategy_reason": reason,
            "market_regime": market_regime,
            "selected_pair": selected,
            "candidate_count": len(pairs),
            "accepted_count": len(accepted),
        }
    qty = int(sizing["qty"])
    order = {
        "order_class": "mleg",
        "strategy": "long_straddle",
        "symbol": selected["strategy_symbol"],
        "side": "buy",
        "type": "limit",
        "qty": qty * 2,
        "strategy_qty": qty,
        "limit_price": round(float(selected["combined_limit_price"]), 2),
        "time_in_force": "day",
        "legs": [
            {"side": "buy", "position_intent": "buy_to_open", "symbol": selected["call"]["symbol"], "ratio_qty": 1},
            {"side": "buy", "position_intent": "buy_to_open", "symbol": selected["put"]["symbol"], "ratio_qty": 1},
        ],
        "policy_reason": "Paper multi-leg long straddle uses one call and one put with a debit limit. No short option legs are used.",
    }
    return {
        "action": "buy_option",
        "underlying": underlying.upper(),
        "option_type": "multi_leg",
        "strategy": "long_straddle",
        "strategy_reason": reason,
        "market_regime": market_regime,
        "selected_pair": selected,
        "trade_quality": selected.get("trade_quality"),
        "selected_contract": {
            "symbol": selected["strategy_symbol"],
            "name": "Long straddle",
            "option_type": "multi_leg",
            "expiration_date": selected["expiration_date"],
            "dte": selected["dte"],
            "strike": selected["strike"],
            "bid": selected["combined_bid"],
            "ask": selected["combined_ask"],
            "mid": selected["combined_mid"],
            "spread_pct": selected["combined_spread_pct"],
            "premium": round(float(selected["combined_limit_price"]) * 100.0, 2),
            "score": selected["score"],
        },
        "order": order,
        "risk": {
            "max_total_debit": config.max_total_debit,
            "estimated_debit": round(float(selected["combined_limit_price"]) * qty * 100.0, 2),
            "max_contracts": config.max_contracts,
            "max_spread_pct": config.max_spread_pct,
            "policy": "Long straddle is a paper-only multi-leg experiment. It buys one call and one put; max loss is limited to paid debit if both legs fill.",
        },
        "exit_plan": {
            "policy": "Multi-leg entries are managed as resulting single-leg positions after fill; each open option leg receives the same existing profit/loss guards.",
            "primary_exit": {"type": "managed_per_leg_after_fill"},
        },
        "sizing": sizing,
        "order_policy_decision": {
            "entry_order_policy": config.entry_order_policy,
            "option_strategy_mode": config.option_strategy_mode,
            "available_order_types_considered": ["limit_multi_leg_debit"],
            "unsupported_or_rejected_for_options": ["market_for_entry", "short_option_legs"],
            "selected_entry_type": "limit_mleg_debit",
        },
        "abandon_plan": {
            "cancel_unfilled_entry_after_seconds": int(config.abandon_entry_after_seconds),
            "replace_policy": "If the debit spread is stale, cancel and recompute both legs from fresh quotes.",
        },
        "candidate_count": len(pairs),
        "accepted_count": len(accepted),
        "top_candidates": accepted[:5],
    }


def _maybe_build_short_iron_butterfly_plan(
    *,
    broker: AlpacaPaperBroker,
    underlying: str,
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
    market_regime: dict[str, Any],
    reason: str,
) -> dict[str, Any] | None:
    if not config.enable_short_option_strategies or config.max_legs < 4 or config.option_strategy_mode not in {"auto", "short_iron_butterfly"}:
        return None
    start_date, end_date = _expiration_window(now, config)
    calls = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=start_date.isoformat(),
        expiration_date_lte=end_date.isoformat(),
        option_type="call",
        limit=1000,
    )
    puts = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=start_date.isoformat(),
        expiration_date_lte=end_date.isoformat(),
        option_type="put",
        limit=1000,
    )
    if not calls or not puts:
        return None
    symbols = [str(contract.get("symbol")) for contract in [*calls, *puts] if contract.get("symbol")]
    snapshots = broker.option_snapshots(symbols)
    candidates = _score_short_iron_butterflies(
        calls=calls,
        puts=puts,
        snapshots=snapshots,
        underlying_price=underlying_price,
        forecast=forecast,
        config=config,
        now=now,
    )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        return {
            "action": "hold",
            "reason": "no_short_iron_butterfly_passed_execution_gates",
            "strategy": "short_iron_butterfly",
            "strategy_reason": reason,
            "market_regime": market_regime,
            "candidate_count": len(candidates),
            "top_rejections": candidates[:10],
        }
    selected = accepted[0]
    max_loss = float(selected["max_loss_usd"])
    strategy_qty = min(int(config.max_contracts), int(float(config.max_total_debit) // max(max_loss, 1.0)))
    if strategy_qty <= 0:
        return {
            "action": "hold",
            "reason": "iron_butterfly_defined_risk_above_budget",
            "strategy": "short_iron_butterfly",
            "strategy_reason": reason,
            "selected_structure": selected,
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
        }
    order = {
        "order_class": "mleg",
        "strategy": "short_iron_butterfly",
        "symbol": selected["strategy_symbol"],
        "side": "sell",
        "type": "limit",
        "qty": strategy_qty * 4,
        "strategy_qty": strategy_qty,
        "limit_price": round(-float(selected["net_credit"]), 2),
        "time_in_force": "day",
        "legs": [
            {"side": "buy", "position_intent": "buy_to_open", "symbol": selected["long_put"]["symbol"], "ratio_qty": 1},
            {"side": "sell", "position_intent": "sell_to_open", "symbol": selected["short_put"]["symbol"], "ratio_qty": 1},
            {"side": "sell", "position_intent": "sell_to_open", "symbol": selected["short_call"]["symbol"], "ratio_qty": 1},
            {"side": "buy", "position_intent": "buy_to_open", "symbol": selected["long_call"]["symbol"], "ratio_qty": 1},
        ],
        "policy_reason": "Defined-risk short iron butterfly. Credit order uses a negative limit price and all four legs must fill together.",
    }
    return {
        "action": "buy_option",
        "underlying": underlying.upper(),
        "option_type": "multi_leg",
        "strategy": "short_iron_butterfly",
        "strategy_reason": reason,
        "market_regime": market_regime,
        "selected_structure": selected,
        "selected_contract": {
            "symbol": selected["strategy_symbol"],
            "name": "Short iron butterfly",
            "option_type": "multi_leg",
            "expiration_date": selected["expiration_date"],
            "dte": selected["dte"],
            "strike": selected["center_strike"],
            "bid": selected["net_credit"],
            "ask": selected["net_credit"],
            "mid": selected["net_credit"],
            "spread_pct": selected["combined_spread_pct"],
            "premium": round(float(selected["max_loss_usd"]), 2),
            "score": selected["score"],
        },
        "trade_quality": selected.get("trade_quality"),
        "order": order,
        "risk": {
            "estimated_credit": round(float(selected["net_credit"]) * strategy_qty * 100.0, 2),
            "max_defined_loss": round(float(selected["max_loss_usd"]) * strategy_qty, 2),
            "max_profit_at_center": round(float(selected["max_profit_usd"]) * strategy_qty, 2),
            "upper_breakeven_price": selected["upper_breakeven_price"],
            "lower_breakeven_price": selected["lower_breakeven_price"],
            "assignment_risk": "Contains short American-style option legs. Agent must close before expiry and monitor assignment/exercise risk.",
            "policy": "Short option strategies require explicit enable-short-option-strategies and defined-risk multi-leg order submission.",
        },
        "exit_plan": {
            "policy": "Manage as a defined-risk spread. Close the whole structure when target credit capture, loss guard, forecast regime break, or expiry guard is hit.",
            "primary_exit": {"type": "managed_defined_risk_multileg_after_fill"},
        },
        "sizing": {"qty": strategy_qty, "strategy_qty": strategy_qty, "estimated_debit": 0.0, "defined_risk_usd": round(float(selected["max_loss_usd"]) * strategy_qty, 2)},
        "order_policy_decision": {
            "entry_order_policy": config.entry_order_policy,
            "option_strategy_mode": config.option_strategy_mode,
            "available_order_types_considered": ["limit_multi_leg_credit"],
            "unsupported_or_rejected_for_options": ["market_for_entry", "naked_short_options"],
            "selected_entry_type": "limit_mleg_credit",
        },
        "abandon_plan": {
            "cancel_unfilled_entry_after_seconds": int(config.abandon_entry_after_seconds),
            "replace_policy": "If the credit is stale, cancel and recompute all four legs from fresh quotes.",
        },
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "top_candidates": accepted[:5],
    }


def _maybe_build_calendar_spread_plan(
    *,
    broker: AlpacaPaperBroker,
    underlying: str,
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
    market_regime: dict[str, Any],
    reason: str,
    option_type: str,
) -> dict[str, Any] | None:
    if not config.enable_short_option_strategies or config.max_legs < 2:
        return None
    expected_mode = "long_call_calendar" if option_type == "call" else "long_put_calendar"
    if config.option_strategy_mode not in {"auto", expected_mode}:
        return None
    near_start = now.date() + timedelta(days=max(1, int(config.calendar_near_min_dte)))
    near_end = now.date() + timedelta(days=max(int(config.calendar_near_min_dte), int(config.calendar_near_max_dte)))
    far_start = now.date() + timedelta(days=max(int(config.calendar_far_min_dte), int(config.calendar_near_max_dte) + 1))
    far_end = now.date() + timedelta(days=max(int(config.calendar_far_max_dte), int(config.calendar_far_min_dte)))
    near = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=near_start.isoformat(),
        expiration_date_lte=near_end.isoformat(),
        option_type=option_type,
        limit=1000,
    )
    far = broker.option_contracts(
        underlying_symbols=underlying,
        expiration_date_gte=far_start.isoformat(),
        expiration_date_lte=far_end.isoformat(),
        option_type=option_type,
        limit=1000,
    )
    if not near or not far:
        return None
    symbols = [str(contract.get("symbol")) for contract in [*near, *far] if contract.get("symbol")]
    snapshots = broker.option_snapshots(symbols)
    candidates = _score_calendar_spreads(
        near_contracts=near,
        far_contracts=far,
        snapshots=snapshots,
        underlying_price=underlying_price,
        forecast=forecast,
        option_type=option_type,
        config=config,
        now=now,
    )
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    strategy = expected_mode
    if not accepted:
        return {
            "action": "hold",
            "reason": f"no_{strategy}_passed_execution_gates",
            "strategy": strategy,
            "strategy_reason": reason,
            "market_regime": market_regime,
            "candidate_count": len(candidates),
            "top_rejections": candidates[:10],
        }
    selected = accepted[0]
    sizing = size_option_position(
        entry_limit_price=float(selected["net_debit"]),
        account_equity=_float_or_none(forecast.get("account_equity")),
        config=config,
    )
    selected["sizing"] = sizing
    if sizing["qty"] <= 0:
        return {
            "action": "hold",
            "reason": "calendar_spread_position_size_below_one_strategy",
            "strategy": strategy,
            "strategy_reason": reason,
            "selected_spread": selected,
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
        }
    strategy_qty = int(sizing["qty"])
    order = {
        "order_class": "mleg",
        "strategy": strategy,
        "symbol": selected["strategy_symbol"],
        "side": "buy",
        "type": "limit",
        "qty": strategy_qty * 2,
        "strategy_qty": strategy_qty,
        "limit_price": round(float(selected["net_debit"]), 2),
        "time_in_force": "day",
        "legs": [
            {"side": "sell", "position_intent": "sell_to_open", "symbol": selected["near_leg"]["symbol"], "ratio_qty": 1},
            {"side": "buy", "position_intent": "buy_to_open", "symbol": selected["far_leg"]["symbol"], "ratio_qty": 1},
        ],
        "policy_reason": "Long calendar spread sells the near expiry and buys the farther expiry at the same strike.",
    }
    return {
        "action": "buy_option",
        "underlying": underlying.upper(),
        "option_type": "multi_leg",
        "strategy": strategy,
        "strategy_reason": reason,
        "market_regime": market_regime,
        "selected_spread": selected,
        "selected_contract": {
            "symbol": selected["strategy_symbol"],
            "name": strategy.replace("_", " ").title(),
            "option_type": "multi_leg",
            "expiration_date": selected["near_expiration_date"],
            "dte": selected["near_dte"],
            "strike": selected["strike"],
            "bid": selected["net_bid"],
            "ask": selected["net_ask"],
            "mid": selected["net_mid"],
            "spread_pct": selected["combined_spread_pct"],
            "premium": round(float(selected["net_debit"]) * 100.0, 2),
            "score": selected["score"],
        },
        "trade_quality": selected.get("trade_quality"),
        "order": order,
        "risk": {
            "estimated_debit": round(float(selected["net_debit"]) * strategy_qty * 100.0, 2),
            "max_loss_at_entry": round(float(selected["net_debit"]) * strategy_qty * 100.0, 2),
            "near_expiry_assignment_risk": "Contains a short near-expiry American-style option. Close or roll before near expiry.",
            "policy": "Calendar spreads are paper multi-leg experiments and require explicit short-option strategy enablement.",
        },
        "exit_plan": {
            "policy": "Close before short near expiry, or earlier if the debit target/loss guard or forecast direction invalidates the thesis.",
            "primary_exit": {"type": "managed_calendar_spread_after_fill"},
        },
        "sizing": sizing,
        "order_policy_decision": {
            "entry_order_policy": config.entry_order_policy,
            "option_strategy_mode": config.option_strategy_mode,
            "available_order_types_considered": ["limit_multi_leg_debit"],
            "unsupported_or_rejected_for_options": ["market_for_entry", "naked_short_options"],
            "selected_entry_type": "limit_mleg_debit",
        },
        "abandon_plan": {
            "cancel_unfilled_entry_after_seconds": int(config.abandon_entry_after_seconds),
            "replace_policy": "If the calendar debit is stale, cancel and recompute both expiries from fresh quotes.",
        },
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "top_candidates": accepted[:5],
    }


def _score_long_straddle_pairs(
    *,
    calls: list[dict[str, Any]],
    puts: list[dict[str, Any]],
    snapshots: dict[str, Any],
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
) -> list[dict[str, Any]]:
    put_by_key = {
        (str(contract.get("expiration_date")), _float_or_none(contract.get("strike_price") or contract.get("strike"))): contract
        for contract in puts
    }
    pairs: list[dict[str, Any]] = []
    for call in calls:
        strike = _float_or_none(call.get("strike_price") or call.get("strike"))
        expiry_raw = str(call.get("expiration_date") or "")
        put = put_by_key.get((expiry_raw, strike))
        if strike is None or put is None:
            continue
        expiry = _parse_expiry(expiry_raw)
        dte = None if expiry is None else max(0, (expiry - now.date()).days)
        call_leg = _multi_leg_snapshot(call, snapshots.get(str(call.get("symbol") or "")) or {}, option_type="call")
        put_leg = _multi_leg_snapshot(put, snapshots.get(str(put.get("symbol") or "")) or {}, option_type="put")
        combined_bid = (call_leg.get("bid") or 0.0) + (put_leg.get("bid") or 0.0)
        combined_ask = (call_leg.get("ask") or 0.0) + (put_leg.get("ask") or 0.0)
        combined_mid = (call_leg.get("mid") or 0.0) + (put_leg.get("mid") or 0.0)
        combined_limit = min(combined_ask, combined_mid * (1.0 + float(config.limit_price_offset_pct))) if combined_mid > 0 and combined_ask > 0 else None
        combined_spread_pct = None if combined_mid <= 0 else (combined_ask - combined_bid) / max(combined_mid, 1e-9)
        premium = None if combined_limit is None else combined_limit * 100.0
        combined_theta = None
        if call_leg.get("theta") is not None or put_leg.get("theta") is not None:
            combined_theta = (call_leg.get("theta") or 0.0) + (put_leg.get("theta") or 0.0)
        horizon_hours = _float_or_none(forecast.get("horizon_hours"))
        theta_decay_usd_for_horizon = None
        if combined_theta is not None and horizon_hours is not None:
            theta_decay_usd_for_horizon = abs(combined_theta) * 100.0 * max(horizon_hours, 0.0) / 24.0
        forecast_price = _float_or_none(forecast.get("predicted_price"))
        reasons = []
        if call.get("tradable") is False or put.get("tradable") is False:
            reasons.append("leg_not_tradable")
        if dte is None:
            reasons.append("missing_expiration")
        elif dte == 0 and not config.allow_0dte:
            reasons.append("zero_dte_blocked")
        elif dte < config.min_dte:
            reasons.append("dte_below_min")
        elif dte > config.max_dte:
            reasons.append("dte_above_max")
        if call_leg.get("bid") is None or call_leg.get("ask") is None or put_leg.get("bid") is None or put_leg.get("ask") is None:
            reasons.append("missing_live_bid_ask")
        if combined_spread_pct is None or combined_spread_pct > config.max_spread_pct:
            reasons.append("combined_spread_too_wide")
        if premium is None:
            reasons.append("missing_combined_premium")
        elif premium > float(config.max_total_debit) * float(config.straddle_max_debit_multiplier):
            reasons.append("combined_premium_above_debit_budget")
        if config.require_greeks:
            for leg_name, leg in (("call", call_leg), ("put", put_leg)):
                if any(leg.get(key) is None for key in ("delta", "gamma", "theta", "vega")):
                    reasons.append(f"{leg_name}_missing_greeks")
        if (call_leg.get("open_interest") is not None and call_leg["open_interest"] < config.min_open_interest) or (
            put_leg.get("open_interest") is not None and put_leg["open_interest"] < config.min_open_interest
        ):
            reasons.append("open_interest_below_min")
        atm_distance_pct = abs(float(strike) - float(underlying_price)) / max(abs(float(underlying_price)), 1e-9)
        forecast_expected_return = abs(_float_or_none(forecast.get("expected_return")) or 0.0)
        score = atm_distance_pct + (combined_spread_pct or 9.0) * 0.75 - forecast_expected_return * 0.20
        rounded_combined_limit = round(float(combined_limit), 2) if combined_limit is not None else None
        trade_quality = _straddle_trade_quality(
            underlying_price=underlying_price,
            forecast_price=forecast_price,
            strike=strike,
            combined_bid=combined_bid,
            combined_ask=combined_ask,
            combined_mid=combined_mid,
            combined_limit_price=rounded_combined_limit,
            combined_spread_pct=combined_spread_pct,
            premium=premium,
            theta_decay_usd_for_horizon=theta_decay_usd_for_horizon,
            call_open_interest=call_leg.get("open_interest"),
            put_open_interest=put_leg.get("open_interest"),
            config=config,
        )
        pairs.append(
            {
                "strategy_symbol": f"{call_leg['symbol']}+{put_leg['symbol']}",
                "strategy": "long_straddle",
                "expiration_date": expiry_raw,
                "dte": dte,
                "strike": strike,
                "call": call_leg,
                "put": put_leg,
                "combined_bid": round(combined_bid, 4),
                "combined_ask": round(combined_ask, 4),
                "combined_mid": round(combined_mid, 4),
                "combined_limit_price": rounded_combined_limit,
                "combined_spread_pct": combined_spread_pct,
                "premium": round(premium, 2) if premium is not None else None,
                "atm_distance_pct": round(atm_distance_pct, 6),
                "trade_quality": trade_quality,
                "accepted": not reasons,
                "reasons": list(dict.fromkeys(reasons)),
                "score": float(score),
            }
        )
    pairs.sort(key=lambda item: (not item["accepted"], item["score"], item.get("premium") or float("inf")))
    return pairs


def _score_short_iron_butterflies(
    *,
    calls: list[dict[str, Any]],
    puts: list[dict[str, Any]],
    snapshots: dict[str, Any],
    underlying_price: float,
    forecast: dict[str, Any],
    config: OptionExecutionConfig,
    now: datetime,
) -> list[dict[str, Any]]:
    call_by_key = _contracts_by_expiry_strike(calls)
    put_by_key = _contracts_by_expiry_strike(puts)
    by_expiry: dict[str, list[float]] = {}
    for expiry, strike in set(call_by_key) | set(put_by_key):
        by_expiry.setdefault(expiry, []).append(float(strike))
    candidates: list[dict[str, Any]] = []
    for expiry_raw, strikes in by_expiry.items():
        strikes = sorted(set(strikes))
        center_candidates = [strike for strike in strikes if (expiry_raw, strike) in call_by_key and (expiry_raw, strike) in put_by_key]
        if not center_candidates:
            continue
        center = min(center_candidates, key=lambda value: abs(value - float(underlying_price)))
        wing_target = max(float(underlying_price) * float(config.iron_butterfly_wing_width_pct), 0.5)
        lower_choices = [strike for strike in strikes if strike < center and (expiry_raw, strike) in put_by_key]
        upper_choices = [strike for strike in strikes if strike > center and (expiry_raw, strike) in call_by_key]
        if not lower_choices or not upper_choices:
            continue
        lower = min(lower_choices, key=lambda value: abs((center - value) - wing_target))
        upper = min(upper_choices, key=lambda value: abs((value - center) - wing_target))
        expiry = _parse_expiry(expiry_raw)
        dte = None if expiry is None else max(0, (expiry - now.date()).days)
        long_put = _multi_leg_snapshot(put_by_key[(expiry_raw, lower)], snapshots.get(str(put_by_key[(expiry_raw, lower)].get("symbol") or "")) or {}, option_type="put")
        short_put = _multi_leg_snapshot(put_by_key[(expiry_raw, center)], snapshots.get(str(put_by_key[(expiry_raw, center)].get("symbol") or "")) or {}, option_type="put")
        short_call = _multi_leg_snapshot(call_by_key[(expiry_raw, center)], snapshots.get(str(call_by_key[(expiry_raw, center)].get("symbol") or "")) or {}, option_type="call")
        long_call = _multi_leg_snapshot(call_by_key[(expiry_raw, upper)], snapshots.get(str(call_by_key[(expiry_raw, upper)].get("symbol") or "")) or {}, option_type="call")
        legs = [long_put, short_put, short_call, long_call]
        net_credit = (short_put.get("bid") or 0.0) + (short_call.get("bid") or 0.0) - (long_put.get("ask") or 0.0) - (long_call.get("ask") or 0.0)
        net_mid = (short_put.get("mid") or 0.0) + (short_call.get("mid") or 0.0) - (long_put.get("mid") or 0.0) - (long_call.get("mid") or 0.0)
        debit_to_close_at_quotes = (short_put.get("ask") or 0.0) + (short_call.get("ask") or 0.0) - (long_put.get("bid") or 0.0) - (long_call.get("bid") or 0.0)
        combined_spread_pct = None if abs(net_mid) <= 1e-9 else abs(debit_to_close_at_quotes - net_credit) / max(abs(net_mid), 1e-9)
        wing_width = min(center - lower, upper - center)
        max_profit_usd = max(0.0, net_credit) * 100.0
        max_loss_usd = max(0.0, wing_width - max(0.0, net_credit)) * 100.0
        forecast_price = _float_or_none(forecast.get("predicted_price"))
        expected_abs_move = None if forecast_price is None else abs(float(forecast_price) - float(underlying_price))
        lower_breakeven = center - max(0.0, net_credit)
        upper_breakeven = center + max(0.0, net_credit)
        reasons = []
        if any(not leg.get("symbol") for leg in legs):
            reasons.append("missing_leg_symbol")
        if any(leg.get("bid") is None or leg.get("ask") is None for leg in legs):
            reasons.append("missing_live_bid_ask")
        if dte is None:
            reasons.append("missing_expiration")
        elif dte == 0 and not config.allow_0dte:
            reasons.append("zero_dte_blocked")
        elif dte < config.min_dte:
            reasons.append("dte_below_min")
        elif dte > config.max_dte:
            reasons.append("dte_above_max")
        if net_credit <= 0:
            reasons.append("non_positive_net_credit")
        if combined_spread_pct is None or combined_spread_pct > config.max_spread_pct * 2.0:
            reasons.append("combined_spread_too_wide")
        if max_loss_usd <= 0 or max_loss_usd > float(config.max_total_debit):
            reasons.append("defined_risk_above_budget")
        if expected_abs_move is not None and expected_abs_move > max(wing_width, max(0.0, net_credit) * 1.5):
            reasons.append("forecast_move_too_large_for_short_volatility")
        if config.require_greeks:
            for leg_name, leg in (("long_put", long_put), ("short_put", short_put), ("short_call", short_call), ("long_call", long_call)):
                if any(leg.get(key) is None for key in ("delta", "gamma", "theta", "vega")):
                    reasons.append(f"{leg_name}_missing_greeks")
        min_open_interest = min(value for value in [leg.get("open_interest") for leg in legs] if value is not None) if any(leg.get("open_interest") is not None for leg in legs) else None
        if min_open_interest is not None and min_open_interest < config.min_open_interest:
            reasons.append("open_interest_below_min")
        atm_distance_pct = abs(center - float(underlying_price)) / max(abs(float(underlying_price)), 1e-9)
        score = atm_distance_pct + (combined_spread_pct or 9.0) * 0.5 - _safe_ratio(max_profit_usd, max_loss_usd) * 0.15 if max_loss_usd > 0 else 99.0
        trade_quality = _defined_risk_credit_quality(
            strategy="short_iron_butterfly",
            underlying_price=underlying_price,
            center_strike=center,
            lower_breakeven=lower_breakeven,
            upper_breakeven=upper_breakeven,
            net_credit=net_credit,
            max_profit_usd=max_profit_usd,
            max_loss_usd=max_loss_usd,
            combined_spread_pct=combined_spread_pct,
            min_open_interest=min_open_interest,
            forecast_price=forecast_price,
            config=config,
        )
        candidates.append(
            {
                "strategy_symbol": f"{long_put['symbol']}+{short_put['symbol']}+{short_call['symbol']}+{long_call['symbol']}",
                "strategy": "short_iron_butterfly",
                "expiration_date": expiry_raw,
                "dte": dte,
                "lower_strike": lower,
                "center_strike": center,
                "upper_strike": upper,
                "long_put": long_put,
                "short_put": short_put,
                "short_call": short_call,
                "long_call": long_call,
                "net_credit": round(float(net_credit), 4),
                "net_mid": round(float(net_mid), 4),
                "combined_spread_pct": combined_spread_pct,
                "max_profit_usd": round(max_profit_usd, 2),
                "max_loss_usd": round(max_loss_usd, 2),
                "lower_breakeven_price": round(lower_breakeven, 4),
                "upper_breakeven_price": round(upper_breakeven, 4),
                "accepted": not reasons,
                "reasons": list(dict.fromkeys(reasons)),
                "score": float(score),
                "trade_quality": trade_quality,
            }
        )
    candidates.sort(key=lambda item: (not item["accepted"], item["score"], item.get("max_loss_usd") or float("inf")))
    return candidates


def _score_calendar_spreads(
    *,
    near_contracts: list[dict[str, Any]],
    far_contracts: list[dict[str, Any]],
    snapshots: dict[str, Any],
    underlying_price: float,
    forecast: dict[str, Any],
    option_type: str,
    config: OptionExecutionConfig,
    now: datetime,
) -> list[dict[str, Any]]:
    far_by_strike: dict[float, list[dict[str, Any]]] = {}
    for contract in far_contracts:
        strike = _float_or_none(contract.get("strike_price") or contract.get("strike"))
        if strike is not None:
            far_by_strike.setdefault(float(strike), []).append(contract)
    candidates: list[dict[str, Any]] = []
    for near in near_contracts:
        strike = _float_or_none(near.get("strike_price") or near.get("strike"))
        if strike is None or float(strike) not in far_by_strike:
            continue
        near_expiry = _parse_expiry(near.get("expiration_date"))
        near_dte = None if near_expiry is None else max(0, (near_expiry - now.date()).days)
        near_leg = _multi_leg_snapshot(near, snapshots.get(str(near.get("symbol") or "")) or {}, option_type=option_type)
        for far in sorted(far_by_strike[float(strike)], key=lambda row: str(row.get("expiration_date") or "")):
            far_expiry = _parse_expiry(far.get("expiration_date"))
            far_dte = None if far_expiry is None else max(0, (far_expiry - now.date()).days)
            if near_dte is None or far_dte is None or far_dte <= near_dte:
                continue
            far_leg = _multi_leg_snapshot(far, snapshots.get(str(far.get("symbol") or "")) or {}, option_type=option_type)
            net_debit = (far_leg.get("ask") or 0.0) - (near_leg.get("bid") or 0.0)
            net_mid = (far_leg.get("mid") or 0.0) - (near_leg.get("mid") or 0.0)
            net_bid = (far_leg.get("bid") or 0.0) - (near_leg.get("ask") or 0.0)
            net_ask = (far_leg.get("ask") or 0.0) - (near_leg.get("bid") or 0.0)
            combined_spread_pct = None if abs(net_mid) <= 1e-9 else abs(net_ask - net_bid) / max(abs(net_mid), 1e-9)
            premium = net_debit * 100.0
            reasons = []
            if near.get("tradable") is False or far.get("tradable") is False:
                reasons.append("leg_not_tradable")
            if near_leg.get("bid") is None or near_leg.get("ask") is None or far_leg.get("bid") is None or far_leg.get("ask") is None:
                reasons.append("missing_live_bid_ask")
            if near_dte < int(config.calendar_near_min_dte) or near_dte > int(config.calendar_near_max_dte):
                reasons.append("near_dte_outside_calendar_window")
            if far_dte < int(config.calendar_far_min_dte) or far_dte > int(config.calendar_far_max_dte):
                reasons.append("far_dte_outside_calendar_window")
            if net_debit <= 0:
                reasons.append("non_positive_calendar_debit")
            if premium > float(config.max_total_debit):
                reasons.append("calendar_debit_above_budget")
            if combined_spread_pct is None or combined_spread_pct > config.max_spread_pct * 2.0:
                reasons.append("combined_spread_too_wide")
            if config.require_greeks:
                for leg_name, leg in (("near", near_leg), ("far", far_leg)):
                    if any(leg.get(key) is None for key in ("delta", "gamma", "theta", "vega")):
                        reasons.append(f"{leg_name}_missing_greeks")
            min_open_interest = min(value for value in [near_leg.get("open_interest"), far_leg.get("open_interest")] if value is not None) if near_leg.get("open_interest") is not None or far_leg.get("open_interest") is not None else None
            if min_open_interest is not None and min_open_interest < config.min_open_interest:
                reasons.append("open_interest_below_min")
            forecast_price = _float_or_none(forecast.get("predicted_price"))
            target_strike_distance = abs(float(strike) - float(underlying_price)) / max(abs(float(underlying_price)), 1e-9)
            calendar_span = far_dte - near_dte
            score = target_strike_distance + (combined_spread_pct or 9.0) * 0.5 + max(0, calendar_span - 21) * 0.003
            trade_quality = _calendar_trade_quality(
                option_type=option_type,
                underlying_price=underlying_price,
                forecast_price=forecast_price,
                strike=float(strike),
                net_debit=net_debit,
                net_mid=net_mid,
                combined_spread_pct=combined_spread_pct,
                premium=premium,
                min_open_interest=min_open_interest,
                near_dte=near_dte,
                far_dte=far_dte,
                config=config,
            )
            candidates.append(
                {
                    "strategy_symbol": f"{near_leg['symbol']}+{far_leg['symbol']}",
                    "strategy": f"long_{option_type}_calendar",
                    "strike": float(strike),
                    "near_expiration_date": str(near.get("expiration_date") or ""),
                    "far_expiration_date": str(far.get("expiration_date") or ""),
                    "near_dte": near_dte,
                    "far_dte": far_dte,
                    "near_leg": near_leg,
                    "far_leg": far_leg,
                    "net_bid": round(float(net_bid), 4),
                    "net_ask": round(float(net_ask), 4),
                    "net_mid": round(float(net_mid), 4),
                    "net_debit": round(float(net_debit), 4),
                    "combined_spread_pct": combined_spread_pct,
                    "premium": round(float(premium), 2),
                    "accepted": not reasons,
                    "reasons": list(dict.fromkeys(reasons)),
                    "score": float(score),
                    "trade_quality": trade_quality,
                }
            )
    candidates.sort(key=lambda item: (not item["accepted"], item["score"], item.get("premium") or float("inf")))
    return candidates


def _multi_leg_snapshot(contract: dict[str, Any], snapshot: dict[str, Any], *, option_type: str) -> dict[str, Any]:
    quote = _snapshot_quote(snapshot)
    greeks = _snapshot_greeks(snapshot)
    bid = quote.get("bid")
    ask = quote.get("ask")
    mid = None if bid is None or ask is None else (bid + ask) / 2.0
    return {
        "symbol": str(contract.get("symbol") or ""),
        "option_type": option_type,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "delta": _float_or_none(greeks.get("delta")),
        "gamma": _float_or_none(greeks.get("gamma")),
        "theta": _float_or_none(greeks.get("theta")),
        "vega": _float_or_none(greeks.get("vega")),
        "open_interest": _float_or_none(contract.get("open_interest")),
    }


def _contracts_by_expiry_strike(contracts: list[dict[str, Any]]) -> dict[tuple[str, float], dict[str, Any]]:
    output: dict[tuple[str, float], dict[str, Any]] = {}
    for contract in contracts:
        expiry = str(contract.get("expiration_date") or "")
        strike = _float_or_none(contract.get("strike_price") or contract.get("strike"))
        if expiry and strike is not None:
            output[(expiry, float(strike))] = contract
    return output


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


def _single_leg_trade_quality(
    *,
    option_type: str,
    underlying_price: float,
    forecast_price: float | None,
    strike: float | None,
    bid: float | None,
    ask: float | None,
    mid: float | None,
    limit_price: float | None,
    spread_pct: float | None,
    open_interest: float | None,
    premium: float | None,
    forecast_edge_usd: float | None,
    theta_decay_usd_for_horizon: float | None,
    theta_premium_pct_per_day: float | None,
    config: OptionExecutionConfig,
) -> dict[str, Any]:
    spread_cost_usd = None if bid is None or ask is None else max(0.0, ask - bid) * 100.0
    half_spread_cost_usd = None if spread_cost_usd is None else spread_cost_usd / 2.0
    breakeven_price = None
    forecast_breakeven_edge = None
    if strike is not None and limit_price is not None:
        breakeven_price = strike + limit_price if option_type == "call" else strike - limit_price
        if forecast_price is not None:
            forecast_breakeven_edge = forecast_price - breakeven_price if option_type == "call" else breakeven_price - forecast_price
    theta_ratio = _safe_ratio(theta_decay_usd_for_horizon, forecast_edge_usd)
    spread_ratio = _safe_ratio(half_spread_cost_usd, forecast_edge_usd)
    edge_after_costs = None
    if forecast_edge_usd is not None:
        edge_after_costs = forecast_edge_usd - (theta_decay_usd_for_horizon or 0.0) - (half_spread_cost_usd or 0.0)
    reward_risk_ratio = _safe_ratio(float(config.take_profit_pct), float(config.stop_loss_pct))
    score = _trade_quality_score(
        spread_pct=spread_pct,
        max_spread_pct=config.max_spread_pct,
        theta_ratio=theta_ratio,
        spread_ratio=spread_ratio,
        edge_after_costs=edge_after_costs,
        premium=premium,
        open_interest=open_interest,
        breakeven_edge=forecast_breakeven_edge,
        reward_risk_ratio=reward_risk_ratio,
    )
    return {
        "score": score,
        "grade": _quality_grade(score),
        "breakeven_price": _round_or_none(breakeven_price, 4),
        "forecast_breakeven_edge": _round_or_none(forecast_breakeven_edge, 4),
        "expected_move_vs_breakeven_pct": _round_or_none(_safe_ratio(forecast_breakeven_edge, underlying_price), 6),
        "spread_pct": _round_or_none(spread_pct, 6),
        "spread_cost_usd": _round_or_none(spread_cost_usd, 2),
        "half_spread_cost_usd": _round_or_none(half_spread_cost_usd, 2),
        "theta_cost_for_horizon_usd": _round_or_none(theta_decay_usd_for_horizon, 2),
        "theta_to_forecast_edge_ratio": _round_or_none(theta_ratio, 4),
        "spread_to_forecast_edge_ratio": _round_or_none(spread_ratio, 4),
        "forecast_edge_usd_delta_adjusted": _round_or_none(forecast_edge_usd, 2),
        "forecast_edge_after_theta_and_half_spread_usd": _round_or_none(edge_after_costs, 2),
        "reward_risk_ratio_from_exit_policy": _round_or_none(reward_risk_ratio, 4),
        "open_interest": _round_or_none(open_interest, 2),
        "premium_usd": _round_or_none(premium, 2),
        "mid": _round_or_none(mid, 4),
        "limit_price": _round_or_none(limit_price, 4),
        "interpretation": _quality_interpretation(score, edge_after_costs, forecast_breakeven_edge, spread_pct, theta_ratio),
    }


def _straddle_trade_quality(
    *,
    underlying_price: float,
    forecast_price: float | None,
    strike: float | None,
    combined_bid: float,
    combined_ask: float,
    combined_mid: float,
    combined_limit_price: float | None,
    combined_spread_pct: float | None,
    premium: float | None,
    theta_decay_usd_for_horizon: float | None,
    call_open_interest: float | None,
    put_open_interest: float | None,
    config: OptionExecutionConfig,
) -> dict[str, Any]:
    spread_cost_usd = max(0.0, combined_ask - combined_bid) * 100.0 if combined_ask > 0 and combined_bid > 0 else None
    half_spread_cost_usd = None if spread_cost_usd is None else spread_cost_usd / 2.0
    upper_breakeven = None if strike is None or combined_limit_price is None else strike + combined_limit_price
    lower_breakeven = None if strike is None or combined_limit_price is None else strike - combined_limit_price
    expected_abs_move = None if forecast_price is None else abs(forecast_price - underlying_price)
    move_needed = combined_limit_price
    move_edge = None if expected_abs_move is None or move_needed is None else expected_abs_move - move_needed
    forecast_edge_usd = None if expected_abs_move is None else expected_abs_move * 100.0
    theta_ratio = _safe_ratio(theta_decay_usd_for_horizon, forecast_edge_usd)
    spread_ratio = _safe_ratio(half_spread_cost_usd, forecast_edge_usd)
    edge_after_costs = None
    if forecast_edge_usd is not None:
        edge_after_costs = forecast_edge_usd - (theta_decay_usd_for_horizon or 0.0) - (half_spread_cost_usd or 0.0)
    min_open_interest = min(value for value in [call_open_interest, put_open_interest] if value is not None) if any(value is not None for value in [call_open_interest, put_open_interest]) else None
    score = _trade_quality_score(
        spread_pct=combined_spread_pct,
        max_spread_pct=config.max_spread_pct,
        theta_ratio=theta_ratio,
        spread_ratio=spread_ratio,
        edge_after_costs=edge_after_costs,
        premium=premium,
        open_interest=min_open_interest,
        breakeven_edge=move_edge,
        reward_risk_ratio=None,
    )
    return {
        "score": score,
        "grade": _quality_grade(score),
        "upper_breakeven_price": _round_or_none(upper_breakeven, 4),
        "lower_breakeven_price": _round_or_none(lower_breakeven, 4),
        "expected_abs_move": _round_or_none(expected_abs_move, 4),
        "expected_move_vs_debit_edge": _round_or_none(move_edge, 4),
        "expected_move_vs_debit_pct": _round_or_none(_safe_ratio(move_edge, underlying_price), 6),
        "spread_pct": _round_or_none(combined_spread_pct, 6),
        "spread_cost_usd": _round_or_none(spread_cost_usd, 2),
        "half_spread_cost_usd": _round_or_none(half_spread_cost_usd, 2),
        "theta_cost_for_horizon_usd": _round_or_none(theta_decay_usd_for_horizon, 2),
        "theta_to_forecast_edge_ratio": _round_or_none(theta_ratio, 4),
        "spread_to_forecast_edge_ratio": _round_or_none(spread_ratio, 4),
        "forecast_edge_after_theta_and_half_spread_usd": _round_or_none(edge_after_costs, 2),
        "minimum_leg_open_interest": _round_or_none(min_open_interest, 2),
        "premium_usd": _round_or_none(premium, 2),
        "combined_mid": _round_or_none(combined_mid, 4),
        "combined_limit_price": _round_or_none(combined_limit_price, 4),
        "interpretation": _quality_interpretation(score, edge_after_costs, move_edge, combined_spread_pct, theta_ratio),
    }


def _defined_risk_credit_quality(
    *,
    strategy: str,
    underlying_price: float,
    center_strike: float,
    lower_breakeven: float,
    upper_breakeven: float,
    net_credit: float,
    max_profit_usd: float,
    max_loss_usd: float,
    combined_spread_pct: float | None,
    min_open_interest: float | None,
    forecast_price: float | None,
    config: OptionExecutionConfig,
) -> dict[str, Any]:
    forecast_inside_profit_zone = None
    if forecast_price is not None:
        forecast_inside_profit_zone = lower_breakeven <= forecast_price <= upper_breakeven
    reward_risk_ratio = _safe_ratio(max_profit_usd, max_loss_usd)
    center_distance_pct = abs(center_strike - underlying_price) / max(abs(underlying_price), 1e-9)
    score = _trade_quality_score(
        spread_pct=combined_spread_pct,
        max_spread_pct=config.max_spread_pct * 2.0,
        theta_ratio=0.0,
        spread_ratio=None,
        edge_after_costs=max_profit_usd - max(0.0, center_distance_pct * underlying_price * 100.0),
        premium=max_loss_usd,
        open_interest=min_open_interest,
        breakeven_edge=1.0 if forecast_inside_profit_zone else (-1.0 if forecast_inside_profit_zone is False else None),
        reward_risk_ratio=reward_risk_ratio,
    )
    return {
        "score": score,
        "grade": _quality_grade(score),
        "strategy": strategy,
        "net_credit": _round_or_none(net_credit, 4),
        "max_profit_usd": _round_or_none(max_profit_usd, 2),
        "max_loss_usd": _round_or_none(max_loss_usd, 2),
        "reward_risk_ratio": _round_or_none(reward_risk_ratio, 4),
        "lower_breakeven_price": _round_or_none(lower_breakeven, 4),
        "upper_breakeven_price": _round_or_none(upper_breakeven, 4),
        "forecast_inside_profit_zone": forecast_inside_profit_zone,
        "center_distance_pct": _round_or_none(center_distance_pct, 6),
        "spread_pct": _round_or_none(combined_spread_pct, 6),
        "minimum_leg_open_interest": _round_or_none(min_open_interest, 2),
        "interpretation": [
            "forecast_inside_profit_zone" if forecast_inside_profit_zone else "forecast_outside_profit_zone_or_missing",
            "defined_risk_credit_structure",
            f"quality_{_quality_grade(score)}",
        ],
    }


def _calendar_trade_quality(
    *,
    option_type: str,
    underlying_price: float,
    forecast_price: float | None,
    strike: float,
    net_debit: float,
    net_mid: float,
    combined_spread_pct: float | None,
    premium: float,
    min_open_interest: float | None,
    near_dte: int,
    far_dte: int,
    config: OptionExecutionConfig,
) -> dict[str, Any]:
    forecast_edge = None
    if forecast_price is not None:
        forecast_edge = forecast_price - strike if option_type == "call" else strike - forecast_price
    strike_distance_pct = abs(strike - underlying_price) / max(abs(underlying_price), 1e-9)
    score = _trade_quality_score(
        spread_pct=combined_spread_pct,
        max_spread_pct=config.max_spread_pct * 2.0,
        theta_ratio=0.25,
        spread_ratio=None,
        edge_after_costs=None if forecast_edge is None else forecast_edge * 100.0 - premium * 0.25,
        premium=premium,
        open_interest=min_open_interest,
        breakeven_edge=forecast_edge,
        reward_risk_ratio=None,
    )
    return {
        "score": score,
        "grade": _quality_grade(score),
        "strategy": f"long_{option_type}_calendar",
        "strike": _round_or_none(strike, 4),
        "net_debit": _round_or_none(net_debit, 4),
        "net_mid": _round_or_none(net_mid, 4),
        "premium_usd": _round_or_none(premium, 2),
        "forecast_strike_edge": _round_or_none(forecast_edge, 4),
        "strike_distance_pct": _round_or_none(strike_distance_pct, 6),
        "near_dte": near_dte,
        "far_dte": far_dte,
        "calendar_span_days": far_dte - near_dte,
        "spread_pct": _round_or_none(combined_spread_pct, 6),
        "minimum_leg_open_interest": _round_or_none(min_open_interest, 2),
        "interpretation": [
            "forecast_supports_calendar_direction" if forecast_edge is not None and forecast_edge >= 0 else "forecast_direction_weak_for_calendar",
            "short_near_expiry_long_far_expiry",
            f"quality_{_quality_grade(score)}",
        ],
    }


def _trade_quality_score(
    *,
    spread_pct: float | None,
    max_spread_pct: float,
    theta_ratio: float | None,
    spread_ratio: float | None,
    edge_after_costs: float | None,
    premium: float | None,
    open_interest: float | None,
    breakeven_edge: float | None,
    reward_risk_ratio: float | None,
) -> float:
    score = 50.0
    if spread_pct is not None:
        score += max(-22.0, min(18.0, (1.0 - spread_pct / max(max_spread_pct, 1e-9)) * 18.0))
    else:
        score -= 18.0
    if theta_ratio is not None:
        score += max(-18.0, min(14.0, (0.75 - theta_ratio) / 0.75 * 14.0))
    else:
        score -= 8.0
    if spread_ratio is not None:
        score += max(-14.0, min(10.0, (0.50 - spread_ratio) / 0.50 * 10.0))
    if edge_after_costs is not None:
        score += max(-18.0, min(18.0, edge_after_costs / max(abs(premium or 100.0), 1.0) * 25.0))
    if breakeven_edge is not None:
        score += 10.0 if breakeven_edge > 0 else -14.0
    if reward_risk_ratio is not None:
        score += max(-8.0, min(8.0, (reward_risk_ratio - 1.0) * 8.0))
    if open_interest is not None:
        score += max(-8.0, min(8.0, np.log10(max(open_interest, 1.0)) * 2.0))
    return round(float(max(0.0, min(100.0, score))), 2)


def _quality_grade(score: float) -> str:
    if score >= 80:
        return "excellent"
    if score >= 65:
        return "good"
    if score >= 50:
        return "fair"
    if score >= 35:
        return "weak"
    return "poor"


def _quality_interpretation(
    score: float,
    edge_after_costs: float | None,
    breakeven_edge: float | None,
    spread_pct: float | None,
    theta_ratio: float | None,
) -> list[str]:
    notes = []
    if breakeven_edge is not None:
        notes.append("forecast_clears_breakeven" if breakeven_edge > 0 else "forecast_does_not_clear_breakeven")
    if edge_after_costs is not None:
        notes.append("positive_edge_after_theta_and_spread" if edge_after_costs > 0 else "costs_overwhelm_forecast_edge")
    if spread_pct is not None:
        notes.append("spread_acceptable" if spread_pct <= 0.12 else "spread_expensive")
    if theta_ratio is not None:
        notes.append("theta_acceptable_for_horizon" if theta_ratio <= 0.5 else "theta_heavy_for_horizon")
    notes.append(f"quality_{_quality_grade(score)}")
    return notes


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(float(denominator)) <= 1e-12:
        return None
    return float(numerator) / float(denominator)


def _round_or_none(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed
