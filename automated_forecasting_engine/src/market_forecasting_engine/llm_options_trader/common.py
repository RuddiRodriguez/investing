from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.deribit_broker import DeribitOptionsBroker
from market_forecasting_engine.llm_options_trader.knowledge_base import load_strategy_knowledge


@dataclass(frozen=True)
class LLMOptionsRuntimeConfig:
    currency: str = "ETH"
    instrument_currency: str = "USDC"
    data_provider: str = "alpaca"
    data_interval: str = "1m"
    lookback_days: int = 20
    max_price_rows: int = 3500
    forecast_hours: tuple[float, ...] = (0.25, 0.5, 1.0)
    option_chain_limit: int = 80
    min_dte: int = 1
    max_dte: int = 14
    max_order_amount: float = 10.0
    max_order_price: float = 5000.0


def testnet_broker() -> DeribitOptionsBroker:
    return DeribitOptionsBroker(account_mode="testnet")


def live_broker() -> DeribitOptionsBroker:
    return DeribitOptionsBroker(account_mode="live")


def build_market_packet(
    *,
    broker: DeribitOptionsBroker,
    config: LLMOptionsRuntimeConfig,
    process: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    prices = load_underlying_prices(config=config, now=now)
    latest_price = _latest_close(prices)
    account = broker.account_summary(currency=config.instrument_currency)
    open_orders = broker.open_orders(currency=config.instrument_currency, kind="option")
    positions = broker.positions(currency=config.instrument_currency, kind="option")
    trades = _safe_user_trades(broker, currency=config.instrument_currency, count=30)
    chain = option_chain_snapshot(
        broker=broker,
        config=config,
        underlying_price=latest_price,
        now=now,
    )
    tape = short_tape_summary(prices)
    transition = regime_transition_warning(prices, forecast_validation={})
    return {
        "packet_version": "llm_options_trader_v1",
        "process": process,
        "generated_at_utc": now.isoformat(),
        "venue": "deribit",
        "currency": config.currency.upper(),
        "instrument_currency": config.instrument_currency.upper(),
        "underlying_ticker": f"{config.currency.upper()}-USD",
        "price_data_provider": config.data_provider,
        "price_data_interval": config.data_interval,
        "latest_underlying_price": latest_price,
        "account": _compact_account(account),
        "open_option_orders": _filter_instruments(open_orders, config=config),
        "option_positions": _filter_instruments(positions, config=config, nonzero_positions=True),
        "recent_option_trades": _filter_instruments(trades, config=config)[:30],
        "price_summary": price_summary(prices),
        "technical_observations": technical_observations(prices),
        "short_tape_summary": tape,
        "regime_transition_warning": transition,
        "option_tradeability_summary": option_tradeability_summary(chain),
        "recent_price_bars": recent_price_bars(prices, rows=3000),
        "strategy_knowledge": load_strategy_knowledge(),
        "strategy_memory": {},
        "option_chain": chain,
        "api_contract": {
            "execution": "Python is only the Deribit API interface. The LLM decision is final.",
            "forecast_policy": "No precomputed forecast is supplied. The LLM must infer direction, timing, confidence, and order parameters from the supplied market data.",
            "entry_order_requirements": {
                "action": "hold or submit_order",
                "order.side": "buy",
                "order.type": "limit",
                "order.reduce_only": False,
                "order.instrument_name": f"must start with {_instrument_prefix(config)}",
            },
            "exit_order_requirements": {
                "action": "hold, cancel_order, or submit_order",
                "order.type": "limit",
                "order.reduce_only": "must be true for reduce-only position exits",
            },
            "mechanical_limits": {
                "max_order_amount": config.max_order_amount,
                "max_order_price": config.max_order_price,
            },
        },
        "trader_memory": {},
    }


def compact_market_packet(packet: dict[str, Any], *, max_contracts_per_side: int = 8, max_price_bars: int = 45, max_trades: int = 12) -> dict[str, Any]:
    chain = packet.get("option_chain") if isinstance(packet.get("option_chain"), list) else []
    calls = [row for row in chain if str(row.get("option_type") or "").lower() == "call"]
    puts = [row for row in chain if str(row.get("option_type") or "").lower() == "put"]
    selected_chain = calls[: max(1, int(max_contracts_per_side))] + puts[: max(1, int(max_contracts_per_side))]
    return {
        "packet_version": "llm_options_trader_compact_v1",
        "process": packet.get("process"),
        "generated_at_utc": packet.get("generated_at_utc"),
        "venue": packet.get("venue"),
        "account_mode": packet.get("account_mode"),
        "execution_mode": packet.get("execution_mode"),
        "currency": packet.get("currency"),
        "instrument_currency": packet.get("instrument_currency"),
        "underlying_ticker": packet.get("underlying_ticker"),
        "price_data_provider": packet.get("price_data_provider"),
        "price_data_interval": packet.get("price_data_interval"),
        "latest_underlying_price": packet.get("latest_underlying_price"),
        "account": packet.get("account") or {},
        "shadow_trading_budget": packet.get("shadow_trading_budget") or {},
        "open_option_orders": packet.get("open_option_orders") or [],
        "option_positions": packet.get("option_positions") or [],
        "recent_option_trades": (packet.get("recent_option_trades") or [])[: max(0, int(max_trades))],
        "shadow_simulation": packet.get("shadow_simulation") or {},
        "price_summary": packet.get("price_summary") or {},
        "technical_observations": packet.get("technical_observations") or {},
        "short_tape_summary": packet.get("short_tape_summary") or {},
        "regime_transition_warning": packet.get("regime_transition_warning") or {},
        "option_tradeability_summary": packet.get("option_tradeability_summary") or {},
        "adaptive_profit_policy": packet.get("adaptive_profit_policy") or {},
        "entry_mandate": packet.get("entry_mandate") or {},
        "strategy_mode": packet.get("strategy_mode") or {},
        "strategy_knowledge": packet.get("strategy_knowledge") or {},
        "strategy_memory": packet.get("strategy_memory") or {},
        "external_forecasts": packet.get("external_forecasts") or {},
        "forecast_validation": packet.get("forecast_validation") or {},
        "forecast_error_feedback": packet.get("forecast_error_feedback") or {},
        "recent_price_bars": (packet.get("recent_price_bars") or [])[-max(2, int(max_price_bars)) :],
        "option_chain": selected_chain,
        "api_contract": packet.get("api_contract") or {},
        "trader_memory": packet.get("trader_memory") or {},
    }


def option_chain_snapshot(
    *,
    broker: DeribitOptionsBroker,
    config: LLMOptionsRuntimeConfig,
    underlying_price: float,
    now: datetime,
) -> list[dict[str, Any]]:
    prefix = _instrument_prefix(config)
    rows: list[dict[str, Any]] = []
    for instrument in broker.instruments(currency=config.instrument_currency, kind="option", expired=False):
        name = str(instrument.get("instrument_name") or "")
        if not name.startswith(prefix):
            continue
        expiry = _expiry_from_ms(instrument.get("expiration_timestamp"))
        dte = None if expiry is None else max(0, (expiry.date() - now.date()).days)
        if dte is None or dte < config.min_dte or dte > config.max_dte:
            continue
        strike = _float_or_none(instrument.get("strike"))
        if strike is None:
            continue
        rows.append({"instrument": instrument, "distance": abs(strike - underlying_price), "expiry": expiry})
    rows.sort(key=lambda item: (item["distance"], item["expiry"] or now + timedelta(days=999)))
    selected = rows[: max(1, int(config.option_chain_limit))]
    snapshots = []
    for item in selected:
        instrument = item["instrument"]
        name = str(instrument.get("instrument_name"))
        book = broker.order_book(name, depth=5)
        stats = book.get("stats") if isinstance(book.get("stats"), dict) else {}
        greeks = book.get("greeks") if isinstance(book.get("greeks"), dict) else {}
        bid = _float_or_none(book.get("best_bid_price"))
        ask = _float_or_none(book.get("best_ask_price"))
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        snapshots.append(
            {
                "instrument_name": name,
                "option_type": instrument.get("option_type"),
                "strike": _float_or_none(instrument.get("strike")),
                "expiration_utc": item["expiry"].isoformat() if item["expiry"] else None,
                "dte": None if item["expiry"] is None else max(0, (item["expiry"].date() - now.date()).days),
                "min_trade_amount": _float_or_none(instrument.get("min_trade_amount")),
                "tick_size": _float_or_none(instrument.get("tick_size")),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "mark_price": _float_or_none(book.get("mark_price")),
                "spread_pct": None if bid is None or ask is None or mid in {None, 0} else (ask - bid) / max(mid, 1e-12),
                "best_bid_amount": _float_or_none(book.get("best_bid_amount")),
                "best_ask_amount": _float_or_none(book.get("best_ask_amount")),
                "open_interest": _float_or_none(book.get("open_interest")),
                "volume": _float_or_none(stats.get("volume")),
                "volume_usd": _float_or_none(stats.get("volume_usd")),
                "price_change": _float_or_none(stats.get("price_change")),
                "greeks": {
                    "delta": _float_or_none(greeks.get("delta")),
                    "gamma": _float_or_none(greeks.get("gamma")),
                    "theta": _float_or_none(greeks.get("theta")),
                    "vega": _float_or_none(greeks.get("vega")),
                    "rho": _float_or_none(greeks.get("rho")),
                },
                "top_bids": book.get("bids", [])[:5],
                "top_asks": book.get("asks", [])[:5],
            }
        )
    return snapshots


def load_underlying_prices(*, config: LLMOptionsRuntimeConfig, now: datetime) -> pd.DataFrame:
    start = (now - timedelta(days=int(config.lookback_days))).isoformat().replace("+00:00", "Z")
    result = load_prices_with_provider(
        config.data_provider,
        DataRequest(ticker=f"{config.currency.upper()}-USD", start=start, interval=config.data_interval, target_column="close"),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    prices = normalize_price_frame(result.frame, target_column="close")
    if config.max_price_rows and len(prices) > int(config.max_price_rows):
        prices = prices.tail(int(config.max_price_rows))
    return prices


def execute_order_payload(
    *,
    broker: DeribitOptionsBroker,
    config: LLMOptionsRuntimeConfig,
    decision: dict[str, Any],
    require_reduce_only: bool,
    execute: bool,
    label: str,
) -> dict[str, Any]:
    order = decision.get("order")
    if not isinstance(order, dict):
        return {"submitted": False, "reason": "no_order_payload"}
    validation = validate_order_payload(order, config=config, require_reduce_only=require_reduce_only)
    if validation.get("blocks"):
        return {"submitted": False, "reason": "format_validation_failed", "blocks": validation["blocks"], "order": order}
    if not execute:
        return {"submitted": False, "reason": "execution_disabled", "validated_order": validation["order"]}
    normalized = validation["order"]
    if normalized["side"] == "buy":
        result = broker.buy_limit(
            instrument_name=normalized["instrument_name"],
            amount=normalized["amount"],
            price=normalized["price"],
            label=label,
            post_only=normalized["post_only"],
            reduce_only=normalized["reduce_only"],
        )
    else:
        result = broker.sell_limit(
            instrument_name=normalized["instrument_name"],
            amount=normalized["amount"],
            price=normalized["price"],
            label=label,
            post_only=normalized["post_only"],
            reduce_only=normalized["reduce_only"],
        )
    return {"submitted": True, "order": result, "validated_order": normalized}


def validate_order_payload(
    order: dict[str, Any],
    *,
    config: LLMOptionsRuntimeConfig,
    require_reduce_only: bool,
) -> dict[str, Any]:
    blocks: list[str] = []
    instrument = str(order.get("instrument_name") or "")
    side = str(order.get("side") or "").lower()
    order_type = str(order.get("type") or "").lower()
    amount = _float_or_none(order.get("amount"))
    price = _float_or_none(order.get("price"))
    post_only = bool(order.get("post_only", False))
    reduce_only = bool(order.get("reduce_only", False))
    if not instrument.startswith(_instrument_prefix(config)):
        blocks.append("instrument_prefix_not_allowed_for_testnet_experiment")
    if side not in {"buy", "sell"}:
        blocks.append("side_must_be_buy_or_sell")
    if order_type != "limit":
        blocks.append("type_must_be_limit")
    if amount is None or amount <= 0:
        blocks.append("amount_must_be_positive_number")
    elif amount > float(config.max_order_amount):
        blocks.append("amount_exceeds_configured_mechanical_cap")
    if price is None or price <= 0:
        blocks.append("price_must_be_positive_number")
    elif price > float(config.max_order_price):
        blocks.append("price_exceeds_configured_mechanical_cap")
    if require_reduce_only and not reduce_only:
        blocks.append("exit_orders_must_be_reduce_only")
    if blocks:
        return {"blocks": blocks}
    return {
        "blocks": [],
        "order": {
            "instrument_name": instrument,
            "side": side,
            "type": "limit",
            "amount": float(amount),
            "price": float(price),
            "post_only": post_only,
            "reduce_only": reduce_only,
            "time_in_force": str(order.get("time_in_force") or "good_til_cancelled"),
        },
    }


def cancel_order_payload(*, broker: DeribitOptionsBroker, decision: dict[str, Any], execute: bool) -> dict[str, Any]:
    order_id = str(decision.get("order_id") or "").strip()
    if not order_id:
        return {"submitted": False, "reason": "missing_order_id"}
    if not execute:
        return {"submitted": False, "reason": "execution_disabled", "order_id": order_id}
    return {"submitted": True, "cancel": broker.cancel_order(order_id)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=_json_default) + "\n")


def price_summary(prices: pd.DataFrame) -> dict[str, Any]:
    close = prices["close"].dropna()
    if close.empty:
        return {"rows": 0}
    returns = close.pct_change().dropna()
    sma9 = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()
    stoch_rsi = stochastic_rsi(close)
    ad_line = accumulation_distribution(prices)
    adaptive_profit = adaptive_profit_protection(prices)
    return {
        "rows": len(prices),
        "start": str(prices.index[0]),
        "end": str(prices.index[-1]),
        "latest_close": float(close.iloc[-1]),
        "return_15m": _window_return(close, 15),
        "return_30m": _window_return(close, 30),
        "return_60m": _window_return(close, 60),
        "realized_vol_60m": None if len(returns) < 60 else float(returns.tail(60).std()),
        "high_60m": None if len(close) < 60 else float(close.tail(60).max()),
        "low_60m": None if len(close) < 60 else float(close.tail(60).min()),
        "sma_9": _series_latest(sma9),
        "sma_21": _series_latest(sma21),
        "sma_9_21_crossover": _ma_crossover_signal(sma9, sma21),
        "stochastic_rsi": stoch_rsi,
        "accumulation_distribution": ad_line,
        "adaptive_profit_protection": adaptive_profit,
    }


def recent_price_bars(prices: pd.DataFrame, *, rows: int) -> list[dict[str, Any]]:
    frame = prices.tail(rows).reset_index()
    timestamp_column = frame.columns[0]
    payload = []
    for row in frame.to_dict(orient="records"):
        payload.append(
            {
                "timestamp": _timestamp_iso_utc(row.get(timestamp_column)),
                "open": _float_or_none(row.get("open")),
                "high": _float_or_none(row.get("high")),
                "low": _float_or_none(row.get("low")),
                "close": _float_or_none(row.get("close")),
                "volume": _float_or_none(row.get("volume")),
            }
        )
    return payload


def short_tape_summary(prices: pd.DataFrame) -> dict[str, Any]:
    close = pd.to_numeric(prices.get("close"), errors="coerce").dropna()
    if len(close) < 12:
        return {"status": "insufficient_data", "minimum_rows": 12}
    frame = prices.copy()
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    opens = pd.to_numeric(frame["open"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    recent = pd.DataFrame({"open": opens, "close": closes}).dropna().tail(30)
    sma9 = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()
    slope9 = _linear_slope(close.tail(9).tolist())
    slope21 = _linear_slope(close.tail(21).tolist())
    green_10 = int((recent.tail(10)["close"] >= recent.tail(10)["open"]).sum())
    red_10 = int((recent.tail(10)["close"] < recent.tail(10)["open"]).sum())
    move_5 = _window_return(close, 5)
    move_10 = _window_return(close, 10)
    move_15 = _window_return(close, 15)
    direction_score = 0
    for value in (move_5, move_10, move_15, slope9):
        if value is None:
            continue
        direction_score += 1 if value > 0 else -1 if value < 0 else 0
    if green_10 >= 7:
        direction_score += 1
    if red_10 >= 7:
        direction_score -= 1
    if direction_score >= 2:
        tape = "short_term_up"
    elif direction_score <= -2:
        tape = "short_term_down"
    else:
        tape = "mixed_or_chop"
    return {
        "status": "ok",
        "window": "recent_1m_bars",
        "last_5m_return": move_5,
        "last_10m_return": move_10,
        "last_15m_return": move_15,
        "green_bars_last_10": green_10,
        "red_bars_last_10": red_10,
        "sma_9": _series_latest(sma9),
        "sma_21": _series_latest(sma21),
        "sma_9_slope_per_bar": slope9,
        "sma_21_slope_per_bar": slope21,
        "sma_context": _ma_crossover_signal(sma9, sma21),
        "tape": tape,
        "instruction": "Use this as a fast tape read. It can reveal a short up/down move inside a broader oscillating range.",
    }


def regime_transition_warning(prices: pd.DataFrame, *, forecast_validation: dict[str, Any] | None = None) -> dict[str, Any]:
    close = pd.to_numeric(prices.get("close"), errors="coerce").dropna()
    if len(close) < 60:
        return {"status": "insufficient_data", "minimum_rows": 60}
    tape = short_tape_summary(prices)
    latest = float(close.iloc[-1])
    short = close.tail(20)
    long = close.tail(60)
    support = float(long.min())
    resistance = float(long.max())
    range_width = resistance - support
    range_width_pct = None if latest == 0 else range_width / latest
    range_position = None if range_width == 0 else (latest - support) / range_width
    sma = tape.get("sma_context") if isinstance(tape.get("sma_context"), dict) else {}
    validation = forecast_validation if isinstance(forecast_validation, dict) else {}
    summary = str(validation.get("summary") or "")
    bias = _forecast_validation_bias(validation)
    directional_accuracy = _forecast_validation_directional_accuracy(validation)
    transition_state = "range_or_chop"
    reasons: list[str] = []
    if range_width_pct is not None and range_width_pct < 0.01:
        reasons.append("tight_60_bar_range")
    if tape.get("tape") == "short_term_up":
        reasons.append("short_tape_up")
    if tape.get("tape") == "short_term_down":
        reasons.append("short_tape_down")
    if sma.get("signal") in {"golden_cross", "short_above_long"}:
        reasons.append("sma9_above_sma21")
    if sma.get("signal") in {"death_cross", "short_below_long"}:
        reasons.append("sma9_below_sma21")
    if bias is not None and bias > 0:
        reasons.append("recent_forecasts_under_predicted_actual_price")
    if bias is not None and bias < 0:
        reasons.append("recent_forecasts_over_predicted_actual_price")
    if directional_accuracy is not None and directional_accuracy < 0.45:
        reasons.append("forecast_direction_accuracy_weak")
    near_top = range_position is not None and range_position >= 0.72
    near_bottom = range_position is not None and range_position <= 0.28
    if tape.get("tape") == "short_term_up" and (near_top or "sma9_above_sma21" in reasons or (bias is not None and bias > 0)):
        transition_state = "chop_transition_up"
    elif tape.get("tape") == "short_term_down" and (near_bottom or "sma9_below_sma21" in reasons or (bias is not None and bias < 0)):
        transition_state = "chop_transition_down"
    instruction = {
        "chop_transition_up": "Do not treat this as pure mean reversion. Consider early call bias only if option spread, theta, liquidity, and confirmation are acceptable.",
        "chop_transition_down": "Do not treat this as pure mean reversion. Consider early put bias only if option spread, theta, liquidity, and confirmation are acceptable.",
        "range_or_chop": "Treat directional entries cautiously unless tape, forecast validation, MA context, A/D, and option tradeability improve together.",
    }[transition_state]
    return {
        "status": "ok",
        "state": transition_state,
        "reason_codes": reasons,
        "latest_price": latest,
        "support_60": support,
        "resistance_60": resistance,
        "range_width_pct": range_width_pct,
        "range_position": range_position,
        "short_range_high_20": float(short.max()),
        "short_range_low_20": float(short.min()),
        "forecast_validation_summary": summary,
        "forecast_validation_bias_actual_minus_predicted": bias,
        "forecast_directional_accuracy": directional_accuracy,
        "instruction": instruction,
    }


def option_tradeability_summary(option_chain: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [row for row in option_chain if str(row.get("option_type") or "").lower() == "call"]
    puts = [row for row in option_chain if str(row.get("option_type") or "").lower() == "put"]
    return {
        "status": "ok" if option_chain else "empty_chain",
        "best_call_tradeability": _best_tradeability(calls),
        "best_put_tradeability": _best_tradeability(puts),
        "instruction": "Use this to avoid forcing trades through bad spreads/theta/liquidity. If transition evidence is strong and tradeability is fair/good, do not hold only because the previous regime was choppy.",
    }


def apply_forecast_validation_to_transition(transition: dict[str, Any], forecast_validation: dict[str, Any]) -> dict[str, Any]:
    updated = dict(transition) if isinstance(transition, dict) else {}
    bias = _forecast_validation_bias(forecast_validation)
    directional_accuracy = _forecast_validation_directional_accuracy(forecast_validation)
    reason_codes = list(updated.get("reason_codes") if isinstance(updated.get("reason_codes"), list) else [])
    if bias is not None and bias > 0:
        reason_codes.append("recent_forecasts_under_predicted_actual_price")
    elif bias is not None and bias < 0:
        reason_codes.append("recent_forecasts_over_predicted_actual_price")
    if directional_accuracy is not None and directional_accuracy < 0.45:
        reason_codes.append("forecast_direction_accuracy_weak")
    tape_up = "short_tape_up" in reason_codes
    tape_down = "short_tape_down" in reason_codes
    state = str(updated.get("state") or "range_or_chop")
    if state == "range_or_chop" and tape_up and (bias is None or bias >= 0):
        state = "chop_transition_up"
    elif state == "range_or_chop" and tape_down and (bias is None or bias <= 0):
        state = "chop_transition_down"
    instruction = {
        "chop_transition_up": "Do not treat this as pure mean reversion. Recent validation may show the forecast under-read upward movement; consider early call bias only if option spread, theta, liquidity, and confirmation are acceptable.",
        "chop_transition_down": "Do not treat this as pure mean reversion. Recent validation may show the forecast over-read price; consider early put bias only if option spread, theta, liquidity, and confirmation are acceptable.",
        "range_or_chop": "Treat directional entries cautiously unless tape, forecast validation, MA context, A/D, and option tradeability improve together.",
    }.get(state, str(updated.get("instruction") or "Use this as descriptive transition context."))
    updated.update(
        {
            "state": state,
            "reason_codes": sorted(set(reason_codes)),
            "forecast_validation_summary": forecast_validation.get("summary"),
            "forecast_validation_bias_actual_minus_predicted": bias,
            "forecast_directional_accuracy": directional_accuracy,
            "instruction": instruction,
        }
    )
    return updated


def _latest_close(prices: pd.DataFrame) -> float:
    close = prices["close"].dropna()
    if close.empty:
        raise ValueError("No close prices available for LLM options trader.")
    return float(close.iloc[-1])


def _filter_instruments(items: list[dict[str, Any]], *, config: LLMOptionsRuntimeConfig, nonzero_positions: bool = False) -> list[dict[str, Any]]:
    prefix = _instrument_prefix(config)
    output = []
    for item in items:
        if not str(item.get("instrument_name") or "").startswith(prefix):
            continue
        if nonzero_positions and abs(_float_or_none(item.get("size")) or 0.0) <= 0:
            continue
        output.append(item)
    return output


def _safe_user_trades(broker: DeribitOptionsBroker, *, currency: str, count: int) -> list[dict[str, Any]]:
    try:
        return broker.user_trades(currency=currency, kind="option", count=count)
    except Exception:
        return []


def _instrument_prefix(config: LLMOptionsRuntimeConfig) -> str:
    if config.instrument_currency.upper() == config.currency.upper():
        return f"{config.currency.upper()}-"
    return f"{config.currency.upper()}_{config.instrument_currency.upper()}-"


def _compact_account(account: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "currency",
        "balance",
        "equity",
        "available_funds",
        "available_withdrawal_funds",
        "margin_balance",
        "options_session_rpl",
        "options_session_upl",
        "initial_margin",
        "maintenance_margin",
    )
    return {key: account.get(key) for key in keys if key in account}


def _expiry_from_ms(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=UTC)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _timestamp_iso_utc(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)
    return timestamp.isoformat().replace("+00:00", "Z")


def _window_return(close: pd.Series, rows: int) -> float | None:
    if len(close) <= rows:
        return None
    previous = float(close.iloc[-rows - 1])
    if previous == 0:
        return None
    return float(close.iloc[-1] / previous - 1.0)


def _series_latest(series: pd.Series) -> float | None:
    if series.empty or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _series_change(series: pd.Series, rows: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= rows:
        return None
    previous = float(clean.iloc[-rows - 1])
    latest = float(clean.iloc[-1])
    if previous == 0:
        return latest - previous
    return float(latest / previous - 1.0)


def _linear_slope(values: list[float]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    n = len(clean)
    x_mean = (n - 1) / 2.0
    y_mean = sum(clean) / n
    denominator = sum((index - x_mean) ** 2 for index in range(n))
    if denominator == 0:
        return 0.0
    return float(sum((index - x_mean) * (value - y_mean) for index, value in enumerate(clean)) / denominator)


def _forecast_validation_bias(validation: dict[str, Any]) -> float | None:
    recent = validation.get("recent_matured") if isinstance(validation.get("recent_matured"), list) else []
    errors = [_float_or_none(row.get("error")) for row in recent if isinstance(row, dict)]
    errors = [value for value in errors if value is not None]
    if not errors:
        return None
    return float(sum(errors) / len(errors))


def _forecast_validation_directional_accuracy(validation: dict[str, Any]) -> float | None:
    by_horizon = validation.get("by_horizon") if isinstance(validation.get("by_horizon"), dict) else {}
    values = []
    for row in by_horizon.values():
        if isinstance(row, dict):
            value = _float_or_none(row.get("directional_accuracy"))
            if value is not None:
                values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def _best_tradeability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [_tradeability(row) for row in rows]
    candidates = [row for row in candidates if row["instrument_name"]]
    if not candidates:
        return {"status": "none_available", "grade": "poor", "reason": "No contracts for this side were present in the compact chain."}
    candidates.sort(key=lambda row: (row["score"], -abs(row.get("delta") or 0.0)), reverse=True)
    return candidates[0]


def _tradeability(row: dict[str, Any]) -> dict[str, Any]:
    spread = _float_or_none(row.get("spread_pct"))
    bid_amount = _float_or_none(row.get("best_bid_amount"))
    ask_amount = _float_or_none(row.get("best_ask_amount"))
    volume = _float_or_none(row.get("volume"))
    open_interest = _float_or_none(row.get("open_interest"))
    greeks = row.get("greeks") if isinstance(row.get("greeks"), dict) else {}
    theta = _float_or_none(greeks.get("theta"))
    delta = _float_or_none(greeks.get("delta"))
    score = 0.0
    reasons: list[str] = []
    if spread is not None:
        if spread <= 0.08:
            score += 3
            reasons.append("tight_spread")
        elif spread <= 0.15:
            score += 2
            reasons.append("acceptable_spread")
        elif spread <= 0.25:
            score += 0.5
            reasons.append("wide_spread")
        else:
            score -= 2
            reasons.append("very_wide_spread")
    if bid_amount and ask_amount and min(bid_amount, ask_amount) >= 0.1:
        score += 1
        reasons.append("top_book_has_min_size")
    if volume and volume > 0:
        score += 1
        reasons.append("recent_volume_present")
    if open_interest and open_interest > 0:
        score += 0.75
        reasons.append("open_interest_present")
    if theta is not None:
        if abs(theta) <= 5:
            score += 1
            reasons.append("theta_moderate")
        elif abs(theta) >= 12:
            score -= 1
            reasons.append("theta_high")
    if delta is not None and 0.25 <= abs(delta) <= 0.65:
        score += 1
        reasons.append("delta_in_tradeable_range")
    if score >= 5:
        grade = "good"
    elif score >= 2.5:
        grade = "fair"
    else:
        grade = "poor"
    return {
        "status": "ok",
        "instrument_name": row.get("instrument_name"),
        "option_type": row.get("option_type"),
        "grade": grade,
        "score": round(score, 4),
        "reason_codes": reasons,
        "bid": row.get("bid"),
        "ask": row.get("ask"),
        "mid": row.get("mid"),
        "spread_pct": spread,
        "delta": delta,
        "theta": theta,
        "volume": volume,
        "open_interest": open_interest,
        "dte": row.get("dte"),
        "instruction": "Fair/good tradeability does not force a trade; poor tradeability should raise the required market edge.",
    }


def _ma_crossover_signal(short_ma: pd.Series, long_ma: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"short": short_ma, "long": long_ma}).dropna()
    if len(frame) < 2:
        return {"status": "insufficient_data", "signal": "unknown"}
    prev = frame.iloc[-2]
    latest = frame.iloc[-1]
    prev_diff = float(prev["short"] - prev["long"])
    latest_diff = float(latest["short"] - latest["long"])
    if prev_diff <= 0 < latest_diff:
        signal = "golden_cross"
        interpretation = "short SMA crossed above long SMA; bullish context if confirmed by candles, liquidity, and option pricing"
    elif prev_diff >= 0 > latest_diff:
        signal = "death_cross"
        interpretation = "short SMA crossed below long SMA; bearish context if confirmed by candles, liquidity, and option pricing"
    elif latest_diff > 0:
        signal = "short_above_long"
        interpretation = "short SMA remains above long SMA; bullish trend context"
    elif latest_diff < 0:
        signal = "short_below_long"
        interpretation = "short SMA remains below long SMA; bearish trend context"
    else:
        signal = "flat"
        interpretation = "short and long SMA are equal; no crossover edge"
    return {
        "status": "ok",
        "signal": signal,
        "short_sma": float(latest["short"]),
        "long_sma": float(latest["long"]),
        "diff": latest_diff,
        "interpretation": interpretation,
    }


def stochastic_rsi(close: pd.Series, *, rsi_period: int = 14, stoch_period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> dict[str, Any]:
    clean = close.dropna().astype(float)
    min_rows = max(rsi_period + stoch_period + smooth_k + smooth_d, 30)
    if len(clean) < min_rows:
        return {
            "status": "insufficient_data",
            "rsi_period": rsi_period,
            "stoch_period": stoch_period,
            "smooth_k": smooth_k,
            "smooth_d": smooth_d,
        }
    delta = clean.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss.replace(0.0, math.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain > 0.0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0.0) & (avg_gain == 0.0)), 50.0)
    lowest_rsi = rsi.rolling(stoch_period, min_periods=stoch_period).min()
    highest_rsi = rsi.rolling(stoch_period, min_periods=stoch_period).max()
    denominator = (highest_rsi - lowest_rsi).replace(0.0, math.nan)
    raw = ((rsi - lowest_rsi) / denominator).clip(lower=0.0, upper=1.0)
    percent_k = raw.rolling(smooth_k, min_periods=smooth_k).mean()
    percent_d = percent_k.rolling(smooth_d, min_periods=smooth_d).mean()
    latest_raw = _series_latest(raw)
    latest_k = _series_latest(percent_k)
    latest_d = _series_latest(percent_d)
    latest_rsi = _series_latest(rsi)
    signal = "unknown"
    interpretation = "Stochastic RSI unavailable or neutral."
    if latest_k is not None and latest_d is not None:
        if latest_k <= 0.2:
            zone = "oversold"
        elif latest_k >= 0.8:
            zone = "overbought"
        else:
            zone = "mid_range"
        frame = pd.DataFrame({"k": percent_k, "d": percent_d}).dropna()
        crossed_up = False
        crossed_down = False
        if len(frame) >= 2:
            prev = frame.iloc[-2]
            latest = frame.iloc[-1]
            crossed_up = float(prev["k"]) <= float(prev["d"]) and float(latest["k"]) > float(latest["d"])
            crossed_down = float(prev["k"]) >= float(prev["d"]) and float(latest["k"]) < float(latest["d"])
        if crossed_up and latest_k <= 0.35:
            signal = "bullish_reversal_timing"
            interpretation = "StochRSI %K crossed above %D from a low zone; possible bullish timing or put-exit warning if confirmed."
        elif crossed_down and latest_k >= 0.65:
            signal = "bearish_reversal_timing"
            interpretation = "StochRSI %K crossed below %D from a high zone; possible bearish timing or call-exit warning if confirmed."
        elif zone == "oversold":
            signal = "oversold_timing_risk"
            interpretation = "Momentum is stretched low; avoid chasing late puts unless breakdown and option pricing still justify it."
        elif zone == "overbought":
            signal = "overbought_timing_risk"
            interpretation = "Momentum is stretched high; avoid chasing late calls unless breakout and option pricing still justify it."
        else:
            signal = "neutral_timing"
            interpretation = "StochRSI is not at an extreme; use price action, trend, liquidity, and Greeks for confirmation."
    return {
        "status": "ok",
        "rsi_period": rsi_period,
        "stoch_period": stoch_period,
        "smooth_k": smooth_k,
        "smooth_d": smooth_d,
        "rsi": latest_rsi,
        "stoch_rsi": latest_raw,
        "percent_k": latest_k,
        "percent_d": latest_d,
        "signal": signal,
        "interpretation": interpretation,
    }


def accumulation_distribution(prices: pd.DataFrame, *, short_window: int = 20, long_window: int = 60) -> dict[str, Any]:
    required = {"high", "low", "close", "volume"}
    if not required.issubset(set(prices.columns)):
        return {"status": "missing_ohlcv_columns", "required": sorted(required)}
    frame = prices[["high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(frame) < max(5, short_window):
        return {"status": "insufficient_data", "short_window": short_window, "long_window": long_window}
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    volume = frame["volume"].clip(lower=0.0)
    candle_range = (high - low).replace(0.0, math.nan)
    money_flow_multiplier = (((close - low) - (high - close)) / candle_range).fillna(0.0).clip(lower=-1.0, upper=1.0)
    money_flow_volume = money_flow_multiplier * volume
    ad = money_flow_volume.cumsum()
    short_change = _series_change(ad, short_window)
    long_change = _series_change(ad, long_window)
    price_short_change = _series_change(close, short_window)
    signal = "neutral_volume_flow"
    interpretation = "A/D is not giving a strong confirmation or divergence signal."
    if short_change is not None and price_short_change is not None:
        if price_short_change > 0 and short_change > 0:
            signal = "buying_pressure_confirms_rise"
            interpretation = "Price and A/D rose together; volume flow confirms bullish pressure."
        elif price_short_change < 0 and short_change < 0:
            signal = "selling_pressure_confirms_drop"
            interpretation = "Price and A/D fell together; volume flow confirms bearish pressure."
        elif price_short_change > 0 and short_change < 0:
            signal = "bearish_distribution_divergence"
            interpretation = "Price rose while A/D fell; rally may be weak or distribution-driven."
        elif price_short_change < 0 and short_change > 0:
            signal = "bullish_accumulation_divergence"
            interpretation = "Price fell while A/D rose; selling may be weakening or buyers may be accumulating."
    return {
        "status": "ok",
        "short_window": short_window,
        "long_window": long_window,
        "latest_ad": _series_latest(ad),
        "short_change": short_change,
        "long_change": long_change,
        "price_short_change_pct": price_short_change,
        "latest_money_flow_multiplier": _series_latest(money_flow_multiplier),
        "latest_money_flow_volume": _series_latest(money_flow_volume),
        "signal": signal,
        "interpretation": interpretation,
    }


def adaptive_profit_protection(prices: pd.DataFrame, *, short_window: int = 20, long_window: int = 60) -> dict[str, Any]:
    close = pd.to_numeric(prices.get("close"), errors="coerce").dropna()
    if len(close) < max(10, short_window):
        return {"status": "insufficient_data", "short_window": short_window, "long_window": long_window}
    short = close.tail(short_window)
    long = close.tail(min(len(close), long_window))
    latest = float(close.iloc[-1])
    short_range_pct = None if latest == 0 else float((short.max() - short.min()) / latest)
    long_range_pct = None if latest == 0 else float((long.max() - long.min()) / latest)
    short_return = _series_change(close, min(short_window - 1, len(close) - 2))
    returns = close.pct_change().dropna()
    short_vol = None if len(returns) < short_window else float(returns.tail(short_window).std())
    long_vol = None if len(returns) < long_window else float(returns.tail(long_window).std())
    range_ratio = None
    if short_range_pct is not None and long_range_pct not in {None, 0}:
        range_ratio = float(short_range_pct / long_range_pct)
    if long_range_pct is not None and long_range_pct >= 0.025 and short_range_pct is not None and short_range_pct >= 0.008:
        regime = "deep_swing_directional"
        instruction = "Recent price behavior has deeper swings. Profit protection may be more permissive, but only while the directional thesis is confirmed by trend, volume flow, StochRSI timing, bid depth, and option spread."
    elif short_range_pct is not None and short_range_pct < 0.0035:
        regime = "narrow_chop"
        instruction = "Current price range is tight; option profits can disappear quickly, so protect small open profits earlier."
    elif range_ratio is not None and range_ratio > 0.75 and short_return is not None and abs(short_return) > 0.004:
        regime = "active_directional"
        instruction = "Recent range is active and directional; allow some room only while price action, volume flow, and option bid support the thesis."
    elif short_vol is not None and long_vol is not None and short_vol > long_vol * 1.4:
        regime = "expanding_volatility"
        instruction = "Volatility is expanding; protect profits with urgency because reversals and spread moves can be abrupt."
    else:
        regime = "balanced"
        instruction = "Use normal profit protection; do not let prior open profit turn negative unless the thesis clearly renewed."
    return {
        "status": "ok",
        "short_window": short_window,
        "long_window": long_window,
        "regime": regime,
        "short_range_pct": short_range_pct,
        "long_range_pct": long_range_pct,
        "range_ratio": range_ratio,
        "short_return_pct": short_return,
        "short_realized_vol": short_vol,
        "long_realized_vol": long_vol,
        "instruction": instruction,
        "policy": "Adaptive context only. Profit protection should respond to current range, volatility, spread, bid depth, theta, and position peak P/L rather than fixed dollar targets.",
    }


def trend_carry_context(prices: pd.DataFrame, *, window: int = 45, pullback_window: int = 9) -> dict[str, Any]:
    close = pd.to_numeric(prices.get("close"), errors="coerce").dropna()
    if len(close) < max(window, 30):
        return {"status": "insufficient_data", "window": window, "pullback_window": pullback_window}
    recent = close.tail(window)
    latest = float(recent.iloc[-1])
    if latest == 0:
        return {"status": "invalid_latest_price", "window": window, "pullback_window": pullback_window}
    sma9 = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()
    recent_sma9 = sma9.tail(window)
    recent_sma21 = sma21.tail(window)
    returns = recent.pct_change().dropna()
    net_return = float(recent.iloc[-1] / recent.iloc[0] - 1.0)
    positive_bar_ratio = float((returns > 0).mean()) if len(returns) else None
    negative_bar_ratio = float((returns < 0).mean()) if len(returns) else None
    monotonicity = None
    if len(returns):
        monotonicity = float(abs(returns.sum()) / returns.abs().sum()) if float(returns.abs().sum()) else 0.0
    rolling_peak = recent.cummax()
    rolling_trough = recent.cummin()
    max_drawdown_from_peak = float(((recent / rolling_peak) - 1.0).min())
    max_bounce_from_trough = float(((recent / rolling_trough) - 1.0).max())
    recent_high = float(recent.max())
    recent_low = float(recent.min())
    distance_from_recent_high_pct = float(latest / recent_high - 1.0) if recent_high else None
    distance_from_recent_low_pct = float(latest / recent_low - 1.0) if recent_low else None
    lower_highs = _swing_lower_high_count(recent)
    higher_lows = _swing_higher_low_count(recent)
    sma_signal = _ma_crossover_signal(sma9, sma21)
    latest_sma9 = _series_latest(sma9)
    latest_sma21 = _series_latest(sma21)
    sma9_slope = _series_change(recent_sma9.dropna(), min(8, max(1, len(recent_sma9.dropna()) - 1)))
    sma21_slope = _series_change(recent_sma21.dropna(), min(8, max(1, len(recent_sma21.dropna()) - 1)))
    pullback = close.tail(max(3, pullback_window))
    failed_reclaim_sma9 = False
    failed_reclaim_sma21 = False
    failed_hold_sma9 = False
    failed_hold_sma21 = False
    if latest_sma9 is not None and len(pullback):
        failed_reclaim_sma9 = bool(float(pullback.max()) >= latest_sma9 and latest < latest_sma9)
        failed_hold_sma9 = bool(float(pullback.min()) <= latest_sma9 and latest > latest_sma9)
    if latest_sma21 is not None and len(pullback):
        failed_reclaim_sma21 = bool(float(pullback.max()) >= latest_sma21 and latest < latest_sma21)
        failed_hold_sma21 = bool(float(pullback.min()) <= latest_sma21 and latest > latest_sma21)
    state = "no_clear_trend_carry"
    reason_codes: list[str] = []
    if net_return <= -0.006:
        reason_codes.append("net_down_window")
    if net_return >= 0.006:
        reason_codes.append("net_up_window")
    if monotonicity is not None and monotonicity >= 0.35:
        reason_codes.append("smooth_directional_drift")
    if negative_bar_ratio is not None and negative_bar_ratio >= 0.52:
        reason_codes.append("negative_bar_majority")
    if positive_bar_ratio is not None and positive_bar_ratio >= 0.52:
        reason_codes.append("positive_bar_majority")
    if lower_highs >= 2:
        reason_codes.append("lower_highs")
    if higher_lows >= 2:
        reason_codes.append("higher_lows")
    if sma_signal.get("signal") in {"death_cross", "short_below_long"}:
        reason_codes.append("sma9_below_sma21")
    if sma_signal.get("signal") in {"golden_cross", "short_above_long"}:
        reason_codes.append("sma9_above_sma21")
    if failed_reclaim_sma9 or failed_reclaim_sma21:
        reason_codes.append("pullback_failed_to_reclaim_ma")
    if failed_hold_sma9 or failed_hold_sma21:
        reason_codes.append("pullback_held_ma_support")
    if distance_from_recent_high_pct is not None and distance_from_recent_high_pct <= -0.003 and ("sma9_below_sma21" in reason_codes or failed_reclaim_sma9 or failed_reclaim_sma21):
        reason_codes.append("rejected_recent_high")
    if distance_from_recent_low_pct is not None and distance_from_recent_low_pct >= 0.003 and ("sma9_above_sma21" in reason_codes or failed_hold_sma9 or failed_hold_sma21):
        reason_codes.append("rejected_recent_low")
    down_score = sum(
        1
        for code in (
            "net_down_window",
            "smooth_directional_drift",
            "negative_bar_majority",
            "lower_highs",
            "sma9_below_sma21",
            "pullback_failed_to_reclaim_ma",
            "rejected_recent_high",
        )
        if code in reason_codes
    )
    up_score = sum(
        1
        for code in (
            "net_up_window",
            "smooth_directional_drift",
            "positive_bar_majority",
            "higher_lows",
            "sma9_above_sma21",
            "pullback_held_ma_support",
            "rejected_recent_low",
        )
        if code in reason_codes
    )
    if down_score >= 4:
        state = "trend_carry_down"
        instruction = "Smooth net downside pressure is present. Consider a put trend-carry setup before a sharp breakdown only if the move is not already exhausted and put spread/theta/liquidity are acceptable."
    elif up_score >= 4:
        state = "trend_carry_up"
        instruction = "Smooth net upside pressure is present. Consider a call trend-carry setup before a sharp breakout only if the move is not already exhausted and call spread/theta/liquidity are acceptable."
    elif down_score >= 3 and "rejected_recent_high" in reason_codes:
        state = "early_trend_carry_down"
        instruction = "A recent high/rejection is rolling into bearish MA structure. Consider a small exploratory put before support breaks if option tradeability is good and the entry is still far enough from support to pay spread/theta."
    elif up_score >= 3 and "rejected_recent_low" in reason_codes:
        state = "early_trend_carry_up"
        instruction = "A recent low/rejection is rolling into bullish MA structure. Consider a small exploratory call before resistance breaks if option tradeability is good and the entry is still far enough from resistance to pay spread/theta."
    else:
        instruction = "No strong trend-carry setup. Prefer scalp, mean-reversion, or hold logic unless evidence improves."
    return {
        "status": "ok",
        "state": state,
        "window": window,
        "pullback_window": pullback_window,
        "net_return_pct": net_return,
        "positive_bar_ratio": positive_bar_ratio,
        "negative_bar_ratio": negative_bar_ratio,
        "monotonicity_score": monotonicity,
        "max_drawdown_from_peak_pct": max_drawdown_from_peak,
        "max_bounce_from_trough_pct": max_bounce_from_trough,
        "recent_high": recent_high,
        "recent_low": recent_low,
        "distance_from_recent_high_pct": distance_from_recent_high_pct,
        "distance_from_recent_low_pct": distance_from_recent_low_pct,
        "lower_high_count": lower_highs,
        "higher_low_count": higher_lows,
        "sma9_slope_pct": sma9_slope,
        "sma21_slope_pct": sma21_slope,
        "failed_reclaim_sma9": failed_reclaim_sma9,
        "failed_reclaim_sma21": failed_reclaim_sma21,
        "failed_hold_sma9": failed_hold_sma9,
        "failed_hold_sma21": failed_hold_sma21,
        "reason_codes": reason_codes,
        "instruction": instruction,
    }


def multi_window_trend_carry_context(prices: pd.DataFrame) -> dict[str, Any]:
    windows = [30, 60, 120, 180]
    contexts = {str(window): trend_carry_context(prices, window=window, pullback_window=9) for window in windows}
    usable = [item for item in contexts.values() if item.get("status") == "ok"]
    states = [str(item.get("state") or "") for item in usable]
    up_states = {"early_trend_carry_up", "trend_carry_up"}
    down_states = {"early_trend_carry_down", "trend_carry_down"}
    up_count = sum(1 for state in states if state in up_states)
    down_count = sum(1 for state in states if state in down_states)
    long_bias = "none"
    if up_count >= 2:
        long_bias = "multi_window_up"
        instruction = "Multiple windows show upward carry. A call may be justified on pullbacks or early continuation, not only after an obvious breakout."
    elif down_count >= 2:
        long_bias = "multi_window_down"
        instruction = "Multiple windows show downward carry. A put may be justified on failed bounces or early continuation, not only after an obvious breakdown."
    elif any(state in up_states for state in states):
        long_bias = "single_window_up"
        instruction = "One window shows upward carry. Treat it as a potential early call clue, not a standalone trade command."
    elif any(state in down_states for state in states):
        long_bias = "single_window_down"
        instruction = "One window shows downward carry. Treat it as a potential early put clue, not a standalone trade command."
    else:
        instruction = "No carry state is confirmed across windows."
    return {
        "status": "ok" if usable else "insufficient_data",
        "windows": contexts,
        "bias": long_bias,
        "up_state_count": up_count,
        "down_state_count": down_count,
        "instruction": instruction,
    }


def technical_observations(prices: pd.DataFrame) -> dict[str, Any]:
    close = prices["close"].dropna()
    if close.empty:
        return {}
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    sma9 = close.rolling(9).mean()
    sma21 = close.rolling(21).mean()
    rolling_high = close.rolling(60, min_periods=5).max()
    rolling_low = close.rolling(60, min_periods=5).min()
    latest = float(close.iloc[-1])
    support = float(rolling_low.iloc[-1]) if not pd.isna(rolling_low.iloc[-1]) else None
    resistance = float(rolling_high.iloc[-1]) if not pd.isna(rolling_high.iloc[-1]) else None
    stoch_rsi = stochastic_rsi(close)
    ad_line = accumulation_distribution(prices)
    adaptive_profit = adaptive_profit_protection(prices)
    carry = trend_carry_context(prices)
    multi_carry = multi_window_trend_carry_context(prices)
    return {
        "latest_close": latest,
        "ema_9": None if pd.isna(fast.iloc[-1]) else float(fast.iloc[-1]),
        "ema_21": None if pd.isna(slow.iloc[-1]) else float(slow.iloc[-1]),
        "ema_9_minus_21_pct": None if pd.isna(fast.iloc[-1]) or pd.isna(slow.iloc[-1]) or slow.iloc[-1] == 0 else float(fast.iloc[-1] / slow.iloc[-1] - 1.0),
        "sma_9": _series_latest(sma9),
        "sma_21": _series_latest(sma21),
        "sma_9_21_crossover": _ma_crossover_signal(sma9, sma21),
        "rolling_60_bar_support": support,
        "rolling_60_bar_resistance": resistance,
        "distance_to_support_pct": None if support in {None, 0} else float(latest / support - 1.0),
        "distance_to_resistance_pct": None if resistance in {None, 0} else float(latest / resistance - 1.0),
        "stochastic_rsi": stoch_rsi,
        "accumulation_distribution": ad_line,
        "adaptive_profit_protection": adaptive_profit,
        "trend_carry_context": carry,
        "multi_window_trend_carry": multi_carry,
        "note": "These are descriptive observations only, not a forecast or trade recommendation.",
    }


def _swing_lower_high_count(series: pd.Series, *, lookback: int = 5) -> int:
    clean = series.dropna().astype(float)
    if len(clean) < lookback * 2:
        return 0
    highs = clean.rolling(lookback, min_periods=lookback).max().dropna()
    if len(highs) < 3:
        return 0
    tail = highs.tail(6)
    return int(sum(float(tail.iloc[index]) < float(tail.iloc[index - 1]) for index in range(1, len(tail))))


def _swing_higher_low_count(series: pd.Series, *, lookback: int = 5) -> int:
    clean = series.dropna().astype(float)
    if len(clean) < lookback * 2:
        return 0
    lows = clean.rolling(lookback, min_periods=lookback).min().dropna()
    if len(lows) < 3:
        return 0
    tail = lows.tail(6)
    return int(sum(float(tail.iloc[index]) > float(tail.iloc[index - 1]) for index in range(1, len(tail))))


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)
