from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.data import normalize_price_frame
from market_forecasting_engine.data_providers import DataRequest, load_prices_with_provider
from market_forecasting_engine.llm_options_trader.common import (
    apply_forecast_validation_to_transition,
    option_tradeability_summary,
    price_summary,
    recent_price_bars,
    regime_transition_warning,
    short_tape_summary,
    technical_observations,
)
from market_forecasting_engine.llm_options_trader.knowledge_base import load_strategy_knowledge


@dataclass(frozen=True)
class AlpacaLLMOptionsRuntimeConfig:
    ticker: str = "NVDA"
    data_provider: str = "alpaca"
    data_interval: str = "1m"
    data_feed: str = "iex"
    lookback_days: int = 20
    max_price_rows: int = 3500
    forecast_hours: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    option_chain_limit: int = 80
    min_dte: int = 1
    max_dte: int = 14
    max_order_qty: int = 1
    max_order_price: float = 25.0
    max_order_debit: float = 1500.0
    max_crypto_notional: float = 100.0


def build_alpaca_market_packet(
    *,
    broker: AlpacaPaperBroker,
    config: AlpacaLLMOptionsRuntimeConfig,
    process: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    prices = load_alpaca_underlying_prices(config=config, now=now)
    latest_price = _latest_close(prices)
    asset_class = alpaca_asset_class(config.ticker)
    account = broker.account()
    market_clock = _safe_clock(broker)
    open_orders = broker.orders(status="open", limit=100)
    positions = broker.positions()
    recent_trades = _recent_closed_fills(broker=broker, ticker=config.ticker, asset_class=asset_class, limit=100)
    chain = (
        alpaca_option_chain_snapshot(broker=broker, config=config, underlying_price=latest_price, now=now)
        if asset_class == "equity_option"
        else []
    )
    transition = regime_transition_warning(prices, forecast_validation={})
    return {
        "packet_version": "alpaca_llm_options_trader_v1",
        "process": process,
        "generated_at_utc": now.isoformat(),
        "venue": "alpaca",
        "account_mode": "paper",
        "asset_class": asset_class,
        "currency": "USD",
        "instrument_currency": "USD",
        "underlying_ticker": config.ticker.upper(),
        "price_data_provider": config.data_provider,
        "price_data_interval": config.data_interval,
        "price_data_feed": config.data_feed,
        "latest_underlying_price": latest_price,
        "spot_instrument": _spot_instrument(config=config, asset_class=asset_class),
        "market_clock": market_clock,
        "market_is_open": bool(market_clock.get("is_open")) if asset_class == "equity_option" else True,
        "market_policy": _market_policy(asset_class=asset_class, market_clock=market_clock),
        "account": _compact_alpaca_account(account),
        "open_option_orders": _filter_alpaca_orders(open_orders, ticker=config.ticker, asset_class=asset_class),
        "option_positions": _filter_alpaca_positions(positions, ticker=config.ticker, asset_class=asset_class),
        "recent_option_trades": recent_trades,
        "price_summary": price_summary(prices),
        "technical_observations": technical_observations(prices),
        "short_tape_summary": short_tape_summary(prices),
        "regime_transition_warning": transition,
        "option_tradeability_summary": option_tradeability_summary(chain),
        "recent_price_bars": recent_price_bars(prices, rows=3000),
        "strategy_knowledge": load_strategy_knowledge(),
        "strategy_memory": {},
        "option_chain": chain,
        "api_contract": {
            "execution": "Python is only the Alpaca API interface. The LLM decision is final inside the configured shadow experiment envelope.",
            "forecast_policy": "No precomputed directional forecast is supplied. The LLM must infer direction, timing, confidence, and order parameters from supplied market data.",
            "entry_order_requirements": {
                "action": "hold or submit_order",
                "order.side": "buy for new entries",
                "order.type": "limit",
                "order.qty": "positive whole-contract integer for equity options; positive decimal amount for crypto spot",
                "order.symbol": _allowed_symbol_instruction(config=config, asset_class=asset_class),
            },
            "exit_order_requirements": {
                "action": "hold, cancel_order, or submit_order",
                "order.side": "sell",
                "order.type": "limit",
                "order.symbol": "must be an existing shadow position symbol when closing",
            },
            "mechanical_limits": {
                "max_order_qty": config.max_order_qty,
                "max_order_price": config.max_order_price,
                "max_order_debit": config.max_order_debit,
                "max_crypto_notional": config.max_crypto_notional,
            },
            "asset_policy": _asset_policy(asset_class=asset_class),
            "paper_safety": "Default Alpaca command is simulation-only shadow trading; no paper order is sent unless a future explicit execution mode is implemented.",
        },
        "trader_memory": {},
    }


def compact_alpaca_market_packet(packet: dict[str, Any], *, max_contracts_per_side: int = 8, max_price_bars: int = 45, max_trades: int = 12) -> dict[str, Any]:
    chain = packet.get("option_chain") if isinstance(packet.get("option_chain"), list) else []
    calls = [row for row in chain if str(row.get("option_type") or "").lower() == "call"]
    puts = [row for row in chain if str(row.get("option_type") or "").lower() == "put"]
    selected_chain = calls[: max(1, int(max_contracts_per_side))] + puts[: max(1, int(max_contracts_per_side))]
    return {
        "packet_version": "alpaca_llm_options_trader_compact_v1",
        "process": packet.get("process"),
        "generated_at_utc": packet.get("generated_at_utc"),
        "venue": packet.get("venue"),
        "asset_class": packet.get("asset_class"),
        "account_mode": packet.get("account_mode"),
        "execution_mode": packet.get("execution_mode"),
        "currency": packet.get("currency"),
        "instrument_currency": packet.get("instrument_currency"),
        "underlying_ticker": packet.get("underlying_ticker"),
        "price_data_provider": packet.get("price_data_provider"),
        "price_data_interval": packet.get("price_data_interval"),
        "price_data_feed": packet.get("price_data_feed"),
        "latest_underlying_price": packet.get("latest_underlying_price"),
        "spot_instrument": packet.get("spot_instrument") or {},
        "market_clock": packet.get("market_clock") or {},
        "market_is_open": packet.get("market_is_open"),
        "market_policy": packet.get("market_policy") or {},
        "account": packet.get("account") or {},
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
        "recent_price_bars": (packet.get("recent_price_bars") or [])[-max(2, int(max_price_bars)) :],
        "option_chain": selected_chain,
        "api_contract": packet.get("api_contract") or {},
        "trader_memory": packet.get("trader_memory") or {},
    }


def load_alpaca_underlying_prices(*, config: AlpacaLLMOptionsRuntimeConfig, now: datetime) -> pd.DataFrame:
    start = (now - timedelta(days=int(config.lookback_days))).isoformat().replace("+00:00", "Z")
    result = load_prices_with_provider(
        config.data_provider,
        DataRequest(ticker=config.ticker.upper(), start=start, interval=config.data_interval, target_column="close"),
        store=None,
        use_cache=False,
        refresh_cache=True,
    )
    prices = normalize_price_frame(result.frame, target_column="close")
    if config.max_price_rows and len(prices) > int(config.max_price_rows):
        prices = prices.tail(int(config.max_price_rows))
    return prices


def alpaca_option_chain_snapshot(
    *,
    broker: AlpacaPaperBroker,
    config: AlpacaLLMOptionsRuntimeConfig,
    underlying_price: float,
    now: datetime,
) -> list[dict[str, Any]]:
    start = (now.date() + timedelta(days=max(1, int(config.min_dte)))).isoformat()
    end = (now.date() + timedelta(days=max(int(config.min_dte), int(config.max_dte)))).isoformat()
    contracts: list[dict[str, Any]] = []
    for option_type in ("call", "put"):
        contracts.extend(
            broker.option_contracts(
                underlying_symbols=config.ticker.upper(),
                expiration_date_gte=start,
                expiration_date_lte=end,
                option_type=option_type,
                limit=1000,
            )
        )
    ranked = []
    for contract in contracts:
        strike = _float_or_none(contract.get("strike_price") or contract.get("strike"))
        if strike is None:
            continue
        expiry = _parse_date(contract.get("expiration_date"))
        dte = None if expiry is None else max(0, (expiry - now.date()).days)
        ranked.append({"contract": contract, "distance": abs(strike - underlying_price), "dte": dte or 999})
    ranked.sort(key=lambda row: (row["distance"], row["dte"]))
    selected = [row["contract"] for row in ranked[: max(1, int(config.option_chain_limit))]]
    snapshots = broker.option_snapshots([str(contract.get("symbol")) for contract in selected if contract.get("symbol")])
    output: list[dict[str, Any]] = []
    for contract in selected:
        symbol = str(contract.get("symbol") or "")
        snapshot = snapshots.get(symbol) or {}
        quote = _snapshot_quote(snapshot)
        greeks = _snapshot_greeks(snapshot)
        bid = quote.get("bid")
        ask = quote.get("ask")
        mid = None if bid is None or ask is None else (bid + ask) / 2.0
        strike = _float_or_none(contract.get("strike_price") or contract.get("strike"))
        expiry = _parse_date(contract.get("expiration_date"))
        dte = None if expiry is None else max(0, (expiry - now.date()).days)
        spread_pct = None if bid is None or ask is None or not mid else (ask - bid) / max(mid, 1e-12)
        output.append(
            {
                "instrument_name": symbol,
                "symbol": symbol,
                "name": contract.get("name"),
                "option_type": str(contract.get("type") or contract.get("option_type") or "").lower(),
                "strike": strike,
                "expiration_utc": None if expiry is None else expiry.isoformat(),
                "dte": dte,
                "tradable": contract.get("tradable"),
                "status": contract.get("status"),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "mark_price": mid,
                "spread_pct": spread_pct,
                "best_bid_amount": _float_or_none(quote.get("bid_size")),
                "best_ask_amount": _float_or_none(quote.get("ask_size")),
                "open_interest": _float_or_none(contract.get("open_interest")),
                "volume": _snapshot_volume(snapshot),
                "volume_usd": None if _snapshot_volume(snapshot) is None or mid is None else _snapshot_volume(snapshot) * mid * 100.0,
                "greeks": {
                    "delta": _float_or_none(greeks.get("delta")),
                    "gamma": _float_or_none(greeks.get("gamma")),
                    "theta": _float_or_none(greeks.get("theta")),
                    "vega": _float_or_none(greeks.get("vega")),
                    "rho": _float_or_none(greeks.get("rho")),
                },
                "top_bids": [] if bid is None else [[bid, quote.get("bid_size")]],
                "top_asks": [] if ask is None else [[ask, quote.get("ask_size")]],
            }
        )
    output.sort(key=lambda row: (row.get("option_type") != "call", abs((_float_or_none(row.get("strike")) or underlying_price) - underlying_price), row.get("dte") or 999))
    return output


def validate_alpaca_order_payload(order: dict[str, Any], *, config: AlpacaLLMOptionsRuntimeConfig, require_exit: bool) -> dict[str, Any]:
    blocks: list[str] = []
    symbol = str(order.get("symbol") or "").upper()
    asset_class = alpaca_asset_class(config.ticker)
    side = str(order.get("side") or "").lower()
    order_type = str(order.get("type") or "").lower()
    qty = _float_or_none(order.get("qty")) if asset_class == "crypto_spot" else _int_or_none(order.get("qty"))
    limit_price = _float_or_none(order.get("limit_price"))
    if asset_class == "crypto_spot":
        allowed_symbols = {_alpaca_crypto_symbol(config.ticker), _alpaca_crypto_symbol(config.ticker).replace("/", "")}
        if symbol not in allowed_symbols:
            blocks.append("symbol_not_allowed_for_alpaca_crypto_experiment")
    elif not symbol.startswith(config.ticker.upper()):
        blocks.append("symbol_not_allowed_for_alpaca_ticker_experiment")
    if side not in {"buy", "sell"}:
        blocks.append("side_must_be_buy_or_sell")
    if not require_exit and side == "sell":
        blocks.append("new_shadow_entries_must_be_buy_orders")
    if require_exit and side != "sell":
        blocks.append("exit_orders_must_sell_existing_shadow_position")
    if order_type != "limit":
        blocks.append("type_must_be_limit")
    if qty is None or qty <= 0:
        blocks.append("qty_must_be_positive_number" if asset_class == "crypto_spot" else "qty_must_be_positive_whole_contract_integer")
    elif asset_class == "equity_option" and qty > int(config.max_order_qty):
        blocks.append("qty_exceeds_configured_mechanical_cap")
    if limit_price is None or limit_price <= 0:
        blocks.append("limit_price_must_be_positive_number")
    elif limit_price > float(config.max_order_price):
        blocks.append("limit_price_exceeds_configured_mechanical_cap")
    if asset_class == "crypto_spot" and qty is not None and limit_price is not None and qty * limit_price > float(config.max_crypto_notional):
        blocks.append("crypto_notional_exceeds_configured_mechanical_cap")
    if asset_class == "equity_option" and qty is not None and limit_price is not None and qty * limit_price * 100.0 > float(config.max_order_debit):
        blocks.append("debit_exceeds_configured_mechanical_cap")
    if blocks:
        return {"blocks": blocks}
    return {
        "blocks": [],
        "order": {
            "symbol": symbol,
            "side": side,
            "type": "limit",
            "qty": float(qty or 0.0) if asset_class == "crypto_spot" else int(qty or 0),
            "limit_price": float(limit_price or 0.0),
            "time_in_force": str(order.get("time_in_force") or "day"),
            "asset_class": asset_class,
        },
    }


def alpaca_asset_class(ticker: str) -> str:
    return "crypto_spot" if _is_crypto_symbol(ticker) else "equity_option"


def apply_forecast_validation_to_alpaca_transition(transition: dict[str, Any], forecast_validation: dict[str, Any]) -> dict[str, Any]:
    return apply_forecast_validation_to_transition(transition, forecast_validation)


def _compact_alpaca_account(account: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "trading_blocked",
        "account_blocked",
        "options_approved_level",
        "options_trading_level",
        "cash",
        "equity",
        "portfolio_value",
        "buying_power",
        "options_buying_power",
        "pattern_day_trader",
    )
    return {key: account.get(key) for key in keys if key in account}


def _filter_alpaca_orders(items: list[dict[str, Any]], *, ticker: str, asset_class: str) -> list[dict[str, Any]]:
    if asset_class == "crypto_spot":
        symbols = {_alpaca_crypto_symbol(ticker), _alpaca_crypto_symbol(ticker).replace("/", "")}
        return [item for item in items if str(item.get("symbol") or "").upper() in symbols]
    prefix = ticker.upper()
    return [item for item in items if str(item.get("symbol") or "").upper().startswith(prefix)]


def _filter_alpaca_positions(items: list[dict[str, Any]], *, ticker: str, asset_class: str) -> list[dict[str, Any]]:
    output = []
    for item in _filter_alpaca_orders(items, ticker=ticker, asset_class=asset_class):
        qty = _float_or_none(item.get("qty"))
        if qty is not None and abs(qty) > 0:
            output.append(item)
    return output


def _recent_closed_fills(*, broker: AlpacaPaperBroker, ticker: str, asset_class: str, limit: int) -> list[dict[str, Any]]:
    rows = []
    for order in broker.orders(status="closed", limit=limit):
        symbol = str(order.get("symbol") or "").upper()
        if asset_class == "crypto_spot":
            wanted = {_alpaca_crypto_symbol(ticker), _alpaca_crypto_symbol(ticker).replace("/", "")}
            matches = symbol in wanted
        else:
            matches = symbol.startswith(ticker.upper())
        if not matches or order.get("status") != "filled":
            continue
        rows.append(
            {
                "order_id": order.get("id"),
                "filled_at": order.get("filled_at"),
                "symbol": symbol,
                "side": order.get("side"),
                "qty": _float_or_none(order.get("filled_qty") or order.get("qty")),
                "price": _float_or_none(order.get("filled_avg_price")),
                "type": order.get("type"),
            }
        )
    return rows


def _safe_clock(broker: AlpacaPaperBroker) -> dict[str, Any]:
    try:
        payload = broker.clock()
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return payload if isinstance(payload, dict) else {}


def _market_policy(*, asset_class: str, market_clock: dict[str, Any]) -> dict[str, Any]:
    if asset_class == "crypto_spot":
        return {
            "status": "continuous_market",
            "can_open_new_shadow_entries": True,
            "instruction": "Alpaca crypto spot trades continuously. Shadow entries may be evaluated even when the US equity/options market is closed.",
        }
    is_open = bool(market_clock.get("is_open"))
    return {
        "status": "equity_options_market_open" if is_open else "equity_options_market_closed",
        "can_open_new_shadow_entries": is_open,
        "instruction": "For non-crypto Alpaca tickers, do not open new shadow option entries when the US market is closed. Existing shadow positions may still be reviewed for risk, but real options orders cannot execute until the market is open.",
    }


def _asset_policy(*, asset_class: str) -> dict[str, Any]:
    if asset_class == "crypto_spot":
        return {
            "instrument": "crypto_spot",
            "option_chain_available": False,
            "instruction": "Alpaca does not provide crypto options here. For crypto tickers, this shadow experiment uses spot crypto buy/sell limit decisions only.",
        }
    return {
        "instrument": "equity_options",
        "option_chain_available": True,
        "instruction": "For equity tickers, this shadow experiment evaluates listed US equity option contracts from Alpaca snapshots.",
    }


def _allowed_symbol_instruction(*, config: AlpacaLLMOptionsRuntimeConfig, asset_class: str) -> str:
    if asset_class == "crypto_spot":
        return f"must be {_alpaca_crypto_symbol(config.ticker)} for Alpaca crypto spot"
    return f"must be a listed {config.ticker.upper()} option symbol from option_chain"


def _spot_instrument(*, config: AlpacaLLMOptionsRuntimeConfig, asset_class: str) -> dict[str, Any]:
    if asset_class != "crypto_spot":
        return {}
    return {
        "symbol": _alpaca_crypto_symbol(config.ticker),
        "asset_class": "crypto_spot",
        "order_type": "limit",
        "allowed_entry_intent": "open_spot_long",
        "allowed_entry_side": "buy",
        "allowed_exit_side": "sell",
        "qty_policy": "decimal crypto quantity, capped by max_crypto_notional",
    }


def _is_crypto_symbol(ticker: str) -> bool:
    normalized = ticker.upper().replace("_", "-").replace("/", "-")
    if "-" not in normalized:
        return normalized in {"BTCUSD", "ETHUSD", "LTCUSD", "DOGEUSD", "SOLUSD", "BTCUSDC", "ETHUSDC"}
    base, quote = normalized.split("-", 1)
    return bool(base) and quote in {"USD", "USDT", "USDC", "BTC", "ETH"}


def _alpaca_crypto_symbol(ticker: str) -> str:
    normalized = ticker.upper().replace("_", "-").replace("/", "-")
    if "-" in normalized:
        base, quote = normalized.split("-", 1)
        return f"{base}/{quote}"
    if normalized.endswith("USDT"):
        return f"{normalized[:-4]}/USDT"
    if normalized.endswith("USDC"):
        return f"{normalized[:-4]}/USDC"
    if normalized.endswith("USD") and len(normalized) > 3:
        return f"{normalized[:-3]}/USD"
    return normalized.replace("-", "/")


def _latest_close(prices: pd.DataFrame) -> float:
    close = prices["close"].dropna()
    if close.empty:
        raise ValueError("No close prices available for Alpaca LLM options trader.")
    return float(close.iloc[-1])


def _snapshot_quote(snapshot: dict[str, Any]) -> dict[str, float | None]:
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or snapshot.get("quote") or {}
    return {
        "bid": _float_or_none(quote.get("bp") or quote.get("bid_price") or quote.get("bid")),
        "ask": _float_or_none(quote.get("ap") or quote.get("ask_price") or quote.get("ask")),
        "bid_size": _float_or_none(quote.get("bs") or quote.get("bid_size")),
        "ask_size": _float_or_none(quote.get("as") or quote.get("ask_size")),
    }


def _snapshot_greeks(snapshot: dict[str, Any]) -> dict[str, Any]:
    return snapshot.get("greeks") or snapshot.get("latestGreeks") or snapshot.get("latest_greeks") or {}


def _snapshot_volume(snapshot: dict[str, Any]) -> float | None:
    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or {}
    return _float_or_none(snapshot.get("volume") or trade.get("s") or trade.get("size"))


def _parse_date(value: Any) -> Any:
    if not value:
        return None
    try:
        return pd.Timestamp(value).date()
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed
