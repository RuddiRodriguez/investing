from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker
from market_forecasting_engine.llm_options_trader.shadow_execution import (
    apply_order_observation,
    evaluate_limit_order_fill,
    normalize_recent_trades,
)

OPTION_CONTRACT_MULTIPLIER = 100.0


def load_and_update_alpaca_shadow_state(*, output_dir: Path, ticker: str, broker: AlpacaPaperBroker) -> dict[str, Any]:
    state = _load_state(output_dir, ticker)
    changed = _update_open_orders(state=state, broker=broker)
    _mark_positions(state=state, broker=broker)
    if changed:
        _save_state(output_dir, ticker, state)
    return _summary(state)


def record_simulated_alpaca_order(
    *,
    output_dir: Path,
    ticker: str,
    broker: AlpacaPaperBroker,
    validated_order: dict[str, Any],
    decision: dict[str, Any],
    checked_at_utc: str,
) -> dict[str, Any]:
    state = _load_state(output_dir, ticker)
    order_id = f"alpaca-shadow-{int(time.time() * 1000)}"
    order = {
        "order_id": order_id,
        "created_at_utc": checked_at_utc,
        "updated_at_utc": checked_at_utc,
        "status": "open",
        "source": "alpaca_llm_live_shadow",
        "decision_reason": decision.get("reason"),
        **validated_order,
    }
    state.setdefault("orders", []).append(order)
    _update_open_orders(state=state, broker=broker)
    _mark_positions(state=state, broker=broker)
    _save_state(output_dir, ticker, state)
    return {"shadow_order_id": order_id, "shadow_state": _summary(state)}


def record_simulated_alpaca_cancels(*, output_dir: Path, ticker: str, order_ids: list[str], checked_at_utc: str) -> dict[str, Any]:
    state = _load_state(output_dir, ticker)
    targets = {str(order_id or "").strip() for order_id in order_ids if str(order_id or "").strip()}
    matched_ids: list[str] = []
    for order in state.get("orders", []):
        if str(order.get("order_id")) in targets and order.get("status") == "open":
            order["status"] = "canceled"
            order["updated_at_utc"] = checked_at_utc
            matched_ids.append(str(order.get("order_id")))
    _save_state(output_dir, ticker, state)
    return {
        "shadow_cancel_matched": bool(matched_ids),
        "shadow_cancel_matched_ids": matched_ids,
        "shadow_cancel_requested_ids": sorted(targets),
        "shadow_state": _summary(state),
    }


def _update_open_orders(*, state: dict[str, Any], broker: AlpacaPaperBroker) -> bool:
    changed = False
    for order in state.get("orders", []):
        if order.get("status") != "open":
            continue
        symbol = str(order.get("symbol") or "")
        quote = _safe_option_quote(broker, symbol)
        side = str(order.get("side") or "").lower()
        limit = _float_or_none(order.get("limit_price"))
        if limit is None:
            continue
        evaluation = evaluate_limit_order_fill(
            order=order,
            quote=quote,
            recent_trades=_safe_latest_trades(broker, symbol),
            side=side,
            limit_price=limit,
        )
        if not evaluation.get("filled"):
            apply_order_observation(order, evaluation)
            continue
        _fill_order(
            state=state,
            order=order,
            fill_price=float(evaluation["fill_price"]),
            mark_price=evaluation.get("mark_price"),
            fill_reason=str(evaluation.get("fill_reason") or "unknown"),
            reference_trade=evaluation.get("reference_trade"),
        )
        changed = True
    return changed


def _fill_order(
    *,
    state: dict[str, Any],
    order: dict[str, Any],
    fill_price: float,
    mark_price: float | None,
    fill_reason: str = "unknown",
    reference_trade: dict[str, Any] | None = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    order["status"] = "filled"
    order["filled_at_utc"] = now
    order["updated_at_utc"] = now
    order["fill_price"] = fill_price
    order["fill_reason"] = fill_reason
    order["reference_trade"] = reference_trade
    qty = float(order.get("qty") or 0.0)
    signed_qty = qty if str(order.get("side")).lower() == "buy" else -qty
    symbol = str(order.get("symbol") or "")
    multiplier = _multiplier(order)
    trade = {
        "trade_id": f"alpaca-shadow-trade-{len(state.get('trades', [])) + 1}",
        "timestamp_utc": now,
        "symbol": symbol,
        "instrument_name": symbol,
        "side": order.get("side"),
        "qty": qty,
        "amount": qty,
        "price": fill_price,
        "notional": round(qty * fill_price * multiplier, 2),
        "source_order_id": order.get("order_id"),
        "fill_reason": fill_reason,
        "reference_trade": reference_trade,
    }
    state.setdefault("trades", []).append(trade)
    positions = state.setdefault("positions", {})
    pos = positions.setdefault(
        symbol,
        {
            "symbol": symbol,
            "instrument_name": symbol,
            "size": 0.0,
            "qty": 0.0,
            "average_price": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "peak_unrealized_pnl": 0.0,
            "profit_giveback": 0.0,
            "profit_giveback_pct": None,
            "mark_price": mark_price,
            "asset_class": order.get("asset_class"),
        },
    )
    old_size = float(pos.get("size") or 0.0)
    old_avg = float(pos.get("average_price") or 0.0)
    if signed_qty > 0:
        new_size = old_size + signed_qty
        pos["average_price"] = ((old_size * old_avg) + (signed_qty * fill_price)) / new_size if new_size else 0.0
        pos["size"] = new_size
        pos["qty"] = new_size
    else:
        close_qty = min(abs(float(signed_qty)), max(0.0, old_size))
        close_realized_pnl = close_qty * (fill_price - old_avg) * multiplier
        peak_before_close = float(pos.get("peak_unrealized_pnl") or 0.0)
        trade["close_realized_pnl"] = round(close_realized_pnl, 2)
        trade["peak_unrealized_pnl_before_close"] = round(peak_before_close, 2)
        trade["unrealized_pnl_before_close"] = pos.get("unrealized_pnl")
        trade["profit_giveback_before_close"] = pos.get("profit_giveback")
        trade["profit_giveback_pct_before_close"] = pos.get("profit_giveback_pct")
        trade["profit_protection_outcome"] = _profit_protection_outcome(peak_before_close=peak_before_close, close_realized_pnl=close_realized_pnl)
        pos["realized_pnl"] = float(pos.get("realized_pnl") or 0.0) + close_realized_pnl
        pos["size"] = old_size - close_qty
        pos["qty"] = pos["size"]
        if pos["size"] <= 1e-12:
            pos["size"] = 0.0
            pos["qty"] = 0.0
            pos["peak_unrealized_pnl"] = 0.0
            pos["profit_giveback"] = 0.0
            pos["profit_giveback_pct"] = None
    pos["mark_price"] = mark_price
    pos["updated_at_utc"] = now


def _mark_positions(*, state: dict[str, Any], broker: AlpacaPaperBroker) -> None:
    for pos in state.get("positions", {}).values():
        symbol = str(pos.get("symbol") or pos.get("instrument_name") or "")
        size = float(pos.get("size") or 0.0)
        if size <= 0:
            pos["unrealized_pnl"] = 0.0
            pos["profit_giveback"] = 0.0
            pos["profit_giveback_pct"] = None
            continue
        quote = _safe_option_quote(broker, symbol)
        bid = _float_or_none(quote.get("bid"))
        ask = _float_or_none(quote.get("ask"))
        mark = None if bid is None or ask is None else (bid + ask) / 2.0
        if mark is None:
            mark = bid
        avg = float(pos.get("average_price") or 0.0)
        pos["mark_price"] = mark
        pos["last_bid"] = bid
        pos["last_ask"] = ask
        multiplier = _position_multiplier(pos)
        current_pnl = None if mark is None else round(size * (mark - avg) * multiplier, 2)
        pos["unrealized_pnl"] = current_pnl
        if current_pnl is not None:
            peak = max(float(pos.get("peak_unrealized_pnl") or 0.0), float(current_pnl))
            giveback = max(0.0, peak - float(current_pnl))
            pos["peak_unrealized_pnl"] = round(peak, 2)
            pos["profit_giveback"] = round(giveback, 2)
            pos["profit_giveback_pct"] = None if peak <= 0 else round(giveback / peak, 6)
            pos["profit_to_loss_guard_triggered"] = bool(peak > 0 and float(current_pnl) <= 0)
            pos["profit_protection_instruction"] = _profit_protection_instruction(
                current_pnl=float(current_pnl),
                peak=peak,
                giveback_pct=pos["profit_giveback_pct"],
            )
        pos["updated_at_utc"] = datetime.now(UTC).isoformat()


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    orders = state.get("orders", [])
    positions = list((state.get("positions") or {}).values())
    open_orders = [order for order in orders if order.get("status") == "open"]
    open_positions = [pos for pos in positions if float(pos.get("size") or 0.0) > 0]
    realized = sum(float(pos.get("realized_pnl") or 0.0) for pos in positions)
    unrealized = sum(float(pos.get("unrealized_pnl") or 0.0) for pos in open_positions)
    protection_audit = _profit_protection_audit(state.get("trades") or [])
    return {
        "mode": "live_shadow_simulation",
        "orders": orders[-50:],
        "open_orders": open_orders,
        "positions": open_positions,
        "trades": (state.get("trades") or [])[-50:],
        "order_count": len(orders),
        "open_order_count": len(open_orders),
        "position_count": len(open_positions),
        "realized_pnl": round(realized, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_pnl": round(realized + unrealized, 2),
        "profit_protection_audit": protection_audit,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }


def _safe_option_quote(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any]:
    if "/" in symbol or symbol.endswith("USD"):
        return _safe_crypto_quote(broker, symbol)
    try:
        snapshot = (broker.option_snapshots([symbol]) or {}).get(symbol) or {}
    except Exception:
        return {}
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or snapshot.get("quote") or {}
    return {
        "bid": _float_or_none(quote.get("bp") or quote.get("bid_price") or quote.get("bid")),
        "ask": _float_or_none(quote.get("ap") or quote.get("ask_price") or quote.get("ask")),
    }


def _safe_crypto_quote(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any]:
    normalized = symbol.upper()
    if "/" not in normalized and normalized.endswith("USD"):
        normalized = f"{normalized[:-3]}/USD"
    try:
        snapshot = (broker.crypto_snapshots([normalized]) or {}).get(normalized) or {}
    except Exception:
        return {}
    quote = snapshot.get("latestQuote") or snapshot.get("latest_quote") or snapshot.get("quote") or {}
    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or snapshot.get("trade") or {}
    bid = _float_or_none(quote.get("bp") or quote.get("bid_price") or quote.get("bid"))
    ask = _float_or_none(quote.get("ap") or quote.get("ask_price") or quote.get("ask"))
    trade_price = _float_or_none(trade.get("p") or trade.get("price"))
    if bid is None and trade_price is not None:
        bid = trade_price
    if ask is None and trade_price is not None:
        ask = trade_price
    return {"bid": bid, "ask": ask}


def _safe_latest_trades(broker: AlpacaPaperBroker, symbol: str) -> list[dict[str, Any]]:
    if not symbol:
        return []
    try:
        snapshot = _safe_crypto_snapshot(broker, symbol) if "/" in symbol or symbol.endswith("USD") else _safe_option_snapshot(broker, symbol)
    except Exception:
        return []
    trade = snapshot.get("latestTrade") or snapshot.get("latest_trade") or snapshot.get("trade") or {}
    if not isinstance(trade, dict) or not trade:
        return []
    return normalize_recent_trades([trade])


def _safe_option_snapshot(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any]:
    try:
        return (broker.option_snapshots([symbol]) or {}).get(symbol) or {}
    except Exception:
        return {}


def _safe_crypto_snapshot(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any]:
    normalized = symbol.upper()
    if "/" not in normalized and normalized.endswith("USD"):
        normalized = f"{normalized[:-3]}/USD"
    try:
        return (broker.crypto_snapshots([normalized]) or {}).get(normalized) or {}
    except Exception:
        return {}


def _multiplier(order: dict[str, Any]) -> float:
    return 1.0 if str(order.get("asset_class") or "") == "crypto_spot" else OPTION_CONTRACT_MULTIPLIER


def _position_multiplier(position: dict[str, Any]) -> float:
    return 1.0 if str(position.get("asset_class") or "") == "crypto_spot" else OPTION_CONTRACT_MULTIPLIER


def _load_state(output_dir: Path, ticker: str) -> dict[str, Any]:
    path = _state_path(output_dir, ticker)
    if not path.exists():
        return {"orders": [], "positions": {}, "trades": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"orders": [], "positions": {}, "trades": []}
    return payload if isinstance(payload, dict) else {"orders": [], "positions": {}, "trades": []}


def _save_state(output_dir: Path, ticker: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _state_path(output_dir: Path, ticker: str) -> Path:
    return output_dir / "shadow" / f"{ticker.upper()}_alpaca_shadow_ledger.json"


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _profit_protection_instruction(*, current_pnl: float, peak: float, giveback_pct: float | None) -> str:
    if peak <= 0:
        return "No prior open profit has been recorded for this position."
    if current_pnl <= 0:
        return "Prior open profit has been lost or turned negative; exit should be prioritized unless there is a very explicit renewed thesis."
    if giveback_pct is not None and giveback_pct > 0:
        return "Open profit is being given back. Use adaptive_profit_protection plus spread, bid depth, theta, and current thesis to decide whether to close before profit turns negative."
    return "Position is at or near peak open profit. Use adaptive_profit_protection to decide whether to harvest, hold, or tighten exit."


def _profit_protection_outcome(*, peak_before_close: float, close_realized_pnl: float) -> str:
    if peak_before_close <= 0:
        return "no_prior_open_profit"
    if close_realized_pnl > 0:
        return "protected_profit"
    if close_realized_pnl == 0:
        return "protected_breakeven_after_prior_profit"
    return "prior_profit_turned_into_realized_loss"


def _profit_protection_audit(trades: list[dict[str, Any]]) -> dict[str, Any]:
    close_trades = [trade for trade in trades if str(trade.get("side") or "").lower() == "sell"]
    protected = [trade for trade in close_trades if trade.get("profit_protection_outcome") == "protected_profit"]
    lost = [trade for trade in close_trades if trade.get("profit_protection_outcome") == "prior_profit_turned_into_realized_loss"]
    prior_profit_closes = [trade for trade in close_trades if float(trade.get("peak_unrealized_pnl_before_close") or 0.0) > 0]
    return {
        "close_trade_count": len(close_trades),
        "prior_profit_close_count": len(prior_profit_closes),
        "protected_profit_count": len(protected),
        "prior_profit_turned_loss_count": len(lost),
        "latest_close": close_trades[-1] if close_trades else None,
        "policy": "Audit compares simulated peak open P/L against realized close P/L using Alpaca option prices and the 100-share contract multiplier.",
    }
