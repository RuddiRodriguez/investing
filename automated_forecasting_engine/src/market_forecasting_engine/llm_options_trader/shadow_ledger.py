from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_and_update_shadow_state(*, output_dir: Path, currency: str, broker) -> dict[str, Any]:
    state = _load_state(output_dir, currency)
    changed = _update_open_orders(state=state, broker=broker)
    _mark_positions(state=state, broker=broker)
    if changed:
        _save_state(output_dir, currency, state)
    return _summary(state)


def record_simulated_order(
    *,
    output_dir: Path,
    currency: str,
    broker,
    validated_order: dict[str, Any],
    decision: dict[str, Any],
    checked_at_utc: str,
) -> dict[str, Any]:
    state = _load_state(output_dir, currency)
    order_id = f"shadow-{int(time.time() * 1000)}"
    order = {
        "order_id": order_id,
        "created_at_utc": checked_at_utc,
        "updated_at_utc": checked_at_utc,
        "status": "open",
        "source": "llm_live_shadow",
        "decision_reason": decision.get("reason"),
        **validated_order,
    }
    state.setdefault("orders", []).append(order)
    _update_open_orders(state=state, broker=broker)
    _mark_positions(state=state, broker=broker)
    _save_state(output_dir, currency, state)
    return {"shadow_order_id": order_id, "shadow_state": _summary(state)}


def record_simulated_cancel(*, output_dir: Path, currency: str, order_id: str | None, checked_at_utc: str) -> dict[str, Any]:
    return record_simulated_cancels(output_dir=output_dir, currency=currency, order_ids=[order_id] if order_id else [], checked_at_utc=checked_at_utc)


def record_simulated_cancels(*, output_dir: Path, currency: str, order_ids: list[str], checked_at_utc: str) -> dict[str, Any]:
    state = _load_state(output_dir, currency)
    targets = {str(order_id or "").strip() for order_id in order_ids if str(order_id or "").strip()}
    matched_ids: list[str] = []
    for order in state.get("orders", []):
        if str(order.get("order_id")) in targets and order.get("status") == "open":
            order["status"] = "canceled"
            order["updated_at_utc"] = checked_at_utc
            matched_ids.append(str(order.get("order_id")))
    _save_state(output_dir, currency, state)
    return {
        "shadow_cancel_matched": bool(matched_ids),
        "shadow_cancel_matched_ids": matched_ids,
        "shadow_cancel_requested_ids": sorted(targets),
        "shadow_state": _summary(state),
    }


def _update_open_orders(*, state: dict[str, Any], broker) -> bool:
    changed = False
    for order in state.get("orders", []):
        if order.get("status") != "open":
            continue
        book = _safe_book(broker, str(order.get("instrument_name") or ""))
        bid = _float_or_none(book.get("best_bid_price"))
        ask = _float_or_none(book.get("best_ask_price"))
        mark = _float_or_none(book.get("mark_price"))
        side = str(order.get("side") or "").lower()
        limit = _float_or_none(order.get("price"))
        if limit is None:
            continue
        fill_price = None
        if side == "buy" and ask is not None and ask <= limit:
            fill_price = ask
        elif side == "sell" and bid is not None and bid >= limit:
            fill_price = bid
        if fill_price is None:
            order["last_bid"] = bid
            order["last_ask"] = ask
            order["last_mark"] = mark
            order["updated_at_utc"] = datetime.now(UTC).isoformat()
            continue
        _fill_order(state=state, order=order, fill_price=fill_price, mark_price=mark)
        changed = True
    return changed


def _fill_order(*, state: dict[str, Any], order: dict[str, Any], fill_price: float, mark_price: float | None) -> None:
    now = datetime.now(UTC).isoformat()
    order["status"] = "filled"
    order["filled_at_utc"] = now
    order["updated_at_utc"] = now
    order["fill_price"] = fill_price
    amount = float(order.get("amount") or 0.0)
    signed_amount = amount if str(order.get("side")).lower() == "buy" else -amount
    instrument = str(order.get("instrument_name") or "")
    trade = {
        "trade_id": f"shadow-trade-{len(state.get('trades', [])) + 1}",
        "timestamp_utc": now,
        "instrument_name": instrument,
        "side": order.get("side"),
        "amount": amount,
        "price": fill_price,
        "notional": round(amount * fill_price, 8),
        "source_order_id": order.get("order_id"),
    }
    state.setdefault("trades", []).append(trade)
    positions = state.setdefault("positions", {})
    pos = positions.setdefault(
        instrument,
        {
            "instrument_name": instrument,
            "size": 0.0,
            "average_price": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "peak_unrealized_pnl": 0.0,
            "profit_giveback": 0.0,
            "profit_giveback_pct": None,
            "mark_price": mark_price,
        },
    )
    old_size = float(pos.get("size") or 0.0)
    old_avg = float(pos.get("average_price") or 0.0)
    if signed_amount > 0:
        new_size = old_size + signed_amount
        pos["average_price"] = ((old_size * old_avg) + (signed_amount * fill_price)) / new_size if new_size else 0.0
        pos["size"] = new_size
    else:
        close_amount = min(abs(signed_amount), max(0.0, old_size))
        close_realized_pnl = close_amount * (fill_price - old_avg)
        peak_before_close = float(pos.get("peak_unrealized_pnl") or 0.0)
        unrealized_before_close = pos.get("unrealized_pnl")
        giveback_before_close = pos.get("profit_giveback")
        giveback_pct_before_close = pos.get("profit_giveback_pct")
        pos["realized_pnl"] = float(pos.get("realized_pnl") or 0.0) + close_realized_pnl
        pos["size"] = old_size - close_amount
        trade["close_realized_pnl"] = round(close_realized_pnl, 8)
        trade["peak_unrealized_pnl_before_close"] = round(peak_before_close, 8)
        trade["unrealized_pnl_before_close"] = unrealized_before_close
        trade["profit_giveback_before_close"] = giveback_before_close
        trade["profit_giveback_pct_before_close"] = giveback_pct_before_close
        trade["profit_protection_outcome"] = _profit_protection_outcome(
            peak_before_close=peak_before_close,
            close_realized_pnl=close_realized_pnl,
        )
        if pos["size"] <= 1e-12:
            pos["size"] = 0.0
            pos["peak_unrealized_pnl"] = 0.0
            pos["profit_giveback"] = 0.0
            pos["profit_giveback_pct"] = None
    pos["mark_price"] = mark_price
    pos["updated_at_utc"] = now


def _mark_positions(*, state: dict[str, Any], broker) -> None:
    for pos in state.get("positions", {}).values():
        instrument = str(pos.get("instrument_name") or "")
        size = float(pos.get("size") or 0.0)
        if size <= 0:
            pos["unrealized_pnl"] = 0.0
            pos["profit_giveback"] = 0.0
            pos["profit_giveback_pct"] = None
            continue
        book = _safe_book(broker, instrument)
        mark = _float_or_none(book.get("mark_price"))
        bid = _float_or_none(book.get("best_bid_price"))
        ask = _float_or_none(book.get("best_ask_price"))
        if mark is None:
            mark = bid
        avg = float(pos.get("average_price") or 0.0)
        pos["mark_price"] = mark
        pos["last_bid"] = bid
        pos["last_ask"] = ask
        current_pnl = None if mark is None else round(size * (mark - avg), 8)
        pos["unrealized_pnl"] = current_pnl
        if current_pnl is not None:
            peak = max(float(pos.get("peak_unrealized_pnl") or 0.0), float(current_pnl))
            giveback = max(0.0, peak - float(current_pnl))
            pos["peak_unrealized_pnl"] = round(peak, 8)
            pos["profit_giveback"] = round(giveback, 8)
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
        "realized_pnl": round(realized, 8),
        "unrealized_pnl": round(unrealized, 8),
        "total_pnl": round(realized + unrealized, 8),
        "profit_protection_audit": protection_audit,
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }


def _load_state(output_dir: Path, currency: str) -> dict[str, Any]:
    path = _state_path(output_dir, currency)
    if not path.exists():
        return {"orders": [], "positions": {}, "trades": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"orders": [], "positions": {}, "trades": []}
    return payload if isinstance(payload, dict) else {"orders": [], "positions": {}, "trades": []}


def _save_state(output_dir: Path, currency: str, state: dict[str, Any]) -> None:
    path = _state_path(output_dir, currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def _state_path(output_dir: Path, currency: str) -> Path:
    return output_dir / "shadow" / f"{currency.upper()}_shadow_ledger.json"


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
        "policy": "Audit compares simulated peak open P/L against realized close P/L to see whether the LLM protected profit or gave it back.",
    }


def _safe_book(broker, instrument_name: str) -> dict[str, Any]:
    if not instrument_name:
        return {}
    try:
        book = broker.order_book(instrument_name, depth=5)
    except Exception:
        return {}
    return book if isinstance(book, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
