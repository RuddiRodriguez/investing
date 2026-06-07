from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def evaluate_limit_order_fill(
    *,
    order: dict[str, Any],
    quote: dict[str, Any],
    recent_trades: list[dict[str, Any]] | None,
    side: str,
    limit_price: float,
) -> dict[str, Any]:
    """Evaluate a simulated limit order using quote crossing plus trade tape.

    The quote crossing rule handles aggressive fills. The trade-tape rule handles
    passive resting orders that can fill when a trade prints at the resting limit
    even if the visible opposite quote never crosses in the sampled snapshots.
    """
    bid = _float_or_none(quote.get("bid") or quote.get("best_bid_price"))
    ask = _float_or_none(quote.get("ask") or quote.get("best_ask_price"))
    mark = _float_or_none(quote.get("mark") or quote.get("mark_price"))

    if side == "buy" and ask is not None and ask <= limit_price:
        return {
            "filled": True,
            "fill_price": ask,
            "mark_price": mark,
            "fill_reason": "quote_crossed_limit",
            "reference_trade": None,
            "cursor": _trade_cursor(order=order, recent_trades=recent_trades),
        }
    if side == "sell" and bid is not None and bid >= limit_price:
        return {
            "filled": True,
            "fill_price": bid,
            "mark_price": mark,
            "fill_reason": "quote_crossed_limit",
            "reference_trade": None,
            "cursor": _trade_cursor(order=order, recent_trades=recent_trades),
        }

    passive_fill = _passive_fill_from_trade_tape(
        order=order,
        recent_trades=recent_trades or [],
        side=side,
        limit_price=limit_price,
    )
    if passive_fill is not None:
        return {
            "filled": True,
            "fill_price": limit_price,
            "mark_price": mark,
            "fill_reason": passive_fill["fill_reason"],
            "reference_trade": passive_fill["reference_trade"],
            "cursor": _trade_cursor(order=order, recent_trades=recent_trades),
        }

    return {
        "filled": False,
        "fill_price": None,
        "mark_price": mark,
        "fill_reason": None,
        "reference_trade": None,
        "cursor": _trade_cursor(order=order, recent_trades=recent_trades),
        "last_bid": bid,
        "last_ask": ask,
        "last_mark": mark,
    }


def normalize_recent_trades(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        rows = payload.get("trades") or payload.get("data") or payload.get("latest_trades") or []
        if isinstance(rows, dict):
            rows = list(rows.values())
    else:
        rows = payload
    if not isinstance(rows, list):
        return []
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        price = _float_or_none(raw.get("price") or raw.get("p"))
        timestamp_ms = _timestamp_ms(raw.get("timestamp") or raw.get("time") or raw.get("t"))
        if price is None or timestamp_ms is None:
            continue
        normalized.append(
            {
                "price": price,
                "timestamp_ms": timestamp_ms,
                "trade_id": raw.get("trade_id") or raw.get("id") or raw.get("i"),
                "direction": _normalize_direction(raw.get("direction") or raw.get("side") or raw.get("taker_side")),
                "raw": {
                    key: raw.get(key)
                    for key in ("trade_id", "id", "timestamp", "time", "t", "price", "p", "direction", "side", "taker_side")
                    if key in raw
                },
            }
        )
    normalized.sort(key=lambda item: int(item["timestamp_ms"]))
    return normalized


def apply_order_observation(order: dict[str, Any], evaluation: dict[str, Any]) -> None:
    order["last_bid"] = evaluation.get("last_bid")
    order["last_ask"] = evaluation.get("last_ask")
    order["last_mark"] = evaluation.get("last_mark")
    cursor = evaluation.get("cursor") or {}
    if cursor.get("last_seen_trade_timestamp_ms") is not None:
        order["last_seen_trade_timestamp_ms"] = cursor.get("last_seen_trade_timestamp_ms")
    if cursor.get("last_seen_trade_id") is not None:
        order["last_seen_trade_id"] = cursor.get("last_seen_trade_id")
    if cursor.get("observed_trade_count") is not None:
        order["observed_trade_count"] = cursor.get("observed_trade_count")
    order["updated_at_utc"] = datetime.now(UTC).isoformat()


def _passive_fill_from_trade_tape(
    *,
    order: dict[str, Any],
    recent_trades: list[dict[str, Any]],
    side: str,
    limit_price: float,
) -> dict[str, Any] | None:
    min_timestamp_ms = _minimum_new_trade_timestamp_ms(order)
    for trade in recent_trades:
        timestamp_ms = _int_or_none(trade.get("timestamp_ms"))
        if timestamp_ms is None or timestamp_ms <= min_timestamp_ms:
            continue
        price = _float_or_none(trade.get("price"))
        if price is None:
            continue
        direction = _normalize_direction(trade.get("direction"))
        if side == "buy" and price <= limit_price and _compatible_passive_direction(direction, expected="sell"):
            return {
                "fill_reason": "public_trade_touched_passive_bid",
                "reference_trade": _public_reference_trade(trade),
            }
        if side == "sell" and price >= limit_price and _compatible_passive_direction(direction, expected="buy"):
            return {
                "fill_reason": "public_trade_touched_passive_ask",
                "reference_trade": _public_reference_trade(trade),
            }
    return None


def _compatible_passive_direction(direction: str | None, *, expected: str) -> bool:
    return direction is None or direction == expected


def _minimum_new_trade_timestamp_ms(order: dict[str, Any]) -> int:
    created = _timestamp_ms(order.get("created_at_utc")) or 0
    last_seen = _int_or_none(order.get("last_seen_trade_timestamp_ms")) or 0
    return max(created, last_seen)


def _trade_cursor(*, order: dict[str, Any], recent_trades: list[dict[str, Any]] | None) -> dict[str, Any]:
    min_timestamp_ms = _minimum_new_trade_timestamp_ms(order)
    newest_timestamp_ms: int | None = None
    newest_trade_id: Any = None
    observed = 0
    for trade in recent_trades or []:
        timestamp_ms = _int_or_none(trade.get("timestamp_ms"))
        if timestamp_ms is None or timestamp_ms <= min_timestamp_ms:
            continue
        observed += 1
        if newest_timestamp_ms is None or timestamp_ms >= newest_timestamp_ms:
            newest_timestamp_ms = timestamp_ms
            newest_trade_id = trade.get("trade_id")
    return {
        "last_seen_trade_timestamp_ms": newest_timestamp_ms,
        "last_seen_trade_id": newest_trade_id,
        "observed_trade_count": observed,
    }


def _public_reference_trade(trade: dict[str, Any]) -> dict[str, Any]:
    return {
        "trade_id": trade.get("trade_id"),
        "timestamp_ms": trade.get("timestamp_ms"),
        "price": trade.get("price"),
        "direction": trade.get("direction"),
        "raw": trade.get("raw"),
    }


def _timestamp_ms(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = int(value)
        if numeric > 10_000_000_000:
            return numeric
        if numeric > 10_000_000:
            return numeric * 1000
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        numeric = _float_or_none(text)
        if numeric is not None:
            return _timestamp_ms(numeric)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return int(parsed.timestamp() * 1000)
    return None


def _normalize_direction(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text in {"buy", "sell"}:
        return text
    return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
