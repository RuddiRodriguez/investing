from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class ReadOnlyBroker(Protocol):
    def account(self) -> dict[str, Any]: ...

    def positions(self) -> list[dict[str, Any]]: ...

    def orders(self, *, status: str = "open", limit: int = 50) -> list[dict[str, Any]]: ...


def build_live_account_report(
    broker: ReadOnlyBroker,
    *,
    venue: str = "alpaca",
    order_limit: int = 100,
    include_closed_orders: bool = True,
) -> dict[str, Any]:
    """Build a read-only live-account report.

    This function intentionally accepts only read methods. It must not submit,
    replace, or cancel orders.
    """

    checked_at = datetime.now(UTC).isoformat()
    account = broker.account()
    positions = broker.positions()
    open_orders = broker.orders(status="open", limit=order_limit)
    recent_orders = broker.orders(status="all", limit=order_limit) if include_closed_orders else open_orders
    stock_positions, option_positions = _split_asset_rows(positions)
    stock_open_orders, option_open_orders = _split_asset_rows(open_orders)
    stock_recent_orders, option_recent_orders = _split_asset_rows(recent_orders)
    stocks = _section_report(
        name="stocks",
        positions=stock_positions,
        open_orders=stock_open_orders,
        recent_orders=stock_recent_orders,
    )
    options = _section_report(
        name="options",
        positions=option_positions,
        open_orders=option_open_orders,
        recent_orders=option_recent_orders,
    )
    return {
        "checked_at": checked_at,
        "mode": "read_only_live_account_report",
        "venue": venue,
        "safety": {
            "order_submission_enabled": False,
            "cancel_enabled": False,
            "live_execution_capability": "disabled_in_this_module",
            "policy": "This report framework is read-only. It fetches account, positions, and order history only.",
        },
        "account": _account_summary(account),
        "overview": {
            "position_count": len(positions),
            "open_order_count": len(open_orders),
            "recent_order_count": len(recent_orders),
            "stock_position_count": len(stock_positions),
            "option_position_count": len(option_positions),
            "stock_open_order_count": len(stock_open_orders),
            "option_open_order_count": len(option_open_orders),
            "total_unrealized_pl": round(stocks["summary"]["unrealized_pl"] + options["summary"]["unrealized_pl"], 2),
            "total_market_value": round(stocks["summary"]["market_value"] + options["summary"]["market_value"], 2),
            "total_cost_basis": round(stocks["summary"]["cost_basis"] + options["summary"]["cost_basis"], 2),
        },
        "stocks": stocks,
        "options": options,
        "raw_counts": {
            "positions": len(positions),
            "open_orders": len(open_orders),
            "recent_orders": len(recent_orders),
        },
    }


def write_live_account_report(report: dict[str, Any], output_dir: Path, *, name: str = "live_account_report") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    checked = str(report.get("checked_at") or datetime.now(UTC).isoformat())
    snapshot_name = checked.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")
    (snapshots / f"{name}_{snapshot_name}.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def _section_report(*, name: str, positions: list[dict[str, Any]], open_orders: list[dict[str, Any]], recent_orders: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_positions = [_position_summary(row) for row in positions]
    normalized_open_orders = [_order_summary(row) for row in open_orders]
    normalized_recent_orders = [_order_summary(row) for row in recent_orders]
    winners = [row for row in normalized_positions if row["unrealized_pl"] > 0]
    losers = [row for row in normalized_positions if row["unrealized_pl"] < 0]
    return {
        "name": name,
        "summary": {
            "position_count": len(normalized_positions),
            "open_order_count": len(normalized_open_orders),
            "recent_order_count": len(normalized_recent_orders),
            "market_value": round(sum(row["market_value"] for row in normalized_positions), 2),
            "cost_basis": round(sum(row["cost_basis"] for row in normalized_positions), 2),
            "unrealized_pl": round(sum(row["unrealized_pl"] for row in normalized_positions), 2),
            "unrealized_pl_pct_weighted": _weighted_pl_pct(normalized_positions),
            "winning_positions": len(winners),
            "losing_positions": len(losers),
        },
        "positions": normalized_positions,
        "open_orders": normalized_open_orders,
        "recent_orders": normalized_recent_orders,
    }


def _split_asset_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stocks: list[dict[str, Any]] = []
    options: list[dict[str, Any]] = []
    for row in rows:
        if _is_option_row(row):
            options.append(row)
        else:
            stocks.append(row)
    return stocks, options


def _is_option_row(row: dict[str, Any]) -> bool:
    asset_class = str(row.get("asset_class") or row.get("asset_class_name") or "").lower()
    if "option" in asset_class:
        return True
    symbol = str(row.get("symbol") or "")
    return _looks_like_occ_option_symbol(symbol)


def _looks_like_occ_option_symbol(symbol: str) -> bool:
    text = symbol.strip().upper().replace(" ", "")
    if len(text) < 15:
        return False
    suffix = text[-15:]
    return suffix[:6].isdigit() and suffix[6] in {"C", "P"} and suffix[7:].isdigit()


def _account_summary(account: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "account_number",
        "status",
        "currency",
        "cash",
        "buying_power",
        "regt_buying_power",
        "daytrading_buying_power",
        "portfolio_value",
        "equity",
        "last_equity",
        "long_market_value",
        "short_market_value",
        "initial_margin",
        "maintenance_margin",
        "pattern_day_trader",
        "trading_blocked",
        "transfers_blocked",
        "account_blocked",
    ]
    return {key: account.get(key) for key in keys if key in account}


def _position_summary(row: dict[str, Any]) -> dict[str, Any]:
    qty = _float(row.get("qty"))
    market_value = _float(row.get("market_value"))
    cost_basis = _float(row.get("cost_basis"))
    unrealized_pl = _float(row.get("unrealized_pl"))
    unrealized_plpc = _float(row.get("unrealized_plpc"))
    return {
        "symbol": row.get("symbol"),
        "asset_class": row.get("asset_class"),
        "exchange": row.get("exchange"),
        "side": row.get("side"),
        "qty": qty,
        "avg_entry_price": _float(row.get("avg_entry_price")),
        "current_price": _float(row.get("current_price")),
        "market_value": market_value,
        "cost_basis": cost_basis,
        "unrealized_pl": unrealized_pl,
        "unrealized_pl_pct": unrealized_plpc,
        "option_details": _parse_occ_symbol(str(row.get("symbol") or "")) if _is_option_row(row) else None,
    }


def _order_summary(row: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "")
    return {
        "id": row.get("id"),
        "client_order_id": row.get("client_order_id"),
        "symbol": symbol,
        "asset_class": row.get("asset_class"),
        "side": row.get("side"),
        "type": row.get("type"),
        "order_class": row.get("order_class"),
        "time_in_force": row.get("time_in_force"),
        "status": row.get("status"),
        "qty": _float(row.get("qty")),
        "filled_qty": _float(row.get("filled_qty")),
        "limit_price": _float(row.get("limit_price")),
        "stop_price": _float(row.get("stop_price")),
        "trail_price": _float(row.get("trail_price")),
        "trail_percent": _float(row.get("trail_percent")),
        "filled_avg_price": _float(row.get("filled_avg_price")),
        "submitted_at": row.get("submitted_at"),
        "filled_at": row.get("filled_at"),
        "canceled_at": row.get("canceled_at"),
        "expired_at": row.get("expired_at"),
        "option_details": _parse_occ_symbol(symbol) if _is_option_row(row) else None,
    }


def _parse_occ_symbol(symbol: str) -> dict[str, Any] | None:
    text = symbol.strip().upper().replace(" ", "")
    if not _looks_like_occ_option_symbol(text):
        return None
    root = text[:-15]
    suffix = text[-15:]
    expiry = suffix[:6]
    option_type = "call" if suffix[6] == "C" else "put"
    strike = int(suffix[7:]) / 1000.0
    return {
        "underlying": root,
        "expiration": f"20{expiry[:2]}-{expiry[2:4]}-{expiry[4:6]}",
        "option_type": option_type,
        "strike": strike,
    }


def _weighted_pl_pct(positions: list[dict[str, Any]]) -> float | None:
    total_cost = sum(abs(row["cost_basis"]) for row in positions)
    if total_cost <= 0:
        return None
    return round(sum(row["unrealized_pl"] for row in positions) / total_cost, 6)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
