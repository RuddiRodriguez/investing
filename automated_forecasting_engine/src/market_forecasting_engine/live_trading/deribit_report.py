from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol


class DeribitReadOnlyAccount(Protocol):
    account_mode: str
    base_url: str

    def account_summary(self, *, currency: str = "ETH") -> dict[str, Any]: ...

    def positions(self, *, currency: str = "ETH", kind: str = "any") -> list[dict[str, Any]]: ...

    def open_orders(self, *, currency: str = "ETH", kind: str = "any") -> list[dict[str, Any]]: ...

    def order_history(self, *, currency: str = "ETH", kind: str = "any", count: int = 100) -> list[dict[str, Any]]: ...

    def user_trades(self, *, currency: str = "ETH", kind: str = "any", count: int = 100) -> list[dict[str, Any]]: ...


def build_deribit_account_report(
    broker: DeribitReadOnlyAccount,
    *,
    currencies: list[str],
    kinds: list[str],
    history_count: int = 100,
) -> dict[str, Any]:
    checked_at = datetime.now(UTC).isoformat()
    access_issues: list[dict[str, Any]] = []
    account_summaries: dict[str, dict[str, Any]] = {}
    positions: list[dict[str, Any]] = []
    open_orders: list[dict[str, Any]] = []
    order_history: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for currency in [item.upper() for item in currencies]:
        account_summaries[currency] = _safe_fetch(access_issues, "account_summary", currency, "account", lambda: broker.account_summary(currency=currency), {})
        for kind in kinds:
            if kind != "spot":
                positions.extend(
                    _tag_rows(
                        _safe_fetch(access_issues, "positions", currency, kind, lambda c=currency, k=kind: broker.positions(currency=c, kind=k), []),
                        currency=currency,
                        kind=kind,
                    )
                )
            open_orders.extend(
                _tag_rows(
                    _safe_fetch(access_issues, "open_orders", currency, kind, lambda c=currency, k=kind: broker.open_orders(currency=c, kind=k), []),
                    currency=currency,
                    kind=kind,
                )
            )
            order_history.extend(
                _tag_rows(
                    _safe_fetch(
                        access_issues,
                        "order_history",
                        currency,
                        kind,
                        lambda c=currency, k=kind: broker.order_history(currency=c, kind=k, count=history_count),
                        [],
                    ),
                    currency=currency,
                    kind=kind,
                )
            )
            trades.extend(
                _tag_rows(
                    _safe_fetch(
                        access_issues,
                        "user_trades",
                        currency,
                        kind,
                        lambda c=currency, k=kind: broker.user_trades(currency=c, kind=k, count=history_count),
                        [],
                    ),
                    currency=currency,
                    kind=kind,
                )
            )

    deduped_open_orders = _dedupe_rows(open_orders, key="order_id")
    deduped_order_history = _dedupe_rows(order_history, key="order_id")
    deduped_trades = _dedupe_rows(trades, key="trade_id")
    option_positions, non_option_positions = _split_deribit_rows(positions)
    option_open_orders, non_option_open_orders = _split_deribit_rows(deduped_open_orders)
    option_order_history, non_option_order_history = _split_deribit_rows(deduped_order_history)
    option_trades, non_option_trades = _split_deribit_rows(deduped_trades)
    options = _section_report(
        name="options",
        positions=option_positions,
        open_orders=option_open_orders,
        order_history=option_order_history,
        trades=option_trades,
    )
    non_options = _section_report(
        name="non_options",
        positions=non_option_positions,
        open_orders=non_option_open_orders,
        order_history=non_option_order_history,
        trades=non_option_trades,
    )
    return {
        "checked_at": checked_at,
        "mode": "read_only_deribit_account_report",
        "venue": f"deribit_{getattr(broker, 'account_mode', 'unknown')}",
        "endpoint": getattr(broker, "base_url", None),
        "safety": {
            "order_submission_enabled": False,
            "cancel_enabled": False,
            "live_execution_capability": "disabled_in_this_module",
            "policy": "This Deribit dashboard is read-only. It fetches account summaries, positions, orders, and trade history only.",
        },
        "currencies": [item.upper() for item in currencies],
        "kinds": kinds,
        "account_summaries": {currency: _account_summary(summary) for currency, summary in account_summaries.items()},
        "overview": {
            "position_count": len(positions),
            "open_order_count": len(deduped_open_orders),
            "order_history_count": len(deduped_order_history),
            "trade_count": len(deduped_trades),
            "option_position_count": len(option_positions),
            "non_option_position_count": len(non_option_positions),
            "option_open_order_count": len(option_open_orders),
            "non_option_open_order_count": len(non_option_open_orders),
            "floating_profit_loss": _round(sum(row["floating_profit_loss"] for row in options["positions"] + non_options["positions"])),
            "total_profit_loss": _round(sum(row["total_profit_loss"] for row in options["positions"] + non_options["positions"])),
            "option_floating_profit_loss": options["summary"]["floating_profit_loss"],
            "non_option_floating_profit_loss": non_options["summary"]["floating_profit_loss"],
            "access_issue_count": len(access_issues),
        },
        "options": options,
        "non_options": non_options,
        "access_issues": access_issues,
        "raw_counts": {
            "positions": len(positions),
            "open_orders": len(deduped_open_orders),
            "order_history": len(deduped_order_history),
            "trades": len(deduped_trades),
        },
    }


def write_deribit_account_report(report: dict[str, Any], output_dir: Path, *, name: str | None = None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = name or f"{report.get('venue', 'deribit')}_account_report"
    path = output_dir / f"{file_name}.json"
    payload = json.dumps(report, indent=2, default=str, allow_nan=False)
    path.write_text(payload, encoding="utf-8")
    snapshots = output_dir / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    checked = str(report.get("checked_at") or datetime.now(UTC).isoformat())
    snapshot_name = checked.replace(":", "").replace("-", "").replace("+", "Z").replace(".", "_")
    (snapshots / f"{file_name}_{snapshot_name}.json").write_text(payload, encoding="utf-8")
    return path


def _safe_fetch(access_issues: list[dict[str, Any]], endpoint: str, currency: str, kind: str, fn: Any, fallback: Any) -> Any:
    try:
        return fn()
    except Exception as exc:
        access_issues.append({"endpoint": endpoint, "currency": currency, "kind": kind, "error": str(exc)})
        return fallback


def _tag_rows(rows: list[dict[str, Any]], *, currency: str, kind: str) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        copy = dict(row)
        copy.setdefault("currency", currency)
        copy.setdefault("kind", kind)
        tagged.append(copy)
    return tagged


def _section_report(
    *,
    name: str,
    positions: list[dict[str, Any]],
    open_orders: list[dict[str, Any]],
    order_history: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_positions = [_position_summary(row) for row in positions]
    normalized_open_orders = [_order_summary(row) for row in open_orders]
    normalized_order_history = [_order_summary(row) for row in order_history]
    normalized_trades = [_trade_summary(row) for row in trades]
    winners = [row for row in normalized_positions if row["floating_profit_loss"] > 0]
    losers = [row for row in normalized_positions if row["floating_profit_loss"] < 0]
    return {
        "name": name,
        "summary": {
            "position_count": len(normalized_positions),
            "open_order_count": len(normalized_open_orders),
            "order_history_count": len(normalized_order_history),
            "trade_count": len(normalized_trades),
            "floating_profit_loss": _round(sum(row["floating_profit_loss"] for row in normalized_positions)),
            "total_profit_loss": _round(sum(row["total_profit_loss"] for row in normalized_positions)),
            "winning_positions": len(winners),
            "losing_positions": len(losers),
            "net_delta": _round(sum(row["delta"] for row in normalized_positions)),
            "net_gamma": _round(sum(row["gamma"] for row in normalized_positions)),
            "net_vega": _round(sum(row["vega"] for row in normalized_positions)),
        },
        "positions": normalized_positions,
        "open_orders": normalized_open_orders,
        "order_history": normalized_order_history,
        "trades": normalized_trades,
    }


def _account_summary(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "currency",
        "equity",
        "balance",
        "available_funds",
        "available_withdrawal_funds",
        "initial_margin",
        "maintenance_margin",
        "margin_balance",
        "options_session_rpl",
        "options_value",
        "futures_session_rpl",
        "futures_pl",
        "total_pl",
        "delta_total",
        "session_upl",
        "session_rpl",
    ]
    return {key: row.get(key) for key in keys if key in row}


def _split_deribit_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    options: list[dict[str, Any]] = []
    non_options: list[dict[str, Any]] = []
    for row in rows:
        if _is_deribit_option(row):
            options.append(row)
        else:
            non_options.append(row)
    return options, non_options


def _is_deribit_option(row: dict[str, Any]) -> bool:
    kind = str(row.get("kind") or row.get("instrument_kind") or "").lower()
    instrument = str(row.get("instrument_name") or "")
    return kind == "option" or bool(_parse_deribit_option_symbol(instrument))


def _position_summary(row: dict[str, Any]) -> dict[str, Any]:
    instrument = str(row.get("instrument_name") or "")
    return {
        "instrument_name": instrument,
        "currency": row.get("currency"),
        "kind": row.get("kind"),
        "direction": row.get("direction"),
        "size": _float(row.get("size")),
        "size_currency": _float(row.get("size_currency")),
        "average_price": _float(row.get("average_price")),
        "mark_price": _float(row.get("mark_price")),
        "index_price": _float(row.get("index_price")),
        "floating_profit_loss": _float(row.get("floating_profit_loss")),
        "total_profit_loss": _float(row.get("total_profit_loss")),
        "realized_profit_loss": _float(row.get("realized_profit_loss")),
        "initial_margin": _float(row.get("initial_margin")),
        "maintenance_margin": _float(row.get("maintenance_margin")),
        "delta": _float(row.get("delta")),
        "gamma": _float(row.get("gamma")),
        "vega": _float(row.get("vega")),
        "theta": _float(row.get("theta")),
        "option_details": _parse_deribit_option_symbol(instrument),
    }


def _order_summary(row: dict[str, Any]) -> dict[str, Any]:
    instrument = str(row.get("instrument_name") or "")
    return {
        "order_id": row.get("order_id"),
        "label": row.get("label"),
        "instrument_name": instrument,
        "currency": row.get("currency"),
        "kind": row.get("kind"),
        "direction": row.get("direction"),
        "order_type": row.get("order_type"),
        "order_state": row.get("order_state"),
        "time_in_force": row.get("time_in_force"),
        "price": _float(row.get("price")),
        "trigger_price": _float(row.get("trigger_price")),
        "trigger": row.get("trigger"),
        "average_price": _float(row.get("average_price")),
        "amount": _float(row.get("amount")),
        "filled_amount": _float(row.get("filled_amount")),
        "oto_order_ids": row.get("oto_order_ids") if isinstance(row.get("oto_order_ids"), list) else [],
        "replaced": bool(row.get("replaced")) if row.get("replaced") is not None else None,
        "reduce_only": bool(row.get("reduce_only")) if row.get("reduce_only") is not None else None,
        "creation_timestamp": _deribit_time(row.get("creation_timestamp")),
        "last_update_timestamp": _deribit_time(row.get("last_update_timestamp")),
        "option_details": _parse_deribit_option_symbol(instrument),
    }


def _trade_summary(row: dict[str, Any]) -> dict[str, Any]:
    instrument = str(row.get("instrument_name") or "")
    return {
        "trade_id": row.get("trade_id"),
        "order_id": row.get("order_id"),
        "instrument_name": instrument,
        "currency": row.get("currency"),
        "kind": row.get("kind"),
        "direction": row.get("direction"),
        "price": _float(row.get("price")),
        "amount": _float(row.get("amount")),
        "fee": _float(row.get("fee")),
        "fee_currency": row.get("fee_currency"),
        "timestamp": _deribit_time(row.get("timestamp")),
        "option_details": _parse_deribit_option_symbol(instrument),
    }


def _parse_deribit_option_symbol(symbol: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"([A-Z]+)-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])", symbol.upper())
    if not match:
        return None
    currency, day, month, year, strike, option_type = match.groups()
    month_number = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }.get(month)
    expiry = f"20{year}-{month_number}-{int(day):02d}" if month_number else None
    return {
        "underlying": currency,
        "expiration": expiry,
        "option_type": "call" if option_type == "C" else "put",
        "strike": _float(strike),
    }


def _dedupe_rows(rows: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            output.append(row)
            continue
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        output.append(row)
    return output


def _deribit_time(value: Any) -> str | None:
    numeric = _float(value)
    if numeric <= 0:
        return None
    return datetime.fromtimestamp(numeric / 1000, UTC).isoformat()


def _round(value: float) -> float:
    return round(float(value), 8)


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
