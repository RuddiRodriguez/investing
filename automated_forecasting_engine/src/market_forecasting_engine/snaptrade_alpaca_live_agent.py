from __future__ import annotations

import argparse
import json
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from market_forecasting_engine.snaptrade_readonly_cli import (
    build_client,
    call_or_error,
    current_timestamp,
    first_non_empty,
    load_env_file,
    response_body,
    safe_error,
    user_credentials,
)


def main() -> None:
    load_env_file()
    args = build_parser().parse_args()
    while True:
        print_snapshot(args)
        if not args.loop:
            return
        time.sleep(max(5, int(args.interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read-only SnapTrade Alpaca live account printer."
    )
    parser.add_argument("--client-id", default=None, help="Defaults to SNAPTRADE_CLIENT_ID.")
    parser.add_argument("--consumer-key", default=None, help="Defaults to SNAPTRADE_CONSUMER_KEY.")
    parser.add_argument("--user-id", default=None, help="Defaults to SNAPTRADE_USER_ID.")
    parser.add_argument("--user-secret", default=None, help="Defaults to SNAPTRADE_USER_SECRET.")
    parser.add_argument("--include-paper", action="store_true", help="Also print Alpaca Paper.")
    parser.add_argument("--loop", action="store_true", help="Keep printing snapshots.")
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of readable text.")
    parser.add_argument("--no-color", action="store_true", help="Disable terminal colors.")
    parser.add_argument("--show-closed-orders", action="store_true", help="Also show executed, cancelled, rejected, and expired orders.")
    return parser


def print_snapshot(args: argparse.Namespace) -> None:
    payload = fetch_alpaca_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=True, default=str), flush=True)
        return
    print_readable(payload, color=not args.no_color, show_closed_orders=args.show_closed_orders)


def fetch_alpaca_payload(args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "broker": "snaptrade",
        "target": "alpaca_live",
        "read_only": True,
        "execution_enabled": False,
        "fetched_at": current_timestamp(),
    }
    try:
        client = build_client(args)
        credentials = user_credentials(args)
        accounts = response_body(client.account_information.list_user_accounts(**credentials))
        alpaca_accounts = [
            account for account in accounts if is_alpaca_account(account, include_paper=args.include_paper)
        ]
        rows = []
        for account in alpaca_accounts:
            account_id = first_non_empty(account, "id", "accountId", "account_id")
            row = {"account": account, "account_id": account_id}
            if account_id:
                row["balance"] = call_or_error(
                    client.account_information.get_user_account_balance,
                    account_id=account_id,
                    **credentials,
                )
                row["details"] = call_or_error(
                    client.account_information.get_user_account_details,
                    account_id=account_id,
                    **credentials,
                )
                row["positions"] = call_or_error(
                    client.account_information.get_all_account_positions,
                    account_id=account_id,
                    **credentials,
                )
                row["open_orders"] = call_or_error(
                    client.account_information.get_user_account_orders,
                    account_id=account_id,
                    state="open",
                    days=90,
                    **credentials,
                )
                row["recent_orders"] = call_or_error(
                    client.account_information.get_user_account_recent_orders,
                    account_id=account_id,
                    **credentials,
                )
            rows.append(row)
        base["accounts"] = rows
        base["account_count"] = len(rows)
        return base
    except Exception as exc:
        base["error"] = safe_error(exc)
        return base


def is_alpaca_account(account: dict[str, Any], *, include_paper: bool) -> bool:
    institution = str(first_non_empty(account, "institution_name", "institutionName") or "").lower()
    name = str(first_non_empty(account, "name") or "").lower()
    is_alpaca = "alpaca" in institution or "alpaca" in name
    is_paper = bool(account.get("is_paper")) or "paper" in institution or "paper" in name
    return is_alpaca and (include_paper or not is_paper)


def print_readable(payload: dict[str, Any], *, color: bool = True, show_closed_orders: bool = False) -> None:
    paint = TerminalColors(enabled=color)
    print("")
    print(paint.blue("=" * 100))
    print(f"{paint.bold('SnapTrade Alpaca Live')}  {paint.dim(str(payload.get('fetched_at')))}  {paint.green('READ ONLY')}")
    print(paint.blue("=" * 100))
    if payload.get("error"):
        print(paint.red(f"ERROR: {payload['error']}"))
        return
    accounts = payload.get("accounts") or []
    if not accounts:
        print(paint.yellow("No Alpaca live account returned by SnapTrade."))
        return
    for row in accounts:
        account = row.get("account") or {}
        balance = row.get("balance") or {}
        positions = row.get("positions") or []
        orders = row.get("open_orders") or []
        position_rows = list_rows(positions, "results")
        cash, buying_power, currency = cash_values(balance)
        total_value = sum_decimal(position_value(position) for position in position_rows)
        total_cost = sum_decimal(position_cost(position) for position in position_rows)
        total_pl = total_value - total_cost
        daily_change = daily_change_value(position_rows)
        print("")
        print(f"{paint.bold('Account')}  {account.get('name')} / {account.get('institution_name')}")
        print(f"{paint.dim('Status')}   {account.get('status')}   Paper: {account.get('is_paper')}   id: {row.get('account_id')}")
        print("")
        print_balance_panel(
            buying_power=buying_power,
            cash=cash,
            daily_change=daily_change,
            open_pl=total_pl,
            currency=currency,
            paint=paint,
        )
        sync = account.get("sync_status") or {}
        print("")
        print(
            paint.dim(
                f"Last sync  holdings: {first_non_empty(sync, 'holdings.last_successful_sync') or '-'}   "
                f"transactions: {first_non_empty(sync, 'transactions.last_successful_sync') or '-'}"
            )
        )
        print("")
        print(paint.bold("Top Positions"))
        if isinstance(positions, dict) and positions.get("error"):
            print(paint.red(f"  ERROR: {positions['error']}"))
        elif not position_rows:
            print(paint.dim("  none"))
        else:
            print_positions_table(sorted(position_rows, key=position_value, reverse=True), paint)
        print("")
        print(paint.bold("Open orders"))
        order_rows = list_rows(orders, "orders")
        if not show_closed_orders:
            order_rows = [order for order in order_rows if is_open_order(order)]
        if isinstance(orders, dict) and orders.get("error"):
            print(paint.red(f"  ERROR: {orders['error']}"))
        elif not order_rows:
            print(paint.dim("  none"))
        else:
            print_orders_table(order_rows[:10], paint)
    print(paint.blue("=" * 100), flush=True)


def money_from_balance(balance: Any) -> str:
    if isinstance(balance, list):
        return ", ".join(money_from_balance(item) for item in balance)
    total = first_non_empty(balance, "total.amount", "amount")
    currency = first_non_empty(balance, "total.currency", "currency")
    if total is None:
        return compact(balance)
    return f"{total} {currency or ''}".strip()


def cash_summary(balance: Any) -> str:
    rows = balance if isinstance(balance, list) else [balance]
    parts = []
    for row in rows:
        currency = first_non_empty(row, "currency.code", "currency")
        cash = first_non_empty(row, "cash")
        buying_power = first_non_empty(row, "buying_power", "buyingPower")
        if cash is not None or buying_power is not None:
            parts.append(f"{currency or ''} cash={cash} buying_power={buying_power}".strip())
    return "; ".join(parts) if parts else compact(balance)


def print_balance_panel(
    *,
    buying_power: Decimal,
    cash: Decimal,
    daily_change: Decimal,
    open_pl: Decimal,
    currency: str,
    paint: "TerminalColors",
) -> None:
    print(paint.bold("Balances"))
    print(paint.dim("-" * 100))
    labels = ["Buying Power", "Cash", "Daily Change", "Open P/L"]
    values = [
        money(buying_power, currency),
        money(cash, currency),
        signed_money(daily_change, currency, paint),
        signed_money(open_pl, currency, paint),
    ]
    print("  ".join(paint.dim(label.ljust(20)) for label in labels))
    print("  ".join(value.ljust(20) for value in values))


def print_positions_table(positions: list[dict[str, Any]], paint: "TerminalColors") -> None:
    widths = [8, 14, 12, 12, 14, 14]
    print(format_row(["Asset", "Price", "Qty", "Market Value", "Total P/L", "P/L %"], widths, paint.dim))
    print(paint.dim("-" * 100))
    for position in positions:
        symbol = first_non_empty(position, "instrument.symbol", "instrument.raw_symbol", "symbol.symbol", "symbol.raw_symbol", "symbol", "ticker") or "?"
        qty = first_non_empty(position, "units", "quantity", "qty")
        price = as_decimal(first_non_empty(position, "price", "current_price"))
        currency = first_non_empty(position, "currency.code", "currency") or ""
        value = position_value(position)
        cost = position_cost(position)
        pl = value - cost
        pl_pct = (pl / cost * Decimal("100")) if cost else Decimal("0")
        print(
            format_row(
                [
                    str(symbol),
                    money(price, currency),
                    fmt_decimal(qty, 6),
                    money(value, currency),
                    signed_money(pl, currency, paint),
                    signed_percent(pl_pct, paint),
                ],
                widths,
            )
        )


def print_orders_table(orders: list[dict[str, Any]], paint: "TerminalColors") -> None:
    widths = [20, 6, 8, 12, 10, 14, 12]
    print(format_row(["Time", "Side", "Symbol", "Qty", "Type", "Price", "Status"], widths, paint.dim))
    print(paint.dim("-" * 100))
    for order in orders:
        action = str(first_non_empty(order, "action") or "")
        side = paint.green(action) if action.upper() == "BUY" else paint.red(action)
        when = shorten(str(first_non_empty(order, "time_executed", "time_placed") or "-").replace("T", " "), 20)
        print(
            format_row(
                [
                    when,
                    side,
                    str(first_non_empty(order, "universal_symbol.symbol", "symbol") or ""),
                    fmt_decimal(first_non_empty(order, "open_quantity", "total_quantity", "filled_quantity"), 6),
                    str(first_non_empty(order, "order_type") or ""),
                    fmt_decimal(first_non_empty(order, "execution_price", "limit_price"), 4),
                    str(first_non_empty(order, "status") or ""),
                ],
                widths,
            )
        )


def is_open_order(order: dict[str, Any]) -> bool:
    status = str(first_non_empty(order, "status") or "").upper()
    closed = {
        "EXECUTED",
        "FILLED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "REJECTED",
        "FAILED",
        "DONE",
    }
    return status not in closed


def list_rows(value: Any, key: str) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get(key), list):
        return value[key]
    return []


def position_value(position: dict[str, Any]) -> Decimal:
    value = as_decimal(first_non_empty(position, "market_value", "marketValue", "current_value"))
    if value:
        return value
    return as_decimal(first_non_empty(position, "units", "quantity", "qty")) * as_decimal(first_non_empty(position, "price", "current_price"))


def position_cost(position: dict[str, Any]) -> Decimal:
    return as_decimal(first_non_empty(position, "units", "quantity", "qty")) * as_decimal(first_non_empty(position, "cost_basis", "costBasis"))


def daily_change_value(positions: list[dict[str, Any]]) -> Decimal:
    total = Decimal("0")
    for position in positions:
        value = first_non_empty(position, "today_pnl", "daily_pnl", "daily_change", "day_change")
        if value is not None:
            total += as_decimal(value)
    return total


def cash_values(balance: Any) -> tuple[Decimal, Decimal, str]:
    rows = balance if isinstance(balance, list) else [balance]
    cash = Decimal("0")
    buying_power = Decimal("0")
    currency = ""
    for row in rows:
        cash += as_decimal(first_non_empty(row, "cash"))
        buying_power += as_decimal(first_non_empty(row, "buying_power", "buyingPower"))
        currency = currency or str(first_non_empty(row, "currency.code", "currency") or "")
    return cash, buying_power, currency


def as_decimal(value: Any) -> Decimal:
    if value in (None, ""):
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def sum_decimal(values: Any) -> Decimal:
    total = Decimal("0")
    for value in values:
        total += value
    return total


def money(value: Any, currency: str | None = None) -> str:
    amount = as_decimal(value)
    text = f"{amount:.2f}"
    return f"{text} {currency or ''}".strip()


def signed_money(value: Any, currency: str | None, paint: "TerminalColors") -> str:
    amount = as_decimal(value)
    text = f"{amount:+.2f} {currency or ''}".strip()
    if amount > 0:
        return paint.green(text)
    if amount < 0:
        return paint.red(text)
    return text


def signed_percent(value: Any, paint: "TerminalColors") -> str:
    amount = as_decimal(value)
    text = f"{amount:+.2f}%"
    if amount > 0:
        return paint.green(text)
    if amount < 0:
        return paint.red(text)
    return text


def fmt_decimal(value: Any, places: int) -> str:
    amount = as_decimal(value)
    text = f"{amount:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return value[: max(0, width - 1)] + "…"


def visible_len(value: str) -> int:
    import re

    return len(re.sub(r"\033\[[0-9;]*m", "", value))


def pad(value: str, width: int) -> str:
    return value + " " * max(0, width - visible_len(value))


def format_row(values: list[str], widths: list[int], style: Any | None = None) -> str:
    cells = []
    for value, width in zip(values, widths):
        text = str(value)
        if style:
            text = style(text)
        cells.append(pad(text, width))
    return "  ".join(cells)


def compact(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=True, default=str)[:1000]


class TerminalColors:
    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        if not self.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def blue(self, text: str) -> str:
        return self._wrap("34", text)


if __name__ == "__main__":
    main()
