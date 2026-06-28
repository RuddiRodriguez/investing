from __future__ import annotations

import argparse
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.ibkr_broker import IBKRBroker, IBKRConnectionConfig


DEFAULT_OUTPUT_DIR = "automated_forecasting_engine/runs/ibkr_manual_buy_agent"


def main() -> None:
    args = build_parser().parse_args()
    plan = load_or_create_plan(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ibkr_manual_buy_plan.json", plan)
    validate_plan(plan)

    broker = None
    if not args.plan_only:
        broker = IBKRBroker(connection_config(args, plan))
        broker.connect()
    try:
        report = run_agent(args=args, plan=plan, broker=broker)
    finally:
        if broker is not None:
            broker.disconnect()
    write_json(output_dir / "ibkr_manual_buy_agent_report.json", report)
    print(json.dumps(console_summary(report), indent=2, sort_keys=True, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual-condition IBKR buy-only limit-order robot.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", help="Path to JSON plan.")
    source.add_argument("--request", help="Plain English request. Example: Use paper. Buy VWCE if price is 120.50 or lower. Use 500 EUR.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--paper-port", type=int, default=7497)
    parser.add_argument("--live-port", type=int, default=7496)
    parser.add_argument("--client-id", type=int, default=7)
    parser.add_argument("--quote-wait-seconds", type=float, default=2.0)
    parser.add_argument("--check-interval-seconds", type=float, default=60.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=0.0, help="0 means no runtime limit when --loop is set.")
    parser.add_argument("--loop", action="store_true", help="Keep checking until all orders are filled/submitted or runtime limit is reached.")
    parser.add_argument("--plan-only", action="store_true", help="Only convert/validate plan. Do not connect to IBKR.")
    parser.add_argument("--execute-orders", action="store_true")
    parser.add_argument("--confirm-risk", action="store_true")
    parser.add_argument("--confirm-live-risk", action="store_true")
    return parser


def load_or_create_plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.plan:
        return read_json(Path(args.plan))
    return parse_plain_english_plan(str(args.request or ""))


def parse_plain_english_plan(text: str) -> dict[str, Any]:
    clean = " ".join(text.strip().split())
    lower = clean.lower()
    account_mode = "live" if re.search(r"\blive\b", lower) else "paper"
    symbol = parse_symbol(clean)
    amount, currency = parse_amount(clean)
    condition_price = parse_condition_price(clean)
    exchange = parse_exchange(clean) or "SMART"
    resubmit = "resubmit" in lower or "submit again" in lower or "if order expires" in lower or "if the order expires" in lower
    return {
        "account_mode": account_mode,
        "orders": [
            {
                "symbol": symbol,
                "exchange": exchange,
                "currency": currency,
                "asset_type": "ETF",
                "side": "BUY",
                "amount_cash": amount,
                "condition": {"type": "price_at_or_below", "price": condition_price},
                "order_type": "LMT",
                "limit_price": condition_price,
                "time_in_force": "DAY",
                "resubmit_if_expired": resubmit,
            }
        ],
    }


def parse_symbol(text: str) -> str:
    patterns = [
        r"\bbuy\s+([A-Z][A-Z0-9.]{1,12})\b",
        r"\bticker\s+([A-Z][A-Z0-9.]{1,12})\b",
        r"\bsymbol\s+([A-Z][A-Z0-9.]{1,12})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate not in {"IF", "AT", "USE", "WITH"}:
                return candidate
    raise ValueError("Could not find ticker. Example: Buy VWCE if price is 120.50 or lower.")


def parse_amount(text: str) -> tuple[float, str]:
    patterns = [
        r"\buse\s+([0-9]+(?:\.[0-9]+)?)\s*(EUR|USD|GBP|CHF)\b",
        r"\bamount\s+([0-9]+(?:\.[0-9]+)?)\s*(EUR|USD|GBP|CHF)\b",
        r"\bfor\s+([0-9]+(?:\.[0-9]+)?)\s*(EUR|USD|GBP|CHF)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1)), match.group(2).upper()
    raise ValueError("Could not find amount. Example: Use 500 EUR.")


def parse_condition_price(text: str) -> float:
    patterns = [
        r"\bprice\s+(?:is\s+)?(?:at\s+)?([0-9]+(?:\.[0-9]+)?)\s*(?:or\s+lower|or\s+less|below|under)\b",
        r"\bat\s+([0-9]+(?:\.[0-9]+)?)\s*(?:or\s+lower|or\s+less|below|under)\b",
        r"\blimit(?:\s+price)?\s+([0-9]+(?:\.[0-9]+)?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    raise ValueError("Could not find condition price. Example: if price is 120.50 or lower.")


def parse_exchange(text: str) -> str | None:
    match = re.search(r"\bexchange\s+([A-Z0-9.]+)\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None


def validate_plan(plan: dict[str, Any]) -> None:
    account_mode = str(plan.get("account_mode") or "").lower()
    if account_mode not in {"paper", "live"}:
        raise ValueError("account_mode must be paper or live.")
    orders = plan.get("orders")
    if not isinstance(orders, list) or not orders:
        raise ValueError("Plan requires at least one order.")
    for index, order in enumerate(orders):
        prefix = f"orders[{index}]"
        if str(order.get("side") or "").upper() != "BUY":
            raise ValueError(f"{prefix}.side must be BUY. First version is buy-only.")
        if str(order.get("order_type") or "").upper() != "LMT":
            raise ValueError(f"{prefix}.order_type must be LMT. Market orders are blocked.")
        if str(order.get("time_in_force") or "").upper() not in {"DAY", "GTC"}:
            raise ValueError(f"{prefix}.time_in_force must be DAY or GTC.")
        if not str(order.get("symbol") or "").strip():
            raise ValueError(f"{prefix}.symbol is required.")
        if float(order.get("amount_cash") or 0) <= 0:
            raise ValueError(f"{prefix}.amount_cash must be positive.")
        condition = order.get("condition") if isinstance(order.get("condition"), dict) else {}
        if condition.get("type") != "price_at_or_below":
            raise ValueError(f"{prefix}.condition.type must be price_at_or_below.")
        condition_price = float(condition.get("price") or 0)
        limit_price = float(order.get("limit_price") or 0)
        if condition_price <= 0 or limit_price <= 0:
            raise ValueError(f"{prefix} condition price and limit_price must be positive.")
        if limit_price > condition_price:
            raise ValueError(f"{prefix}.limit_price cannot be above condition price.")


def connection_config(args: argparse.Namespace, plan: dict[str, Any]) -> IBKRConnectionConfig:
    mode = str(plan.get("account_mode") or "paper").lower()
    return IBKRConnectionConfig(
        host=str(args.host),
        port=int(args.live_port if mode == "live" else args.paper_port),
        client_id=int(args.client_id),
        account_mode=mode,
    )


def run_agent(*, args: argparse.Namespace, plan: dict[str, Any], broker: IBKRBroker | None) -> dict[str, Any]:
    if args.plan_only:
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "mode": "ibkr_manual_buy_agent",
            "dry_run": True,
            "policy": base_policy(),
            "plan": plan,
            "cycles": [],
            "status": "plan_valid",
        }
    started = datetime.now(UTC)
    cycles: list[dict[str, Any]] = []
    terminal: set[str] = set()
    while True:
        cycle = run_cycle(args=args, plan=plan, broker=broker, terminal=terminal)
        cycles.append(cycle)
        for row in cycle["orders"]:
            if row.get("terminal"):
                terminal.add(str(row["symbol"]))
        if not args.loop:
            break
        if len(terminal) >= len(plan["orders"]):
            break
        if args.max_runtime_seconds and (datetime.now(UTC) - started).total_seconds() >= float(args.max_runtime_seconds):
            break
        if broker is None:
            break
        broker.sleep(max(1.0, float(args.check_interval_seconds)))
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "mode": "ibkr_manual_buy_agent",
        "dry_run": not bool(args.execute_orders),
        "policy": base_policy(),
        "plan": plan,
        "cycles": cycles,
    }


def base_policy() -> dict[str, Any]:
    return {
        "buy_only": True,
        "limit_orders_only": True,
        "plain_english_allowed": True,
        "live_requires_execute_orders_and_confirm_live_risk": True,
        "no_credentials_or_2fa_stored": True,
    }


def run_cycle(*, args: argparse.Namespace, plan: dict[str, Any], broker: IBKRBroker | None, terminal: set[str]) -> dict[str, Any]:
    now = datetime.now(UTC)
    account_mode = str(plan.get("account_mode") or "paper").lower()
    base_blocks = execution_blocks(args=args, account_mode=account_mode)
    account_values = broker.account_values() if broker is not None else []
    open_orders = broker.open_orders() if broker is not None else []
    rows: list[dict[str, Any]] = []
    for raw_order in plan["orders"]:
        order = normalize_order(raw_order)
        symbol = order["symbol"]
        row = {"symbol": symbol, "blocks": list(base_blocks), "quote": None, "payload": None, "effects": [], "terminal": False}
        if symbol in terminal:
            row["blocks"].append("already_terminal")
            row["terminal"] = True
            rows.append(row)
            continue
        duplicate = matching_open_order(open_orders, order)
        if duplicate:
            row["effects"].append({"action": "open_order_already_exists", "order": duplicate})
            rows.append(row)
            continue
        contract = None
        if broker is not None:
            contract = broker.make_stock_contract(symbol=symbol, exchange=order["exchange"], currency=order["currency"])
            row["contract"] = {
                "symbol": getattr(contract, "symbol", None),
                "secType": getattr(contract, "secType", None),
                "exchange": getattr(contract, "exchange", None),
                "currency": getattr(contract, "currency", None),
                "conId": getattr(contract, "conId", None),
            }
            row["quote"] = broker.snapshot_quote(contract, wait_seconds=float(args.quote_wait_seconds))
        quote_price = execution_reference_price(row.get("quote") or {}, order)
        if quote_price is None:
            row["blocks"].append("no_quote_price")
            rows.append(row)
            continue
        if quote_price > order["condition"]["price"]:
            row["effects"].append(
                {
                    "action": "condition_not_met",
                    "reference_price": quote_price,
                    "required_price_at_or_below": order["condition"]["price"],
                }
            )
            rows.append(row)
            continue
        quantity = round(float(order["amount_cash"]) / float(order["limit_price"]), 6)
        if quantity <= 0:
            row["blocks"].append("calculated_quantity_zero")
        row["payload"] = {
            "symbol": symbol,
            "exchange": order["exchange"],
            "currency": order["currency"],
            "action": "BUY",
            "orderType": "LMT",
            "totalQuantity": quantity,
            "lmtPrice": order["limit_price"],
            "tif": order["time_in_force"],
        }
        if row["blocks"]:
            rows.append(row)
            continue
        if not args.execute_orders:
            row["effects"].append({"action": "would_submit_limit_buy", "payload": row["payload"]})
        elif broker is not None and contract is not None:
            row["effects"].append(
                {
                    "action": "submitted_limit_buy",
                    "trade": broker.place_limit_buy(contract, quantity=quantity, limit_price=order["limit_price"], tif=order["time_in_force"]),
                }
            )
            row["terminal"] = not bool(order["resubmit_if_expired"])
        rows.append(row)
    return {"generated_at": now.isoformat(), "orders": rows, "account_values": compact_account_values(account_values), "open_orders_count": len(open_orders)}


def execution_blocks(*, args: argparse.Namespace, account_mode: str) -> list[str]:
    blocks: list[str] = []
    if args.execute_orders and not args.confirm_risk:
        blocks.append("missing_confirm_risk")
    if account_mode == "live" and (not args.execute_orders or not args.confirm_live_risk):
        blocks.append("live_requires_execute_orders_and_confirm_live_risk")
    return blocks


def normalize_order(order: dict[str, Any]) -> dict[str, Any]:
    condition = order.get("condition") if isinstance(order.get("condition"), dict) else {}
    return {
        "symbol": str(order.get("symbol") or "").upper(),
        "exchange": str(order.get("exchange") or "SMART").upper(),
        "currency": str(order.get("currency") or "USD").upper(),
        "amount_cash": float(order.get("amount_cash") or 0),
        "condition": {"type": "price_at_or_below", "price": float(condition.get("price") or 0)},
        "order_type": "LMT",
        "limit_price": float(order.get("limit_price") or 0),
        "time_in_force": str(order.get("time_in_force") or "DAY").upper(),
        "resubmit_if_expired": bool(order.get("resubmit_if_expired")),
    }


def execution_reference_price(quote: dict[str, Any], order: dict[str, Any]) -> float | None:
    return number(quote.get("ask")) or number(quote.get("last")) or number(quote.get("marketPrice")) or number(quote.get("close"))


def matching_open_order(open_orders: list[dict[str, Any]], order: dict[str, Any]) -> dict[str, Any] | None:
    for row in open_orders:
        contract = row.get("contract") if isinstance(row.get("contract"), dict) else {}
        payload = row.get("order") if isinstance(row.get("order"), dict) else {}
        if str(contract.get("symbol") or "").upper() != order["symbol"]:
            continue
        if str(payload.get("action") or "").upper() != "BUY":
            continue
        if str(payload.get("orderType") or "").upper() != "LMT":
            continue
        return row
    return None


def compact_account_values(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {"NetLiquidation", "AvailableFunds", "BuyingPower", "CashBalance"}
    return [
        {"tag": row.get("tag"), "value": row.get("value"), "currency": row.get("currency")}
        for row in values
        if row.get("tag") in wanted
    ]


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") == "plan_valid":
        return {"mode": report["mode"], "status": "plan_valid", "account_mode": report["plan"].get("account_mode"), "plan": report["plan"]}
    last = report["cycles"][-1] if report.get("cycles") else {"orders": []}
    return {
        "mode": report["mode"],
        "dry_run": report["dry_run"],
        "account_mode": report["plan"].get("account_mode"),
        "orders": [
            {
                "symbol": row.get("symbol"),
                "blocks": row.get("blocks"),
                "quote": row.get("quote"),
                "payload": row.get("payload"),
                "effects": row.get("effects"),
            }
            for row in last.get("orders", [])
        ],
    }


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
