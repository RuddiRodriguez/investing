from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker, load_env_file


LIVE_BASE_URL = "https://api.alpaca.markets"


def main() -> None:
    args = build_parser().parse_args()
    load_env_file()
    broker = live_broker(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state = read_state(output_dir / f"{args.symbol.upper()}_breakout_state.json")
    while True:
        report = run_once(args=args, broker=broker, state=state)
        write_json(output_dir / f"{args.symbol.upper()}_breakout_report.json", report)
        write_json(output_dir / f"{args.symbol.upper()}_breakout_state.json", state)
        print(json.dumps(console_summary(report), indent=2, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(15, int(args.check_interval_seconds)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live Alpaca stock/ETF breakout monitor with explicit execution gates.")
    parser.add_argument("--symbol", default="IWM")
    parser.add_argument("--trigger-price", type=float, default=294.50)
    parser.add_argument("--entry-limit-price", type=float, default=295.00)
    parser.add_argument("--hold-bars", type=int, default=3, help="Required consecutive 1m closes at or above trigger.")
    parser.add_argument("--volume-window-bars", type=int, default=10, help="Recent 1m bars used for volume pace confirmation.")
    parser.add_argument("--opening-hour-bars", type=int, default=60)
    parser.add_argument("--volume-pace-multiplier", type=float, default=1.05)
    parser.add_argument("--stop-price", type=float, default=286.00)
    parser.add_argument("--target1-price", type=float, default=305.00)
    parser.add_argument("--target2-price", type=float, default=312.00)
    parser.add_argument("--invalidation-price", type=float, default=292.80)
    parser.add_argument("--protective-order-style", choices=("oco", "agent_managed"), default="oco")
    parser.add_argument("--stop-limit-offset-pct", type=float, default=0.003)
    parser.add_argument("--max-notional", type=float, default=None, help="Hard cap. Defaults to available buying power times --buying-power-fraction.")
    parser.add_argument("--buying-power-fraction", type=float, default=0.95)
    parser.add_argument("--min-notional", type=float, default=1.0)
    parser.add_argument("--exit-limit-offset-pct", type=float, default=0.003)
    parser.add_argument("--data-feed", choices=("iex", "sip", "auto"), default="iex")
    parser.add_argument("--check-interval-seconds", type=int, default=60)
    parser.add_argument("--output-dir", default="automated_forecasting_engine/runs/live_alpaca_breakout_iwm")
    parser.add_argument("--execute-live-orders", action="store_true")
    parser.add_argument("--confirm-live-order-risk", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--base-url", default=None)
    return parser


def run_once(*, args: argparse.Namespace, broker: AlpacaPaperBroker, state: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now(UTC)
    symbol = args.symbol.upper()
    account = broker.account()
    clock = broker.clock()
    asset = safe_asset(broker, symbol)
    position = broker.position(symbol)
    open_orders = [
        order
        for order in broker.orders(status="open", limit=100)
        if str(order.get("symbol") or "").upper() == symbol
    ]
    bars = load_intraday_bars(broker, symbol=symbol, feed=None if args.data_feed == "auto" else args.data_feed)
    confirmation = evaluate_breakout_confirmation(
        bars=bars,
        trigger_price=float(args.trigger_price),
        hold_bars=int(args.hold_bars),
        volume_window_bars=int(args.volume_window_bars),
        opening_hour_bars=int(args.opening_hour_bars),
        volume_pace_multiplier=float(args.volume_pace_multiplier),
    )
    buying_power = number(account.get("buying_power"))
    cash = number(account.get("cash"))
    latest_price = confirmation.get("latest_price")
    position_qty = number((position or {}).get("qty")) or 0.0
    side_effects: list[dict[str, Any]] = []
    execution_blocks = live_execution_blocks(args=args, account=account, clock=clock, asset=asset)
    sizing = size_entry(
        buying_power=buying_power,
        latest_price=float(args.entry_limit_price),
        max_notional=args.max_notional,
        buying_power_fraction=float(args.buying_power_fraction),
        min_notional=float(args.min_notional),
    )

    action = "wait"
    reason = "waiting_for_breakout_confirmation"
    order_payload: dict[str, Any] | None = None
    if position_qty > 0:
        sell_orders = [order for order in open_orders if str(order.get("side") or "").lower() == "sell"]
        if sell_orders:
            action, reason, order_payload, managed_effects = manage_existing_sell_orders(
                broker=broker,
                args=args,
                sell_orders=sell_orders,
                position_qty=position_qty,
                latest_price=latest_price,
                execution_blocks=execution_blocks,
                now=now,
            )
            side_effects.extend(managed_effects)
        else:
            protective_payload = build_protective_order_payload(args=args, position_qty=position_qty, now=now)
            if protective_payload:
                action = "submit_protective_oco"
                reason = "filled_position_missing_resting_protection"
                order_payload = protective_payload
                side_effects.extend(
                    submit_if_allowed(
                        broker=broker,
                        args=args,
                        payload=protective_payload,
                        execution_blocks=execution_blocks,
                        label=action,
                    )
                )
            else:
                action, reason, order_payload = manage_existing_position(args=args, position_qty=position_qty, latest_price=latest_price, state=state)
                if order_payload:
                    side_effects.extend(submit_if_allowed(broker=broker, args=args, payload=order_payload, execution_blocks=execution_blocks, label=action))
    elif open_orders:
        action = "wait"
        reason = "open_order_already_exists"
    elif not confirmation["confirmed"]:
        action = "wait"
        reason = "breakout_not_confirmed"
    elif sizing["blocked"]:
        action = "blocked"
        reason = "insufficient_buying_power_for_configured_min_notional"
    else:
        action = "buy_breakout"
        reason = "confirmed_breakout_with_volume_pace"
        order_payload = {
            "symbol": symbol,
            "side": "buy",
            "type": "limit",
            "qty": str(sizing["qty"]),
            "limit_price": str(round(float(args.entry_limit_price), 2)),
            "time_in_force": "day",
            "client_order_id": client_order_id("breakout-entry", symbol, now),
        }
        side_effects.extend(submit_if_allowed(broker=broker, args=args, payload=order_payload, execution_blocks=execution_blocks, label=action))

    report = {
        "generated_at": now.isoformat(),
        "mode": "live_alpaca_breakout_monitor",
        "symbol": symbol,
        "policy": "WAIT_FOR_BREAKOUT",
        "account": {
            "status": account.get("status"),
            "cash": cash,
            "buying_power": buying_power,
            "portfolio_value": number(account.get("portfolio_value")),
            "trading_blocked": account.get("trading_blocked"),
            "account_blocked": account.get("account_blocked"),
        },
        "clock": {key: clock.get(key) for key in ("timestamp", "is_open", "next_open", "next_close")},
        "asset": {
            "symbol": (asset or {}).get("symbol"),
            "tradable": (asset or {}).get("tradable"),
            "fractionable": (asset or {}).get("fractionable"),
            "status": (asset or {}).get("status"),
        },
        "plan": {
            "trigger": float(args.trigger_price),
            "entry_limit": float(args.entry_limit_price),
            "required_confirmation": f"{args.hold_bars} closes above trigger and recent volume pace > opening hour pace x {args.volume_pace_multiplier}",
            "stop": float(args.stop_price),
            "target1": float(args.target1_price),
            "target2": float(args.target2_price),
            "invalidation": float(args.invalidation_price),
            "protective_order_style": str(args.protective_order_style),
            "reason": "strongest trend in scan; wait for resistance shelf to clear before buying",
        },
        "confirmation": confirmation,
        "sizing": sizing,
        "position": position,
        "open_orders": open_orders,
        "execution_blocks": execution_blocks,
        "action": action,
        "reason": reason,
        "order_payload": order_payload,
        "side_effects": side_effects,
        "state": state,
    }
    return report


def evaluate_breakout_confirmation(
    *,
    bars: list[dict[str, Any]],
    trigger_price: float,
    hold_bars: int,
    volume_window_bars: int,
    opening_hour_bars: int,
    volume_pace_multiplier: float,
) -> dict[str, Any]:
    normalized = [bar for bar in bars if number(bar.get("c")) is not None and number(bar.get("v")) is not None]
    latest = normalized[-1] if normalized else {}
    latest_price = number(latest.get("c"))
    recent = normalized[-max(1, volume_window_bars) :]
    opening = normalized[: max(1, opening_hour_bars)]
    hold_slice = normalized[-max(1, hold_bars) :]
    hold_confirmed = len(hold_slice) >= hold_bars and all((number(bar.get("c")) or 0.0) >= trigger_price for bar in hold_slice)
    recent_volume_pace = average([number(bar.get("v")) or 0.0 for bar in recent])
    opening_volume_pace = average([number(bar.get("v")) or 0.0 for bar in opening])
    volume_confirmed = (
        len(opening) >= opening_hour_bars
        and len(recent) >= volume_window_bars
        and opening_volume_pace > 0
        and recent_volume_pace >= opening_volume_pace * volume_pace_multiplier
    )
    confirmed = bool(hold_confirmed and volume_confirmed)
    return {
        "confirmed": confirmed,
        "latest_price": latest_price,
        "latest_bar_time": latest.get("t"),
        "bar_count": len(normalized),
        "hold_confirmed": hold_confirmed,
        "hold_bars": hold_bars,
        "recent_closes": [number(bar.get("c")) for bar in hold_slice],
        "volume_confirmed": volume_confirmed,
        "recent_volume_pace": round(recent_volume_pace, 2),
        "opening_hour_volume_pace": round(opening_volume_pace, 2),
        "volume_pace_multiplier": volume_pace_multiplier,
        "required_recent_volume_pace": round(opening_volume_pace * volume_pace_multiplier, 2),
    }


def manage_existing_position(
    *,
    args: argparse.Namespace,
    position_qty: float,
    latest_price: float | None,
    state: dict[str, Any],
) -> tuple[str, str, dict[str, Any] | None]:
    if latest_price is None:
        return "hold_position", "missing_latest_price", None
    symbol = args.symbol.upper()
    sell_qty = round(float(position_qty), 6)
    if latest_price <= float(args.stop_price):
        return "sell_stop", "stop_price_hit", sell_payload(args=args, symbol=symbol, qty=sell_qty, reference_price=latest_price, tag="stop")
    if latest_price < float(args.invalidation_price):
        return "sell_invalidation", "rejected_back_below_invalidation", sell_payload(args=args, symbol=symbol, qty=sell_qty, reference_price=latest_price, tag="invalid")
    if latest_price >= float(args.target2_price):
        state["target1_taken"] = True
        return "sell_target2", "second_target_hit", sell_payload(args=args, symbol=symbol, qty=sell_qty, reference_price=latest_price, tag="target2")
    if latest_price >= float(args.target1_price) and not state.get("target1_taken"):
        partial_qty = round(max(0.000001, sell_qty / 2.0), 6)
        if partial_qty >= sell_qty:
            partial_qty = sell_qty
        state["target1_taken"] = True
        return "sell_target1_partial", "first_target_hit", sell_payload(args=args, symbol=symbol, qty=partial_qty, reference_price=latest_price, tag="target1")
    return "hold_position", "position_open_no_exit_trigger", None


def manage_existing_sell_orders(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    sell_orders: list[dict[str, Any]],
    position_qty: float,
    latest_price: float | None,
    execution_blocks: list[str],
    now: datetime,
) -> tuple[str, str, dict[str, Any] | None, list[dict[str, Any]]]:
    if latest_price is None:
        return "hold_position", "missing_latest_price", None, []
    profit_orders = [order for order in sell_orders if is_profit_target_order(order, target_price=float(args.target1_price))]
    protective_orders = [order for order in sell_orders if not is_profit_target_order(order, target_price=float(args.target1_price))]
    if profit_orders:
        if latest_price < float(args.target1_price):
            protective_payload = build_protective_order_payload(args=args, position_qty=position_qty, now=now)
            effects = cancel_orders_if_allowed(
                broker=broker,
                args=args,
                orders=profit_orders,
                execution_blocks=execution_blocks,
                label="cancel_profit_target_after_pullback",
            )
            if protective_payload:
                effects.extend(
                    submit_if_allowed(
                        broker=broker,
                        args=args,
                        payload=protective_payload,
                        execution_blocks=execution_blocks,
                        label="restore_protective_stop_after_profit_pullback",
                    )
                )
            return "restore_protection_after_profit_pullback", "profit_target_not_filled_and_price_pulled_back", protective_payload, effects
        return "hold_position", "profit_target_order_working", None, []
    if latest_price >= float(args.target1_price):
        profit_payload = build_profit_target_payload(args=args, position_qty=position_qty, now=now, target_price=float(args.target1_price))
        effects = cancel_orders_if_allowed(
            broker=broker,
            args=args,
            orders=protective_orders,
            execution_blocks=execution_blocks,
            label="cancel_protective_stop_for_profit_target",
        )
        effects.extend(
            submit_if_allowed(
                broker=broker,
                args=args,
                payload=profit_payload,
                execution_blocks=execution_blocks,
                label="submit_profit_target_sell",
            )
        )
        return "submit_profit_target_sell", "target1_hit_cancel_stop_and_sell_for_profit", profit_payload, effects
    return "hold_position", "protective_sell_order_already_exists", None, []


def build_protective_order_payload(*, args: argparse.Namespace, position_qty: float, now: datetime) -> dict[str, Any] | None:
    if str(args.protective_order_style) != "oco":
        return None
    qty = round(float(position_qty), 6)
    if qty <= 0:
        return None
    stop_price = round(float(args.invalidation_price), 2)
    stop_limit_price = round(max(0.01, stop_price * (1.0 - max(0.0, float(args.stop_limit_offset_pct)))), 2)
    return {
        "symbol": args.symbol.upper(),
        "side": "sell",
        "type": "limit",
        "qty": str(qty),
        "time_in_force": "day",
        "order_class": "oco",
        "take_profit": {
            "limit_price": str(round(float(args.target1_price), 2)),
        },
        "stop_loss": {
            "stop_price": str(stop_price),
            "limit_price": str(stop_limit_price),
        },
        "client_order_id": client_order_id("breakout-oco", args.symbol.upper(), now),
    }


def build_profit_target_payload(*, args: argparse.Namespace, position_qty: float, now: datetime, target_price: float) -> dict[str, Any]:
    return {
        "symbol": args.symbol.upper(),
        "side": "sell",
        "type": "limit",
        "qty": str(round(float(position_qty), 6)),
        "limit_price": str(round(float(target_price), 2)),
        "time_in_force": "day",
        "client_order_id": client_order_id("breakout-target1", args.symbol.upper(), now),
    }


def is_profit_target_order(order: dict[str, Any], *, target_price: float) -> bool:
    order_type = str(order.get("type") or order.get("order_type") or "").lower()
    stop_price = number(order.get("stop_price"))
    limit_price = number(order.get("limit_price"))
    client_id = str(order.get("client_order_id") or "").lower()
    if "target" in client_id or "profit" in client_id:
        return True
    return order_type == "limit" and stop_price is None and limit_price is not None and limit_price >= float(target_price)


def sell_payload(*, args: argparse.Namespace, symbol: str, qty: float, reference_price: float, tag: str) -> dict[str, Any]:
    limit_price = round(max(0.01, float(reference_price) * (1.0 - max(0.0, float(args.exit_limit_offset_pct)))), 2)
    return {
        "symbol": symbol,
        "side": "sell",
        "type": "limit",
        "qty": str(qty),
        "limit_price": str(limit_price),
        "time_in_force": "day",
        "client_order_id": client_order_id(f"breakout-{tag}", symbol, datetime.now(UTC)),
    }


def submit_if_allowed(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    payload: dict[str, Any],
    execution_blocks: list[str],
    label: str,
) -> list[dict[str, Any]]:
    if execution_blocks:
        return [{"action": f"{label}_blocked", "blocks": execution_blocks, "payload": payload}]
    if not args.execute_live_orders:
        return [{"action": f"would_{label}", "payload": payload}]
    try:
        return [{"action": f"submitted_{label}", "order": broker._request("POST", "/v2/orders", payload)}]
    except RuntimeError as exc:
        fallback = simple_stop_from_rejected_oco(payload) if label == "submit_protective_oco" else None
        if fallback and "fractional orders must be simple orders" in str(exc):
            try:
                return [
                    {"action": f"{label}_rejected", "error": str(exc), "payload": payload},
                    {"action": "submitted_fractional_simple_stop_fallback", "order": broker._request("POST", "/v2/orders", fallback), "payload": fallback},
                ]
            except RuntimeError as fallback_exc:
                return [
                    {"action": f"{label}_rejected", "error": str(exc), "payload": payload},
                    {"action": "fractional_simple_stop_fallback_rejected", "error": str(fallback_exc), "payload": fallback},
                ]
        return [{"action": f"{label}_rejected", "error": str(exc), "payload": payload}]


def cancel_orders_if_allowed(
    *,
    broker: AlpacaPaperBroker,
    args: argparse.Namespace,
    orders: list[dict[str, Any]],
    execution_blocks: list[str],
    label: str,
) -> list[dict[str, Any]]:
    if not orders:
        return []
    if execution_blocks:
        return [{"action": f"{label}_blocked", "blocks": execution_blocks, "order_ids": [order.get("id") for order in orders]}]
    if not args.execute_live_orders:
        return [{"action": f"would_{label}", "order_ids": [order.get("id") for order in orders]}]
    effects: list[dict[str, Any]] = []
    for order in orders:
        order_id = order.get("id")
        if not order_id:
            continue
        try:
            effects.append({"action": label, "order_id": order_id, "response": broker.cancel_order(str(order_id))})
        except RuntimeError as exc:
            effects.append({"action": f"{label}_rejected", "order_id": order_id, "error": str(exc)})
            break
    return effects


def simple_stop_from_rejected_oco(payload: dict[str, Any]) -> dict[str, Any] | None:
    stop_loss = payload.get("stop_loss") if isinstance(payload.get("stop_loss"), dict) else {}
    stop_price = stop_loss.get("stop_price")
    limit_price = stop_loss.get("limit_price")
    if not stop_price or not limit_price:
        return None
    return {
        "symbol": payload.get("symbol"),
        "side": payload.get("side", "sell"),
        "type": "stop_limit",
        "qty": payload.get("qty"),
        "time_in_force": "day",
        "stop_price": str(stop_price),
        "limit_price": str(limit_price),
        "client_order_id": str(payload.get("client_order_id") or "iwm-breakout-stop")[:40] + "-stp",
    }


def live_execution_blocks(*, args: argparse.Namespace, account: dict[str, Any], clock: dict[str, Any], asset: dict[str, Any] | None) -> list[str]:
    blocks: list[str] = []
    if "paper-api" in str(args.base_url or ""):
        blocks.append("refusing_live_monitor_with_paper_endpoint")
    if not bool(clock.get("is_open")):
        blocks.append("market_closed")
    if str(account.get("status") or "").upper() != "ACTIVE":
        blocks.append("account_not_active")
    if account.get("trading_blocked") or account.get("account_blocked") or account.get("trade_suspended_by_user"):
        blocks.append("account_trading_blocked")
    if asset is None:
        blocks.append("asset_not_found")
    else:
        if not asset.get("tradable"):
            blocks.append("asset_not_tradable")
        if not asset.get("fractionable"):
            blocks.append("asset_not_fractionable_for_small_account")
    if args.execute_live_orders and not args.confirm_live_order_risk:
        blocks.append("missing_confirm_live_order_risk")
    return blocks


def size_entry(
    *,
    buying_power: float | None,
    latest_price: float,
    max_notional: float | None,
    buying_power_fraction: float,
    min_notional: float,
) -> dict[str, Any]:
    available = max(0.0, float(buying_power or 0.0) * max(0.0, min(1.0, float(buying_power_fraction))))
    notional = min(available, float(max_notional)) if max_notional is not None else available
    qty = round(max(0.0, notional / max(float(latest_price), 1e-9)), 6)
    blocked = notional < float(min_notional) or qty <= 0
    return {
        "buying_power": buying_power,
        "buying_power_fraction": buying_power_fraction,
        "max_notional": max_notional,
        "planned_notional": round(notional, 2),
        "reference_price": latest_price,
        "qty": qty,
        "min_notional": min_notional,
        "blocked": blocked,
    }


def load_intraday_bars(broker: AlpacaPaperBroker, *, symbol: str, feed: str | None) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    start = (now - timedelta(hours=8)).isoformat().replace("+00:00", "Z")
    end = now.isoformat().replace("+00:00", "Z")
    return broker.stock_bars(symbol, start=start, end=end, timeframe="1Min", feed=feed, limit=1000)


def live_broker(args: argparse.Namespace) -> AlpacaPaperBroker:
    base_url = str(args.base_url or os.getenv("ALPACA_TRADING_BASE_URL_LIVE") or os.getenv("ALPACA_API_BASE_URL_LIVE") or LIVE_BASE_URL).rstrip("/")
    if "paper-api" in base_url:
        raise SystemExit("Refusing to run live breakout monitor against paper Alpaca endpoint.")
    key = (
        os.getenv("ALPACA_API_KEY_ID_LIVE")
        or os.getenv("APCA_API_KEY_ID_LIVE")
        or os.getenv("ALPACA_LIVE_API_KEY_ID")
    )
    secret = (
        os.getenv("ALPACA_API_SECRET_KEY_LIVE")
        or os.getenv("APCA_API_SECRET_KEY_LIVE")
        or os.getenv("ALPACA_LIVE_API_SECRET_KEY")
    )
    if not key or not secret:
        raise SystemExit("Live Alpaca credentials are required: ALPACA_API_KEY_ID_LIVE/ALPACA_API_SECRET_KEY_LIVE or ALPACA_LIVE_API_KEY_ID/ALPACA_LIVE_API_SECRET_KEY.")
    return AlpacaPaperBroker(base_url=base_url, key_id=key, secret_key=secret)


def safe_asset(broker: AlpacaPaperBroker, symbol: str) -> dict[str, Any] | None:
    try:
        return broker._request("GET", f"/v2/assets/{symbol.upper()}")
    except RuntimeError:
        return None


def console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report": report["mode"],
        "symbol": report["symbol"],
        "action": report["action"],
        "reason": report["reason"],
        "buying_power": report["account"]["buying_power"],
        "planned_notional": report["sizing"]["planned_notional"],
        "planned_qty": report["sizing"]["qty"],
        "latest_price": report["confirmation"]["latest_price"],
        "confirmed": report["confirmation"]["confirmed"],
        "execution_blocks": report["execution_blocks"],
        "side_effects": report["side_effects"],
    }


def client_order_id(prefix: str, symbol: str, now: datetime) -> str:
    return f"{prefix}-{symbol}-{now.strftime('%Y%m%d%H%M%S')}"[:48]


def read_state(path: Path) -> dict[str, Any]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
