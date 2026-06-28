from __future__ import annotations

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.deribit_broker import DeribitLiveSpotBroker


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor a Deribit spot buy order and submit protection after fill.")
    parser.add_argument("--instrument", default="ETH_USDC")
    parser.add_argument("--buy-order-id", required=True)
    parser.add_argument("--expected-label", default=None)
    parser.add_argument("--trigger-price", type=float, required=True)
    parser.add_argument("--limit-price", type=float, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-seconds", type=int, default=172800)
    parser.add_argument("--log-dir", default="automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent/manual_orders")
    args = parser.parse_args()

    broker = DeribitLiveSpotBroker()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / f"protect-monitor-{args.buy_order_id}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    state_path = log_dir / f"protect-monitor-{args.buy_order_id}.state.json"
    started = time.monotonic()
    protection_submitted = False

    while True:
        now = datetime.now(UTC)
        try:
            order = broker.private_get("get_order_state", {"order_id": args.buy_order_id})
            if not isinstance(order, dict):
                order = {"raw": order}
        except Exception as exc:
            append_event(event_log, {"checked_at_utc": now.isoformat(), "status": "order_state_error", "error": str(exc)})
            time.sleep(max(5, int(args.poll_seconds)))
            continue

        state = str(order.get("order_state") or "").lower()
        filled_amount = finite_float(order.get("filled_amount"))
        amount = finite_float(order.get("amount"))
        label = str(order.get("label") or "")
        append_event(
            event_log,
            {
                "checked_at_utc": now.isoformat(),
                "status": "checked",
                "buy_order_id": args.buy_order_id,
                "order_state": state,
                "amount": amount,
                "filled_amount": filled_amount,
                "label": label,
                "protection_submitted": protection_submitted,
            },
        )
        state_path.write_text(
            json.dumps(
                {
                    "updated_at_utc": now.isoformat(),
                    "buy_order_id": args.buy_order_id,
                    "order_state": state,
                    "amount": amount,
                    "filled_amount": filled_amount,
                    "protection_submitted": protection_submitted,
                    "event_log": str(event_log),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if args.expected_label and label and label != args.expected_label:
            append_event(event_log, {"checked_at_utc": now.isoformat(), "status": "label_mismatch_abort", "actual_label": label})
            raise SystemExit(f"Order label mismatch: {label}")
        if state in {"cancelled", "rejected"}:
            append_event(event_log, {"checked_at_utc": now.isoformat(), "status": "buy_order_not_live_exit", "order_state": state})
            return
        if state == "filled" and filled_amount > 0 and not protection_submitted:
            details = broker.public_get("get_instrument", {"instrument_name": args.instrument})
            step = finite_float(details.get("contract_size")) or 0.000001
            min_amount = finite_float(details.get("min_trade_amount"))
            protect_amount = math.floor(filled_amount / step) * step
            protect_amount = round(protect_amount, 8)
            if protect_amount < min_amount:
                append_event(
                    event_log,
                    {
                        "checked_at_utc": now.isoformat(),
                        "status": "filled_but_below_min_protection",
                        "filled_amount": filled_amount,
                        "min_amount": min_amount,
                    },
                )
                return
            stop_label = f"codex-auto-protect-{args.instrument.lower()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
            result = broker.submit_spot_order(
                side="sell",
                instrument_name=args.instrument,
                amount=protect_amount,
                order_type="stop_limit",
                price=args.limit_price,
                trigger_price=args.trigger_price,
                trigger="index_price",
                label=stop_label,
            )
            protection_submitted = True
            append_event(
                event_log,
                {
                    "checked_at_utc": datetime.now(UTC).isoformat(),
                    "status": "protection_submitted",
                    "buy_order_id": args.buy_order_id,
                    "stop_label": stop_label,
                    "amount": protect_amount,
                    "trigger_price": args.trigger_price,
                    "limit_price": args.limit_price,
                    "broker_result": result,
                },
            )
            return

        if time.monotonic() - started > max(60, int(args.max_seconds)):
            append_event(event_log, {"checked_at_utc": now.isoformat(), "status": "timeout_exit"})
            return
        time.sleep(max(5, int(args.poll_seconds)))


def append_event(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


if __name__ == "__main__":
    main()
