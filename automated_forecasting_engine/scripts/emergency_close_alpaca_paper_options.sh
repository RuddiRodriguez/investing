#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
RUN_DIR="${RUN_DIR:-automated_forecasting_engine/runs/emergency_close_alpaca_paper_options}"
REPORT_DIR="${REPORT_DIR:-$RUN_DIR}"
STOP_LOCAL_PROCESSES="${STOP_LOCAL_PROCESSES:-1}"
CANCEL_OPTION_ORDERS="${CANCEL_OPTION_ORDERS:-1}"
CLOSE_OPTION_POSITIONS="${CLOSE_OPTION_POSITIONS:-1}"
MAX_CLOSE_ATTEMPTS="${MAX_CLOSE_ATTEMPTS:-3}"
FILL_WAIT_SECONDS="${FILL_WAIT_SECONDS:-5}"
LIMIT_BID_OFFSET_PCT="${LIMIT_BID_OFFSET_PCT:-0.03}"
LIMIT_MARK_OFFSET_PCT="${LIMIT_MARK_OFFSET_PCT:-0.10}"
FINAL_LIMIT_MARK_OFFSET_PCT="${FINAL_LIMIT_MARK_OFFSET_PCT:-0.25}"

usage() {
  cat <<'EOF'
Emergency close for Alpaca PAPER options.

Default behavior:
  1. Stops local Alpaca paper-options agents/dashboards and keep-awake screens.
  2. Cancels open Alpaca paper option orders.
  3. Submits aggressive sell-to-close limit orders for open Alpaca paper option positions.
  4. Retries briefly and verifies paper options are flat.
  5. Writes an audit JSON report.

Run:
  automated_forecasting_engine/scripts/emergency_close_alpaca_paper_options.sh

Environment overrides:
  PROJECT_DIR=/Users/ruddigarcia/Projects/invest
  PYTHON=/Users/ruddigarcia/Projects/invest/venv/bin/python
  REPORT_DIR=automated_forecasting_engine/runs/emergency_close_alpaca_paper_options
  STOP_LOCAL_PROCESSES=1
  CANCEL_OPTION_ORDERS=1
  CLOSE_OPTION_POSITIONS=1
  MAX_CLOSE_ATTEMPTS=3
  FILL_WAIT_SECONDS=5
  LIMIT_BID_OFFSET_PCT=0.03
  LIMIT_MARK_OFFSET_PCT=0.10
  FINAL_LIMIT_MARK_OFFSET_PCT=0.25
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$PROJECT_DIR"
mkdir -p "$REPORT_DIR"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

echo "$(timestamp) | ALPACA_PAPER_OPTIONS_EMERGENCY_CLOSE | start"

if [[ "$STOP_LOCAL_PROCESSES" == "1" ]]; then
  echo "$(timestamp) | stopping local screen sessions"
  if command -v screen >/dev/null 2>&1; then
    for session in "${(@f)$(screen -ls 2>/dev/null | awk '/alpaca_options_|daily_target_options_|keep_awake_trading/ {print $1}' || true)}"; do
      [[ -n "$session" ]] || continue
      screen -S "$session" -X quit 2>/dev/null || true
    done
  fi

  echo "$(timestamp) | stopping local paper-options processes"
  PIDS=("${(@f)$(ps -axo pid,command | awk '
    /market_forecasting_engine\.paper_options_agent/ ||
    /market_forecasting_engine\.paper_options_daily_target_agent/ ||
    /market_forecasting_engine\.paper_options_dashboard/ ||
    /market_forecasting_engine\.paper_options_performance_dashboard/ ||
    /automated_forecasting_engine\/scripts\/keep_awake\.sh/ ||
    /caffeinate -dimsu/ {
        if ($0 !~ /awk/ && $0 !~ /emergency_close_alpaca_paper_options/) print $1
    }' || true)}")
  PIDS=("${(@)PIDS:#}")
  if (( ${#PIDS[@]} > 0 )); then
    echo "$(timestamp) | killing pids=${PIDS[*]}"
    kill "${PIDS[@]}" 2>/dev/null || true
    sleep 1
    SURVIVORS=("${(@f)$(ps -axo pid,command | awk '
      /market_forecasting_engine\.paper_options_agent/ ||
      /market_forecasting_engine\.paper_options_daily_target_agent/ ||
      /market_forecasting_engine\.paper_options_dashboard/ ||
      /market_forecasting_engine\.paper_options_performance_dashboard/ ||
      /automated_forecasting_engine\/scripts\/keep_awake\.sh/ ||
      /caffeinate -dimsu/ {
        if ($0 !~ /awk/ && $0 !~ /emergency_close_alpaca_paper_options/) print $1
      }' || true)}")
    SURVIVORS=("${(@)SURVIVORS:#}")
    if (( ${#SURVIVORS[@]} > 0 )); then
      echo "$(timestamp) | force killing survivors=${SURVIVORS[*]}"
      kill -9 "${SURVIVORS[@]}" 2>/dev/null || true
    fi
  fi
fi

echo "$(timestamp) | broker cleanup"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" - "$REPORT_DIR" "$CANCEL_OPTION_ORDERS" "$CLOSE_OPTION_POSITIONS" "$MAX_CLOSE_ATTEMPTS" "$FILL_WAIT_SECONDS" "$LIMIT_BID_OFFSET_PCT" "$LIMIT_MARK_OFFSET_PCT" "$FINAL_LIMIT_MARK_OFFSET_PCT" <<'PY'
from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_forecasting_engine.alpaca_broker import AlpacaPaperBroker


report_dir = Path(sys.argv[1])
cancel_option_orders = sys.argv[2] == "1"
close_option_positions = sys.argv[3] == "1"
max_close_attempts = max(1, int(float(sys.argv[4])))
fill_wait_seconds = max(1.0, float(sys.argv[5]))
limit_bid_offset_pct = max(0.0, float(sys.argv[6]))
limit_mark_offset_pct = max(0.0, float(sys.argv[7]))
final_limit_mark_offset_pct = max(0.0, float(sys.argv[8]))


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def is_option_order(order: dict[str, Any]) -> bool:
    if str(order.get("asset_class") or "").lower() == "us_option":
        return True
    symbol = str(order.get("symbol") or "")
    # Alpaca option symbols are compact OCC-style symbols like TSLA260612C00385000.
    return len(symbol) >= 15 and symbol[-9:-8] in {"C", "P"} and symbol[-8:].isdigit()


def option_positions(broker: AlpacaPaperBroker) -> list[dict[str, Any]]:
    return [
        position
        for position in broker.positions()
        if str(position.get("asset_class") or "").lower() == "us_option"
        and abs(as_float(position.get("qty"))) > 0
    ]


def option_open_orders(broker: AlpacaPaperBroker) -> list[dict[str, Any]]:
    return [order for order in broker.orders(status="open", limit=500) if is_option_order(order)]


def cancel_orders(broker: AlpacaPaperBroker, orders: list[dict[str, Any]], report: dict[str, Any], label: str) -> None:
    for order in orders:
        order_id = order.get("id")
        if not order_id:
            continue
        try:
            result = broker.cancel_order(str(order_id))
            report["cancelled_orders"].append(
                {
                    "label": label,
                    "id": order_id,
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "type": order.get("type"),
                    "status": order.get("status"),
                    "result": result,
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit every broker exception
            report["cancel_errors"].append(
                {
                    "label": label,
                    "id": order_id,
                    "symbol": order.get("symbol"),
                    "error": str(exc),
                }
            )


def snapshot_prices(broker: AlpacaPaperBroker, positions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    symbols = [str(position.get("symbol") or "") for position in positions if position.get("symbol")]
    return broker.option_snapshots(symbols) if symbols else {}


def close_limit_price(position: dict[str, Any], snapshot: dict[str, Any], attempt: int) -> tuple[float | None, dict[str, Any]]:
    quote = snapshot.get("latestQuote") or {}
    trade = snapshot.get("latestTrade") or {}
    minute = snapshot.get("minuteBar") or {}
    bid = as_float(quote.get("bp"))
    ask = as_float(quote.get("ap"))
    current = as_float(position.get("current_price")) or as_float(trade.get("p")) or as_float(minute.get("c"))
    if attempt < max_close_attempts:
        if bid > 0:
            price = max(0.01, round(bid * (1.0 - limit_bid_offset_pct * attempt), 2))
        elif current > 0:
            price = max(0.01, round(current * (1.0 - limit_mark_offset_pct * attempt), 2))
        else:
            return None, {"bid": bid, "ask": ask, "current": current, "reason": "no_bid_or_mark"}
    elif current > 0:
        price = max(0.01, round(current * (1.0 - final_limit_mark_offset_pct), 2))
    elif bid > 0:
        price = max(0.01, round(bid * 0.75, 2))
    else:
        return None, {"bid": bid, "ask": ask, "current": current, "reason": "no_final_price"}
    return price, {"bid": bid, "ask": ask, "current": current}


def close_positions_once(
    broker: AlpacaPaperBroker,
    positions: list[dict[str, Any]],
    report: dict[str, Any],
    attempt: int,
) -> None:
    snapshots = snapshot_prices(broker, positions)
    for position in positions:
        symbol = str(position.get("symbol") or "")
        qty = abs(as_float(position.get("qty")))
        if not symbol or qty <= 0:
            continue
        price, price_context = close_limit_price(position, snapshots.get(symbol) or {}, attempt)
        if price is None:
            report["close_errors"].append(
                {"attempt": attempt, "symbol": symbol, "qty": qty, "error": "no executable close limit", **price_context}
            )
            continue
        try:
            order = broker.submit_order(
                symbol=symbol,
                side="sell",
                order_type="limit",
                qty=qty,
                limit_price=price,
                time_in_force="day",
                client_order_id=f"codex-close-{symbol[-16:]}-{datetime.now(UTC).strftime('%H%M%S%f')}"[:48],
            )
            report["close_orders"].append(
                {
                    "attempt": attempt,
                    "symbol": symbol,
                    "qty": qty,
                    "limit_price": price,
                    **price_context,
                    "order": order,
                }
            )
        except Exception as exc:  # noqa: BLE001 - audit every broker exception
            report["close_errors"].append(
                {"attempt": attempt, "symbol": symbol, "qty": qty, "limit_price": price, "error": str(exc), **price_context}
            )


report_dir.mkdir(parents=True, exist_ok=True)
report_path = report_dir / f"alpaca_paper_options_emergency_close_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
broker = AlpacaPaperBroker()
report: dict[str, Any] = {
    "started_at": now_iso(),
    "mode": "alpaca_paper_options_only",
    "config": {
        "cancel_option_orders": cancel_option_orders,
        "close_option_positions": close_option_positions,
        "max_close_attempts": max_close_attempts,
        "fill_wait_seconds": fill_wait_seconds,
        "limit_bid_offset_pct": limit_bid_offset_pct,
        "limit_mark_offset_pct": limit_mark_offset_pct,
        "final_limit_mark_offset_pct": final_limit_mark_offset_pct,
    },
    "open_orders_before": [],
    "positions_before": [],
    "cancelled_orders": [],
    "cancel_errors": [],
    "close_orders": [],
    "close_errors": [],
    "attempts": [],
    "open_orders_after": [],
    "positions_after": [],
}

report["open_orders_before"] = option_open_orders(broker)
report["positions_before"] = option_positions(broker)

if cancel_option_orders:
    cancel_orders(broker, report["open_orders_before"], report, "initial_option_order_cancel")
    time.sleep(1)

if close_option_positions:
    for attempt in range(1, max_close_attempts + 1):
        remaining_positions = option_positions(broker)
        remaining_orders = option_open_orders(broker)
        report["attempts"].append(
            {
                "attempt": attempt,
                "remaining_position_count_before": len(remaining_positions),
                "remaining_order_count_before": len(remaining_orders),
                "checked_at": now_iso(),
            }
        )
        if not remaining_positions:
            break
        if remaining_orders:
            cancel_orders(broker, remaining_orders, report, f"pre_attempt_{attempt}_option_order_cancel")
            time.sleep(1)
            remaining_positions = option_positions(broker)
        close_positions_once(broker, remaining_positions, report, attempt)
        time.sleep(fill_wait_seconds)

report["positions_after"] = option_positions(broker)
report["open_orders_after"] = option_open_orders(broker)
report["finished_at"] = now_iso()
report["flat"] = not report["positions_after"] and not report["open_orders_after"]
report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

summary = {
    "report": str(report_path),
    "open_orders_before": len(report["open_orders_before"]),
    "positions_before": len(report["positions_before"]),
    "cancelled_orders": len(report["cancelled_orders"]),
    "cancel_errors": len(report["cancel_errors"]),
    "close_orders": len(report["close_orders"]),
    "close_errors": len(report["close_errors"]),
    "open_orders_after": len(report["open_orders_after"]),
    "positions_after": len(report["positions_after"]),
    "flat": report["flat"],
}
print(json.dumps(summary, indent=2))
if not report["flat"]:
    raise SystemExit("Emergency close did not finish flat; inspect the report and broker UI before restarting.")
PY

echo "$(timestamp) | ALPACA_PAPER_OPTIONS_EMERGENCY_CLOSE | completed"
