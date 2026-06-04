#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/deribit_options_agent}"
GRACE_SECONDS="${GRACE_SECONDS:-3}"

cd "$PROJECT_DIR"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | stop_request currency=$CURRENCY"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --currency "$CURRENCY" \
  --output-dir "$STATE_DIR" \
  --stop-agent

sleep "$GRACE_SECONDS"

AGENT_PIDS=("${(@f)$(pgrep -f "market_forecasting_engine.deribit_options_agent.*--currency $CURRENCY" || true)}")
if (( ${#AGENT_PIDS[@]} > 0 )); then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | stopping_local_agent_processes pids=${AGENT_PIDS[*]}"
  kill "${AGENT_PIDS[@]}" 2>/dev/null || true
  sleep 1
fi

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | liquidate_and_stop currency=$CURRENCY"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --currency "$CURRENCY" \
  --output-dir "$STATE_DIR" \
  --execute-paper-orders \
  --liquidate-and-stop

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | verify_flat currency=$CURRENCY"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" - <<PY
import json
from market_forecasting_engine.deribit_broker import DeribitTestnetBroker

currency = "$CURRENCY"
broker = DeribitTestnetBroker()
positions = [
    position
    for position in broker.positions(currency=currency, kind="option")
    if abs(float(position.get("size") or 0)) > 0
]
orders = broker.open_orders(currency=currency, kind="option")
payload = {
    "currency": currency,
    "open_position_count": len(positions),
    "open_order_count": len(orders),
    "positions": positions,
    "orders": orders,
}
print(json.dumps(payload, indent=2, default=str))
if positions or orders:
    raise SystemExit("Deribit cleanup did not finish flat; inspect positions/orders before restarting.")
PY

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | clear_stop_request currency=$CURRENCY"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --currency "$CURRENCY" \
  --output-dir "$STATE_DIR" \
  --clear-stop-request

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT | completed currency=$CURRENCY"
