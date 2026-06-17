#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-automated_forecasting_engine/runs/live_alpaca_breakout_iwm}"
MAX_NOTIONAL="${MAX_NOTIONAL:-}"
EXECUTION_FLAGS=()

if [[ "${EXECUTE_LIVE_ORDERS:-0}" == "1" ]]; then
  EXECUTION_FLAGS+=(--execute-live-orders --confirm-live-order-risk)
fi
if [[ -n "$MAX_NOTIONAL" ]]; then
  EXECUTION_FLAGS+=(--max-notional "$MAX_NOTIONAL")
fi

cd "$PROJECT_DIR"
PYTHONPATH=automated_forecasting_engine/src "$PYTHON" -m market_forecasting_engine.live_trading.alpaca_breakout_monitor \
  --symbol IWM \
  --trigger-price 294.50 \
  --entry-limit-price 295.00 \
  --hold-bars 3 \
  --volume-window-bars 10 \
  --opening-hour-bars 60 \
  --volume-pace-multiplier 1.05 \
  --stop-price 286.00 \
  --target1-price 305.00 \
  --target2-price 312.00 \
  --invalidation-price 292.80 \
  --buying-power-fraction 0.95 \
  --min-notional 1.00 \
  --data-feed iex \
  --check-interval-seconds 60 \
  --output-dir "$OUTPUT_DIR" \
  "${EXECUTION_FLAGS[@]}" \
  "$@"
