#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/deribit_scalping_testnet}"

cd "$PROJECT_DIR"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | liquidate_and_stop currency=$CURRENCY instrument_currency=$INSTRUMENT_CURRENCY"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --account-mode testnet \
  --currency "$CURRENCY" \
  --instrument-currency "$INSTRUMENT_CURRENCY" \
  --output-dir "$STATE_DIR" \
  --execute-paper-orders \
  --liquidate-and-stop

screen -S deribit_scalping_testnet_agent -X quit 2>/dev/null || true
screen -S deribit_scalping_testnet_dashboard -X quit 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | clear_stop_request"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --account-mode testnet \
  --currency "$CURRENCY" \
  --instrument-currency "$INSTRUMENT_CURRENCY" \
  --output-dir "$STATE_DIR" \
  --clear-stop-request

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | completed"
