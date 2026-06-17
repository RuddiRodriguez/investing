#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-automated_forecasting_engine/runs}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
MAX_NOTIONAL_PER_TRADE="${MAX_NOTIONAL_PER_TRADE:-2.20}"
EXECUTION_FLAGS=()

if [[ "${EXECUTE_LIVE_ORDERS:-0}" == "1" ]]; then
  EXECUTION_FLAGS+=(--execute-live-orders --confirm-live-order-risk)
fi

cd "$PROJECT_DIR"
mkdir -p "$RUNS_ROOT/live_alpaca_breakout_spy" "$RUNS_ROOT/live_alpaca_breakout_tlt"

screen -S spy_live_breakout_monitor -X quit 2>/dev/null || true
screen -S tlt_live_breakout_monitor -X quit 2>/dev/null || true

screen -dmS spy_live_breakout_monitor zsh -lc "cd '$PROJECT_DIR' && PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.live_trading.alpaca_breakout_monitor \
  --symbol SPY \
  --trigger-price 760.40 \
  --entry-limit-price 760.80 \
  --hold-bars 3 \
  --volume-window-bars 10 \
  --opening-hour-bars 60 \
  --volume-pace-multiplier 0 \
  --stop-price 736.00 \
  --target1-price 780.00 \
  --target2-price 780.00 \
  --invalidation-price 736.00 \
  --max-notional '$MAX_NOTIONAL_PER_TRADE' \
  --buying-power-fraction 0.95 \
  --min-notional 1.00 \
  --data-feed iex \
  --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
  --output-dir '$RUNS_ROOT/live_alpaca_breakout_spy' \
  ${EXECUTION_FLAGS[*]} >> '$RUNS_ROOT/live_alpaca_breakout_spy/monitor.out' 2>&1"

screen -dmS tlt_live_breakout_monitor zsh -lc "cd '$PROJECT_DIR' && PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.live_trading.alpaca_breakout_monitor \
  --symbol TLT \
  --trigger-price 86.05 \
  --entry-limit-price 86.10 \
  --hold-bars 3 \
  --volume-window-bars 10 \
  --opening-hour-bars 60 \
  --volume-pace-multiplier 0 \
  --stop-price 84.70 \
  --target1-price 87.70 \
  --target2-price 87.70 \
  --invalidation-price 84.70 \
  --max-notional '$MAX_NOTIONAL_PER_TRADE' \
  --buying-power-fraction 0.95 \
  --min-notional 1.00 \
  --data-feed iex \
  --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
  --output-dir '$RUNS_ROOT/live_alpaca_breakout_tlt' \
  ${EXECUTION_FLAGS[*]} >> '$RUNS_ROOT/live_alpaca_breakout_tlt/monitor.out' 2>&1"

echo "SPY/TLT live breakout monitors started."
echo "Live execution: ${EXECUTE_LIVE_ORDERS:-0}"
echo "Per-trade cap: $MAX_NOTIONAL_PER_TRADE"
echo "SPY report: $RUNS_ROOT/live_alpaca_breakout_spy/SPY_breakout_report.json"
echo "TLT report: $RUNS_ROOT/live_alpaca_breakout_tlt/TLT_breakout_report.json"
