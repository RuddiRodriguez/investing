#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/llm_options_trader_testnet}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8798}"

cd "$PROJECT_DIR"
mkdir -p "$STATE_DIR"

if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 1
fi

screen -S llm_options_dashboard -X quit 2>/dev/null || true
screen -dmS llm_options_dashboard zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.llm_options_trader.dashboard \
    --currency '$CURRENCY' \
    --state-dir '$STATE_DIR' \
    --refresh-seconds 15 \
    --port '$DASHBOARD_PORT' \
    >> '$STATE_DIR/dashboard.out' 2>&1"

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/?currency=$CURRENCY"
echo "Dashboard log: $PROJECT_DIR/$STATE_DIR/dashboard.out"
