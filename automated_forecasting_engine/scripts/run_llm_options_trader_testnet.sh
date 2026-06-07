#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/llm_options_trader_testnet}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-300}"
MAX_ORDER_AMOUNT="${MAX_ORDER_AMOUNT:-10}"
MAX_ORDER_PRICE="${MAX_ORDER_PRICE:-5000}"
LLM_PROVIDER="${LLM_PROVIDER:-openai}"
DEFAULT_LLM_MODEL="$(
  PYTHONPATH=automated_forecasting_engine/src "$PYTHON" - <<'PY'
from market_forecasting_engine.openai_models import DEFAULT_OPENAI_MODEL
print(DEFAULT_OPENAI_MODEL)
PY
)"
LLM_MODEL="${LLM_MODEL:-$DEFAULT_LLM_MODEL}"
TRADER_PROFILE="${TRADER_PROFILE:-gambit}"
MEMORY_EVENTS="${MEMORY_EVENTS:-40}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8798}"
RUN_DASHBOARD="${RUN_DASHBOARD:-1}"
ENABLE_CHRONOS_FORECAST="${ENABLE_CHRONOS_FORECAST:-1}"
CHRONOS_MODEL="${CHRONOS_MODEL:-amazon/chronos-t5-tiny}"
CHRONOS_REFRESH_SECONDS="${CHRONOS_REFRESH_SECONDS:-300}"
CHRONOS_CONTEXT_ROWS="${CHRONOS_CONTEXT_ROWS:-512}"
CHRONOS_NUM_SAMPLES="${CHRONOS_NUM_SAMPLES:-40}"
if [[ "$ENABLE_CHRONOS_FORECAST" == "1" ]]; then
  CHRONOS_FLAG="--enable-chronos-forecast"
else
  CHRONOS_FLAG="--no-enable-chronos-forecast"
fi

cd "$PROJECT_DIR"
mkdir -p "$STATE_DIR"

screen -S llm_options_entry_testnet -X quit 2>/dev/null || true
screen -S llm_options_exit_testnet -X quit 2>/dev/null || true
screen -S llm_options_agent_testnet -X quit 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.entry_agent.*$STATE_DIR" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.exit_agent.*$STATE_DIR" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*$STATE_DIR" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_TESTNET | start combined LLM trader"
screen -dmS llm_options_agent_testnet zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.llm_options_trader.agent \
    --currency '$CURRENCY' \
    --instrument-currency '$INSTRUMENT_CURRENCY' \
    --data-provider alpaca \
    --data-interval 1m \
    --lookback-days 20 \
    --forecast-hours 0.25,0.5,1 \
    --option-chain-limit 120 \
    --min-dte 1 \
    --max-dte 14 \
    --max-order-amount '$MAX_ORDER_AMOUNT' \
    --max-order-price '$MAX_ORDER_PRICE' \
    $CHRONOS_FLAG \
    --chronos-model '$CHRONOS_MODEL' \
    --chronos-refresh-seconds '$CHRONOS_REFRESH_SECONDS' \
    --chronos-context-rows '$CHRONOS_CONTEXT_ROWS' \
    --chronos-num-samples '$CHRONOS_NUM_SAMPLES' \
    --llm-provider '$LLM_PROVIDER' \
    --llm-model '$LLM_MODEL' \
    --trader-profile '$TRADER_PROFILE' \
    --memory-events '$MEMORY_EVENTS' \
    --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
    --output-dir '$STATE_DIR' \
    --execute-testnet-orders \
    >> '$STATE_DIR/agent.out' 2>&1"

if [[ "$RUN_DASHBOARD" == "1" ]]; then
  if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
    sleep 1
  fi
  screen -S llm_options_dashboard -X quit 2>/dev/null || true
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_TESTNET | start dashboard"
  screen -dmS llm_options_dashboard zsh -lc "
    cd '$PROJECT_DIR' && \
    PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.llm_options_trader.dashboard \
      --currency '$CURRENCY' \
      --state-dir '$STATE_DIR' \
      --refresh-seconds 15 \
      --port '$DASHBOARD_PORT' \
      >> '$STATE_DIR/dashboard.out' 2>&1"
fi

echo "Agent log: $PROJECT_DIR/$STATE_DIR/agent.out"
echo "Reports: $PROJECT_DIR/$STATE_DIR"
echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/?currency=$CURRENCY"
