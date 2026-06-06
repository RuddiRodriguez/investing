#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/llm_options_trader_live_shadow_hf}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8800}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
FORECAST_HOURS="${FORECAST_HOURS:-0.083333,0.166667,0.25,0.5}"
DATA_PROVIDER="${DATA_PROVIDER:-deribit}"
DATA_INTERVAL="${DATA_INTERVAL:-1m}"
LLM_PROVIDER="${LLM_PROVIDER:-huggingface}"
LLM_MODEL="${LLM_MODEL:-$(
  PYTHONPATH=automated_forecasting_engine/src "$PYTHON" - <<'PY'
from market_forecasting_engine.llm_model_catalog import DEFAULT_HUGGINGFACE_TRADER_MODEL
print(DEFAULT_HUGGINGFACE_TRADER_MODEL)
PY
)}"
MAX_ORDER_AMOUNT="${MAX_ORDER_AMOUNT:-10}"
MAX_ORDER_PRICE="${MAX_ORDER_PRICE:-5000}"
TRADER_PROFILE="${TRADER_PROFILE:-gambit}"
MEMORY_EVENTS="${MEMORY_EVENTS:-40}"

cd "$PROJECT_DIR"
mkdir -p "$STATE_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

if [[ -z "${HF_TOKEN:-${HUGGINGFACE_API_KEY:-}}" ]]; then
  echo "HF_TOKEN or HUGGINGFACE_API_KEY is required for Hugging Face hosted inference." >&2
  exit 2
fi

screen -S llm_options_live_shadow_hf_agent -X quit 2>/dev/null || true
screen -S llm_options_live_shadow_hf_dashboard -X quit 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*$STATE_DIR" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_HF_SHADOW | start simulated live-market trader"
screen -dmS llm_options_live_shadow_hf_agent zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.llm_options_trader.agent \
    --account-mode live \
    --simulation-only \
    --currency '$CURRENCY' \
    --instrument-currency '$INSTRUMENT_CURRENCY' \
    --data-provider '$DATA_PROVIDER' \
    --data-interval '$DATA_INTERVAL' \
    --lookback-days 20 \
    --forecast-hours '$FORECAST_HOURS' \
    --option-chain-limit 120 \
    --min-dte 1 \
    --max-dte 14 \
    --max-order-amount '$MAX_ORDER_AMOUNT' \
    --max-order-price '$MAX_ORDER_PRICE' \
    --enable-chronos-forecast \
    --chronos-model amazon/chronos-t5-tiny \
    --chronos-refresh-seconds 300 \
    --chronos-context-rows 512 \
    --chronos-num-samples 40 \
    --llm-provider '$LLM_PROVIDER' \
    --llm-model '$LLM_MODEL' \
    --trader-profile '$TRADER_PROFILE' \
    --memory-events '$MEMORY_EVENTS' \
    --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
    --output-dir '$STATE_DIR' \
    >> '$STATE_DIR/agent.out' 2>&1"

if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 1
fi

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_HF_SHADOW | start dashboard"
screen -dmS llm_options_live_shadow_hf_dashboard zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.llm_options_trader.dashboard \
    --currency '$CURRENCY' \
    --state-dir '$STATE_DIR' \
    --refresh-seconds 15 \
    --port '$DASHBOARD_PORT' \
    >> '$STATE_DIR/dashboard.out' 2>&1"

echo "Agent log: $PROJECT_DIR/$STATE_DIR/agent.out"
echo "Reports: $PROJECT_DIR/$STATE_DIR"
echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/?currency=$CURRENCY"
