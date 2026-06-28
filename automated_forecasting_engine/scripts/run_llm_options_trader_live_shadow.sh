#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
AWS_PROFILE="${AWS_PROFILE:-support-ML-QA}"
AWS_REGION="${AWS_REGION:-us-east-2}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/llm_options_trader_live_shadow}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
FORECAST_HOURS="${FORECAST_HOURS:-0.083333,0.166667,0.25,0.5,2,3}"
DATA_PROVIDER="${DATA_PROVIDER:-deribit}"
DATA_INTERVAL="${DATA_INTERVAL:-1m}"
MAX_ORDER_AMOUNT="${MAX_ORDER_AMOUNT:-10}"
MAX_ORDER_PRICE="${MAX_ORDER_PRICE:-5000}"
SHADOW_ACCOUNT_EQUITY="${SHADOW_ACCOUNT_EQUITY:-500}"
SHADOW_MAX_ENTRY_DEBIT="${SHADOW_MAX_ENTRY_DEBIT:-75}"
SHADOW_MAX_SESSION_DEBIT="${SHADOW_MAX_SESSION_DEBIT:-250}"
LLM_PROVIDER="${LLM_PROVIDER:-llm_studio}"
ENTRY_BIAS="${ENTRY_BIAS:-unrestricted}"
STRATEGY_MODE="${STRATEGY_MODE:-exploratory_trend_probe}"
DEFAULT_LLM_MODEL="$(
  LLM_PROVIDER="$LLM_PROVIDER" PYTHONPATH=automated_forecasting_engine/src "$PYTHON" - <<'PY'
import os
from market_forecasting_engine.llm_model_catalog import DEFAULT_FULL_LLM_OPTIONS_MODEL
from market_forecasting_engine.llm_trader.run import resolve_llm_provider, resolve_llm_model

provider = resolve_llm_provider(os.environ.get("LLM_PROVIDER"))
print(DEFAULT_FULL_LLM_OPTIONS_MODEL if provider == "llm_studio" else resolve_llm_model(None, provider=provider))
PY
)"
LLM_MODEL="${LLM_MODEL:-$DEFAULT_LLM_MODEL}"
TRADER_PROFILE="${TRADER_PROFILE:-gambit}"
MEMORY_EVENTS="${MEMORY_EVENTS:-40}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8799}"
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

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

export AWS_PROFILE
export AWS_REGION

screen -S llm_options_live_shadow_agent -X quit 2>/dev/null || true
screen -S llm_options_live_shadow_dashboard -X quit 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*$STATE_DIR" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_LIVE_SHADOW | start simulated live-market trader"
screen -dmS llm_options_live_shadow_agent zsh -lc "
  export AWS_PROFILE='$AWS_PROFILE' AWS_REGION='$AWS_REGION' && \
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
    --shadow-account-equity '$SHADOW_ACCOUNT_EQUITY' \
    --shadow-max-entry-debit '$SHADOW_MAX_ENTRY_DEBIT' \
    --shadow-max-session-debit '$SHADOW_MAX_SESSION_DEBIT' \
    $CHRONOS_FLAG \
    --chronos-model '$CHRONOS_MODEL' \
    --chronos-refresh-seconds '$CHRONOS_REFRESH_SECONDS' \
    --chronos-context-rows '$CHRONOS_CONTEXT_ROWS' \
    --chronos-num-samples '$CHRONOS_NUM_SAMPLES' \
    --llm-provider '$LLM_PROVIDER' \
    --llm-model '$LLM_MODEL' \
    --entry-bias '$ENTRY_BIAS' \
    --strategy-mode '$STRATEGY_MODE' \
    --trader-profile '$TRADER_PROFILE' \
    --memory-events '$MEMORY_EVENTS' \
    --check-interval-seconds '$CHECK_INTERVAL_SECONDS' \
    --output-dir '$STATE_DIR' \
    >> '$STATE_DIR/agent.out' 2>&1"

if [[ "$RUN_DASHBOARD" == "1" ]]; then
  if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
    sleep 1
  fi
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_LIVE_SHADOW | start dashboard"
  screen -dmS llm_options_live_shadow_dashboard zsh -lc "
    export AWS_PROFILE='$AWS_PROFILE' AWS_REGION='$AWS_REGION' && \
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
