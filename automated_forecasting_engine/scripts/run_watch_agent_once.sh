#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
TICKER="${TICKER:-ASML}"
PROFILE="${PROFILE:-medium}"
HOLDING_STATUS="${HOLDING_STATUS:-not_owned}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/watch_agent_state}"
REFRESH_AFTER_HOURS="${REFRESH_AFTER_HOURS:-12}"
HORIZONS="${HORIZONS:-1,5,30}"
LLM_ENV_FILE="${LLM_ENV_FILE:-$PROJECT_DIR/.env}"
QUIET_UNCHANGED="${QUIET_UNCHANGED:-1}"
OPENAI_USAGE_PROCESS_NAME="${OPENAI_USAGE_PROCESS_NAME:-watch_agent}"
CALENDAR="${CALENDAR:-XNYS}"
LIVE_PRICE_PROVIDER="${LIVE_PRICE_PROVIDER:-auto}"
ENTRY_PRICE="${ENTRY_PRICE:-}"
QUANTITY="${QUANTITY:-}"
POSITION_VALUE="${POSITION_VALUE:-}"
ACCOUNT_EQUITY="${ACCOUNT_EQUITY:-}"
PORTFOLIO_NOTES="${PORTFOLIO_NOTES:-}"
PORTFOLIO_CONTEXT_JSON="${PORTFOLIO_CONTEXT_JSON:-}"
PORTFOLIO_CONTEXT_FILE="${PORTFOLIO_CONTEXT_FILE:-}"
STARTUP_DELAY_SECONDS="${STARTUP_DELAY_SECONDS:-0}"
ACTIVE_TIMEZONE="${ACTIVE_TIMEZONE:-Europe/Amsterdam}"
ACTIVE_START_LOCAL_TIME="${ACTIVE_START_LOCAL_TIME:-}"
STOP_AFTER_LOCAL_TIME="${STOP_AFTER_LOCAL_TIME:-}"
VALIDATION_WORKERS="${VALIDATION_WORKERS:-0}"
STRATEGY_KNOWLEDGE_DISABLED="${STRATEGY_KNOWLEDGE_DISABLED:-0}"
STRATEGY_KNOWLEDGE_CORPUS_DIR="${STRATEGY_KNOWLEDGE_CORPUS_DIR:-automated_forecasting_engine/strategy_knowledge/corpus}"
STRATEGY_KNOWLEDGE_INDEX="${STRATEGY_KNOWLEDGE_INDEX:-automated_forecasting_engine/strategy_knowledge/indexes/strategy_knowledge.faiss}"
STRATEGY_KNOWLEDGE_MAX_CHUNKS="${STRATEGY_KNOWLEDGE_MAX_CHUNKS:-8}"
STRATEGY_KNOWLEDGE_REBUILD_INDEX="${STRATEGY_KNOWLEDGE_REBUILD_INDEX:-0}"

if [[ "$STATE_DIR" = /* ]]; then
  STATE_ROOT="$STATE_DIR"
else
  STATE_ROOT="$PROJECT_DIR/$STATE_DIR"
fi

mkdir -p "$STATE_ROOT/logs"
mkdir -p /private/tmp/mfe_mpl

cd "$PROJECT_DIR"

local_minutes() {
  TZ="$ACTIVE_TIMEZONE" date '+%H:%M' | awk -F: '{print ($1 * 60) + $2}'
}

clock_minutes() {
  echo "$1" | awk -F: '{print ($1 * 60) + $2}'
}

if [[ -n "$ACTIVE_START_LOCAL_TIME" || -n "$STOP_AFTER_LOCAL_TIME" ]]; then
  NOW_MINUTES="$(local_minutes)"
  if [[ -n "$ACTIVE_START_LOCAL_TIME" ]]; then
    START_MINUTES="$(clock_minutes "$ACTIVE_START_LOCAL_TIME")"
    if (( NOW_MINUTES < START_MINUTES )); then
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LAUNCHD | skipped_before_active_window ticker=$TICKER timezone=$ACTIVE_TIMEZONE active_start=$ACTIVE_START_LOCAL_TIME"
      exit 0
    fi
  fi
  if [[ -n "$STOP_AFTER_LOCAL_TIME" ]]; then
    STOP_MINUTES="$(clock_minutes "$STOP_AFTER_LOCAL_TIME")"
    if (( NOW_MINUTES >= STOP_MINUTES )); then
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LAUNCHD | skipped_after_stop_time ticker=$TICKER timezone=$ACTIVE_TIMEZONE stop_after=$STOP_AFTER_LOCAL_TIME"
      exit 0
    fi
  fi
fi

if [[ "$STARTUP_DELAY_SECONDS" != "0" && "$STARTUP_DELAY_SECONDS" != "" ]]; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LAUNCHD | delayed_start ticker=$TICKER profile=$PROFILE delay_seconds=$STARTUP_DELAY_SECONDS"
  sleep "$STARTUP_DELAY_SECONDS"
fi

WATCH_ARGS=(
  --ticker "$TICKER"
  --profile "$PROFILE"
  --holding-status "$HOLDING_STATUS"
  --state-dir "$STATE_DIR"
  --refresh-after-hours "$REFRESH_AFTER_HOURS"
  --horizons "$HORIZONS"
  --llm-env-file "$LLM_ENV_FILE"
  --calendar "$CALENDAR"
  --live-price-provider "$LIVE_PRICE_PROVIDER"
  --validation-workers "$VALIDATION_WORKERS"
  --strategy-knowledge-corpus-dir "$STRATEGY_KNOWLEDGE_CORPUS_DIR"
  --strategy-knowledge-index "$STRATEGY_KNOWLEDGE_INDEX"
  --strategy-knowledge-max-chunks "$STRATEGY_KNOWLEDGE_MAX_CHUNKS"
  --once
)

if [[ "$STRATEGY_KNOWLEDGE_DISABLED" == "1" || "$STRATEGY_KNOWLEDGE_DISABLED" == "true" || "$STRATEGY_KNOWLEDGE_DISABLED" == "TRUE" ]]; then
  WATCH_ARGS+=(--disable-strategy-knowledge)
fi
if [[ "$STRATEGY_KNOWLEDGE_REBUILD_INDEX" == "1" || "$STRATEGY_KNOWLEDGE_REBUILD_INDEX" == "true" || "$STRATEGY_KNOWLEDGE_REBUILD_INDEX" == "TRUE" ]]; then
  WATCH_ARGS+=(--strategy-knowledge-rebuild-index)
fi

if [[ -n "$ENTRY_PRICE" ]]; then
  WATCH_ARGS+=(--entry-price "$ENTRY_PRICE")
fi
if [[ -n "$QUANTITY" ]]; then
  WATCH_ARGS+=(--quantity "$QUANTITY")
fi
if [[ -n "$POSITION_VALUE" ]]; then
  WATCH_ARGS+=(--position-value "$POSITION_VALUE")
fi
if [[ -n "$ACCOUNT_EQUITY" ]]; then
  WATCH_ARGS+=(--account-equity "$ACCOUNT_EQUITY")
fi
if [[ -n "$PORTFOLIO_NOTES" ]]; then
  WATCH_ARGS+=(--portfolio-notes "$PORTFOLIO_NOTES")
fi
if [[ -n "$PORTFOLIO_CONTEXT_JSON" ]]; then
  WATCH_ARGS+=(--portfolio-context-json "$PORTFOLIO_CONTEXT_JSON")
fi
if [[ -n "$PORTFOLIO_CONTEXT_FILE" ]]; then
  WATCH_ARGS+=(--portfolio-context-file "$PORTFOLIO_CONTEXT_FILE")
fi

if [[ "$QUIET_UNCHANGED" == "1" || "$QUIET_UNCHANGED" == "true" || "$QUIET_UNCHANGED" == "TRUE" ]]; then
  WATCH_ARGS+=(--quiet-unchanged)
else
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LAUNCHD | wake-up ticker=$TICKER profile=$PROFILE holding=$HOLDING_STATUS refresh_after_hours=$REFRESH_AFTER_HOURS"
fi

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=automated_forecasting_engine/src \
MPLCONFIGDIR=/private/tmp/mfe_mpl \
OPENAI_USAGE_PROCESS_NAME="$OPENAI_USAGE_PROCESS_NAME" \
"$PYTHON" -m market_forecasting_engine.watch_agent.cli "${WATCH_ARGS[@]}"

if [[ "$QUIET_UNCHANGED" != "1" && "$QUIET_UNCHANGED" != "true" && "$QUIET_UNCHANGED" != "TRUE" ]]; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LAUNCHD | completed ticker=$TICKER profile=$PROFILE"
fi
