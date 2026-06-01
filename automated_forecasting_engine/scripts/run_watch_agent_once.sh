#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
TICKER="${TICKER:-ASML}"
PROFILE="${PROFILE:-medium}"
HOLDING_STATUS="${HOLDING_STATUS:-not_owned}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/watch_agent_state}"
REFRESH_AFTER_HOURS="${REFRESH_AFTER_HOURS:-12}"
LLM_ENV_FILE="${LLM_ENV_FILE:-$PROJECT_DIR/.env}"
QUIET_UNCHANGED="${QUIET_UNCHANGED:-1}"
OPENAI_USAGE_PROCESS_NAME="${OPENAI_USAGE_PROCESS_NAME:-watch_agent}"

if [[ "$STATE_DIR" = /* ]]; then
  STATE_ROOT="$STATE_DIR"
else
  STATE_ROOT="$PROJECT_DIR/$STATE_DIR"
fi

mkdir -p "$STATE_ROOT/logs"
mkdir -p /private/tmp/mfe_mpl

cd "$PROJECT_DIR"

WATCH_ARGS=(
  --ticker "$TICKER"
  --profile "$PROFILE"
  --holding-status "$HOLDING_STATUS"
  --state-dir "$STATE_DIR"
  --refresh-after-hours "$REFRESH_AFTER_HOURS"
  --llm-env-file "$LLM_ENV_FILE"
  --once
)

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
