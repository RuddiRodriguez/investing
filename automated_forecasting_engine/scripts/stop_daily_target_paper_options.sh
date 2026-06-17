#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
REPORT_DIR="${REPORT_DIR:-automated_forecasting_engine/runs/emergency_close_alpaca_paper_options}"

cd "$PROJECT_DIR"

if command -v screen >/dev/null 2>&1; then
  for session in daily_target_options_agent daily_target_options_dashboard keep_awake_trading; do
    screen -S "$session" -X quit 2>/dev/null || true
  done
fi

STOP_LOCAL_PROCESSES=1 \
CANCEL_OPTION_ORDERS=1 \
CLOSE_OPTION_POSITIONS=1 \
REPORT_DIR="$REPORT_DIR" \
automated_forecasting_engine/scripts/emergency_close_alpaca_paper_options.sh
