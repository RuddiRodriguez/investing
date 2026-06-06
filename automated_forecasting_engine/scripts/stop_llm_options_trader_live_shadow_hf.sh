#!/bin/zsh
set -euo pipefail

DASHBOARD_PORT="${DASHBOARD_PORT:-8800}"

screen -S llm_options_live_shadow_hf_agent -X quit 2>/dev/null || true
screen -S llm_options_live_shadow_hf_dashboard -X quit 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*llm_options_trader_live_shadow_hf" 2>/dev/null || true
if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
fi
echo "Stopped HF live-shadow LLM options trader."
