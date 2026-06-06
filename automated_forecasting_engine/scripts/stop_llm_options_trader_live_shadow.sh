#!/bin/zsh
set -euo pipefail

screen -S llm_options_live_shadow_agent -X quit 2>/dev/null || true
screen -S llm_options_live_shadow_dashboard -X quit 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*llm_options_trader_live_shadow" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.dashboard.*llm_options_trader_live_shadow" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_LIVE_SHADOW | stopped agent and dashboard"
