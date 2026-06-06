#!/bin/zsh
set -euo pipefail

screen -S llm_options_entry_testnet -X quit 2>/dev/null || true
screen -S llm_options_exit_testnet -X quit 2>/dev/null || true
screen -S llm_options_agent_testnet -X quit 2>/dev/null || true
screen -S llm_options_dashboard -X quit 2>/dev/null || true

pkill -f "market_forecasting_engine.llm_options_trader.entry_agent" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.exit_agent" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.dashboard" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_TESTNET | stopped entry, exit, and dashboard processes"
