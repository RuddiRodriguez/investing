#!/bin/zsh
set -euo pipefail

STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/llm_options_trader_testnet}"

screen -S llm_options_entry_testnet -X quit 2>/dev/null || true
screen -S llm_options_exit_testnet -X quit 2>/dev/null || true
screen -S llm_options_agent_testnet -X quit 2>/dev/null || true
screen -S llm_options_dashboard -X quit 2>/dev/null || true

pkill -f "market_forecasting_engine.llm_options_trader.entry_agent.*$STATE_DIR" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.exit_agent.*$STATE_DIR" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.agent.*$STATE_DIR" 2>/dev/null || true
pkill -f "market_forecasting_engine.llm_options_trader.dashboard.*$STATE_DIR" 2>/dev/null || true

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | LLM_OPTIONS_TESTNET | stopped entry, exit, and dashboard processes"
