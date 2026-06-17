#!/usr/bin/env bash
set -euo pipefail

LABEL="com.marketforecasting.deribit.ethusdc.dailyagent"
PLIST="/Users/ruddigarcia/Library/LaunchAgents/${LABEL}.plist"

launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
pkill -f "market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent" || true
pkill -f "market_forecasting_engine.cli --ticker ETH-USDC" || true

echo "Stopped Deribit ETH/USDC daily forecast LaunchAgent if it was running."
