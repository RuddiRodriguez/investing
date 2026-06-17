#!/usr/bin/env bash
set -euo pipefail

cd /Users/ruddigarcia/Projects/invest

EXTRA_ARGS=()
if [[ "${FORCE_FORECAST_REFRESH_ON_START:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--force-forecast-refresh)
fi

PYTHONPATH=automated_forecasting_engine/src ./venv/bin/python -m market_forecasting_engine.live_trading.deribit_eth_usdc_daily_agent \
  --project-dir /Users/ruddigarcia/Projects/invest \
  --output-dir automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent \
  --llm-env-file /Users/ruddigarcia/Projects/invest/.env \
  --instrument ETH_USDC \
  --ticker ETH-USDC \
  --forecast-provider deribit \
  --forecast-interval 1h \
  --forecast-horizons 12,24,48 \
  --forecast-timeout-seconds 5400 \
  --daily-forecast-local-time 07:00 \
  --active-timezone Europe/Amsterdam \
  --check-interval-seconds 3600 \
  --max-notional-usdc 100 \
  --max-base-position 0.25 \
  --inventory-scope codex_only \
  "${EXTRA_ARGS[@]}"
