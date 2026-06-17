#!/usr/bin/env bash
set -euo pipefail

cd /Users/ruddigarcia/Projects/invest

PYTHONPATH=automated_forecasting_engine/src ./venv/bin/streamlit run \
  automated_forecasting_engine/src/market_forecasting_engine/live_trading/deribit_streamlit_dashboard.py \
  --server.address 127.0.0.1 \
  --server.port "${DERIBIT_STREAMLIT_DASHBOARD_PORT:-8796}" \
  --server.headless true \
  -- \
  --report-path automated_forecasting_engine/runs/live_trading/deribit_live_account_report.json \
  --agent-report-path automated_forecasting_engine/runs/live_deribit_eth_usdc_daily_agent/ETH_USDC_daily_agent_report.json \
  --refresh-seconds 20
