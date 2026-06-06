#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/deribit_scalping_testnet}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8796}"
RUN_DASHBOARD="${RUN_DASHBOARD:-1}"
RUN_AGENT="${RUN_AGENT:-1}"

cd "$PROJECT_DIR"
mkdir -p "$STATE_DIR"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | clear prior stop request"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --account-mode testnet \
  --currency "$CURRENCY" \
  --instrument-currency "$INSTRUMENT_CURRENCY" \
  --output-dir "$STATE_DIR" \
  --clear-stop-request

if [[ "$RUN_AGENT" == "1" ]]; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | start agent screen=deribit_scalping_testnet_agent"
  screen -dmS deribit_scalping_testnet_agent zsh -lc "
    cd '$PROJECT_DIR' && \
    PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.deribit_options_agent \
      --account-mode testnet \
      --currency '$CURRENCY' \
      --instrument-currency '$INSTRUMENT_CURRENCY' \
      --risk-profile aggressive \
      --data-provider alpaca \
      --data-interval 1m \
      --lookback-days 7 \
      --forecast-hours 0.0417,0.0833,0.1667 \
      --check-interval-seconds 5 \
      --forecast-refresh-seconds 15 \
      --max-training-rows 1500 \
      --min-dte 1 \
      --max-dte 3 \
      --max-contracts 5 \
      --max-total-debit-usd 5000 \
      --max-position-equity-pct 0.80 \
      --max-spread-pct 0.20 \
      --target-delta 0.45 \
      --max-delta-distance 0.30 \
      --max-theta-edge-ratio 2.50 \
      --max-theta-premium-pct-per-day 2.00 \
      --limit-price-offset-pct 9.00 \
      --entry-order-policy limit \
      --exit-order-policy auto \
      --enable-chart-patterns \
      --no-block-chart-pattern-conflicts \
      --no-enable-fibonacci \
      --no-enable-market-regime-filter \
      --enable-impulse-entry \
      --impulse-lookback-bars 5 \
      --min-impulse-move-pct 0.0005 \
      --min-impulse-directional-bars 2 \
      --enable-late-entry-filter \
      --no-enable-late-entry-filter \
      --max-late-entry-move-pct 0.050 \
      --max-ema-extension-pct 0.050 \
      --exhaustion-reversal-bars 2 \
      --market-regime-lookback-rows 60 \
      --market-regime-breakout-buffer-pct 0.001 \
      --market-regime-middle-zone-width 0.12 \
      --min-trend-strength-pct 0.001 \
      --allow-range-edge-reversal-entry \
      --min-entry-expected-return-pct 0.0002 \
      --max-total-unrealized-profit-usd 8.00 \
      --total-profit-close-mode winning_only \
      --max-total-unrealized-loss-usd 12.00 \
      --total-loss-close-mode all \
      --max-position-unrealized-profit-usd 1.50 \
      --take-profit-position-pl-usd 1.00 \
      --profit-retrace-from-peak-pct 0.25 \
      --profit-close-limit-offset-pct 0.01 \
      --take-profit-pct 0.18 \
      --max-position-unrealized-loss-usd 3.00 \
      --stop-loss-pct 0.35 \
      --max-daily-realized-loss-usd 500 \
      --loss-cooldown-minutes 0 \
      --greeks-mode off \
      --max-open-option-orders 5 \
      --max-open-option-positions 5 \
      --max-open-option-contracts 25 \
      --max-open-option-premium-usd 5000 \
      --no-enable-feedback-loop \
      --feedback-min-matured 10 \
      --feedback-min-direction-accuracy 0.55 \
      --feedback-max-abs-pct-error 0.025 \
      --feedback-ledger-window 60 \
      --output-dir '$STATE_DIR' \
      --execute-paper-orders \
      >> '$STATE_DIR/scalping_agent.out' 2>&1"
fi

if [[ "$RUN_DASHBOARD" == "1" ]]; then
  if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | dashboard already listening port=$DASHBOARD_PORT"
  else
    echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_SCALPING_TESTNET | start dashboard screen=deribit_scalping_testnet_dashboard port=$DASHBOARD_PORT"
    screen -dmS deribit_scalping_testnet_dashboard zsh -lc "
      cd '$PROJECT_DIR' && \
      PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.deribit_options_dashboard \
        --currency '$CURRENCY' \
        --state-dir '$STATE_DIR' \
        --refresh-seconds 15 \
        --port '$DASHBOARD_PORT' \
        >> '$STATE_DIR/scalping_dashboard.out' 2>&1"
  fi
fi

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/?currency=$CURRENCY"
echo "Agent log: $PROJECT_DIR/$STATE_DIR/scalping_agent.out"
echo "Dashboard log: $PROJECT_DIR/$STATE_DIR/scalping_dashboard.out"
