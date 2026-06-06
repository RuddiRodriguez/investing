#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
CURRENCY="${CURRENCY:-ETH}"
INSTRUMENT_CURRENCY="${INSTRUMENT_CURRENCY:-USDC}"
STATE_DIR="${STATE_DIR:-automated_forecasting_engine/runs/deribit_live_strategy_testnet}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8796}"

cd "$PROJECT_DIR"
mkdir -p "$STATE_DIR"

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_LIVE_STRATEGY_TESTNET | clear prior stop request"
PYTHONPATH=automated_forecasting_engine/src \
"$PYTHON" -m market_forecasting_engine.deribit_options_agent \
  --account-mode testnet \
  --currency "$CURRENCY" \
  --instrument-currency "$INSTRUMENT_CURRENCY" \
  --output-dir "$STATE_DIR" \
  --clear-stop-request

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_LIVE_STRATEGY_TESTNET | start agent screen=deribit_live_strategy_testnet_agent"
screen -dmS deribit_live_strategy_testnet_agent zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.deribit_options_agent \
    --account-mode testnet \
    --currency '$CURRENCY' \
    --instrument-currency '$INSTRUMENT_CURRENCY' \
    --risk-profile aggressive \
    --data-provider alpaca \
    --data-interval 1m \
    --lookback-days 20 \
    --forecast-hours 0.25,0.5,1 \
    --check-interval-seconds 60 \
    --forecast-refresh-seconds 300 \
    --max-training-rows 3500 \
    --min-dte 1 \
    --max-dte 7 \
    --max-total-debit-usd 5 \
    --max-position-equity-pct 0.75 \
    --risk-budget-pct 0.0075 \
    --max-spread-pct 0.18 \
    --max-contracts 0.1 \
    --target-delta 0.45 \
    --max-delta-distance 0.22 \
    --greeks-mode required \
    --max-theta-edge-ratio 0.55 \
    --max-theta-premium-pct-per-day 0.25 \
    --no-enable-fibonacci \
    --enable-chart-patterns \
    --block-chart-pattern-conflicts \
    --enable-market-regime-filter \
    --market-regime-lookback-rows 120 \
    --market-regime-breakout-buffer-pct 0.0015 \
    --market-regime-middle-zone-width 0.22 \
    --min-trend-strength-pct 0.0045 \
    --enable-impulse-entry \
    --impulse-lookback-bars 8 \
    --min-impulse-move-pct 0.005 \
    --min-impulse-directional-bars 6 \
    --enable-late-entry-filter \
    --max-late-entry-move-pct 0.014 \
    --max-ema-extension-pct 0.008 \
    --exhaustion-reversal-bars 2 \
    --entry-order-policy limit \
    --exit-order-policy auto \
    --limit-price-offset-pct 0.03 \
    --stop-loss-pct 0.18 \
    --take-profit-pct 0.12 \
    --max-total-unrealized-loss-usd 2.5 \
    --total-loss-close-mode all \
    --max-total-unrealized-profit-usd 1.2 \
    --total-profit-close-mode winning_only \
    --max-position-unrealized-loss-usd 1.2 \
    --max-position-unrealized-profit-usd 0.8 \
    --take-profit-position-pl-usd 0.7 \
    --profit-retrace-from-peak-pct 0.15 \
    --profit-close-limit-offset-pct 0.005 \
    --enable-forecast-reversal-exit \
    --min-reversal-edge-pct 0.001 \
    --close-before-expiry-hours 12 \
    --expiry-warning-hours 24 \
    --liquidation-limit-offset-pct 0.05 \
    --max-open-option-orders 1 \
    --max-open-option-positions 1 \
    --max-open-option-contracts 0.1 \
    --max-open-option-premium-usd 5 \
    --max-daily-realized-loss-usd 18 \
    --loss-cooldown-minutes 20 \
    --min-entry-expected-return-pct 0.008 \
    --enable-feedback-loop \
    --feedback-min-matured 5 \
    --feedback-min-direction-accuracy 0.55 \
    --feedback-max-abs-pct-error 0.04 \
    --feedback-ledger-window 30 \
    --output-dir '$STATE_DIR' \
    --execute-paper-orders \
    >> '$STATE_DIR/live_strategy_testnet_agent.out' 2>&1"

if lsof -nP -iTCP:"$DASHBOARD_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_LIVE_STRATEGY_TESTNET | stopping existing dashboard listener on port=$DASHBOARD_PORT"
  lsof -tiTCP:"$DASHBOARD_PORT" -sTCP:LISTEN | xargs kill 2>/dev/null || true
  sleep 1
fi

echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') | DERIBIT_LIVE_STRATEGY_TESTNET | start dashboard screen=deribit_live_strategy_testnet_dashboard port=$DASHBOARD_PORT"
screen -dmS deribit_live_strategy_testnet_dashboard zsh -lc "
  cd '$PROJECT_DIR' && \
  PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.deribit_options_dashboard \
    --currency '$CURRENCY' \
    --state-dir '$STATE_DIR' \
    --refresh-seconds 30 \
    --port '$DASHBOARD_PORT' \
    >> '$STATE_DIR/live_strategy_testnet_dashboard.out' 2>&1"

echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/?currency=$CURRENCY"
echo "Agent log: $PROJECT_DIR/$STATE_DIR/live_strategy_testnet_agent.out"
echo "Dashboard log: $PROJECT_DIR/$STATE_DIR/live_strategy_testnet_dashboard.out"
