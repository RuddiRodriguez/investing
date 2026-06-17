#!/usr/bin/env zsh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-$PROJECT_DIR/venv/bin/python}"
RUNS_ROOT="${RUNS_ROOT:-automated_forecasting_engine/runs}"
TICKERS="${TICKERS:-SOFI,HOOD,BAC,F,AMD,PLTR}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8806}"
MAX_UNDERLYING_PRICE="${MAX_UNDERLYING_PRICE:-100}"
DAILY_PROFIT_TARGET="${DAILY_PROFIT_TARGET:-45}"
DAILY_LOSS_LIMIT="${DAILY_LOSS_LIMIT:-30}"
MAX_TOTAL_TRADES="${MAX_TOTAL_TRADES:-4}"
MAX_ACTIVE_TICKERS="${MAX_ACTIVE_TICKERS:-1}"
MAX_TOTAL_DEBIT="${MAX_TOTAL_DEBIT:-100}"
MAX_CONTRACTS="${MAX_CONTRACTS:-1}"
OPTION_STRATEGY_MODE="${OPTION_STRATEGY_MODE:-auto}"
ENABLE_MULTI_LEG="${ENABLE_MULTI_LEG:-1}"
ENABLE_SHORT_OPTION_STRATEGIES="${ENABLE_SHORT_OPTION_STRATEGIES:-1}"
MAX_LEGS="${MAX_LEGS:-4}"
MAX_OPEN_OPTION_CONTRACTS="${MAX_OPEN_OPTION_CONTRACTS:-4}"
IRON_BUTTERFLY_WING_WIDTH_PCT="${IRON_BUTTERFLY_WING_WIDTH_PCT:-0.05}"
CALENDAR_NEAR_MIN_DTE="${CALENDAR_NEAR_MIN_DTE:-1}"
CALENDAR_NEAR_MAX_DTE="${CALENDAR_NEAR_MAX_DTE:-7}"
CALENDAR_FAR_MIN_DTE="${CALENDAR_FAR_MIN_DTE:-8}"
CALENDAR_FAR_MAX_DTE="${CALENDAR_FAR_MAX_DTE:-35}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"
FORECAST_REFRESH_SECONDS="${FORECAST_REFRESH_SECONDS:-300}"
SELECTION_REFRESH_SECONDS="${SELECTION_REFRESH_SECONDS:-300}"
FORECAST_ENGINE="${FORECAST_ENGINE:-advanced}"
SELECTION_METRIC="${SELECTION_METRIC:-mae}"
SEARCH_LEVEL="${SEARCH_LEVEL:-realtime}"

cd "$PROJECT_DIR"
mkdir -p "$RUNS_ROOT"

stop_screen() {
  local name="$1"
  if command -v screen >/dev/null 2>&1; then
    screen -S "$name" -X quit 2>/dev/null || true
  fi
}

stop_screen daily_target_options_agent
stop_screen daily_target_options_dashboard
stop_screen keep_awake_trading
pkill -f "market_forecasting_engine.paper_options_daily_target_agent .*--output-root" 2>/dev/null || true
pkill -f "market_forecasting_engine.paper_options_performance_dashboard .*--port ${DASHBOARD_PORT}" 2>/dev/null || true
pkill -f "automated_forecasting_engine/scripts/keep_awake.sh" 2>/dev/null || true

screen -dmS keep_awake_trading zsh -lc "cd '$PROJECT_DIR' && automated_forecasting_engine/scripts/keep_awake.sh >> '$RUNS_ROOT/keep_awake.out' 2>&1"

MULTI_LEG_FLAGS=""
if [[ "$ENABLE_MULTI_LEG" == "1" ]]; then
  SHORT_STRATEGY_FLAG=""
  if [[ "$ENABLE_SHORT_OPTION_STRATEGIES" == "1" ]]; then
    SHORT_STRATEGY_FLAG="--enable-short-option-strategies"
  fi
  MULTI_LEG_FLAGS="--enable-multi-leg $SHORT_STRATEGY_FLAG --option-strategy-mode '$OPTION_STRATEGY_MODE' --max-legs '$MAX_LEGS' --straddle-max-debit-multiplier 1.0 --iron-butterfly-wing-width-pct '$IRON_BUTTERFLY_WING_WIDTH_PCT' --calendar-near-min-dte '$CALENDAR_NEAR_MIN_DTE' --calendar-near-max-dte '$CALENDAR_NEAR_MAX_DTE' --calendar-far-min-dte '$CALENDAR_FAR_MIN_DTE' --calendar-far-max-dte '$CALENDAR_FAR_MAX_DTE'"
else
  MULTI_LEG_FLAGS="--no-enable-multi-leg --option-strategy-mode directional"
fi

screen -dmS daily_target_options_agent zsh -lc "cd '$PROJECT_DIR' && PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.paper_options_daily_target_agent --candidate-tickers '$TICKERS' --max-underlying-price '$MAX_UNDERLYING_PRICE' --risk-profile aggressive --provider alpaca --alpaca-data-feed iex --interval 1m --lookback-days 20 --forecast-hours 0.25,0.5,1 --forecast-engine '$FORECAST_ENGINE' --selection-metric '$SELECTION_METRIC' --search-level '$SEARCH_LEVEL' --check-interval-seconds '$CHECK_INTERVAL_SECONDS' --forecast-refresh-seconds '$FORECAST_REFRESH_SECONDS' --selection-refresh-seconds '$SELECTION_REFRESH_SECONDS' --max-training-rows 3500 --min-dte 1 --max-dte 7 --max-total-debit '$MAX_TOTAL_DEBIT' --max-contracts '$MAX_CONTRACTS' --max-active-tickers '$MAX_ACTIVE_TICKERS' --daily-profit-target '$DAILY_PROFIT_TARGET' --daily-loss-limit '$DAILY_LOSS_LIMIT' --max-total-trades '$MAX_TOTAL_TRADES' --take-profit-position-pl 12 --max-position-unrealized-loss 12 --max-total-unrealized-profit 18 --max-total-unrealized-loss 15 --profit-retrace-from-peak-pct 0.18 --profit-close-limit-offset-pct 0.01 --take-profit-pct 0.12 --stop-loss-pct 0.12 --entry-order-policy limit --exit-order-policy auto $MULTI_LEG_FLAGS --require-greeks --enable-market-regime-filter --enable-impulse-entry --max-open-option-positions 1 --max-open-option-contracts '$MAX_OPEN_OPTION_CONTRACTS' --max-open-option-orders 1 --max-trades-per-day 2 --entry-cooldown-minutes 5 --loss-cooldown-minutes 10 --one-trade-per-forecast --output-root '$RUNS_ROOT' --execute-paper-orders >> '$RUNS_ROOT/daily_target_options_agent.out' 2>&1"

screen -dmS daily_target_options_dashboard zsh -lc "cd '$PROJECT_DIR' && PYTHONPATH=automated_forecasting_engine/src '$PYTHON' -m market_forecasting_engine.paper_options_performance_dashboard --runs-root '$RUNS_ROOT' --tickers '$TICKERS' --run-prefixes alpaca_daily_target_options_ --refresh-seconds 30 --port '$DASHBOARD_PORT' >> '$RUNS_ROOT/daily_target_options_dashboard.out' 2>&1"

echo "Daily target paper options flow started."
echo "Agent screen: daily_target_options_agent"
echo "Dashboard screen: daily_target_options_dashboard"
echo "Keep-awake screen: keep_awake_trading"
echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT/"
echo "Strategy mode: $OPTION_STRATEGY_MODE | multi-leg enabled: $ENABLE_MULTI_LEG | short-leg strategies enabled: $ENABLE_SHORT_OPTION_STRATEGIES"
echo "Forecast engine: $FORECAST_ENGINE | selection metric: $SELECTION_METRIC | search level: $SEARCH_LEVEL"
