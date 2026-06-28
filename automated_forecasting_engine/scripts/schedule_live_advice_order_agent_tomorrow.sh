#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/Users/ruddigarcia/Projects/invest}"
PYTHON="${PYTHON:-python}"
TICKERS="${TICKERS:-ASML,MMM,WM}"
RUN_HOUR="${RUN_HOUR:-15}"
RUN_MINUTE="${RUN_MINUTE:-35}"
EXECUTE_LIVE_ORDERS="${EXECUTE_LIVE_ORDERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/automated_forecasting_engine/runs/live_advice_order_agent}"
LABEL="${LABEL:-com.marketforecasting.live-advice-order-agent.tomorrow}"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOG_DIR="$PROJECT_DIR/automated_forecasting_engine/runs/live_advice_order_agent"
TOMORROW="$(date -v+1d +%Y-%m-%d)"
YEAR="$(date -j -f %Y-%m-%d "$TOMORROW" +%Y)"
MONTH="$(date -j -f %Y-%m-%d "$TOMORROW" +%-m)"
DAY="$(date -j -f %Y-%m-%d "$TOMORROW" +%-d)"

mkdir -p "$LOG_DIR" "$HOME/Library/LaunchAgents"

FLAGS=(--only-if-expired)
if [[ "$EXECUTE_LIVE_ORDERS" == "1" ]]; then
  FLAGS+=(--execute-live-orders --confirm-live-order-risk)
fi

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>WorkingDirectory</key>
  <string>$PROJECT_DIR</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>cd "$PROJECT_DIR" &amp;&amp; PYTHONPATH=automated_forecasting_engine/src "$PYTHON" -m market_forecasting_engine.live_trading.stocks.advice_order_agent --tickers "$TICKERS" --output-dir "$OUTPUT_DIR" --env-file "$PROJECT_DIR/.env" ${FLAGS[*]}</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Year</key><integer>$YEAR</integer>
    <key>Month</key><integer>$MONTH</integer>
    <key>Day</key><integer>$DAY</integer>
    <key>Hour</key><integer>$RUN_HOUR</integer>
    <key>Minute</key><integer>$RUN_MINUTE</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>$LOG_DIR/launchd_stdout.log</string>
  <key>StandardErrorPath</key>
  <string>$LOG_DIR/launchd_stderr.log</string>
</dict>
</plist>
PLIST

launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"

echo "Scheduled $LABEL for $TOMORROW $RUN_HOUR:$RUN_MINUTE local time."
echo "Plist: $PLIST"
echo "Live execution: $EXECUTE_LIVE_ORDERS"
echo "To wake the Mac before the run, execute:"
echo "sudo pmset schedule wakeorpoweron \"$(date -j -f '%Y-%m-%d %H:%M' "$TOMORROW $RUN_HOUR:$((RUN_MINUTE - 5))" '+%m/%d/%y %H:%M:%S')\""
