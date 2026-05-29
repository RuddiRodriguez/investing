#!/bin/zsh
set -euo pipefail

PROJECT_DIR="/Users/ruddigarcia/Projects/invest"
TICKER=""
PROFILE="medium"
LABEL=""
HOLDING_STATUS="not_owned"
REFRESH_AFTER_HOURS="12"
LLM_ENV_FILE="$PROJECT_DIR/.env"
START_INTERVAL="3600"
REPLACE="0"
QUIET_UNCHANGED="1"

usage() {
  cat <<'USAGE'
Usage:
  automated_forecasting_engine/scripts/install_watch_agent_launchd.sh \
    --ticker ASML \
    --profile medium \
    --label com.marketforecasting.watchagent.asml.medium

Options:
  --ticker SYMBOL                 Required, for example ASML or NVDA.
  --profile PROFILE               aggressive, medium, or conservative. Default: medium.
  --label LABEL                   Optional unique launchd label. Default: com.marketforecasting.watchagent.<ticker>.<profile>
  --holding-status STATUS         not_owned or owned. Default: not_owned.
  --refresh-after-hours HOURS     Full forecast + LLM refresh cadence. Default: 12.
  --interval-seconds SECONDS      launchd interval. Default: 3600.
  --llm-env-file PATH             .env file with OPENAI_API_KEY. Default: /Users/ruddigarcia/Projects/invest/.env.
  --project-dir PATH              Project folder. Default: /Users/ruddigarcia/Projects/invest.
  --quiet-unchanged               Only print when action changes, BUY/SELL triggers, or forecast refreshes. Default.
  --print-unchanged               Print every hourly check.
  --replace                       Replace an already loaded LaunchAgent with the same label.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ticker)
      TICKER="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --holding-status)
      HOLDING_STATUS="$2"
      shift 2
      ;;
    --refresh-after-hours)
      REFRESH_AFTER_HOURS="$2"
      shift 2
      ;;
    --interval-seconds)
      START_INTERVAL="$2"
      shift 2
      ;;
    --llm-env-file)
      LLM_ENV_FILE="$2"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="$2"
      LLM_ENV_FILE="$PROJECT_DIR/.env"
      shift 2
      ;;
    --quiet-unchanged)
      QUIET_UNCHANGED="1"
      shift
      ;;
    --print-unchanged)
      QUIET_UNCHANGED="0"
      shift
      ;;
    --replace)
      REPLACE="1"
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$TICKER" ]]; then
  echo "--ticker is required." >&2
  usage >&2
  exit 2
fi

TICKER_UPPER="$(echo "$TICKER" | tr '[:lower:]' '[:upper:]')"
TICKER_LOWER="$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')"
PROFILE_LOWER="$(echo "$PROFILE" | tr '[:upper:]' '[:lower:]')"

if [[ -z "$LABEL" ]]; then
  LABEL="com.marketforecasting.watchagent.${TICKER_LOWER}.${PROFILE_LOWER}"
fi

PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$PLIST_DIR/${LABEL}.plist"
SCRIPT_PATH="$PROJECT_DIR/automated_forecasting_engine/scripts/run_watch_agent_once.sh"
STATE_DIR="automated_forecasting_engine/runs/watch_agent_state"
STDOUT_PATH="$PROJECT_DIR/automated_forecasting_engine/runs/watch_agent_state/${LABEL}.stdout.log"
STDERR_PATH="$PROJECT_DIR/automated_forecasting_engine/runs/watch_agent_state/${LABEL}.stderr.log"
SERVICE="gui/$(id -u)/${LABEL}"

mkdir -p "$PLIST_DIR"
mkdir -p "$PROJECT_DIR/automated_forecasting_engine/runs/watch_agent_state"

if launchctl print "$SERVICE" >/dev/null 2>&1; then
  if [[ "$REPLACE" != "1" ]]; then
    echo "LaunchAgent already loaded: $LABEL" >&2
    echo "Use --replace to unload and reinstall only this label." >&2
    exit 1
  fi
  launchctl bootout "gui/$(id -u)" "$PLIST_PATH" >/dev/null 2>&1 || true
fi

cat > "$PLIST_PATH" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>

  <key>ProgramArguments</key>
  <array>
    <string>${SCRIPT_PATH}</string>
  </array>

  <key>WorkingDirectory</key>
  <string>${PROJECT_DIR}</string>

  <key>EnvironmentVariables</key>
  <dict>
    <key>PROJECT_DIR</key>
    <string>${PROJECT_DIR}</string>
    <key>TICKER</key>
    <string>${TICKER_UPPER}</string>
    <key>PROFILE</key>
    <string>${PROFILE_LOWER}</string>
    <key>HOLDING_STATUS</key>
    <string>${HOLDING_STATUS}</string>
    <key>REFRESH_AFTER_HOURS</key>
    <string>${REFRESH_AFTER_HOURS}</string>
    <key>LLM_ENV_FILE</key>
    <string>${LLM_ENV_FILE}</string>
    <key>STATE_DIR</key>
    <string>${STATE_DIR}</string>
    <key>QUIET_UNCHANGED</key>
    <string>${QUIET_UNCHANGED}</string>
  </dict>

  <key>RunAtLoad</key>
  <true/>

  <key>StartInterval</key>
  <integer>${START_INTERVAL}</integer>

  <key>StandardOutPath</key>
  <string>${STDOUT_PATH}</string>

  <key>StandardErrorPath</key>
  <string>${STDERR_PATH}</string>
</dict>
</plist>
PLIST

plutil -lint "$PLIST_PATH"
launchctl bootstrap "gui/$(id -u)" "$PLIST_PATH"
launchctl kickstart -k "$SERVICE"

echo "Installed and started LaunchAgent:"
echo "  label: $LABEL"
echo "  plist: $PLIST_PATH"
echo "  ticker: $TICKER_UPPER"
echo "  profile: $PROFILE_LOWER"
echo "  quiet unchanged: $QUIET_UNCHANGED"
echo "  stdout: $STDOUT_PATH"
echo "  stderr: $STDERR_PATH"
