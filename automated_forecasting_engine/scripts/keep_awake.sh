#!/usr/bin/env zsh
set -euo pipefail

MODE="system"
INTERVAL_SECONDS="60"
PIXELS="1"

usage() {
  cat <<'EOF'
Keep the Mac awake while trading agents/dashboard are running.

Recommended:
  automated_forecasting_engine/scripts/keep_awake.sh

Optional mouse nudge mode:
  automated_forecasting_engine/scripts/keep_awake.sh --mouse --interval-seconds 60

Options:
  --system                  Use macOS caffeinate only. Default.
  --mouse                   Also nudge the pointer by a few pixels.
  --interval-seconds N      Mouse nudge interval. Default: 60.
  --pixels N                Mouse nudge distance. Default: 1.
  -h, --help                Show help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      MODE="system"
      shift
      ;;
    --mouse)
      MODE="mouse"
      shift
      ;;
    --interval-seconds)
      INTERVAL_SECONDS="${2:?missing value for --interval-seconds}"
      shift 2
      ;;
    --pixels)
      PIXELS="${2:?missing value for --pixels}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == "system" ]]; then
  echo "Keeping macOS awake with caffeinate. Press Ctrl+C to stop."
  exec caffeinate -dimsu
fi

echo "Keeping macOS awake with caffeinate and mouse nudge every ${INTERVAL_SECONDS}s. Press Ctrl+C to stop."
caffeinate -dimsu &
CAFFEINATE_PID=$!
trap 'kill "$CAFFEINATE_PID" 2>/dev/null || true' EXIT INT TERM

python3 - "$INTERVAL_SECONDS" "$PIXELS" <<'PY'
import ctypes
import sys
import time

interval = max(1.0, float(sys.argv[1]))
pixels = max(1.0, float(sys.argv[2]))

class CGPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

app = ctypes.CDLL("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
core = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

app.CGEventCreate.argtypes = [ctypes.c_void_p]
app.CGEventCreate.restype = ctypes.c_void_p
app.CGEventGetLocation.argtypes = [ctypes.c_void_p]
app.CGEventGetLocation.restype = CGPoint
app.CGEventCreateMouseEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint32, CGPoint, ctypes.c_uint32]
app.CGEventCreateMouseEvent.restype = ctypes.c_void_p
app.CGEventPost.argtypes = [ctypes.c_uint32, ctypes.c_void_p]
core.CFRelease.argtypes = [ctypes.c_void_p]

K_CG_HID_EVENT_TAP = 0
K_CG_EVENT_MOUSE_MOVED = 5
direction = 1.0

while True:
    event = app.CGEventCreate(None)
    if not event:
        raise RuntimeError("Could not read current mouse location.")
    point = app.CGEventGetLocation(event)
    core.CFRelease(event)

    target = CGPoint(point.x + direction * pixels, point.y)
    move = app.CGEventCreateMouseEvent(None, K_CG_EVENT_MOUSE_MOVED, target, 0)
    if move:
        app.CGEventPost(K_CG_HID_EVENT_TAP, move)
        core.CFRelease(move)
    direction *= -1.0
    time.sleep(interval)
PY
